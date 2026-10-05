/* llmengine_moe_adapter_gguf.c — ingest DIRETTO di un modello MoE in GGUF (nessuna conversione offline).
 *
 * load: legge i metadati (llmengine_moe_discover_gguf) e i tensori dense dal GGUF (dequant a f32),
 * con TRASPOSIZIONE dal layout GGML `[in,out]` a quello atteso dal forward `[out,in]` dove serve
 * (q/k/v/o, router); norm e per-head norm sono vettori 1D. Gli esperti restano nel file e sono
 * dequant-on-read per (layer, expert) via llmengine_moe_gguf_expert_swiglu.
 * forward: riusa la stessa matematica di Qwen3 (llmengine_moe_qwen3_forward.c).
 * Primo slice: F16/F32/Q8_0/Q4_0; i tipi non supportati danno errore esplicito al load. */
#include "llmengine_moe_adapter_gguf.h"
#include "llmengine_moe.h"
#include "llmengine_moe_discover.h"
#include "llmengine_moe_gguf.h"
#include "llmengine_moe_gguf_expert.h"
#include "llmengine_moe_gguf_store.h"
#include "llmengine_moe_qwen3_forward.h"
#include "llmengine_moe_arch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GA_MAX_LAYERS 128
#define GA_MAX_CTX    4096     /* cap di capacità KV per lo slice 1 (evita alloc enormi su ctx GGUF) */

typedef struct {
    float *input_ln, *post_ln, *q, *k, *v, *o, *q_norm, *k_norm, *router;
    float *sh_gate, *sh_up, *sh_down;   /* M7.1 shared-expert (NULL se assente) */
    uint32_t sh_inter;
} GALayerW;

typedef struct {
    FILE*      f;
    LlmEngineMoeGguf    g;
    Q3Cfg      cfg;
    const LlmCodecGgufTensor* embed;    /* token_embd [hidden, vocab] */
    const LlmCodecGgufTensor* lmhead;   /* output.weight [hidden, vocab], o token_embd se tied */
    float*     final_norm;
    GALayerW   ly[GA_MAX_LAYERS];
    const LlmCodecGgufTensor* gate[GA_MAX_LAYERS];
    const LlmCodecGgufTensor* up[GA_MAX_LAYERS];
    const LlmCodecGgufTensor* down[GA_MAX_LAYERS];
    Q3Layer    lv[GA_MAX_LAYERS];
    float*     kv_k; float* kv_v;
    uint32_t   pos;
    /* Fase 2 (stile DS4): store degli esperti (mmap RESIDENT / pread+LRU SWAP), aperto al primo
     * forward dal config del runtime (mode + cache_mb). NULL = fallback dequant-on-read via FILE*. */
    GgufStore* store;
    uint64_t   store_mb;   /* budget (MB) con cui lo store e' stato aperto (per il resize CACHE) */
    LlmEngineMoeRuntime* rt;   /* runtime del forward corrente (route-cb: trace ROUTER + step K2) */
    char       path[1024];
} GgufSelf;

/* Dequant di un vettore 1D dal GGUF (norm, per-head norm). NULL se assente. */
static float* deq_vec(GgufSelf* s, const char* name, uint32_t len) {
    const LlmCodecGgufTensor* t = llmengine_moe_gguf_tensor(&s->g, name);
    if (t == NULL) return NULL;
    float* out = (float*)malloc((size_t)len * sizeof(float));
    if (out == NULL) return NULL;
    if (llmengine_moe_gguf_tensor_f32(s->f, &s->g, t, out, len) != LLMENGINE_MOE_OK) { free(out); return NULL; }
    return out;
}

/* Dequant di una matrice GGML [in,out] (ne0=in più veloce) nel layout f32 atteso dal forward.
 * NB: con ne0 più veloce, l'elemento (i0=in i, i1=out o) sta a `o*in + i` = `i + o*in`, che è
 * GIÀ il row-major [out,in] del forward (`out[o*in+i]`): nessuna trasposizione, si copia. */
static float* load_dense_t(GgufSelf* s, const char* name, uint32_t out_dim, uint32_t in_dim) {
    const LlmCodecGgufTensor* t = llmengine_moe_gguf_tensor(&s->g, name);
    if (t == NULL) return NULL;
    size_t n = (size_t)in_dim * out_dim;
    float* buf = (float*)malloc(n * sizeof(float));
    if (buf == NULL) return NULL;
    if (llmengine_moe_gguf_tensor_f32(s->f, &s->g, t, buf, n) != LLMENGINE_MOE_OK) { free(buf); return NULL; }
    return buf;
}

static void lname(char* buf, size_t cap, uint32_t l, const char* suffix) {
    snprintf(buf, cap, "blk.%u.%s", l, suffix);
}

/* K5: esperti disponibili nel tensor stacked (dims[2]); 0 = tensor non 3D. */
static uint32_t gguf_tensor_experts(const LlmCodecGgufTensor* t) {
    return (t != NULL && t->n_dims >= 3) ? (uint32_t)t->dims[2] : 0;
}

static LlmEngineMoeStatus gguf_expert_cb(void* ectx, uint32_t layer, uint32_t expert, const float* x, float* out) {
    GgufSelf* s = (GgufSelf*)ectx;
    if (layer >= s->cfg.n_layers) return LLMENGINE_MOE_ERR_INVALID;
    if (s->gate[layer] == NULL || s->up[layer] == NULL || s->down[layer] == NULL) return LLMENGINE_MOE_ERR_NOT_IMPLEMENTED;
    /* K5: il router puo' selezionare un esperto che nel container NON esiste (tensor piu'
     * corto / corrotto): stato DISTINTO, mai un risultato silenzioso. */
    uint32_t ne = gguf_tensor_experts(s->gate[layer]);
    if (ne == 0 || expert >= ne) {
        if (s->store != NULL) s->store->unresolvable++;
        return LLMENGINE_MOE_ERR_EXPERT_MISSING;
    }
    if (s->store != NULL) {   /* stile DS4: byte grezzi da mmap (RESIDENT) o cache LRU (SWAP) */
        const uint8_t *pu, *pd;
        const uint8_t* pg = llmengine_moe_gguf_store_expert(s->store, layer, expert, &pu, &pd);
        if (pg != NULL)
            return llmengine_moe_gguf_expert_swiglu_mem(s->gate[layer]->type, s->up[layer]->type,
                                                  s->down[layer]->type, pg, pu, pd,
                                                  s->cfg.hidden, s->cfg.inter, x, out);
    }
    return llmengine_moe_gguf_expert_swiglu(s->f, &s->g, s->gate[layer], s->up[layer], s->down[layer],
                                      expert, s->cfg.hidden, s->cfg.inter, x, out);
}

/* M6.4 — variante BATCH: l'esperto viene dequantizzato UNA volta e applicato a tutte le
 * posizioni del prompt che lo hanno selezionato (usato dal prefill batched). */
static LlmEngineMoeStatus gguf_expert_batch_cb(void* ectx, uint32_t layer, uint32_t expert,
                                         const float* X, uint32_t n, float* out) {
    GgufSelf* s = (GgufSelf*)ectx;
    if (layer >= s->cfg.n_layers) return LLMENGINE_MOE_ERR_INVALID;
    if (s->gate[layer] == NULL || s->up[layer] == NULL || s->down[layer] == NULL) return LLMENGINE_MOE_ERR_NOT_IMPLEMENTED;
    uint32_t ne = gguf_tensor_experts(s->gate[layer]);   /* K5: bounds, mai silenzioso */
    if (ne == 0 || expert >= ne) {
        if (s->store != NULL) s->store->unresolvable++;
        return LLMENGINE_MOE_ERR_EXPERT_MISSING;
    }
    if (s->store != NULL) {   /* batch: l'esperto viene dequantizzato UNA volta dallo store */
        const uint8_t *pu, *pd;
        const uint8_t* pg = llmengine_moe_gguf_store_expert(s->store, layer, expert, &pu, &pd);
        if (pg != NULL)
            return llmengine_moe_gguf_expert_swiglu_batch_mem(s->gate[layer]->type, s->up[layer]->type,
                                                        s->down[layer]->type, pg, pu, pd,
                                                        s->cfg.hidden, s->cfg.inter, X, n, out);
    }
    return llmengine_moe_gguf_expert_swiglu_batch(s->f, &s->g, s->gate[layer], s->up[layer], s->down[layer],
                                            expert, s->cfg.hidden, s->cfg.inter, X, n, out);
}

/* lm_head: riga v del GGUF contigua ([hidden, vocab]) → out[v] = riga · xn (dequant-on-read). */
static LlmEngineMoeStatus gguf_lmhead(GgufSelf* s, const float* xn, float* out) {
    float* row = (float*)malloc((size_t)s->cfg.hidden * sizeof(float));
    if (row == NULL) return LLMENGINE_MOE_ERR_NO_MEMORY;
    for (uint32_t v = 0; v < s->cfg.vocab; ++v) {
        if (llmengine_moe_gguf_row_f32(s->f, &s->g, s->lmhead, v, row, s->cfg.hidden) != LLMENGINE_MOE_OK) {
            free(row); return LLMENGINE_MOE_ERR_NOT_IMPLEMENTED;
        }
        double acc = 0.0;
        for (uint32_t i = 0; i < s->cfg.hidden; ++i) acc += (double)row[i] * (double)xn[i];
        out[v] = (float)acc;
    }
    free(row);
    return LLMENGINE_MOE_OK;
}

/* M6 v2 + K1/K2: il router del layer notifica gli esperti scelti (feed predittore/prefetch),
 * emette il ROUTER nel trace (confine di step per il replay) e avanza lo step dello store
 * (un hit su un bundle di uno step precedente conta in `resident_hits`). */
/* M6: prefetch REALE dal predittore cross-layer (route_prob/future) — carica l'esperto nel store
 * (warm cache) cosi' l'accesso al forward del layer successivo diventa un HIT. */
static int gguf_prefetch_fn(void* ctx, uint32_t layer, uint32_t expert) {
    GgufSelf* s = (GgufSelf*)ctx;
    if (s == NULL || s->store == NULL) return -1;
    return llmengine_moe_gguf_store_prefetch(s->store, layer, expert);
}

static void gguf_route_cb(void* ctx, uint32_t layer, const uint32_t* experts, uint32_t n) {
    GgufSelf* s = (GgufSelf*)ctx;
    if (s->store != NULL) {
        s->store->step++;
        if (s->rt != NULL) s->store->prof = llmengine_moe_profiler(s->rt);   /* K1: trace avviato dopo il load */
        llmengine_moe_gguf_store_note_route(s->store, layer, experts, n);    /* D19: hotness sul cache reale */
    }
    if (s->rt != NULL) llmengine_moe_notify_routing(s->rt, layer, experts, n);
}

/* Pubblica le metriche dello store nel runtime (per AI.STREAMSTATS / LLM.STREAMSTATS). no-op se
 * rt==NULL (uso diretto dell'adapter nei test) o store assente. */
static void gguf_publish(GgufSelf* s, LlmEngineMoeRuntime* rt) {
    if (s->store == NULL) return;
    llmengine_moe_set_stream_metrics(rt, s->store->bytes_read, s->store->hits, s->store->misses,
                               s->store->evictions, s->store->resident_hits,
                               llmengine_moe_gguf_store_resident_bytes(s->store),
                               s->store_mb * 1024ull * 1024ull, s->store->unresolvable);
}

static LlmEngineMoeStatus gguf_forward(void* self, LlmEngineMoeRuntime* rt, const float* in, uint32_t seq,
                                 uint32_t hidden, float* out) {
    GgufSelf* s = (GgufSelf*)self;
    if (hidden != s->cfg.hidden) return LLMENGINE_MOE_ERR_INVALID;
    s->rt = rt;
    /* Fase 2 (stile DS4): apre lo store al primo forward, dal config del runtime (mode+cache_mb).
     * RESIDENT → mmap zero-copy; SWAP → pread+LRU bundle. Fallback su dequant-on-read se assente.
     * `LLM MODEL CACHE`/`DEVICE` possono cambiare mode/budget → lo store viene riaperto. */
    if (s->path[0] != '\0') {
        int swap = (llmengine_moe_mode(rt) != LLMENGINE_MOE_MODE_RESIDENT);
        uint64_t mb = llmengine_moe_cache_mb(rt);
        if (s->store != NULL && (s->store_mb != mb || s->store->swap != swap)) {
            llmengine_moe_gguf_store_close(s->store); s->store = NULL;
        }
        if (s->store == NULL) {
            s->store = llmengine_moe_gguf_store_open(s->path, &s->g, s->cfg.n_layers, swap, mb);
            s->store_mb = mb;
            if (s->store != NULL) {
                /* D19: hotness appresa sul cache REALE; profilo `<path>.stats` se opt-in. */
                llmengine_moe_gguf_store_set_dims(s->store, s->cfg.n_layers, s->cfg.n_experts);
                llmengine_moe_gguf_store_set_hotness(s->store, llmengine_moe_hotness(rt));
                if (llmengine_moe_hotness(rt)) {
                    char pp[1200]; snprintf(pp, sizeof(pp), "%s.stats", s->path);
                    /* advisory: profilo assente = ok; se presente, preload top-K (warm-up). */
                    if (llmengine_moe_gguf_store_profile_load(s->store, pp) == 0)
                        (void)llmengine_moe_gguf_store_preload_topk(s->store, LLMENGINE_MOE_GGUF_PRELOAD_TOPK);
                }
            }
        }
        if (s->store != NULL) {
            s->store->prof = llmengine_moe_profiler(rt);   /* K1: trace */
            /* M6: collega il prefetch cross-layer (route_prob/future) allo store. */
            llmengine_moe_set_prefetch_fn(rt, gguf_prefetch_fn, s);
        }
    }
    Q3KvCache kv; kv.k = s->kv_k; kv.v = s->kv_v;
    float* x = (float*)malloc((size_t)hidden * sizeof(float));
    float* y = (float*)malloc((size_t)hidden * sizeof(float));
    float* xn = (float*)malloc((size_t)hidden * sizeof(float));
    if (!x || !y || !xn) { free(x); free(y); free(xn); return LLMENGINE_MOE_ERR_NO_MEMORY; }

    if (seq > 1) {
        /* M6.4: prefill BATCHED — esperti condivisi tra le posizioni del prompt. */
        float* xa = (float*)malloc((size_t)seq * hidden * sizeof(float));
        float* oa = (float*)malloc((size_t)seq * hidden * sizeof(float));
        if (!xa || !oa) { free(x); free(y); free(xn); free(xa); free(oa); return LLMENGINE_MOE_ERR_NO_MEMORY; }
        for (uint32_t t = 0; t < seq; ++t) {
            uint32_t tok = (uint32_t)in[t];
            if (tok >= s->cfg.vocab) { free(x); free(y); free(xn); free(xa); free(oa); return LLMENGINE_MOE_ERR_INVALID; }
            if (llmengine_moe_gguf_row_f32(s->f, &s->g, s->embed, tok, xa + (size_t)t * hidden, hidden) != LLMENGINE_MOE_OK) {
                free(x); free(y); free(xn); free(xa); free(oa); return LLMENGINE_MOE_ERR_NOT_IMPLEMENTED;
            }
        }
        LlmEngineMoeStatus ps = llmengine_moe_qwen3_prefill_batchfn(&s->cfg, s->lv, s->cfg.n_layers,
                                                        gguf_expert_batch_cb, s, gguf_route_cb, s,
                                                        seq, s->pos, xa, oa, &kv);
        if (ps != LLMENGINE_MOE_OK) {   /* K5: pubblica anche le metriche di errore (unresolvable) */
            gguf_publish(s, rt); free(x); free(y); free(xn); free(xa); free(oa); return ps;
        }
        s->pos += seq;
        llmengine_moe_rmsnorm(oa + (size_t)(seq - 1) * hidden, s->final_norm, s->cfg.eps, hidden, xn);
        LlmEngineMoeStatus ls = gguf_lmhead(s, xn, out);
        gguf_publish(s, rt);
        free(x); free(y); free(xn); free(xa); free(oa);
        return ls;
    }

    for (uint32_t t = 0; t < seq; ++t) {           /* decode a un token (per-token) */
        uint32_t tok = (uint32_t)in[t];
        if (tok >= s->cfg.vocab) { free(x); free(y); free(xn); return LLMENGINE_MOE_ERR_INVALID; }
        if (llmengine_moe_gguf_row_f32(s->f, &s->g, s->embed, tok, x, hidden) != LLMENGINE_MOE_OK) {
            free(x); free(y); free(xn); return LLMENGINE_MOE_ERR_NOT_IMPLEMENTED;
        }
        for (uint32_t l = 0; l < s->cfg.n_layers; ++l) {
            LlmEngineMoeStatus st = llmengine_moe_qwen3_decode_layer(&s->cfg, &s->lv[l], gguf_expert_cb, s,
                                                         gguf_route_cb, s, l, s->pos, x, y, &kv);
            if (st != LLMENGINE_MOE_OK) {   /* K5: pubblica anche le metriche di errore (unresolvable) */
                gguf_publish(s, rt); free(x); free(y); free(xn); return st;
            }
            float* tmp = x; x = y; y = tmp;
        }
        s->pos++;
    }
    llmengine_moe_rmsnorm(x, s->final_norm, s->cfg.eps, hidden, xn);
    LlmEngineMoeStatus ls = gguf_lmhead(s, xn, out);
    gguf_publish(s, rt);
    free(x); free(y); free(xn);
    return ls;
}

static LlmEngineMoeStatus gguf_load(void* self, const char* path, LlmEngineMoeModelMeta* meta) {
    GgufSelf* s = (GgufSelf*)self;
    LlmEngineMoeDiscover disc;
    LlmEngineMoeStatus ds = llmengine_moe_discover_gguf(path, &disc);
    if (ds != LLMENGINE_MOE_OK) return ds;
    const LlmEngineMoeArchDesc* d = &disc.desc;
    if (d->hidden == 0 || d->n_layers == 0 || d->head_dim == 0 || d->n_experts == 0 || d->topk == 0)
        return LLMENGINE_MOE_ERR_FORMAT;
    if (d->n_layers > GA_MAX_LAYERS) return LLMENGINE_MOE_ERR_FORMAT;

    s->cfg.hidden = d->hidden; s->cfg.n_heads = d->n_heads; s->cfg.n_kv_heads = d->n_kv_heads;
    s->cfg.head_dim = d->head_dim; s->cfg.n_layers = d->n_layers; s->cfg.n_experts = d->n_experts;
    s->cfg.topk = d->topk; s->cfg.inter = d->inter; s->cfg.vocab = d->vocab;
    s->cfg.ctx = (d->ctx == 0) ? 2048 : (d->ctx < GA_MAX_CTX ? d->ctx : GA_MAX_CTX);
    s->cfg.eps = d->eps; s->cfg.rope_theta = d->rope_theta;
    s->cfg.rope_variant = d->rope_variant; s->cfg.router_kind = d->router_kind;
    s->cfg.norm_kind = d->norm_kind; s->cfg.flags = d->flags;
    s->cfg.arch_family = d->arch_family;
    if (s->cfg.vocab == 0 || s->cfg.inter == 0) return LLMENGINE_MOE_ERR_FORMAT;

    if (llmengine_moe_gguf_read(path, &s->g) != LLMENGINE_MOE_OK) return LLMENGINE_MOE_ERR_FORMAT;
    s->f = fopen(path, "rb");
    if (s->f == NULL) return LLMENGINE_MOE_ERR_IO;
    snprintf(s->path, sizeof(s->path), "%s", path);   /* per aprire lo store al primo forward */

    /* variante router: `expert_weights_norm` assente/0 → top-k NON rinormalizzati (es. OLMoE). */
    {   char ek[140];
        snprintf(ek, sizeof(ek), "%s.expert_weights_norm", disc.model_type);
        if (llmengine_moe_gguf_num(&s->g, ek, 0.0) == 0.0) s->cfg.flags |= LLMENGINE_MOE_FLAG_NO_RENORM;
    }

    s->embed = llmengine_moe_gguf_tensor(&s->g, "token_embd.weight");
    s->lmhead = llmengine_moe_gguf_tensor(&s->g, "output.weight");
    if (s->lmhead == NULL) s->lmhead = s->embed;     /* tied embeddings */
    if (s->embed == NULL || s->lmhead == NULL) return LLMENGINE_MOE_ERR_FORMAT;

    uint32_t H = s->cfg.n_heads, KVH = s->cfg.n_kv_heads, D = s->cfg.head_dim, hidden = s->cfg.hidden;
    s->final_norm = deq_vec(s, "output_norm.weight", hidden);
    if (s->final_norm == NULL) return LLMENGINE_MOE_ERR_FORMAT;

    char nm[128];
    for (uint32_t l = 0; l < s->cfg.n_layers; ++l) {
        GALayerW* w = &s->ly[l];
        lname(nm, sizeof(nm), l, "attn_norm.weight");          w->input_ln = deq_vec(s, nm, hidden);
        lname(nm, sizeof(nm), l, "ffn_norm.weight");           w->post_ln  = deq_vec(s, nm, hidden);
        lname(nm, sizeof(nm), l, "attn_q.weight");             w->q = load_dense_t(s, nm, H * D, hidden);
        lname(nm, sizeof(nm), l, "attn_k.weight");             w->k = load_dense_t(s, nm, KVH * D, hidden);
        lname(nm, sizeof(nm), l, "attn_v.weight");             w->v = load_dense_t(s, nm, KVH * D, hidden);
        lname(nm, sizeof(nm), l, "attn_output.weight");        w->o = load_dense_t(s, nm, hidden, H * D);
        lname(nm, sizeof(nm), l, "attn_q_norm.weight");
        { const LlmCodecGgufTensor* qt = llmengine_moe_gguf_tensor(&s->g, nm);
          uint32_t qlen = qt ? (uint32_t)qt->dims[0] : 0;
          if (qlen == hidden) s->cfg.flags |= LLMENGINE_MOE_FLAG_QK_NORM_FULL;   /* es. OLMoE */
          w->q_norm = qlen ? deq_vec(s, nm, qlen) : NULL; }
        lname(nm, sizeof(nm), l, "attn_k_norm.weight");
        { const LlmCodecGgufTensor* kt = llmengine_moe_gguf_tensor(&s->g, nm);
          uint32_t klen = kt ? (uint32_t)kt->dims[0] : 0;
          w->k_norm = klen ? deq_vec(s, nm, klen) : NULL; }
        lname(nm, sizeof(nm), l, "ffn_gate_inp.weight");       w->router = load_dense_t(s, nm, s->cfg.n_experts, hidden);
        lname(nm, sizeof(nm), l, "ffn_gate_exps.weight");      s->gate[l] = llmengine_moe_gguf_tensor(&s->g, nm);
        lname(nm, sizeof(nm), l, "ffn_up_exps.weight");        s->up[l]   = llmengine_moe_gguf_tensor(&s->g, nm);
        lname(nm, sizeof(nm), l, "ffn_down_exps.weight");      s->down[l] = llmengine_moe_gguf_tensor(&s->g, nm);
        /* M7.1 shared-expert (GLM/DeepSeek/Qwen2-MoE): MLP SwiGLU 2D sempre applicato */
        { char ng[128], nu[128], nd[128];
          snprintf(ng, sizeof(ng), "blk.%u.ffn_gate_shexp.weight", l);
          snprintf(nu, sizeof(nu), "blk.%u.ffn_up_shexp.weight", l);
          snprintf(nd, sizeof(nd), "blk.%u.ffn_down_shexp.weight", l);
          const LlmCodecGgufTensor* sg = llmengine_moe_gguf_tensor(&s->g, ng);
          if (sg != NULL && sg->n_dims >= 2) {
              uint32_t sh = (uint32_t)sg->dims[1];   /* n_ff_shexp */
              w->sh_gate = load_dense_t(s, ng, sh, hidden);
              w->sh_up   = load_dense_t(s, nu, sh, hidden);
              w->sh_down = load_dense_t(s, nd, hidden, sh);
              w->sh_inter = sh;
          }
        }
        if (!w->input_ln || !w->post_ln || !w->q || !w->k || !w->v || !w->o ||
            !w->router || !s->gate[l] || !s->up[l] || !s->down[l])
            return LLMENGINE_MOE_ERR_NOT_IMPLEMENTED;   /* modello non-MoE stacked o tipo non supportato */
        s->lv[l].input_ln = w->input_ln; s->lv[l].post_attn_ln = w->post_ln;
        s->lv[l].q_w = w->q; s->lv[l].k_w = w->k; s->lv[l].v_w = w->v; s->lv[l].o_w = w->o;
        s->lv[l].q_norm = w->q_norm; s->lv[l].k_norm = w->k_norm; s->lv[l].router = w->router;
        s->lv[l].shared_gate = w->sh_gate; s->lv[l].shared_up = w->sh_up;
        s->lv[l].shared_down = w->sh_down; s->lv[l].shared_inter = w->sh_inter;
    }
    size_t kv_per = (size_t)s->cfg.n_layers * KVH * s->cfg.ctx * D;
    s->kv_k = (float*)calloc(kv_per, sizeof(float));
    s->kv_v = (float*)calloc(kv_per, sizeof(float));
    if (s->kv_k == NULL || s->kv_v == NULL) return LLMENGINE_MOE_ERR_NO_MEMORY;
    s->pos = 0;

    if (meta != NULL) {
        meta->layer_count = s->cfg.n_layers; meta->expert_count = s->cfg.n_experts;
        meta->experts_per_tok = s->cfg.topk; meta->hidden_size = s->cfg.hidden;
        meta->vocab = s->cfg.vocab;
        /* K1: byte totali degli esperti routed (per l'header del trace: byte/esperto =
         * routed_bytes / (layer*expert)); somma dei bundle gate+up+down per layer. */
        uint64_t routed = 0;
        for (uint32_t l = 0; l < s->cfg.n_layers; ++l) {
            char b[160];
            GgufStoreTensor tg, tu, td;
            snprintf(b, sizeof(b), "blk.%u.ffn_gate_exps.weight", l);
            if (!gguf_store_tensor(&s->g, b, &tg)) continue;
            snprintf(b, sizeof(b), "blk.%u.ffn_up_exps.weight", l);
            if (!gguf_store_tensor(&s->g, b, &tu)) continue;
            snprintf(b, sizeof(b), "blk.%u.ffn_down_exps.weight", l);
            if (!gguf_store_tensor(&s->g, b, &td)) continue;
            routed += (uint64_t)(tg.nb + tu.nb + td.nb) * tg.n_experts;
        }
        meta->routed_bytes = routed;
    }
    return LLMENGINE_MOE_OK;
}

static void gguf_destroy(void* self) {
    GgufSelf* s = (GgufSelf*)self;
    if (s == NULL) return;
    free(s->final_norm);
    for (uint32_t l = 0; l < s->cfg.n_layers && l < GA_MAX_LAYERS; ++l) {
        free(s->ly[l].input_ln); free(s->ly[l].post_ln); free(s->ly[l].q); free(s->ly[l].k);
        free(s->ly[l].v); free(s->ly[l].o); free(s->ly[l].q_norm); free(s->ly[l].k_norm); free(s->ly[l].router);
        free(s->ly[l].sh_gate); free(s->ly[l].sh_up); free(s->ly[l].sh_down);
    }
    free(s->kv_k); free(s->kv_v);
    llmengine_moe_gguf_store_close(s->store);
    llmengine_moe_gguf_free(&s->g);
    if (s->f) fclose(s->f);
    free(s);
}

LlmEngineMoeModelAdapter llmengine_moe_gguf_adapter(void) {
    LlmEngineMoeModelAdapter a;
    memset(&a, 0, sizeof(a));
    a.name = "gguf";
    a.self = calloc(1, sizeof(GgufSelf));
    a.load = gguf_load;
    a.forward = gguf_forward;
    a.destroy = gguf_destroy;
    return a;
}
