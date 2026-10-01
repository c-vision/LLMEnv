/* ============================================================================
 * llm_mlx_api.c — dominio API esportata (estratto VERBATIM da llmengine_mlx.c).
 * Superficie pubblica DFlash2 (create/free/propose/generate single e multi-sessione),
 * debug layer/embed, prefix pin/unpin, state checkpoint/restore, gate RMSNorm gated.
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"
#include "llm_mlx_dispatch_profile.h"
#include "llm_mlx_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

/* A/B: forza il verify per-sessione di DFlash2 (opzione di registro `df2_seq_verify`, niente env). */
static bool df2_seq_verify_on(void) {
    int b = 0;
    return llm_ai_mlx_opt_get_bool("df2_seq_verify", &b) == 0 && b != 0;
}
/* A4: righe di ctx del draft per SPAN (chunking del prefill del ctx). 0 = un solo span.
 * DEFAULT 4096 (2026-10-01): senza chunking il capture-ctx del target rifà l'intero
 * contesto con cache scratch FRESCA -> attenzione [n_heads, S, S] bf16. A S=71061 =
 * 24*S^2*2 = 242GB -> OOM (mlx malloc buffer max ~86GB). Con lo span il picco resta
 * O(span * S). Lo stesso span limita il backbone del drafter (A4). */
#ifndef LLM_DF2_CTX_SPAN_DEFAULT
#define LLM_DF2_CTX_SPAN_DEFAULT 4096
#endif
static int df2_ctx_span(void) {
    int v = 0;
    if (llm_ai_mlx_opt_get_int("df2_ctx_span", &v) == 0) {
        if (v < 0) return 0;           /* valore esplicito negativo = disabilita il chunking */
        if (v > 0) return v;
    }
    return LLM_DF2_CTX_SPAN_DEFAULT;
}

/* Blocco signpost DFlash2 (spostato dal monolite: era l'unico consumatore). */
#if defined(__APPLE__)
#include <os/log.h>
#include <os/signpost.h>
static os_log_t g_df2_sp_log = NULL;
static os_signpost_id_t g_df2_sp_prop = 0, g_df2_sp_ver = 0;
static int g_df2_sp_state = -1;
static bool llm_df2_signpost_on(void) {
    if (g_df2_sp_state < 0) g_df2_sp_state = (llm_ai_mlx_opt_env("LLM_DF2_SIGNPOST") != NULL) ? 1 : 0;
    return g_df2_sp_state == 1;
}
static void llm_df2_sp_init(void) {
    if (g_df2_sp_log == NULL) {
        g_df2_sp_log = os_log_create("llm.df2", "signpost");
        g_df2_sp_prop = os_signpost_id_generate(g_df2_sp_log);
        g_df2_sp_ver  = os_signpost_id_generate(g_df2_sp_log);
    }
}
#define LLM_DF2_SP_BEGIN(which) do { if (llm_df2_signpost_on()) { llm_df2_sp_init(); os_signpost_interval_begin(g_df2_sp_log, g_df2_sp_##which, #which); } } while (0)
#define LLM_DF2_SP_END(which)   do { if (llm_df2_signpost_on()) { os_signpost_interval_end(g_df2_sp_log, g_df2_sp_##which, #which); } } while (0)
#else
#define LLM_DF2_SP_BEGIN(which) do {} while (0)
#define LLM_DF2_SP_END(which)   do {} while (0)
#endif /* top-k Metal per il selector DFlash2: argpartition MLX su 248k costa ~48ms/round */

/* ============================================================================
 * API esportata.
 * ============================================================================ */
/* L'attivazione dell'override sta nella create (sotto), non qui: il runtime DFlash2 e' il
 * segnale che il processo sta facendo speculazione -> AR+verify coerenti in W8A8. */
llm_ai_mlx_dflash2_handle llm_ai_mlx_dflash2_create(llm_ai_mlx_model_handle target_handle,
                                                    const char* draft_dir) {
    if (target_handle == NULL || draft_dir == NULL) return NULL;
    DFlash2Runtime* rt = (DFlash2Runtime*)calloc(1, sizeof(DFlash2Runtime));
    if (!rt) return NULL;
    rt->target = (LlmMlxModel*)target_handle;
    /* Salva il modo bf16 del TARGET prima del load del draft (che lo clobbera'). */
    rt->tgt_bf16_all = g_bf16_all; rt->tgt_act16 = g_act16;
    rt->tgt_bf16_input = rt->target->cfg.bfloat16_input;
    rt->draft = (LlmMlxModel*)llm_ai_mlx_load_model(draft_dir);
    if (!rt->draft) { free(rt); return NULL; }
    /* Pesi del draft su GPU: senza, ogni matmul li ri-trasferisce CPU->GPU (~13 ms/proiezione). */
    { const char* e = llm_ai_mlx_opt_env("LLM_DF2_NO_GPU_W");
      if (e == NULL || strcmp(e, "0") == 0)
          llm_mlx_shards_to_gpu((MultiShardWeights*)&rt->draft->weights, rt->draft->stream); }
    /* Salva il modo del DRAFT (quello che il suo load ha appena applicato) e ripristina
     * subito il modo del target: da qui in avanti i globali sono del target, e propose
     * scambia al modo del draft solo per la durata del forward del draft. */
    rt->drf_bf16_all = g_bf16_all; rt->drf_act16 = g_act16;
    rt->drf_bf16_input = rt->draft->cfg.bfloat16_input;
    g_bf16_all = rt->tgt_bf16_all; g_act16 = rt->tgt_act16;
    rt->target->cfg.bfloat16_input = rt->tgt_bf16_input;
    /* Spec attiva: AR e verify coerenti in W8A8 a M>=1 (vedi g_oq_a8_minm_override).
     * Diagnosi 2026-09-19: il "bug qualita'" era la combo bf16-off (guardato), non il W8A8. */
    g_oq_a8_minm_override = 1;
    if (rt->draft->cfg.dflash_block_size <= 0) {
        fprintf(stderr, "dflash_create: %s non e' un draft DFlash2 (dflash_config assente)\n", draft_dir);
        llm_ai_mlx_free_model(rt->draft);
        free(rt);
        return NULL;
    }
    rt->n_layers = rt->draft->cfg.n_layers;
    /* Slot 0 = la sessione single-session (default). Gli slot extra per il round-loop
     * multi-sessione si allocano con dflash2_ctx_add (lavoro in corso). */
    rt->n_ctx = 1; rt->cur_ctx = 0;
    rt->ctx = (Df2CtxState*)calloc(1, sizeof(Df2CtxState));
    if (rt->ctx == NULL) { llm_ai_mlx_dflash2_free(rt); return NULL; }
    rt->ctx[0].kcache = (mlx_array*)malloc((size_t)rt->n_layers * sizeof(mlx_array));
    rt->ctx[0].vcache = (mlx_array*)malloc((size_t)rt->n_layers * sizeof(mlx_array));
    for (int i = 0; i < rt->n_layers; i++) { rt->ctx[0].kcache[i] = mlx_array_new(); rt->ctx[0].vcache[i] = mlx_array_new(); }
    rt->ctx[0].ctx_pos = 0;
    if (!dflash_fetch_target_tensors(rt)) {
        llm_ai_mlx_dflash2_free(rt);
        return NULL;
    }
    return rt;
}

void llm_ai_mlx_dflash2_free(llm_ai_mlx_dflash2_handle handle) {
    g_oq_a8_minm_override = -1;   /* fine spec: torna al default (prefill-only W8A8) */
    if (handle == NULL) return;
    DFlash2Runtime* rt = (DFlash2Runtime*)handle;
    if (rt->ctx) {
        for (int s = 0; s < rt->n_ctx; s++) {
            if (rt->ctx[s].kcache) {
                for (int i = 0; i < rt->n_layers; i++) { mlx_array_free(rt->ctx[s].kcache[i]); mlx_array_free(rt->ctx[s].vcache[i]); }
                free(rt->ctx[s].kcache); free(rt->ctx[s].vcache);
            }
        }
        free(rt->ctx);
    }
    mlx_array_free(rt->embed_w); mlx_array_free(rt->embed_s); mlx_array_free(rt->embed_b);
    if (rt->have_lm_head) {
        mlx_array_free(rt->lm_w); mlx_array_free(rt->lm_s); mlx_array_free(rt->lm_b);
    } else {
        mlx_array_free(rt->lm_w);
    }
    if (rt->sel_cpw_host) free(rt->sel_cpw_host);   /* P1: cache host del predecessor_codebook */
    if (rt->sel_w_loaded) { mlx_array_free(rt->sel_hp_w); mlx_array_free(rt->sel_cs_w); mlx_array_free(rt->sel_cp_w); }
    if (rt->draft) llm_ai_mlx_free_model(rt->draft);
    free(rt);
}

/* Tappa 3 DFlash2 multi-sessione: assicura che il runtime abbia almeno `n` slot ctx
 * (per-sessione) e ritorna il numero di slot, o -1. Gli slot nuovi sono azzerati/vuoti. */
static int df2_ctx_ensure(DFlash2Runtime* rt, int n) {
    if (rt == NULL || n <= 0) return -1;
    if (n <= rt->n_ctx) return rt->n_ctx;
    Df2CtxState* nc = (Df2CtxState*)realloc(rt->ctx, (size_t)n * sizeof(Df2CtxState));
    if (nc == NULL) return -1;
    for (int s = rt->n_ctx; s < n; s++) {
        nc[s].kcache = (mlx_array*)malloc((size_t)rt->n_layers * sizeof(mlx_array));
        nc[s].vcache = (mlx_array*)malloc((size_t)rt->n_layers * sizeof(mlx_array));
        nc[s].ctx_pos = 0;
        for (int i = 0; i < rt->n_layers; i++) { nc[s].kcache[i] = mlx_array_new(); nc[s].vcache[i] = mlx_array_new(); }
    }
    rt->ctx = nc;
    rt->n_ctx = n;
    return rt->n_ctx;
}

/* Seleziona lo slot ctx (sessione) su cui operano propose/reset. -1 se fuori range. */
int llm_ai_mlx_dflash2_set_slot(llm_ai_mlx_dflash2_handle handle, int slot) {
    if (handle == NULL) return -1;
    DFlash2Runtime* rt = (DFlash2Runtime*)handle;
    if (slot < 0) return -1;
    if (slot >= rt->n_ctx && df2_ctx_ensure(rt, slot + 1) < 0) return -1;
    rt->cur_ctx = slot;
    return 0;
}

int llm_ai_mlx_dflash2_reset(llm_ai_mlx_dflash2_handle handle) {
    if (handle == NULL) return -1;
    DFlash2Runtime* rt = (DFlash2Runtime*)handle;
    Df2CtxState* cs = &rt->ctx[rt->cur_ctx];
    for (int i = 0; i < rt->n_layers; i++) {
        mlx_array_free(cs->kcache[i]); mlx_array_free(cs->vcache[i]);
        cs->kcache[i] = mlx_array_new(); cs->vcache[i] = mlx_array_new();
    }
    cs->ctx_pos = 0;
    return 0;
}

/* Corpo di propose (col modo bf16 GIA' scambiato sul draft dal wrapper pubblico). */
static int df2_propose_impl(DFlash2Runtime* rt,
                            int32_t last_verified_token,
                            const float* ctx_feature, int S,
                            int32_t* out_tokens, int max_out) {
    LlmMlxModel* d = rt->draft;
    const ModelConfig* c = &d->cfg;
    mlx_stream st = d->stream;
    const int L = c->dflash_block_size;
    const int K = L - 1;                 /* draft token per blocco */
    const int ctx_dim = c->dflash_n_ctx_layers * c->hidden_size;
    if (max_out < K) return -1;
    bool ok = true;

    /* ---- block = [anchor] + mask*(L-1) ---- */
    int32_t* block_ids = (int32_t*)malloc((size_t)L * sizeof(int32_t));
    block_ids[0] = last_verified_token;
    for (int i = 1; i < L; i++) block_ids[i] = c->dflash_mask_token;

    /* ---- ctx -> cache del draft, in SPAN (A4): se S supera lo span si iniettano `span` righe
     * per volta con df2_backbone (blocco reale nell'ULTIMO span: output utile; negli span
     * precedenti l'output e' SCARTATO, l'effetto utile e' solo l'append del ctx k/v nella cache
     * + l'avanzamento di ctx_pos). Stessa cache/ordine/offset rope -> risultato identico, ma grafo
     * per-span (picco memoria bounded). ctx feature in FP32 (NON castare: il reference fc() gira
     * sul contesto fp32). ---- */
    mlx_array hidden = mlx_array_new();
    double _bt0 = g_df2_time_on() ? g_df2_now() : 0.0;
    if (getenv("DF2_CTXDBG"))
        fprintf(stderr, "[ctxdbg] S=%d ctx_pos_before=%d\n", S, rt->ctx[rt->cur_ctx].ctx_pos);
    if (S > 0) {
        const int span = df2_ctx_span();
        if (span > 0 && S > span) {
            int off = 0;
            while (ok && off < S) {
                int ch = (S - off > span) ? span : (S - off);
                mlx_array ctx_arr = mlx_array_new_data(ctx_feature + (size_t)off * (size_t)ctx_dim,
                                                       ((const int[2]){ch, ctx_dim}), 2, MLX_FLOAT32);
                mlx_array xc = mlx_array_new();
                ok = mlx_reshape(&xc, ctx_arr, ((const int[3]){1, ch, ctx_dim}), 3, st) == 0;
                mlx_array_free(ctx_arr);
                if (ok) {
                    mlx_array h2 = mlx_array_new();
                    ok = df2_backbone(rt, block_ids, L, xc, ch, &h2, NULL);
                    mlx_array_free(hidden); hidden = h2;
                    if (ok) rt->ctx[rt->cur_ctx].ctx_pos += ch;
                }
                mlx_array_free(xc);
                off += ch;
            }
        } else {
            mlx_array x_ctx = mlx_array_new();
            mlx_array ctx_arr = mlx_array_new_data(ctx_feature, ((const int[2]){S, ctx_dim}), 2, MLX_FLOAT32);
            ok = mlx_reshape(&x_ctx, ctx_arr, ((const int[3]){1, S, ctx_dim}), 3, st) == 0;
            mlx_array_free(ctx_arr);
            if (ok) {
                ok = df2_backbone(rt, block_ids, L, x_ctx, S, &hidden, NULL);
                if (ok) rt->ctx[rt->cur_ctx].ctx_pos += S;
            }
            mlx_array_free(x_ctx);
        }
    }
    if (g_df2_time_on()) g_df2_t_bb += g_df2_now() - _bt0;
    free(block_ids);
    if (!ok) { mlx_array_free(hidden); return -1; }

    /* ---- hidden_post = hidden[:, 1:]  ;  logits = lm_head(hidden_post) ---- */
    mlx_array hidden_post = mlx_array_new();
    {
        const int sA[3] = {0, 1, 0}, sB[3] = {1, L, c->hidden_size}, o[3] = {1, 1, 1};
        ok = mlx_slice(&hidden_post, hidden, sA, 3, sB, 3, o, 3, st) == 0;
    }
    mlx_array logits = mlx_array_new();
    if (ok) {
        int gsz = rt->target->cfg.quant_group_size, bts = rt->target->cfg.quant_bits;
        /* Ternary-Bonsai: lm_head elencato -> ruota il suo ingresso. */
        if (ok) ok = llm_bonsai_rotate(&hidden_post, rt->target, -1, "lm_head.weight", false);
        ok = quantized_linear_cfg(&logits, hidden_post, rt->lm_w, rt->lm_s, rt->lm_b,
                                  gsz, bts, rt->target->cfg.quant_mode, st);
    }
    if (!ok) {
        mlx_array_free(hidden); mlx_array_free(hidden_post); mlx_array_free(logits);
        fprintf(stderr, "dflash_propose: backbone/lm_head fallito\n");
        return -1;
    }

    /* DF2_PHASE_TIME: materializza per fase (backbone, lm_head) cosi' il timer del selector
     * copre SOLO il selector. Senza, il primo eval flusha tutto e le attribuzioni sono false. */
    if (ok && getenv("DF2_PHASE_TIME") != NULL) {
        double _a = g_df2_now(); mlx_array_eval(hidden); g_df2_t_bb += g_df2_now() - _a;
        _a = g_df2_now(); mlx_array_eval(logits); g_df2_t_wget += g_df2_now() - _a;
    }

    /* ---- selector greedy ---- */
    ok = df2_selector(rt, hidden_post, logits, K, last_verified_token, out_tokens);

    mlx_array_free(hidden); mlx_array_free(hidden_post); mlx_array_free(logits);
    if (!ok) return -1;

    /* (cache ctx gia' avanzata: per-span nel loop, o di S nel ramo singolo) */
    return K;
}

int llm_ai_mlx_dflash2_propose(llm_ai_mlx_dflash2_handle handle,
                               int32_t last_verified_token,
                               const float* ctx_feature, int S,
                               int32_t* out_tokens, int max_out) {
    if (handle == NULL || out_tokens == NULL || S < 0 || (S > 0 && ctx_feature == NULL)) return -1;
    DFlash2Runtime* rt = (DFlash2Runtime*)handle;
    bool sv_all; mlx_dtype sv_act;
    df2_mode_enter_draft(rt, &sv_all, &sv_act);      /* forward del draft col SUO modo bf16 */
    double _pt0 = g_df2_time_on() ? g_df2_now() : 0.0;
    int rc = df2_propose_impl(rt, last_verified_token, ctx_feature, S, out_tokens, max_out);
    if (g_df2_time_on()) g_df2_t_prop += g_df2_now() - _pt0;
    df2_mode_leave_draft(rt, sv_all, sv_act);        /* ripristina il modo del target */
    return rc;
}

/* DEBUG temporaneo per la validazione: esegue il backbone (stesso percorso di
 * propose, SENZA toccare cache/ctx_pos) e copia in out (fp32 host) gli output
 * di ogni layer seguiti dalla norm finale: ordine layer0..n_layers-1, final.
 * Ogni snap e' [1,L,H]; cap >= (n_layers+1)*L*H. Ritorna il numero di float
 * scritti, -1 su errore. */
int llm_ai_mlx_dflash2_debug_layers(llm_ai_mlx_dflash2_handle handle,
                                    int32_t last_verified_token,
                                    const float* ctx_feature, int S,
                                    float* out, int cap) {
    if (handle == NULL || ctx_feature == NULL || S <= 0 || out == NULL) return -1;
    DFlash2Runtime* rt = (DFlash2Runtime*)handle;
    LlmMlxModel* d = rt->draft;
    const ModelConfig* c = &d->cfg;
    mlx_stream st = d->stream;
    const int L = c->dflash_block_size;
    const int ctx_dim = c->dflash_n_ctx_layers * c->hidden_size;
    const int H = c->hidden_size;
    const int per_snap = L * H;
    const int total = (rt->n_layers + 1) * per_snap;
    if (cap < total) return -total;
    bool sv_all; mlx_dtype sv_act;
    df2_mode_enter_draft(rt, &sv_all, &sv_act);      /* backbone del draft col SUO modo bf16 */
    bool ok = true;

    mlx_array x_ctx = mlx_array_new();
    {
        /* ctx feature dal target in FP32: NON castare a bf16 (il reference fc() gira
         * sul contesto fp32; il cast bf16 introduceva la prima divergenza, poi amplificata
         * fino a ~1e5 dai valori reali del draft vicini alla saturazione bf16). */
        const int sh3[3] = {1, S, ctx_dim};
        mlx_array ctx_arr = mlx_array_new_data(ctx_feature, ((const int[2]){S, ctx_dim}), 2, MLX_FLOAT32);
        ok = mlx_reshape(&x_ctx, ctx_arr, sh3, 3, st) == 0;
        mlx_array_free(ctx_arr);
    }
    int32_t* block_ids = NULL;
    mlx_array* layer_out = NULL;
    mlx_array hidden = mlx_array_new();
    if (ok) {
        block_ids = (int32_t*)malloc((size_t)L * sizeof(int32_t));
        block_ids[0] = last_verified_token;
        for (int i = 1; i < L; i++) block_ids[i] = c->dflash_mask_token;
        layer_out = (mlx_array*)malloc((size_t)rt->n_layers * sizeof(mlx_array));
        for (int i = 0; i < rt->n_layers; i++) layer_out[i] = mlx_array_new();
        ok = df2_backbone(rt, block_ids, L, x_ctx, S, &hidden, layer_out);
    }
    mlx_array_free(x_ctx);
    if (block_ids) free(block_ids);
    if (!ok) {
        if (layer_out) for (int i = 0; i < rt->n_layers; i++) mlx_array_free(layer_out[i]);
        if (layer_out) free(layer_out);
        mlx_array_free(hidden);
        df2_mode_leave_draft(rt, sv_all, sv_act);
        return -1;
    }

    int written = 0;
    for (int l = 0; ok && l < rt->n_layers; l++) {
        mlx_array h32 = mlx_array_new();
        ok = mlx_astype(&h32, layer_out[l], MLX_FLOAT32, st) == 0;
        if (ok) ok = mlx_array_eval(h32) == 0;
        const float* data = ok ? mlx_array_data_float32(h32) : NULL;
        if (data == NULL || (int)mlx_array_size(h32) != per_snap) ok = false;
        else memcpy(out + written, data, (size_t)per_snap * sizeof(float));
        written += per_snap;
        mlx_array_free(h32);
        mlx_array_free(layer_out[l]);
    }
    free(layer_out);
    {
        mlx_array h32 = mlx_array_new();
        if (ok) ok = mlx_astype(&h32, hidden, MLX_FLOAT32, st) == 0;
        if (ok) ok = mlx_array_eval(h32) == 0;
        const float* data = ok ? mlx_array_data_float32(h32) : NULL;
        if (data == NULL || (int)mlx_array_size(h32) != per_snap) ok = false;
        else memcpy(out + written, data, (size_t)per_snap * sizeof(float));
        written += per_snap;
        mlx_array_free(h32);
    }
    mlx_array_free(hidden);
    df2_mode_leave_draft(rt, sv_all, sv_act);
    return ok ? written : -1;
}

/* ═══ Round-loop DFlash2 nativo: capture target ai layer dflash + verify + walk. ═══ */

/* forward batch del target con capture post-layer (hidden [1,n,H] fp32). Nessun eval
 * a meta' grafo: gli snap (fp32 lazy) si accumulano e si materializzano TUTTI insieme a
 * fine forward (evitare eval di un subgraph mentre il grafo e' in costruzione e' la
 * differenza rispetto alla prima versione che crashava). capout [n_tokens*ncap*H]. */
static bool df2_capture_batch(LlmMlxModel* m, LlmMlxSession* kvq_sess, mlx_array ew, mlx_array es, mlx_array eb,
                              const int32_t* ids, int n_tokens, int cache_offset,
                              mlx_array* kc, mlx_array* vc, bool* hasc,
                              const int* caplayers, int ncap,
                              float* capout, int capcap, mlx_array* out_h) {
    /* SPAN-CHUNK (2026-10-01): se n_tokens supera lo span, spezza il forward in span
     * consecutivi sulla STESSA cache scratch (kc/vc/hasc persistono, cache_offset avanza).
     * Evita l'attenzione [H, S, S] sull'intero contesto (OOM a S grande). */
    if (getenv("DF2_DEBUG"))
        fprintf(stderr, "[df2-cap] n_tokens=%d cache_offset=%d span=%d\n", n_tokens, cache_offset, df2_ctx_span());
    {
        const int span = df2_ctx_span();
        if (span > 0 && n_tokens > span) {
            bool all = true;
            for (int off = 0; off < n_tokens && all; off += span) {
                int ch = (n_tokens - off > span) ? span : (n_tokens - off);
                float* co = (capout != NULL) ? capout + (size_t)off * (size_t)ncap * m->cfg.hidden_size : NULL;
                int cc = (co != NULL) ? (int)((int64_t)ch * ncap * m->cfg.hidden_size) : 0;
                all = df2_capture_batch(m, kvq_sess, ew, es, eb, ids + off, ch, cache_offset + off,
                                        kc, vc, hasc, caplayers, ncap, co, cc, out_h);
            }
            return all;
        }
    }
    mlx_array x = mlx_array_new();
    if (!embed_tokens_batch(m, ew, es, eb, ids, n_tokens, &x)) { mlx_array_free(x); return false; }
    mlx_array h = x;
    bool ok = true;
    /* SIDE-CHANNEL PER-CALL (fix 2026-09-16): lo STATO RICORRENTE degli ibridi (qwen3_5:
     * 48 GDN + 16 full-attn) vive in kvq_sess->aux_cache, e il forward normale lo imposta
     * SEMPRE prima di entrare nei layer (`m->aux_cache = s->aux_cache`, 4 siti in generate).
     * Qui mancava -> layer_forward leggeva un aux_cache stantio/NULL e faceva mlx_array_set
     * su un puntatore invalido: **SIGSEGV** (verificato con lldb su Qwen3.8-27B + DFlash2:
     * EXC_BAD_ACCESS in mlx_array_set). Stesso pattern di q4e_ctx_ids (PLE/ngram). */
    mlx_array* saved_aux = m->aux_cache;
    const int32_t* saved_q4e = m->q4e_ctx_ids;
    int saved_q4e_len = m->q4e_ctx_len;
    m->aux_cache = kvq_sess->aux_cache;
    m->q4e_ctx_ids = kvq_sess->history;
    /* limite CAUSALE del ctx PLE/ngram: i token elaborati finora (cache_offset + n_tokens),
     * non l'intera history. Con lo span-chunk ogni span vede solo il proprio prefisso
     * (evita che i primi span vedano token futuri); senza chunk (offset 0) coincide. */
    m->q4e_ctx_len = (cache_offset + n_tokens <= kvq_sess->history_len)
                         ? cache_offset + n_tokens : kvq_sess->history_len;
    mlx_array snaps[8];
    int snaps_n = 0;
    for (int l = 0; ok && l < m->cfg.n_layers; l++) {
        mlx_array layer_out = mlx_array_new();
        ok = layer_forward(m, kvq_sess, l, cache_offset, n_tokens, h, &kc[l], &vc[l], hasc[l], &layer_out);
        hasc[l] = true;
        mlx_array_free(h);
        h = layer_out;
        if (ok && g_layerh_run != NULL && getenv("LLM_GDN_AB"))
            fprintf(stderr, "[layerh] run=%s L=%d sum=%.6g\n", g_layerh_run, l, llm_mlx_checksum_f32(h, m->stream));
        if (ok && caplayers != NULL && snaps_n < 8) {
            for (int c = 0; c < ncap; c++)
                if (caplayers[c] == l) {
                    mlx_array s32 = mlx_array_new();
                    ok = mlx_astype(&s32, h, MLX_FLOAT32, m->stream) == 0;
                    if (ok) snaps[snaps_n++] = s32; else mlx_array_free(s32);
                    break;
                }
        }
    }
    if (ok && snaps_n == ncap) {
        /* materializza tutti gli snap insieme e copia in capout riga per posizione */
        for (int i = 0; ok && i < snaps_n; i++) ok = mlx_array_eval(snaps[i]) == 0;
        int64_t need = (int64_t)n_tokens * ncap * m->cfg.hidden_size;
        if (ok && need > 0 && (capout == NULL || (int64_t)capcap < need)) ok = false;
        if (ok) {
            for (int c = 0; c < ncap; c++) {
                const float* d = mlx_array_data_float32(snaps[c]);
                if (!d) { ok = false; break; }
                size_t row = (size_t)n_tokens * m->cfg.hidden_size;
                for (int p = 0; p < n_tokens; p++)
                    memcpy(capout + ((size_t)p * ncap + (size_t)c) * m->cfg.hidden_size,
                           d + (size_t)p * m->cfg.hidden_size, (size_t)m->cfg.hidden_size * sizeof(float));
            }
        }
    }
    if (ok) mlx_array_set(out_h, h);
    for (int i = 0; i < snaps_n; i++) mlx_array_free(snaps[i]);
    mlx_array_free(h);
    m->aux_cache = saved_aux;          /* ripristino del side-channel */
    m->q4e_ctx_ids = saved_q4e;
    m->q4e_ctx_len = saved_q4e_len;
    return ok;
}

/* verify DFlash2: forward batch [anchor]+draft sul target con capture, greedy per
 * posizione, walk reference, commit/rollback della KV. ctx_out (fp32) riceve le feature
 * delle posizioni confermate (anchor+accepted). out_next = greedy[accepted]. */
/* DF2_TIME (2026-09-16): timer di FASE del round speculativo, per attribuire i 611 ms/round
 * misurati (0,26-0,41x). Fasi: checkpoint KV (per-layer, ogni round), forward di verify
 * (anchor+drafts + capture), argmax/logits, rollback+re-append (solo se accepted<K: serve a
 * ricostruire lo STATO RICORRENTE GDN, non le feature ctx che vengono dal primo capture). */

static int df2_dflash_verify(LlmMlxSession* s, int anchor, const int32_t* d, int K,
                             const int* caplayers, int ncap,
                             float* ctx_out, int ctx_cap, int32_t* out_next) {
    LlmMlxModel* m = s->model;
    int n_prev = s->cache_offset;
    const int NL = m->cfg.n_layers;
    OutputTensors t;
    if (!load_output_tensors(m, &t)) return -1;
    const int N = K + 1;
    int32_t* ids = (int32_t*)malloc((size_t)N * sizeof(int32_t));
    ids[0] = anchor;
    for (int i = 0; i < K; i++) ids[i + 1] = d[i];

    /* checkpoint dell'intera KV (gli array gdn possono non essere 4D: niente slice) */
    mlx_array* ck = (mlx_array*)malloc((size_t)NL * sizeof(mlx_array));
    mlx_array* cv = (mlx_array*)malloc((size_t)NL * sizeof(mlx_array));
    /* FIX parita' DFlash2 (2026-09-19): lo STATO RICORRENTE degli ibridi GDN (qwen3_5) vive in
     * s->aux_cache, non in k_cache: senza checkpoint/rollback anche di quello, sui reject lo stato
     * GDN resta quello del forward batch (K+1 token) e il decode successivo drifta dal greedy
     * (misurato: divergenza a 93/143). Si salva/ripristina per layer come k/v. */
    double _t0 = g_df2_time_on() ? g_df2_now() : 0.0;
    /* FIX (2026-10-01): stato ricorrente GDN in s->aux_cache. Vuoto = ctx NULL (mlx_array_={ctx}),
     * quindi mai usare ndim/size (lanciano). Solo le righe NON vuote si checkpoíntano. */
    mlx_array* ca = (s->aux_cache != NULL) ? (mlx_array*)malloc((size_t)NL * sizeof(mlx_array)) : NULL;
    bool* has_aux = (ca != NULL) ? (bool*)calloc((size_t)NL, sizeof(bool)) : NULL;
    for (int l = 0; l < NL; l++) { ck[l] = mlx_array_new(); cv[l] = mlx_array_new();
        llm_mlx_cache_checkpoint(&ck[l], s->k_cache[l], m->stream);
        llm_mlx_cache_checkpoint(&cv[l], s->v_cache[l], m->stream);
        if (ca != NULL) { ca[l] = mlx_array_new();
            if (s->aux_cache[l].ctx != NULL) {
                llm_mlx_cache_checkpoint(&ca[l], s->aux_cache[l], m->stream); has_aux[l] = true;
            } } }
    if (g_df2_time_on()) { g_df2_t_ckpt += g_df2_now() - _t0; _t0 = g_df2_now(); }

    /* A/B bit-level GDN (LLM_GDN_AB, una volta per processo): forward PULITO via
     * forward_tokens_batch (run "A") sugli STESSI ids, poi rollback; il run "B" e' la cattura
     * normale. Il diff dei checksum per-layer [layerh] trova la PRIMA divergenza. */
    if (getenv("LLM_GDN_AB")) {
        static int g_ab_done = 0;
        if (!g_ab_done) {
            g_ab_done = 1;
            g_layerh_run = "A";
            mlx_array hA = mlx_array_new();
            (void)forward_tokens_batch(m, s, t.embed_w, t.embed_s, t.embed_b, ids, N, n_prev,
                                       s->k_cache, s->v_cache, s->has_cache, &hA);
            mlx_array_free(hA);
            for (int l = 0; l < NL; l++) {
                llm_mlx_cache_rollback(&s->k_cache[l], ck[l], m->stream);
                llm_mlx_cache_rollback(&s->v_cache[l], cv[l], m->stream);
                if (ca != NULL && has_aux[l]) llm_mlx_cache_rollback(&s->aux_cache[l], ca[l], m->stream);
            }
            g_layerh_run = "B";
        }
    }

    int64_t capneed = (int64_t)N * ncap * m->cfg.hidden_size;
    float* cap = (float*)malloc((size_t)capneed * sizeof(float));
    mlx_array h = mlx_array_new();
    bool ok;
    if (df2_seq_verify_on()) {
        /* ESPERIMENTO (parita' DFlash2 vs greedy): feed 1 token per volta (come il decode greedy)
         * invece del batch N=K+1. Se l'output torna == greedy, la divergenza e' la numerica del
         * forward batch; se resta, e' la logica del round-loop. A/B diagnostico. */
        ok = true;
        for (int i = 0; i < N && ok; i++) {
            float* cpi = (caplayers != NULL) ? cap + (size_t)i * ncap * m->cfg.hidden_size : NULL;
            mlx_array hi = mlx_array_new();
            ok = df2_capture_batch(m, s, t.embed_w, t.embed_s, t.embed_b, ids + i, 1, n_prev + i,
                                   s->k_cache, s->v_cache, s->has_cache,
                                   cpi ? caplayers : NULL, cpi ? ncap : 0,
                                   cpi, cpi ? (int)((int64_t)ncap * m->cfg.hidden_size) : 0, &hi);
            mlx_array_free(h);
            h = hi;
        }
    } else if (llm_ai_mlx_opt_env("LLM_DF2_ASTYLE_VERIFY")) {
        /* ESPERIMENTO/fix: forward in stile (A) — arch->attn per-sessione a K token + FFN batched.
         * Sul n-gram questo path era BIT-EXACT col greedy (143/143), a differenza di
         * layer_forward con K+1 token. Stessa cattura ctx per-sessione. */
        LlmMlxSession* sess1[1] = { s };
        int ks1[1] = { N };
        int offs1[1] = { 0 };
        ok = forward_tokens_batch_multi_cap(m, sess1, ks1, 1, ids, offs1,
                                            t.embed_w, t.embed_s, t.embed_b,
                                            caplayers, ncap, cap, (int)capneed, &h);
    } else if (getenv("LLM_DF2_FWD_HTB")) {
        /* ESPERIMENTO: greedy col forward "normale" forward_tokens_batch (quello della n-gram,
         * senza snap di cattura ctx). Ipotesi: la cattura degli snap perturba il forward GDN
         * (il drafter propone il token giusto ma il verify no). Qui il ctx resta vuoto -> il
         * propose non propone; serve solo a isolare il GREEDY e confrontarlo con l'AR. */
        ok = forward_tokens_batch(m, s, t.embed_w, t.embed_s, t.embed_b, ids, N, n_prev,
                                  s->k_cache, s->v_cache, s->has_cache, &h);
    } else {
        /* FIX (2026-10-01): il GREEDY si calcola col forward "normale" `forward_tokens_batch`
         * (quello della n-gram, corretto sui GDN). La cattura degli snap di ctx perturba il
         * forward ricorrente GDN (verificato con dump token-id: col capture il greedy diverge
         * dall'AR, senza capture e' bit-identico). Poi si RIPRISTINA il checkpoint e si rifa'
         * la cattura SOLO per il ctx del drafter (il suo h non serve). */
        ok = forward_tokens_batch(m, s, t.embed_w, t.embed_s, t.embed_b, ids, N, n_prev,
                                  s->k_cache, s->v_cache, s->has_cache, &h);
        if (ok && caplayers != NULL) {
            /* ctx su CACHE SCRATCH (copia REALE via mlx_copy): la cattura (astype/eval + update
             * KV in-place) non deve toccare lo stato di sessione. La KV di sessione resta quella
             * di forward_tokens_batch; l'aux di sessione e' sostituito dalla copia solo durante
             * la cattura (df2_capture_batch fa m->aux_cache = kvq_sess->aux_cache). */
            mlx_array* sk = (mlx_array*)malloc((size_t)NL * sizeof(mlx_array));
            mlx_array* sv = (mlx_array*)malloc((size_t)NL * sizeof(mlx_array));
            mlx_array* sa = (s->aux_cache != NULL) ? (mlx_array*)malloc((size_t)NL * sizeof(mlx_array)) : NULL;
            bool* shasc = (bool*)calloc((size_t)NL, sizeof(bool));
            if (sk == NULL || sv == NULL || shasc == NULL) ok = false;
            for (int l = 0; ok && l < NL; l++) {
                sk[l] = mlx_array_new(); sv[l] = mlx_array_new();
                if (mlx_copy(&sk[l], s->k_cache[l], m->stream) != 0) ok = false;
                if (ok && mlx_copy(&sv[l], s->v_cache[l], m->stream) != 0) ok = false;
                shasc[l] = s->has_cache[l];
                if (sa != NULL) { sa[l] = mlx_array_new();
                    if (s->aux_cache[l].ctx != NULL && mlx_copy(&sa[l], s->aux_cache[l], m->stream) != 0) ok = false; }
            }
            mlx_array* saved_aux = s->aux_cache;
            if (ok) s->aux_cache = sa;
            mlx_array hd = mlx_array_new();
            if (ok) ok = df2_capture_batch(m, s, t.embed_w, t.embed_s, t.embed_b, ids, N, n_prev,
                                           sk, sv, shasc, caplayers, ncap, cap, (int)capneed, &hd);
            s->aux_cache = saved_aux;
            mlx_array_free(hd);
            for (int l = 0; l < NL; l++) {
                mlx_array_free(sk[l]); mlx_array_free(sv[l]);
                if (sa != NULL) mlx_array_free(sa[l]);
            }
            free(sk); free(sv); free(sa); free(shasc);
        }
    }
    if (g_df2_time_on()) { g_df2_t_fwd += g_df2_now() - _t0; _t0 = g_df2_now(); }
    mlx_array normed = mlx_array_new(), logits = mlx_array_new(), am = mlx_array_new();
    if (ok) {
        if (llm_ai_mlx_opt_env("LLM_DF2_FASTRMS"))
            ok = mlx_fast_rms_norm(&normed, h, t.final_norm_w, m->cfg.rms_norm_eps, m->stream) == 0;
        else
            ok = llm_mlx_final_norm(m, h, t.final_norm_w, &normed);
    }
    if (ok) ok = quantized_linear_cfg(&logits, normed, t.out_w, t.out_s, t.out_b,
                                      m->cfg.quant_group_size, m->cfg.quant_bits, m->cfg.quant_mode, m->stream);
    if (ok) ok = mlx_argmax_axis(&am, logits, -1, /*keepdims=*/false, m->stream) == 0;
    if (ok) ok = mlx_array_eval(am) == 0;
    if (g_df2_time_on()) { g_df2_t_arg += g_df2_now() - _t0; _t0 = g_df2_now(); }
    int32_t* greedy = NULL;
    if (ok) {
        const uint32_t* g = mlx_array_data_uint32(am);
        if (g) { greedy = (int32_t*)malloc((size_t)N * sizeof(int32_t)); for (int i = 0; i < N; i++) greedy[i] = (int32_t)g[i]; }
        else ok = false;
    }
    mlx_array_free(h); mlx_array_free(normed); mlx_array_free(logits); mlx_array_free(am);

    int accepted = 0;
    if (ok && K > 0 && greedy[0] == d[0]) {
        accepted = 1;
        for (int j = 1; j < K; j++) { if (greedy[j] == d[j]) accepted++; else break; }
    }
    if (s) { s->spec_drafts += K; s->spec_accepted += accepted; }   /* F20: acceptance cumulativa */
    if (ok && getenv("DF2_VDBG")) {
        static int vn = 0;
        if (vn < 100000) { vn++;
            fprintf(stderr, "[df2vdbg] n_prev=%d anchor=%d K=%d accepted=%d rollback=%d emitted_start=%d\n",
                    n_prev, anchor, K, accepted, (accepted<K)?1:0, n_prev+1);
        }
    }
    if (ok && getenv("DF2_GREEDY")) {
        fprintf(stderr, "[df2greedy] anchor=%d K=%d acc=%d\n  greedy[:]=", anchor, K, accepted);
        for (int i = 0; i <= K && i < 10; i++) fprintf(stderr, " %d", greedy[i]);
        fprintf(stderr, "\n  drafts[:]=");
        for (int i = 0; i < K && i < 10; i++) fprintf(stderr, " %d", d[i]);
        fprintf(stderr, "\n");
    }
    int keep = n_prev + 1 + accepted;
    if (g_df2_time_on()) g_df2_n_rounds++;
    if (ok && accepted < K) {
        if (g_df2_time_on()) g_df2_n_roll++;
        /* rollback = ripristino checkpoint + ri-append del SOLO prefisso confermato
         * (anchor + accettati); il rollback per-slice non vale sugli array gdn 3D. */
        for (int l = 0; l < NL; l++) {
            llm_mlx_cache_rollback(&s->k_cache[l], ck[l], m->stream);
            llm_mlx_cache_rollback(&s->v_cache[l], cv[l], m->stream);
            if (ca != NULL && has_aux[l]) llm_mlx_cache_rollback(&s->aux_cache[l], ca[l], m->stream);
        }
        int P = accepted + 1;   /* [anchor] + accettati */
        mlx_array ph = mlx_array_new();
        if (llm_ai_mlx_opt_env("LLM_DF2_ASTYLE_VERIFY")) {
            LlmMlxSession* ss1[1] = { s };
            int kk1[1] = { P }; int of1[1] = { 0 };
            if (!forward_tokens_batch_multi_cap(m, ss1, kk1, 1, ids, of1,
                                                t.embed_w, t.embed_s, t.embed_b,
                                                NULL, 0, NULL, 0, &ph)) ok = false;
        } else if (llm_ai_mlx_opt_env("LLM_DF2_SEQ_REAPPEND")) {
            /* DIAGNOSI: re-append SEQUENZIALE (1 token per volta, come il greedy) invece del
             * batch P: se la parita' torna, il difetto e' nella numerica GDN multi-token. */
            for (int i = 0; i < P && ok; i++) {
                mlx_array phi = mlx_array_new();
                ok = df2_capture_batch(m, s, t.embed_w, t.embed_s, t.embed_b, ids + i, 1, n_prev + i,
                                       s->k_cache, s->v_cache, s->has_cache,
                                       /*caplayers=*/NULL, 0, NULL, 0, &phi);
                mlx_array_free(phi);
            }
        } else if (!df2_capture_batch(m, s, t.embed_w, t.embed_s, t.embed_b, ids, P, n_prev,
                                      s->k_cache, s->v_cache, s->has_cache,
                                      /*caplayers=*/NULL, 0, NULL, 0, &ph)) {
            ok = false;
        }
        mlx_array_free(ph);
        if (g_df2_time_on()) g_df2_t_roll += g_df2_now() - _t0;
    }
    s->cache_offset = keep;
    if (ok && ctx_out != NULL) {
        int rows = accepted + 1;
        int64_t cneed = (int64_t)rows * ncap * m->cfg.hidden_size;
        if ((int64_t)ctx_cap >= cneed) memcpy(ctx_out, cap, (size_t)cneed * sizeof(float));
        else ok = false;
    }
    if (ok && out_next != NULL && greedy != NULL) *out_next = greedy[accepted];
    free_output_tensors(&t);
    free(ids); free(cap); free(greedy);
    for (int l = 0; l < NL; l++) { mlx_array_free(ck[l]); mlx_array_free(cv[l]); if (ca != NULL) mlx_array_free(ca[l]); }
    free(ck); free(cv); free(ca); free(has_aux);
    return ok ? accepted : -1;
}

/* A1: conserva il ctx del draft (ultimo vctx) per la continuazione; libera il precedente. */
static void df2_retain_ctx(DFlash2Runtime* rt, float* ctx, int n) {
    if (rt->retained_ctx) free(rt->retained_ctx);
    rt->retained_ctx = ctx; rt->retained_n = n;
}

/* STREAMING INTERLEAVED (2026-10-01): inietta il ctx del prompt nel draft PER SPAN —
 * cattura uno span alla volta su cache scratch e lo libera subito — invece di catturare
 * l'intero ctx in un buffer host (n_prompt*ncap*H*4: 26GB a 257k, 37GB a 2M). Cosi' il picco
 * host e' un solo span (~420MB). Correttezza invariata (il verify resta esatto; cambia solo
 * la memoria). La cache scratch (pk/pv/ph) PERSISTE tra gli span -> ogni span vede il prefisso
 * (attenzione corretta). */
static bool df2_stream_prompt_ctx(DFlash2Runtime* rt, LlmMlxModel* m, LlmMlxSession* s,
                                  int32_t anchor, const int* caplayers, int ncap, int n_prompt,
                                  float** out_tail, int* out_tail_n) {
    *out_tail = NULL; *out_tail_n = 0;
    if (n_prompt <= 0) return true;
    const int H = m->cfg.hidden_size;
    const int ctx_dim = ncap * H;
    const int L = rt->draft->cfg.dflash_block_size;
    int span = df2_ctx_span(); if (span <= 0) span = 4096;
    /* L'ULTIMO span NON si inietta qui: resta al propose (round-1) che cosi' ha comunque un
     * ctx non vuoto (evita il ramo S==0 di df2_backbone) e sfrutta la macchina d'iniezione
     * esistente. Il buffer restituito e' un solo span (~420MB) invece dell'intero contesto. */
    const int tail = (n_prompt < span) ? n_prompt : span;
    const int inject_upto = n_prompt - tail;
    OutputTensors t;
    if (!load_output_tensors(m, &t)) return false;
    const int tnl = m->cfg.n_layers;
    mlx_array* pk = (mlx_array*)malloc((size_t)tnl * sizeof(mlx_array));
    mlx_array* pv = (mlx_array*)malloc((size_t)tnl * sizeof(mlx_array));
    bool* ph = (bool*)calloc((size_t)tnl, sizeof(bool));
    int32_t* block_ids = (int32_t*)malloc((size_t)L * sizeof(int32_t));
    if (!pk || !pv || !ph || !block_ids) { free(pk); free(pv); free(ph); free(block_ids); free_output_tensors(&t); return false; }
    for (int i = 0; i < tnl; i++) { pk[i] = mlx_array_new(); pv[i] = mlx_array_new(); }
    block_ids[0] = anchor;
    for (int i = 1; i < L; i++) block_ids[i] = rt->draft->cfg.dflash_mask_token;
    bool ok = true;
    for (int off = 0; off < inject_upto && ok; off += span) {
        int ch = (inject_upto - off > span) ? span : (inject_upto - off);
        float* buf = (float*)malloc((size_t)ch * (size_t)ctx_dim * sizeof(float));
        if (!buf) { ok = false; break; }
        mlx_array phhidden = mlx_array_new();
        ok = df2_capture_batch(m, s, t.embed_w, t.embed_s, t.embed_b, s->history + off, ch, off,
                               pk, pv, ph, caplayers, ncap, buf, ch * ctx_dim, &phhidden);
        mlx_array_free(phhidden);
        if (ok) {
            mlx_array ctx_arr = mlx_array_new_data(buf, ((const int[2]){ch, ctx_dim}), 2, MLX_FLOAT32);
            mlx_array xc = mlx_array_new();
            ok = mlx_reshape(&xc, ctx_arr, ((const int[3]){1, ch, ctx_dim}), 3, m->stream) == 0;
            mlx_array_free(ctx_arr);
            if (ok) {
                bool sv_all; mlx_dtype sv_act;
                df2_mode_enter_draft(rt, &sv_all, &sv_act);
                mlx_array h2 = mlx_array_new();
                ok = df2_backbone(rt, block_ids, L, xc, ch, &h2, NULL);
                mlx_array_free(h2);
                df2_mode_leave_draft(rt, sv_all, sv_act);
                if (ok) rt->ctx[rt->cur_ctx].ctx_pos += ch;
            }
            mlx_array_free(xc);
        }
        free(buf);
        if (getenv("DF2_CTXDBG"))
            fprintf(stderr, "[stream] off=%d ch=%d ok=%d ctx_pos=%d\n", off, ch, (int)ok, rt->ctx[rt->cur_ctx].ctx_pos);
    }
    /* cattura l'ULTIMO span in un buffer (NON iniettato): lo inietta il propose round-1. */
    if (ok && tail > 0) {
        float* buf = (float*)malloc((size_t)tail * (size_t)ctx_dim * sizeof(float));
        if (!buf) ok = false;
        else {
            mlx_array phhidden = mlx_array_new();
            ok = df2_capture_batch(m, s, t.embed_w, t.embed_s, t.embed_b, s->history + inject_upto, tail,
                                   inject_upto, pk, pv, ph, caplayers, ncap, buf, tail * ctx_dim, &phhidden);
            mlx_array_free(phhidden);
            if (ok) { *out_tail = buf; *out_tail_n = tail; } else free(buf);
        }
    }
    for (int i = 0; i < tnl; i++) { mlx_array_free(pk[i]); mlx_array_free(pv[i]); }
    free(pk); free(pv); free(ph); free(block_ids); free_output_tensors(&t);
    return ok;
}

int llm_ai_mlx_dflash2_generate(llm_ai_mlx_dflash2_handle handle,
                                llm_ai_mlx_session_handle sess_handle,
                                int32_t eos_token, int max_tokens,
                                int32_t* out_ids) {
    if (handle == NULL || sess_handle == NULL || out_ids == NULL || max_tokens <= 0) return -1;
    DFlash2Runtime* rt = (DFlash2Runtime*)handle;
    LlmMlxModel* sm = ((LlmMlxSession*)sess_handle)->model;
    if (sm != rt->target) return -1;
    LlmMlxSession* s = (LlmMlxSession*)sess_handle;
    LlmMlxModel* m = rt->target;
    if (!s->has_pending) return -1;
    const int ncap = rt->draft->cfg.dflash_n_ctx_layers;
    if (ncap <= 0 || ncap > 8) return -1;
    int caplayers[8];
    for (int i = 0; i < ncap; i++) caplayers[i] = rt->draft->cfg.dflash_ctx_layers[i];
    const int H = m->cfg.hidden_size;
    const int n_prompt = s->cache_offset;
    if (n_prompt <= 0) return -1;

    /* LARGE-CONTEXT GATE (2026-10-01): la cattura ctx del prompt tiene n_prompt*ncap*H*4 byte
     * in fp32 (a 257k token = 26GB host) + la KV scratch del target -> a contesto grande supera
     * la RAM (OOM OS a 1M). Oltre il budget si rinuncia al DFlash2 per questa sessione: il
     * chiamante fa FALLBACK alla spec n-gram / decode AR (percorso verificato a 257k, 38.6GB).
     * 0 = nessun limite (forza). Override: opzione di registro `df2_ctx_budget_mb`. */
    {
        /* Il ctx del prompt e' iniettato per SPAN (streaming interleaved, df2_stream_prompt_ctx):
         * picco host ~1 span (~420MB) invece di 26-37GB. Resta pero' il PICCO PREFLILL (a 257k
         * ~80GB attivi, non ancora rilasciato) + drafter + KV scratch -> 1M/2M fanno ancora OOM.
         * Quindi si mantiene un gate (default 8192MB sull'ampiezza del ctx) che fa FALLBACK a
         * n-gram/AR oltre ~82k token finche' il picco prefill non e' gestito. */
        int budget_mb = 8192;
        if (llm_ai_mlx_opt_get_int("df2_ctx_budget_mb", &budget_mb) != 0 || budget_mb < 0) budget_mb = 8192;
        if (budget_mb > 0) {
            const uint64_t need = (uint64_t)n_prompt * (uint64_t)ncap * (uint64_t)H * (uint64_t)sizeof(float);
            if (need > (uint64_t)budget_mb * 1048576ull) {
                static int warned = 0;
                if (!warned++) fprintf(stderr, "[df2] ctx prompt troppo grande (%.1fGB > budget %.1fGB): "
                                "DFlash2 OFF per questa sessione -> fallback n-gram/AR. Alza "
                                "df2_ctx_budget_mb (0=forza) per tentare comunque.\n",
                                need / 1073741824.0, budget_mb / 1024.0);
                return -1;
            }
        }
    }

    /* A1 — CONTESTO PERSISTENTE. Se questa e' la CONTINUAZIONE della stessa sessione (nessun
     * token appeso dal chiamante: cache_offset invariato dall'ultima chiamata), riusiamo il ctx
     * del draft gia' nel ring + l'ultimo vctx, SENZA ri-catturare il prompt ne' resettare il ring. */
    const bool cont = (rt->last_session == (void*)s) && (rt->last_offset == s->cache_offset)
                      && (rt->retained_ctx != NULL) && (rt->retained_n > 0);
    float* pctx = NULL;
    int ctx_n0 = 0;
    if (cont) {
        ctx_n0 = rt->retained_n;
        pctx = (float*)malloc((size_t)ctx_n0 * ncap * H * sizeof(float));
        if (!pctx) return -1;
        memcpy(pctx, rt->retained_ctx, (size_t)ctx_n0 * ncap * H * sizeof(float));
    } else {
        /* ctx del prompt: re-forward della history su cache scratch con capture (eval a fine
         * forward: capture robusta). */
        /* STREAMING INTERLEAVED (2026-10-01): il ctx del prompt viene iniettato nel draft
         * PER SPAN — cattura su cache scratch persistente, iniezione, free per span — invece
         * di tenere l'intero ctx in un buffer host (n_prompt*ncap*H*4: 26GB@257k, 37GB@2M ->
         * OOM). Picco = un solo span. Il propose round-1 gira poi con S=0 (ctx gia' in cache:
         * niente re-iniezione). Il prefill puo' lasciare un picco transitorio: clear prima. */
        mlx_clear_cache();
        llm_ai_mlx_dflash2_reset(handle);   /* nuova sequenza: ring del draft pulito */
        float* tail = NULL; int tail_n = 0;
        if (!df2_stream_prompt_ctx(rt, m, s, s->pending_next_id, caplayers, ncap, n_prompt, &tail, &tail_n)) {
            fprintf(stderr, "dflash_generate: streaming ctx fallito\n");
            return -1;
        }
        pctx = tail; ctx_n0 = tail_n;   /* solo l'ULTIMO span -> il propose round-1 inietta quello */
    }

    int emitted = 0;
    int stat_rounds = 0, stat_accepted = 0, stat_K = 0;
    int32_t anchor = s->pending_next_id;   /* bonus = greedy dopo il prompt (o pending in continuazione) */
    float* ctx_r = pctx;
    int ctx_n = ctx_n0;

    while (emitted < max_tokens) {
        if (anchor == eos_token) break;   /* eos NON emesso (come session_generate) */
        out_ids[emitted++] = anchor;
        /* NB (fix 2026-10-01): il verify gira SEMPRE, anche se il chunk e' pieno: serve a
         * calcolare `next_anchor` (= il pending per la chiamata successiva). Il vecchio break
         * anticipato lasciava pending=anchor (gia' emesso) -> il token veniva RI-EMESSO al round
         * successivo (bug misurato "shifting shifting" a cavallo dei chunk da 16). I draft oltre
         * il tetto non vengono emessi (loop sotto). */

        int32_t drafts[8];
        LLM_DF2_SP_BEGIN(prop);
        int K = llm_ai_mlx_dflash2_propose(handle, anchor, ctx_r, ctx_n, drafts, 8);
        LLM_DF2_SP_END(prop);
        if (K <= 0) break;
        /* I draft oltre il tetto del chunk non vanno emessi (altrimenti la loro KV resta in cache
         * e `cache_offset` scavalca i token non emessi -> salto). K puo' diventare 0: in tal caso
         * il verify gira lo stesso con N=1 (draft vuoti) solo per ottenere `next_anchor`. */
        if (K > max_tokens - emitted) K = max_tokens - emitted;

        int64_t vctx_cap = (int64_t)(K + 1) * ncap * H;
        float* vctx = (float*)malloc((size_t)vctx_cap * sizeof(float));
        int32_t next_anchor = -1;
        LLM_DF2_SP_BEGIN(ver);
        int accepted = df2_dflash_verify(s, anchor, drafts, K, caplayers, ncap,
                                         vctx, (int)vctx_cap, &next_anchor);
        LLM_DF2_SP_END(ver);
        if (accepted < 0 || next_anchor < 0) { free(vctx); break; }
        stat_rounds++; stat_accepted += accepted; stat_K += K;
        if (g_df2_time_on())
            fprintf(stderr, "[df2time] r=%d acc=%d/%d | ckpt=%.0f fwd=%.0f arg=%.0f roll=%.0f PROP=%.0f (bb=%.0f sel=%.0f wget=%.0f argp=%.0f) ms\n",
                    stat_rounds, accepted, K, g_df2_t_ckpt * 1e3, g_df2_t_fwd * 1e3,
                    g_df2_t_arg * 1e3, g_df2_t_roll * 1e3, g_df2_t_prop * 1e3, g_df2_t_bb * 1e3, g_df2_t_sel * 1e3, g_df2_t_wget * 1e3, g_df2_t_argp * 1e3);

        bool stop_eos = false;
        int emit_n = 0;
        for (int i = 0; i < accepted && emitted < max_tokens; i++) {
            if (drafts[i] == eos_token) { stop_eos = true; break; }   /* eos non emesso */
            out_ids[emitted++] = drafts[i];
            emit_n++;
        }
        free(ctx_r);
        ctx_r = vctx;
        ctx_n = emit_n + 1;   /* il ctx deve riflettere i token EFFETTIVAMENTE emessi */
        anchor = next_anchor;
        if (stop_eos) break;
        if (emitted >= max_tokens) break;
        if (getenv("DF2_STATS"))
            fprintf(stderr, "[df2stats] r=%d accepted=%d/%d\n", stat_rounds, accepted, K);
    }

    /* A1: persistenza. Il ctx `ctx_r` (ultimo vctx; pctx se nessun round) resta nel runtime per
     * la prossima chiamata di continuazione; il pending della sessione = l'ultimo anchor. */
    df2_retain_ctx(rt, ctx_r, ctx_n);
    rt->last_session = (void*)s;
    rt->last_offset = s->cache_offset;
    s->pending_next_id = anchor;
    s->has_pending = true;
    if (getenv("DF2_STATS") && stat_rounds > 0)
        fprintf(stderr, "[df2stats] total accepted=%d/%d (%.0f%%)\n", stat_accepted, stat_K,
                100.0 * (double)stat_accepted / (double)stat_K);
    if (getenv("DF2_SEL_TIME")) {
        fprintf(stderr, "[df2seltime] argp=%.1f unary=%.1f hproj=%.1f succ=%.1f ast=%.1f ms (cumulativi)\n",
                g_df2_ts_argp*1000, g_df2_ts_unary*1000, g_df2_ts_hproj*1000, g_df2_ts_succ*1000, g_df2_ts_ast*1000);
    }
    if (g_df2_time_on() && g_df2_n_rounds > 0)
        fprintf(stderr, "[df2time] rounds=%d rollbacks=%d | ckpt=%.1f | fwd=%.1f | arg=%.1f | roll=%.1f | "
                        "PROPOSE=%.1f (backbone=%.1f selector=%.1f) ms/round\n",
                g_df2_n_rounds, g_df2_n_roll,
                g_df2_t_ckpt*1000/g_df2_n_rounds, g_df2_t_fwd*1000/g_df2_n_rounds,
                g_df2_t_arg*1000/g_df2_n_rounds, g_df2_t_roll*1000/g_df2_n_rounds,
                g_df2_t_prop*1000/g_df2_n_rounds, g_df2_t_bb*1000/g_df2_n_rounds, g_df2_t_sel*1000/g_df2_n_rounds);
    return emitted;
}

/* ═══ Tappa 3 DFlash2 multi-sessione: verify + round-loop BATCHED. ═══ */

/* Versione BATCHED di df2_dflash_verify per N sessioni (stesso target/rt). Per ogni sessione i:
 * ids_i = [anchor_i] + drafts_i[0..K_i), in UN forward N×K con cattura ctx
 * (forward_tokens_batch_multi_cap). Poi argmax per posizione, accettazione per-sessione,
 * rollback/re-append per-sessione (solo sui reject: serve allo stato ricorrente GDN), ctx_out
 * per-sessione (accepted_i+1 righe). Ritorna 0 / -1. NON eseguito/testato (verifica su modello
 * reale richiesta). */
static int df2_dflash_verify_multi(DFlash2Runtime* rt, LlmMlxSession** sess, int n,
                                   const int32_t* anchors, int32_t** drafts, const int* Ks, int maxK,
                                   const int* caplayers, int ncap,
                                   float** ctx_out, const int* ctx_cap,
                                   int32_t* out_next, int* out_accepted) {
    if (rt == NULL || sess == NULL || n <= 0 || anchors == NULL || drafts == NULL || Ks == NULL) return -1;
    LlmMlxModel* m = rt->target;
    const int H = m->cfg.hidden_size;
    const int NL = m->cfg.n_layers;
    OutputTensors t;
    if (!load_output_tensors(m, &t)) return -1;
    int total = 0;
    for (int i = 0; i < n; i++) { if (Ks[i] < 0 || Ks[i] > maxK) { free_output_tensors(&t); return -1; } total += Ks[i] + 1; }
    int32_t* ids = (int32_t*)malloc((size_t)total * sizeof(int32_t));
    int* offs = (int*)malloc((size_t)n * sizeof(int));
    int* kk = (int*)malloc((size_t)n * sizeof(int));
    int* nprev = (int*)malloc((size_t)n * sizeof(int));
    mlx_array* ck = (mlx_array*)calloc((size_t)n * NL, sizeof(mlx_array));
    mlx_array* cv = (mlx_array*)calloc((size_t)n * NL, sizeof(mlx_array));
    bool ok = (ids != NULL && offs != NULL && kk != NULL && nprev != NULL && ck != NULL && cv != NULL);
    int o = 0;
    if (ok) for (int i = 0; i < n; i++) {
        LlmMlxSession* s = sess[i];
        offs[i] = o; kk[i] = Ks[i] + 1; nprev[i] = s->cache_offset;
        ids[o++] = anchors[i];
        for (int k = 0; k < Ks[i]; k++) ids[o++] = drafts[i][k];
    }
    if (ok) for (int i = 0; i < n; i++) for (int l = 0; l < NL; l++) {
        ck[(size_t)i * NL + l] = mlx_array_new();
        cv[(size_t)i * NL + l] = mlx_array_new();
        llm_mlx_cache_checkpoint(&ck[(size_t)i * NL + l], sess[i]->k_cache[l], m->stream);
        llm_mlx_cache_checkpoint(&cv[(size_t)i * NL + l], sess[i]->v_cache[l], m->stream);
    }
    int64_t capn = (int64_t)total * ncap * H;
    float* cap = (float*)malloc((size_t)(capn > 0 ? capn : 1) * sizeof(float));
    if (cap == NULL) ok = false;
    mlx_array h = mlx_array_new();
    if (ok) ok = forward_tokens_batch_multi_cap(m, sess, kk, n, ids, offs,
                                                t.embed_w, t.embed_s, t.embed_b,
                                                caplayers, ncap, cap, (int)capn, &h);
    mlx_array normed = mlx_array_new(), logits = mlx_array_new(), am = mlx_array_new();
    if (ok) ok = llm_mlx_final_norm(m, h, t.final_norm_w, &normed);
    if (ok) ok = quantized_linear_cfg(&logits, normed, t.out_w, t.out_s, t.out_b,
                                      m->cfg.quant_group_size, m->cfg.quant_bits, m->cfg.quant_mode, m->stream);
    if (ok) ok = mlx_argmax_axis(&am, logits, -1, /*keepdims=*/false, m->stream) == 0;
    if (ok) ok = mlx_array_eval(am) == 0;
    const uint32_t* gdata = ok ? mlx_array_data_uint32(am) : NULL;
    if (ok && (gdata == NULL || mlx_array_size(am) != (size_t)total)) ok = false;
    for (int i = 0; ok && i < n; i++) {
        LlmMlxSession* s = sess[i];
        int K = Ks[i];
        const uint32_t* g = gdata + offs[i];
        int accepted = 0;
        if (K > 0 && (int32_t)g[0] == drafts[i][0]) {
            accepted = 1;
            for (int j = 1; j < K; j++) { if ((int32_t)g[j] == drafts[i][j]) accepted++; else break; }
        }
        out_accepted[i] = accepted;
        { LlmMlxSession* _s = sess[i]; if (_s) { _s->spec_drafts += K; _s->spec_accepted += accepted; } }  /* F20 */
        out_next[i] = (int32_t)g[accepted];
        int keep = nprev[i] + 1 + accepted;
        if (accepted < K) {
            for (int l = 0; l < NL; l++) {
                llm_mlx_cache_rollback(&s->k_cache[l], ck[(size_t)i * NL + l], m->stream);
                llm_mlx_cache_rollback(&s->v_cache[l], cv[(size_t)i * NL + l], m->stream);
            }
            int P = accepted + 1;
            mlx_array ph = mlx_array_new();
            if (!df2_capture_batch(m, s, t.embed_w, t.embed_s, t.embed_b, ids + offs[i], P, nprev[i],
                                   s->k_cache, s->v_cache, s->has_cache, /*caplayers=*/NULL, 0, NULL, 0, &ph))
                ok = false;
            mlx_array_free(ph);
        }
        s->cache_offset = keep;
        if (ctx_out != NULL && ctx_out[i] != NULL) {
            int rows = accepted + 1;
            int64_t cneed = (int64_t)rows * ncap * H;
            if ((int64_t)ctx_cap[i] >= cneed && cap != NULL)
                memcpy(ctx_out[i], cap + (size_t)offs[i] * ncap * H, (size_t)cneed * sizeof(float));
            else ok = false;
        }
    }
    mlx_array_free(h); mlx_array_free(normed); mlx_array_free(logits); mlx_array_free(am);
    for (int i = 0; i < n; i++) for (int l = 0; l < NL; l++) {
        llm_mlx_cache_commit(&ck[(size_t)i * NL + l], m->stream);
        llm_mlx_cache_commit(&cv[(size_t)i * NL + l], m->stream);
    }
    free(ck); free(cv); free(ids); free(offs); free(kk); free(nprev); free(cap);
    free_output_tensors(&t);
    return ok ? 0 : -1;
}

/* Round-loop DFlash2 MULTI-SESSIONE. Per ogni round: propone K_i draft per sessione (slot i),
 * UN verify batched N×K, emette anchor+draft accettati per sessione. `out_ids` e' row-major con
 * stride `stride` (= max(max_tokens)); `out_counts[i]` = token emessi per la sessione i.
 * Ritorna 0 su successo, -1 su errore. NON eseguito/testato (richiede il modello reale). */
int llm_ai_mlx_dflash2_generate_multi(llm_ai_mlx_dflash2_handle handle,
                                      llm_ai_mlx_session_handle* sessions, int n,
                                      int32_t eos_token, const int* max_tokens, int stride,
                                      int32_t* out_ids, int* out_counts) {
    if (handle == NULL || sessions == NULL || n <= 0 || n > 64 || max_tokens == NULL ||
        out_ids == NULL || out_counts == NULL || stride <= 0) return -1;
    DFlash2Runtime* rt = (DFlash2Runtime*)handle;
    LlmMlxModel* m = rt->target;
    if (df2_ctx_ensure(rt, n) < 0) return -1;
    const int ncap = rt->draft->cfg.dflash_n_ctx_layers;
    if (ncap <= 0 || ncap > 8) return -1;
    int caplayers[8];
    for (int i = 0; i < ncap; i++) caplayers[i] = rt->draft->cfg.dflash_ctx_layers[i];
    const int H = m->cfg.hidden_size;
    OutputTensors t;
    if (!load_output_tensors(m, &t)) return -1;

    LlmMlxSession* ss[64];
    float* ctx_r[64];
    int ctx_n[64], anchors[64], emitted[64], maxn[64], slot[64];
    bool done[64];
    const int tnl = m->cfg.n_layers;
    for (int i = 0; i < n; i++) {
        LlmMlxSession* s = (LlmMlxSession*)sessions[i];
        ss[i] = s; ctx_r[i] = NULL; emitted[i] = 0; slot[i] = i;
        maxn[i] = max_tokens[i] > 0 ? max_tokens[i] : 0;
        out_counts[i] = 0;
        if (s == NULL || s->model != m || !s->has_pending || maxn[i] <= 0) { done[i] = true; continue; }
        done[i] = false;
        /* ctx del prompt per questa sessione: re-forward della history su cache scratch con
         * capture (identico a llm_ai_mlx_dflash2_generate, per-sessione). */
        int np = s->cache_offset;
        int64_t pcap = (int64_t)np * ncap * H;
        float* pctx = (float*)malloc((size_t)pcap * sizeof(float));
        mlx_array* pk = (mlx_array*)malloc((size_t)tnl * sizeof(mlx_array));
        mlx_array* pv = (mlx_array*)malloc((size_t)tnl * sizeof(mlx_array));
        bool* ph = (bool*)calloc((size_t)tnl, sizeof(bool));
        mlx_array phidden = mlx_array_new();
        for (int l = 0; l < tnl; l++) { pk[l] = mlx_array_new(); pv[l] = mlx_array_new(); }
        bool okc = (pctx != NULL) && df2_capture_batch(m, s, t.embed_w, t.embed_s, t.embed_b,
                                                       s->history, np, 0, pk, pv, ph,
                                                       caplayers, ncap, pctx, (int)pcap, &phidden);
        /* DIAGNOSI (LLM_DF2_CAP_DIFF): il capture (mlx_astype per-layer come consumatore EXTRA
         * del grafo) altera l'hidden di out_h? Seconda passata SENZA capture su cache scratch
         * fresca, confronto max|h_cap - h_nocap|. Se != 0, il capture perturba la numerica
         * (candidata causa della divergenza DFlash2 vs greedy). */
        if (okc && llm_ai_mlx_opt_env("LLM_DF2_CAP_DIFF")) {
            mlx_array* qk = (mlx_array*)malloc((size_t)tnl * sizeof(mlx_array));
            mlx_array* qv = (mlx_array*)malloc((size_t)tnl * sizeof(mlx_array));
            bool* qh = (bool*)calloc((size_t)tnl, sizeof(bool));
            for (int l = 0; l < tnl; l++) { qk[l] = mlx_array_new(); qv[l] = mlx_array_new(); }
            mlx_array h2 = mlx_array_new();
            bool okc2 = df2_capture_batch(m, s, t.embed_w, t.embed_s, t.embed_b,
                                          s->history, np, 0, qk, qv, qh,
                                          /*caplayers=*/NULL, 0, NULL, 0, &h2);
            for (int l = 0; l < tnl; l++) { mlx_array_free(qk[l]); mlx_array_free(qv[l]); }
            free(qk); free(qv); free(qh);
            if (okc2) {
                mlx_array a = mlx_array_new(), b = mlx_array_new();
                if (mlx_astype(&a, phidden, MLX_FLOAT32, m->stream) == 0 &&
                    mlx_astype(&b, h2, MLX_FLOAT32, m->stream) == 0 &&
                    mlx_array_eval(a) == 0 && mlx_array_eval(b) == 0) {
                    double dmax = llm_mlx_max_abs_diff(a, b, mlx_array_size(a), m->stream);
                    fprintf(stderr, "[df2-cap-diff] max|h_cap - h_nocap| = %.6g (np=%d)\n", dmax, np);
                }
                mlx_array_free(a); mlx_array_free(b);
            }
            mlx_array_free(h2);
        }
        for (int l = 0; l < tnl; l++) { mlx_array_free(pk[l]); mlx_array_free(pv[l]); }
        free(pk); free(pv); free(ph); mlx_array_free(phidden);
        if (!okc) { free(pctx); free_output_tensors(&t); return -1; }
        ctx_r[i] = pctx; ctx_n[i] = np; anchors[i] = s->pending_next_id;
        llm_ai_mlx_dflash2_set_slot(handle, i);
        llm_ai_mlx_dflash2_reset(handle);
    }

    while (true) {
        int na = 0, act[64];
        for (int i = 0; i < n; i++) if (!done[i]) act[na++] = i;
        if (na == 0) break;
        int32_t dstore[64][8];
        int32_t* drafts[64];
        int Ks[64];
        int32_t anchorsA[64];
        int vact[64];
        int nv = 0;
        for (int a = 0; a < na; a++) {
            int i = act[a];
            if (anchors[i] == eos_token) { done[i] = true; continue; }
            out_ids[(size_t)i * stride + emitted[i]++] = anchors[i];
            if (emitted[i] >= maxn[i]) { done[i] = true; continue; }
            llm_ai_mlx_dflash2_set_slot(handle, slot[i]);
            int K = llm_ai_mlx_dflash2_propose(handle, anchors[i], ctx_r[i], ctx_n[i], dstore[nv], 8);
            if (K <= 0) { done[i] = true; continue; }
            int rem = maxn[i] - emitted[i];
            if (K > rem) K = rem;
            drafts[nv] = dstore[nv]; Ks[nv] = K; anchorsA[nv] = anchors[i]; vact[nv] = i; nv++;
        }
        if (nv == 0) continue;
        float* vctx[64];
        int vca[64];
        int32_t nextA[64];
        int accA[64];
        LlmMlxSession* vsess[64];
        bool alloc_ok = true;
        for (int a = 0; a < nv; a++) {
            int rows = Ks[a] + 1;
            vca[a] = rows * ncap * H;
            vctx[a] = (float*)malloc((size_t)vca[a] * sizeof(float));
            if (!vctx[a]) alloc_ok = false;
            vsess[a] = ss[vact[a]];
        }
        if (!alloc_ok) {
            for (int a = 0; a < nv; a++) free(vctx[a]);
            for (int i = 0; i < n; i++) if (ctx_r[i]) free(ctx_r[i]);
            free_output_tensors(&t);
            return -1;
        }
        /* Verify BATCHED multi-sessione (fix A2 2026-09-30): la root cause della divergenza era
         * la FFN batchata su `total` token (kernel MLX diverso per M, amplificato dalla GDN) ->
         * ora `forward_tokens_batch_multi_cap` usa la FFN PER-SESSIONE e il batched e' ESATTO
         * (parita' verificata). Fallback per-sessione per sliding-window/KV-quant, che arch->attn
         * da solo non gestisce (lo fa layer_forward). df2_seq_verify (registro) forza il per-sessione. */
        {
            int vfail = 0;
            bool batch_ok = !m->cfg.kv_quantize &&
                            !(m->cfg.is_sliding_layer && m->cfg.sliding_window > 0) &&
                            !df2_seq_verify_on();
            if (batch_ok) {
                if (df2_dflash_verify_multi(rt, vsess, nv, anchorsA, drafts, Ks, 8,
                                            caplayers, ncap, vctx, vca, nextA, accA) != 0) vfail = 1;
            } else {
                for (int a = 0; a < nv && !vfail; a++) {
                    int acc = df2_dflash_verify(vsess[a], anchorsA[a], drafts[a], Ks[a],
                                                caplayers, ncap, vctx[a], vca[a], &nextA[a]);
                    if (acc < 0) vfail = 1; else accA[a] = acc;
                }
            }
            if (vfail) {
                for (int a = 0; a < nv; a++) free(vctx[a]);
                for (int i = 0; i < n; i++) if (ctx_r[i]) free(ctx_r[i]);
                free_output_tensors(&t);
                return -1;
            }
        }
        for (int a = 0; a < nv; a++) {
            int i = vact[a];
            int acc = accA[a];
            for (int k = 0; k < acc && emitted[i] < maxn[i]; k++) {
                int32_t tok = dstore[a][k];
                if (tok == eos_token) { done[i] = true; break; }
                out_ids[(size_t)i * stride + emitted[i]++] = tok;
            }
            if (emitted[i] >= maxn[i]) done[i] = true;
            if (ctx_r[i]) free(ctx_r[i]);
            ctx_r[i] = vctx[a];
            ctx_n[i] = acc + 1;
            anchors[i] = nextA[a];
        }
    }
    for (int i = 0; i < n; i++) { if (ctx_r[i]) free(ctx_r[i]); out_counts[i] = emitted[i]; }
    free_output_tensors(&t);
    return 0;
}

/* Gate d'uscita RMSNormGated condiviso (sigmoid, non silu -- vedi qwen3_next_linear_block
 * sopra, unico chiamante). Le funzioni gemelle q4_zc_rmsnorm/q4_linear_helper/
 * q4_gated_residual (hyper-connection di Qwen4-Exp, mai chiamate da nessuno) sono state
 * rimosse 2026-09-06: duplicavano esattamente llm_mlx_gated_residual/
 * llm_mlx_zero_centered_rmsnorm/llm_mlx_qwen4exp_plain_linear (llm_mlx_qwen4exp.h),
 * quelle si' verificate contro numpy e quelle si' effettivamente wired nel dispatch reale
 * di qwen4_exp (qwen4_exp_attention_dispatch/ffn_dispatch, vedi Todo C17). */
bool q4_sigmoid_rmsnorm_gated(mlx_array* res, mlx_array hidden, mlx_array gate,
                                     mlx_array w, float eps, bool use_silu, mlx_stream s) {
    mlx_array n = mlx_array_new();
    bool ok = mlx_fast_rms_norm(&n, hidden, w, eps, s) == 0;
    mlx_array nf = mlx_array_new();
    if (ok) ok = mlx_astype(&nf, n, MLX_FLOAT32, s) == 0;
    mlx_array gf = mlx_array_new();
    if (ok) ok = mlx_astype(&gf, gate, MLX_FLOAT32, s) == 0;
    mlx_array act = mlx_array_new();
    if (ok) ok = mlx_sigmoid(&act, gf, s) == 0;                    /* sigmoid */
    mlx_array out = mlx_array_new();
    if (ok) ok = mlx_multiply(&out, nf, act, s) == 0;
    mlx_dtype dt = mlx_array_dtype(hidden);
    mlx_array outd = mlx_array_new();
    if (ok) ok = mlx_astype(&outd, out, dt, s) == 0;
    if (ok) ok = mlx_array_set(res, outd) == 0;
    mlx_array_free(n); mlx_array_free(nf); mlx_array_free(gf); mlx_array_free(act);
    mlx_array_free(out); mlx_array_free(outd);
    (void)use_silu;
    return ok;
}
/* DEBUG correttezza: hidden fp32 dopo OGNI layer per un forward di n token da zero.
 * out = [n_tokens][n_layers][hidden] (row-major: per posizione, poi per layer).
 * Ritorna il numero di float scritti, -1 su errore. */
int llm_ai_mlx_debug_all_layers_hidden(llm_ai_mlx_session_handle sess, const int32_t* ids, int n,
                                       float* out, int cap) {
    if (sess == NULL || ids == NULL || n <= 0 || out == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)sess;
    LlmMlxModel* m = s->model;
    const int L = m->cfg.n_layers, H = m->cfg.hidden_size;
    int64_t need = (int64_t)n * L * H;
    if ((int64_t)cap < need) return -(int)need;
    int* caplayers = (int*)malloc((size_t)L * sizeof(int));
    for (int i = 0; i < L; i++) caplayers[i] = i;
    OutputTensors t;
    bool ok = load_output_tensors(m, &t);
    mlx_array h = mlx_array_new();
    if (ok) ok = df2_capture_batch(m, s, t.embed_w, t.embed_s, t.embed_b, ids, n,
                                   s->cache_offset, s->k_cache, s->v_cache, s->has_cache,
                                   caplayers, L, out, (int)need, &h);
    mlx_array_free(h);
    free_output_tensors(&t);
    free(caplayers);
    return ok ? (int)need : -1;
}

/* DEBUG: output dell'embed (fp32) per n token — decide se il ~0 di gpt_oss nasce nell'embed. */
int llm_ai_mlx_debug_embed(llm_ai_mlx_session_handle sess, const int32_t* ids, int n, float* out, int cap) {
    if (sess == NULL || ids == NULL || n <= 0 || out == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)sess;
    LlmMlxModel* m = s->model;
    int need = n * m->cfg.hidden_size;
    if (cap < need) return -need;
    OutputTensors t;
    bool ok = load_output_tensors(m, &t);
    mlx_array x = mlx_array_new();
    if (ok) ok = embed_tokens_batch(m, t.embed_w, t.embed_s, t.embed_b, ids, n, &x);
    mlx_array hf = mlx_array_new();
    if (ok) ok = mlx_astype(&hf, x, MLX_FLOAT32, m->stream) == 0;
    if (ok) ok = mlx_array_eval(hf) == 0;
    if (ok) { const float* d = mlx_array_data_float32(hf); if (d) memcpy(out, d, (size_t)need * sizeof(float)); else ok = false; }
    mlx_array_free(x); mlx_array_free(hf);
    free_output_tensors(&t);
    return ok ? need : -1;
}

/* F1 follow-on GDN — pin/unpin prefix in-memory e checkpoint stato ricorrente (FaseC) */
int llm_ai_mlx_prefix_pin(llm_ai_mlx_session_handle h, int n_tokens) {
    if (h == NULL || n_tokens <= 0) return -1;
    LlmMlxSession* s = (LlmMlxSession*)h;
    // Full-attention: prefix tree già gestisce pin implicito dei primi n_tokens (non evictabili)
    // GDN: marca i primi n_tokens come pinned (non evictabili dal radix)
    s->pinned_tokens = n_tokens;
    return 0;
}
int llm_ai_mlx_prefix_unpin(llm_ai_mlx_session_handle h) {
    if (h == NULL) return -1;
    ((LlmMlxSession*)h)->pinned_tokens = 0;
    return 0;
}
int llm_ai_mlx_state_checkpoint(llm_ai_mlx_session_handle h, llm_ai_mlx_state_handle *out) {
    if (h == NULL || out == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)h;
    // Full-attention: no-op (KV già nel prefix tree), ritorna handle finto per uniformità
    // GDN: alloca copia dello stato ricorrente (conv1d/GDN per-layer)
    llm_ai_mlx_state_handle st = malloc(sizeof(struct { int dummy; }));
    if (!st) return -1;
    *(int*)st = s->cache_offset;
    *out = st;
    return 0;
}
int llm_ai_mlx_state_restore(llm_ai_mlx_session_handle h, llm_ai_mlx_state_handle st) {
    if (h == NULL || st == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)h;
    // Full-attention: no-op
    // GDN: ripristina conv-state (per ora solo cache_offset, il vero stato GDN è in s->gdn_state per-layer)
    s->cache_offset = *(int*)st;
    return 0;
}
void llm_ai_mlx_state_free(llm_ai_mlx_state_handle st) { free(st); }
