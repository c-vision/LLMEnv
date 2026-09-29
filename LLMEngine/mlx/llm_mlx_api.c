/* ============================================================================
 * llm_mlx_api.c — dominio API esportata (estratto VERBATIM da llmengine_mlx.c).
 * Superficie pubblica DFlash2 (create/free/propose/generate single e multi-sessione),
 * debug layer/embed, prefix pin/unpin, state checkpoint/restore, gate RMSNorm gated.
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

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

    /* ---- ctx host fp32 -> bf16 [1,S,ctx_dim] ---- */
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
    if (!ok) { fprintf(stderr, "dflash_propose: ctx astype/reshape fallito\n"); return -1; }

    /* ---- block = [anchor] + mask*(L-1) ---- */
    int32_t* block_ids = (int32_t*)malloc((size_t)L * sizeof(int32_t));
    block_ids[0] = last_verified_token;
    for (int i = 1; i < L; i++) block_ids[i] = c->dflash_mask_token;

    /* ---- backbone ---- */
    mlx_array hidden = mlx_array_new();
    double _bt0 = g_df2_time_on() ? g_df2_now() : 0.0;
    ok = df2_backbone(rt, block_ids, L, x_ctx, S, &hidden, NULL);
    if (g_df2_time_on()) g_df2_t_bb += g_df2_now() - _bt0;
    free(block_ids);
    mlx_array_free(x_ctx);
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

    /* ---- selector greedy ---- */
    ok = df2_selector(rt, hidden_post, logits, K, last_verified_token, out_tokens);

    mlx_array_free(hidden); mlx_array_free(hidden_post); mlx_array_free(logits);
    if (!ok) return -1;

    /* ---- cache ctx avanza di S (dopo il forward completo dei layer) ---- */
    rt->ctx[rt->cur_ctx].ctx_pos += S;
    return K;
}

int llm_ai_mlx_dflash2_propose(llm_ai_mlx_dflash2_handle handle,
                               int32_t last_verified_token,
                               const float* ctx_feature, int S,
                               int32_t* out_tokens, int max_out) {
    if (handle == NULL || ctx_feature == NULL || S <= 0 || out_tokens == NULL) return -1;
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
    m->q4e_ctx_len = kvq_sess->history_len;
    mlx_array snaps[8];
    int snaps_n = 0;
    for (int l = 0; ok && l < m->cfg.n_layers; l++) {
        mlx_array layer_out = mlx_array_new();
        ok = layer_forward(m, kvq_sess, l, cache_offset, n_tokens, h, &kc[l], &vc[l], hasc[l], &layer_out);
        hasc[l] = true;
        mlx_array_free(h);
        h = layer_out;
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
    for (int l = 0; l < NL; l++) { ck[l] = mlx_array_new(); cv[l] = mlx_array_new();
        llm_mlx_cache_checkpoint(&ck[l], s->k_cache[l], m->stream);
        llm_mlx_cache_checkpoint(&cv[l], s->v_cache[l], m->stream); }
    if (g_df2_time_on()) { g_df2_t_ckpt += g_df2_now() - _t0; _t0 = g_df2_now(); }

    int64_t capneed = (int64_t)N * ncap * m->cfg.hidden_size;
    float* cap = (float*)malloc((size_t)capneed * sizeof(float));
    mlx_array h = mlx_array_new();
    bool ok;
    if (llm_ai_mlx_opt_env("LLM_DF2_SEQ_VERIFY")) {
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
    } else {
        ok = df2_capture_batch(m, s, t.embed_w, t.embed_s, t.embed_b, ids, N, n_prev,
                               s->k_cache, s->v_cache, s->has_cache,
                               caplayers, ncap, cap, (int)capneed, &h);
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
    int keep = n_prev + 1 + accepted;
    if (g_df2_time_on()) g_df2_n_rounds++;
    if (ok && accepted < K) {
        if (g_df2_time_on()) g_df2_n_roll++;
        /* rollback = ripristino checkpoint + ri-append del SOLO prefisso confermato
         * (anchor + accettati); il rollback per-slice non vale sugli array gdn 3D. */
        for (int l = 0; l < NL; l++) {
            llm_mlx_cache_rollback(&s->k_cache[l], ck[l], m->stream);
            llm_mlx_cache_rollback(&s->v_cache[l], cv[l], m->stream);
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
    for (int l = 0; l < NL; l++) { mlx_array_free(ck[l]); mlx_array_free(cv[l]); }
    free(ck); free(cv);
    return ok ? accepted : -1;
}

/* A1: conserva il ctx del draft (ultimo vctx) per la continuazione; libera il precedente. */
static void df2_retain_ctx(DFlash2Runtime* rt, float* ctx, int n) {
    if (rt->retained_ctx) free(rt->retained_ctx);
    rt->retained_ctx = ctx; rt->retained_n = n;
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
         * forward: capture robusta). Round1 usa queste righe come nel reference. */
        int64_t pctx_cap = (int64_t)n_prompt * ncap * H;
        pctx = (float*)malloc((size_t)pctx_cap * sizeof(float));
        if (!pctx) return -1;
        ctx_n0 = n_prompt;
        OutputTensors t;
        bool okc = load_output_tensors(m, &t);
        const int tnl = m->cfg.n_layers;   /* cache scratch del TARGET (64), NON del draft (5)! */
        mlx_array* pk = (mlx_array*)malloc((size_t)tnl * sizeof(mlx_array));
        mlx_array* pv = (mlx_array*)malloc((size_t)tnl * sizeof(mlx_array));
        bool* ph = (bool*)calloc((size_t)tnl, sizeof(bool));
        for (int i = 0; i < tnl; i++) { pk[i] = mlx_array_new(); pv[i] = mlx_array_new(); }
        mlx_array phidden = mlx_array_new();
        if (okc) okc = df2_capture_batch(m, s, t.embed_w, t.embed_s, t.embed_b,
                                         s->history, n_prompt, 0, pk, pv, ph,
                                         caplayers, ncap, pctx, (int)pctx_cap, &phidden);
        free_output_tensors(&t);
        for (int i = 0; i < tnl; i++) { mlx_array_free(pk[i]); mlx_array_free(pv[i]); }
        free(pk); free(pv); free(ph); mlx_array_free(phidden);
        if (!okc) { free(pctx); fprintf(stderr, "dflash_generate: capture prompt fallita\n"); return -1; }
        llm_ai_mlx_dflash2_reset(handle);   /* nuova sequenza: ring del draft pulito */
    }

    int emitted = 0;
    int stat_rounds = 0, stat_accepted = 0, stat_K = 0;
    int32_t anchor = s->pending_next_id;   /* bonus = greedy dopo il prompt (o pending in continuazione) */
    float* ctx_r = pctx;
    int ctx_n = ctx_n0;

    while (emitted < max_tokens) {
        if (anchor == eos_token) break;   /* eos NON emesso (come session_generate) */
        out_ids[emitted++] = anchor;
        if (emitted >= max_tokens) break;

        int32_t drafts[8];
        LLM_DF2_SP_BEGIN(prop);
        int K = llm_ai_mlx_dflash2_propose(handle, anchor, ctx_r, ctx_n, drafts, 8);
        LLM_DF2_SP_END(prop);
        if (K <= 0) break;
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
        for (int i = 0; i < accepted && emitted < max_tokens; i++) {
            if (drafts[i] == eos_token) { stop_eos = true; break; }   /* eos non emesso */
            out_ids[emitted++] = drafts[i];
        }
        free(ctx_r);
        ctx_r = vctx;
        ctx_n = accepted + 1;
        anchor = next_anchor;
        if (stop_eos) break;
        if (emitted >= max_tokens) break;
    }

    /* A1: persistenza. Il ctx `ctx_r` (ultimo vctx; pctx se nessun round) resta nel runtime per
     * la prossima chiamata di continuazione; il pending della sessione = l'ultimo anchor. */
    df2_retain_ctx(rt, ctx_r, ctx_n);
    rt->last_session = (void*)s;
    rt->last_offset = s->cache_offset;
    s->pending_next_id = anchor;
    s->has_pending = true;
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
        if (!alloc_ok || df2_dflash_verify_multi(rt, vsess, nv, anchorsA, drafts, Ks, 8,
                                                 caplayers, ncap, vctx, vca, nextA, accA) != 0) {
            for (int a = 0; a < nv; a++) free(vctx[a]);
            for (int i = 0; i < n; i++) if (ctx_r[i]) free(ctx_r[i]);
            free_output_tensors(&t);
            return -1;
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
