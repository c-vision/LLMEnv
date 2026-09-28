/* ============================================================================
 * Dominio FFN denso SwiGLU + Ternary-Bonsai: llm_mlx_fuse_swiglu_metal_enabled, llm_mlx_fuse_swiglu_compile_enabled, llm_mlx_dense_ffn_quant (cache pesi), llm_bonsai_rotate, dense_swiglu_ffn.
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include "llm_mlx_lora.h"
#include "llm_mlx_swiglu_kernel.h"
#include <cJSON.h>
#include "llm_mlx_bonsai_hadamard.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
/* LLMMETAL_FUSE_SWIGLU_METAL: fusione SwiGLU dei FFN densi (llama/qwen3/qwen3_5/qwen2)
 * via kernel Metal reale (llm_mlx_swiglu_kernel.h, mlx_fast_metal_kernel) -- 1 dispatch
 * invece di 3 (sigmoid+mul+mul), thread-per-elemento, NESSUN mlx_compile/closure (la
 * via closure fu PROVATA CONTROPRODUCENTE, vedi commento llmengine_mlx.c:7626). Gemello
 * del gate LLMMETAL_FUSE_GEGLU_METAL, ma definito QUI perche' dense_swiglu_ffn precede nel file
 * l'area geglu. Default OFF: comportamento identico a oggi finche' non si promuove a misura
 * vinta + IDS greedy parita'. */
static int g_fuse_swiglu_metal_state = -1;
bool llm_mlx_fuse_swiglu_metal_enabled(void) {
    /* DEFAULT ON (2026-09-13): il kernel Metal fuso sostituisce le 3 op eager (Sigmoid+2x
     * Multiply). Misurato Llama-1B decode: 476,5 -> 446 nodi/token (~= mlx_lm 441 -> l'intero
     * gap di 8 op) e +2,4% tok/s (424,3 -> 434,6); Qwen3.6-27B +1%. IDS greedy IDENTICI a
     * 512/2048/8192. Opt-out LLMMETAL_FUSE_SWIGLU_METAL=0. (La nota storica "falsificato per
     * parita'" era su un kernel/GEGLU diverso: qui la parita' IDS e' verificata.) */
    if (g_fuse_swiglu_metal_state < 0) {
        const char* e = getenv("LLMMETAL_FUSE_SWIGLU_METAL");
        g_fuse_swiglu_metal_state = (e != NULL && strcmp(e, "0") == 0) ? 0 : 1;
    }
    return g_fuse_swiglu_metal_state == 1;
}

/* #86 (2026-09-11): fold SwiGLU dei FFN densi via `mx.compile` per-attivazione
 * (llm_mlx_swiglu_compiled, gia' usato dal ramo sorted del prefill MoE). Fonde le
 * PRIMITIVE MLX stesse (Sigmoid/Multiply), quindi eredita il `mx.sigmoid` bf16 del
 * path eager ed e' PARITA'-ESATTO bit-per-bit (validato model-free, §5-octies del
 * tracker). Distinto da L7 (kernel Metal hand-written, falsificato per parita') e dal
 * whole-step compile (falsificato per costo). Default OFF: A/B a finestra idle prima
 * di qualunque promozione. */
static int g_fuse_swiglu_compile_state = -1;
static bool llm_mlx_fuse_swiglu_compile_enabled(void) {
    if (g_fuse_swiglu_compile_state < 0) g_fuse_swiglu_compile_state = (getenv("LLM_FUSE_SWIGLU_COMPILE") != NULL) ? 1 : 0;
    return g_fuse_swiglu_compile_state == 1;
}

/* FASE 1b.1: FFN denso SwiGLU singolo (nessun router/nessun esperto) --
 * target di Qwen3 non-MoE (8B/32B) e, in seguito, Llama/Mistral dense
 * (Fase1b-MLX-Multi-Architecture-Plan.md §1.2 punto A, Fase 1b.2/1b.3: stessa
 * funzione, riusata). Naming dei tensori verificato contro
 * mlx_lm/models/qwen3.py (classe MLP, righe 92-100 circa nel venv locale):
 * "mlp.gate_proj"/"mlp.up_proj"/"mlp.down_proj" -- SENZA lo strato
 * "switch_mlp" della variante MoE. Stessa formula SwiGLU della FFN MoE
 * (silu(gate)*up), stesso schema quantizzato (quantized_linear_cfg). h:
 * hidden state POST-attenzione (da qwen3_moe_attention_block, riusata
 * identica -- vedi commento lì su QK-norm/RoPE condivisi tra Qwen3 MoE e
 * denso). *out riceve l'hidden state finale del layer. */
/* Cache pesi FFN denso per-layer (lazy). Ritorna NULL se disabilitata (LLM_FFN_WCACHE=0 o
 * LLM_WCACHE off) -> il chiamante ricade sui lookup per chiamata. */
GdenseFfnQuant* llm_mlx_dense_ffn_quant(LlmMlxModel* m, int layer_idx, int gemma4) {
    static int off = -1;
    if (off < 0) {
        const char* e = getenv("LLM_FFN_WCACHE");
        off = (e != NULL && atoi(e) == 0) ? 1 : 0;
    }
    if (off || m == NULL || !llm_mlx_wcache_enabled()) return NULL;
    if (m->dense_ffn == NULL) {
        int n = m->cfg.n_layers;
        if (n <= 0) return NULL;
        m->dense_ffn = (GdenseFfnQuant*)calloc((size_t)n, sizeof(GdenseFfnQuant));
        if (m->dense_ffn == NULL) return NULL;
        m->dense_ffn_n = n;
    }
    if (layer_idx < 0 || layer_idx >= m->dense_ffn_n) return NULL;
    GdenseFfnQuant* c = &m->dense_ffn[layer_idx];
    if (c->valid && c->gemma4 == gemma4) return c;
    if (c->valid) return NULL;   /* stessa entry richiesta con variante diversa: non mescolare */
    char nb[256];
    const char* norm_name = gemma4 ? "pre_feedforward_layernorm.weight" : "post_attention_layernorm.weight";
    if (gemma4) tname_gemma4(layer_idx, norm_name, nb, sizeof(nb));
    else tname(m->cfg.layer_prefix, layer_idx, norm_name, nb, sizeof(nb));
    if (!multishard_get(&m->weights, nb, &c->ln2)) return NULL;
    if (gemma4) tname_gemma4(layer_idx, "mlp.gate_proj", nb, sizeof(nb));
    else tname(m->cfg.layer_prefix, layer_idx, "mlp.gate_proj", nb, sizeof(nb));
    if (!multishard_get_triple(&m->weights, nb, &c->gate_w, &c->gate_s, &c->gate_b)) return NULL;
    if (gemma4) tname_gemma4(layer_idx, "mlp.up_proj", nb, sizeof(nb));
    else tname(m->cfg.layer_prefix, layer_idx, "mlp.up_proj", nb, sizeof(nb));
    if (!multishard_get_triple(&m->weights, nb, &c->up_w, &c->up_s, &c->up_b)) return NULL;
    if (gemma4) tname_gemma4(layer_idx, "mlp.down_proj", nb, sizeof(nb));
    else tname(m->cfg.layer_prefix, layer_idx, "mlp.down_proj", nb, sizeof(nb));
    if (!multishard_get_triple(&m->weights, nb, &c->down_w, &c->down_s, &c->down_b)) return NULL;
    c->gemma4 = gemma4;
    c->valid = true;
    return c;
}

/* ============================================================================
 * Ternary-Bonsai 2 27B (PrismML): i pesi elencati in hadamard.json sono FOLDED in basis
 * ruotata (Hadamard block-diagonale + sign espliciti). La de-rotazione si applica alle
 * ATTIVAZIONI in ingresso della proiezione (de-rotare i pesi richiederebbe l'unpack a f16:
 * ~54 GB, non praticabile). `suffix` e' il nome del tensore come nel pack (es.
 * "linear_attn.in_proj_qkv.weight"): per i nomi NON elencati (es. in_proj_a/b, che il pack
 * esclude) e' un no-op, come per qualunque altro modello (`m->bonsai_had == NULL`).
 * `invert=true` serve all'USCITA di embed_tokens (l'inverso). */
bool llm_bonsai_rotate(mlx_array* x, LlmMlxModel* m, int layer_idx, const char* suffix,
                              bool invert) {
    if (x == NULL || x->ctx == NULL || m == NULL || m->bonsai_had == NULL) return true;
    char name[256];
    if (layer_idx >= 0) tname(m->cfg.layer_prefix, layer_idx, suffix, name, sizeof(name));
    else {
        /* embed/lm_head non hanno numero di layer: il prefix di qwen3_5 finisce con ".layers."
         * e va tolto (altrimenti si ottiene "language_model.model.layers..embed_tokens.weight"). */
        char base[256];
        snprintf(base, sizeof(base), "%s", m->cfg.layer_prefix);
        size_t bl = strlen(base);
        if (bl >= 8 && strcmp(base + bl - 8, ".layers.") == 0) base[bl - 8] = 0;
        snprintf(name, sizeof(name), "%s.%s", base, suffix);
    }
    int kind = llm_bonsai_hadamard_kind(m->bonsai_had, name);
    if (kind == 0) {
        /* Fallback: il pack mette lm_head FUORI da ".model" (language_model.lm_head.weight). */
        char alt[256];
        const char* pfx = m->cfg.layer_prefix;
        const char* dm = strstr(pfx, ".model");
        if (dm != NULL && layer_idx < 0) {
            int head = (int)(dm - pfx);
            if (head > 0 && head < 200) {
                memcpy(alt, pfx, (size_t)head); alt[head] = 0;
                snprintf(alt + head, sizeof(alt) - (size_t)head, ".%s", suffix);
                kind = llm_bonsai_hadamard_kind(m->bonsai_had, alt);
                if (kind != 0) snprintf(name, sizeof(name), "%s", alt);
            }
        }
        if (kind == 0) {
            if (getenv("LLM_BONSAI_DBG"))
                fprintf(stderr, "[bonsai] NON trovato: '%s' (prefix='%s', layer_idx=%d, alt provato per lm_head)\n",
                        name, m->cfg.layer_prefix, layer_idx);
            return true;
        }
    }
    if ((kind == -1) != invert) return true;
    int nd = mlx_array_ndim(*x);
    if (nd < 1) return true;
    int width = (int)mlx_array_dim(*x, nd - 1);
    const float* signs = llm_bonsai_hadamard_signs_for(m->bonsai_had, width);
    if (signs != NULL && getenv("LLM_BONSAI_DBG") && strstr(name, "out_proj") != NULL)
        fprintf(stderr, "[gdn] out_proj signs(%d)[0:6]= %.0f %.0f %.0f %.0f %.0f %.0f\n", width,
                signs[0],signs[1],signs[2],signs[3],signs[4],signs[5]);
    if (signs == NULL) return true;
    mlx_array r = mlx_array_new();
    bool dir_inv = invert;
    if (kind == 1 && getenv("LLM_BONSAI_FWD_INV") != NULL && atoi(getenv("LLM_BONSAI_FWD_INV")) != 0) dir_inv = !dir_inv;   /* A/B direzione forward */
    if (getenv("LLM_BONSAI_DBG") && strstr(name, "embed_tokens") != NULL) {
        mlx_array vf = mlx_array_new();
        if (mlx_astype(&vf, *x, MLX_FLOAT32, m->stream) == 0 && mlx_array_eval(vf) == 0) {
            const float* d = mlx_array_data_float32(vf);
            if (d) { fprintf(stderr, "[bonsai] EMBED in[:8] ="); for (int q=0;q<8;q++) fprintf(stderr," %.6f", d[q]); fprintf(stderr, "\n"); }
        }
        mlx_array_free(vf);
    }
    bool ok = llm_bonsai_hadamard_rotate(&r, *x, width, m->bonsai_had->block, signs,
                                        llm_bonsai_signs_right(), dir_inv, m->stream);
    if (ok) {
        mlx_array_set(x, r);
        if (getenv("LLM_BONSAI_DBG") && strstr(name, "embed_tokens") != NULL) {
            mlx_array vf = mlx_array_new();
            if (mlx_astype(&vf, *x, MLX_FLOAT32, m->stream) == 0 && mlx_array_eval(vf) == 0) {
                const float* d = mlx_array_data_float32(vf);
                if (d) fprintf(stderr, "[bonsai] EMBED out[:6] = %.5f %.5f %.5f %.5f %.5f %.5f\n",
                               d[0], d[1], d[2], d[3], d[4], d[5]);
            }
            mlx_array_free(vf);
        }
        if (getenv("LLM_BONSAI_DBG"))
            fprintf(stderr, "[bonsai] ruotata %s (width=%d invert=%d)%s\n", name, width, (int)invert,
                    getenv("LLM_BONSAI_ONCE") ? "" : "");
    }
    else if (getenv("LLM_BONSAI_DBG"))
        fprintf(stderr, "[bonsai] rotate FALLITA %s width=%d invert=%d\n", name, width, (int)invert);
    mlx_array_free(r);
    return ok;
}

bool dense_swiglu_ffn(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    (void)n_tokens;
    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    char namebuf[256];

#define CHK(cond, what) do { \
        if (!(cond)) { fprintf(stderr, "dense_swiglu_ffn[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } \
    } while (0)

    if (layer_idx == 0 && getenv("LLM_BONSAI_DBG")) {
        mlx_array vf = mlx_array_new();
        if (mlx_astype(&vf, h, MLX_FLOAT32, m->stream) == 0 && mlx_array_eval(vf) == 0) {
            const float* d = mlx_array_data_float32(vf);
            if (d) { fprintf(stderr, "[mlp0] in[:6] ="); for (int q=0;q<6;q++) fprintf(stderr, " %.6f", d[q]); fprintf(stderr, "\n"); }
        }
        mlx_array_free(vf);
    }

    mlx_array ln2_w = mlx_array_new();
    mlx_array gate_w = mlx_array_new(), gate_s = mlx_array_new(), gate_b = mlx_array_new();
    mlx_array up_w = mlx_array_new(), up_s = mlx_array_new(), up_b = mlx_array_new();
    mlx_array down_w = mlx_array_new(), down_s = mlx_array_new(), down_b = mlx_array_new();
    GdenseFfnQuant* fc = llm_mlx_dense_ffn_quant(m, layer_idx, /*gemma4=*/0);
    if (fc != NULL) {
        ok = ok && mlx_array_set(&ln2_w, fc->ln2) == 0;
        ok = ok && mlx_array_set(&gate_w, fc->gate_w) == 0 && mlx_array_set_optional(&gate_s, fc->gate_s) &&
                   mlx_array_set_optional(&gate_b, fc->gate_b);
        ok = ok && mlx_array_set(&up_w, fc->up_w) == 0 && mlx_array_set_optional(&up_s, fc->up_s) &&
                   mlx_array_set_optional(&up_b, fc->up_b);
        ok = ok && mlx_array_set(&down_w, fc->down_w) == 0 && mlx_array_set_optional(&down_s, fc->down_s) &&
                   mlx_array_set_optional(&down_b, fc->down_b);
        CHK(ok, "dense ffn weights (cache)");
    } else {
        tname(m->cfg.layer_prefix, layer_idx, "post_attention_layernorm.weight", namebuf, sizeof(namebuf));
        ok = ok && multishard_get(&m->weights, namebuf, &ln2_w);
        tname(m->cfg.layer_prefix, layer_idx, "mlp.gate_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &gate_w, &gate_s, &gate_b);
        CHK(ok, "mlp.gate_proj lookup");
        tname(m->cfg.layer_prefix, layer_idx, "mlp.up_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &up_w, &up_s, &up_b);
        CHK(ok, "mlp.up_proj lookup");
        tname(m->cfg.layer_prefix, layer_idx, "mlp.down_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &down_w, &down_s, &down_b);
        CHK(ok, "mlp.down_proj lookup");
    }
    mlx_array normed2 = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed2, h, ln2_w, m->cfg.rms_norm_eps, m->stream); CHK(ok, "post_attention_layernorm"); }
    /* Ternary-Bonsai: mlp.gate_proj/mlp.up_proj sono entrambi elencati e hanno la STESSA
     * attivazione in ingresso: una sola rotazione (i sign dipendono dalla width). */
    if (ok) ok = llm_bonsai_rotate(&normed2, m, layer_idx, "mlp.gate_proj.weight", false);

    mlx_array gate_out = mlx_array_new(), up_out = mlx_array_new();
    /* FEDELTA' oMLX: Stage A una volta, condivisa fra gate e up. */
    bool a8_shared_gu = false;
    if (ok) {
        mlx_array s_qa, s_sa, s_ra; s_qa.ctx = NULL; s_sa.ctx = NULL; s_ra.ctx = NULL;
        if (llm_mlx_oq_a8_stage_try(normed2, gate_w, gate_s, gate_b, gsz, bts, m->cfg.quant_mode, m->stream, &s_qa, &s_sa, &s_ra)) {
            /* N-FUSIONE (env LLMMETAL_OQ_A8_FUSE_MLP) — **FALSIFICATA 2026-09-19**: in isolamento la
             * griglia 2x sembrava valere ~2x banda (curva N del probe), ma nel motore reale
             * perde **-5,5%** a chunk 8 (4 reps interleaved: 348-355 vs 371-375 tok/s) e -10%
             * con swiglu-metal OFF. Il "2x" del probe era l'overhead eval per-call contato due
             * volte. Default OFF; NON riattivare senza una ragione nuova (es. il layout
             * weight-major contiguo invece di [N,K]). */
            bool gu_fused = false;
            if (getenv("LLMMETAL_OQ_A8_FUSE_MLP") && gate_b.ctx != NULL && up_b.ctx != NULL) {
                mlx_array gu = mlx_array_new();
                const int wb = (mlx_array_dtype(normed2) == MLX_BFLOAT16) ? 1 : 0;
                if (llmmetal_oq_a8_qmm_fused2(&gu, s_qa, s_sa, s_ra, gate_w, gate_s, gate_b,
                                         up_w, up_s, up_b, m->stream, bts, wb) == 0) {
                    const int nd = mlx_array_ndim(gu);
                    const int* gsh = mlx_array_shape(gu);
                    if (nd >= 2 && nd <= 8 && gsh != NULL && (gsh[nd - 1] % 2) == 0) {
                        const int ntot = gsh[nd - 1];
                        int st[8], sp[8], sx[8];
                        for (int i = 0; i < nd; i++) { st[i] = 0; sp[i] = gsh[i]; sx[i] = 1; }
                        sp[nd - 1] = ntot / 2;
                        if (mlx_slice(&gate_out, gu, st, nd, sp, nd, sx, nd, m->stream) == 0) {
                            st[nd - 1] = ntot / 2; sp[nd - 1] = ntot;
                            if (mlx_slice(&up_out, gu, st, nd, sp, nd, sx, nd, m->stream) == 0) gu_fused = true;
                        }
                    }
                    mlx_array_free(gu);
                }
                if (!gu_fused) { mlx_array_free(gate_out); gate_out = mlx_array_new(); mlx_array_free(up_out); up_out = mlx_array_new(); }
            }
            if (gu_fused) { a8_shared_gu = true; }
            else {
                bool g_ok = llm_mlx_oq_a8_qmm_stage_try(&gate_out, s_qa, s_sa, s_ra, normed2, gate_w, gate_s, gate_b, gsz, bts, m->cfg.quant_mode, m->stream);
                bool u_ok = g_ok && llm_mlx_oq_a8_qmm_stage_try(&up_out, s_qa, s_sa, s_ra, normed2, up_w, up_s, up_b, gsz, bts, m->cfg.quant_mode, m->stream);
                if (g_ok && u_ok) { a8_shared_gu = true; }
                else { mlx_array_free(gate_out); gate_out = mlx_array_new(); mlx_array_free(up_out); up_out = mlx_array_new(); }
            }
            mlx_array_free(s_qa); mlx_array_free(s_sa); mlx_array_free(s_ra);
        }
    }
    if (ok && !a8_shared_gu) { ok = quantized_linear_cfg(&gate_out, normed2, gate_w, gate_s, gate_b, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "gate_proj matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "gate_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&gate_out, normed2, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }
    if (ok && !a8_shared_gu) { ok = quantized_linear_cfg(&up_out, normed2, up_w, up_s, up_b, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "up_proj matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "up_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&up_out, normed2, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }

    /* swiglu(gate, up) = silu(gate) * up = (gate * sigmoid(gate)) * up --
     * stessa formula/ordine di operazioni della FFN MoE sopra, e verificata
     * identica a mlx_lm.models.activations.swiglu. */
    mlx_array sig = mlx_array_new(), silu = mlx_array_new(), act = mlx_array_new();
    bool swiglu_done = false;
    if (ok && llm_mlx_fuse_swiglu_metal_enabled()) {
        swiglu_done = llm_mlx_swiglu_metal(gate_out, up_out, &act, m->stream);
        if (swiglu_done && llm_env_isset("LLMMETAL_FUSE_SWIGLU_METAL_VERIFY")) {
            /* Confronta il kernel col riferimento CPU float32 per layer; stampa max-diff/NaN
             * su stderr; USA SEMPRE act del kernel (nessun rischio). Il kernel fonde il
             * calcolo in float32 con UN arrotondamento finale (path eager: uno per op) -- lo
             * scarto atteso e' ~1-2 ULP, non bit-identicita'; la parita' vera e' IDS greedy. */
            mlx_array gf = mlx_array_new(), uf = mlx_array_new(), af = mlx_array_new();
            mlx_astype(&gf, gate_out, MLX_FLOAT32, m->stream);
            mlx_astype(&uf, up_out, MLX_FLOAT32, m->stream);
            mlx_astype(&af, act, MLX_FLOAT32, m->stream);
            if (mlx_array_eval(gf) == 0 && mlx_array_eval(uf) == 0 && mlx_array_eval(af) == 0) {
                const float* gd = mlx_array_data_float32(gf);
                const float* ud = mlx_array_data_float32(uf);
                const float* ad = mlx_array_data_float32(af);
                size_t n = mlx_array_size(gf);
                double maxdiff = 0.0; size_t nanact = 0;
                for (size_t i = 0; i < n && gd && ud && ad; i++) {
                    float x = gd[i];
                    float ref = x * (1.0f / (1.0f + expf(-x))) * ud[i];
                    double diff = fabs((double)ad[i] - (double)ref);
                    if (diff > maxdiff) maxdiff = diff;
                    if (isnan(ad[i]) || isinf(ad[i])) nanact++;
                }
                fprintf(stderr, "[LLMMETAL_FUSE_SWIGLU_METAL_VERIFY] layer=%d n=%zu maxdiff=%g nan/inf=%zu\n",
                        layer_idx, n, maxdiff, nanact);
            }
            mlx_array_free(gf); mlx_array_free(uf); mlx_array_free(af);
        }
    }
    /* NB (2026-09-13): attivare qui llm_mlx_swiglu_compiled di default E' UNA REGRESSIONE
     * (misurato Llama-1B decode @2048: 415,9 vs 422,0 tok/s = -1,5%). La compilata conviene solo
     * sul prefill batched (costo fisso di mlx_closure_apply ammortizzato su N righe), non sul
     * decode n_tokens==1. Resta opt-in via LLM_FUSE_SWIGLU_COMPILE (default OFF), come in origine. */
    if (ok && !swiglu_done && llm_mlx_fuse_swiglu_compile_enabled()) {
        /* #86: fold `mx.compile` per-attivazione al posto delle 3 op eager
         * (Sigmoid+Multiply+Multiply). Parita' bit-esatta validata model-free. */
        swiglu_done = llm_mlx_swiglu_compiled(gate_out, up_out, &act, m->stream);
    }
    if (ok && !swiglu_done) {
        /* eager (default, e fallback se il kernel/il fold fallisce): identico a oggi */
        if (ok) { ok = mlx_sigmoid(&sig, gate_out, m->stream) == 0; CHK(ok, "sigmoid"); }
        if (ok) { ok = mlx_multiply(&silu, gate_out, sig, m->stream) == 0; CHK(ok, "silu"); }
        if (ok) { ok = mlx_multiply(&act, silu, up_out, m->stream) == 0; CHK(ok, "swiglu"); }
    }

    /* Q2 (misuratore): imatrix anche per i layer DENSE (slot esperto 0). x3d = normed2 (input di
     * gate/up) e act = uscita SwiGLU (input di down), esattamente come i blocchi MoE.
     * Guardia anti-layout: la larghezza reale di act DEVE combaciare con im->down_cols (allocato
     * da cfg.intermediate_size): altrimenti `arow = ad + r*M` leggerebbe il layout sbagliato
     * (silent-wrong). In quel caso si salta e si avvisa UNA volta. */
    if (ok && m->im.active && act.ctx != NULL && n_tokens > 0) {
        int m_real = 0;
        int nd = mlx_array_ndim(act);
        if (nd >= 1) m_real = (int)mlx_array_dim(act, nd - 1);
        if (m_real == m->im.down_cols) {
            uint32_t* ex = (uint32_t*)malloc((size_t)n_tokens * sizeof(uint32_t));
            uint32_t* src = (uint32_t*)malloc((size_t)n_tokens * sizeof(uint32_t));
            if (ex != NULL && src != NULL) {
                for (int r = 0; r < n_tokens; r++) { ex[r] = 0u; src[r] = (uint32_t)r; }
                llm_mlx_imatrix_accumulate_rows(m, layer_idx, normed2, act, ex, src, n_tokens);
            }
            free(ex); free(src);
        } else if (m_real > 0) {
            static bool im_dense_warned = false;
            if (!im_dense_warned) {
                im_dense_warned = true;
                fprintf(stderr, "imatrix(dense): larghezza attivazione %d != down_cols %d "
                                "(intermediate_size=%d) — accumulo denso SALTATO\n",
                        m_real, m->im.down_cols, m->cfg.intermediate_size);
            }
        }
    }

    /* Ternary-Bonsai: mlp.down_proj e' elencato: si ruota l'uscita dello swiglu (width
     * 17408) DOPO lo swiglu, qualunque percorso l'abbia prodotta. */
    if (ok) ok = llm_bonsai_rotate(&act, m, layer_idx, "mlp.down_proj.weight", false);
    mlx_array down_out = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&down_out, act, down_w, down_s, down_b, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "down_proj matmul"); }
    { int li = llm_mlx_model_lora_get(m, layer_idx, "down_proj"); if (li >= 0 && ok) ok = llm_mlx_lora_apply_mlx(&down_out, act, m->lora_A[li], m->lora_B[li], m->lora_scale[li], m->stream); }

    if (ok) { ok = mlx_add(out, h, down_out, m->stream) == 0; CHK(ok, "final residual add"); }

    mlx_array all[] = {
        ln2_w, normed2, gate_w, gate_s, gate_b, up_w, up_s, up_b,
        down_w, down_s, down_b, gate_out, up_out, sig, silu, act, down_out,
    };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) mlx_array_free(all[i]);

#undef CHK
    return ok;
}
