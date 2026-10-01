/* ============================================================================
 * Dominio FFN denso GeGLU (gemma4): dense_geglu_ffn (proiezioni mlp.gate/up/down, attivazione GeGLU con kernel/compile/eager).
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include "llm_mlx_geglu_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
bool dense_geglu_ffn(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out) {
    (void)n_tokens;
    bool ok = true;
    const int gsz = m->cfg.quant_group_size;
    const int bts = m->cfg.quant_bits;
    char namebuf[256];

#define CHK(cond, what) do { \
        if (!(cond)) { fprintf(stderr, "dense_geglu_ffn[layer=%d]: %s failed\n", layer_idx, (what)); ok = false; } \
    } while (0)

    mlx_array ln_pre_w = mlx_array_new();
    mlx_array gate_w = mlx_array_new(), gate_s = mlx_array_new(), gate_b = mlx_array_new();
    mlx_array up_w = mlx_array_new(), up_s = mlx_array_new(), up_b = mlx_array_new();
    mlx_array down_w = mlx_array_new(), down_s = mlx_array_new(), down_b = mlx_array_new();
    GdenseFfnQuant* fc = llm_mlx_dense_ffn_quant(m, layer_idx, /*gemma4=*/1);
    if (fc != NULL) {
        ok = ok && mlx_array_set(&ln_pre_w, fc->ln2) == 0;
        ok = ok && mlx_array_set(&gate_w, fc->gate_w) == 0 && mlx_array_set_optional(&gate_s, fc->gate_s) &&
                   mlx_array_set_optional(&gate_b, fc->gate_b);
        ok = ok && mlx_array_set(&up_w, fc->up_w) == 0 && mlx_array_set_optional(&up_s, fc->up_s) &&
                   mlx_array_set_optional(&up_b, fc->up_b);
        ok = ok && mlx_array_set(&down_w, fc->down_w) == 0 && mlx_array_set_optional(&down_s, fc->down_s) &&
                   mlx_array_set_optional(&down_b, fc->down_b);
        CHK(ok, "dense geglu weights (cache)");
    } else {
        tname_gemma4(layer_idx, "pre_feedforward_layernorm.weight", namebuf, sizeof(namebuf));
        ok = ok && multishard_get(&m->weights, namebuf, &ln_pre_w);
        CHK(ok, "pre_feedforward_layernorm lookup");
        tname_gemma4(layer_idx, "mlp.gate_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &gate_w, &gate_s, &gate_b);
        CHK(ok, "mlp.gate_proj lookup");
        tname_gemma4(layer_idx, "mlp.up_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &up_w, &up_s, &up_b);
        CHK(ok, "mlp.up_proj lookup");
        tname_gemma4(layer_idx, "mlp.down_proj", namebuf, sizeof(namebuf));
        ok = ok && multishard_get_triple(&m->weights, namebuf, &down_w, &down_s, &down_b);
        CHK(ok, "mlp.down_proj lookup");
    }
    mlx_array normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&normed, h, ln_pre_w, m->cfg.rms_norm_eps, m->stream); CHK(ok, "pre_feedforward_layernorm"); }

    mlx_array gate_out = mlx_array_new(), up_out = mlx_array_new();
    /* FEDELTA' oMLX: Stage A una volta, condivisa fra gate e up. */
    bool a8_shared_gu = false;
    if (ok) {
        mlx_array s_qa, s_sa, s_ra; s_qa.ctx = NULL; s_sa.ctx = NULL; s_ra.ctx = NULL;
        if (llm_mlx_oq_a8_stage_try(normed, gate_w, gate_s, gate_b, gsz, bts, m->cfg.quant_mode, m->stream, &s_qa, &s_sa, &s_ra)) {
            bool g_ok = llm_mlx_oq_a8_qmm_stage_try(&gate_out, s_qa, s_sa, s_ra, normed, gate_w, gate_s, gate_b, gsz, bts, m->cfg.quant_mode, m->stream);
            bool u_ok = g_ok && llm_mlx_oq_a8_qmm_stage_try(&up_out, s_qa, s_sa, s_ra, normed, up_w, up_s, up_b, gsz, bts, m->cfg.quant_mode, m->stream);
            if (g_ok && u_ok) { a8_shared_gu = true; }
            else { mlx_array_free(gate_out); gate_out = mlx_array_new(); mlx_array_free(up_out); up_out = mlx_array_new(); }
            mlx_array_free(s_qa); mlx_array_free(s_sa); mlx_array_free(s_ra);
        }
    }
    if (ok && !a8_shared_gu) { ok = quantized_linear_cfg(&gate_out, normed, gate_w, gate_s, gate_b, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "gate_proj matmul"); }
    if (ok && !a8_shared_gu) { ok = quantized_linear_cfg(&up_out, normed, up_w, up_s, up_b, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "up_proj matmul"); }

    /* gelu_approx(gate) = 0.5*gate*(1 + tanh(sqrt(2/pi)*(gate + 0.044715*gate^3)))
     * LLMMETAL_FUSE_GEGLU_METAL=1 (default OFF): 1 dispatch via kernel Metal reale, nessun
     * compile/closure (llm_mlx_geglu_metal, vedi llm_mlx_geglu_kernel.h). LLM_FUSE_GEGLU=1
     * (default OFF, PROVATO CONTROPRODUCENTE, vedi commento sopra): via mlx_compile/closure. */
    mlx_array gate3 = mlx_array_new(), gate3_scaled = mlx_array_new(), inner = mlx_array_new();
    mlx_array inner_scaled = mlx_array_new(), tanh_out = mlx_array_new(), one_plus_tanh = mlx_array_new();
    mlx_array gelu_out = mlx_array_new(), act = mlx_array_new();
    if (ok && llm_mlx_fuse_geglu_metal_enabled()) {
        ok = llm_mlx_geglu_metal(gate_out, up_out, &act, m->stream);
        CHK(ok, "geglu metal");
        /* LLMMETAL_FUSE_GEGLU_METAL_VERIFY=1 (default OFF, stesso pattern di LLMMETAL_KVMETAL_VERIFY/
         * LLM_GEMVQ4_VERIFY): confronta il kernel contro il riferimento CPU per layer,
         * stampa max-diff/NaN su stderr, usa SEMPRE act (nessun rischio anche se attivo. */
        if (ok && llm_env_isset("LLMMETAL_FUSE_GEGLU_METAL_VERIFY")) {
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
                    float ref = 0.5f*x*(1.0f+tanhf(0.7978845608028654f*(x+0.044715f*x*x*x)))*ud[i];
                    double diff = fabs((double)ad[i] - (double)ref);
                    if (diff > maxdiff) maxdiff = diff;
                    if (isnan(ad[i]) || isinf(ad[i])) nanact++;
                }
                fprintf(stderr, "[LLMMETAL_FUSE_GEGLU_METAL_VERIFY] layer=%d n=%zu maxdiff=%g nan/inf=%zu\n",
                        layer_idx, n, maxdiff, nanact);
            }
            mlx_array_free(gf); mlx_array_free(uf); mlx_array_free(af);
        }
    } else if (ok && llm_mlx_fuse_geglu_enabled()) {
        ok = llm_mlx_geglu_compiled(gate_out, up_out, &act, m->stream);
        CHK(ok, "geglu fuso");
    } else if (ok) {
        mlx_array three = mlx_array_new_float32(3.0f);
        mlx_array c0044715 = mlx_array_new_float32(0.044715f);
        mlx_array sqrt2_pi = mlx_array_new_float32(0.7978845608028654f); /* sqrt(2/pi) */
        mlx_array one = mlx_array_new_float32(1.0f);
        mlx_array half = mlx_array_new_float32(0.5f);
        ok = mlx_power(&gate3, gate_out, three, m->stream) == 0; CHK(ok, "gelu x^3");
        if (ok) { ok = mlx_multiply(&gate3_scaled, gate3, c0044715, m->stream) == 0; CHK(ok, "gelu 0.044715*x^3"); }
        if (ok) { ok = mlx_add(&inner, gate_out, gate3_scaled, m->stream) == 0; CHK(ok, "gelu x+0.044715*x^3"); }
        if (ok) { ok = mlx_multiply(&inner_scaled, inner, sqrt2_pi, m->stream) == 0; CHK(ok, "gelu *sqrt(2/pi)"); }
        if (ok) { ok = mlx_tanh(&tanh_out, inner_scaled, m->stream) == 0; CHK(ok, "gelu tanh"); }
        if (ok) { ok = mlx_add(&one_plus_tanh, tanh_out, one, m->stream) == 0; CHK(ok, "gelu 1+tanh"); }
        if (ok) { ok = mlx_multiply(&gelu_out, gate_out, one_plus_tanh, m->stream) == 0; CHK(ok, "gelu gate*(1+tanh)"); }
        if (ok) { ok = mlx_multiply(&gelu_out, gelu_out, half, m->stream) == 0; CHK(ok, "gelu *0.5"); }
        if (ok) { ok = mlx_multiply(&act, gelu_out, up_out, m->stream) == 0; CHK(ok, "geglu gelu(gate)*up"); }
        mlx_array_free(three); mlx_array_free(c0044715); mlx_array_free(sqrt2_pi);
        mlx_array_free(one); mlx_array_free(half);
    }

    /* Q2-b: come `dense_swiglu_ffn` — energia del FFN denso (slot 0): x3d = normed (input
     * gate/up), act = uscita GeGLU (input down). Stessa guardia anti-layout. */
    if (ok && m->im.active && act.ctx != NULL && n_tokens > 0) {
        int nd = mlx_array_ndim(act);
        int m_real = nd >= 1 ? (int)mlx_array_dim(act, nd - 1) : 0;
        if (m_real == m->im.down_cols) {
            uint32_t* ex = (uint32_t*)malloc((size_t)n_tokens * sizeof(uint32_t));
            uint32_t* src = (uint32_t*)malloc((size_t)n_tokens * sizeof(uint32_t));
            if (ex != NULL && src != NULL) {
                for (int r = 0; r < n_tokens; r++) { ex[r] = 0u; src[r] = (uint32_t)r; }
                llm_mlx_imatrix_accumulate_rows(m, layer_idx, normed, act, ex, src, n_tokens);
            }
            free(ex); free(src);
        } else if (m_real > 0) {
            static bool im_geglu_warned = false;
            if (!im_geglu_warned) {
                im_geglu_warned = true;
                fprintf(stderr, "imatrix(dense/geglu): larghezza attivazione %d != down_cols %d "
                                "(intermediate_size=%d) — accumulo SALTATO\n",
                        m_real, m->im.down_cols, m->cfg.intermediate_size);
            }
        }
    }

    mlx_array down_out = mlx_array_new();
    if (ok) { ok = quantized_linear_cfg(&down_out, act, down_w, down_s, down_b, gsz, bts, m->cfg.quant_mode, m->stream); CHK(ok, "down_proj matmul"); }

    /* post_feedforward_layernorm: applicata al risultato della FFN PRIMA del residuo --
     * a differenza di dense_swiglu_ffn, che non ha questa seconda norma. */
    mlx_array ln_post_w = mlx_array_new();
    tname_gemma4(layer_idx, "post_feedforward_layernorm.weight", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &ln_post_w);
    CHK(ok, "post_feedforward_layernorm lookup");
    mlx_array down_normed = mlx_array_new();
    if (ok) { ok = rms_norm_cfg(&down_normed, down_out, ln_post_w, m->cfg.rms_norm_eps, m->stream); CHK(ok, "post_feedforward_layernorm"); }

    mlx_array residual_out = mlx_array_new();
    if (ok) { ok = mlx_add(&residual_out, h, down_normed, m->stream) == 0; CHK(ok, "final residual add"); }

    /* layer_scalar: scalare appreso per layer (shape [1]), moltiplicato incondizionatamente
     * sull'intero output del layer -- verificato in gemma4_text.py, riga finale di
     * DecoderLayer.__call__, applicato SEMPRE (non condizionato da MoE/per-layer-gating,
     * entrambi spenti per questo modello comunque). */
    mlx_array layer_scalar = mlx_array_new();
    tname_gemma4(layer_idx, "layer_scalar", namebuf, sizeof(namebuf));
    ok = ok && multishard_get(&m->weights, namebuf, &layer_scalar);
    CHK(ok, "layer_scalar lookup");
    if (ok) { ok = mlx_multiply(out, residual_out, layer_scalar, m->stream) == 0; CHK(ok, "layer_scalar multiply"); }

    mlx_array all[] = {
        ln_pre_w, normed, gate_w, gate_s, gate_b, up_w, up_s, up_b,
        down_w, down_s, down_b, gate_out, up_out,
        gate3, gate3_scaled, inner, inner_scaled, tanh_out, one_plus_tanh, gelu_out, act,
        down_out, ln_post_w, down_normed, residual_out, layer_scalar,
    };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) mlx_array_free(all[i]);

#undef CHK
    return ok;
}
