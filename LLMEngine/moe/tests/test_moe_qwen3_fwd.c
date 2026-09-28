/* test_moe_qwen3_fwd.c — M15.3 gate: primitive del forward Qwen3 (RMSNorm/softmax/RoPE/matvec)
 * vs riferimenti naive + smoke del decode layer. */
#include "llmengine_moe_qwen3_forward.h"
#include "llmengine_moe_arch.h"   /* LLMENGINE_MOE_NORM_LAYER (M7.2) */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float frnd(unsigned* s) { *s = *s * 1103515245u + 12345u; return (float)((*s >> 8) & 0xFFFF) / 32768.0f - 1.0f; }

static LlmEngineMoeStatus fake_expert(void* ctx, uint32_t layer, uint32_t expert, const float* x, float* out) {
    (void)ctx; (void)layer; (void)expert;
    for (uint32_t i = 0; i < 32; ++i) out[i] = x[i] * 0.5f;
    return LLMENGINE_MOE_OK;
}

int main(void) {
    /* 1. RMSNorm vs riferimento */
    {
        const uint32_t n = 64; float x[64], w[64], out[64];
        unsigned s = 7; for (uint32_t i = 0; i < n; ++i) { x[i] = frnd(&s); w[i] = 0.5f + 0.5f * frnd(&s) * frnd(&s); }
        llmengine_moe_rmsnorm(x, w, 1e-6f, n, out);
        double ss = 0; for (uint32_t i = 0; i < n; ++i) ss += (double)x[i] * x[i];
        double r = 1.0 / sqrt(ss / n + 1e-6);
        double md = 0; for (uint32_t i = 0; i < n; ++i) md = fmax(md, fabs(out[i] - x[i] * r * w[i]));
        printf("[rmsnorm] maxdiff=%.3e\n", md); assert(md < 1e-5);
    }
    /* 1b. LayerNorm vs riferimento (M7.2) */
    {
        const uint32_t n = 64; float x[64], w[64], out[64];
        unsigned s = 13; for (uint32_t i = 0; i < n; ++i) { x[i] = frnd(&s) + 0.3f; w[i] = 0.5f + 0.5f * frnd(&s); }
        llmengine_moe_layernorm(x, w, 1e-5f, n, out);
        double mean = 0; for (uint32_t i = 0; i < n; ++i) mean += x[i]; mean /= n;
        double var = 0; for (uint32_t i = 0; i < n; ++i) { double d = (double)x[i] - mean; var += d * d; } var /= n;
        double inv = 1.0 / sqrt(var + 1e-5);
        double md = 0; for (uint32_t i = 0; i < n; ++i) md = fmax(md, fabs(out[i] - (((double)x[i] - mean) * inv) * w[i]));
        printf("[layernorm] maxdiff=%.3e\n", md); assert(md < 1e-4);
    }
    /* 2. softmax: somma 1 + monotona */
    {
        const uint32_t n = 32; float x[32], ref[32];
        unsigned s = 11; for (uint32_t i = 0; i < n; ++i) { x[i] = 3.0f * frnd(&s); ref[i] = x[i]; }
        llmengine_moe_softmax(x, n);
        double sum = 0; for (uint32_t i = 0; i < n; ++i) sum += x[i];
        assert(fabs(sum - 1.0) < 1e-5);
        double m = ref[0]; for (uint32_t i = 1; i < n; ++i) m = fmax(m, ref[i]);
        double z = 0; for (uint32_t i = 0; i < n; ++i) z += exp(ref[i] - m);
        double md = 0; for (uint32_t i = 0; i < n; ++i) md = fmax(md, fabs(x[i] - exp(ref[i] - m) / z));
        printf("[softmax] maxdiff=%.3e\n", md); assert(md < 1e-6);
    }
    /* 3. matvec vs riferimento */
    {
        const uint32_t R = 16, C = 24; float w[16 * 24], x[24], out[16];
        unsigned s = 3; for (uint32_t i = 0; i < R * C; ++i) w[i] = frnd(&s);
        for (uint32_t i = 0; i < C; ++i) x[i] = frnd(&s);
        llmengine_moe_matvec(w, R, C, x, out);
        double md = 0;
        for (uint32_t r = 0; r < R; ++r) { double a = 0; for (uint32_t c = 0; c < C; ++c) a += (double)w[r * C + c] * x[c]; md = fmax(md, fabs(out[r] - a)); }
        printf("[matvec] maxdiff=%.3e\n", md); assert(md < 1e-4);
    }
    /* 4. RoPE: preserva la norma (rotazione) + cambia i valori a pos>0 */
    {
        const uint32_t H = 4, D = 16; float v[4 * 16], v0[4 * 16];
        unsigned s = 5; for (uint32_t i = 0; i < H * D; ++i) { v[i] = frnd(&s); v0[i] = v[i]; }
        double n0 = 0; for (uint32_t i = 0; i < H * D; ++i) n0 += (double)v[i] * v[i];
        llmengine_moe_rope(v, H, D, 7, 10000000.0f);
        double n1 = 0; for (uint32_t i = 0; i < H * D; ++i) n1 += (double)v[i] * v[i];
        printf("[rope] |n1-n0|=%.3e\n", fabs(n1 - n0)); assert(fabs(n1 - n0) < 1e-3);
        int changed = 0; for (uint32_t i = 0; i < H * D; ++i) if (fabs(v[i] - v0[i]) > 1e-6f) changed = 1;
        assert(changed);
    }
    /* 5. decode layer smoke (esperto identità scalato): output finito, dims corrette */
    {
        Q3Cfg cfg; memset(&cfg, 0, sizeof(cfg));
        cfg.hidden = 32; cfg.n_heads = 4; cfg.n_kv_heads = 2; cfg.head_dim = 8;
        cfg.n_layers = 1; cfg.n_experts = 4; cfg.topk = 2; cfg.inter = 16; cfg.ctx = 4;
        cfg.eps = 1e-6f; cfg.rope_theta = 10000000.0f;
        const uint32_t H = cfg.hidden, D = cfg.head_dim;
        float input_ln[32], post[32], qw[32 * 32], kw[16 * 32], vw[16 * 32], ow[32 * 32], qn[8], kn[8], router[4 * 32];
        unsigned s = 9;
        for (uint32_t i = 0; i < H; ++i) { input_ln[i] = 1.0f; post[i] = 1.0f; }
        for (uint32_t i = 0; i < H * H; ++i) { qw[i] = frnd(&s) * 0.1f; ow[i] = frnd(&s) * 0.1f; }
        for (uint32_t i = 0; i < 2 * D * H; ++i) { kw[i] = frnd(&s) * 0.1f; vw[i] = frnd(&s) * 0.1f; }
        for (uint32_t i = 0; i < D; ++i) { qn[i] = 1.0f; kn[i] = 1.0f; }
        for (uint32_t i = 0; i < 4 * H; ++i) router[i] = frnd(&s);
        Q3Layer ly; memset(&ly, 0, sizeof(ly));
        ly.input_ln = input_ln; ly.post_attn_ln = post; ly.q_w = qw; ly.k_w = kw; ly.v_w = vw;
        ly.o_w = ow; ly.q_norm = qn; ly.k_norm = kn; ly.router = router;
        static float kbuf[1 * 2 * 4 * 8], vbuf[1 * 2 * 4 * 8]; Q3KvCache kv; kv.k = kbuf; kv.v = vbuf;
        float x[32], out[32]; for (uint32_t i = 0; i < H; ++i) x[i] = frnd(&s);
        int fake = 0;
        LlmEngineMoeStatus st = llmengine_moe_qwen3_decode_layer(&cfg, &ly, fake_expert, &fake, NULL, NULL, 0, 0, x, out, &kv);
        assert(st == LLMENGINE_MOE_OK);
        int finite = 1; for (uint32_t i = 0; i < H; ++i) if (!isfinite(out[i])) finite = 0;
        printf("[decode smoke] finite=%d\n", finite); assert(finite);
        /* 5b. dispatch norm_kind: LayerNorm cambia l'output rispetto a RMSNorm (M7.2) */
        float out_rms[32]; memcpy(out_rms, out, sizeof(out_rms));
        cfg.norm_kind = LLMENGINE_MOE_NORM_LAYER;
        memset(kbuf, 0, sizeof(kbuf)); memset(vbuf, 0, sizeof(vbuf));
        float out_ln[32];
        st = llmengine_moe_qwen3_decode_layer(&cfg, &ly, fake_expert, &fake, NULL, NULL, 0, 0, x, out_ln, &kv);
        assert(st == LLMENGINE_MOE_OK);
        int norm_changed = 0;
        for (uint32_t i = 0; i < H; ++i) if (fabs(out_ln[i] - out_rms[i]) > 1e-5) norm_changed = 1;
        printf("[decode norm dispatch] changed=%d\n", norm_changed); assert(norm_changed);
    }
    printf("test_moe_qwen3_fwd: PASS\n");
    return 0;
}
