/* test_moe_mla.c — confronto llmengine_moe_mla_decode vs un riferimento naive indipendente.
 * Dims piccole, RoPE interleaved (DeepSeek) senza yarn. Verifica pos=0 e pos=1 (uso cache). */
#include "llmengine_moe_mla.h"
#include "llmengine_moe_qwen3_forward.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)

#define HID 8
#define H   2
#define QN  4
#define QR  2
#define VHD 3
#define KVL 6
#define QH  (QN + QR)

static float rnd(unsigned* s) { *s = *s * 1103515245u + 12345u; return (float)((*s >> 8) & 0xFFFF) / 32768.0f - 1.0f; }

/* riferimento naive: stessa matematica, loop espliciti indipendenti. */
static void naive_layer(const MlaCfg* c, const MlaLayer* ly, const float* x, uint32_t pos, uint32_t kvctx,
                        const float* keys, const float* vals, float* key_out, float* val_out, float* out) {
    float xn[HID]; llmengine_moe_rmsnorm(x, ly->attn_norm, c->eps, HID, xn);
    float q[H * QH]; llmengine_moe_matvec(ly->wq, H * QH, HID, xn, q);
    float ckv[KVL + QR]; llmengine_moe_matvec(ly->wkv_a_mqa, KVL + QR, HID, xn, ckv);
    float ckn[KVL]; llmengine_moe_rmsnorm(ckv, ly->kv_a_norm, c->eps, KVL, ckn);
    float kvb[H * (QN + VHD)]; llmengine_moe_matvec(ly->wkv_b, H * (QN + VHD), KVL, ckn, kvb);
    /* rope interleaved sui q_pe per head e su kpe */
    float inv[QR / 2];
    for (uint32_t i = 0; i < QR / 2; ++i) inv[i] = powf(c->rope_theta, -2.0f * (float)i / (float)QR);
    for (uint32_t h = 0; h < H; ++h)
        for (uint32_t i = 0; i < QR / 2; ++i) {
            float* v = q + h * QH + QN;
            float ang = (float)pos * inv[i], co = cosf(ang), si = sinf(ang);
            float a = v[2 * i], b = v[2 * i + 1];
            v[2 * i] = a * co - b * si; v[2 * i + 1] = b * co + a * si;
        }
    for (uint32_t i = 0; i < QR / 2; ++i) {
        float* v = ckv + KVL;
        float ang = (float)pos * inv[i], co = cosf(ang), si = sinf(ang);
        float a = v[2 * i], b = v[2 * i + 1];
        v[2 * i] = a * co - b * si; v[2 * i + 1] = b * co + a * si;
    }
    float* kpe = ckv + KVL;
    float ctx[H * VHD];
    for (uint32_t h = 0; h < H; ++h) {
        for (uint32_t d = 0; d < QN; ++d) key_out[h * QH + d] = kvb[h * (QN + VHD) + d];
        for (uint32_t d = 0; d < QR; ++d) key_out[h * QH + QN + d] = kpe[d];
        for (uint32_t d = 0; d < VHD; ++d) val_out[h * VHD + d] = kvb[h * (QN + VHD) + QN + d];
        float sc[8];
        for (uint32_t t = 0; t <= pos; ++t) {
            const float* kt = (t < pos) ? (keys + (h * kvctx + t) * QH) : (key_out + h * QH);
            double a = 0; for (uint32_t d = 0; d < QH; ++d) a += (double)(q[h * QH + d]) * (double)kt[d];
            sc[t] = (float)a / sqrtf((float)QH);
        }
        llmengine_moe_softmax(sc, pos + 1);
        for (uint32_t d = 0; d < VHD; ++d) {
            double a = 0;
            for (uint32_t t = 0; t <= pos; ++t) {
                const float* vt = (t < pos) ? (vals + (h * kvctx + t) * VHD) : (val_out + h * VHD);
                a += (double)sc[t] * (double)vt[d];
            }
            ctx[h * VHD + d] = (float)a;
        }
    }
    llmengine_moe_matvec(ly->wo, HID, H * VHD, ctx, out);
}

int main(void) {
    MlaCfg c; memset(&c, 0, sizeof(c));
    c.hidden = HID; c.n_heads = H; c.q_lora = 0; c.kv_lora = KVL; c.qk_nope = QN; c.qk_rope = QR;
    c.v_head = VHD; c.eps = 1e-6f; c.rope_theta = 10000.0f; c.yarn_factor = 1.0f;
    c.yarn_mscale = 0.707f; c.yarn_mscale_all_dim = 0.707f; c.rope_variant = 1;   /* interleaved */

    unsigned s = 12345;
    float attn_norm[HID], wq[H * QH * HID], wkv_a[(KVL + QR) * HID], kv_a_norm[KVL], wkv_b[H * (QN + VHD) * KVL], wo[HID * H * VHD];
    for (uint32_t i = 0; i < HID; ++i) attn_norm[i] = 0.5f + 0.5f * fabsf(rnd(&s));
    for (uint32_t i = 0; i < (uint32_t)(H * QH * HID); ++i) wq[i] = rnd(&s) * 0.3f;
    for (uint32_t i = 0; i < (uint32_t)((KVL + QR) * HID); ++i) wkv_a[i] = rnd(&s) * 0.3f;
    for (uint32_t i = 0; i < KVL; ++i) kv_a_norm[i] = 0.5f + 0.5f * fabsf(rnd(&s));
    for (uint32_t i = 0; i < (uint32_t)(H * (QN + VHD) * KVL); ++i) wkv_b[i] = rnd(&s) * 0.3f;
    for (uint32_t i = 0; i < (uint32_t)(HID * H * VHD); ++i) wo[i] = rnd(&s) * 0.3f;

    MlaLayer ly; memset(&ly, 0, sizeof(ly));
    ly.attn_norm = attn_norm; ly.wq = wq; ly.wkv_a_mqa = wkv_a; ly.kv_a_norm = kv_a_norm; ly.wkv_b = wkv_b; ly.wo = wo;

    const uint32_t CTX = 4;
    MlaKv kv; memset(&kv, 0, sizeof(kv));
    kv.n_layers = 1; kv.n_heads = H; kv.ctx = CTX; kv.qh = QH; kv.vhd = VHD;
    kv.k = (float*)calloc((size_t)H * CTX * QH, sizeof(float));
    kv.v = (float*)calloc((size_t)H * CTX * VHD, sizeof(float));

    /* reference cache (indipendente) */
    float* rk = (float*)calloc((size_t)H * CTX * QH, sizeof(float));
    float* rv = (float*)calloc((size_t)H * CTX * VHD, sizeof(float));

    for (uint32_t pos = 0; pos < 2; ++pos) {
        float x[HID]; for (uint32_t i = 0; i < HID; ++i) x[i] = rnd(&s);
        float out[HID], rout[HID], krow[H * QH], vrow[H * VHD];
        CHECK(llmengine_moe_mla_decode(&c, &ly, 0, pos, x, out, &kv) == LLMENGINE_MOE_OK);
        naive_layer(&c, &ly, x, pos, CTX, rk, rv, krow, vrow, rout);
        for (uint32_t i = 0; i < HID; ++i) CHECK(fabsf(out[i] - rout[i]) < 1e-4f);
        /* confronto cache */
        for (uint32_t h = 0; h < H; ++h) {
            for (uint32_t d = 0; d < QH; ++d) CHECK(fabsf(kv.k[(h * CTX + pos) * QH + d] - krow[h * QH + d]) < 1e-5f);
            for (uint32_t d = 0; d < VHD; ++d) CHECK(fabsf(kv.v[(h * CTX + pos) * VHD + d] - vrow[h * VHD + d]) < 1e-5f);
        }
        /* copia nella cache reference (l'ordine è [h][t]) */
        for (uint32_t h = 0; h < H; ++h) {
            memcpy(rk + (h * CTX + pos) * QH, krow + h * QH, QH * sizeof(float));
            memcpy(rv + (h * CTX + pos) * VHD, vrow + h * VHD, VHD * sizeof(float));
        }
        float md=0; for (uint32_t i=0;i<HID;++i) md=fmaxf(md,fabsf(out[i]-rout[i]));
        printf("[mla] pos=%u maxdiff=%.3e\n", pos, md);
    }
    free(kv.k); free(kv.v); free(rk); free(rv);
    if (fails == 0) printf("test_moe_mla: OK\n");
    else fprintf(stderr, "test_moe_mla: %d FAIL\n", fails);
    return fails ? 1 : 0;
}
