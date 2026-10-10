/* gmb_asr_decoder.c — decoder Qwen3 (GQA + per-head Q/K RMSNorm + NeoX RoPE + SwiGLU) con KV cache.
 * Port della struttura di antirez/qwen-asr `qwen_asr_decoder.c` (MIT) via i kernel di questo modulo. */
#include "gmb_asr.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

void gmb_asr_kv_init(GmbAsrKv* kv, int layers, int kv_dim, int max_seq) {
    kv->len = 0; kv->max = max_seq;
    kv->k = (float*)calloc((size_t)layers * max_seq * kv_dim, sizeof(float));
    kv->v = (float*)calloc((size_t)layers * max_seq * kv_dim, sizeof(float));
}
void gmb_asr_kv_free(GmbAsrKv* kv) { if (kv) { free(kv->k); free(kv->v); kv->k = kv->v = NULL; kv->len = kv->max = 0; } }

void gmb_asr_decoder_embed(const GmbAsrDecConfig* cfg, const GmbAsrDecoder* d, const int* ids, int seq, float* out) {
    for (int s = 0; s < seq; s++) {
        const uint16_t* row = d->tok_embeddings + (size_t)ids[s] * cfg->hidden;
        float* o = out + (size_t)s * cfg->hidden;
        for (int i = 0; i < cfg->hidden; i++) { uint32_t u = (uint32_t)row[i] << 16; float f; memcpy(&f, &u, 4); o[i] = f; }
    }
}

int gmb_asr_decoder_forward(const GmbAsrDecConfig* cfg, const GmbAsrDecoder* d, GmbAsrKv* kv,
                            const float* embeds, int seq, float* logits) {
    const int H = cfg->hidden, L = cfg->layers, NH = cfg->heads, NKV = cfg->kv_heads, HD = cfg->head_dim;
    const int inter = cfg->intermediate, q_dim = NH * HD, kv_dim = NKV * HD;
    const int start = kv->len;
    if (start + seq > kv->max) return -1;

    float* x = (float*)malloc((size_t)seq * H * sizeof(float));
    memcpy(x, embeds, (size_t)seq * H * sizeof(float));
    float* xn = (float*)malloc((size_t)seq * H * sizeof(float));
    float* q = (float*)malloc((size_t)seq * q_dim * sizeof(float));
    float* k = (float*)malloc((size_t)seq * kv_dim * sizeof(float));
    float* v = (float*)malloc((size_t)seq * kv_dim * sizeof(float));
    float* attn = (float*)malloc((size_t)seq * q_dim * sizeof(float));
    float* tmp = (float*)malloc((size_t)seq * (inter > H ? inter : H) * sizeof(float));
    float* gate = (float*)malloc((size_t)seq * inter * sizeof(float));
    float* up = (float*)malloc((size_t)seq * inter * sizeof(float));
    float* cos = (float*)malloc((size_t)seq * HD * sizeof(float));
    float* sin = (float*)malloc((size_t)seq * HD * sizeof(float));
    int* pos = (int*)malloc((size_t)seq * sizeof(int));
    for (int i = 0; i < seq; i++) pos[i] = start + i;
    gmb_asr_rope_neox_prep(cos, sin, pos, seq, HD, cfg->rope_theta);

    for (int layer = 0; layer < L; layer++) {
        const GmbAsrDecLayer* ly = &d->layers[layer];
        float* kcache = kv->k + (size_t)layer * kv->max * kv_dim;
        float* vcache = kv->v + (size_t)layer * kv->max * kv_dim;

        gmb_asr_rms_norm(xn, x, ly->input_norm, seq, H, cfg->rms_eps);
        gmb_asr_linear_bf16(q, xn, ly->wq, NULL, seq, H, q_dim);
        gmb_asr_linear_bf16(k, xn, ly->wk, NULL, seq, H, kv_dim);
        gmb_asr_linear_bf16(v, xn, ly->wv, NULL, seq, H, kv_dim);

        /* per-head Q/K RMSNorm (in-place) */
        gmb_asr_rms_norm_per_head(q, ly->q_norm, seq, NH, HD, cfg->rms_eps);
        gmb_asr_rms_norm_per_head(k, ly->k_norm, seq, NKV, HD, cfg->rms_eps);

        /* RoPE NeoX (in-place) */
        gmb_asr_rope_neox_apply(q, cos, sin, seq, NH, HD);
        gmb_asr_rope_neox_apply(k, cos, sin, seq, NKV, HD);

        /* append K/V alla cache */
        memcpy(kcache + (size_t)start * kv_dim, k, (size_t)seq * kv_dim * sizeof(float));
        memcpy(vcache + (size_t)start * kv_dim, v, (size_t)seq * kv_dim * sizeof(float));

        /* attenzione causale GQA su tutto il prefisso [0, start+seq) */
        gmb_asr_attn_causal_gqa(attn, q, kcache, vcache, seq, start + seq, NH, NKV, HD,
                                1.0f / sqrtf((float)HD), start);
        gmb_asr_linear_bf16(tmp, attn, ly->wo, NULL, seq, q_dim, H);
        gmb_asr_add_inplace(x, tmp, seq * H);

        gmb_asr_rms_norm(xn, x, ly->post_attn_norm, seq, H, cfg->rms_eps);
        gmb_asr_linear_bf16(gate, xn, ly->gate, NULL, seq, H, inter);
        gmb_asr_linear_bf16(up, xn, ly->up, NULL, seq, H, inter);
        gmb_asr_silu(gate, seq * inter);
        for (int i = 0; i < seq * inter; i++) gate[i] *= up[i];
        gmb_asr_linear_bf16(tmp, gate, ly->down, NULL, seq, inter, H);
        gmb_asr_add_inplace(x, tmp, seq * H);
    }
    kv->len = start + seq;

    if (logits) {
        const float* last = x + (size_t)(seq - 1) * H;
        float* nl = (float*)malloc((size_t)H * sizeof(float));
        gmb_asr_rms_norm(nl, last, d->norm, 1, H, cfg->rms_eps);
        gmb_asr_linear_bf16(logits, nl, d->tok_embeddings, NULL, 1, H, cfg->vocab);  /* lm_head tied */
        free(nl);
    }

    free(x); free(xn); free(q); free(k); free(v); free(attn); free(tmp); free(gate); free(up); free(cos); free(sin); free(pos);
    return 0;
}
