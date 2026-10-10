/* gmb_asr_encoder.c — forward dell'encoder audio Qwen3-ASR.
 * Port di antirez/qwen-asr `qwen_asr_encoder.c` (MIT) via i kernel di gmb_asr_kernels.c.
 *   Conv2D stem (3x conv 3x3 stride2 pad1 + GELU) -> reshape [480,16,w3]->[w3,7680] -> proj d_model
 *   + PE sinusoidale per-chunk -> transformer bidirezionale a finestre -> ln_post -> proj1(GELU)->proj2.
 * Pesi PRE-convertiti in F32 (come l'originale, che converte i bf16 all'load). */
#include "gmb_asr.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CONV_HIDDEN 480
#define CONV_PROJ_DIM (CONV_HIDDEN * 16) /* 7680 */

float* gmb_asr_encoder_forward(const GmbAsrEncConfig* cfg, const GmbAsrEncoder* enc,
                               const float* mel, int mel_frames, int* out_seq_len) {
    if (cfg == NULL || enc == NULL || mel == NULL || mel_frames <= 0) return NULL;
    const int d_model = cfg->d_model, n_heads = cfg->heads, head_dim = cfg->head_dim;
    const int ffn_dim = cfg->ffn_dim, output_dim = cfg->output_dim, chunk_size = cfg->chunk_size;

    /* conteggio token (w3 per chunk) */
    int n_chunks = (mel_frames + chunk_size - 1) / chunk_size;
    int total_tokens = 0;
    for (int c = 0; c < n_chunks; c++) {
        int start = c * chunk_size, end = start + chunk_size; if (end > mel_frames) end = mel_frames;
        int cw = end - start, w1 = (cw + 2 - 3) / 2 + 1, w2 = (w1 + 2 - 3) / 2 + 1, w3 = (w2 + 2 - 3) / 2 + 1;
        total_tokens += w3;
    }
    if (total_tokens <= 0) return NULL;

    float* x = (float*)calloc((size_t)total_tokens * d_model, sizeof(float));
    if (!x) return NULL;
    int token_offset = 0;

    for (int c = 0; c < n_chunks; c++) {
        int start = c * chunk_size, end = start + chunk_size; if (end > mel_frames) end = mel_frames;
        int cw = end - start;
        float* chunk_mel = (float*)malloc((size_t)128 * cw * sizeof(float));
        for (int m = 0; m < 128; m++) memcpy(chunk_mel + (size_t)m * cw, mel + (size_t)m * mel_frames + start, (size_t)cw * sizeof(float));

        int h1 = 64, w1 = (cw + 2 - 3) / 2 + 1;
        float* c1 = (float*)malloc((size_t)CONV_HIDDEN * h1 * w1 * sizeof(float));
        gmb_asr_conv2d(c1, chunk_mel, enc->conv1_w, enc->conv1_b, 1, CONV_HIDDEN, 128, cw, 3, 3, 2, 1);
        gmb_asr_gelu(c1, CONV_HIDDEN * h1 * w1); free(chunk_mel);

        int h2 = 32, w2 = (w1 + 2 - 3) / 2 + 1;
        float* c2 = (float*)malloc((size_t)CONV_HIDDEN * h2 * w2 * sizeof(float));
        gmb_asr_conv2d(c2, c1, enc->conv2_w, enc->conv2_b, CONV_HIDDEN, CONV_HIDDEN, h1, w1, 3, 3, 2, 1);
        gmb_asr_gelu(c2, CONV_HIDDEN * h2 * w2); free(c1);

        int h3 = 16, w3 = (w2 + 2 - 3) / 2 + 1;
        float* c3 = (float*)malloc((size_t)CONV_HIDDEN * h3 * w3 * sizeof(float));
        gmb_asr_conv2d(c3, c2, enc->conv3_w, enc->conv3_b, CONV_HIDDEN, CONV_HIDDEN, h2, w2, 3, 3, 2, 1);
        gmb_asr_gelu(c3, CONV_HIDDEN * h3 * w3); free(c2);

        float* reshaped = (float*)malloc((size_t)w3 * CONV_PROJ_DIM * sizeof(float));
        for (int t = 0; t < w3; t++)
            for (int ch = 0; ch < CONV_HIDDEN; ch++)
                for (int f = 0; f < h3; f++)
                    reshaped[(size_t)t * CONV_PROJ_DIM + ch * h3 + f] = c3[((size_t)ch * h3 + f) * w3 + t];
        free(c3);

        float* projected = x + (size_t)token_offset * d_model;
        gmb_asr_linear(projected, reshaped, enc->conv_out_w, NULL, w3, CONV_PROJ_DIM, d_model);
        free(reshaped);

        float* pe = (float*)malloc((size_t)w3 * d_model * sizeof(float));
        gmb_asr_sinusoidal_pe(pe, w3, d_model);
        gmb_asr_add_inplace(projected, pe, w3 * d_model);
        free(pe);
        token_offset += w3;
    }

    /* finestre di attenzione */
    int tokens_per_chunk = 0; { int cw = chunk_size, w1 = (cw + 2 - 3) / 2 + 1, w2 = (w1 + 2 - 3) / 2 + 1; tokens_per_chunk = (w2 + 2 - 3) / 2 + 1; }
    int window_token_size = tokens_per_chunk * (cfg->n_window_infer / chunk_size);
    if (window_token_size <= 0) window_token_size = total_tokens;
    int n_windows = (total_tokens + window_token_size - 1) / window_token_size;
    int* ws = (int*)malloc((size_t)(n_windows + 1) * sizeof(int));
    for (int w = 0; w < n_windows; w++) ws[w] = w * window_token_size;
    ws[n_windows] = total_tokens;

    float* x_norm = (float*)malloc((size_t)total_tokens * d_model * sizeof(float));
    float* q = (float*)malloc((size_t)total_tokens * d_model * sizeof(float));
    float* k = (float*)malloc((size_t)total_tokens * d_model * sizeof(float));
    float* v = (float*)malloc((size_t)total_tokens * d_model * sizeof(float));
    float* attn_out = (float*)malloc((size_t)total_tokens * d_model * sizeof(float));
    float* proj_out = (float*)malloc((size_t)total_tokens * d_model * sizeof(float));
    float* ffn_mid = (float*)malloc((size_t)total_tokens * ffn_dim * sizeof(float));
    float* ffn_out = (float*)malloc((size_t)total_tokens * d_model * sizeof(float));
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int layer = 0; layer < cfg->layers; layer++) {
        const GmbAsrEncLayer* l = &enc->layers[layer];
        gmb_asr_layer_norm(x_norm, x, l->attn_norm_w, l->attn_norm_b, total_tokens, d_model, 1e-5f);
        gmb_asr_linear(q, x_norm, l->wq, l->bq, total_tokens, d_model, d_model);
        gmb_asr_linear(k, x_norm, l->wk, l->bk, total_tokens, d_model, d_model);
        gmb_asr_linear(v, x_norm, l->wv, l->bv, total_tokens, d_model, d_model);
        gmb_asr_attn_bidirectional(attn_out, q, k, v, total_tokens, n_heads, head_dim, scale, ws, n_windows);
        gmb_asr_linear(proj_out, attn_out, l->wo, l->bo, total_tokens, d_model, d_model);
        gmb_asr_add_inplace(x, proj_out, total_tokens * d_model);

        gmb_asr_layer_norm(x_norm, x, l->ffn_norm_w, l->ffn_norm_b, total_tokens, d_model, 1e-5f);
        gmb_asr_linear(ffn_mid, x_norm, l->fc1, l->bfc1, total_tokens, d_model, ffn_dim);
        gmb_asr_gelu(ffn_mid, total_tokens * ffn_dim);
        gmb_asr_linear(ffn_out, ffn_mid, l->fc2, l->bfc2, total_tokens, ffn_dim, d_model);
        gmb_asr_add_inplace(x, ffn_out, total_tokens * d_model);
    }

    gmb_asr_layer_norm(x, x, enc->ln_post_w, enc->ln_post_b, total_tokens, d_model, 1e-5f);
    float* proj_mid = (float*)malloc((size_t)total_tokens * d_model * sizeof(float));
    gmb_asr_linear(proj_mid, x, enc->proj1_w, enc->bproj1, total_tokens, d_model, d_model);
    gmb_asr_gelu(proj_mid, total_tokens * d_model);
    float* enc_output = (float*)malloc((size_t)total_tokens * output_dim * sizeof(float));
    gmb_asr_linear(enc_output, proj_mid, enc->proj2_w, enc->bproj2, total_tokens, d_model, output_dim);
    free(proj_mid);

    free(x); free(x_norm); free(q); free(k); free(v); free(attn_out); free(proj_out); free(ffn_mid); free(ffn_out); free(ws);
    *out_seq_len = total_tokens;
    return enc_output;
}
