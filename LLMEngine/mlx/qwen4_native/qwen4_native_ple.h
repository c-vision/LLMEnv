/* qwen4_native_ple.h — PLE (n-gram) nativo: hash ds4 `qwen4_ple_step` + lettura righe
 * dagli shard `ple.shard.N.weight` (GGUF, F16/Q8) + layer PLE
 * (key/value proj, gate, conv dilatata) come `qwen4_ref_ple`. */
#ifndef LLM_QWEN4_NATIVE_PLE_H
#define LLM_QWEN4_NATIVE_PLE_H

#include "mlx/c/mlx.h"
#include "llm_ggml.h"
#include "qwen4_native_gguf.h"
#include "qwen4_native_ple_kernel.h"
#include "qwen4_native_layer.h"

typedef struct { int prev[4]; mlx_array hist; } Q4nPleState;

typedef struct { float offsets[64], vocab[64], mult[8]; int n_heads; } Q4nPleMeta;

static int q4n_ple_meta(const LlmEngineMoeGguf* g, Q4nHp* hp, Q4nPleMeta* m) {
    int64_t n = 0;
    float* off = llm_gguf_load_f32(g, "ple.head_offsets", &n);
    if (!off || n > 64) { free(off); return -1; }
    memcpy(m->offsets, off, (size_t)n * sizeof(float)); free(off);
    float* voc = llm_gguf_load_f32(g, "ple.head_vocab_sizes", &n);
    if (!voc || n > 64) { free(voc); return -1; }
    memcpy(m->vocab, voc, (size_t)n * sizeof(float)); free(voc);
    float* mul = llm_gguf_load_f32(g, "blk.1.ple_layer_multipliers", &n);
    if (!mul) mul = llm_gguf_load_f32(g, "blk.0.ple_layer_multipliers", &n);
    if (mul && n <= 8) memcpy(m->mult, mul, (size_t)n * sizeof(float));
    free(mul);
    m->n_heads = (hp->ple_ng - 1) * hp->ple_heads;
    return (m->n_heads > 0) ? 0 : -1;
}

/* righe n-gram per il token corrente (prev newest-first), come qwen4_ple_step. */
static void q4n_ple_rows(int token, int* prev, const Q4nPleMeta* m, int ng, int eos, uint32_t* rows) {
    int ctx[4]; int cut = 0; ctx[0] = token;
    for (int s = 1; s < ng; s++) { int t = cut ? eos : prev[s - 1]; cut = cut || t == eos; ctx[s] = cut ? eos : t; }
    for (int n = 2; n <= ng; n++) {
        uint64_t mix = (uint64_t)ctx[0] * (uint64_t)m->mult[0];
        for (int j = 1; j < n; j++) mix ^= (uint64_t)ctx[j] * (uint64_t)m->mult[j];
        for (int gg = 0; gg < (m->n_heads / (ng - 1)); gg++) {
            int h = (n - 2) * (m->n_heads / (ng - 1)) + gg;
            uint64_t voc = (uint64_t)m->vocab[h];
            rows[h] = (uint32_t)(voc ? (mix % voc) : 0) + (uint32_t)m->offsets[h];
        }
    }
    for (int s = ng - 1; s > 0; s--) prev[s] = prev[s - 1];
    prev[0] = token;
}

/* riga globale -> (shard, riga locale) assumendo shard di ugual numero di righe. */
static int q4n_ple_row_read(const LlmEngineMoeGguf* g, uint32_t row, float* out) {
    char nm[64];
    int per_shard = 0, shard = -1;
    for (int s = 0; s < 128; s++) {
        snprintf(nm, sizeof nm, "ple.shard.%d.weight", s);
        int i = llm_gguf_find_tensor(g, nm);
        if (i < 0) break;
        int rows = (int)g->ti[i].dims[1];
        if (!per_shard) per_shard = rows;
        if ((int)row < (s + 1) * rows) { shard = s; break; }
    }
    if (shard < 0 || per_shard <= 0) return -1;
    snprintf(nm, sizeof nm, "ple.shard.%d.weight", shard);
    int i = llm_gguf_find_tensor(g, nm);
    const int type = g->ti[i].type;
    const uint64_t in = g->ti[i].dims[0];
    const int rb = (int)llm_ggml_nbytes(type, (int64_t)in);
    const uint8_t* base = llm_gguf_tensor_data(g, i, NULL);
    if (!base || rb <= 0) return -1;
    base += (uint64_t)(row - (uint32_t)shard * (uint32_t)per_shard) * (uint64_t)rb;
    return llmcodec_ggml_dequant_row(type, base, out, (int64_t)in);
}

/* applica il layer PLE a R [1][hc*E]; token = token corrente. */
static int qwen4_native_ple_layer(mlx_array* R, const LlmEngineMoeGguf* g, int il,
                                  const Q4nLayerRefs* r, Q4nPleMeta* pm, Q4nPleState* st,
                                  int token, int E, int hc, int ng, int eos_id, int conv_k,
                                  float eps, Q4nMvFn mv, mlx_stream s) {
    const int nh = pm->n_heads, edim = E / nh;   /* 160 */
    if (nh <= 0 || E % nh) return -1;
    uint32_t rows[64];
    q4n_ple_rows(token, st->prev, pm, ng, eos_id, rows);
    float* emb = malloc((size_t)nh * edim * sizeof(float));
    if (!emb) return -1;
    for (int h = 0; h < nh; h++) if (q4n_ple_row_read(g, rows[h], emb + (size_t)h * edim) != 0) { free(emb); return -1; }
    float* key = malloc((size_t)hc * E * sizeof(float));
    float* value = malloc((size_t)E * sizeof(float));
    if (!key || !value) { free(emb); free(key); free(value); return -1; }
    if (mv(NULL, r->ple_key, emb, key, nh * edim, hc * E) != 0 ||
        mv(NULL, r->ple_value, emb, value, nh * edim, E) != 0) { free(emb); free(key); free(value); return -1; }
    int ksh[2] = {1, hc * E}, vsh[2] = {1, E}, esh[1] = {hc * E}, dsh[1] = {E};
    mlx_array aKey = mlx_array_new_data(key, ksh, 2, MLX_FLOAT32);
    mlx_array aVal = mlx_array_new_data(value, vsh, 2, MLX_FLOAT32);
    mlx_array aGK = mlx_array_new_data(r->ple_norm_key, esh, 1, MLX_FLOAT32);
    mlx_array aGQ = mlx_array_new_data(r->ple_norm_query, esh, 1, MLX_FLOAT32);
    mlx_array aGC = mlx_array_new_data(r->ple_norm_conv, esh, 1, MLX_FLOAT32);
    mlx_array gated = mlx_array_new(), normed = mlx_array_new();
    if (llm_q4n_ple_gate(&gated, &normed, *R, aKey, aVal, aGK, aGQ, aGC, eps, s) != 0) return -1;
    { /* conv: weight [C][K] */
        mlx_array aW = mlx_array_new_data(r->ple_conv, (int[]){hc * E, conv_k}, 2, MLX_FLOAT32);
        mlx_array Ro = mlx_array_new(), Ho = mlx_array_new();
        if (llm_q4n_ple_conv(&Ro, &Ho, *R, gated, normed, st->hist, aW, conv_k, ng, s) != 0) return -1;
        mlx_vector_array ev = mlx_vector_array_new();
        mlx_vector_array_append_value(ev, Ro);
        mlx_vector_array_append_value(ev, Ho);
        mlx_eval(ev); mlx_vector_array_free(ev);
        mlx_array_set(R, Ro);
        mlx_array_set(&st->hist, Ho);
        mlx_array_free(Ro); mlx_array_free(Ho); mlx_array_free(aW);
    }
    mlx_array_free(gated); mlx_array_free(normed);
    mlx_array_free(aKey); mlx_array_free(aVal); mlx_array_free(aGK); mlx_array_free(aGQ); mlx_array_free(aGC);
    free(emb); free(key); free(value);
    return 0;
}

#endif
