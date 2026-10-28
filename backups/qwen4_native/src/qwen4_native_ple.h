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

/* KV array u64/u32 per nome (owned dal reader): alloca out (da liberare). */
static long q4n_kv_u64(const LlmEngineMoeGguf* g, const char* key, uint64_t** out) {
    if (!g || !g->_kvown) return -1;
    for (int i = 0; i < g->nkv; i++) {
        if (strcmp(g->keys[i], key) != 0) continue;
        const LlmCodecGgufMetaKVOwned* kv = &g->_kvown[i];
        if (kv->type != GGARR || kv->arr_raw == NULL) return -1;
        long n = (long)kv->arr_count;
        uint64_t* a = (uint64_t*)malloc(sizeof(uint64_t) * (size_t)n);
        if (!a && n) return -1;
        if (kv->arr_type == GGU64 || kv->arr_type == GGI64) memcpy(a, kv->arr_raw, sizeof(uint64_t) * (size_t)n);
        else if (kv->arr_type == GGU32 || kv->arr_type == GGI32) { const uint32_t* r = (const uint32_t*)kv->arr_raw; for (long j = 0; j < n; j++) a[j] = r[j]; }
        else { free(a); return -1; }
        *out = a; return n;
    }
    return -1;
}

typedef struct { float offsets[64], vocab[64]; uint64_t mult[8]; int n_heads; } Q4nPleMeta;

static int q4n_ple_meta(const LlmEngineMoeGguf* g, Q4nHp* hp, Q4nPleMeta* m) {
    int64_t n = 0;
    { uint64_t* a = NULL; long na = q4n_kv_u64(g, "qwen4exp.ple.head_offsets", &a);
      if (na > 0 && na <= 64) { for (long i = 0; i < na; i++) m->offsets[i] = (float)a[i]; free(a); }
      else { free(a);
             float* off = llm_gguf_load_f32(g, "ple.head_offsets", &n);
             if (!off || n > 64) { free(off); return -1; }
             memcpy(m->offsets, off, (size_t)n * sizeof(float)); free(off); } }
    { uint64_t* a = NULL; long na = q4n_kv_u64(g, "qwen4exp.ple.head_vocab_sizes", &a);
      if (na > 0 && na <= 64) { for (long i = 0; i < na; i++) m->vocab[i] = (float)a[i]; free(a); }
      else { free(a);
             float* voc = llm_gguf_load_f32(g, "ple.head_vocab_sizes", &n);
             if (!voc || n > 64) { free(voc); return -1; }
             memcpy(m->vocab, voc, (size_t)n * sizeof(float)); free(voc); } }
    { uint64_t* a = NULL; long na = q4n_kv_u64(g, "qwen4exp.ple.layer_multipliers", &a);
      if (na > 0 && na <= 8) { for (long i = 0; i < na; i++) m->mult[i] = a[i]; free(a); m->n_heads = (hp->ple_ng - 1) * hp->ple_heads; return (m->n_heads > 0) ? 0 : -1; }
      free(a); }
    int64_t nm = 0;
    float* mul = llm_gguf_load_f32(g, "blk.1.ple_layer_multipliers", &nm);
    if (!mul) mul = llm_gguf_load_f32(g, "blk.0.ple_layer_multipliers", &nm);
    if (mul) {
        if (nm == 12) {   /* 4 u16 per valore (esatto, > 2^24) */
            for (int i = 0; i < 4 && i*4+3 < nm; i++) {
                uint64_t u = ((uint64_t)mul[i*4+0] << 48) | ((uint64_t)mul[i*4+1] << 32) |
                             ((uint64_t)mul[i*4+2] << 16) | (uint64_t)mul[i*4+3];
                m->mult[i] = u;
            }
        } else if (nm <= 8) {
            for (int64_t i = 0; i < nm; i++) m->mult[i] = (uint64_t)mul[i];
        }
    }
    free(mul);
    m->n_heads = (hp->ple_ng - 1) * hp->ple_heads;
    return (m->n_heads > 0) ? 0 : -1;
}

/* righe n-gram per il token corrente (prev newest-first), come qwen4_ple_step. */
static void q4n_ple_rows(int token, int* prev, const Q4nPleMeta* m, int ng, int eos, uint32_t* rows) {
    int ctx[4]; int cut = 0; ctx[0] = token;
    for (int s = 1; s < ng; s++) { int t = cut ? eos : prev[s - 1]; cut = cut || t == eos; ctx[s] = cut ? eos : t; }
    for (int n = 2; n <= ng; n++) {
        uint64_t mix = (uint64_t)ctx[0] * m->mult[0];
        for (int j = 1; j < n; j++) mix ^= (uint64_t)ctx[j] * m->mult[j];
        for (int gg = 0; gg < (m->n_heads / (ng - 1)); gg++) {
            int h = (n - 2) * (m->n_heads / (ng - 1)) + gg;
            uint64_t voc = (uint64_t)m->vocab[h];
            rows[h] = (uint32_t)(voc ? (mix % voc) : 0) + (uint32_t)m->offsets[h];
        }
    }
    for (int s = ng - 1; s > 0; s--) prev[s] = prev[s - 1];
    prev[0] = token;
}

/* riga globale -> (shard, riga locale). Supporta PLE packed affine (qs I8 + scales/biases F16)
 * e il vecchio layout .weight (F16/Q8). */
static int q4n_ple_row_read(const LlmEngineMoeGguf* g, uint32_t row, float* out) {
    /* Formato ds4: tabella unica BF16 [160 x rows]; row e' gia' globale. */
    int fi = llm_gguf_find_tensor(g, "per_layer_token_embd.weight");
    if (fi >= 0) {
        const int type = g->ti[fi].type;
        const uint64_t in = g->ti[fi].dims[0];
        const int rb = (int)llm_ggml_nbytes(type, (int64_t)in);
        const uint8_t* base = llm_gguf_tensor_data(g, fi, NULL);
        if (!base || rb <= 0 || row >= g->ti[fi].dims[1]) return -1;
        return llmcodec_ggml_dequant_row(type, base + (uint64_t)row * (uint64_t)rb, out, (int64_t)in);
    }
    char nm[64];
    int per_shard = 0, shard = -1, packed = 0;
    for (int s = 0; s < 128; s++) {
        snprintf(nm, sizeof nm, "ple.shard.%d.qs", s);
        int i = llm_gguf_find_tensor(g, nm);
        if (i >= 0) packed = 1;
        else { snprintf(nm, sizeof nm, "ple.shard.%d.weight", s); i = llm_gguf_find_tensor(g, nm); }
        if (i < 0) break;
        int rows = (int)g->ti[i].dims[1];
        if (!per_shard) per_shard = rows;
        if ((int)row < (s + 1) * rows) { shard = s; break; }
    }
    if (shard < 0 || per_shard <= 0) return -1;
    const uint32_t lr = row - (uint32_t)shard * (uint32_t)per_shard;
    if (packed) {
        char nq[64], ns[64], nb[64];
        snprintf(nq, sizeof nq, "ple.shard.%d.qs", shard);
        snprintf(ns, sizeof ns, "ple.shard.%d.scales", shard);
        snprintf(nb, sizeof nb, "ple.shard.%d.biases", shard);
        int iq = llm_gguf_find_tensor(g, nq), is = llm_gguf_find_tensor(g, ns), ib = llm_gguf_find_tensor(g, nb);
        if (iq < 0 || is < 0 || ib < 0) return -1;
        const uint64_t cols = g->ti[iq].dims[0] * 2u;
        const uint64_t ng = cols / 32u;
        const uint8_t* qb = llm_gguf_tensor_data(g, iq, NULL);
        const uint8_t* sb = llm_gguf_tensor_data(g, is, NULL);
        const uint8_t* bb = llm_gguf_tensor_data(g, ib, NULL);
        if (!qb || !sb || !bb) return -1;
        const size_t qrb = (size_t)ng * 16u;          /* cols/2 byte */
        const size_t srb = (size_t)ng * 2u;           /* F16 per gruppo */
        const uint8_t* qrow = qb + (uint64_t)lr * qrb;
        const uint8_t* srow = sb + (uint64_t)lr * srb;
        const uint8_t* brow = bb + (uint64_t)lr * srb;
        float* sc = malloc((size_t)ng * sizeof(float));
        float* bi = malloc((size_t)ng * sizeof(float));
        if (!sc || !bi) { free(sc); free(bi); return -1; }
        if (llmcodec_ggml_dequant_row(1, srow, sc, (int64_t)ng) != 0 ||
            llmcodec_ggml_dequant_row(1, brow, bi, (int64_t)ng) != 0) { free(sc); free(bi); return -1; }
        for (uint64_t i = 0; i < cols; i++) {
            uint8_t byte = qrow[i >> 1];
            int q = (i & 1u) ? (byte >> 4) : (byte & 0x0Fu);
            const uint64_t gg = i >> 5;
            out[i] = (float)q * sc[gg] + bi[gg];
        }
        free(sc); free(bi);
        return 0;
    }
    snprintf(nm, sizeof nm, "ple.shard.%d.weight", shard);
    int i = llm_gguf_find_tensor(g, nm);
    const int type = g->ti[i].type;
    const uint64_t in = g->ti[i].dims[0];
    const int rb = (int)llm_ggml_nbytes(type, (int64_t)in);
    const uint8_t* base = llm_gguf_tensor_data(g, i, NULL);
    if (!base || rb <= 0) return -1;
    base += (uint64_t)lr * (uint64_t)rb;
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
