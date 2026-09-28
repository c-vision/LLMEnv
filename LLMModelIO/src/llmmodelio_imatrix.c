/* llmmodelio_imatrix.c — lettura imatrix GGUF (llama-imatrix). Vedi llmmodelio_imatrix.h. */
#include "llmmodelio_imatrix.h"
#include "llmcodec_gguf_writer.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static uint8_t* read_tensor(const LlmCodecGgufFileInfo* fi, const char* path, uint64_t idx, uint64_t* bl) {
    const LlmCodecGgufTensorInfo* t = &fi->tensors[idx];
    uint64_t numel = 1;
    for (int d = 0; d < t->ndim; d++) numel *= t->dims[d];
    /* Canale d'errore = NULL: la dimensione NON calcolabile (tipo ignoto o `numel` non multiplo
     * del blocco) non deve mai diventare un `malloc(0)` che "legge" un tensore vuoto. */
    if (llmcodec_ggml_type_bytes(t->ggml_type, numel, bl) != 0) { *bl = 0; return NULL; }
    uint8_t* buf = (uint8_t*)malloc((size_t)*bl);
    if (!buf) return NULL;
    FILE* f = fopen(path, "rb");
    if (!f) { free(buf); return NULL; }
    fseek(f, (long)(fi->data_start + t->offset), SEEK_SET);
    if (fread(buf, 1, (size_t)*bl, f) != *bl) { /* parziale: accetta */ }
    fclose(f);
    return buf;
}
float* llmmodelio_imatrix_weights(const char* imatrix_path, const char* model_tensor_name, int64_t* out_len) {
    if (out_len) *out_len = 0;
    if (!imatrix_path || !model_tensor_name) return NULL;
    LlmCodecGgufFileInfo fi;
    if (llmcodec_gguf_read_info(imatrix_path, &fi) != 0) return NULL;
    char n_sum[512], n_cnt[512];
    snprintf(n_sum, sizeof n_sum, "%s.in_sum2", model_tensor_name);
    snprintf(n_cnt, sizeof n_cnt, "%s.counts",  model_tensor_name);
    int64_t is = -1, ic = -1;
    for (uint64_t i = 0; i < fi.n_tensors; i++) {
        if (strcmp(fi.tensors[i].name, n_sum) == 0) is = (int64_t)i;
        else if (strcmp(fi.tensors[i].name, n_cnt) == 0) ic = (int64_t)i;
    }
    if (is < 0) { llmcodec_gguf_free_info(&fi); return NULL; }
    uint64_t bs = 0, bc = 0;
    uint8_t* ps = read_tensor(&fi, imatrix_path, (uint64_t)is, &bs);
    uint8_t* pc = (ic >= 0) ? read_tensor(&fi, imatrix_path, (uint64_t)ic, &bc) : NULL;
    /* llama.cpp conserva TUTTE le matrici per-esperto: per un tensore MoE il risultato ha
     * ne[0]*ne[2] elementi, non la media fra esperti. Il quantizzatore seleziona poi la fetta
     * dell'esperto corrente. */
    uint64_t n = 1;
    for (int d = 0; d < fi.tensors[is].ndim; d++) n *= fi.tensors[is].dims[d];
    uint64_t nc = 0;                                     /* numel dei counts (da dims) */
    if (ic >= 0) { nc = 1; for (int d = 0; d < fi.tensors[ic].ndim; d++) nc *= fi.tensors[ic].dims[d]; }
    float* w = NULL;
    if (ps && n > 0 && bs >= n*4) {
        const float* fs = (const float*)ps;
        const float* fc = pc ? (const float*)pc : NULL;
        w = (float*)malloc((size_t)n*sizeof(float));
        if (w && (!fc || nc == 0 || n % nc == 0)) {
            const uint64_t per_count = (fc && nc > 0) ? n/nc : n;
            for (uint64_t i = 0; i < n; i++) {
                const float c = fc ? fc[i/per_count] : 1.0f;
                w[i] = c > 0.0f ? fs[i]/c : 1.0f;       /* identico a tools/quantize/quantize.cpp */
            }
            if (out_len) *out_len = (int64_t)n;
        } else {
            free(w); w = NULL;
        }
    }
    free(ps); free(pc); llmcodec_gguf_free_info(&fi);
    /* GUARD (2026-09-22, causa-radice del modello rotto): se un peso non e' finito, l'im_sum2 di
     * quel tensore contiene inf (le attivazioni del forward esplodono a certi layer — es. blk.15);
     * usarlo come weighting genera d/dmin NON FINITI -> logits corrotti. In tal caso NON pesare:
     * NULL -> il chiamante ricade sul percorso non-pesato (per-ruolo = 12.27 == baseline). */
    if (w != NULL) for (uint64_t i = 0; i < n; i++) if (!isfinite(w[i])) { free(w); w = NULL; break; }
    return w;
}

/* Uso REALE per esperto dal routing di calibrazione: <tensor>.counts = [1, E] (F32). */
float* llmmodelio_imatrix_expert_usage(const char* imatrix_path, const char* model_tensor_name, int64_t* out_n) {
    if (out_n) *out_n = 0;
    if (!imatrix_path || !model_tensor_name) return NULL;
    LlmCodecGgufFileInfo fi;
    if (llmcodec_gguf_read_info(imatrix_path, &fi) != 0) return NULL;
    char n_cnt[512]; snprintf(n_cnt, sizeof n_cnt, "%s.counts", model_tensor_name);
    int64_t ic = -1;
    for (uint64_t i = 0; i < fi.n_tensors; i++) if (strcmp(fi.tensors[i].name, n_cnt) == 0) { ic = (int64_t)i; break; }
    if (ic < 0) { llmcodec_gguf_free_info(&fi); return NULL; }
    uint64_t n = 1; for (int d = 0; d < fi.tensors[ic].ndim; d++) n *= fi.tensors[ic].dims[d];
    uint64_t b = 0;
    if (llmcodec_ggml_type_bytes(fi.tensors[ic].ggml_type, n, &b) != 0) b = 0;
    float* w = NULL;
    if (fi.tensors[ic].ggml_type == 0 && b >= n*4) {
        uint8_t* raw = read_tensor(&fi, imatrix_path, (uint64_t)ic, &b);
        if (raw) { w = (float*)malloc((size_t)n*sizeof(float)); for (uint64_t k = 0; k < n; k++) w[k] = ((float*)raw)[k]; free(raw); if (out_n) *out_n = (int64_t)n; }
    }
    llmcodec_gguf_free_info(&fi);
    return w;
}

float* llmmodelio_imatrix_samples(const char* imatrix_path, const char* model_tensor_name, int64_t* out_k_total, int64_t* out_cols) {
    if (out_k_total) *out_k_total = 0;
    if (out_cols) *out_cols = 0;
    if (!imatrix_path || !model_tensor_name) return NULL;
    LlmCodecGgufFileInfo fi;
    if (llmcodec_gguf_read_info(imatrix_path, &fi) != 0) return NULL;
    char n_samp[512]; snprintf(n_samp, sizeof n_samp, "%s.samples", model_tensor_name);
    int64_t is = -1;
    for (uint64_t i = 0; i < fi.n_tensors; i++) if (strcmp(fi.tensors[i].name, n_samp) == 0) { is = (int64_t)i; break; }
    if (is < 0) { llmcodec_gguf_free_info(&fi); return NULL; }
    uint64_t cols = fi.tensors[is].ndim > 0 ? fi.tensors[is].dims[0] : 0;
    uint64_t kt   = fi.tensors[is].ndim > 1 ? fi.tensors[is].dims[1] : 0;
    float* w = NULL;
    if (cols > 0 && kt > 0) {
        uint64_t bs = 0;
        uint8_t* ps = read_tensor(&fi, imatrix_path, (uint64_t)is, &bs);
        if (ps && bs >= cols*kt*4) {
            w = (float*)malloc((size_t)(cols*kt)*sizeof(float));
            memcpy(w, ps, (size_t)(cols*kt)*sizeof(float));
            if (out_k_total) *out_k_total = (int64_t)kt;
            if (out_cols) *out_cols = (int64_t)cols;
        }
        free(ps);
    }
    llmcodec_gguf_free_info(&fi);
    return w;
}
