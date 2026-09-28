/* llmengine_moe_quant_alloc.c — §E3: misura di sensibilità per-tensore + allocatore bit greedy.
 * Vedi include/llmengine_moe_quant_alloc.h. */
#include "llmengine_moe_quant_alloc.h"
#include "llmcodec_moe_quant.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int llmengine_moe_quant_rel_error(const float* x, const float* imp, uint32_t rows, uint32_t cols,
                        uint32_t group, uint32_t bits, double* out_rel) {
    if (x == NULL || out_rel == NULL || rows == 0 || cols == 0) return -1;
    size_t wb = llmcodec_moe_affine_w_bytes(rows, cols, bits);
    size_t sb = llmcodec_moe_affine_sc_bytes(rows, cols, group);
    if (wb == 0 || sb == 0) return -1;
    uint32_t* w = (uint32_t*)malloc(wb);
    uint16_t* sc = (uint16_t*)malloc(sb);
    uint16_t* bs = (uint16_t*)malloc(sb);
    float* dq = (float*)malloc((size_t)rows * cols * sizeof(float));
    if (!w || !sc || !bs || !dq) { free(w); free(sc); free(bs); free(dq); return -1; }
    if (llmcodec_moe_affine_quantize(x, rows, cols, group, bits, w, sc, bs) != 0) {
        free(w); free(sc); free(bs); free(dq); return -1;
    }
    llmcodec_moe_affine_dequantize(w, sc, bs, rows, cols, group, bits, /*sc_f16=*/1, dq);
    double err = 0.0, sig = 0.0;
    for (uint32_t o = 0; o < rows; ++o) {
        const float* xr = x + (size_t)o * cols;
        const float* dr = dq + (size_t)o * cols;
        for (uint32_t k = 0; k < cols; ++k) {
            double wi = imp ? (double)imp[k] : 1.0;
            double d = (double)xr[k] - (double)dr[k];
            err += wi * d * d;
            sig += wi * (double)xr[k] * (double)xr[k];
        }
    }
    free(w); free(sc); free(bs); free(dq);
    *out_rel = (sig > 0.0) ? sqrt(err / sig) : 0.0;
    return 0;
}

int llmengine_moe_quant_alloc_bits(const uint64_t* nparams, const double* rel, int n, int ncand,
                         const int* cand, double target_avg_bits, int* out_cand) {
    if (!nparams || !rel || !cand || !out_cand || n <= 0 || ncand <= 0) return -1;
    for (int i = 1; i < ncand; ++i) if (cand[i] <= cand[i - 1]) return -1;  /* bit crescenti */
    uint64_t total = 0;
    for (int i = 0; i < n; ++i) { if (nparams[i] == 0) return -1; total += nparams[i]; }
    if (total == 0) return -1;

    for (int i = 0; i < n; ++i) out_cand[i] = 0;
    double bits_now = 0.0;
    for (int i = 0; i < n; ++i) bits_now += (double)nparams[i] * (double)cand[0];

    for (;;) {
        int best_i = -1;
        double best_score = 0.0, best_cost = 0.0;
        for (int i = 0; i < n; ++i) {
            int c = out_cand[i];
            if (c + 1 >= ncand) continue;
            double gain = rel[(size_t)i * ncand + c] - rel[(size_t)i * ncand + c + 1];
            if (gain <= 0.0) continue;
            double cost = (double)nparams[i] * (double)(cand[c + 1] - cand[c]);
            if ((bits_now + cost) / (double)total > target_avg_bits + 1e-9) continue;
            double score = gain / cost;
            if (score > best_score) { best_score = score; best_i = i; best_cost = cost; }
        }
        if (best_i < 0) break;
        out_cand[best_i]++;
        bits_now += best_cost;
    }
    return 0;
}
