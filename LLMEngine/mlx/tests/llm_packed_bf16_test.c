/* llm_packed_bf16_test.c — gemv Q4 packed su tensori REALI (scales/biases BF16, componenti in shard
 * diversi) contro dequantizzazione CPU di riferimento. Uso: llm_packed_bf16_test <model_dir> <base2d> <base3d> <expert>
 * es.: ... Qwen3.8-Flash-Next-4bit language_model.model.layers.2.linear_attn.out_proj \
 *          language_model.model.layers.8.mlp.switch_mlp.up_proj 5 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "llm_mlx_wpacked.h"
#include "llm_gemvq4_bridge.h"

static float bf16f(uint16_t u) { uint32_t v = (uint32_t)u << 16; float f; memcpy(&f, &v, 4); return f; }

/* y[o] = sum_i x[i] * (nib(o,i) * s + b), riferimento f64 */
static void ref_gemv(const uint32_t* w, const uint16_t* sc, const uint16_t* bi, int bf, int rows, int cols, int group,
                     const float* x, double* y) {
    const int wcols = cols / 8, ng = cols / group;
    for (int o = 0; o < rows; o++) {
        double acc = 0;
        for (int i = 0; i < cols; i++) {
            uint32_t word = w[(size_t)o * wcols + i / 8];
            unsigned nib = (word >> (4 * (i % 8))) & 0xF;
            int g = i / group;
            float s = bf ? bf16f(sc[(size_t)o * ng + g]) : 0.f, b = bf ? bf16f(bi[(size_t)o * ng + g]) : 0.f;
            acc += (double)x[i] * ((double)nib * s + b);
        }
        y[o] = acc;
    }
}

static int check(const char* tag, const uint32_t* w, const uint16_t* sc, const uint16_t* bi, int bf, int rows, int cols, int group) {
    float* x = malloc(sizeof(float) * cols); uint16_t* xh = malloc(2 * cols);
    for (int i = 0; i < cols; i++) {
        x[i] = 0.5f * sinf(0.37f * (float)i) + 0.1f;
        _Float16 h = (_Float16)x[i]; memcpy(&xh[i], &h, 2); x[i] = (float)h;   /* il kernel vede l'x in half */
    }
    double* yr = malloc(sizeof(double) * rows); uint16_t* yh = malloc(2 * rows);
    ref_gemv(w, sc, bi, bf, rows, cols, group, x, yr);
    LlmGemvQ4Ctx* g = llm_gemvq4_create();
    int rc = -1; double maxrel = 0, maxabs = 0, ymax = 0;
    if (g && llm_gemvq4_load_weights_wonly_nocopy(g, (const uint8_t*)w, sc, bi, (uint32_t)cols, (uint32_t)rows, (uint32_t)group) == 0) {
        llm_gemvq4_set_scale_bf16(g, bf);
        if (llm_gemvq4_run(g, xh, yh) == 0) {
            rc = 0;
            for (int o = 0; o < rows; o++) { _Float16 h; memcpy(&h, &yh[o], 2); if (fabs(yr[o]) > ymax) ymax = fabs(yr[o]); }
            for (int o = 0; o < rows; o++) {
                _Float16 h; memcpy(&h, &yh[o], 2); double d = fabs((double)h - yr[o]);
                if (d > maxabs) maxabs = d;
            }
            maxrel = maxabs / (ymax > 0 ? ymax : 1);
        }
    }
    if (g) llm_gemvq4_destroy(g);
    printf("%s: rows=%d cols=%d bf16=%d  max|y|=%.4g max|diff|=%.3e relativo=%.3e  %s\n", tag, rows, cols, bf, ymax, maxabs, maxrel,
           (rc == 0 && maxrel < 2e-3) ? "OK" : "FAIL");
    free(x); free(xh); free(yr); free(yh);
    return (rc == 0 && maxrel < 2e-3) ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "uso: %s <model_dir> <base2d> <base3d> <expert>\n", argv[0]); return 2; }
    char idx[1400], p[1600]; (void)idx;
    int fail = 0;
    /* shard del weight: via index.json con ricerca testuale */
    for (int t = 0; t < 2; t++) {
        const char* base = t == 0 ? argv[2] : argv[3];
        char ip[1500]; snprintf(ip, sizeof ip, "%s/model.safetensors.index.json", argv[1]);
        FILE* f = fopen(ip, "rb"); if (!f) { perror("index"); return 2; }
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        char* j = malloc((size_t)n + 1); if (fread(j, 1, (size_t)n, f) != (size_t)n) return 2; j[n] = 0; fclose(f);
        char key[700]; snprintf(key, sizeof key, "\"%s.weight\"", base);
        char* q = strstr(j, key); if (!q) { fprintf(stderr, "tensore assente: %s\n", base); return 2; }
        q = strchr(q + strlen(key), '"'); char* e = strchr(q + 1, '"'); *e = 0;
        snprintf(p, sizeof p, "%s/%s", argv[1], q + 1); free(j);
        if (t == 0) {
            LlmMlxPackedW w;
            if (llm_mlx_packed_open(p, base, 4, 32, &w) != 0) { fprintf(stderr, "open 2D fallita: %s\n", base); fail++; continue; }
            fail += check("2D", w.w, w.scales, w.biases, w.scale_bf16, (int)w.rows, (int)w.cols, w.group);
            llm_mlx_packed_close(&w);
        } else {
            LlmMlxPackedW3 w; int ex = atoi(argv[4]);
            if (llm_mlx_packed_open3d(p, base, 4, 32, &w) != 0) { fprintf(stderr, "open 3D fallita: %s\n", base); fail++; continue; }
            fail += check("3D", (const uint32_t*)((const uint8_t*)w.w + (size_t)ex * w.w_stride),
                          (const uint16_t*)((const uint8_t*)w.scales + (size_t)ex * w.s_stride),
                          w.biases ? (const uint16_t*)((const uint8_t*)w.biases + (size_t)ex * w.s_stride) : NULL,
                          w.scale_bf16, (int)w.rows, (int)w.cols, w.group);
            llm_mlx_packed_close3d(&w);
        }
    }
    printf(fail ? "FAIL\n" : "PASS\n");
    return fail ? 1 : 0;
}
