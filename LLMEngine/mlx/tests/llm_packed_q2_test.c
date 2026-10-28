/* llm_packed_q2_test.c — gemv packed lazy (kernel Q2/Q4 con bits per tensore) su un tensore REALE contro la dequantizzazione CPU.
 * Uso: llm_packed_q2_test <model_dir> <tensor_base> <bits> [righe_di_x]
 * es.: llm_packed_q2_test ~/ai/Models/Qwen3.8-Flash-Next-2bit language_model.model.layers.2.linear_attn.out_proj 2
 * Compilazione: clang -O2 -I libs/mlx-deps/mlx-c-src -I LLMEngine/mlx -I LLMCodec/include tests/llm_packed_q2_test.c \
 *   -L LLMEngine/mlx/build-gmbasic -lllm_ai_mlx -framework Accelerate -o /tmp/llm_packed_q2_test */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "llm_mlx_wpacked.h"
#include "llm_mlx_packed_lazy.h"

static float bf16f(uint16_t u) { uint32_t v = (uint32_t)u << 16; float f; memcpy(&f, &v, 4); return f; }
static float halff(uint16_t u) { _Float16 h; memcpy(&h, &u, 2); return (float)h; }

/* locate dello shard che contiene <base>.weight (via index.json): ricerca testuale come negli altri moduli */
static int shard_of(const char* dir, const char* base, char* path, size_t cap) {
    char ip[1200]; snprintf(ip, sizeof ip, "%s/model.safetensors.index.json", dir);
    snprintf(path, cap, "%s/model.safetensors", dir);
    FILE* f = fopen(ip, "rb"); if (!f) return 0;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* j = (char*)malloc((size_t)n + 1); if (!j || fread(j, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(j); return 0; } fclose(f); j[n] = 0;
    char key[700]; snprintf(key, sizeof key, "\"%s.weight\"", base);
    char* p = strstr(j, key);
    if (p) { char* q = strstr(p, ".safetensors"); if (q) { char* b = q; while (b > j && b[-1] != '"') --b;
        snprintf(path, cap, "%s/%.*s", dir, (int)(q + 12 - b), b); } }
    free(j); return 0;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "uso: %s <model_dir> <tensor_base> <bits> [nt]\n", argv[0]); return 2; }
    const int bits = atoi(argv[3]), nt = argc > 4 ? atoi(argv[4]) : 1;
    char path[1600]; shard_of(argv[1], argv[2], path, sizeof path);
    LlmMlxPackedW w;
    if (llm_mlx_packed_open(path, argv[2], bits, 0, &w) != 0 || !w.biases) { fprintf(stderr, "open fallita: %s\n", path); return 1; }
    printf("%s: rows=%u cols=%u bits=%d group=%d bf16=%d\n", argv[2], w.rows, w.cols, w.bits, w.group, w.scale_bf16);
    const int rows = (int)w.rows, cols = (int)w.cols, ng = cols / w.group;
    float* x = (float*)malloc(sizeof(float) * (size_t)nt * cols);
    for (int t = 0; t < nt; t++) for (int i = 0; i < cols; i++) x[(size_t)t * cols + i] = 0.5f * sinf(0.37f * (float)i + (float)t) + 0.1f;
    double* yr = (double*)malloc(sizeof(double) * (size_t)nt * rows);
    const uint8_t* wb = (const uint8_t*)w.w; const uint8_t* sb = (const uint8_t*)w.scales; const uint8_t* bb = (const uint8_t*)w.biases;
    const uint32_t mask = (1u << bits) - 1u;
    for (int t = 0; t < nt; t++) for (int o = 0; o < rows; o++) {
        double acc = 0;
        for (int i = 0; i < cols; i++) {
            const size_t bitpos = ((size_t)o * cols + (size_t)i) * (size_t)bits;
            uint32_t word; memcpy(&word, wb + (bitpos / 32) * 4, 4);
            const unsigned q = (word >> (bitpos % 32)) & mask;
            const size_t si = (size_t)o * ng + (size_t)(i / w.group);
            uint16_t su, bu; memcpy(&su, sb + si * 2, 2); memcpy(&bu, bb + si * 2, 2);
            const float s = w.scale_bf16 ? bf16f(su) : halff(su), b = w.scale_bf16 ? bf16f(bu) : halff(bu);
            acc += (double)x[(size_t)t * cols + i] * ((double)q * s + b);
        }
        yr[(size_t)t * rows + o] = acc;
    }
    mlx_stream s = mlx_default_gpu_stream_new();
    LlmMlxPackedLazy z; memset(&z, 0, sizeof z);
    if (llm_mlx_lazy_build(&z, &w) != 0) { fprintf(stderr, "lazy_build fallita\n"); return 1; }
    int sh[2] = { nt, cols };
    mlx_array xa = mlx_array_new_data(x, sh, 2, MLX_FLOAT32), y = mlx_array_new();
    if (llm_mlx_lazy_linear(&z, &w, xa, &y, s) != 0) { fprintf(stderr, "lazy_linear fallita\n"); return 1; }
    mlx_vector_array ev = mlx_vector_array_new(); mlx_vector_array_append_value(ev, y); mlx_eval(ev); mlx_vector_array_free(ev);
    const float* yp = mlx_array_data_float32(y);
    double maxd = 0, ymax = 0;
    for (size_t k = 0; k < (size_t)nt * rows; k++) { double d = fabs((double)yp[k] - yr[k]); if (!(d <= maxd)) maxd = d; if (fabs(yr[k]) > ymax) ymax = fabs(yr[k]); }
    const double rel = maxd / (ymax > 0 ? ymax : 1);
    printf("max|y|=%.4g max|diff|=%.3e rel=%.3e  %s\n", ymax, maxd, rel, rel < 2e-3 ? "OK" : "FAIL");
    return rel < 2e-3 ? 0 : 1;
}
