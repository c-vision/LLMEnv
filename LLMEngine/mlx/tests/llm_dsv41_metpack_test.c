/* llm_dsv41_metpack_test.c — verifica che i kernel di packing Metal emettano gli STESSI byte
 * di `llmcodec_pack_fp8` (fp8 e4m3 grp32) e `llmcodec_pack_activation` (fp4 gs16 e4m3,
 * fp4 gs32 potenza-di-2). Nessun modello, solo device MLX. Atteso: ALL PASS.
 * Uso: tools/dsv41_metpack_test.sh */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mlx/c/mlx.h>

#include "llm_dsv41_fp8.h"
#include "llm_dsv41_fp4.h"
#include "llm_dsv41_metal.h"

static void fill(float* x, int n, unsigned seed) {
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        x[i] = ((float)((s >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f) * 4.0f;
    }
}

int main(void) {
    mlx_stream s = mlx_default_gpu_stream_new();
    int fail = 0;

    /* fp8 grp32 */
    {
        const int R = 5, D = 64, W = D + D / 32;
        float x[R * D]; uint8_t host[R * W], got[R * W];
        fill(x, R * D, 12345);
        for (int r = 0; r < R; r++) llmcodec_pack_fp8(x + r * D, D, host + r * W);
        mlx_array xa = mlx_array_new_data(x, (const int[]){ R, D }, 2, MLX_FLOAT32);
        mlx_array out = mlx_array_new();
        bool ok = llm_dsv41_metal_pack8(&out, xa, R, D, s);
        if (!ok) { printf("FAIL pack8 disponibile\n"); fail++; }
        else {
            mlx_array_eval(out);
            const uint8_t* p = mlx_array_data_uint8(out);
            memcpy(got, p, sizeof got);
            int bad = memcmp(host, got, sizeof host) != 0;
            printf("%s pack8 == host (fp8 grp32)\n", bad ? "FAIL" : "PASS");
            if (bad) { for (int i = 0; i < R * W && fail < 99; i++) if (host[i] != got[i]) { fprintf(stderr, "  byte[%d] host=%u got=%u\n", i, host[i], got[i]); break; } fail++; }
        }
    }
    /* fp4 gs16 e4m3 (pooled) */
    {
        const int R = 4, D = 32, GS = 16, W = D / 2 + D / GS;
        float x[R * D]; uint8_t host[R * W], got[R * W];
        fill(x, R * D, 987654);
        for (int r = 0; r < R; r++) llmcodec_pack_activation(x + r * D, D, 4, GS, 1, host + r * W);
        mlx_array xa = mlx_array_new_data(x, (const int[]){ R, D }, 2, MLX_FLOAT32);
        mlx_array out = mlx_array_new();
        bool ok = llm_dsv41_metal_pack4(&out, xa, R, D, GS, 1, s);
        if (!ok) { printf("FAIL pack4 gs16 disponibile\n"); fail++; }
        else {
            mlx_array_eval(out);
            memcpy(got, mlx_array_data_uint8(out), sizeof got);
            int bad = memcmp(host, got, sizeof host) != 0;
            printf("%s pack4 == host (fp4 gs16 e4m3)\n", bad ? "FAIL" : "PASS");
            if (bad) fail++;
        }
    }
    /* fp4 gs32 potenza-di-2 (index q/k) */
    {
        const int R = 3, D = 32, GS = 32, W = D / 2 + D / GS;
        float x[R * D]; uint8_t host[R * W], got[R * W];
        fill(x, R * D, 55555);
        for (int r = 0; r < R; r++) llmcodec_pack_activation(x + r * D, D, 4, GS, 0, host + r * W);
        mlx_array xa = mlx_array_new_data(x, (const int[]){ R, D }, 2, MLX_FLOAT32);
        mlx_array out = mlx_array_new();
        bool ok = llm_dsv41_metal_pack4(&out, xa, R, D, GS, 0, s);
        if (!ok) { printf("FAIL pack4 gs32 disponibile\n"); fail++; }
        else {
            mlx_array_eval(out);
            memcpy(got, mlx_array_data_uint8(out), sizeof got);
            int bad = memcmp(host, got, sizeof host) != 0;
            printf("%s pack4 == host (fp4 gs32 pow2)\n", bad ? "FAIL" : "PASS");
            if (bad) fail++;
        }
    }

    printf(fail ? "FAIL (%d)\n" : "ALL PASS\n", fail);
    return fail ? 1 : 0;
}
