/* dflash2_validate.c -- harness di validazione del forward nativo DFlash2
 * (Qwen3.8-27B-DFlash2) contro i golden reference.
 *
 * Carica il TARGET con llm_ai_mlx_load_model (serve per embed/lm_head, ~28G),
 * crea il runtime DFlash2 via llm_ai_mlx_dflash2_create(draft_dir), poi esegue
 * UN draft_block (round0 del golden): propose(anchor, ctx[5,25600], S=5) e
 * confronta i 7 token prodotti con round0_draft atteso.
 *
 * Build (linka la lib):
 *   clang -O2 dflash2_validate.c -I ../.. -L ../../build -lllm_ai_mlx \
 *         -o dflash2_validate
 * Uso:
 *   ./dflash2_validate [target_dir] [draft_dir] [round0.bin] [--dump-snaps]
 *
 * Criterio: i 7 token devono ESSERE IDENTICI a round0_draft. Con --dump-snaps
 * stampa anche maxdiff dei layer snaps (layer0..4, final_norm) per isolare
 * conv vs attenzione vs fc vs selector in caso di mismatch.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "llmengine_mlx.h"

int main(int argc, char** argv) {
    const char* target = argc > 1 ? argv[1] : "/Users/gaetano/ai/Models/Qwen3.8-27B-8bit";
    const char* draft  = argc > 2 ? argv[2] : "/Users/gaetano/ai/Models/Qwen3.8-27B-DFlash2";
    const char* bin    = argc > 3 ? argv[3] : NULL;
    const bool dump_snaps = argc > 4 && strcmp(argv[4], "--dump-snaps") == 0;

    char binbuf[1024];
    if (!bin) {
        snprintf(binbuf, sizeof(binbuf), "%s/goldens/dflash2/round0.bin", ".");
        bin = binbuf;
    }

    FILE* f = fopen(bin, "rb");
    if (!f) { fprintf(stderr, "impossibile aprire %s\n", bin); return 2; }
    int S = 0, anchor = 0, n_exp = 0;
    if (fread(&S, 4, 1, f) != 1) return 2;
    if (fread(&anchor, 4, 1, f) != 1) return 2;
    if (fread(&n_exp, 4, 1, f) != 1) return 2;
    int* expected = (int*)malloc((size_t)n_exp * sizeof(int));
    if (fread(expected, 4, (size_t)n_exp, f) != (size_t)n_exp) return 2;
    int n_ctx = 0;
    if (fread(&n_ctx, 4, 1, f) != 1) return 2;
    float* ctx = (float*)malloc((size_t)n_ctx * sizeof(float));
    if (fread(ctx, sizeof(float), (size_t)n_ctx, f) != (size_t)n_ctx) return 2;
    int n_snaps = 0, sh0 = 0, sh1 = 0, sh2 = 0;
    float* snaps = NULL;
    size_t snap_count = 0;
    if (dump_snaps) {
        if (fread(&n_snaps, 4, 1, f) != 1) return 2;
        if (fread(&sh0, 4, 1, f) != 1) return 2;
        if (fread(&sh1, 4, 1, f) != 1) return 2;
        if (fread(&sh2, 4, 1, f) != 1) return 2;
        snap_count = (size_t)sh0 * sh1 * sh2;
        snaps = (float*)malloc(snap_count * (size_t)n_snaps * sizeof(float));
        if (fread(snaps, sizeof(float), snap_count * (size_t)n_snaps, f) != snap_count * (size_t)n_snaps) return 2;
    }
    fclose(f);

    printf("[validate] S=%d anchor=%d expected(%d) ctx_float=%d\n", S, anchor, n_exp, n_ctx);

    llm_ai_mlx_model_handle target_h = llm_ai_mlx_load_model(target);
    if (!target_h) { fprintf(stderr, "caricamento target fallito: %s\n", target); return 1; }
    printf("[validate] target caricato\n");

    llm_ai_mlx_dflash2_handle rt = llm_ai_mlx_dflash2_create(target_h, draft);
    if (!rt) { fprintf(stderr, "create DFlash2 fallito\n"); llm_ai_mlx_free_model(target_h); return 1; }
    printf("[validate] runtime DFlash2 creato\n");

    if (dump_snaps) {
        /* dump per-layer (layer0..4 + final_norm) e confronto maxdiff col golden */
        size_t cap = 6 * snap_count;
        float* mine = (float*)malloc(cap * sizeof(float));
        int nw = llm_ai_mlx_dflash2_debug_layers(rt, (int32_t)anchor, ctx, S, mine, (int)cap);
        if (nw < 0) {
            printf("[validate] debug_layers fallito (%d)\n", nw);
        } else {
            const char* tags[6] = {"layer0", "layer1", "layer2", "layer3", "layer4", "final_norm"};
            for (int si = 0; si < n_snaps && si < 6; si++) {
                const float* g = snaps + (size_t)si * snap_count;
                const float* m = mine + (size_t)si * snap_count;
                double maxd = 0.0, sum = 0.0, sq = 0.0;
                for (size_t i = 0; i < snap_count; i++) {
                    double dd = fabs((double)g[i] - (double)m[i]);
                    if (dd > maxd) maxd = dd;
                    sq += dd * dd;
                }
                double rmse = sqrt(sq / (double)snap_count);
                printf("[validate] %-10s maxdiff=%.6g rmse=%.6g  mine[0..3]=%.4g %.4g %.4g %.4g  gold[0..3]=%.4g %.4g %.4g %.4g\n",
                       tags[si], maxd, rmse,
                       m[0], m[1], m[2], m[3], g[0], g[1], g[2], g[3]);
            }
        }
        free(mine);
    }

    int32_t out[16];
    int got = llm_ai_mlx_dflash2_propose(rt, (int32_t)anchor, ctx, S, out, 16);
    if (got < 0) {
        fprintf(stderr, "propose fallito\n");
        llm_ai_mlx_dflash2_free(rt);
        llm_ai_mlx_free_model(target_h);
        return 1;
    }
    printf("[validate] draft prodotti (%d):", got);
    for (int i = 0; i < got; i++) printf(" %d", (int)out[i]);
    printf("\n");

    int match = 1;
    for (int i = 0; i < n_exp && i < got; i++)
        if (out[i] != expected[i]) match = 0;

    printf("[validate] expected          :");
    for (int i = 0; i < n_exp; i++) printf(" %d", expected[i]);
    printf("\n");

    free(expected); free(ctx);
    llm_ai_mlx_dflash2_free(rt);
    llm_ai_mlx_free_model(target_h);
    if (!dump_snaps) free(snaps);

    printf("[validate] round0 %s\n", match && got == n_exp ? "MATCH" : "NON-MATCH");
    return (match && got == n_exp) ? 0 : 1;
}
