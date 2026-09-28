/* llm_imatrix_smoke_test.c — §E2: smoke test del collector imatrix generalizzato.
 * Uso: LLM_... llm_imatrix_smoke_test <model_dir> [out.txt]
 * Carica un MoE, attiva l'imatrix, appende token di calibrazione (>=64 -> path sorted), scrive
 * il report e ne verifica il contenuto (righe > 0, energie finite e non nulle). */
#include "llmengine_mlx.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "uso: %s <model_dir> [out.txt]\n", argv[0]); return 2; }
    const char* out = argc > 2 ? argv[2] : "/tmp/imatrix_smoke.txt";

    llm_ai_mlx_model_handle mh = llm_ai_mlx_load_model(argv[1]);
    if (!mh) { fprintf(stderr, "load_model fallita\n"); return 3; }

    if (llm_ai_mlx_model_imatrix_begin(mh) != 0) { fprintf(stderr, "imatrix_begin fallito (MoE?)\n"); return 4; }

    llm_ai_mlx_session_handle sh = llm_ai_mlx_create_session(mh);
    int32_t toks[256];
    for (int i = 0; i < 256; i++) toks[i] = 1 + (i % 1000);
    for (int off = 0; off < 256; off += 128) {   /* 2 chunk: il 2o e' prefill multi-chunk */
        if (llm_ai_mlx_session_append(sh, toks + off, 128) != 0) { fprintf(stderr, "append fallito\n"); return 5; }
    }
    for (int i = 0; i < 3; i++) {                 /* decode: n_tokens=1 -> path UNSORTED */
        int32_t one = 100 + i;
        if (llm_ai_mlx_session_append(sh, &one, 1) != 0) { fprintf(stderr, "append(1) fallito\n"); return 5; }
    }
    if (llm_ai_mlx_model_imatrix_write(mh, out) != 0) { fprintf(stderr, "imatrix_write fallito\n"); return 6; }

    FILE* f = fopen(out, "r");
    if (!f) { fprintf(stderr, "report non scritto\n"); return 7; }
    char line[8192];
    long rows = 0, nonfinite = 0; double gsum = 0.0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        rows++;
        /* somma la 4^ colonna (prima energia gate) come checksum */
        double v = 0; int col = 0; char* p = line;
        while (*p && col < 4) { char* e = NULL; v = strtod(p, &e); if (e == p) break; p = e; col++; while (*p == ' ') p++; }
        if (!isfinite(v)) nonfinite++;
        gsum += v;
    }
    fclose(f);
    printf("imatrix: rows=%ld nonfinite=%ld sum(col4)=%.6g\n", rows, nonfinite, gsum);
    int ok = (rows > 0 && nonfinite == 0 && gsum > 0.0);
    printf("smoke: %s\n", ok ? "OK (collector generalizzato produce energia)" : "FAIL");
    llm_ai_mlx_destroy_session(sh);
    return ok ? 0 : 1;
}
