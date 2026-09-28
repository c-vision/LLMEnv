/* llm_bw_test.c — Fase 1: banda sostenibile dello stack (nessun modello).
 * Uso: llm_bw_test <size_mb> <mode 0=triad 1=read 2=write> <reps> */
#include "llmengine_mlx.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char** argv) {
    if (argc < 4) { printf("uso: %s <size_mb> <mode> <reps>\n", argv[0]); return 2; }
    char buf[256];
    int rc = llm_ai_mlx_bw_selftest(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), buf, (int)sizeof buf);
    printf("rc=%d %s\n", rc, buf);
    return rc == 0 ? 0 : 1;
}
