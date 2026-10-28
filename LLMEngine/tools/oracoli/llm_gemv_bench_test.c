/* llm_gemv_bench_test.c — driver CLI di benchmark GEMV (quant/dequant) dell'engine MLX:
 * uso: gemv_bench <m> <n> <k> <bits> <group> <adt> <sdt> <reps> <pipe>; stampa il report. */
#include "llmengine_mlx.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char** argv) {
    if (argc < 10) { printf("uso: gemv_bench <m> <n> <k> <bits> <group> <adt> <sdt> <reps> <pipe>\n"); return 2; }
    char buf[192];
    extern int mlx_set_qmv_split_k(int v);
    if (argc >= 11) mlx_set_qmv_split_k(atoi(argv[10]));
    int rc = llm_ai_mlx_gemv_bench(atoi(argv[1]),atoi(argv[2]),atoi(argv[3]),atoi(argv[4]),atoi(argv[5]),atoi(argv[6]),atoi(argv[7]),atoi(argv[8]),atoi(argv[9]),buf,(int)sizeof buf);
    if (rc != 0) { printf("FAILED rc=%d\n", rc); return 1; }
    printf("%s\n", buf);
    return 0;
}
