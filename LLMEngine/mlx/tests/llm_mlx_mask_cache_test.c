/* llm_mlx_mask_cache_test.c — driver CLI del self-test della cache delle maschere
 * attenzione (sliding-window) dell'engine MLX: llm_ai_mlx_mask_cache_selftest(Lq,Nk,win). */
#include "llmengine_mlx.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char** argv) {
    int Lq = argc > 1 ? atoi(argv[1]) : 8, Nk = argc > 2 ? atoi(argv[2]) : 16, win = argc > 3 ? atoi(argv[3]) : 8;
    char buf[128];
    if (llm_ai_mlx_mask_cache_selftest(Lq, Nk, win, buf, (int)sizeof buf) != 0) { printf("SELFTEST FAILED\n"); return 1; }
    printf("%s\n", buf);
    return 0;
}
