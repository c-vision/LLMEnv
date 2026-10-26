/* generazione greedy nativa su GGUF qwen4exp (parita' col path MLX). */
#include "mlx/c/mlx.h"
#include "qwen4_native/qwen4_native_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int main(int argc, char** argv) {
    if (argc < 5) { printf("uso: %s <gguf> <n_new> <id...>\n", argv[0]); return 2; }
    int n_new = atoi(argv[2]);
    int np = argc - 3; int32_t* prompt = malloc((size_t)np * sizeof(int32_t));
    for (int i = 0; i < np; i++) prompt[i] = atoi(argv[3+i]);
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    if (dev.ctx == NULL || mlx_set_default_device(dev) != 0) return 1;
    mlx_stream s = mlx_stream_new_device(dev);
    Q4nModel m; if (q4n_model_open(&m, argv[1], 64, 1) != 0) { printf("open fail\n"); return 1; }
    const uint64_t V = llm_gguf_u(&m.gguf, "qwen4exp.vocab_size", 0, "qwen4_exp.vocab_size");
    float* logits = malloc((size_t)V * sizeof(float));
    for (int i = 0; i < np; i++) if (q4n_model_forward(&m, prompt[i], logits, s) != 0) return 1;
    printf("prompt:");
    for (int i = 0; i < np; i++) printf(" %d", prompt[i]);
    printf("\ngen:");
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    int last = prompt[np-1];
    for (int k = 0; k < n_new; k++) {
        int best = 0; float bv = logits[0];
        for (uint64_t v = 1; v < V; v++) if (logits[v] > bv) { bv = logits[v]; best = (int)v; }
        printf(" %d", best);
        last = best;
        if (k + 1 < n_new && q4n_model_forward(&m, last, logits, s) != 0) return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    printf("\n%.2f s  %.2f tok/s (incl. argmax)\n", dt, n_new / dt);
    return 0;
}
