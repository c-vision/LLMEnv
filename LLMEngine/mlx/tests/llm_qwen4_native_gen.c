#include "mlx/c/mlx.h"
#include "qwen4_native/qwen4_native_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
extern double g_q4n_prof[12];
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
    for (int i = 0; i < np; i++) q4n_model_forward(&m, prompt[i], logits, s);
    printf("prompt:"); for (int i = 0; i < np; i++) printf(" %d", prompt[i]);
    printf("\ngen:");
    memset(g_q4n_prof, 0, sizeof(double)*12);
    struct timespec a,b; clock_gettime(CLOCK_MONOTONIC,&a);
    int last = prompt[np-1];
    for (int k = 0; k < n_new; k++) {
        int best = 0; float bv = logits[0];
        for (uint64_t v = 1; v < V; v++) if (logits[v] > bv) { bv = logits[v]; best = (int)v; }
        printf(" %d", best); last = best;
        if (k + 1 < n_new) q4n_model_forward(&m, last, logits, s);
    }
    clock_gettime(CLOCK_MONOTONIC,&b);
    double dt = (b.tv_sec-a.tv_sec)+1e-9*(b.tv_nsec-a.tv_nsec);
    printf("\n%.2f s  %.2f tok/s\n", dt, n_new/dt);
    { int fw = n_new > 1 ? n_new - 1 : 1;
      double tot = g_q4n_prof[10], moe = g_q4n_prof[4]+g_q4n_prof[5]+g_q4n_prof[6]+g_q4n_prof[7]+g_q4n_prof[8];
      fprintf(stderr,"[prof/tok] tot=%.3f load=%.3f hc=%.3f comb+hc2=%.3f moe=%.3f lm=%.3f | router=%.3f gateup=%.3f swi=%.3f down=%.3f shared=%.3f gemv_r=%.3f\n",
              tot/fw,g_q4n_prof[0]/fw,g_q4n_prof[1]/fw,g_q4n_prof[2]/fw,moe/fw,g_q4n_prof[9]/fw,
              g_q4n_prof[4]/fw,g_q4n_prof[5]/fw,g_q4n_prof[6]/fw,g_q4n_prof[7]/fw,g_q4n_prof[8]/fw,g_q4n_prof[11]/fw); }
    return 0;
}
