/* smoke: forward nativo su uno slice GGUF reale (use_moe=0). */
#include "mlx/c/mlx.h"
#include "qwen4_native/qwen4_native_model.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc < 2) { printf("uso: %s <slice.gguf>\n", argv[0]); return 2; }
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    if (dev.ctx == NULL || mlx_set_default_device(dev) != 0) { printf("FAIL device\n"); return 1; }
    mlx_stream s = mlx_stream_new_device(dev);
    Q4nModel m;
    fprintf(stderr, "[smoke] prima di open\n");
    int orc = q4n_model_open(&m, argv[1], 8, argc > 2 ? atoi(argv[2]) : 0);
    fprintf(stderr, "[smoke] open rc=%d L=%d\n", orc, m.hp.L);
    if (orc != 0) { printf("FAIL open\n"); return 1; }
    printf("model: E=%d L=%d hc=%d rank=%d lin(Hk=%d Hv=%d D=%d) full_interval=%d\n",
           m.hp.E, m.hp.L, m.hp.hc, m.hp.rank, m.hp.lin_Hk, m.hp.lin_Hv, m.hp.lin_D, m.hp.full_interval);
    { int nl = 0, nf = 0; for (int i = 0; i < m.hp.L; i++) if (m.layer_linear && m.layer_linear[i]) nl++; else nf++;
      printf("layer_types: linear=%d full=%d\n", nl, nf); }
    const uint64_t V = llm_gguf_u(&m.gguf, "qwen4exp.vocab_size", 0, "qwen4_exp.vocab_size");
    float* logits = malloc((size_t)V * sizeof(float));
    int tokens[2] = {248044, 9707};
    for (int t = 0; t < 2; t++) {
        if (q4n_model_forward(&m, tokens[t], logits, s) != 0) { printf("FAIL forward %d\n", t); return 1; }
        int best = 0; double bv = logits[0];
        for (uint64_t v = 1; v < V; v++) if (logits[v] > bv) { bv = logits[v]; best = (int)v; }
        printf("tok %d -> argmax=%d (%.4f)\n", tokens[t], best, bv);
    }
    free(logits);
    q4n_model_free(&m);
    printf("P7f SMOKE: OK\n");
    return 0;
}
