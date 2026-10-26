/* PPL nativa teacher-forced su testo (tokenizer del modello). */
#include "mlx/c/mlx.h"
#include "qwen4_native/qwen4_native_model.h"
#include "llmtoken.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
int main(int argc, char** argv) {
    if (argc < 4) { printf("uso: %s <gguf> <model_dir> <testo>\n", argv[0]); return 2; }
    char tp[1300]; snprintf(tp, sizeof tp, "%s/tokenizer.json", argv[2]);
    llmtoken_handle th = llmtoken_load(tp);
    if (!th) { printf("tokenizer fail\n"); return 1; }
    int32_t ids[256]; int n = llmtoken_encode(th, argv[3], ids, 256);
    if (n <= 1) { printf("encode fail\n"); return 1; }
    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    if (dev.ctx == NULL || mlx_set_default_device(dev) != 0) return 1;
    mlx_stream s = mlx_stream_new_device(dev);
    Q4nModel m; if (q4n_model_open(&m, argv[1], 256, 1) != 0) { printf("open fail\n"); return 1; }
    { struct rusage ru; getrusage(RUSAGE_SELF, &ru); fprintf(stderr, "[rss] open=%ldMB\n", ru.ru_maxrss/1024/1024); }
    const uint64_t V = llm_gguf_u(&m.gguf, "qwen4exp.vocab_size", 0, "qwen4_exp.vocab_size");
    float* logits = malloc((size_t)V * sizeof(float));
    double nll = 0; int cnt = 0;
    printf("ids(%d):", n); for (int i = 0; i < n; i++) printf(" %d", ids[i]); printf("\n");
    for (int i = 0; i < n; i++) {
        if (q4n_model_forward(&m, ids[i], logits, s) != 0) { printf("fwd fail %d\n", i); return 1; }
        { struct rusage ru; getrusage(RUSAGE_SELF, &ru); fprintf(stderr, "[rss] tok%d=%ldMB\n", i, ru.ru_maxrss/1024/1024); }
        if (i + 1 < n) {
            double mx = logits[0]; for (uint64_t v = 1; v < V; v++) if (logits[v] > mx) mx = logits[v];
            double sum = 0; for (uint64_t v = 0; v < V; v++) sum += exp((double)logits[v] - mx);
            double lse = mx + log(sum);
            nll += lse - (double)logits[ids[i+1]];
            cnt++;
        }
    }
    printf("NLL medio=%.6f  PPL=%.4f (n=%d)\n", nll / cnt, exp(nll / cnt), cnt);
    return 0;
}
