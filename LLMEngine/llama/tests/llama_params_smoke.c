/* Smoke test (2026-10-30) per i nuovi parametri del ramo llama di LLMEnv.
 * Verifica: default, risoluzione AUTO di n_cpu_moe, load di un MoE GGUF con esperti
 * in CPU, sessione con batch/ubatch separati, generazione di qualche token.
 * Non e' un test di performance: serve a provare che il placement e i due budget
 * esistano e siano onorati. Uso:
 *   ./llama_params_smoke <model.gguf> [n_cpu_moe] [ram_budget_mb]
 */
#include "llmengine_llama.h"
#include <stdio.h>
/* teardown ordinato: libera modelli/sessioni PRIMA di ggml-metal (assert rsets) */
extern void llm_ai_shutdown(void);
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int path_exists(const char* p) {
    FILE* f = fopen(p, "rb");
    if (f) { fclose(f); return 1; }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "uso: %s <model.gguf> [n_cpu_moe] [ram_budget_mb]\n", argv[0]); return 2; }
    const char* path = argv[1];
    int n_cpu_moe = (argc > 2) ? atoi(argv[2]) : -1;
    int ram_mb    = (argc > 3) ? atoi(argv[3]) : 0;
    int ngl       = (argc > 4) ? atoi(argv[4]) : 0;
    int hold_s    = (argc > 5) ? atoi(argv[5]) : 0;

    llm_ai_model_params p;
    llm_ai_model_params_defaults(&p);
    if (p.size != (int)sizeof(p) || p.n_gpu_layers != -1 || p.n_cpu_moe != -2) {
        fprintf(stderr, "FAIL: default inattesi (size=%d ngl=%d ncmoe=%d)\n", p.size, p.n_gpu_layers, p.n_cpu_moe);
        return 1;
    }
    p.n_gpu_layers = ngl;   /* 0 = CPU, -1 = tutto su Metal */
    p.n_cpu_moe = n_cpu_moe;
    p.ram_budget_mb = ram_mb;

    int mh = llm_ai_load_model_ex(path, &p);
    if (mh < 0) { fprintf(stderr, "FAIL: load\n"); return 1; }
    printf("model handle=%d n_cpu_moe_effective=%d\n", mh, llm_ai_model_n_cpu_moe(mh));

    int64_t nparams = 0, size = 0; int nlayer = 0, nembd = 0, ntrain = 0;
    if (llm_ai_model_stats(mh, &nparams, &size, &nlayer, &nembd, &ntrain) != 0) {
        fprintf(stderr, "FAIL: stats\n"); return 1;
    }
    printf("params=%lld bytes=%lld layer=%d embd=%d ctx_train=%d\n",
           (long long)nparams, (long long)size, nlayer, nembd, ntrain);

    llm_ai_session_params sp;
    llm_ai_session_params_defaults(&sp);
    sp.n_ctx = 2048;
    sp.n_batch = 512;
    sp.n_ubatch = 128;
    sp.n_threads = 4;
    sp.n_threads_batch = 4;
    int sh = llm_ai_create_session_ex(mh, &sp);
    if (sh < 0) { fprintf(stderr, "FAIL: create_session_ex\n"); return 1; }

    /* prompt di 4 token: bastano per esercitare router + esperti */
    int32_t ids[4] = { 1, 2, 3, 4 };
    if (llm_ai_session_append_ids(sh, ids, 4) != 0) { fprintf(stderr, "FAIL: append_ids\n"); return 1; }
    int32_t out[8] = {0};
    int n = llm_ai_session_generate_ids(sh, 8, 0, out);
    if (n <= 0) { fprintf(stderr, "FAIL: generate_ids (%d)\n", n); return 1; }
    printf("generated %d token: ", n);
    for (int i = 0; i < n; ++i) printf("%d ", out[i]);
    printf("\n");

    if (hold_s > 0) { printf("pid=%d hold=%ds\n", (int)getpid(), hold_s); fflush(stdout); sleep(hold_s); }
    llm_ai_destroy_session(sh);
    llm_ai_unload_model(mh);
    llm_ai_shutdown();
    printf("OK\n");
    return 0;
}
