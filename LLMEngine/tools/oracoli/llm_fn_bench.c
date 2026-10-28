/* llm_fn_bench.c — ORACOLO per-FUNZIONE (model-free): chiama l'entry del motore
 * `llm_ai_mlx_core_bench(op, iters, out, cap)` via dlopen e stampa il report
 * per-primitiva (rms_norm, qlinear, sdpa, swiglu, kvq_sdpa, zero_centered_rmsnorm,
 * sigmoid_rmsnorm_gated, qk_l2norm, moe_mxfp4, layer, ...). Nessun modello, array
 * sintetici. Uso:  llm_fn_bench <dylib> [op|all] [iters]
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (*bench_fn)(const char* op, int iters, char* out, size_t cap);

int main(int argc, char** argv) {
    const char* lib = (argc > 1) ? argv[1] : "build/LLMEngine/mlx/libllm_ai_mlx.dylib";
    const char* op  = (argc > 2) ? argv[2] : "all";
    int iters       = (argc > 3) ? atoi(argv[3]) : 200;

    void* h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen(%s): %s\n", lib, dlerror()); return 1; }
    bench_fn f = (bench_fn)dlsym(h, "llm_ai_mlx_core_bench");
    if (!f) { fprintf(stderr, "dlsym: %s\n", dlerror()); return 1; }

    size_t cap = 1u << 20;
    char* out = (char*)malloc(cap);
    if (!out) return 2;
    int rc = f(op, iters, out, cap);
    fputs(out, stdout);
    fprintf(stderr, "rc=%d\n", rc);
    free(out);
    dlclose(h);
    return 0;
}
