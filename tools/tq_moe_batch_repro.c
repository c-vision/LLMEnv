/* tools/tq_moe_batch_repro.c — repro DETERMINISTICO del fallimento di `forward_tokens_batch`
 * per n piccolo (2..7) sul MoE TurboQuant (ramo unsorted).
 *
 * Carica il modello una volta (via dlopen sulla dylib del motore) e, per ogni lunghezza
 * n = 1..8 e 14, crea una sessione nuova, fa `llm_ai_mlx_session_append(ids[0..n))` (che
 * internamente usa `forward_tokens_batch(n)` nel prefill) e stampa l'esito. Prima del fix
 * n=2..7 falliva (`shared_expert lookup failed`); n=1 e n>=8 ok. Niente env, input fisso.
 *
 * Uso:
 *   clang -O2 tools/tq_moe_batch_repro.c -o /tmp/tqrepro
 *   /tmp/tqrepro <dylib> <model_dir> <id,id,...>
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef void* model_h;
typedef void* sess_h;

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "uso: %s <dylib> <model_dir> [id,id,...]\n", argv[0]); return 2; }
    const char* dylib = argv[1];
    const char* model = argv[2];

    int32_t seed[4096]; int n_seed = 0;
    if (argc >= 4) { const char* p = argv[3]; char* end = NULL;
        while (*p && n_seed < 4096) { long v = strtol(p, &end, 10); if (end == p) break; seed[n_seed++] = (int32_t)v; p = end; if (*p == ',') p++; } }
    if (n_seed <= 0) { seed[0] = 1; n_seed = 1; }

    void* h = dlopen(dylib, RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 3; }
    model_h (*load)(const char*) = (model_h (*)(const char*))dlsym(h, "llm_ai_mlx_load_model");
    sess_h  (*create)(model_h) = (sess_h (*)(model_h))dlsym(h, "llm_ai_mlx_create_session");
    int     (*append)(sess_h, const int32_t*, int) = (int (*)(sess_h, const int32_t*, int))dlsym(h, "llm_ai_mlx_session_append");
    if (!load || !create || !append) { fprintf(stderr, "dlsym fallito\n"); return 3; }

    model_h m = load(model);
    if (!m) { fprintf(stderr, "load_model NULL\n"); return 4; }

    const int lens[] = { 1, 2, 3, 4, 5, 6, 7, 8, 14 };
    int nfail = 0;
    for (unsigned li = 0; li < sizeof(lens) / sizeof(lens[0]); li++) {
        int n = lens[li];
        int32_t ids[64];
        for (int i = 0; i < n; i++) ids[i] = seed[i % n_seed];
        sess_h s = create(m);
        if (!s) { printf("append n=%-2d CREATE_NULL\n", n); nfail++; continue; }
        int rc = append(s, ids, n);
        printf("append n=%-2d rc=%d%s\n", n, rc, rc != 0 ? "   <== FALLIMENTO" : "");
        if (rc != 0) nfail++;
    }
    printf("FALLIMENTI: %d\n", nfail);
    return nfail ? 1 : 0;
}
