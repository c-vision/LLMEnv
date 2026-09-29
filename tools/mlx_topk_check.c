/* mlx_topk_check.c — verifica dell'accessore ABI v3 `token_topk` del motore MLX.
 *
 * Carica una dylib del motore via dlopen, risolve l'accessor della vtable e, dopo un prefill
 * di ids noti, controlla che la distribuzione top-k alla posizione "pending" sia COERENTE e
 * NON distruttiva:
 *   1. token_topk(k) ritorna esattamente k voci con logprob DECRESCENTI;
 *   2. ogni logprob coincide con `token_logprob(id)` (stessa distribuzione, stesso log-softmax);
 *   3. la voce 0 (argmax) coincide con `last_logprob` del token greedy pending;
 *   4. chiamare token_topk NON avanza lo stato: il prossimo generate(1) emette proprio i
 *      l'argmax restituito;
 *   5. due chiamate consecutive danno lo stesso risultato (lettura pura).
 *
 * Uso: mlx_topk_check <dylib> <accessor> <model_dir> <ids_csv> [K]
 * Stampa righe chiave=valore, poi OK / ERROR: <msg>.
 */
#include "llmengine.h"

#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef const LlmEngineLlmEngine* (*accessor_fn)(void);

static int parse_ids(const char* csv, int32_t** out, int* n_out) {
    int cap = 64, n = 0;
    int32_t* ids = (int32_t*)malloc((size_t)cap * sizeof(int32_t));
    if (!ids) return -1;
    const char* p = csv;
    while (*p) {
        char* end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        if (n == cap) {
            cap *= 2;
            int32_t* t = (int32_t*)realloc(ids, (size_t)cap * sizeof(int32_t));
            if (!t) { free(ids); return -1; }
            ids = t;
        }
        ids[n++] = (int32_t)v;
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    if (n == 0) { free(ids); return -1; }
    *out = ids;
    *n_out = n;
    return 0;
}

#define EPS 1e-3

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "uso: %s <dylib> <accessor> <model_dir> <ids_csv> [K]\n", argv[0]);
        return 2;
    }
    const char* dylib_path = argv[1];
    const char* accessor = argv[2];
    const char* model_dir = argv[3];
    const char* ids_csv = argv[4];
    int K = (argc > 5) ? atoi(argv[5]) : 5;
    if (K < 1) K = 5;

    void* lib = dlopen(dylib_path, RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        fprintf(stderr, "ERROR: dlopen(%s): %s\n", dylib_path, dlerror());
        return 3;
    }
    accessor_fn fn = (accessor_fn)dlsym(lib, accessor);
    if (!fn) {
        fprintf(stderr, "ERROR: dlsym(%s): %s\n", accessor, dlerror());
        return 4;
    }
    const LlmEngineLlmEngine* eng = fn();
    if (!eng) {
        fprintf(stderr, "ERROR: accessor NULL\n");
        return 5;
    }
    unsigned (*abi_ver)(void) = (unsigned (*)(void))dlsym(lib, "llmengine_abi_version");
    printf("ABI=%u\n", abi_ver ? abi_ver() : 0u);
    if (!eng->token_topk || !eng->token_logprob || !eng->last_logprob || !eng->generate_ids) {
        fprintf(stderr, "ERROR: la vtable non espone token_topk/logprob (ABI v3?)\n");
        return 6;
    }

    int32_t* ids = NULL;
    int n_ids = 0;
    if (parse_ids(ids_csv, &ids, &n_ids) != 0) {
        fprintf(stderr, "ERROR: ids non valide\n");
        return 7;
    }

    LlmEngineLlmModel model = eng->load_model(model_dir);
    if (!model) {
        fprintf(stderr, "ERROR: load_model(%s) fallita\n", model_dir);
        free(ids);
        return 8;
    }
    LlmEngineLlmSession sess = eng->create_session(model);
    if (!sess) {
        fprintf(stderr, "ERROR: create_session fallita\n");
        eng->free_model(model);
        free(ids);
        return 9;
    }
    LlmEngineLlmSampling sp;
    memset(&sp, 0, sizeof sp);
    sp.temperature = 0.0f;
    sp.top_p = 1.0f;
    sp.repetition_penalty = 1.0f;
    if (eng->set_sampling) eng->set_sampling(sess, &sp);
    if (eng->append_ids(sess, ids, n_ids) != 0) {
        fprintf(stderr, "ERROR: append_ids\n");
        return 10;
    }

    int32_t* top = (int32_t*)malloc((size_t)K * sizeof *top);
    float* lp = (float*)malloc((size_t)K * sizeof *lp);
    int32_t* top2 = (int32_t*)malloc((size_t)K * sizeof *top2);
    float* lp2 = (float*)malloc((size_t)K * sizeof *lp2);
    int rc = eng->token_topk(sess, K, top, lp);
    printf("TOPK_RC=%d\n", rc);
    if (rc != K) {
        fprintf(stderr, "ERROR: token_topk ha ritornato %d, atteso %d\n", rc, K);
        return 11;
    }
    int bad = 0;
    for (int i = 0; i < K; i++) {
        float ref = 0.0f;
        int trc = eng->token_logprob(sess, top[i], &ref);
        if (trc != 0 || fabsf(ref - lp[i]) > EPS) {
            fprintf(stderr, "ERROR: voce %d id=%d lp=%.6f vs token_logprob=%.6f (rc=%d)\n",
                    i, top[i], lp[i], ref, trc);
            bad = 1;
        }
        if (i > 0 && lp[i] > lp[i - 1] + EPS) {
            fprintf(stderr, "ERROR: ordine non decrescente in %d (%.6f > %.6f)\n", i, lp[i],
                    lp[i - 1]);
            bad = 1;
        }
    }
    float greedy = 0.0f;
    if (eng->last_logprob(sess, &greedy) != 0 || fabsf(greedy - lp[0]) > EPS) {
        fprintf(stderr, "ERROR: argmax topk lp=%.6f vs last_logprob=%.6f\n", lp[0], greedy);
        bad = 1;
    }
    /* Lettura pura: seconda chiamata identica. */
    if (eng->token_topk(sess, K, top2, lp2) != K || memcmp(top, top2, (size_t)K * sizeof *top) != 0) {
        fprintf(stderr, "ERROR: token_topk non e' idempotente (stato cambiato?)\n");
        bad = 1;
    }
    /* Non distruttiva: il prossimo token e' l'argmax. */
    int32_t next = -1;
    if (eng->generate_ids(sess, 1, -1, &next) != 1 || next != top[0]) {
        fprintf(stderr, "ERROR: generate(1)=%d, atteso argmax=%d (token_topk ha avanzato?)\n", next,
                top[0]);
        bad = 1;
    }

    printf("TOPK_IDS=");
    for (int i = 0; i < K; i++) printf("%s%d", i ? "," : "", top[i]);
    printf("\nTOPK_LP=");
    for (int i = 0; i < K; i++) printf("%s%.5f", i ? "," : "", lp[i]);
    printf("\nNEXT=%d\n", next);
    if (bad) {
        printf("ERROR: incoerenza top-k\n");
    } else {
        printf("OK\n");
    }

    free(top); free(lp); free(top2); free(lp2);
    eng->destroy_session(sess);
    eng->free_model(model);
    free(ids);
    dlclose(lib);
    return bad ? 1 : 0;
}
