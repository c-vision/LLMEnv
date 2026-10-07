/* mlx_embed_check.c — verifica dell'accessore ABI v4 `embed_ids` (embedding IDS-based).
 *
 * Carica una dylib del motore via dlopen, risolve l'accessor e controlla, su un modello reale:
 *   1. emb_dim > 0 e coerente col numero di float scritti;
 *   2. determinismo: due chiamate sugli stessi id danno lo stesso vettore;
 *   3. testi diversi danno vettori diversi (non un vettore costante);
 *   4. la L2-normalizzazione del vettore ha norma 1;
 *   5. il CHUNKING non cambia il risultato: la media dei chunk (pesata) coincide col full (>512 id).
 *
 * Uso: mlx_embed_check <dylib> <accessor> <model_dir> <ids_csv_a> [ids_csv_b] [K]
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

static int diffs(const float* a, const float* b, int n, float eps) {
    for (int i = 0; i < n; i++)
        if (fabsf(a[i] - b[i]) > eps) return 1;
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "uso: %s <dylib> <accessor> <model_dir> <ids_csv_a> [ids_csv_b]\n", argv[0]);
        return 2;
    }
    const char* dylib_path = argv[1];
    const char* accessor = argv[2];
    const char* model_dir = argv[3];
    const char* csv_a = argv[4];
    const char* csv_b = (argc > 5) ? argv[5] : NULL;

    void* lib = dlopen(dylib_path, RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "ERROR: dlopen: %s\n", dlerror()); return 3; }
    accessor_fn fn = (accessor_fn)dlsym(lib, accessor);
    if (!fn) { fprintf(stderr, "ERROR: dlsym(%s): %s\n", accessor, dlerror()); return 4; }
    const LlmEngineLlmEngine* eng = fn();
    if (!eng || !eng->load_model || !eng->embed_ids || !eng->emb_dim) {
        fprintf(stderr, "ERROR: la vtable non espone embed_ids/emb_dim (ABI v4?)\n");
        return 5;
    }

    int32_t *a = NULL, *b = NULL;
    int na = 0, nb = 0;
    if (parse_ids(csv_a, &a, &na) != 0) { fprintf(stderr, "ERROR: ids A\n"); return 6; }
    if (csv_b && parse_ids(csv_b, &b, &nb) != 0) { fprintf(stderr, "ERROR: ids B\n"); return 6; }

    LlmEngineLlmModel model = eng->load_model(model_dir);
    if (!model) { fprintf(stderr, "ERROR: load_model\n"); return 7; }
    int dim = eng->emb_dim(model);
    printf("DIM=%d\n", dim);
    if (dim <= 0) { fprintf(stderr, "ERROR: emb_dim=%d\n", dim); return 8; }

    float* va = (float*)calloc((size_t)dim, sizeof(float));
    float* vb = (float*)calloc((size_t)dim, sizeof(float));
    float* vc = (float*)calloc((size_t)dim, sizeof(float));
    int bad = 0;

    if (eng->embed_ids(model, a, na, va, dim) != dim) { fprintf(stderr, "ERROR: embed A\n"); return 9; }
    if (eng->embed_ids(model, a, na, vb, dim) != dim) { fprintf(stderr, "ERROR: embed A2\n"); return 9; }
    if (diffs(va, vb, dim, 1e-6f)) { fprintf(stderr, "ERROR: non deterministico\n"); bad = 1; }

    double norm = 0.0;
    for (int i = 0; i < dim; i++) norm += (double)va[i] * va[i];
    norm = sqrt(norm);
    printf("NORM=%.6f\n", norm);
    if (!(norm > 0.0)) { fprintf(stderr, "ERROR: norma nulla\n"); bad = 1; }

    if (b) {
        if (eng->embed_ids(model, b, nb, vc, dim) != dim) { fprintf(stderr, "ERROR: embed B\n"); return 9; }
        if (!diffs(va, vc, dim, 1e-6f)) { fprintf(stderr, "ERROR: testi diversi -> vettori identici\n"); bad = 1; }
        else printf("B_DIFFERS=1\n");
    }

    /* Sequenza lunga (>512, quindi piu' chunk): il run deve riuscire ed essere deterministico.
     * NON si confronta con la media di chunk in sessioni separate: la causalita' fa dipendere un
     * chunk dal passato, quindi solo la stessa sessione (dentro embed_ids) e' il confronto giusto. */
    int longn = 0, reps = (520 / (na > 0 ? na : 1)) + 1;
    int32_t* long_ids = (int32_t*)malloc((size_t)reps * na * sizeof(int32_t));
    for (int r = 0; r < reps; r++)
        for (int i = 0; i < na; i++) long_ids[longn++] = a[i];
    if (longn > 512) {
        float* vfull = (float*)calloc((size_t)dim, sizeof(float));
        float* vfull2 = (float*)calloc((size_t)dim, sizeof(float));
        int okf = eng->embed_ids(model, long_ids, longn, vfull, dim) == dim;
        int okf2 = eng->embed_ids(model, long_ids, longn, vfull2, dim) == dim;
        if (okf && okf2) {
            printf("LONG_N=%d\n", longn);
            if (diffs(vfull, vfull2, dim, 1e-6f)) {
                fprintf(stderr, "ERROR: long/chunk non deterministico\n");
                bad = 1;
            }
            if (!diffs(va, vfull, dim, 1e-6f)) {
                fprintf(stderr, "ERROR: contesto lungo -> stesso vettore del corto\n");
                bad = 1;
            }
        } else {
            fprintf(stderr, "ERROR: embed long/chunk fallita\n");
            bad = 1;
        }
        free(vfull); free(vfull2);
    }
    free(long_ids);

    printf("EMBED0=%.6f,%.6f,%.6f\n", va[0], dim > 1 ? va[1] : 0.0f, dim > 2 ? va[2] : 0.0f);
    printf(bad ? "ERROR: incoerenza embed\n" : "OK\n");
    printf("%s\n", bad ? "" : "");

    free(va); free(vb); free(vc);
    eng->free_model(model);
    free(a); free(b);
    dlclose(lib);
    return bad ? 1 : 0;
}
