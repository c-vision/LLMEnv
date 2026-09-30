/* tools/dflash_multi.c — A2: verifica `llm_ai_mlx_dflash2_generate_multi` (N sessioni in
 * round-loop batched) contro N esecuzioni single-sessione. Parità = stessi id greedy.
 *
 * Uso: dflash_multi <dylib> <target_dir> <draft_dir> <N_out> <nsess>
 * (prompt: lista id fissa, valida per il modello; la parità non dipende dal testo.)
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

typedef void* M; typedef void* S; typedef void* DF;

int main(int argc, char** argv) {
    if (argc < 6) { fprintf(stderr, "uso: %s <dylib> <target> <draft> <N_out> <nsess>\n", argv[0]); return 2; }
    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 3; }
    M  (*load)(const char*)             = dlsym(h, "llm_ai_mlx_load_model");
    S  (*create)(M)                     = dlsym(h, "llm_ai_mlx_create_session");
    int(*append)(S, const int32_t*, int)= dlsym(h, "llm_ai_mlx_session_append");
    DF (*dfcreate)(M, const char*)      = dlsym(h, "llm_ai_mlx_dflash2_create");
    int(*dfgen)(DF, S, int32_t, int, int32_t*) = dlsym(h, "llm_ai_mlx_dflash2_generate");
    int(*dfmulti)(DF, S*, int, int32_t, const int*, int, int32_t*, int*) = dlsym(h, "llm_ai_mlx_dflash2_generate_multi");
    int(*optset)(const char*, const char*) = dlsym(h, "llm_ai_mlx_opt_set");
    if (!load || !create || !append || !dfcreate || !dfgen || !dfmulti) { fprintf(stderr, "dlsym fallito\n"); return 3; }
    if (optset) optset("spec_ngram", "0");

    int N = atoi(argv[4]); int NS = atoi(argv[5]);
    if (N <= 0 || N > 256) N = 32;
    if (NS <= 0 || NS > 8) NS = 3;

    /* prompt DISTINTO per sessione (id validi): verifica l'indipendenza degli slot. */
    int32_t base[] = {9707, 11, 1879, 1917, 374, 220, 323, 358, 1079, 13};
    int nb = (int)(sizeof(base)/sizeof(base[0]));
    int32_t ids[16]; int nids;

    M m = load(argv[2]);
    if (!m) { fprintf(stderr, "load target NULL\n"); return 4; }
    DF d = dfcreate(m, argv[3]);
    if (!d) { fprintf(stderr, "dflash2_create NULL\n"); return 5; }

    /* riferimento: NS sessioni single (una per volta), prompt distinto per sessione. */
    int32_t ref[8][256]; int rn[8];
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < NS; i++) {
        nids = 0; for (int k = 0; k < nb; k++) ids[nids++] = base[k];
        ids[nids++] = 1000 + i * 13;   /* token distinto per sessione */
        S s = create(m); append(s, ids, nids);
        rn[i] = dfgen(d, s, 248044 /*eos*/, N, ref[i]);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double single_ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    /* multi: NS sessioni in batch, stesso prompt distinto. */
    S ss[8];
    for (int i = 0; i < NS; i++) {
        nids = 0; for (int k = 0; k < nb; k++) ids[nids++] = base[k];
        ids[nids++] = 1000 + i * 13;
        ss[i] = create(m); append(ss[i], ids, nids);
    }
    int maxarr[8]; for (int i = 0; i < NS; i++) maxarr[i] = N;
    int stride = N; int32_t* out = (int32_t*)calloc((size_t)NS*stride, sizeof(int32_t));
    int counts[8] = {0};
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int rc = dfmulti(d, ss, NS, 248044, maxarr, stride, out, counts);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double multi_ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("generate_multi rc=%d | tempo: %d single = %.0f ms, multi = %.0f ms (speedup %.2fx)\n",
           rc, NS, single_ms, multi_ms, multi_ms > 0 ? single_ms / multi_ms : 0.0);

    int all_ok = (rc == 0);
    for (int i = 0; i < NS; i++) {
        int mc = counts[i], rc2 = rn[i];
        int mn = mc < rc2 ? mc : rc2; int first = -1;
        for (int k = 0; k < mn; k++) if (out[(size_t)i*stride+k] != ref[i][k]) { first = k; break; }
        int ok = (first < 0 && mc == rc2);
        all_ok = all_ok && ok;
        printf("sess %d: single=%d multi=%d parita=%s%s\n", i, rc2, mc, ok ? "OK" : "DIVERGE",
               first >= 0 ? " (primo diff)" : "");
        if (first >= 0) printf("   ref[%d]=%d multi=%d\n", first, ref[i][first], out[(size_t)i*stride+first]);
    }
    printf("ESITO: %s\n", all_ok ? "OK" : "FAIL");
    free(out);
    return all_ok ? 0 : 1;
}
