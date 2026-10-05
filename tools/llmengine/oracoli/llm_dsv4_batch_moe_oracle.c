/* llm_dsv4_batch_moe_oracle.c — ORACOLO **MODEL-FREE** per la radice del bug DS4
 * "garbage a freddo nel prefill a chunk" (Bugs/(open) - DS4-Garbage-Cold-Prefill-Batch-Zeri).
 *
 * NON carica il MODELLO e NON usa la GPU: costruisce un file SINTETICO di record-esperto con un
 * pattern noto e verifica la PROPRIETA' che il batch MoE assume
 * (`llm_dsv4_gpu_moe.h:263`): «gli esperti dell'unione sono letti una volta -> puntatori stabili
 * (cap >= used*nt)».
 *
 * METODO (il "collect-then-use" del batch, riprodotto senza modello):
 *   1. `llm_dsv4_store_prefetch(store, layer, union, n)`  — come fa `llm_dsv4_gpu_moe_batch` (riga 304);
 *   2. raccoglie TUTTI i puntatori `gp[i]=store_get(e_i)` PRIMA di usarli (come righe 305-321);
 *   3. verifica che il puntatore i-esimo contenga ANCORA il pattern dell'esperto i.
 *   - `cap >= n`: atteso PASS (nessun riuso di slot -> puntatori stabili);
 *   - `cap <  n`: se lo store riusa gli slot (LRU), i puntatori raccolti vedono il contenuto di un
 *                 ALTRO esperto -> FAIL = **proprieta' violata, bug riprodotto model-free**.
 *
 * Il fix e' rendere STABILI i puntatori quando `cap < used*nt` (processare il chunk in sub-chunk di
 * `cap/used` righe, oppure ricadere sul path per-token). Questo oracolo e' la guardia: prima del fix
 * il caso `cap < union` DEVE fallire; dopo il fix DEVE passare.
 *
 * NON ESEGUIRE senza necessita' (e' comunque model-free: nessun caricamento di pesi). Build: vedi
 * `tools/llmengine/oracoli/build_probes.sh` (aggiungere questo sorgente all'elenco).
 */
#include "llm_dsv4_store.h"   /* store SWAP/LRU + prefetch/get (static inline) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

#define NB  64    /* byte per record-esperto (gate / up / down) */
#define NEX 16    /* esperti per layer */
#define NLY 2     /* layer */

static size_t part_off(int layer, int part) { return ((size_t)layer * 3 + (size_t)part) * ((size_t)NEX * NB); }
static unsigned char pat(int layer, int part, int e, size_t k) {
    return (unsigned char)((layer * 13 + part * 5 + e + (int)k) & 0xFF);
}

/* Scrive il file sintetico: per (layer, part, expert) un record di NB byte col pattern `pat`. */
static int make_file(const char* path) {
    size_t total = (size_t)NLY * 3 * NEX * NB;
    unsigned char* p = (unsigned char*)calloc(1, total);
    if (!p) return -1;
    for (int L = 0; L < NLY; L++)
        for (int part = 0; part < 3; part++)
            for (int e = 0; e < NEX; e++)
                for (size_t k = 0; k < NB; k++)
                    p[part_off(L, part) + (size_t)e * NB + k] = pat(L, part, e, k);
    FILE* f = fopen(path, "wb");
    if (!f) { free(p); return -1; }
    size_t w = fwrite(p, 1, total, f);
    fclose(f);
    free(p);
    return w == total ? 0 : -1;
}

/* Store SWAP sintetico: come `llm_dsv4_store_open` ma senza GGUF (offset/stride dei layer fissati). */
static void store_init(LlmDsv4Store* s, const char* path, int cap) {
    memset(s, 0, sizeof *s);
    s->mode = LLM_DSV4_STORE_SWAP;
    s->fd = open(path, O_RDONLY);
    s->n_layer = NLY;
    s->n_expert = NEX;
    s->resident_base = NULL;
    s->layers = (LlmDsv4StoreLayer*)calloc(NLY, sizeof(LlmDsv4StoreLayer));
    for (int L = 0; L < NLY; L++) {
        s->layers[L].off_g = part_off(L, 0);
        s->layers[L].off_u = part_off(L, 1);
        s->layers[L].off_d = part_off(L, 2);
        s->layers[L].nb_g = s->layers[L].nb_u = s->layers[L].nb_d = NB;
        s->layers[L].stride_g = s->layers[L].stride_u = s->layers[L].stride_d = NB;
    }
    s->cap = cap > 0 ? cap : 64;
    s->ents = (LlmDsv4StoreEnt*)calloc((size_t)s->cap, sizeof(LlmDsv4StoreEnt));
    int hcap = 16; while (hcap < s->cap * 2) hcap <<= 1;
    s->ht_cap = hcap;
    s->ht_keys = (int*)malloc(sizeof(int) * (size_t)hcap);
    s->ht_vals = (int*)malloc(sizeof(int) * (size_t)hcap);
    for (int i = 0; i < hcap; i++) s->ht_keys[i] = -1;
}

static void store_destroy(LlmDsv4Store* s) {
    if (s->ents) { for (int i = 0; i < s->cap; i++) free(s->ents[i].buf); free(s->ents); }
    free(s->ht_keys); free(s->ht_vals); free(s->layers);
    if (s->fd >= 0) close(s->fd);
}

/* Riproduce il collect-then-use del batch: ritorna il numero di esperti il cui puntatore NON contiene
 * piu' il proprio pattern. */
static int test_case(int cap, int n) {
    char path[512];
    snprintf(path, sizeof path, "/tmp/dsv4_oracle_%d_%d.bin", getpid(), cap);
    if (make_file(path) != 0) { printf("  cap=%-4d union=%-4d [skip: tmp file]\n", cap, n); return -1; }

    LlmDsv4Store s;
    store_init(&s, path, cap);

    int experts[64];
    for (int i = 0; i < n; i++) experts[i] = i % NEX;

    /* 1) prefetch dell'unione (come `llm_dsv4_gpu_moe_batch`: 304) */
    llm_dsv4_store_prefetch(&s, 0, experts, n);

    /* 2) raccolta dei puntatori TUTTI prima dell'uso (come righe 305-321) */
    const uint8_t* gp[64]; const uint8_t* up[64]; const uint8_t* dp[64];
    for (int i = 0; i < n; i++) llm_dsv4_store_expert(&s, 0, experts[i], &gp[i], &up[i], &dp[i]);

    /* 3) verifica: il puntatore i-esimo deve avere il pattern dell'esperto i */
    int bad = 0;
    for (int i = 0; i < n; i++) {
        int e = experts[i];
        if (!gp[i] || !up[i] || !dp[i]) { bad++; continue; }
        if (gp[i][0] != pat(0, 0, e, 0) || up[i][0] != pat(0, 1, e, 0) || dp[i][0] != pat(0, 2, e, 0)) bad++;
    }
    printf("  cap=%-4d union=%-4d -> %-20s (%d/%d esperti con puntatore sbagliato)\n",
           cap, n, bad ? "FAIL (instabile!)" : "PASS", bad, n);

    store_destroy(&s);
    unlink(path);
    return bad;
}

/* Col FIX (sub-chunk di cap/used token): ogni sotto-unione entra nel cap -> puntatori stabili ->
 * atteso PASS per qualunque cap>=used. Riproduce il fix di `llm_dsv4_gpu_moe_batch`. */
static int test_fixed(int cap, int used, int nt) {
    char path[512];
    snprintf(path, sizeof path, "/tmp/dsv4_oracle_fix_%d.bin", getpid());
    if (make_file(path) != 0) { printf("  [skip] tmp file\n"); return -1; }
    LlmDsv4Store s;
    store_init(&s, path, cap);
    int sub = cap / (used > 0 ? used : 1); if (sub < 1) sub = 1;
    int bad = 0, tot = 0;
    for (int t0 = 0; t0 < nt; t0 += sub) {
        int m = nt - t0 < sub ? nt - t0 : sub, u = m * used;
        int ex[256]; for (int i = 0; i < u && i < 256; i++) ex[i] = (t0 * used + i) % NEX;
        llm_dsv4_store_prefetch(&s, 0, ex, u);
        /* COLLECT-THEN-USE come il batch: raccogli TUTTI i puntatori, POI verifica. */
        const uint8_t* gp[256]; const uint8_t* up[256]; const uint8_t* dp[256];
        for (int i = 0; i < u && i < 256; i++) llm_dsv4_store_expert(&s, 0, ex[i], &gp[i], &up[i], &dp[i]);
        for (int i = 0; i < u && i < 256; i++) {
            tot++;
            if (!gp[i] || gp[i][0] != pat(0, 0, ex[i], 0)) bad++;
        }
    }
    printf("  FIX cap=%-4d used=%d nt=%-4d sub=%-4d -> %-10s (%d/%d sbagliati)\n",
           cap, used, nt, sub, bad ? "FAIL" : "PASS", bad, tot);
    store_destroy(&s); unlink(path);
    return bad;
}

int main(void) {
    printf("[dsv4-batch-moe-oracle] stabilita' dei puntatori dello store (MODEL-FREE, no modello/GPU)\n");
    printf("Proprieta' del batch MoE (llm_dsv4_gpu_moe.h:263): \"puntatori stabili: cap >= used*nt\".\n\n");

    printf("-- SENZA fix (comportamento attuale) --\n");
    int bad_big   = test_case(64, 12);   /* cap >= union: atteso PASS */
    int bad_small = test_case(4, 12);    /* cap <  union: se >0 -> proprieta' violata (BUG) */
    printf("\n-- CON fix (sub-chunk cap/used) --\n");
    int bad_fix = test_fixed(10, 6, 16); /* cap(=10)>=used(6): sub-chunk tiene l'unione nel cap -> PASS */
    printf("  (cap<used -> il batch fa fallback per-token: return false, puntatori sempre stabili)\n");

    printf("\nEsito: senza-fix cap>=union %s | senza-fix cap<union %s | con-fix cap<union %s\n",
           bad_big > 0 ? "FAIL" : "PASS",
           bad_small > 0 ? "FAIL (bug)" : "PASS",
           bad_fix > 0 ? "FAIL" : "PASS");
    printf("Atteso: il bug si manifesta a cap<union (senza fix); col fix (sub-chunk) i puntatori\n"
           "restano stabili anche a cap<union.\n");
    return (!bad_big && bad_small > 0 && !bad_fix) ? 0 : 1;
}
