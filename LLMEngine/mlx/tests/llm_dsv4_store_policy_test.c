/* llm_dsv4_store_policy_test.c — D19-follow-up A: ORACOLO di policy dello store dsv4.
 *
 * Nessun modello, nessun file: fabbrica lo store (ents/cap/freq/ht) e verifica
 *  1) `llm_dsv4_store_evict_choice`: LRU pura (default) vs min-freq (hotness ON);
 *  2) la feed osservata in `llm_dsv4_store_claim` (hit e miss contano come accesso);
 *  3) l'eviction con hotness ON risparmia l'esperto caldo.
 * Deterministico e offline. Compila/run: tools/llm_dsv4_store_policy_test.sh */
#include "llm_dsv4_store.h"

#include <stdio.h>
#include <string.h>

static int fails;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); fails++; } } while (0)

/* Store fabbricato per `claim`: 1 layer con bundle 48B, 2 slot, ht minimale, fd=-1 (nessun I/O). */
static void mk_store(LlmDsv4Store* s, LlmDsv4StoreLayer* lay, LlmDsv4StoreEnt* ents,
                     int* hk, int* hv) {
    memset(s, 0, sizeof(*s));
    memset(lay, 0, sizeof(*lay));
    memset(ents, 0, sizeof(LlmDsv4StoreEnt) * 2);
    for (int i = 0; i < 16; ++i) hk[i] = -1;
    lay->nb_g = 16; lay->nb_u = 16; lay->nb_d = 16;
    s->layers = lay; s->n_layer = 1; s->ents = ents; s->cap = 2;
    s->ht_keys = hk; s->ht_vals = hv; s->ht_cap = 16; s->fd = -1;
    llm_dsv4_store_set_dims(s, 1, 3);
}

static void free_store(LlmDsv4Store* s) {
    for (int i = 0; i < s->cap; ++i) free(s->ents[i].buf);
    free(s->freq);
}

int main(void) {
    /* ── 1. scelta pura: cap=3, slot (0,0) age1, (0,1) age2, (0,2) age3 ── */
    {
        LlmDsv4Store s; memset(&s, 0, sizeof s);
        LlmDsv4StoreEnt ents[3]; memset(ents, 0, sizeof ents);
        for (int i = 0; i < 3; ++i) {
            ents[i].valid = 1; ents[i].layer = 0; ents[i].e = i; ents[i].age = (uint64_t)(i + 1);
        }
        uint32_t freq[3] = { 100, 1, 1 };
        s.ents = ents; s.cap = 3; s.freq = freq; s.f_layers = 1; s.f_experts = 3;

        s.hotness = 0;
        CHECK(llm_dsv4_store_evict_choice(&s) == 0);   /* LRU: il più vecchio */
        s.hotness = 1;
        CHECK(llm_dsv4_store_evict_choice(&s) == 1);   /* freq: evince il freddo (e1, age minore) */
        freq[0] = 0;
        CHECK(llm_dsv4_store_evict_choice(&s) == 0);   /* ora e0 è il minimo */
        freq[0] = freq[1] = freq[2] = 7;
        CHECK(llm_dsv4_store_evict_choice(&s) == 0);   /* freq pari → LRU */
    }

    /* ── 2. feed in claim + eviction hotness (store fabbricato) ── */
    {
        LlmDsv4Store s; LlmDsv4StoreLayer lay; LlmDsv4StoreEnt ents[2]; int hk[16], hv[16];
        mk_store(&s, &lay, ents, hk, hv);
        int miss = 0;
        CHECK(llm_dsv4_store_claim(&s, 0, 0, &miss) == 0 && miss == 1);   /* e0 miss */
        CHECK(llm_dsv4_store_claim(&s, 0, 0, &miss) == 0 && miss == 0);   /* e0 hit */
        CHECK(llm_dsv4_store_freq(&s, 0, 0) == 2);                        /* 2 accessi osservati */
        CHECK(llm_dsv4_store_claim(&s, 0, 1, &miss) == 1 && miss == 1);   /* e1 riempie il 2o slot */
        CHECK(llm_dsv4_store_freq(&s, 0, 1) == 1);
        llm_dsv4_store_set_hotness(&s, 1);
        CHECK(llm_dsv4_store_claim(&s, 0, 2, &miss) == 1 && miss == 1);   /* evince e1 (freq 1), NON e0 (freq 2) */
        CHECK(llm_dsv4_store_claim(&s, 0, 0, &miss) == 0 && miss == 0);   /* e0 (caldo) ancora residente */
        CHECK(s.evictions == 1);
        free_store(&s);
    }

    /* ── 3. default OFF = LRU: stesso scenario, ma evince e0 ── */
    {
        LlmDsv4Store s; LlmDsv4StoreLayer lay; LlmDsv4StoreEnt ents[2]; int hk[16], hv[16];
        mk_store(&s, &lay, ents, hk, hv);
        int miss = 0;
        CHECK(llm_dsv4_store_claim(&s, 0, 0, &miss) == 0 && miss == 1);
        CHECK(llm_dsv4_store_claim(&s, 0, 0, &miss) == 0 && miss == 0);
        CHECK(llm_dsv4_store_claim(&s, 0, 1, &miss) == 1 && miss == 1);
        CHECK(llm_dsv4_store_claim(&s, 0, 2, &miss) == 0 && miss == 1);   /* LRU: evince e0 (slot 0) */
        CHECK(llm_dsv4_store_claim(&s, 0, 0, &miss) == 1 && miss == 1);   /* e0 evitto → miss (ricaricato in slot 1) */
        free_store(&s);
    }

    if (fails == 0) { printf("llm_dsv4_store_policy_test: OK\n"); return 0; }
    printf("llm_dsv4_store_policy_test: %d FAIL\n", fails);
    return 1;
}
