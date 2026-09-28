/* llm_mlx_ngram.h — tabella N-GRAM per self-speculative decoding (P1 del piano).
 *
 * Fonte: `Piani/LLMENV_LLM_MODEL_DECOMPOSITION_AND_PLUGIN_ARCHITECTURE.md` e
 * `AI-LongContext-Modular-Plan.md §3.1/§8.4/§8.10`. Il N-Gram speculative NON richiede
 * un secondo modello: il modello verifica i propri token. Il modulo qui e' la struttura
 * dati pura (nessuna dipendenza MLX) che, dato il contesto, propone il token successivo
 * piu' probabile gia' visto (base per il draft proposto al modello per la verifica).
 *
 * Convenzione (come llm_mlx_ops.h): funzioni `static`, pura, nessuno stato globale.
 * Usa solo C standard (stdlib/string). Testabile in isolamento senza MLX.
 *
 * Semantica:
 *   - ordine `order` = lunghezza dell'n-gram di contesto (es. 3 = trigram).
 *   - La tabella impara da una sequenza di token: per ogni posizione i>=order,
 *     la chiave token[i-order..i-1] ha successore token[i].
 *   - `llm_mlx_ngram_predict` data la coda di contesto ritorna il successore piu'
 *     frequente (tie-break: il piu' recente / primo inserito a parita').
 *
 * NOTA sull'uso (speculative): il guadagno reale si ottiene proponendo MULTIPLI draft
 * token e verificandoli con UN forward batch (il forward del draft produce i logit per
 * il passo successivo). Con 1 solo token nel nostro decode sequenziale il rapporto
 * forward:token resta 1:1 (il guadagno ~+17% citato per Qwen3.6 presuppone draft batch).
 * Questa tabella e' il mattoncino; l'integrazione batch e' il passo successivo.
 */
#ifndef LLM_MLX_NGRAM_H
#define LLM_MLX_NGRAM_H

#include <stdint.h>
#include <stdlib.h>

/* FNV-1a 64-bit per un n-gram (combinazione dei token). */
static inline uint64_t llm_ngram_hash(const int32_t* tok, int order) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < order; i++) {
        h ^= (uint64_t)(uint32_t)tok[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/* Slot: per una chiave (n-gram), il conteggio di un particolare successore. */
typedef struct NGramNode {
    uint64_t key;
    int32_t next_token;
    uint32_t count;
    struct NGramNode* next; /* collisione / piu' successori della stessa chiave */
} NGramNode;

typedef struct {
    NGramNode** buckets; /* hash table: indice = key % nbuckets */
    size_t nbuckets;
    int order;           /* lunghezza n-gram */
} NGramTable;

static inline NGramTable llm_mlx_ngram_new(int order, size_t nbuckets) {
    NGramTable t;
    t.order = order > 0 ? order : 1;
    t.nbuckets = nbuckets > 0 ? nbuckets : 1024;
    t.buckets = (NGramNode**)calloc(t.nbuckets, sizeof(NGramNode*));
    return t;
}

/* Libera tutta la tabella (nodi e bucket). */
static inline void llm_mlx_ngram_free(NGramTable* t) {
    if (!t || !t->buckets) return;
    for (size_t b = 0; b < t->nbuckets; b++) {
        NGramNode* n = t->buckets[b];
        while (n) { NGramNode* nx = n->next; free(n); n = nx; }
    }
    free(t->buckets);
    t->buckets = NULL;
}

/* Aggiorna la tabella da una sequenza di token: per ogni posizione i>=order,
 * la chiave tok[i-order..i-1] ha successore tok[i] (incrementa conteggio).
 * Se `last_suffix` e' una coda di `order` token gia' vista (contesto che precede
 * token_ids), la usa come chiave per il primo token di token_ids. */
static inline void llm_mlx_ngram_add(NGramTable* t, const int32_t* token_ids,
                                     int len, const int32_t* last_suffix /* may be NULL */) {
    if (!t || !t->buckets || len <= 0 || t->order <= 0) return;
    int32_t window[64]; /* finestra scorrevole dell'n-gram corrente (order <= 64) */
    if (t->order > 64) t->order = 64;
    /* inizializza la finestra con last_suffix (contesto precedente) se fornito,
     * altrimenti con i primi token di token_ids. */
    int wlen = 0;
    if (last_suffix) {
        for (int i = 0; i < t->order && i < 64; i++) window[wlen++] = last_suffix[i];
    }
    for (int i = 0; i < len; i++) {
        if (wlen < t->order) {
            window[wlen++] = token_ids[i];
            continue; /* non ancora un n-gram completo */
        }
        /* la finestra attuale (t->order token) e' la chiave; successore = token_ids[i] */
        uint64_t key = llm_ngram_hash(window, t->order);
        size_t b = (size_t)(key % t->nbuckets);
        NGramNode* n = t->buckets[b];
        while (n && (n->key != key || n->next_token != token_ids[i])) n = n->next;
        if (n) {
            n->count++;
        } else {
            NGramNode* nn = (NGramNode*)malloc(sizeof(NGramNode));
            if (!nn) return;
            nn->key = key; nn->next_token = token_ids[i]; nn->count = 1;
            nn->next = t->buckets[b];
            t->buckets[b] = nn;
        }
        /* scorre la finestra: esce il piu' vecchio, entra token_ids[i] */
        for (int j = 0; j < t->order - 1; j++) window[j] = window[j + 1];
        window[t->order - 1] = token_ids[i];
    }
}

/* Prevede il successore piu' probabile per il contesto dato (gli ultimi `t->order`
 * token in `ctx`). Ritorna il token con conteggio massimo, oppure -1 se sconosciuto. */
static inline int32_t llm_mlx_ngram_predict(NGramTable* t, const int32_t* ctx, int ctx_len) {
    if (!t || !t->buckets || ctx_len < t->order) return -1;
    /* gli ultimi order token */
    const int32_t* last = ctx + (ctx_len - t->order);
    uint64_t key = llm_ngram_hash(last, t->order);
    size_t b = (size_t)(key % t->nbuckets);
    NGramNode* n = t->buckets[b];
    NGramNode* best = NULL;
    while (n) {
        if (n->key == key) {
            if (!best || n->count > best->count) best = n;
        }
        n = n->next;
    }
    return best ? best->next_token : -1;
}

#endif /* LLM_MLX_NGRAM_H */
