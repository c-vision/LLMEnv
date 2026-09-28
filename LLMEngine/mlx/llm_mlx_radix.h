/* llm_mlx_radix.h — RADIX-TRIE PREFIX CACHE (fix del coding a contesto lungo).
 *
 * Indice content-addressed dei prefissi di token: dato un nuovo prompt, trova in O(prefix_len)
 * il prefisso piu' lungo GIÀ visto (longest-prefix-match). Il riuso del prefill tra turni
 * (coding: cronologia stabile) evita di ricalcolare O(N²) del prefill completo a ogni turno.
 *
 * Questo è il fondamento PURO (trie di token, nessuna dipendenza MLX): mappa token -> nodo.
 * L'integrazione con la KV-cache (ogni nodo mantiene lo stato di prefill riusabile) avviene nel
 * motore via la prefix cache nativa (`llm_ai_mlx_model_save_prefix`/`append_reuse_prefix`), dove
 * il trie sostituisce il match lineare con la ricerca O(prefix_len). Ispirato a
 * `RadixPrefixIndex` di rapidMLX (vllm_mlx/runtime/radix_index.py).
 *
 * Convenzione: puro C, nessuno stato globale.
 */
#ifndef LLM_MLX_RADIX_H
#define LLM_MLX_RADIX_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/* Nodo del trie: una posizione (token) in un prefisso. Ogni nodo può avere un `id` di prefisso
 * (indice nella prefix-cache del motore) se a quel punto il prefisso e' stato salvato. */
typedef struct llm_radix_node {
    int32_t token;
    int depth;                     /* profondita' (= n. token fino a qui) */
    int prefix_id;                 /* id del prefisso salvato a questo nodo, -1 se non salvato */
    struct llm_radix_node* children[64]; /* slot per token -> figlio (table: token%64, collisioni lineari) */
    struct llm_radix_node* next_slot;    /* overflow per collisione di slot */
} llm_radix_node;

typedef struct {
    llm_radix_node* root;
    int n_nodes;
} llm_radix_trie;

static inline int llm_radix_slot(int32_t token) {
    /* slot stabile (token mod 64, con shift per distribuire) */
    return (int)(((uint32_t)token * 2654435761u) >> 26) & 63;
}

static inline void llm_radix_init(llm_radix_trie* t) {
    t->root = (llm_radix_node*)calloc(1, sizeof(llm_radix_node));
    t->root->token = -1;
    t->root->depth = 0;
    t->root->prefix_id = -1;
    t->n_nodes = 1;
}

static void llm_radix_free_node(llm_radix_node* n) {
    if (!n) return;
    for (int i = 0; i < 64; i++) llm_radix_free_node(n->children[i]);
    llm_radix_free_node(n->next_slot);
    free(n);
}

static inline void llm_radix_free(llm_radix_trie* t) {
    if (!t || !t->root) return;
    llm_radix_free_node(t->root);
    t->root = NULL;
}

static llm_radix_node* llm_radix_child(llm_radix_node* n, int32_t token) {
    int s = llm_radix_slot(token);
    llm_radix_node* c = n->children[s];
    while (c) {
        if (c->token == token) return c;
        c = c->next_slot;
    }
    return NULL;
}

static llm_radix_node* llm_radix_child_add(llm_radix_node* n, int32_t token, int depth, int n_nodes) {
    int s = llm_radix_slot(token);
    llm_radix_node* c = n->children[s];
    llm_radix_node* prev = NULL;
    while (c) {
        if (c->token == token) return c;
        prev = c; c = c->next_slot;
    }
    llm_radix_node* nn = (llm_radix_node*)calloc(1, sizeof(llm_radix_node));
    nn->token = token;
    nn->depth = depth;
    nn->prefix_id = -1;
    if (prev) prev->next_slot = nn; else n->children[s] = nn;
    return nn;
}

/* Inserisce una sequenza di token. `prefix_id` applicato all'ultimo nodo (il prefisso salvato).
 * Se `ids` è una lista di prefissi da marcare (opzionale), marca i nodi alle posizioni indicate.
 * Semplicita': segna l'ultimo token della sequenza con `prefix_id` se >= 0. */
static inline void llm_radix_insert(llm_radix_trie* t, const int32_t* toks, int len, int prefix_id) {
    llm_radix_node* cur = t->root;
    for (int i = 0; i < len; i++) {
        cur = llm_radix_child_add(cur, toks[i], cur->depth + 1, t->n_nodes);
        t->n_nodes++;
    }
    if (prefix_id >= 0) cur->prefix_id = prefix_id;
}

/* Longest-prefix-match: cammina il trie finche' i token del prompt combaciano con un percorso.
 * Ritorna la profondita' massima raggiunta, e (opzionale) il prefix_id salvato a quel punto.
 * `out_prefix_id` = -1 se nessun prefisso salvato matcha. */
static inline int llm_radix_longest_match(const llm_radix_trie* t, const int32_t* prompt, int plen,
                                          int* out_prefix_id) {
    llm_radix_node* cur = t->root;
    int depth = 0;
    int matched_prefix = -1;
    for (int i = 0; i < plen; i++) {
        llm_radix_node* c = llm_radix_child(cur, prompt[i]);
        if (!c) break;
        cur = c;
        depth++;
        if (cur->prefix_id >= 0) matched_prefix = cur->prefix_id;
    }
    if (out_prefix_id) *out_prefix_id = matched_prefix;
    return depth;
}

#endif /* LLM_MLX_RADIX_H */
