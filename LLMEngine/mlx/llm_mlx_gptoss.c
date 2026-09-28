/* ============================================================================
 * llm_mlx_gptoss.c — dominio GPT-OSS: memo nome->array per-layer dei pesi, pre-broadcast
 * dei bias per il decode e getter (sfx/sub/triple). Estratto VERBATIM da llmengine_mlx.c.
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Memo nome->array per-layer (gpt-oss). true se il tensore esiste. I MISS sono memoizzati
 * (handle vuoto): dopo il load i pesi non cambiano. */
/* Memo (suffisso, sub) -> array per-layer. Il suffisso e' un LETTERALE: chiave = puntatore
 * (confronto O(1), nessuna stringa) e sul hit si salta ANCHE il tname (snprintf), che il
 * memo-per-nome non evitava. Sul miss costruisce il nome col tname e delega a multishard_get.
 * I MISS sono memoizzati (handle vuoto): dopo il load i pesi non cambiano. */
static bool llm_mlx_gptoss_memo_lookup(LlmMlxModel* m, int layer_idx, const char* sfx, int sub, mlx_array* out) {
    GptossLayerWE* c = &m->gptoss_we[layer_idx];
    for (int i = 0; i < c->n; i++) {
        if (c->e[i].sfx == sfx && c->e[i].sub == sub) {
            mlx_array_free(*out);
            *out = mlx_array_new();
            if (c->e[i].a.ctx == NULL) return false;
            return mlx_array_set(out, c->e[i].a) == 0;
        }
    }
    return false;   /* non memoizzato */
}

static void llm_mlx_gptoss_memo_store(LlmMlxModel* m, int layer_idx, const char* sfx, int sub, mlx_array a) {
    GptossLayerWE* c = &m->gptoss_we[layer_idx];
    if (c->n == c->cap) {
        int ncap = c->cap == 0 ? 32 : c->cap * 2;
        GptossWE* ne = (GptossWE*)realloc(c->e, (size_t)ncap * sizeof(GptossWE));
        if (ne == NULL) return;
        c->e = ne; c->cap = ncap;
    }
    c->e[c->n].sfx = sfx;
    c->e[c->n].sub = sub;
    c->e[c->n].a = mlx_array_new();
    if (a.ctx != NULL) mlx_array_set(&c->e[c->n].a, a);
    c->n++;
}

static bool llm_mlx_gptoss_ensure(LlmMlxModel* m, int layer_idx) {
    if (m->gptoss_we == NULL) {
        int n = m->cfg.n_layers;
        if (n <= 0 || layer_idx >= n) return false;
        m->gptoss_we = (GptossLayerWE*)calloc((size_t)n, sizeof(GptossLayerWE));
        if (m->gptoss_we == NULL) return false;
        m->gptoss_we_n = n;
    }
    return layer_idx >= 0 && layer_idx < m->gptoss_we_n;
}

/* Pre-broadcast di un bias COSTANTE a [1,1,dim] per il passo di DECODE: l'add bias<->[1,1,dim]
 * emette un nodo Broadcast per token; calcolandolo UNA volta (memo per (layer, slot)) resta il
 * solo Add. La chiave e' lo SLOT esplicito, non .ctx: mlx_array_set COPIA l'oggetto array (il
 * puntatore .ctx cambia a ogni chiamata -> una memo su .ctx non colpisce mai: hits=0/600 misurato).
 * Solo n_tokens==1 (prefill invariato). LLM_GPTOSS_BIASPREP=0 -> disattiva (A/B). */
bool llm_mlx_gptoss_bias_prep(LlmMlxModel* m, int layer_idx, int slot, mlx_array src, int dim, mlx_array* out) {
    mlx_array_free(*out); *out = mlx_array_new();
    static int off = -1;
    if (off < 0) { const char* e = getenv("LLM_GPTOSS_BIASPREP"); off = (e != NULL && strcmp(e, "0") == 0) ? 1 : 0; }
    if (off || m == NULL || layer_idx < 0 || layer_idx >= 64 || slot < 0 || slot > 5 ||
        mlx_array_size(src) != (size_t)dim)
        return mlx_array_set(out, src) == 0;
    static int c_dim[64][6]; static signed char c_ok[64][6]; static mlx_array c_arr[64][6];
    if (c_ok[layer_idx][slot] > 0 && c_dim[layer_idx][slot] == dim)
        return mlx_array_set(out, c_arr[layer_idx][slot]) == 0;
    const int shape[3] = {1, 1, dim};
    mlx_array tmp = mlx_array_new();
    if (mlx_broadcast_to(&tmp, src, shape, 3, m->stream) != 0) {
        mlx_array_free(tmp); c_ok[layer_idx][slot] = -1;
        return mlx_array_set(out, src) == 0;
    }
    if (c_arr[layer_idx][slot].ctx != NULL) mlx_array_free(c_arr[layer_idx][slot]);
    c_arr[layer_idx][slot] = mlx_array_new();
    mlx_array_set(&c_arr[layer_idx][slot], tmp);
    c_dim[layer_idx][slot] = dim; c_ok[layer_idx][slot] = 1;
    bool rok = mlx_array_set(out, tmp) == 0;
    mlx_array_free(tmp);
    return rok;
}

bool llm_mlx_gptoss_get_sfx(LlmMlxModel* m, int layer_idx, const char* sfx, mlx_array* out) {
    static int off = -1;
    if (off < 0) { const char* e = getenv("LLM_GPTOSS_WCACHE"); off = (e != NULL && atoi(e) == 0) ? 1 : 0; }
    char nb[256];
    mlx_array_free(*out); *out = mlx_array_new();
    if (off || m == NULL || !llm_mlx_wcache_enabled() || !llm_mlx_gptoss_ensure(m, layer_idx)) {
        if (m == NULL || layer_idx < 0) return false;
        tname(m->cfg.layer_prefix, layer_idx, sfx, nb, sizeof(nb));
        return multishard_get(&m->weights, nb, out);
    }
    if (llm_mlx_gptoss_memo_lookup(m, layer_idx, sfx, -1, out)) return out->ctx != NULL;
    tname(m->cfg.layer_prefix, layer_idx, sfx, nb, sizeof(nb));
    mlx_array tmp = mlx_array_new();
    bool ok = multishard_get(&m->weights, nb, &tmp);
    llm_mlx_gptoss_memo_store(m, layer_idx, sfx, -1, tmp);
    mlx_array_free(*out);
    *out = mlx_array_new();
    bool rok = ok && mlx_array_set(out, tmp) == 0;
    mlx_array_free(tmp);
    return rok;
}

/* sub 0/1/2 = .weight/.scales/.biases (semantica di multishard_get_triple: obbligatorio solo il
 * weight; scales/biases possono mancare -> il chiamante usa set_optional/quantized_linear_cfg). */
static bool llm_mlx_gptoss_get_sub(LlmMlxModel* m, int layer_idx, const char* sfx, int sub, mlx_array* out) {
    char nm[256], nb[256];
    mlx_array_free(*out); *out = mlx_array_new();
    size_t bl = strlen(sfx);
    if (bl + 8 > sizeof(nm)) return false;
    memcpy(nm, sfx, bl);
    memcpy(nm + bl, sub == 0 ? ".weight" : (sub == 1 ? ".scales" : ".biases"), 8);
    static int off = -1;
    if (off < 0) { const char* e = getenv("LLM_GPTOSS_WCACHE"); off = (e != NULL && atoi(e) == 0) ? 1 : 0; }
    if (off || m == NULL || !llm_mlx_wcache_enabled() || layer_idx < 0 || !llm_mlx_gptoss_ensure(m, layer_idx)) {
        if (m == NULL || layer_idx < 0) return false;
        tname(m->cfg.layer_prefix, layer_idx, nm, nb, sizeof(nb));
        return multishard_get(&m->weights, nb, out);
    }
    if (llm_mlx_gptoss_memo_lookup(m, layer_idx, sfx, sub, out)) return out->ctx != NULL;
    tname(m->cfg.layer_prefix, layer_idx, nm, nb, sizeof(nb));
    mlx_array tmp = mlx_array_new();
    bool ok = multishard_get(&m->weights, nb, &tmp);
    llm_mlx_gptoss_memo_store(m, layer_idx, sfx, sub, tmp);
    mlx_array_free(*out);
    *out = mlx_array_new();
    bool rok = ok && mlx_array_set(out, tmp) == 0;
    mlx_array_free(tmp);
    return rok;
}

bool llm_mlx_gptoss_get_triple_sfx(LlmMlxModel* m, int layer_idx, const char* sfx,
                                          mlx_array* w, mlx_array* s, mlx_array* b) {
    bool okw = llm_mlx_gptoss_get_sub(m, layer_idx, sfx, 0, w);
    if (!okw) return false;
    (void)llm_mlx_gptoss_get_sub(m, layer_idx, sfx, 1, s);
    (void)llm_mlx_gptoss_get_sub(m, layer_idx, sfx, 2, b);
    return true;
}
