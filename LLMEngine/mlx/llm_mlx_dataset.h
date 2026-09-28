/* llm_mlx_dataset.h — §D R4.4: dataset di fine-tune da JSONL (messages + prompt/completion),
 * con MASK (loss solo sulla completion) e BUCKETING/PACKING per l'efficienza dei batch.
 *
 * PURO (nessuna dipendenza MLX): la tokenizzazione e' un CALLBACK (il chiamante usa il tokenizer
 * del modello). Formato input: una riga JSON per esempio, con
 *   {"prompt":"...","completion":"..."}                        (prompt-completion)
 *   {"messages":[{"role":"system","content":"..."},{"role":"user",...},{"role":"assistant",...}]}
 * Per `messages`: loss SOLO sui turni assistant (mask=1), come SFT standard.
 *
 * ids[i] predice targets[i]=ids[i+1]; mask[i]=1 se la predizione riguarda un token di completion.
 * Bucketing: gli indici sono raggruppati in bucket di lunghezza (riduce il padding).
 * Packing: piu' campioni corti sono concatenati in blocchi di lunghezza fissa `seq_len` (con EOS
 * tra i campioni), con mask per-token e azzeramento dei target cross-campione. */
#ifndef LLM_MLX_DATASET_H
#define LLM_MLX_DATASET_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int32_t* ids; int32_t* tgt; uint8_t* mask; int n; } GdsSample;
typedef struct { GdsSample* s; int n; } Gds;
typedef int (*GdsTokFn)(void* ctx, const char* text, int32_t* out, int cap);

static inline void gds_free(Gds* d) {
    if (!d) return;
    for (int i = 0; i < d->n; i++) { free(d->s[i].ids); free(d->s[i].tgt); free(d->s[i].mask); }
    free(d->s); d->s = NULL; d->n = 0;
}

/* Estrae il valore stringa di `key` da `p` (JSON minimale: gestisce \" \\ \n \t). Ritorna il
 * puntatore dopo il valore o NULL. out=NULL -> solo skip. */
static inline const char* gds_json_str(const char* p, const char* key, char* out, int cap) {
    char pat[64]; snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char* q = strstr(p, pat);
    if (!q) return NULL;
    q = strchr(q + strlen(pat), ':'); if (!q) return NULL;
    q++; while (*q == ' ' || *q == '\t') q++;
    if (*q != '"') return NULL;
    q++;
    int k = 0;
    while (*q && *q != '"') {
        if (*q == '\\' && q[1]) { q++; char c = *q; if (out && k < cap - 1) out[k++] = (c == 'n') ? '\n' : (c == 't' ? '\t' : c); }
        else { if (out && k < cap - 1) out[k++] = *q; }
        q++;
    }
    if (out) out[k] = 0;
    return (*q == '"') ? q + 1 : NULL;
}

/* Costruisce un campione da prompt+completion: ids = tok(prompt+completion), mask=1 solo sui
 * token di completion (posizioni i con i+1 >= ntok_prompt). 0 ok / -1. */
static inline int gds_make_sample(GdsTokFn tok, void* ctx, const char* prompt, const char* completion, GdsSample* out) {
    char buf[1 << 16];
    snprintf(buf, sizeof(buf), "%s%s", prompt ? prompt : "", completion ? completion : "");
    int32_t* tmp = (int32_t*)malloc(sizeof(int32_t) * (1 << 16));
    int np = 0; if (prompt && prompt[0]) np = tok(ctx, prompt, tmp, 1 << 15);
    if (np < 0) { free(tmp); return -1; }
    int n = tok(ctx, buf, tmp, 1 << 16);
    if (n <= 1) { free(tmp); return -1; }
    out->ids = (int32_t*)malloc((size_t)n * sizeof(int32_t));
    out->tgt = (int32_t*)malloc((size_t)n * sizeof(int32_t));
    out->mask = (uint8_t*)calloc((size_t)n, 1);
    if (!out->ids || !out->tgt || !out->mask) { free(tmp); free(out->ids); free(out->tgt); free(out->mask); return -1; }
    for (int i = 0; i < n; i++) { out->ids[i] = tmp[i]; out->tgt[i] = (i + 1 < n) ? tmp[i + 1] : tmp[i]; }
    for (int i = 0; i < n; i++) out->mask[i] = (i + 1 >= np) ? 1 : 0;   /* predice un token di completion */
    out->n = n;
    free(tmp);
    return 0;
}

/* Carica un JSONL. `max_ctx` = lunghezza massima (campioni piu' lunghi troncati). 0 ok. */
static inline int gds_load_jsonl(const char* path, GdsTokFn tok, void* ctx, int max_ctx, Gds* out) {
    out->s = NULL; out->n = 0;
    FILE* f = fopen(path, "r"); if (!f) return -1;
    char line[1 << 18];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '\n' || line[0] == 0) continue;
        char prompt[1 << 15] = "", completion[1 << 15] = "";
        const char* p = line;
        if (gds_json_str(p, "prompt", prompt, sizeof(prompt))) {
            gds_json_str(p, "completion", completion, sizeof(completion));
        } else {
            /* messages: prompt = concat dei turni NON-assistant, completion = concat degli
             * assistant (loss solo qui). Approssimazione deterministica per SFT. */
            char role[64] = "", content[1 << 15];
            const char* q = line;
            while ((q = gds_json_str(q, "role", role, sizeof(role))) != NULL) {
                if (!gds_json_str(q, "content", content, sizeof(content))) break;
                if (strstr(role, "assistant") != NULL) strncat(completion, content, sizeof(completion) - 1 - strlen(completion));
                else strncat(prompt, content, sizeof(prompt) - 1 - strlen(prompt));
                q = strstr(q, "\"role\"");
                if (!q) break;
            }
        }
        if (!prompt[0] && !completion[0]) continue;
        GdsSample sm;
        if (gds_make_sample(tok, ctx, prompt, completion, &sm) != 0) continue;
        if (sm.n > max_ctx) { sm.n = max_ctx; }   /* tronca */
        out->s = (GdsSample*)realloc(out->s, (size_t)(out->n + 1) * sizeof(GdsSample));
        out->s[out->n++] = sm;
    }
    fclose(f);
    return 0;
}

/* Bucketing: ordina gli indici in bucket di ampiezza `bucket` (per lunghezza). */
static inline void gds_bucket_order(const Gds* d, int bucket, int* order) {
    if (bucket <= 0) bucket = 1;
    int n = d->n;
    for (int i = 0; i < n; i++) order[i] = i;
    /* insertion sort per (bucket(len), len) */
    for (int i = 1; i < n; i++) {
        int k = order[i], L = d->s[k].n / bucket;
        int j = i - 1;
        while (j >= 0 && (d->s[order[j]].n / bucket) > L) { order[j + 1] = order[j]; j--; }
        order[j + 1] = k;
    }
}

/* Packing: concatena i campioni (nell'ordine `order`) in blocchi di `seq_len`. Scrive in
 * out_ids/out_tgt/out_mask (max_blocks*seq_len) e setta *out_nblocks. Ritorna il numero di
 * token effettivamente impacchettati. */
static inline int gds_pack(const Gds* d, const int* order, int seq_len, int eos_id,
                           int32_t* out_ids, int32_t* out_tgt, uint8_t* out_mask, int max_blocks, int* out_nblocks) {
    int blk = 0, pos = 0;
    for (int i = 0; i < d->n && blk < max_blocks; i++) {
        const GdsSample* s = &d->s[order[i]];
        int j = 0;
        while (j < s->n && blk < max_blocks) {
            int room = seq_len - pos;
            int take = s->n - j; if (take > room) take = room;
            for (int k = 0; k < take; k++) {
                out_ids[blk * seq_len + pos + k] = s->ids[j + k];
                out_tgt[blk * seq_len + pos + k] = s->tgt[j + k];
                out_mask[blk * seq_len + pos + k] = (j + k + 1 < s->n) ? s->mask[j + k] : 1; /* ultimo: predice EOS */
            }
            pos += take; j += take;
            if (pos == seq_len) { blk++; pos = 0; }
        }
        /* EOS tra campioni (se c'e' spazio) */
        if (pos > 0 && j >= s->n && blk < max_blocks) {
            out_ids[blk * seq_len + pos] = eos_id; out_tgt[blk * seq_len + pos] = eos_id; out_mask[blk * seq_len + pos] = 1;
            pos++; if (pos == seq_len) { blk++; pos = 0; }
        }
    }
    if (pos > 0) blk++;   /* ultimo blocco parziale (maskato) */
    *out_nblocks = blk;
    return blk * seq_len;
}

#endif /* LLM_MLX_DATASET_H */
