/* llm_mlx_kvpagesel.h — SPARSE PAGE-LEVEL ATTENTION: selezione delle pagine (§G-bis, leva 1).
 *
 * Problema: il decode legge O(pos) KV ad ogni token; a contesto molto lungo (100K) il vincolo
 * e' la bandwidth. Soluzione (UNIQUE/RocketKV two-stage): stimare l'importanza per PAGINA — non
 * token-per-token — usando un SUMMARY economico per pagina (la media delle key della pagina,
 * mantenuto INCREMENTALMENTE, senza rileggere tutto il K), poi attendere solo sulle top-k pagine.
 * La pagina e' l'unita' del KvPagePool esistente.
 *
 * Questo modulo e' PURO (nessuna dipendenza MLX, come kvpage/kvdisk): mantiene i summary
 * (somma + conteggio per pagina/kv-head), calcola lo score query-vs-pagina e seleziona le pagine.
 * L'integrazione col kernel di attenzione (gather delle pagine + SDPA) e' il passo successivo.
 *
 * Scoring (proxy two-stage, stile RocketKV/Quest):
 *   mean[p,h] = sum[p,h]/count[p];  score[p] = max_h  <q[h], mean[p, map(h)]>
 * (map(h) = kv head di h in GQA). Le pagine RECENTI (sliding) e la pagina SINK (la prima) sono
 * sempre incluse: l'attenzione locale e i "sink" non sono selezionabili per score.
 */
#ifndef LLM_MLX_KVPAGESEL_H
#define LLM_MLX_KVPAGESEL_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int nkv;        /* kv heads */
    int hd;         /* head dim */
    int page;       /* token per pagina */
    int n_pages;    /* capacita' (numero massimo di pagine) */
    float* sum;     /* [n_pages * nkv * hd] somma delle key */
    int*   count;   /* [n_pages] token accumulati */
    int    used;    /* pagine toccate (<= n_pages) */
} KvPageSum;

static inline KvPageSum llm_kvps_new(int nkv, int hd, int page, int max_pages) {
    KvPageSum s; memset(&s, 0, sizeof(s));
    s.nkv = nkv > 0 ? nkv : 1;
    s.hd = hd > 0 ? hd : 1;
    s.page = page > 0 ? page : 64;
    s.n_pages = max_pages > 0 ? max_pages : 1024;
    s.sum = (float*)calloc((size_t)s.n_pages * s.nkv * s.hd, sizeof(float));
    s.count = (int*)calloc((size_t)s.n_pages, sizeof(int));
    return s;
}
static inline void llm_kvps_free(KvPageSum* s) {
    if (!s) return;
    free(s->sum); free(s->count); s->sum = NULL; s->count = NULL;
}

/* Accumula i summary di `nt` key (layout [t][nkv][hd]) nella pagina `page`. Ritorna 0/-1. */
static inline int llm_kvps_add(KvPageSum* s, int page, const float* k, int nt) {
    if (!s || !k || page < 0 || page >= s->n_pages || nt < 0) return -1;
    if (nt > s->page - s->count[page]) nt = s->page - s->count[page]; /* non oltre la pagina */
    float* base = s->sum + (size_t)page * s->nkv * s->hd;
    for (int t = 0; t < nt; t++) {
        const float* kt = k + (size_t)t * s->nkv * s->hd;
        for (int i = 0; i < s->nkv * s->hd; i++) base[i] += kt[i];
    }
    s->count[page] += nt;
    if (page + 1 > s->used) s->used = page + 1;
    return 0;
}

/* score[p] = max_h <q[h], mean[p, map(h)]> (mean = sum/count; pagine vuote -> score 0).
 * q: [nqh][hd]. Ritorna 0/-1. out_scores: [n_pages]. */
static inline int llm_kvps_score(const KvPageSum* s, const float* q, int nqh, float* out_scores) {
    if (!s || !q || !out_scores || nqh <= 0) return -1;
    const int g = (nqh + s->nkv - 1) / s->nkv;   /* query heads per kv head */
    for (int p = 0; p < s->n_pages; p++) {
        if (s->count[p] <= 0) { out_scores[p] = -1e30f; continue; }
        const float inv = 1.0f / (float)s->count[p];
        const float* base = s->sum + (size_t)p * s->nkv * s->hd;
        float best = -1e30f;
        for (int h = 0; h < nqh; h++) {
            int kh = h / g; if (kh >= s->nkv) kh = s->nkv - 1;
            const float* mean = base + (size_t)kh * s->hd;
            const float* qh = q + (size_t)h * s->hd;
            float dot = 0.0f;
            for (int d = 0; d < s->hd; d++) dot += qh[d] * (mean[d] * inv);
            if (dot > best) best = dot;
        }
        out_scores[p] = best;
    }
    return 0;
}

/* Seleziona le pagine da attendere: le top-k per score + SEMPRE le ultime `window` pagine
 * (recent, sliding) e la pagina 0 (sink). Scrive indici UNIVOCI ORDINATI CRESCENTI in out
 * (capienza out_cap), ritorna il conteggio (<= out_cap). topk<=0 -> solo recent+sink. */
static inline int llm_kvps_select(const float* scores, int n_pages, int topk, int window,
                                  int* out, int out_cap) {
    if (!scores || !out || n_pages <= 0 || out_cap <= 0) return 0;
    unsigned char* keep = (unsigned char*)calloc((size_t)n_pages, 1);
    if (!keep) return 0;
    /* top-k per score (selezione parziale semplice: O(n*k), k piccolo) */
    for (int i = 0; i < topk; i++) {
        int best = -1; float bv = -1e30f;
        for (int p = 0; p < n_pages; p++) if (!keep[p] && scores[p] > bv) { bv = scores[p]; best = p; }
        if (best < 0) break;
        keep[best] = 1;
    }
    /* recent + sink */
    keep[0] = 1;
    for (int p = n_pages - 1; p >= 0 && p >= n_pages - window; p--) keep[p] = 1;
    int n = 0;
    for (int p = 0; p < n_pages && n < out_cap; p++) if (keep[p]) out[n++] = p;
    free(keep);
    return n;
}

#endif /* LLM_MLX_KVPAGESEL_H */
