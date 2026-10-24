/* llm_mlx_xs_cache.h — cache HOT/LRU degli esperti streamati (path MLX safetensors).
 *
 * Perche': il path streamed rilegge i byte di un esperto a OGNI token (memcpy da mmap);
 * il routing MoE e' temporalmente stabile (~48% di riuso tra token adiacenti misurato su
 * Qwen3.6-35B-A3B, coerente con 13,4% esperti -> 66% richieste del vault). Con modelli che
 * superano il wired-limit le pagine sorgente non restano in page cache -> ogni token paga
 * ~720 MB di letture. Qui si tiene un set LIMITATO di slice esperto in RAM (arena propria,
 * budget in byte) con politica LRU 4-way: i "hot" restano, i freddi si rileggono.
 *
 * Parametro: opzione registro `moe_expert_cache_mb` (default 2048; 0 = OFF). Nessun env.
 * Pura libc; single-thread (il forward MoE e' single-thread, come g_tq_full/xs_pool).
 * Testabile senza MLX: tests/test_xs_cache.c */
#ifndef LLM_MLX_XS_CACHE_H
#define LLM_MLX_XS_CACHE_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint64_t key; uint64_t used; uint8_t* buf; uint32_t len; } LlmXsCacheEnt;

static struct {
    int       init;
    size_t    budget;
    size_t    bytes;
    uint32_t  nsets;      /* n. set (4 vie ciascuno) */
    uint32_t  hand;       /* mano per l'eviction globale quando si supera il budget */
    uint64_t  tick;
    uint64_t  hits, misses;
    LlmXsCacheEnt* tab;   /* nsets*4 */
} g_xsc;

int llm_ai_mlx_opt_get_int(const char* key, int* out);   /* registro opzioni del motore */

static inline uint64_t llm_xsc_hash(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
static inline uint64_t llm_xsc_key(int layer, int proj, int kind, int e) {
    return ((uint64_t)(uint32_t)layer << 40) | ((uint64_t)(uint32_t)proj << 36) |
           ((uint64_t)(uint32_t)kind << 32) | (uint32_t)e;
}
static inline void llm_xsc_free_ent(LlmXsCacheEnt* e) {
    if (e->buf) { free(e->buf); e->buf = NULL; }
    g_xsc.bytes -= e->len; e->len = 0; e->key = 0;
}
static inline void llm_xsc_evict_hand(void) {
    if (!g_xsc.tab || g_xsc.nsets == 0) return;
    const uint32_t n = g_xsc.nsets * 4u;
    for (uint32_t i = 0; i < n; i++) {
        LlmXsCacheEnt* e = &g_xsc.tab[g_xsc.hand];
        g_xsc.hand = (g_xsc.hand + 1u) % n;
        if (e->buf) { llm_xsc_free_ent(e); return; }
    }
}
static inline void llm_xsc_reset(void) {
    if (g_xsc.tab) {
        const uint32_t n = g_xsc.nsets * 4u;
        for (uint32_t i = 0; i < n; i++) if (g_xsc.tab[i].buf) llm_xsc_free_ent(&g_xsc.tab[i]);
        free(g_xsc.tab);
    }
    memset(&g_xsc, 0, sizeof g_xsc);
}
static inline void llm_xsc_init(void) {
    if (g_xsc.init) return;
    g_xsc.init = 1;
    int mb = 2048;
    if (llm_ai_mlx_opt_get_int("moe_expert_cache_mb", &mb) == 0 && mb < 0) mb = 0;
    if (mb <= 0) { g_xsc.budget = 0; return; }
    g_xsc.budget = (size_t)mb << 20;
    uint32_t nsets = (uint32_t)(g_xsc.budget / (256u * 1024u * 4u)); /* ~256KB per entry */
    if (nsets < 16u) nsets = 16u;
    if (nsets > (1u << 18)) nsets = 1u << 18;
    g_xsc.nsets = nsets;
    g_xsc.tab = (LlmXsCacheEnt*)calloc((size_t)nsets * 4u, sizeof(LlmXsCacheEnt));
    if (!g_xsc.tab) { g_xsc.nsets = 0; g_xsc.budget = 0; }
}
/* hit: copia in dst e ritorna 1; miss: 0. */
static inline int llm_xsc_get(uint64_t key, uint32_t len, uint8_t* dst) {
    if (!g_xsc.init) llm_xsc_init();
    if (!g_xsc.tab) { g_xsc.misses++; return 0; }
    const size_t ns = g_xsc.nsets;
    const size_t s = (size_t)(llm_xsc_hash(key) % ns) * 4u;
    for (size_t i = 0; i < 4u; i++) {
        LlmXsCacheEnt* e = &g_xsc.tab[s + i];
        if (e->buf && e->key == key && e->len == len) {
            memcpy(dst, e->buf, len);
            e->used = ++g_xsc.tick;
            g_xsc.hits++;
            return 1;
        }
    }
    g_xsc.misses++;
    return 0;
}
/* inserisce (copia len byte). Se il budget e' superato evince con la mano. */
static inline void llm_xsc_put(uint64_t key, const uint8_t* src, uint32_t len) {
    if (!g_xsc.init) llm_xsc_init();
    if (!g_xsc.tab || len == 0) return;
    if (len > g_xsc.budget) return;
    while (g_xsc.bytes + len > g_xsc.budget) llm_xsc_evict_hand();
    const size_t ns = g_xsc.nsets;
    const size_t s = (size_t)(llm_xsc_hash(key) % ns) * 4u;
    LlmXsCacheEnt* victim = NULL;
    for (size_t i = 0; i < 4u; i++) {
        LlmXsCacheEnt* e = &g_xsc.tab[s + i];
        if (!e->buf) { victim = e; break; }
        if (e->key == key) { /* aggiorna in place */
            if (e->len == len) { memcpy(e->buf, src, len); e->used = ++g_xsc.tick; return; }
            llm_xsc_free_ent(e); victim = e; break;
        }
        if (!victim || e->used < victim->used) victim = e;
    }
    if (!victim) return;
    if (victim->buf) llm_xsc_free_ent(victim);
    uint8_t* b = (uint8_t*)malloc(len);
    if (!b) return;
    memcpy(b, src, len);
    victim->buf = b; victim->len = len; victim->key = key; victim->used = ++g_xsc.tick;
    g_xsc.bytes += len;
}
#endif /* LLM_MLX_XS_CACHE_H */
