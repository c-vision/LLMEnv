/* llm_dsv41_numpy_rng.h — port C esatto di `np.random.default_rng(seed).integers(0, high, n,
 * dtype=int64)` (NumPy 2.x): SeedSequence (bit_generator.pyx) + PCG64 XSL-RR-128/64 (pcg64.h) +
 * Lemire bounded (distributions.c). Serve a costruire i multipliers Engram a load-time senza
 * Python. C puro, host-only, nessun device.
 *
 * Verifica: `llm_dsv41_numpy_rng_test.c` vs `llm_dsv41_numpy_rng_ref.h` (gen col NumPy reale). */
#ifndef LLM_DSV41_NUMPY_RNG_H
#define LLM_DSV41_NUMPY_RNG_H

#include <stdint.h>

typedef unsigned __int128 llm_rng_u128;

/* ── SeedSequence ─────────────────────────────────────────────────────────── */
#define LLM_RNG_INIT_A 0x43b0d7e5u
#define LLM_RNG_MULT_A 0x931e8875u
#define LLM_RNG_INIT_B 0x8b51f9ddu
#define LLM_RNG_MULT_B 0x58f38dedu
#define LLM_RNG_MIX_L  0xca01f9ddu
#define LLM_RNG_MIX_R  0x4973f715u
#define LLM_RNG_XSHIFT 16u
#define LLM_RNG_POOL   4

static inline uint32_t llm_rng_hashmix(uint32_t value, uint32_t* hc) {
    value ^= *hc;
    *hc *= LLM_RNG_MULT_A;
    value *= *hc;
    value ^= value >> LLM_RNG_XSHIFT;
    return value;
}
static inline uint32_t llm_rng_mix(uint32_t x, uint32_t y) {
    uint32_t r = LLM_RNG_MIX_L * x - LLM_RNG_MIX_R * y;
    r ^= r >> LLM_RNG_XSHIFT;
    return r;
}
/* entropy (uint32, low-first) -> pool[4] (mix_entropy) */
static inline void llm_rng_seedseq_pool(const uint32_t* ent, int n, uint32_t pool[LLM_RNG_POOL]) {
    uint32_t hc = LLM_RNG_INIT_A;
    for (int i = 0; i < LLM_RNG_POOL; i++) pool[i] = llm_rng_hashmix(i < n ? ent[i] : 0u, &hc);
    for (int s = 0; s < LLM_RNG_POOL; s++)
        for (int d = 0; d < LLM_RNG_POOL; d++)
            if (s != d) pool[d] = llm_rng_mix(pool[d], llm_rng_hashmix(pool[s], &hc));
    for (int s = LLM_RNG_POOL; s < n; s++)
        for (int d = 0; d < LLM_RNG_POOL; d++) pool[d] = llm_rng_mix(pool[d], llm_rng_hashmix(ent[s], &hc));
}
/* generate_state(4, uint64): 8 uint32 -> 4 uint64 little-endian */
static inline void llm_rng_generate_state64(const uint32_t pool[LLM_RNG_POOL], uint64_t out[4]) {
    uint32_t hc = LLM_RNG_INIT_B, s32[8];
    for (int i = 0; i < 8; i++) {
        uint32_t d = pool[i % LLM_RNG_POOL];
        d ^= hc; hc *= LLM_RNG_MULT_B; d *= hc; d ^= d >> LLM_RNG_XSHIFT;
        s32[i] = d;
    }
    for (int k = 0; k < 4; k++) out[k] = (uint64_t)s32[2 * k] | ((uint64_t)s32[2 * k + 1] << 32);
}

/* ── PCG64 XSL-RR 128/64 ──────────────────────────────────────────────────── */
typedef struct { llm_rng_u128 state, inc; uint32_t uinteger; int has_uint32; } llm_rng_pcg64;
#define LLM_PCG_MULT_HIGH 2549297995355413924ULL
#define LLM_PCG_MULT_LOW  4865540595714422341ULL
static inline llm_rng_u128 llm_rng_u128c(uint64_t hi, uint64_t lo) { return ((llm_rng_u128)hi << 64) | lo; }
static inline uint64_t llm_rng_rotr64(uint64_t v, unsigned r) { return (v >> r) | (v << ((-r) & 63)); }
static inline uint64_t llm_rng_pcg_output(llm_rng_u128 s) {
    uint64_t hi = (uint64_t)(s >> 64), lo = (uint64_t)s;
    return llm_rng_rotr64(hi ^ lo, (unsigned)(hi >> 58));
}
static inline void llm_rng_pcg_set_seed(llm_rng_pcg64* g, uint64_t sh, uint64_t sl, uint64_t ih, uint64_t il) {
    llm_rng_u128 initstate = llm_rng_u128c(sh, sl), initseq = llm_rng_u128c(ih, il);
    llm_rng_u128 mult = llm_rng_u128c(LLM_PCG_MULT_HIGH, LLM_PCG_MULT_LOW);
    g->state = (llm_rng_u128)0;
    g->inc = (initseq << 1) | (llm_rng_u128)1;
    g->state = g->state * mult + g->inc;
    g->state += initstate;
    g->state = g->state * mult + g->inc;
    g->has_uint32 = 0; g->uinteger = 0;
}
static inline uint64_t llm_rng_pcg_next(llm_rng_pcg64* g) {
    g->state = g->state * llm_rng_u128c(LLM_PCG_MULT_HIGH, LLM_PCG_MULT_LOW) + g->inc;
    return llm_rng_pcg_output(g->state);
}
/* default_rng(seed) -> PCG64 seminato */
static inline void llm_rng_default(int64_t seed, llm_rng_pcg64* g) {
    uint32_t ent[16]; int n = 0;
    if (seed == 0) { ent[n++] = 0; }
    else { uint64_t s = (uint64_t)seed; while (s > 0) { ent[n++] = (uint32_t)(s & 0xFFFFFFFFu); s >>= 32; } }
    uint32_t pool[LLM_RNG_POOL]; llm_rng_seedseq_pool(ent, n, pool);
    uint64_t st[4]; llm_rng_generate_state64(pool, st);
    llm_rng_pcg_set_seed(g, st[0], st[1], st[2], st[3]);
}

static inline uint32_t llm_rng_pcg_next32(llm_rng_pcg64* g) {
    if (g->has_uint32) { g->has_uint32 = 0; return g->uinteger; }
    uint64_t next = llm_rng_pcg_next(g);
    g->has_uint32 = 1;
    g->uinteger = (uint32_t)(next >> 32);
    return (uint32_t)(next & 0xffffffffu);
}
/* Lemire32 (buffered_bounded_lemire_uint32). */
static inline uint32_t llm_rng_bounded32(llm_rng_pcg64* g, uint32_t rng) {
    const uint32_t rng_excl = rng + 1u;
    uint64_t m = (uint64_t)llm_rng_pcg_next32(g) * rng_excl;
    uint32_t leftover = (uint32_t)(m & 0xFFFFFFFFu);
    if (leftover < rng_excl) {
        const uint32_t threshold = (UINT32_MAX - rng) % rng_excl;
        while (leftover < threshold) {
            m = (uint64_t)llm_rng_pcg_next32(g) * rng_excl;
            leftover = (uint32_t)(m & 0xFFFFFFFFu);
        }
    }
    return (uint32_t)(m >> 32);
}
/* Lemire bounded: uniforme in [0, rng] INCLUSIVO (come random_bounded_uint64). */
static inline uint64_t llm_rng_bounded(llm_rng_pcg64* g, uint64_t rng) {
    if (rng == 0) return 0;
    if (rng <= 0xFFFFFFFFULL) {
        if (rng == 0xFFFFFFFFULL) return (uint64_t)llm_rng_pcg_next32(g);
        return (uint64_t)llm_rng_bounded32(g, (uint32_t)rng);
    }
    if (rng == 0xFFFFFFFFFFFFFFFFULL) return llm_rng_pcg_next(g);
    const uint64_t rng_excl = rng + 1;
    llm_rng_u128 m = (llm_rng_u128)llm_rng_pcg_next(g) * rng_excl;
    uint64_t leftover = (uint64_t)m;
    if (leftover < rng_excl) {
        const uint64_t threshold = (UINT64_MAX - rng) % rng_excl;
        while (leftover < threshold) {
            m = (llm_rng_u128)llm_rng_pcg_next(g) * rng_excl;
            leftover = (uint64_t)m;
        }
    }
    return (uint64_t)(m >> 64);
}

/* np.random.default_rng(seed).integers(0, high, n, dtype=int64) (endpoint=False). */
static inline void llm_dsv41_numpy_integers_i64(int64_t seed, int64_t high, int64_t* out, int n) {
    llm_rng_pcg64 g; llm_rng_default(seed, &g);
    uint64_t rng = (uint64_t)high - 1u;   /* high > 0, open endpoint -> rng = high-1 inclusivo */
    for (int i = 0; i < n; i++) out[i] = (int64_t)llm_rng_bounded(&g, rng);
}

#endif /* LLM_DSV41_NUMPY_RNG_H */
