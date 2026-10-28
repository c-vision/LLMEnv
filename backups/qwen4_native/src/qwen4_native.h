/* qwen4_native.h — Port nativo dei kernel Metal di `antirez/ds4` (`metal/qwen4.metal`, MIT)
 * per Qwen3.8-Flash-Next (`qwen4_exp`) in LLMEngine.  P0: scaffold host-side.
 *
 * Meccanismo (coerente con gli altri `llm_mlx_*_kernel.h` del motore): ogni kernel Metal di ds4
 * (MSL, `[[threadgroup_position_in_grid]]`, template) viene incapsulato come SORGENTE stringa e
 * lanciato con `mlx_fast_metal_kernel_new(...)` (funzioni `*_kernel.h`).  Il calcolo resta fedele
 * a `qwen4_ref_*` di `ds4.c` (byte-identico dove ds4 usa le varianti `*_pf` con FP-contract pinned).
 *
 * Fasi: P0 scaffold · P1 hc · P2 GDN · P3 PLE · P4 router/MoE · P5 QSA/GQA · P6 dense/MTP/argmax · P7 wiring.
 * Gate model-free per fase (build + parità vs il kernel MLX esistente).  Nessun run del modello.
 */
#ifndef LLM_QWEN4_NATIVE_H
#define LLM_QWEN4_NATIVE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Tipi di peso qwen4 (ds4 `qwen4_row_dot` / function constants) ─────────────────────────── */
enum {
    QWEN4_W_F32   = 0,
    QWEN4_W_F16   = 1,
    QWEN4_W_Q4_0  = 2,
    QWEN4_W_Q8_0  = 8,
    QWEN4_W_Q2_K  = 10,
    QWEN4_W_Q4_K  = 12,
    QWEN4_W_IQ2_XXS = 16,
    QWEN4_W_BF16  = 30,
    QWEN4_W_MXFP4 = 39,
};

/* ── Costanti di forma qwen4_exp (dal config reale; usate dalle struct args) ───────────────── */
enum {
    QWEN4_HIDDEN      = 2560,
    QWEN4_HC_COUNT    = 4,
    QWEN4_HC_LOWRANK  = 320,
    QWEN4_EXPERTS     = 512,
    QWEN4_TOP_K       = 10,
    QWEN4_MOE_INTER   = 640,
    QWEN4_N_Q_HEAD    = 24,
    QWEN4_N_KV_HEAD   = 2,
    QWEN4_HEAD_DIM    = 256,
    QWEN4_IDX_N_HEAD  = 4,
    QWEN4_IDX_DIM     = 128,
    QWEN4_IDX_BUDGET  = 2048,
    QWEN4_IDX_RATIO   = 4,
};

/* ── Helper di riferimento (host, per i test di parità) — fedeli a ds4 `qwen4_*` ─────────── */
#include <math.h>
static inline float qwen4_native_sigmoid(float x) {          /* ds4 qwen4_sigmoid */
    if (x >= 0.0f) { const float e = expf(-x); return 1.0f / (1.0f + e); }
    const float e = expf(x); return e / (1.0f + e);
}
static inline float qwen4_native_softplus(float x) {         /* ds4 qwen4_softplus */
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return logf(1.0f + expf(x));
}
static inline float qwen4_native_silu(float x) {             /* ds4 qwen4_silu */
    return x * qwen4_native_sigmoid(x);
}

#ifdef __cplusplus
}
#endif
#endif /* LLM_QWEN4_NATIVE_H */
