/* ============================================================================
 * llm_mlx_rope.c — dominio RoPE: array di frequenze custom (proportional, llama3, YaRN)
 * e applicazione YaRN (mscale). Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* FASE 1b.4.2 (Gemma4): array di frequenze custom per il RoPE "proportional" (RoPE
 * parziale) dei layer full/global -- capacità nuova nel motore: fino a qui
 * mlx_fast_rope è sempre stato chiamato con un array freqs VUOTO (usa solo `base`),
 * qui invece costruiamo un array esplicito e lo passiamo, con `base` disabilitato
 * (has_value=false), esattamente come fa mlx_lm (`base=None, freqs=self._freqs`).
 *
 * Formula verificata identica a ProportionalRoPE in mlx_lm/models/rope_utils.py:
 *   rotated_dims = dims * partial_rotary_factor (già arrotondato per difetto dal
 *                  chiamante, coerente con Python `int(dims * factor)`)
 *   exponents[i] = (2*i) / dims,  per i in [0, rotated_dims/2)
 *   freqs[i]     = factor * base^exponents[i],  per i in [0, rotated_dims/2)
 *   freqs[i]     = +inf,                        per i in [rotated_dims/2, dims/2)
 *                  (frequenza infinita = nessuna rotazione per quelle dimensioni --
 *                  la stessa identica primitiva mlx_fast_rope viene usata da Python
 *                  e da qui, quindi qualunque sia la convenzione interna con cui MLX
 *                  interpreta un freq infinito, è garantita identica in entrambi:
 *                  non serve replicare la matematica di mlx_fast_rope, solo l'array).
 * `factor` è quasi sempre 1.0 (assente da rope_parameters nel modello reale osservato,
 * cfg.get("factor", 1.0) lato Python) -- parametro comunque esposto per fedeltà. */
mlx_array llm_mlx_proportional_rope_freqs(int dims, int rotated_dims, float base, float factor) {
    int n = dims / 2;
    int half_rotated = rotated_dims / 2;
    float* freqs = (float*)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        if (i < half_rotated) {
            float exponent = (float)(2 * i) / (float)dims;
            freqs[i] = factor * powf(base, exponent);
        } else {
            freqs[i] = INFINITY;
        }
    }
    const int shape[1] = {n};
    mlx_array arr = mlx_array_new_data(freqs, shape, 1, MLX_FLOAT32);
    free(freqs);
    return arr;
}

/* FASE 1b.2 (Llama 3.x): array di frequenze custom per il RoPE "llama3" (scaling a 3
 * fasce di frequenza) -- verificato identico a Llama3RoPE in mlx_lm/models/rope_utils.py:
 *   freqs[i]   = base^(2*i/dims),                     per i in [0, dims/2)
 *   wavelen[i] = 2*pi*freqs[i]
 *   scaled[i]  = wavelen[i] > low_freq_wavelen ? freqs[i]*factor : freqs[i]
 *   (bassa frequenza/wavelength lunga: rallentata di `factor` -- estrapolazione)
 *   fascia media (high_freq_wavelen < wavelen[i] < low_freq_wavelen): interpolazione
 *   "smooth" fra la versione non scalata e quella scalata, calcolata qui usando SEMPRE
 *   freqs[i] non scalata (per costruzione, in questa fascia scaled[i]==freqs[i]: la
 *   condizione "wavelen > low_freq_wavelen" è falsa per definizione di fascia media --
 *   verificato algebricamente confrontando con l'ordine esatto delle operazioni Python,
 *   dove la variabile `freqs` usata dentro smooth_freqs è quella POST il primo `where`,
 *   che per la fascia media coincide con l'originale non scalata)
 *   altrimenti (alta frequenza, wavelength corta): scaled[i] invariata.
 * dove low_freq_wavelen = old_context_len/low_freq_factor, high_freq_wavelen =
 * old_context_len/high_freq_factor. */
mlx_array llm_mlx_llama3_rope_freqs(int dims, float base, float factor,
                                            float low_freq_factor, float high_freq_factor,
                                            float old_context_len) {
    int n = dims / 2;
    float low_freq_wavelen = old_context_len / low_freq_factor;
    float high_freq_wavelen = old_context_len / high_freq_factor;
    float* freqs = (float*)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) {
        float exponent = (float)(2 * i) / (float)dims;
        float f = powf(base, exponent);
        float wavelen = 2.0f * 3.14159265358979323846f * f;
        float scaled = (wavelen > low_freq_wavelen) ? (f * factor) : f;
        bool is_medium = (wavelen > high_freq_wavelen) && (wavelen < low_freq_wavelen);
        if (is_medium) {
            float smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
            freqs[i] = f / ((1.0f - smooth) / factor + smooth);
        } else {
            freqs[i] = scaled;
        }
    }
    const int shape[1] = {n};
    mlx_array arr = mlx_array_new_data(freqs, shape, 1, MLX_FLOAT32);
    free(freqs);
    return arr;
}

/* FASE 1b.7 (YaRN, 2026-09-06): replica 1:1 YarnRoPE in mlx_lm/models/rope_utils.py
 * (initialize_rope -> YarnRoPE, stessa formula usata da transformers). Necessario per
 * gpt-oss-20b/120b e Ternary-Bonsai-4B (rope_type "yarn"). Convenzione indice freqs
 * identica a llama3/proportional: i in [0, dims/2), exponent = 2*i/dims. */
static double llm_mlx_yarn_correction_dim(int dims, double orig, double num_rotations, double base) {
    return (double)dims * log(orig / (num_rotations * 2.0 * 3.14159265358979323846)) /
           (2.0 * log(base));
}
float llm_mlx_yarn_mscale(float factor, float mscale) {
    /* yarn_get_mscale: scale<=1 -> 1; altrimenti 0.1*mscale*ln(scale)+1. In LLMEnv
     * mscale_all_dim resta 0 (non presente nei config locali) -> denominatore = 1. */
    if (factor <= 1.0f) return 1.0f;
    return 0.1f * mscale * logf(factor) + 1.0f;
}
static mlx_array llm_mlx_yarn_rope_freqs(int dims, float base, float factor,
                                         int orig_max, float beta_fast, float beta_slow) {
    int n = dims / 2;
    double low_d = floor(llm_mlx_yarn_correction_dim(dims, orig_max, beta_fast, base));
    double high_d = ceil(llm_mlx_yarn_correction_dim(dims, orig_max, beta_slow, base));
    if (low_d < 0) low_d = 0;
    if (high_d > dims - 1) high_d = dims - 1;
    double denom = high_d - low_d;
    if (denom == 0.0) denom = 0.001; /* come python: max_val += 0.001 se min==max */
    float* freqs = (float*)malloc((size_t)n * sizeof(float));
    if (!freqs) return mlx_array_new();
    for (int i = 0; i < n; i++) {
        float exponent = (float)(2 * i) / (float)dims;
        float extra = powf(base, exponent);          /* freq_extra */
        float inter = factor * extra;                /* freq_inter */
        float ramp = (float)((i - low_d) / denom);   /* yarn_linear_ramp_mask */
        if (ramp < 0.0f) ramp = 0.0f;
        if (ramp > 1.0f) ramp = 1.0f;
        float mask = 1.0f - ramp;                    /* freq_mask */
        float d = inter * mask + extra * (1.0f - mask);
        freqs[i] = (d != 0.0f) ? (inter * extra) / d : extra;
    }
    const int shape[1] = {n};
    mlx_array arr = mlx_array_new_data(freqs, shape, 1, MLX_FLOAT32);
    free(freqs);
    return arr;
}
/* Cache statica delle frequenze YaRN: ricompila solo se i parametri cambiano (multi-modello
 * nello stesso processo, es. Ternary poi gpt-oss). Le chiamate per layer/token nel modello
 * corrente sono un hit. */
mlx_array llm_mlx_yarn_rope_freqs_cached(int dims, float base, float factor,
                                                int orig_max, float beta_fast, float beta_slow) {
    static int c_dims = 0;
    static float c_base = 0, c_factor = 0, c_bf = 0, c_bs = 0;
    static int c_orig = 0;
    static mlx_array c_arr = {0};
    static bool c_built = false;
    if (c_built && c_dims == dims && c_base == base && c_factor == factor &&
        c_orig == orig_max && c_bf == beta_fast && c_bs == beta_slow)
        return c_arr;
    if (c_built) mlx_array_free(c_arr);
    c_arr = llm_mlx_yarn_rope_freqs(dims, base, factor, orig_max, beta_fast, beta_slow);
    c_dims = dims; c_base = base; c_factor = factor; c_orig = orig_max;
    c_bf = beta_fast; c_bs = beta_slow; c_built = true;
    return c_arr;
}
/* Applica il RoPE YaRN a q_t/k_t ([B, heads, T, head_dim]) con offset; ritorna false su
 * errore. Dopo il rope applica la riscalatura mscale (il rope e' lineare -> post-scale
 * equivale al pre-scale di YarnRoPE su x). */
bool llm_mlx_yarn_rope_apply(mlx_array* q_out, mlx_array* k_out,
                                    int head_dim, float rope_theta, float factor, int orig_max,
                                    float beta_fast, float beta_slow, float mscale_param,
                                    mlx_array q_t, mlx_array k_t, int offset, mlx_stream stream) {
    mlx_array freqs_f32 = llm_mlx_yarn_rope_freqs_cached(head_dim, rope_theta, factor, orig_max,
                                                         beta_fast, beta_slow);
    if (freqs_f32.ctx == NULL) return false;
    /* La tabella in f32 PROMUOVE q/k a f32 (mlx_fast_rope promuove x al dtype delle freqs):
     * misurato su gpt-oss -> q/k f32, **cache K in f32** (8 byte/token/head in piu' di V, che
     * resta bf16) e 5 AsType/layer nella SDPA a dtype misti. Con una tabella convertita al
     * dtype dell'attivazione il rope resta a 16 bit (e la cache KV dimezza i byte di K).
     * LLM_YARN_FREQS_MATCH=0 -> tabella f32 (comportamento pre-fix, per A/B). */
    /* LLM_GPTOSS_ROPE16=1 (default 0 = pre-fix): applica lo scalare mscale (e l'eventuale cast
     * finale) nel dtype dell'ATTIVAZIONE invece che f32. Il pre-fix usava `new_float32(msc)`:
     * lo scalare f32 PROMUOVEVA q/k a f32 -> **cache K in f32** (4 byte/valore) e SDPA a dtype
     * misti (5 AsType/layer). MISURATO (gpt-oss-20b, 512 token, coppie affiancate): **-4,4%/-5,5%
     * di wall**; a 48 token neutro (la KV pesa poco). -108 nodi/token.
     * *** PROMOSSO A DEFAULT 2026-09-16 *** dopo aver corretto un bug NOSTRO (non del motore):
     * `mlx_array_new_data(&sf, ..., bf16)` INTERPRETA il buffer host nel dtype dichiarato (non
     * converte) -> lo scalare mscale diventava spazzatura (0xCCCD) e q/k erano distrutti (gate:
     * 11/300 con loop). Ora si costruisce f32 e si CONVERTE con mlx_astype.
     * Gate dopo il fix (teacher-forcing, corpus reale, skip 16):
     *   300 posizioni: default 84/300 vs ROPE16 85/300 (+1)
     *   600 posizioni: default 212/600 vs ROPE16 210/600 (**-2 = -0,33%**, dentro la barra usata
     *                  per le altre promozioni dtype di questo motore)
     * Wall end-to-end: **NON MISURABILE** con questa macchina carica. Tre esperimenti a coppie
     * A ORDINE ALTERNATO danno segni discordi: 6 coppie a 512 tok = +9,7%; 4 coppie a 512 tok =
     * -7,5%; 5 coppie a 2048 tok = mediana 0,959 (= neutro, spread +/-30%). L'effetto atteso
     * (K dimezzata) e' piccolo rispetto ai pesi (a 2048 tok la KV e' ~5% del traffico) e sta
     * sotto il rumore. **NON si rivendica uno speedup**: il valore dimostrato e' (a) **meta'
     * byte di cache K** (f32 4B -> bf16 2B: ~50 MB in meno a 2048 tok, capacita' per il contesto
     * lungo), (b) **-84 nodi/token**, (c) qualita' gate-equivalente. Kill-switch
     * LLM_GPTOSS_ROPE16=0 per A/B.
     * Storico: prima del fix la leva sembrava "rotta" (*** NON E' UNA PERDITA DI PRECISIONE:
     * ERA ROTTO ***) -- il gate teacher-forcing
     * (gate teacher-forcing, corpus samples/Fixtures/nll/corpus.txt, 300 posizioni,
     * skip 16) da' top-1 **84/300 = 72%** con il default contro **11/300 = 3,7%** con questa
     * leva, con **loop di ripetizione** nella sequenza greedy (410 410 410..., 11713 11713...).
     * Un KV a 16 bit non puo' costare 68 punti di top-1: c'e' un BUG nel percorso KV a 16 bit di
     * gpt-oss (sliding window/SDPA o append), non un trade-off numerico. La leva resta OPT-IN e
     * NON va promossa; il gate e' la riproduzione per indagarlo (vale -108 nodi/token e dimezza
     * i byte di K). */
    static int rope16 = -1;
    if (rope16 < 0) {
        const char* e = llm_ai_mlx_opt_env("LLM_GPTOSS_ROPE16");
        rope16 = (e != NULL && strcmp(e, "0") == 0) ? 0 : 1;   /* PROMOSSO a default 2026-09-16 */
    }
    mlx_dtype xdt = mlx_array_dtype(q_t);
    mlx_dtype use = MLX_FLOAT32;   /* frequenze SEMPRE f32 (identiche al reference) */
    const int scalar_dt_mode = rope16;
    mlx_array freqs = mlx_array_new();
    if (use == MLX_FLOAT32) {
        mlx_array_set(&freqs, freqs_f32);
    } else {
        static int c_dims = 0, c_orig = 0; static float c_base = 0, c_factor = 0, c_bf = 0, c_bs = 0;
        static mlx_array c_arr = {0}; static bool c_built = false; static int c_dt = -1;
        if (c_built && c_dims == head_dim && c_base == rope_theta && c_factor == factor &&
            c_orig == orig_max && c_bf == beta_fast && c_bs == beta_slow && c_dt == (int)use) {
            mlx_array_set(&freqs, c_arr);
        } else {
            if (c_built) { mlx_array_free(c_arr); c_arr = mlx_array_new(); c_built = false; }
            mlx_array cvt = mlx_array_new();
            if (mlx_astype(&cvt, freqs_f32, use, stream) == 0) {
                c_arr = cvt; c_built = true; c_dt = (int)use;
                c_dims = head_dim; c_base = rope_theta; c_factor = factor; c_orig = orig_max;
                c_bf = beta_fast; c_bs = beta_slow;
                mlx_array_set(&freqs, c_arr);
            } else {
                mlx_array_free(cvt);
                mlx_array_set(&freqs, freqs_f32);   /* fallback: tabella f32 */
            }
        }
    }
    if (freqs.ctx == NULL) return false;
    mlx_optional_float no_base = {0.0f, false};
    bool ok = mlx_fast_rope(q_out, q_t, head_dim, /*traditional=*/false, no_base, /*scale=*/1.0f,
                            offset, freqs, stream) == 0 &&
              mlx_fast_rope(k_out, k_t, head_dim, false, no_base, 1.0f, offset, freqs, stream) == 0;
    float msc = llm_mlx_yarn_mscale(factor, mscale_param);
    if (ok && msc != 1.0f) {
        /* lo scalare mscale VA nel dtype di lavoro: one scalar f32 promuoveva q/k a f32.
         * ATTENZIONE (bug trovato col gate 2026-09-16): mlx_array_new_data interpreta il buffer
         * host NEL DTYPE DICHIARATO, non lo converte. Passare `float sf` dichiarando bf16
         * reinterpreta i 4 byte come 2 di bf16 -> scalare SPazzatura (0xCCCD...) -> q/k distrutti
         * (top-1 3,7% con loop). Qui si costruisce f32 e si CONVERTE con mlx_astype. */
        mlx_dtype wdt = scalar_dt_mode ? mlx_array_dtype(*q_out) : MLX_FLOAT32;
        float sf = msc;
        mlx_array s32 = mlx_array_new_data(&sf, (const int[]){1}, 1, MLX_FLOAT32);
        mlx_array s = mlx_array_new();
        if (wdt == MLX_FLOAT16 || wdt == MLX_BFLOAT16) ok = mlx_astype(&s, s32, wdt, stream) == 0;
        else ok = mlx_array_set(&s, s32) == 0;
        mlx_array_free(s32);
        mlx_array q2 = mlx_array_new(), k2 = mlx_array_new();
        ok = ok && mlx_multiply(&q2, *q_out, s, stream) == 0 && mlx_multiply(&k2, *k_out, s, stream) == 0;
        if (ok) { mlx_array_set(q_out, q2); mlx_array_set(k_out, k2); }
        mlx_array_free(s); mlx_array_free(q2); mlx_array_free(k2);
    }
    /* modo 2: rope in f32 (frequenze IDENTICHE al reference) e output arrotondato al dtype
     * dell'attivazione -> cache/SDPA a 16 bit senza toccare le frequenze. */
    if (ok && rope16 && (xdt == MLX_FLOAT16 || xdt == MLX_BFLOAT16) && mlx_array_dtype(*q_out) != xdt) {
        mlx_array qc = mlx_array_new(), kc = mlx_array_new();
        if (mlx_astype(&qc, *q_out, xdt, stream) == 0 && mlx_astype(&kc, *k_out, xdt, stream) == 0) {
            mlx_array_set(q_out, qc); mlx_array_set(k_out, kc);
        }
        mlx_array_free(qc); mlx_array_free(kc);
    }
    mlx_array_free(freqs);
    return ok;
}
