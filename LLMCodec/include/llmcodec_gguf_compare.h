/* llmcodec_gguf_compare.h — confronto NUMERICO per-tensore tra due GGUF.
 *
 * Analisi (sola lettura): dequantizza i tensori COMUNI (stesso nome + stesse dimensioni) di due
 * modelli e ne misura la differenza numerica. Serve a due domande reali:
 *   - "questo modello deriva da quello?" / "quali layer sono FROZEN" in un finetune
 *     (per-tensore avg_diff ≈ 0 -> congelato; grande -> modificato);
 *   - "quale RUOLO/layer e' degradato" dopo quant/surgery (breakdown per-tensore, non un solo
 *     numero aggregato come `LLM TENSOR QUANTERROR`).
 *
 * Metrica per tensore (antirez, gguf-tools — normalizzazione uniforme su [-N,N] -> 100%):
 *   rel_l2        = sqrt( Σ(cand-ref)² / Σ ref² )                     (ricostruzione)
 *   avg_diff_pct  = ( Σ|cand-ref| / n ) / avg_mag * 75                (0 ≈ identico, ~100 = random)
 *   avg_mag       = ( Σ|ref| + Σ|cand| ) / (2n)
 *   max_abs       = max|cand-ref|
 * Il dequant e' a BLOCCHI (chunk) per non materializzare tensori enormi (embedding 2.5 GB).
 *
 * Licenza: il glob matcher e' adattato da antirez/gguf-tools (MIT). La metrica non ha vincoli.
 */
#ifndef LLMCODEC_GGUF_COMPARE_H
#define LLMCODEC_GGUF_COMPARE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double   rel_l2;        /* sqrt(ΣΔ²/Σref²) sul tensore (0 se ref tutto zero) */
    double   avg_diff_pct;  /* avg|Δ|/avg_mag*75 : "quanto e' diverso" in % (id uguali -> 0) */
    double   max_abs;       /* max|Δ| */
    double   avg_mag;       /* media di |w| sui due tensori (magnitudine tipica) */
    uint64_t numel;         /* numero di pesi del tensore */
    int      ggml_type_a;   /* tipo GGML del tensore in A */
    int      ggml_type_b;   /* tipo GGML del tensore in B */
} LlmCodecGgufCompareStat;

typedef struct {
    uint64_t n_a, n_b;       /* tensori totali nei due file */
    uint64_t matched;        /* nome comune, dims uguali, dequantizzati */
    uint64_t only_a, only_b; /* presenti in un solo file (rispetto al filtro) */
    uint64_t mismatch;       /* nome comune ma dims diverse */
    uint64_t skipped;        /* tipo GGML non dequantizzabile */
    double   rel_l2_agg;     /* sqrt(ΣΔ²/Σref²) aggregato su tutti i matched */
    double   avg_diff_pct;   /* media dei avg_diff_pct dei matched */
    double   max_diff_pct;   /* il massimo (il tensore piu' diverso) */
    char     max_diff_name[256];
} LlmCodecGgufCompareSummary;

/* Chiamata per ogni tensore confrontato. `st` valido solo durante la call.
 * Ritorna 0 per continuare, !=0 per interrompere l'iterazione. */
typedef int (*llmcodec_gguf_compare_cb)(const char* name, const LlmCodecGgufCompareStat* st, void* ud);

/* Confronta due GGUF. `glob` (NULL/"" = tutti): filtro glob stile antirez sui nomi tensore.
 * `cb` opzionale (dettaglio per-tensore). `out` riempito col riepilogo.
 * Ritorna 0 su successo, -1 su errore (file mancante o non GGUF). */
int llmcodec_gguf_compare(const char* path_a, const char* path_b, const char* glob,
                          llmcodec_gguf_compare_cb cb, void* ud,
                          LlmCodecGgufCompareSummary* out);

/* Glob matcher (adattato da antirez/gguf-tools, MIT): 1 se `pattern` matcha `string`.
 * Supporta `*`, `?`, `[...]`/`[^...]`/`[a-z]`, `\` escape; `nocase` per match case-insensitive. */
int llmcodec_globs_match(const char* pattern, const char* string, int nocase);

#ifdef __cplusplus
}
#endif

#endif /* LLMCODEC_GGUF_COMPARE_H */
