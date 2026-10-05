/* llm_mlx_kvformat.h — §9.A M10c: contratto GMB_KV_FORMAT (dtype di STORAGE del KV).
 *
 * Il KV ha un dtype NATIVO (quello con cui il modello è addestrato/calibrato) e può essere
 * memorizzato in dtype diversi SOLO se ammessi (supportati/quantizzabili/ricostruibili); un
 * dtype non ammesso va RIFIUTATO — mai una conversione silenziosa. Classi (dal paper
 * DeepSeek-V4.1: il KV FP4 è nativo del modello, non una conversione runtime universale):
 *   NATIVE         : == dtype nativo del modello;
 *   SUPPORTED      : f16/bf16/f32 (calcolo, senza perdita rilevante);
 *   QUANTIZZABILE  : q8_0/int8/fp8 (quantizzazione a runtime);
 *   RICOSTRUIBILE  : fp4/mxfp4/nvfp4 (SOLO se è il dtype nativo; altrimenti rifiutato);
 *   UNSUPPORTED    : rifiutato.
 */
#ifndef LLM_MLX_KVFORMAT_H
#define LLM_MLX_KVFORMAT_H

#include <stddef.h>

typedef enum {
    LLM_KV_DTYPE_NATIVE = 0,
    LLM_KV_DTYPE_SUPPORTED,
    LLM_KV_DTYPE_QUANTIZABLE,
    LLM_KV_DTYPE_RECONSTRUCTABLE,
    LLM_KV_DTYPE_UNSUPPORTED
} LlmKvDtypeClass;

/* Classe di `requested` rispetto al dtype nativo `native`. Scrive il nome della classe in
 * `cls_out` ("NATIVE"|"SUPPORTED"|"QUANTIZABLE"|"RECONSTRUCTABLE"|"UNSUPPORTED"). */
int llm_mlx_kvformat_classify(const char* native_dtype, const char* requested, char* cls_out, size_t cls_cap);

/* Valida una richiesta di storage KV: 0 = ammessa, -1 = RIFIUTATA (mai conversione silenziosa).
 * `reason` (opzionale) riceve il motivo del rifiuto. */
int llm_mlx_kvformat_validate(const char* native_dtype, const char* requested, char* reason, size_t reason_cap);

#endif /* LLM_MLX_KVFORMAT_H */
