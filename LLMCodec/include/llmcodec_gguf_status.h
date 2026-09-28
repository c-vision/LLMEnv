/* llmcodec_gguf_status.h — TASSONOMIA D'ERRORE del lettore GGUF (CONTRATTO STABILE).
 *
 * Un codice DISTINTO per ogni causa di rifiuto: "file troncato", "tipo ignoto", "tensori
 * sovrapposti", "versione non supportata" non condividono mai lo stesso `-1`. E' l'asset
 * assorbito dalla lane GGUF di FastLLM (vedi il piano LLMEnv §5): un `return -1` ambiguo
 * costringe chi legge a indovinare la reazione giusta, che invece e' OPPOSTA nei due casi
 * ("riscrivo il file" vs "correggo lo slicing").
 *
 * Direzione dei valori: 0 = OK, 1..N = errore. NON si rinumerano: il numero e' parte del
 * contratto (chi lega la libreria distingue le cause per numero e non per testo). Il valore
 * 14 e' un BUCO storico (era `ERR_ORDER`, rimosso: GGUF non richiede offset crescenti, vieta
 * la SOVRAPPOSIZIONE, che e' `OVERLAP`) e resta libero per non spostare i successivi.
 *
 * NOTE DI CONFINE: questa tassonomia e' quella del CODEC; l'enum locale `LLMCODEC_GGUF_ERR_*`
 * dentro `LLMEngine/mlx/llm_gguf.h` e' un'altra cosa (errori di OPEN/mmap del runner) e NON
 * va fusa con questa. I prefissi sono distinti di proposito.
 */
#ifndef LLMCODEC_GGUF_STATUS_H
#define LLMCODEC_GGUF_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LLMCODEC_GGUF_STATUS_OK = 0,
    LLMCODEC_GGUF_STATUS_NOT_GGUF = 1,      /* magic diversa da "GGUF"                    */
    LLMCODEC_GGUF_STATUS_VERSION = 2,       /* versione fuori da 2..4                     */
    LLMCODEC_GGUF_STATUS_TRUNCATED = 3,     /* buffer finito dentro header/KV/tensor-info */
    LLMCODEC_GGUF_STATUS_IMPLAUSIBLE = 4,   /* lunghezza stringa/array assurda            */
    LLMCODEC_GGUF_STATUS_KV_TYPE = 5,       /* tipo di valore KV sconosciuto              */
    LLMCODEC_GGUF_STATUS_NDIMS = 6,         /* n_dims 0 o oltre il massimo                */
    LLMCODEC_GGUF_STATUS_BAD_ALIGN = 7,     /* general.alignment non potenza di 2 / 0     */
    LLMCODEC_GGUF_STATUS_NEG_DIM = 8,       /* una dimensione di tensore = 0              */
    LLMCODEC_GGUF_STATUS_TYPE_UNKNOWN = 9,  /* ggml_type fuori tabella                    */
    LLMCODEC_GGUF_STATUS_TYPE_REMOVED = 10, /* Q4_2/Q4_3, tipi ritirati upstream          */
    LLMCODEC_GGUF_STATUS_NUMEL_BLOCK = 11,  /* numel non multiplo del blocco              */
    LLMCODEC_GGUF_STATUS_OVERLAP = 12,      /* intervalli di dati sovrapposti             */
    LLMCODEC_GGUF_STATUS_DATA_SHORT = 13,   /* dati dichiarati oltre la fine del file     */
    /* 14 = BUCO storico (ERR_ORDER): vedi l'intestazione. NON riusare. */
    LLMCODEC_GGUF_STATUS_NO_MEM = 15,       /* memoria esaurita                           */
    LLMCODEC_GGUF_STATUS_IO = 16,           /* errore di I/O (open/stat/mmap)             */
    LLMCODEC_GGUF_STATUS_NAME_DUP = 17,     /* due tensori con lo stesso nome             */
    LLMCODEC_GGUF_STATUS_COUNT = 18,        /* n_tensors/n_kv impossibili per la size     */
    LLMCODEC_GGUF_STATUS_DST_SMALL = 19,    /* buffer di destinazione troppo piccolo      */
    LLMCODEC_GGUF_STATUS_NOT_COVERED = 20   /* tipo valido ma non coperto (gap dichiarato)*/
} LlmCodecGgufStatus;

/* Testo del codice, SEMPRE non-NULL: anche per un numero che non conosce. */
const char* llmcodec_gguf_status_str(LlmCodecGgufStatus s);

#ifdef __cplusplus
}
#endif
#endif /* LLMCODEC_GGUF_STATUS_H */
