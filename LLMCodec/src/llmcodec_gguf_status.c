/* llmcodec_gguf_status.c — testi della tassonomia d'errore GGUF (asset assorbito da lane_gguf).
 * Vedi include/llmcodec_gguf_status.h per il contratto. */
#include "llmcodec_gguf_status.h"

const char* llmcodec_gguf_status_str(LlmCodecGgufStatus s) {
    switch (s) {
        case LLMCODEC_GGUF_STATUS_OK: return "ok";
        case LLMCODEC_GGUF_STATUS_NOT_GGUF: return "magic non GGUF";
        case LLMCODEC_GGUF_STATUS_VERSION: return "versione GGUF non supportata";
        case LLMCODEC_GGUF_STATUS_TRUNCATED: return "file troncato nell'header";
        case LLMCODEC_GGUF_STATUS_IMPLAUSIBLE: return "header implausibile";
        case LLMCODEC_GGUF_STATUS_KV_TYPE: return "tipo di metadata KV sconosciuto";
        case LLMCODEC_GGUF_STATUS_NDIMS: return "n_dims fuori range";
        case LLMCODEC_GGUF_STATUS_BAD_ALIGN: return "general.alignment non valido";
        case LLMCODEC_GGUF_STATUS_NEG_DIM: return "dimensione di tensore = 0";
        case LLMCODEC_GGUF_STATUS_TYPE_UNKNOWN: return "ggml_type sconosciuto";
        case LLMCODEC_GGUF_STATUS_TYPE_REMOVED: return "ggml_type ritirato upstream";
        case LLMCODEC_GGUF_STATUS_NUMEL_BLOCK: return "numel non multiplo del blocco di quantizzazione";
        case LLMCODEC_GGUF_STATUS_OVERLAP: return "intervalli di dati sovrapposti";
        case LLMCODEC_GGUF_STATUS_DATA_SHORT: return "dati dichiarati oltre la fine del file";
        case LLMCODEC_GGUF_STATUS_NO_MEM: return "memoria esaurita";
        case LLMCODEC_GGUF_STATUS_IO: return "errore di I/O";
        case LLMCODEC_GGUF_STATUS_NAME_DUP: return "nome di tensore duplicato";
        case LLMCODEC_GGUF_STATUS_COUNT: return "conteggi incoerenti con la dimensione del file";
        case LLMCODEC_GGUF_STATUS_DST_SMALL: return "buffer di destinazione troppo piccolo";
        case LLMCODEC_GGUF_STATUS_NOT_COVERED: return "tipo valido ma non coperto da questa lane (gap dichiarato)";
        default: return "codice di errore sconosciuto";
    }
}
