/* llmcodec.h — ABI dei PLUGIN CODEC (dylib) + registry di carico.
 *
 * DIRETTIVA: il CORE non contiene NESSUNA implementazione di encoder/decoder.
 * Ogni formato numerico e' una dylib `libllmcodec_<nome>.dylib` che esporta
 *   const LlmCodec* llmcodec_describe(void);
 * Il `.llmarch` (`codecs` / `weight_source.codec`) lo nomina; il converter lo carica a runtime.
 *
 * Una FAMIGLIA copre piu' formati selezionati da `fmt` (es. la famiglia "ggml" copre tutti i
 * ggml_type; "mlx_affine" seleziona bits/group; "fp8" seleziona block/fp4/row).
 *
 * `aux` = buffer secondario (es. la scala, per i formati block-scaled), altrimenti NULL.
 */
#ifndef LLMCODEC_CODEC_H
#define LLMCODEC_CODEC_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LLMCODEC_CODEC_ABI_VERSION 3

/* Layout dell'AUX (buffer secondario). La FAMIGLIA lo DICHIARA: il converter non ha elenchi cablati. */
enum { LLMCODEC_CODEC_AUX_NONE = 0, LLMCODEC_CODEC_AUX_SCALE = 1, LLMCODEC_CODEC_AUX_SCALES_BIASES = 2 };

typedef struct LlmCodec {
    const char* name;                 /* famiglia: "ggml", "mlx_affine", "fp8", ... */
    int         abi_version;
    int         aux_kind;             /* LLMCODEC_CODEC_AUX_* */
    /* dtype safetensors dei buffer prodotti da encode()/attesi da decode():
     * `which` 0 = principale, 1 = aux (o NULL se non applicabile). */
    const char* (*dtype_of)(int which, int fmt);
    /* Dims LOGICHE (rows,cols) dalla forma sorgente (d0,d1) nel formato `fmt` (params p0/p1). */
    int         (*shape_of)(int fmt, int p0, int p1, size_t d0, size_t d1, size_t* rows, size_t* cols);
    /* Forma di MEMORIA del buffer (which 0=principale, 1=aux) dati rows/cols logici. */
    int         (*store_shape_of)(int which, int fmt, int p0, int p1, size_t rows, size_t cols, size_t* d0, size_t* d1);

    /* Byte necessari per `rows`x`cols` valori nel formato `fmt` (params p0/p1) + byte dell'aux
     * (0 se la famiglia non usa l'aux). Ritorna 0 ok. */
    int    (*sizes)(int fmt, int p0, int p1, size_t rows, size_t cols,
                    size_t* out_bytes, size_t* aux_bytes);

    /* Decodifica: `in` (in_len byte) + `aux` (aux_len) -> `out` (rows*cols float). 0 ok. */
    int    (*decode)(const void* in, size_t in_len, int fmt, int p0, int p1,
                     size_t rows, size_t cols, const void* aux, size_t aux_len, float* out);

    /* Codifica: `in` (rows*cols float) -> `out` (out_cap) + `aux` (aux_cap). 0 ok + lunghezze. */
    int    (*encode)(const float* in, size_t rows, size_t cols, int fmt, int p0, int p1,
                     void* out, size_t out_cap, size_t* out_len, void* aux, size_t aux_cap, size_t* aux_len);
} LlmCodec;

/* Risolve un plugin codec per nome (dlopen + cache): $LLMCODEC_PLUGIN_DIR, <repo>/Resources/plugins/,
 * accanto all'eseguibile. NULL = non trovato (gap VISIBILE, mai un fallback silenzioso). */
const LlmCodec* llmcodec_resolve(const char* name);

/* Directory aggiuntive dove cercare i plugin (es. la cache librerie dell'app). */
void llmcodec_add_plugin_dir(const char* dir);
/* Simbolo esportato dal plugin `name` (dopo averlo caricato). NULL se assente. Per il bridge. */
void* llmcodec_symbol(const char* name, const char* symbol);

#ifdef __cplusplus
}
#endif
#endif /* LLMCODEC_CODEC_H */
