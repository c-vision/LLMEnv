/* llmengine_plugin.h — ABI dei PLUGIN nativi: CODEC di formato + PRIMITIVE d'architettura.
 *
 * PERCHÉ. Un modello nuovo NON deve richiedere la ricompilazione del compilatore/core:
 *  - descrizione (misure, nomi, ruoli, quant, scale)  -> un file `.llmarch`      [dati]
 *  - FORMATO numerico nuovo (dtype)                   -> un plugin CODEC (dylib) [codice isolato]
 *  - PRIMITIVA d'architettura nuova (attn/comp/...)    -> un plugin PRIM  (dylib) [codice isolato]
 * Il `.llmarch` dichiara i plugin; il motore/compiler li carica a RUNTIME (dlopen). Il core non
 * cambia. Vedi Knowledge/LLMENV-LLM-Plugin-Schema.md (vault) per lo schema completo.
 *
 * Convenzione di caricamento: `llm_<tipo>_get("<nome>")` cerca `libllmcodec_<nome>.dylib` /
 * `libllmengine_plugin_prim_<nome>.dylib` in: $LLMENGINE_PLUGIN_DIR, <repo>/Resources/plugins/<rid>/, accanto
 * all'eseguibile. Ogni dylib esporta UN descrittore:
 *   CODEC -> const LlmCodec* llmcodec_describe(void);
 *   PRIM  -> const LlmEnginePluginPrim*  llmengine_plugin_prim_describe(void);
 */
#ifndef LLMENGINE_PLUGIN_H
#define LLMENGINE_PLUGIN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LLMENGINE_PLUGIN_ABI_VERSION 1

/* ── CODEC: un formato numerico (eventualmente quantizzato) ── */
typedef struct LlmCodec {
    const char* name;                 /* "q4_0", "iq2_xxs", "fp8_e4m3", "mxfp4", ... */
    int         ggml_type;            /* id GGML del formato (o -1 se non-GGUF) */
    int         block_n;              /* quanti valori per blocco (0 = non a blocchi) */
    size_t      (*bytes_for)(size_t n);                 /* byte per n valori (0 = errore) */
    int         (*quantize)(const float* in, size_t n, void* out, size_t out_cap);
    int         (*dequantize)(const void* in, size_t n, float* out);
} LlmCodec;

/* ── "REFERENCE IO": la forma più comune di una primitiva (op 1): un tensore [n] f32 in -> out,
 * con un tensore ausiliario opzionale (es. i pesi di una norm) e due parametri scalari. Le
 * primitive elementwise (norm, attivazioni, rope mono-tensore) targettano questa forma: così un
 * plugin resta indipendente dall'engine e testabile. */
#define LLMENGINE_PLUGIN_PRIM_OP_ELEMENTWISE 1
typedef struct LlmEnginePluginPrimIO {
    int          op;       /* LLMENGINE_PLUGIN_PRIM_OP_ELEMENTWISE */
    const float* in; float* out; size_t n;
    const float* aux; size_t aux_n;   /* ausiliario (es. pesi norm); può essere NULL */
    int          i0;                  /* parametro intero opzionale */
    float        f0;                  /* parametro float opzionale (es. eps) */
} LlmEnginePluginPrimIO;

/* ── PRIMITIVA: un'operazione d'architettura ──
 * L'engine istanzia la primitiva coi parametri della `.llmarch` (`structures.<kind>`), poi la
 * guida per-fase. `io` è il contesto dell'engine (opaco al plugin: l'ABI version la lega). */
typedef struct LlmEnginePluginPrim {
    const char* name;                 /* "mla", "csa_hca", "topk_selector", "vit", ... */
    const char* kind;                 /* "attention"|"compression"|"indexer"|"engram"|"vision"|"norm"|"moe" */
    int         abi_version;          /* LLMENGINE_PLUGIN_ABI_VERSION */
    void*       (*create)(const char* params_json);      /* dai `structures` del .llmarch */
    void        (*destroy)(void* handle);
    int         (*forward)(void* handle, void* io);      /* una fase; 0 = ok */
} LlmEnginePluginPrim;

/* ── PRIMITIVA MLX-AWARE ──
 * Per le primitive che VIVONO nel forward del motore MLX (attention/ffn/norm): il plugin lavora
 * sui handle MLX OPACHI (void*), ZERO copie -> nessuna regressione di velocità. L'op seleziona la
 * fase; la forma dell'io e' versionata (abi_version). */
#define LLMENGINE_PLUGIN_PRIMMLX_OP_NORM 1   /* out = rmsnorm(x) * w  (x/w/out = mlx_array* opachi) */
#define LLMENGINE_PLUGIN_PRIMMLX_OP_ROPE 2
typedef struct LlmEnginePluginPrimMLXIO {
    int     op;
    void*   x; void* w; void* out;   /* mlx_array* (opachi) */
    void*   stream;                  /* mlx_stream* (opaco) */
    size_t  n;
    float   eps;
    int     pos;
    float   theta;
} LlmEnginePluginPrimMLXIO;
typedef struct LlmEnginePluginPrimMLX {
    const char* name;
    const char* kind;               /* "norm" | "attention" | "ffn" | ... */
    int         abi_version;
    void*       (*create)(const char* params_json);
    void        (*destroy)(void* handle);
    int         (*forward)(void* handle, LlmEnginePluginPrimMLXIO* io);
} LlmEnginePluginPrimMLX;
/* Ogni plugin MLX esporta: const LlmEnginePluginPrimMLX* llmengine_plugin_prim_mlx_describe(void); */
const LlmEnginePluginPrimMLX* llmengine_plugin_prim_mlx_get(const char* name);

/* ── Registry (dlopen + cache). Ritorna il descrittore del plugin o NULL (gap visibile). ── */
const LlmCodec* llmengine_plugin_codec_get(const char* name);
const LlmEnginePluginPrim*  llmengine_plugin_prim_get(const char* name);

/* Elenca e valida i plugin disponibili nel dir (log). Ritorna il numero trovato. */
int  llmengine_plugin_list(void);

/* Percorso risolto di un plugin (per diagnostica); NULL se assente. */
const char* llmengine_plugin_path(const char* kind, const char* name);

#ifdef __cplusplus
}
#endif

#endif /* LLMENGINE_PLUGIN_H */
