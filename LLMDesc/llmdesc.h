#ifndef LLMDESC_H
#define LLMDESC_H
#include <stdint.h>
#include <stddef.h>
/* .llmarch — descrizione d'architettura data-driven. Un modello = una descrizione (.llmarch JSON):
 * config_map (chiavi del config.json), roles (match->ruolo, prima che combacia vince), quant (formato
 * per ruolo + la ricetta), structures (le primitive: attention/moe/compression/indexer/engram/vision). */
typedef struct { char match[96]; char role[32]; char submodel[32]; } LlmDescRole;
typedef struct { char role[32]; char fmt[16]; } LlmDescQuant;
/* Una struttura puo' essere una PIPELINE ORDINATA di primitive (`steps`: [{op, prim}]). E' la
 * "composizione" (livello 3): una struttura nuova = una lista di primitive esistenti, non codice. */
typedef struct { char op[24]; char prim[32]; } LlmDescStep;
typedef struct { char key[32]; char type[32]; char prim[32];
                 LlmDescStep steps[12]; int n_steps; } LlmDescStruct;  /* struttura dichiarata (attention/moe/vision/...) */
typedef struct {
    char arch[64];
    LlmDescRole roles[128];  int n_roles;
    LlmDescQuant quant[128]; int n_quant;
    LlmDescStruct structures[16]; int n_struct;
    /* codecs: dichiarazione formato -> plugin che lo implementa (es. "f16" -> "llmcodec_f16"). */
    char codec_fmt[16][32]; char codec_plugin[16][32]; int n_codec;
    char recipe[16], boost[64], boost_fmt[16];
    char scale_suffix[16];
    /* runtime: metadati ENGINE (activation/kv/state) — NON codec di checkpoint. Esposti alla
     * superficie `LLM KV` come informativi: `state` (stringa, es. "csa2"/"attention_kv"), il
     * dtype del KV (`kv.dtype`) e dell'attivazione (`activation.dtype`). Vuoti se assenti. */
    char runtime_state[32];
    char runtime_kv_dtype[16];
    char runtime_activation_dtype[16];
    /* source (dichiarazione MODALITA': come il modello e' composto) + runtime.multimodal/decode.
     * Prima non erano letti: la modalita' era dichiarata nell'ABI ma invisibile al model compiler. */
    char source_format[16];       /* source.format (gguf|safetensors) */
    char source_mmproj[256];      /* source.mmproj (file del projector multimodale) */
    char mm_vision[32];           /* runtime.multimodal.vision (es. "vit") */
    int  mm_aligner;              /* runtime.multimodal.aligner */
    int  mm_present;              /* runtime.multimodal.present */
    int  mm_mtp;                  /* runtime.decode.mtp */
    int  mm_dspark;               /* runtime.decode.dspark */
} LlmDesc;
int llmdesc_load(const char* path, LlmDesc* out);
/* Risolve la descrizione per un modello: legge `architectures`/`model_type` dal config.json, cerca
 * `<arch>.llmarch` in LLMDESC_ARCH_DIR (o `<hf_dir>/arch/`, o `<cwd>/arch/`, o accanto all'eseguibile),
 * la carica. Ritorna 0 e scrive `out`; `out_path` (opz.) riceve il percorso trovato. */
int llmdesc_resolve(const char* hf_dir, LlmDesc* out, char* out_path, size_t out_path_cap);
/* Applica le regole `roles` (prima che combacia). Ritorna il ruolo o "OTHER". */
const char* llmdesc_role(const LlmDesc* a, const char* tensor_name);
/* Formato per ruolo (da `quant`), o NULL. */
const char* llmdesc_fmt(const LlmDesc* a, const char* role);
/* Tipo dichiarato di una struttura (es. "attention" -> "mla"), o NULL. */
const char* llmdesc_struct_type(const LlmDesc* a, const char* key);
/* Nome del plugin PRIMITIVA dichiarato per una struttura (campo "prim"), o NULL. */
const char* llmdesc_struct_prim(const LlmDesc* a, const char* key);
/* Trova la struttura `key` (o NULL). */
const LlmDescStruct* llmdesc_struct(const LlmDesc* a, const char* key);
/* Cartella ABI/ canonica (risale dal cwd). Ritorna 1 e scrive il path in `out` se trovata. */
int llmdesc_abi_dir(char* out, size_t cap);
/* Strutture DICHIARATE che il CONVERTER non sa ancora mappare (attention/mla|gqa e moe sono ok).
 * Scrive un report in out e ritorna il numero di strutture NON supportate (0 = tutte mappabili). */
int llmdesc_unsupported(const LlmDesc* a, char* out, size_t cap);
/* Modalita' DICHIARATA dall'ABI: "text"|"vision"|"audio"|"ocr"|"mtp"|"omni" (piu' di una).
 * Derivata da source.mmproj + ruoli VISION/VISION_ALIGNER/AUDIO_/MTP + structures.{vision,audio}. */
const char* llmdesc_modality(const LlmDesc* a);
/* File mmproj dichiarato in source.mmproj (o "" se assente). */
const char* llmdesc_source_mmproj(const LlmDesc* a);
/* Nome del plugin che implementa il formato `fmt` (da `codecs`), o NULL. */
const char* llmdesc_codec_plugin(const LlmDesc* a, const char* fmt);
#endif
