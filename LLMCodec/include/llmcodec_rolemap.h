/* llmcodec_rolemap.h — mappa RUOLI di un modello, letta da una descrizione `.llmarch`.
 *
 * Il converter GGUF<->MLX e' GENERICO (basato sui nomi): questa mappa gli da' la
 * SEMANTICA (ruolo di ogni tensore + suffisso della scala) senza dipendere dal
 * runtime. Stessa semantica di `llmdesc_role()` (prima regola che combacia vince),
 * ma self-contained: il codec resta autonomo.
 */
#ifndef LLMCODEC_ROLEMAP_H
#define LLMCODEC_ROLEMAP_H
#include <stddef.h>

/* Regola ruolo->tensore. `gguf` (opzionale) = nome canonico GGUF di destinazione: template con
 * `%d` sostituito dall'indice layer estratto dal nome sorgente (es. "blk.%d.attn_q_a.weight");
 * vuoto = si conserva il nome sorgente. `stack`=1 -> gli esperti per-esperto vanno IMPILATI in un
 * unico tensore 3D `[cols, rows, E]` col nome `gguf` (es. ffn_gate_exps). */
typedef struct { char match[96]; char role[32]; char submodel[32]; char gguf[64]; int stack; } LlmCodecRoleRule;
typedef struct { char role[32]; char fmt[16]; } LlmCodecRoleFmt;
typedef struct { char key[32]; char type[32]; } LlmCodecRoleStruct;
/* Sorgente pesi: formato -> come decodificarlo. `codec` seleziona l'implementazione dal file
 * (la mappa `codecs` lo risolve: es. "fp8_block" -> "fp8_block"). */
typedef struct { char name[24]; char dtype[16]; char codec[24]; char role[32]; char scale_suffix[16]; char bias_suffix[16]; int expert_axis; int fmt, p0, p1; char mode[16]; char global_suffix[16]; } LlmCodecWeightSource;
typedef struct { char fmt[24]; char plugin[32]; } LlmCodecDecl;
typedef struct { char role[32]; char codec[24]; int fmt, p0, p1; } LlmCodecConvertRole;

typedef struct {
    char        arch[64];
    LlmCodecRoleRule r[128];
    int         n;
    char        scale_suffix[16];  /* default ".scales"; es. ".scale" per DS4.1 */
    char        bias_suffix[16];   /* default ".biases" */
    LlmCodecRoleFmt  q[128];  int n_q;  /* sezione `quant`: ruolo -> formato (es. NORM->f32, OUTPUT->q6_k) */
    LlmCodecRoleStruct st[16]; int n_st;  /* sezione `structures`: chiave -> tipo (attention:mla, engram, vision:vit, ...) */
    int         expert_axis;         /* dalla struttura `moe` (default 3, come da .llmarch DS4.1) */
    LlmCodecWeightSource ws[8]; int n_ws;      /* sezione `weight_source`: formato sorgente -> decodifica */
    int         q_group, q_bits;          /* sezione `quant_params`: parametri affine (0 = non dichiarati) */
    LlmCodecDecl    codec[16]; int n_codec;/* sezione `codecs`: fmt -> plugin/codec */
    char ct_codec[24]; int ct_fmt, ct_p0, ct_p1;   /* sezione `convert_target`.default */
    LlmCodecConvertRole  ctr[16]; int n_ctr;            /* sezione `convert_target`.roles */
} LlmCodecRoleMap;

/* Carica `roles`/`scale_suffix`/`arch` da una `.llmarch` (JSON). 0 ok / -1. */
int llmcodec_rolemap_load(const char* llmarch_path, LlmCodecRoleMap* out);

/* Risolve il ruolo di `name`: prima regola la cui `match` compare in `name`, o "OTHER". */
const char* llmcodec_rolemap_of(const LlmCodecRoleMap* m, const char* name);

/* Nome canonico GGUF della regola che combacia su `name` (o NULL se assente/vuoto). */
const char* llmcodec_rolemap_gguf(const LlmCodecRoleMap* m, const char* name);
/* 1 se la regola che combacia richiede lo STACKING 3D (esperti per-esperto -> un tensore 3D). */
int llmcodec_rolemap_stack(const LlmCodecRoleMap* m, const char* name);

/* Vero se il ruolo e' un esperto (ROUTED_EXPERT/SHARED_EXPERT/...). */
int llmcodec_rolemap_is_expert(const char* role);

/* Formato dichiarato per ruolo (sezione `quant`), o NULL. */
const char* llmcodec_rolemap_fmt(const LlmCodecRoleMap* m, const char* role);

/* Tipo dichiarato di una struttura (es. "attention"->"mla", "engram"->"engram"), o NULL. */
const char* llmcodec_rolemap_struct_type(const LlmCodecRoleMap* m, const char* key);
/* expert_axis dalla struttura `moe` (default 3). */
int llmcodec_rolemap_expert_axis(const LlmCodecRoleMap* m);
/* Strutture dichiarate che il CONVERTER non sa mappare (supportate: attention gqa|mla, moe).
 * Scrive un report e ritorna il n. di strutture NON supportate (0 = tutte mappabili). */
int llmcodec_rolemap_unsupported(const LlmCodecRoleMap* m, char* out, size_t cap);

/* Sorgente pesi il cui `dtype` combacia (es. "F8_E4M3", "I8"), o NULL. */
const LlmCodecWeightSource* llmcodec_rolemap_source_for_dtype(const LlmCodecRoleMap* m, const char* dtype);
/* Entry piu' specifica per (dtype, role): preferisce quella col `role` combaciante. */
const LlmCodecWeightSource* llmcodec_rolemap_source_for(const LlmCodecRoleMap* m, const char* dtype, const char* role);
/* Entry per (dtype, mode): preferisce il `mode` combaciante, poi quella senza mode. */
const LlmCodecWeightSource* llmcodec_rolemap_source_mode(const LlmCodecRoleMap* m, const char* dtype, const char* mode);
/* Plugin/codec dichiarato per un fmt (sezione `codecs`), o NULL. */
const char* llmcodec_rolemap_codec_plugin(const LlmCodecRoleMap* m, const char* fmt);
/* Target di conversione (A1) per ruolo: scrive fmt/p0/p1 e ritorna il NOME-codec (o NULL). */
const char* llmcodec_rolemap_convert(const LlmCodecRoleMap* m, const char* role, int* fmt, int* p0, int* p1);

/* Quantization config MLX per-tensore (da config.json): mode/bits/group di default + override. */
typedef struct { char name[160]; char mode[16]; int bits, group; } LlmCodecQuantEntry;
typedef struct { char mode[16]; int bits, group; LlmCodecQuantEntry e[256]; int n; } LlmCodecQuantCfg;
int  llmcodec_quantcfg_load(const char* config_path, LlmCodecQuantCfg* out);        /* 0 ok / -1 */
const LlmCodecQuantEntry* llmcodec_quantcfg_find(const LlmCodecQuantCfg* c, const char* name); /* override o NULL */
/* compressed-tensors (vLLM/HF): legge quantization_config.{format,config_groups.weights} e
 * popola mode/bits/group. NON e' un codec: e' un ADAPTER che identifica il codec reale. 0 ok / -1. */
int  llmcodec_ct_load(const char* config_path, LlmCodecQuantCfg* out);
/* Riconoscitore: quant_method/format dal config (bitsandbytes, hqq, aqlm, vptq, spqr, quanto,
 * torchao, quark, autoround, compressed-tensors, fp8, gptq, awq, ...). NULL se assente.
 * `out_supported` = 1 se LLMEnv ha un codec per quel metodo. */
const char* llmcodec_quant_method(const char* config_path, int* out_supported);
/* Parametri affine dichiarati per il file (0 se assenti). */
int llmcodec_rolemap_group(const LlmCodecRoleMap* m);
int llmcodec_rolemap_bits(const LlmCodecRoleMap* m);

#endif
