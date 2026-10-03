/* ============================================================================
 * llmengine_mlx.c
 * Implementation -- see llmengine_mlx.h.
 *
 * Riscrittura in C puro (2026-08-24) della precedente implementazione C++.
 * Stesso identico algoritmo/comportamento, stessa superficie pubblica
 * (llmengine_mlx.h non e' stata toccata). Vedi
 * Fase0b-LlamaCpp-Wrapper-C-Rewrite-Plan.md per il piano complessivo.
 *
 * Sostituzioni: nlohmann::json -> cJSON, std::string -> buffer C fissi/sds
 * dove serve concatenazione, std::vector<mlx_map_string_to_array> (shard) ->
 * array C con malloc/realloc, std::set<std::string> (dedup shard file) ->
 * array C con ricerca lineare (N tipicamente piccolo, poche decine di shard
 * al massimo), std::vector<mlx_array>/std::vector<bool> (k_cache/v_cache/
 * has_cache di sessione) -> array C a dimensione fissa (n_layers, mai
 * ridimensionati dopo la creazione), struct con distruttore C++ -> funzioni
 * *_destroy esplicite chiamate dai wrapper extern "C" originali (ora solo C).
 * mlx-c stesso resta C++ internamente (Metal/C++), esposto via API C pura --
 * non e' in discussione, e' codice upstream vendorizzato invariato.
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */

#include "llm_mlx_array_count.h"
#include "llmengine_plugin.h"
#include "llmdesc.h"  /* K4: contatore handle (va PRIMA degli header di progetto) */
#include "llmcodec_rolemap.h" /* codec: mappa HF<->GGUF da `.llmarch` (load GGUF generico) */
#include "llmcodec_fp16.h"    /* llmcodec_f16_from_f32 (pesi GGUF -> F16) */
#include "llmcodec_quant_awq.h" /* llmcodec_awq_dequant: decoder AWQ int4 GEMM (checkpoint autoawq) */
#include "llmcodec_quant_gptq.h" /* llmcodec_gptq_dequant: decoder GPTQ int2/4/8 (autogptq/gptqmodel) */
#include "llmcodec_quant_nf4.h"  /* llmcodec_nf4_dequant: decoder bitsandbytes NF4 (+double-quant) */
#include "../../LLMMetal/oq_a8/llmmetal_oq_a8_bridge.h"  /* P10: GEMM int8-act x Q4/Q5 sui tensor unit M5 (port oMLX) */
#include "llm_mlx_ops.h"   /* moduli componibili riusabili (SDPA window, RoPE) */
#include "llm_mlx_which_path.h"  /* W1: dump del ramo attivo (LLM_WHICH_PATH=1) */
#include "llm_mlx_packed_qkv.h"
#include "llm_mlx_ngram.h"  /* tabella n-gram per self-speculative decoding (P1) */
#include "llm_mlx_kvpage.h"  /* KV-cache paging (roadmap 8.11 fase 3) */
#include "llm_mlx_kvdisk.h"     /* §G: tier SSD dei blocchi KV (offload prefissi; opzione di registro `kvdisk`) */
#include "llm_mlx_kvdisk_io.h"  /* §G: serializzazione MLX <-> blob KV */
#include "llm_mlx_kernel.h"   /* runtime kernel replacement (roadmap 8.11 fase 3) */
#include "llm_mlx_lora.h"     /* adapter LoRA (roadmap 8.11 fase 4) */
#include "llm_mlx_rewrite.h"  /* model rewrite (roadmap 8.11 fase 5) */
#include "llm_mlx_merge.h"    /* model merging (roadmap 8.11 fase 6) */
#include "llm_mlx_radix.h"    /* radix-trie prefix cache (fix coding contesto lungo) */
#include "llm_mlx_gdn_kernel.h"
#include "llm_mlx_gdn_gate_kernel.h" /* kernel Metal fuso del gate GDN (beta + g) — LLMMETAL_GDN_GATE_METAL */
#include "llm_mlx_mpp_q4_kernel.h" /* GEMM q4 affine su MPP (tensor unit Apple), LLMMETAL_MPP_Q4 */
#include "llmmetal_mpp_q4_bridge.h"   /* bridge metallib MPP q4 (le tensor-op NON passano da mlx_fast_metal_kernel) */
 /* kernel Metal GDN fuso prefill (collo) */
#include "llm_mlx_router_kernel.h" /* kernel Metal router MoE fuso (LLMMETAL_GLM_ROUTER_METAL) */
#include "llm_mlx_geglu_kernel.h" /* kernel Metal GeGLU fuso decode (LLMMETAL_FUSE_GEGLU_METAL) */
#include "llm_mlx_swiglu_kernel.h" /* kernel Metal SwiGLU fuso FFN densi (LLMMETAL_FUSE_SWIGLU_METAL) */
#include "llm_mlx_swiglu_gptoss_kernel.h" /* kernel Metal SwiGLU gpt-oss (LLMMETAL_GPTOSS_SWIGLU_METAL) */
#include "llm_mlx_moe_fused_kernel.h" /* kernel Metal MoE decode fuso (LLMENGINE_MOE_FUSE_MOE_DECODE), da mlxcel */
#include "llm_mlx_flash_decode_kernel.h" /* kernel Metal SDPA decode per head_dim fuori flash MLX (gemma4 hd=512, LLM_GEMMA_FLASH_DECODE) */
#include "llm_mlx_kv_quant_sdpa_kernel.h" /* kernel Metal dequant-on-read KV-quant decode (LLM_KV_DEQUANT_READ, fase 1) */
#include "llm_mlx_qwen4exp.h"  /* Qwen4-Exp (Flash-Next): indexer QSA, N-gram/PLE, hyper-connection */
#include "llm_dsv41_engine.h"  /* DeepSeek V4.1: forward whole-model (mHC/CSA2/Engram) con cache 7 slot */
#include "llm_dsv4_engine.h"  /* DS4 (DeepSeek V4 Flash): forward whole-model da GGUF singolo */
#include "llm_dsv4_tokenizer_json.h" /* builder JSON tokenizer DS4 dal GGUF (no llmtoken) */
#include "llm_kvmetal_bridge.h" /* bridge ABI-C verso KV page-pool Metal reale (LLMMETAL_KVMETAL, sperimentale) */
#include "llm_gemvq4_bridge.h"  /* bridge ABI-C verso kernel GEMV Q4 Metal reale (LLM_GEMVQ4, sperimentale) */
#include "llm_mlx_fused_rope_append.h" /* kernel Metal fuso layout+RoPE decode (LLM_FUSE_ROPE, G1) */
#include "llm_mlx_dispatch_profile.h"   /* G0: contatore deterministico dei nodi di dispatch (LLM_DISPATCH) */
#include "llm_pflash.h"  /* PFlash (#287): compressione del prompt a contesto lungo (LLM_PFLASH, default AUTO) */
#include "mlx/c/mlx.h"
#include "llm_mlx_internal.h"  /* tipi condivisi del motore (split) */
#include <cJSON.h>
#include "llm_mlx_bonsai_hadamard.h"   /* Ternary-Bonsai 2 27B (PrismML): contratto Hadamard */

#include <math.h>
#include <dlfcn.h>   /* dladdr: directory della dylib (default path risorse oQ A8) */
#include <locale.h>
#include <stdint.h>
#include <stdio.h>
#include <execinfo.h>   /* LLM_MLX_ERRTRACE: backtrace sull'error handler MLX */
#include <sys/sysctl.h> /* P4: iogpu.wired_limit_mb / hw.memsize */
#include <stdlib.h>
#include <string.h>
#include <dirent.h>     /* llm_mlx_dir_safetensors_bytes: footprint del modello (fix wired 2026-09-17) */
#include <sys/stat.h>
#include <unistd.h>   /* pread/close -- reader mirato a righe di ngram_table.bin */
#include <sys/sysctl.h> /* sysctlbyname(hw.memsize) -- policy auto wired-limit (Mixtral) */
#include <fcntl.h>    /* open/O_RDONLY -- idem */

/* K4: il contatore di ownership degli handle mlx_array vive in llm_mlx_array_count.h,
 * incluso SOPRA (prima degli header di progetto) cosi' conta anche gli header inline. */

/* Handle MLX VUOTO condiviso (ctx==NULL) per gli argomenti opzionali "assenti"
 * (es. `biases` di `mlx_dequantize`/`mlx_quantize`): non alloca nulla e non va liberato.
 * Elimina il pattern `mlx_array_new()` inline mai liberato (leak di handle, +1/token
 * sull'embedding: vedi Knowledge/LLMEnv-Core-Difetti-vs-mlxlm.md). */
mlx_array g_mlx_empty_arr = {0};

/* --------------------------------------------------------------------------
 * x @ w^T dequantizzato al volo -- la shape condivisa da ogni proiezione
 * Q/K/V/O nel blocco di attenzione. group_size/bits fissi a 64/8 per questo
 * modello (verificato contro il blocco "quantization" di config.json), non
 * ancora un parametro generico -- usata solo dalle funzioni di checkpoint
 * (Fase 1.3/1.4), il motore generico sotto usa quantized_linear_cfg.
 * -------------------------------------------------------------------------- */
bool quantized_linear(mlx_array* res, mlx_array x, mlx_array w, mlx_array scales, mlx_array biases, mlx_stream stream) {
    mlx_optional_int group_size = {64, true};
    mlx_optional_int bits = {8, true};
    return mlx_quantized_matmul(res, x, w, scales, biases, /*transpose=*/true, group_size, bits, "affine", stream) == 0;
}

/* Cerca "{base}.weight" / "{base}.scales" / "{base}.biases" come gruppo --
 * ogni tripla di tensori quantizzati in questo modello segue questa
 * convenzione di naming. */
bool get_quantized_triple(mlx_map_string_to_array map, const char* base,
                                  mlx_array* w, mlx_array* scales, mlx_array* biases) {
    char name[256];
    snprintf(name, sizeof(name), "%s.weight", base);
    if (mlx_map_string_to_array_get(w, map, name) != 0) return false;
    snprintf(name, sizeof(name), "%s.scales", base);
    if (mlx_map_string_to_array_get(scales, map, name) != 0) return false;
    snprintf(name, sizeof(name), "%s.biases", base);
    if (mlx_map_string_to_array_get(biases, map, name) != 0) return false;
    return true;
}

/* --- FASE 1.5: modello generico (dimensioni + pesi multi-shard), non più
 * hardcodato/singolo-shard come le funzioni di checkpoint sopra. --- */

bool plain_linear(mlx_array* res, mlx_array x, mlx_array w, mlx_stream stream); /* FASE 1b.3, definita sotto */
bool rms_norm_cfg(mlx_array* res, mlx_array x, mlx_array w, float eps, mlx_stream stream); /* bf16 refactor §9 */
/* WIRING PLUGIN: primitiva MLX per la NORM (dalla .llmarch `structures.norm.prim`). Se presente, */
/* rms_norm_cfg la usa al posto del built-in (handle opachi, zero copie). */
const LlmEnginePluginPrimMLX* g_primmlx_norm = NULL;
void* g_primmlx_norm_h = NULL;

/* Come quantized_linear() sopra, ma con group_size/bits letti da config.json
 * invece che fissati a 64/8 -- usata solo dal motore generico sotto, le
 * funzioni di checkpoint restano invariate (già verificate). */
/* §quant per-tensore: deriva il bits CORRETTO dalle shape di un tensore quantizzato.
 * Per affine quant, w e' [out, wcols] (packing 32/bits), scales e' [out, scols]
 * con in = scols*group. Quindi bits = 32*wcols/in. Ritorna il bits derivato se
 * valido (4/8/16), altrimenti `fallback`. Modelli a precisione mista (Qwen3.6-35B:
 * embed/linear bits8, esperti bits4) richiedono questo per ogni tensore. */
int llm_mlx_derive_bits(mlx_array w, mlx_array scales, int group_size, int fallback) {
    if (group_size <= 0 || scales.ctx == NULL || w.ctx == NULL) return fallback;
    if (mlx_array_ndim(w) < 2 || mlx_array_ndim(scales) < 2) return fallback;
    const int* ws = mlx_array_shape(w);
    const int* ss = mlx_array_shape(scales);
    int wcols = ws[mlx_array_ndim(w) - 1];
    int scols = ss[mlx_array_ndim(scales) - 1];
    int in = scols * group_size;
    if (in <= 0 || wcols <= 0) return fallback;
    int derived = (32 * wcols) / in;
    return (derived == 4 || derived == 8 || derived == 16) ? derived : fallback;
}


/* bfloat16 refactor (§9): quando attivo, le attivazioni dei layer girano in bf16 come mlx_lm
 * (input del matmul + output del rms_norm). Default OFF -> nessuna variazione sul denso.
 * Sincronizzata da llm_ai_mlx_model_set_bfloat16. Il matmul quantizzato MLX produce in bf16
 * quando l'input e' bf16 (promotion). */
bool g_bf16_all = false;
/* L19 (2026-09-08): il "bit a 16 bit" NON e' sempre bf16. Micro-harness canarizzato sulle shape
 * reali (/tmp/llm-l19/gemv3.py): la GEMV 8-bit affine di mlx prende il kernel veloce in funzione
 * della COPPIA (dtype di x, dtype delle SCALE del checkpoint) -- su DeepSeek-R1-32B-8bit (scale
 * F16) x=f16 batte fp32 del 21-27% e batte bf16 del 24-28%; su Qwen3.8-27B-8bit (scale BF16)
 * vince bf16. Chiedere bf16 a un modello con scale f16 e' il mix PIU' LENTO di tutti (peggio di
 * fp32): e' l'unica cosa che riconcilia "per-op bf16 +25%" con "model-level bf16 su qwen2 1,03x"
 * (misura L8/A-B). Un solo dtype a 16 bit per il modello carico, letto da llm_mlx_apply_bf16_mode.
 * Default = bf16: lo stato validato e promosso fin qui resta identico se nessuno chiede "f16". */
mlx_dtype g_act16 = MLX_BFLOAT16;
mlx_dtype llm_mlx_act_dt(bool on) { return on ? g_act16 : MLX_FLOAT32; }
const char* llm_mlx_act16_name(void) {
    return g_act16 == MLX_FLOAT16 ? "f16" : (g_act16 == MLX_BFLOAT16 ? "bf16" : "fp32");
}
/* KV quant: riferimento alla sessione che ha il KV quant attivo (per accedere ai campi kvq_*
 * per-layer dal forward senza cambiare le firme di ARCH_TABLE). Fragile con piu' sessioni KV
 * quant contemporanee: accettato per il KV quant (scenario long-context, tipicamente una sessione). */

/* RMLX10 Fase A (2026-09-03, vault Todo/(open) - RMLX10-handoff): decode-step con
 * KV-cache a buffer fisso shape-statica (prerequisito per la closure mlx_compile di
 * Fase B). Gated da env LLM_FIXKV=1, DEFAULT OFF -> il percorso eager resta intatto.
 * Attivo SOLO per model_type=="mixtral" (slice verticale, vedi layer_forward). */
static int g_fixkv_state = -1; /* -1 = non inizializzato (letto al primo uso) */
bool llm_mlx_fixkv_enabled(void) {
    if (g_fixkv_state < 0) g_fixkv_state = (llm_ai_mlx_opt_env("LLM_FIXKV") != NULL) ? 1 : 0;
    return g_fixkv_state == 1;
}

/* LLM_KVSEG (#61, 2026-09-11) — KV cache SEGMENTATA per il decode denso a contesto lungo.
 *
 * CAUSA MISURATA (kvf16_idle_bench 2026-09-11, Llama-3.2-1B-4bit, N=16384, idle,
 * adiacente: LLM_KVF16 154,9 tok/s vs rapid 243,9 = 0,64x FUORI soglia): anche con la
 * cache a buffer fisso (KVSLICE) la SCRITTURA passa da mlx_slice_update che e'
 * FUNZIONALE in MLX (ritorna un nuovo array): a ogni step MLX ricopia l'INTERO buffer
 * da `cap` per inserire 1 token -> O(pos) byte copiati per token per layer, esattamente
 * il costo del concat che KVSLICE voleva eliminare. La LETTURA (mlx_slice del prefisso,
 * zero-copy) non e' il problema.
 *
 * CURA: segmentare la cache in un MAIN statico (mai riscritto durante il decode:
 * nessuna copia O(pos) per token) + un SEGMENTO corrente piccolo (capacita' fissa
 * LLM_KVSEG_SIZE token): per ogni step si scrive O(SEG) nel segmento (concat+set su
 * buffer piccolo, costo costante) e si fa attention in DUE PEZZI (main + segmento)
 * con merge softmax online (max-trick):
 *   m = max(l1, l2);  l = exp(l1-m) + exp(l2-m);
 *   o = (exp(l1-m)*o1 + exp(l2-m)*o2) / l;
 * dove l1/l2 sono i denominatori softmax (somma degli esponenziali) di ciascuna parte.
 * Il main cresce SOLO di un flush ogni LLM_KVSEG_SIZE step (concat ammortizzato, il
 * vecchio main liberato solo dopo che il nuovo e' pronto); il segmento si riscrive a
 * costo O(SEG) costante.
 *
 * STORIA (2026-09-10/11, tutte misurate, vedi AiWiki MLX-Engine-Speed-Gap):
 * - LLMMETAL_KVMETAL_AUTH (pool Metal via bridge ObjC): 2,7-4x PEGGIO anche a concat
 *   soppresso — il round-trip host->device per token per layer spezza il pipelining
 *   MLX (un solo sync per token su tutti i layer). NON ripetere quell'errore: qui NON
 *   c'e' nessun bridge, nessun sync per-layer, tutto resta nel grafo MLX.
 * - KVSLICE (buffer fisso): vince sulla LETTURA ma la slice_update funzionale
 *   ricopia comunque tutto -> a N=16K resta ~0,64x. Questa segmentazione elimina
 *   l'ULTIMA copia O(pos).
 *
 * Gating: LLM_KVSEG=1 (default OFF). A/B: LLM_KVSEG_VERIFY=1 calcola ENTRAMBI i
 * risultati (segmentato e baseline KVSLICE) e stampa il max_abs_diff su stderr,
 * usando SEMPRE il risultato baseline (zero rischio, validazione su modello reale).
 * Contratto: richiede il ramo KVSLICE attivo (stessi gate: decode mono-token, cache
 * presente, kv_max_tokens<=0) — e' un RIFINIMENTO del suo layout, non un terzo path. */
#define MAX_KVSEG_LAYERS 256
static int g_kvseg_state = -1;
bool llm_mlx_kvseg_enabled(void) {
    if (g_kvseg_state < 0) g_kvseg_state = (llm_ai_mlx_opt_env("LLM_KVSEG") != NULL) ? 1 : 0;
    return g_kvseg_state == 1;
}
static int g_kvseg_verify_state = -1;
bool llm_mlx_kvseg_verify_enabled(void) {
    if (g_kvseg_verify_state < 0) g_kvseg_verify_state = (llm_ai_mlx_opt_env("LLM_KVSEG_VERIFY") != NULL) ? 1 : 0;
    return g_kvseg_verify_state == 1;
}
/* Dimensione fissa del segmento (token). Override: LLM_KVSEG_SIZE. Il main e' il
 * prefill + i segmenti chiusi; per costruzione seg_cap <= capacita' del buffer
 * KVSLICE cresciuto, quindi la riscrittura del segmento resta a costo costante. */
static int llm_mlx_kvseg_size(void) {
    static int v = 0;
    if (v == 0) {
        const char* e = llm_ai_mlx_opt_env("LLM_KVSEG_SIZE");
        v = (e != NULL && atoi(e) > 0) ? atoi(e) : 512;
    }
    return v;
}

/* Forward declare del modello MLX (la struct è definita più sotto nel file, ~riga 2537):
 * serve alle funzioni di dominio KVSEG/flush che ricevono LlmHlxModel*. */
typedef struct LlmMlxModel LlmMlxModel;

/* Stato per-layer della cache segmentata. Pivot del design: k_main/v_main (il grosso
 * della sequenza) NON vengono riscritti durante il decode — crescono solo di un flush
 * ogni LLM_KVSEG_SIZE step (concat ammortizzato, il vecchio main liberato solo dopo
 * che il nuovo e' pronto). k_seg/v_seg sono il segmento corrente, riscritto a costo
 * O(SEG) costante a ogni step. La sessione possiede questi array; i k_cache/v_cache
 * dichiarativi del layer restano i canali verso il resto del motore: mantengono la
 * shape logica [1,nkv,base_len+seg_len,hd] via una SLICE ZERO-COPY pubblicata a ogni
 * step, cosi' qualunque altro consumatore (dump, ispezioni) vede la dimensione giusta
 * senza copie O(pos). */
static LlmKvSegState g_kvseg[MAX_KVSEG_LAYERS];

void llm_mlx_kvseg_reset_layer(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= MAX_KVSEG_LAYERS) return;
    LlmKvSegState* s = &g_kvseg[layer_idx];
    if (!s->active) return;
    mlx_array_free(s->k_main); mlx_array_free(s->v_main);
    mlx_array_free(s->k_seg);  mlx_array_free(s->v_seg);
    memset(s, 0, sizeof(*s));
}
/* Reset di tutte le layer (chiamato dal free del modello — vedi llm_mlx_model_free). */
void llm_mlx_kvseg_reset_all(void) {
    for (int i = 0; i < MAX_KVSEG_LAYERS; i++) llm_mlx_kvseg_reset_layer(i);
}

/* Flush: il segmento (pieno o parziale) entra nel main. UNA concatenazione ogni
 * seg_cap step invece che una copia O(pos) a ogni step: e' questa la cura dell'
 * O(pos) del KVSLICE (la slice_update funzionale ricopia l'intero buffer). */
static bool llm_mlx_kvseg_flush(mlx_stream stream, int nkv, int hd, int layer_idx) {
    LlmKvSegState* s = &g_kvseg[layer_idx];
    if (s->seg_len == 0) return true;
    if (s->base_len <= 0) return false;   /* main non ancora pronto: non si può flushare */
    /* Il main nuovo contiene [0..base_len) vecchio + [0..seg_len) del segmento:
     * slice del segmento alla sua lunghezza REALE (il buffer è a seg_cap). */
    mlx_array kseg_v = mlx_array_new(), vseg_v = mlx_array_new();
    int p0[4] = {0,0,0,0}, p1[4] = {1,nkv,s->seg_len,hd}, ps[4] = {1,1,1,1};
    bool ok = mlx_slice(&kseg_v, s->k_seg, p0, 4, p1, 4, ps, 4, stream) == 0 &&
              mlx_slice(&vseg_v, s->v_seg, p0, 4, p1, 4, ps, 4, stream) == 0;
    mlx_vector_array kv = mlx_vector_array_new(), vv = mlx_vector_array_new();
    if (ok) {
        mlx_vector_array_append_value(kv, s->k_main);
        mlx_vector_array_append_value(kv, kseg_v);
        mlx_vector_array_append_value(vv, s->v_main);
        mlx_vector_array_append_value(vv, vseg_v);
        mlx_array k_new = mlx_array_new(), v_new = mlx_array_new();
        ok = mlx_concatenate_axis(&k_new, kv, 2, stream) == 0 &&
             mlx_concatenate_axis(&v_new, vv, 2, stream) == 0;
        if (!ok) { mlx_array_free(k_new); mlx_array_free(v_new); }
        else {
            mlx_array_free(s->k_main); mlx_array_free(s->v_main);
            s->k_main = k_new; s->v_main = v_new;
            s->base_len += s->seg_len;
            /* Buffer del segmento riportato a costo O(SEG): il segmento svuotato viene
             * re-inizializzato a zeros(seg_cap) — così la shape resta fissa e il prossimo
             * append parte da un buffer pulito (nessuna riscrittura O(pos)). */
            mlx_array kz = mlx_array_new(), vz = mlx_array_new();
            int zsh[4] = {1, nkv, s->seg_cap, hd};
            bool zok = mlx_zeros(&kz, zsh, 4, mlx_array_dtype(s->k_main), stream) == 0 &&
                       mlx_zeros(&vz, zsh, 4, mlx_array_dtype(s->v_main), stream) == 0;
            if (zok) {
                mlx_array_free(s->k_seg); mlx_array_free(s->v_seg);
                s->k_seg = kz; s->v_seg = vz;
            } else {
                mlx_array_free(kz); mlx_array_free(vz);
                /* segmento non re-inizializzato: shape stantia, l'append sotto usa la
                 * slice fino a seg_len (0 dopo il flush) — corretto comunque */
            }
        }
    }
    mlx_vector_array_free(kv); mlx_vector_array_free(vv);
    mlx_array_free(kseg_v); mlx_array_free(vseg_v);
    if (!ok) { llm_mlx_kvseg_reset_layer(layer_idx); return false; }
    s->seg_len = 0;
    return true;
}
/* Setup/append: gestisce lo stato segmentato per questo layer e PUBBLICA le pezzi che
 * l'attention deve leggere. Ritorna false = NON usare il path segmentato (il chiamante
 * ripiega sul ramo KVSLICE esistente, mai un errore bloccante). k_rope/v_t sono i
 * tensori rope-ati del token corrente gia' in layout cache [1,H,1,D]. */
bool llm_mlx_kvseg_step(mlx_stream stream, int nkv, int hd, int layer_idx,
                               int cache_offset, int n_tokens,
                               mlx_array k_rope, mlx_array v_t,
                               mlx_array* k_cache_out, mlx_array* v_cache_out,
                               mlx_array* k_piece1, mlx_array* v_piece1,
                               mlx_array* k_piece2, mlx_array* v_piece2,
                               int* n_pieces) {
    if (k_piece1 == NULL || v_piece1 == NULL || n_pieces == NULL) return false;
    if (layer_idx < 0 || layer_idx >= MAX_KVSEG_LAYERS) return false;
    if (k_cache_out == NULL || v_cache_out == NULL) return false;
    if ((*k_cache_out).ctx == NULL || (*v_cache_out).ctx == NULL) return false;
    if (nkv <= 0 || hd <= 0 || cache_offset < 0 || n_tokens != 1) return false;
    if (mlx_array_ndim(*k_cache_out) != 4) return false;
    const mlx_dtype cdt = mlx_array_dtype(*k_cache_out);
    LlmKvSegState* s = &g_kvseg[layer_idx];
    const int seg_cap = llm_mlx_kvseg_size();

    /* Avvio: spezza la storia gia' in cache (cache_offset token REALI, la shape del
     * buffer KVSLICE e' PADDED a capacita' -- usare solo cache_offset, mai shape) in
     * base allineata al segmento + residuo:
     *   main = cache [0..base_len)      (zero-copy view)
     *   seg  = cache [base_len..offset) (zero-copy view)
     * Sotto la soglia la segmentazione non paga: fallback a KVSLICE. */
    if (!s->active) {
        /* Buffer dichiarativo deve già contenere i cache_offset token reali (capacità
         * fisica >= cache_offset), altrimenti la segmentazione non può partire. */
        if (cache_offset > (int)mlx_array_shape(*k_cache_out)[2]) return false;
        int have_len = cache_offset;
        if (have_len + n_tokens < seg_cap) return false;   /* contesto corto: KVSLICE basta */
        s->seg_cap = seg_cap;
        s->base_len = (have_len / seg_cap) * seg_cap;
        if (s->base_len < 1) return false;                 /* < seg_cap non pagherebbe */
        s->seg_len = have_len - s->base_len;
        /* Segmento: buffer a capacita' fissa, dtype della cache (MAI fp16 forzato:
         * su cache bf16/fp32 sarebbe una perdita di precisione silenziosa). */
        mlx_array kz = mlx_array_new(), vz = mlx_array_new();
        int zsh[4] = {1, nkv, s->seg_cap, hd};
        bool sok = mlx_zeros(&kz, zsh, 4, cdt, stream) == 0 &&
                   mlx_zeros(&vz, zsh, 4, cdt, stream) == 0;
        if (!sok) { mlx_array_free(kz); mlx_array_free(vz); return false; }
        /* Il residuo NON parte da 0 ma da base_len (i primi base_len token sono nel main). */
        if (s->seg_len > 0) {
            mlx_array kr = mlx_array_new(), vr = mlx_array_new();
            int p0[4] = {0, 0, s->base_len, 0}, p1[4] = {1, nkv, s->base_len + s->seg_len, hd}, ps[4] = {1, 1, 1, 1};
            sok = mlx_slice(&kr, *k_cache_out, p0, 4, p1, 4, ps, 4, stream) == 0 &&
                  mlx_slice(&vr, *v_cache_out, p0, 4, p1, 4, ps, 4, stream) == 0 &&
                  mlx_array_set(&s->k_seg, kr) == 0 && mlx_array_set(&s->v_seg, vr) == 0;
            mlx_array_free(kr); mlx_array_free(vr);
            if (!sok) { mlx_array_free(kz); mlx_array_free(vz);
                        llm_mlx_kvseg_reset_layer(layer_idx); return false; }
            mlx_array_free(kz); mlx_array_free(vz);
        } else {
            s->k_seg = kz; s->v_seg = vz;
        }
        /* Main: view zero-copy dei primi base_len token della cache. */
        mlx_array km = mlx_array_new(), vm = mlx_array_new();
        int p0[4] = {0,0,0,0}, p1[4] = {1,nkv,s->base_len,hd}, ps[4] = {1,1,1,1};
        bool mok = mlx_slice(&km, *k_cache_out, p0, 4, p1, 4, ps, 4, stream) == 0 &&
                   mlx_slice(&vm, *v_cache_out, p0, 4, p1, 4, ps, 4, stream) == 0;
        if (!mok) { mlx_array_free(km); mlx_array_free(vm);
                    llm_mlx_kvseg_reset_layer(layer_idx); return false; }
        s->k_main = km; s->v_main = vm;
        s->active = true;
    }

    /* Append del token nel segmento: concat+set su buffer PICCOLO (seg_cap) —
     * costo O(seg_cap) costante, mai O(pos). Se il segmento e' pieno, flush prima. */
    if (s->seg_len >= s->seg_cap && !llm_mlx_kvseg_flush(stream, nkv, hd, layer_idx)) return false;
    mlx_array k_base = mlx_array_new(), v_base = mlx_array_new();
    mlx_array k_buf = mlx_array_new(), v_buf = mlx_array_new();
    int b0[4] = {0,0,0,0}, b1[4] = {1,nkv,s->seg_len,hd}, bs[4] = {1,1,1,1};
    bool wok = mlx_slice(&k_base, s->k_seg, b0, 4, b1, 4, bs, 4, stream) == 0 &&
               mlx_slice(&v_base, s->v_seg, b0, 4, b1, 4, bs, 4, stream) == 0;
    if (wok) {
        mlx_vector_array kv = mlx_vector_array_new(), vv = mlx_vector_array_new();
        mlx_vector_array_append_value(kv, k_base);
        mlx_vector_array_append_value(kv, k_rope);
        mlx_vector_array_append_value(vv, v_base);
        mlx_vector_array_append_value(vv, v_t);
        wok = mlx_concatenate_axis(&k_buf, kv, 2, stream) == 0 &&
              mlx_concatenate_axis(&v_buf, vv, 2, stream) == 0;
        mlx_vector_array_free(kv); mlx_vector_array_free(vv);
    }
    mlx_array_free(k_base); mlx_array_free(v_base);
    if (wok) wok = mlx_array_set(&s->k_seg, k_buf) == 0 && mlx_array_set(&s->v_seg, v_buf) == 0;
    mlx_array_free(k_buf); mlx_array_free(v_buf);
    if (!wok) { llm_mlx_kvseg_reset_layer(layer_idx); return false; }
    s->seg_len += n_tokens;

    /* Pubblica le pezzi per l'attention:
     * 2 pezzi = main + segmento (caso normale) -> il chiamante fa DUE SDPA e il merge
     * softmax; 1 pezzo = solo main (segmento appena svuotato dal flush).
     * ATTENZIONE ownership: TUTTE le pezzi pubblicate sono SLICE zero-copy
     * (mlx_slice), MAI alias diretti degli array interni (s->k_main/s->k_seg),
     * cosi' il chiamante puo' liberarle senza toccare lo stato segmentato. */
    int p0[4] = {0,0,0,0}, ps[4] = {1,1,1,1};
    if (s->base_len > 0) {
        if (s->seg_len > 0) {
            int p1s[4] = {1,nkv,s->seg_len,hd};
            if (mlx_slice(k_piece1, s->k_seg, p0, 4, p1s, 4, ps, 4, stream) != 0 ||
                mlx_slice(v_piece1, s->v_seg, p0, 4, p1s, 4, ps, 4, stream) != 0) {
                llm_mlx_kvseg_reset_layer(layer_idx); return false;
            }
            int p1m[4] = {1,nkv,s->base_len,hd};
            if (mlx_slice(k_piece2, s->k_main, p0, 4, p1m, 4, ps, 4, stream) != 0 ||
                mlx_slice(v_piece2, s->v_main, p0, 4, p1m, 4, ps, 4, stream) != 0) {
                mlx_array_free(*k_piece1); mlx_array_free(*v_piece1);
                llm_mlx_kvseg_reset_layer(layer_idx); return false;
            }
            *n_pieces = 2;
        } else {
            int p1m[4] = {1,nkv,s->base_len,hd};
            if (mlx_slice(k_piece1, s->k_main, p0, 4, p1m, 4, ps, 4, stream) != 0 ||
                mlx_slice(v_piece1, s->v_main, p0, 4, p1m, 4, ps, 4, stream) != 0) {
                llm_mlx_kvseg_reset_layer(layer_idx); return false;
            }
            *n_pieces = 1;
        }
        return true;
    }
    /* Difensivo (non raggiungibile per costruzione: all'avvio si richiede
     * cache_offset >= seg_cap quindi base_len >= seg_cap > 0): se mai un flush
     * avesse svuotato anche il main, reset e fallback al KVSLICE (mai un errore). */
    llm_mlx_kvseg_reset_layer(layer_idx);
    return false;
}
/* LLM_KVSEG fine — la logica step/flush e' piu' sotto, vicino al ramo KVSLICE che la usa. */

/* Bug P0 (2026-09-03): GDN eval per-layer. Default = NO eval (stati lazy, materializzati
 * dall' eval a fine token/chunk). LLM_GDN_EVAL=1 forza il vecchio comportamento. Letto una
 * sola volta (mai getenv nel loop per-layer). */
static int g_gdn_eval_state = -1;
bool llm_mlx_gdn_eval_forced(void) {
    if (g_gdn_eval_state < 0) g_gdn_eval_state = (llm_ai_mlx_opt_env("LLM_GDN_EVAL") != NULL) ? 1 : 0;
    return g_gdn_eval_state == 1;
}

/* paged-KV step 3 (2026-09-03, sperimentale): decode dense con buffer a capacita' + slice_update.
 * LLM_KVSLICE=1 (default OFF). Se non vince la misura resta OFF. */
static int g_kvslice_state = -1;
bool llm_mlx_kvslice_enabled(void) {
    if (g_kvslice_state < 0) g_kvslice_state = (llm_ai_mlx_opt_env("LLM_KVSLICE") != NULL) ? 1 : 0;
    return g_kvslice_state == 1;
}

/* KV fp16 (task #61, 2026-09-10): il decode denso a contesto lungo e' dominato dal
 * Concatenate O(pos) della cache KV — misurato 256,53 di 261,64 MB/token @4112 su Llama
 * (98% del traffico), e cresce linearmente colla posizione (4,53 MB @80). Su Llama/Bonsai
 * l'attivazione e' fp32 (task #5), quindi la cache KV e' fp32 = 2x i byte del KV fp16/bf16
 * di rapidMLX, sia nella copia per-step del concat sia nella lettura di SDPA. LLM_KVF16
 * IMMAGAZZINA k/v in fp16: dimezza il termine dominante. NON e' bit-parita' col path fp32
 * (e' un trade numerico come bf16), ma l'SDPA gia' lavora in fp16 internamente, quindi
 * l'errore e' l'arrotondamento k/v a fp16.
 * #61 PROMOVED default-ON (2026-09-10): parita' greedy IDS (temp=0) su TUTTI i 4 blocchi
 * f16-KV cablati — llama@7355, qwen3_moe@2922 (via model_type "qwen3"), qwen2@3460,
 * glm4_moe@5627 — + gate score teacher-forcing su testo reale (512 posizioni AGENTS.md:
 * prob/acc/argmax IDENTICI, top1 225/512 in entrambe le braccia). Speed-up misurato §X.
 * Opt-out (A/B e rollback): LLM_KVF16=0 (o "off") riporterà la cache KV a fp32. */
static int g_kvf16_state = -1;
bool llm_mlx_kv_f16_enabled(void) {
    if (g_kvf16_state < 0) {
        const char* e = llm_ai_mlx_opt_env("LLM_KVF16");
        g_kvf16_state = (e != NULL && (strcmp(e, "0") == 0 || strcmp(e, "off") == 0)) ? 0 : 1;
    }
    return g_kvf16_state == 1;
}
/* LLM_KVF16 esteso a gpt-oss (2026-09-11): fp16 SOLO sui layer FULL. Parita' IDS greedy
 * VERIFICATA su gpt-oss-20b (fp32==fp16, 16/16). Velocita' NON concludente a load 19-21
 * (una finestra dava +39%, tre il contrario) -> GATE SEPARATO, default OFF: non si attiva
 * finche' un A/B a finestra idle non lo promuove. */
static int g_kvf16_gptoss_state = -1;
bool llm_mlx_kv_f16_gptoss_enabled(void) {
    if (g_kvf16_gptoss_state < 0) g_kvf16_gptoss_state = (llm_ai_mlx_opt_env("LLM_KVF16_GPTOSS") != NULL) ? 1 : 0;
    return g_kvf16_gptoss_state == 1;
}

/* G1 (piano antigravity 2026-09-08, collo "dispatch-count" di §3.2 Gap-Closure):
 * un solo kernel Metal fonde reshape+transpose+RoPE di Q/K e transpose di V nel
 * decode a 1 token (8 dispatch -> 1). LLM_FUSE_ROPE=1 attiva, DEFAULT OFF finche'
 * non vince la misura. LLM_FUSE_ROPE_VERIFY=1 calcola ENTRAMBI i risultati, stampa
 * il max_abs_diff su stderr e USA il path originale -- zero rischio, serve a
 * validare la correttezza (formula RoPE + layout [B,H,L,D]) prima di fidarsi.
 * Fallback silenzioso su ANY condizione non soddisfatta: mai un errore bloccante. */
static int g_fuse_rope_state = -1;
bool llm_mlx_fuse_rope_enabled(void) {
    if (g_llm_mlx_train_composable) return false;   /* training: no kernel Metal (no vjp) */
    if (g_fuse_rope_state < 0) {
        /* DEFAULT ON (2026-10-01). Misura deterministica decode llama-1B: 364 -> 300 nodi/token
         * (-64, -17.6%) a `MB_in` invariato (pura riduzione di launch); `LLM_FUSE_ROPE_VERIFY`
         * max_abs_diff q=k=v=0 (bit-exact). Opt-out: LLM_FUSE_ROPE=0. */
        const char* e = llm_ai_mlx_opt_env("LLM_FUSE_ROPE");
        g_fuse_rope_state = (e != NULL && e[0] == '0') ? 0 : 1;
    }
    return g_fuse_rope_state == 1;
}
static int g_fuse_rope_verify_state = -1;
bool llm_mlx_fuse_rope_verify_enabled(void) {
    if (g_fuse_rope_verify_state < 0) g_fuse_rope_verify_state = (llm_ai_mlx_opt_env("LLM_FUSE_ROPE_VERIFY") != NULL) ? 1 : 0;
    return g_fuse_rope_verify_state == 1;
}

/* Kernel Metal custom per il decode SDPA a head_dim fuori dal fused MLX. I 10 layer
 * FULL di gemma4 (head_dim=512) ripiegano sul percorso NAIVE di MLX (should_use_naive:
 * vector<=256 / flash-full<=128) che MATERIALIZZA repeat_kv [B,H,N,512] e i punteggi
 * -> crollo del decode (§V/§18bis). LLM_GEMMA_FLASH_DECODE=1 rotta quel SDPA (solo
 * full-layer, n_tokens==1) sul kernel vettoriale single-pass che evita la
 * materializzazione. DEFAULT OFF finche' non vince la misura.
 * LLM_GEMMA_FLASH_DECODE_VERIFY=1 calcola ENTRAMBI (kernel custom + naive MLX), stampa
 * max_abs_diff su stderr e USA il naive: gate di correttezza (§7 parita' prima della
 * velocita'), zero rischio. Fallback silenzioso su ANY condizione non soddisfatta. */
static int g_flash_decode_state = -1;
bool llm_mlx_flash_decode_enabled(void) {
    if (g_llm_mlx_train_composable) return false;   /* training: no kernel Metal (no vjp) */
    if (g_flash_decode_state < 0) g_flash_decode_state = (llm_ai_mlx_opt_env("LLM_GEMMA_FLASH_DECODE") != NULL) ? 1 : 0;
    return g_flash_decode_state == 1;
}
static int g_flash_decode_verify_state = -1;
bool llm_mlx_flash_decode_verify_enabled(void) {
    if (g_flash_decode_verify_state < 0) g_flash_decode_verify_state = (llm_ai_mlx_opt_env("LLM_GEMMA_FLASH_DECODE_VERIFY") != NULL) ? 1 : 0;
    return g_flash_decode_verify_state == 1;
}
/* LLM_GEMMA_FLASH_PREFILL (2026-09-11): prefill (n_tokens>1) dei layer FULL di gemma4 (hd=512)
 * sul kernel custom: evita la materializzazione [Lq,Lk] della naive MLX (§44.1). Default OFF. */
static int g_flash_prefill_state = -1;
bool llm_mlx_flash_prefill_enabled(void) {
    if (g_llm_mlx_train_composable) return false;   /* training: no kernel Metal (no vjp) */
    /* OPT-IN (LLM_GEMMA_FLASH_PREFILL=1). Era stato promosso a default ON il 2026-09-16 su
     * misure fatte col vecchio llm_prefill_bench (che appende piu' volte nella STESSA sessione:
     * primo tocco dei pesi + contesto crescente -> misura CONTAMINATA). Con
     * `llm_batch_scale_probe` (sessione nuova per N + warm-up) la verita' e' l'OPPOSTO:
     *   gemma-4-31b prefill 4096/8192 ms:
     *     kernel OFF + chunk 128 (default originale) = 11088 / 23609  <-- MIGLIORE
     *     kernel ON  + chunk 128                     = 14292 / 37317  (+29% / +58%!)
     *     kernel ON  + chunk 2048                    = 11544 / 34298  (+4% / +45%)
     * -> il kernel costa piu' di quanto faccia risparmiare (e' per-query SENZA key-tiling).
     * Resta OPT-IN e il chunk gemma4 torna 128 per default. La SOGLIA Nk (16384) sotto resta
     * come protezione per chi lo abilita: a Nk grande il kernel e' catastrofico (misurato). */
    if (g_flash_prefill_state < 0)
        g_flash_prefill_state = (llm_ai_mlx_opt_env("LLM_GEMMA_FLASH_PREFILL") != NULL) ? 1 : 0;
    return g_flash_prefill_state == 1;
}

/* G5 (piano antigravity): strumento di MISURA, non una feature. Il fix del sampler
 * (saltare heap+qsort del vocabolario quando nessun filtro e' attivo) e' promosso di
 * default perche' parita' IDS verificata; questo toggle riattiva il percorso LEGACY cosi'
 * il "prima" si puo' misurare con lo STESSO binario sui modelli grandi, dove il dato
 * storico non esiste. Default OFF = percorso veloce. */
static int g_sampler_legacy_state = -1;
bool llm_mlx_sampler_legacy_enabled(void) {
    if (g_sampler_legacy_state < 0) g_sampler_legacy_state = (llm_ai_mlx_opt_env("LLM_SAMPLER_LEGACY") != NULL) ? 1 : 0;
    return g_sampler_legacy_state == 1;
}
/* S1 (Todo/(open) - MLX-Decode-Composizione §11): sampling ON-DEVICE (opt-in, default OFF).
 * Evita la lettura del vocabolario in host (~512 KB/token) del path storico; le sequenze con
 * il flag ON NON sono identiche a quelle del path host (RNG diverso: mlx_random_key) — da qui
 * l'opt-in e la nota nel todo. Greedy (temperature<=0) non toccato. */
static int g_sampler_ondevice_state = -1;
bool llm_mlx_sampler_ondevice_enabled(void) {
    if (g_sampler_ondevice_state < 0) g_sampler_ondevice_state = (llm_ai_mlx_opt_env("LLM_SAMPLER_ONDEVICE") != NULL) ? 1 : 0;
    return g_sampler_ondevice_state == 1;
}

/* Kernel Metal MoE decode fuso (2026-09-08, adattato da mlxcel -- vedi
 * llm_mlx_moe_fused_kernel.h). Verificato corretto (rel_rms 0.004-0.013, atteso e coerente
 * con la fonte) su qwen3_moe/qwen3_5_moe; velocita' 0.95x-1.28x su macchina condivisa sotto
 * carico variabile (nessuna misura chiaramente peggiorativa, a differenza di LLM_KVSLICE su
 * GLM) -- promosso a DEFAULT ON su richiesta esplicita (2026-09-08): il guadagno atteso
 * (~3.5% per la fonte) e' sotto il rumore di misura qui, ma potrebbe emergere a macchina
 * fredda/scarica. Fallback sempre sicuro al path gather_qmm esistente per ogni condizione
 * non soddisfatta (n_tokens>1, bits non affine/non in {4,8}, LoRA moe_down attivo, ecc.) --
 * mai un errore bloccante. LLMENGINE_MOE_FUSE_MOE_DECODE=0 disattiva esplicitamente. */
static int g_fuse_moe_decode_state = -1;
bool llm_mlx_fuse_moe_decode_enabled(void) {
    if (g_fuse_moe_decode_state < 0) {
        const char* e = llm_ai_mlx_opt_env("LLMENGINE_MOE_FUSE_MOE_DECODE");
        g_fuse_moe_decode_state = (e != NULL && strcmp(e, "0") == 0) ? 0 : 1;
    }
    return g_fuse_moe_decode_state == 1;
}
static int g_fuse_moe_decode_verify_state = -1;
bool llm_mlx_fuse_moe_decode_verify_enabled(void) {
    if (g_fuse_moe_decode_verify_state < 0) g_fuse_moe_decode_verify_state = (llm_ai_mlx_opt_env("LLMENGINE_MOE_FUSE_MOE_DECODE_VERIFY") != NULL) ? 1 : 0;
    return g_fuse_moe_decode_verify_state == 1;
}

/* MXFP4 fused-MoE decode (gpt-oss, 2026-09-09 "procedi"): kernel DEDICATO per gli esperti
 * mxfp4 (scale uint8 e8m0, nessun bias di quant, bias lineare per-esperto + attivazione
 * clamped oml2) -- vedi llm_mlx_moe_fused_decode_mxfp4 in llm_mlx_moe_fused_kernel.h. A
 * differenza del fuse affine (promosso default-ON), questo entra DEFAULT OFF: creato e da
 * validare numericamente (LLMENGINE_MOE_FUSE_MOE_MXFP4_VERIFY=1 stampa il diff vs gather_qmm, usa il
 * risultato gather_qmm) PRIMA di qualsiasi attivazione, che richiede una finestra di misura
 * pulita. LLMENGINE_MOE_FUSE_MOE_MXFP4=1 forza il risultato del kernel; fallback silenzioso a
 * gather_qmm su qualunque condizione non soddisfatta (mai un errore bloccante). */
static int g_fuse_moe_mxfp4_state = -1;
bool llm_mlx_fuse_moe_mxfp4_enabled(void) {
    if (g_fuse_moe_mxfp4_state < 0) {
        /* DEFAULT ON (2026-09-13): kernel Metal fuso MoE MXFP4 (gpt-oss), ora BIT-FEDELE
         * all'eager (replica i round a T di gather_qmm/SwiGLU gpt-oss e il raggruppamento
         * qdot di fp_quantized.h): rel_rms=0 su ogni layer, IDS greedy identici su 256
         * token. Misurato ctx2048: +8% decode (126->134 tok/s). LLMENGINE_MOE_FUSE_MOE_MXFP4=0
         * kill-switch; _VERIFY=1 per il diff. Applicato solo a decode puro mxfp4 4/g32
         * (top-k experts), altrove fallback silenzioso a gather_qmm. */
        const char* e = llm_ai_mlx_opt_env("LLMENGINE_MOE_FUSE_MOE_MXFP4");
        g_fuse_moe_mxfp4_state = (e != NULL && strcmp(e, "0") == 0) ? 0 : 1;
    }
    return g_fuse_moe_mxfp4_state == 1;
}
static int g_fuse_moe_mxfp4_verify_state = -1;
bool llm_mlx_fuse_moe_mxfp4_verify_enabled(void) {
    if (g_fuse_moe_mxfp4_verify_state < 0) g_fuse_moe_mxfp4_verify_state = (llm_ai_mlx_opt_env("LLMENGINE_MOE_FUSE_MOE_MXFP4_VERIFY") != NULL) ? 1 : 0;
    return g_fuse_moe_mxfp4_verify_state == 1;
}
/* Leva f16-attivazione sugli esperti mxfp4 PURI (2026-09-10, probe tools/moe_mxfp4_gather_probe.py):
 * gather_qmm mxfp4 con lhs f16 e' ~1,9-2,2x piu' veloce che con lhs f32 sulla forma gpt-oss
 * (2880x2880), e NON crasha (a differenza del set_bf16_mode modello-intero, 10x piu' lento/crash).
 * Sul percorso mxfp4 PURO l'attivazione arriva f32 (g_bf16_all falso -> rms_norm non riduce),
 * quindi le 3 gather (gate/up/down) girano in f32. LLM_MXFP4_ACT_F16=1 casta la lhs delle gather
 * esperti a f16 SENZA toccare il resto del modello (e' la singola hot-spot, non il dtype globale).
 * NON e' bit-parita': e' un trade numerico come bf16/f16 (arrotonda l'attivazione a f16 prima della
 * matmul low-bit) -> da validare con gate greedy-IDS (temperatura 0) + score PRIMA di qualsiasi
 * claim di velocita'. DEFAULT OFF: finche' spento, comportamento identico a oggi.
 * REFUTATO 2026-09-17 (tools/moe_mxfp4_floor_probe.py): la "floor ~0,19 ms/call" delle forme
 * 122B/coder era un ARTEFATTO del sync per-chiamata (il probe faceva mx.eval() dentro il loop).
 * Nel path reale (grafo lazy, UN eval/token come l'engine) la stessa gather mxfp4 costa
 * ~0,021 ms/call (7,5x meno) ed e' dtype-invariante (f16~=f32): le 144 gather di Coder-Next =
 * ~3,0 ms/token a ~264 GB/s. NON c'e' una floor di dispatch da aggredire: il kernel fuso
 * mono-dispatch NON e' giustificato da questo dato. Il cast resta neutro (f16~=f32). */
static int g_mxfp4_act_f16_state = -1;
bool llm_mlx_mxfp4_act_f16_enabled(void) {
    if (g_mxfp4_act_f16_state < 0) {
        const char* e = llm_ai_mlx_opt_env("LLM_MXFP4_ACT_F16");
        g_mxfp4_act_f16_state = (e != NULL && strcmp(e, "0") != 0) ? 1 : 0;
    }
    return g_mxfp4_act_f16_state == 1;
}

/* Cast IN-PLACE della lhs di una gather esperto a f16 per la leva LLM_MXFP4_ACT_F16.
 * Ri-binda l'handle del chiamante sull'f16 se l'astype riesce; su fallimento lascia
 * l'array originale com'era (fallback fp32 silenzioso, mai bloccante). Ritorna true
 * se ha applicato il cast. Duplica la logica dei due cast inline in gpt_oss_ffn_block. */
bool llm_mlx_cast_f16_inplace(mlx_array* lhs, mlx_stream stream) {
    mlx_array h = mlx_array_new();
    if (mlx_astype(&h, *lhs, MLX_FLOAT16, stream) == 0) {
        mlx_array_free(*lhs);
        *lhs = h;
        return true;
    }
    mlx_array_free(h);
    return false;
}

/* Scalar costante per le ATTIVAZIONI, nel dtype di attivazione quando g_bf16_all e' attivo.
 * Un f32 "forte" (mlx_array_new_float32) PROMUOVE a f32 tutto il sotto-grafo a valle: su gpt-oss
 * l'SwiGLU inline (min/clip/moltiplicazioni/sigmoid) finiva f32 -> 382 AsType e ~2,2 GB/token di
 * cast sul prefill (misurato 2026-09-13). mlx_lm usa scalari deboli (float Python) e resta nel
 * dtype di attivazione. Fuori dal regime a 16 bit (g_bf16_all spento) resta f32, invariato. */
mlx_array llm_mlx_scalar_act(float v, mlx_stream stream) {
    mlx_array s = mlx_array_new_float32(v);
    if (!g_bf16_all || g_act16 == MLX_FLOAT32) return s;
    mlx_array c = mlx_array_new();
    if (mlx_astype(&c, s, g_act16, stream) == 0) { mlx_array_free(s); return c; }
    mlx_array_free(c);
    return s;
}

/* KV Metal reale (2026-09-06, sperimentale, default OFF): page-pool con buffer Metal
 * (MTLResourceStorageModeShared) + decode attention paged online-softmax split-K, invece
 * di zeri+concat+where -- vedi native/llmmetal_kv/llmmetal_kv.h e piano AiWiki
 * Proposta-614GBs-Banda-Metal-DeepDive.md §4 L4. NON sostituisce k_cache/v_cache (che
 * restano gestiti come oggi, invariati, per compatibilita' con trim/kv-quant/resto del
 * motore): scrive in un buffer Metal SEPARATO, mantenuto in lockstep, usato SOLO per
 * calcolare l'attention in alternativa a mlx_fast_scaled_dot_product_attention.
 * LLMMETAL_KVMETAL_VERIFY=1: calcola ENTRAMBI i risultati e stampa il diff su stderr, USA
 * SEMPRE il risultato normale -- zero rischio, pensato per validare la correttezza contro
 * un modello reale prima di fidarsi del path nuovo. LLMMETAL_KVMETAL=1 (senza _VERIFY): usa il
 * risultato del kernel Metal al posto della SDPA normale. Nessuna delle due variazioni ha
 * effetto se LLMMETAL_KVMETAL/_VERIFY non sono impostate (comportamento identico a prima). */
static int g_kvmetal_state = -1;
bool llm_mlx_kvmetal_enabled(void) {
    if (g_kvmetal_state < 0) g_kvmetal_state = (llm_ai_mlx_opt_env("LLMMETAL_KVMETAL") != NULL) ? 1 : 0;
    return g_kvmetal_state == 1;
}
static int g_kvmetal_verify_state = -1;
bool llm_mlx_kvmetal_verify_enabled(void) {
    if (g_kvmetal_verify_state < 0) g_kvmetal_verify_state = (llm_ai_mlx_opt_env("LLMMETAL_KVMETAL_VERIFY") != NULL) ? 1 : 0;
    return g_kvmetal_verify_state == 1;
}
/* Stato ombra della sessione LLMMETAL_KVMETAL attiva, accessibile da llama_attention_block
 * (definita PRIMA della struct LlmMlxSession piu' avanti nel file, quindi non puo'
 * dereferenziare un LlmMlxSession* qui). Sincronizzato da llm_mlx_session_create/_destroy
 * insieme alla guardia multi-sessione vera (g_kvmetal_act/_disabled, dichiarati vicino a
 * LlmMlxSession dove il tipo e' completo). g_kvmetal_id serve SOLO per confronto di identita'
 * in session_destroy (mai dereferenziato come puntatore a struct in questo scope). */
void* g_kvmetal_id = NULL;
void** g_kvmetal_ctx = NULL;   /* == sessione_attiva->kvmetal_ctx, o NULL se nessuna */
int g_kvmetal_n_layers = 0;    /* == sessione_attiva->n_layers, o 0 se nessuna */
int g_kvmetal_disabled = 0;    /* 1 = disabilitato (sessioni concorrenti), fallback normale */
/* LLMMETAL_KVMETAL_AUTH (#68, default OFF): rende il pool Metal fp16 O(1) AUTORITATIVO sul decode
 * denso: l'attention viene dalle pagine del pool e il Concatenate O(pos) della KV eager (98%
 * dei byte/decode, causa #61) viene SOPPRESSO per i layer il cui pool e' stato popolato dal
 * prefill. La guardia `cache_offset < kv_max_tokens` rende anche il trim eager un no-op, quindi
 * pool ed eager attendono sulle STESSI posizioni => comportamento identico salvo i numeri del
 * kernel (online-softmax fp16 vs SDPA: ~1e-4, validato da greedy IDS + score, non bit-parita).
 * CONTRATT per chi lo attiva: LLM_KV_MAX_TOKENS >= lunghezza totale della sessione (il pool non
 * evince). Implica l'allocazione del pool (vedi llm_mlx_session_create). */
static int g_kvmetal_auth_state = -1;
bool llm_mlx_kvmetal_auth_enabled(void) {
    if (g_kvmetal_auth_state < 0) g_kvmetal_auth_state = (llm_ai_mlx_opt_env("LLMMETAL_KVMETAL_AUTH") != NULL) ? 1 : 0;
    return g_kvmetal_auth_state == 1;
}
/* Bit per layer: 1 se il pool di quel layer e' stato popolato con successo dal prefill
 * (llm_mlx_kvmetal_populate_batch ha scritto TUTTI i token del batch). Allocato in
 * session_create insieme a g_kvmetal_ctx, liberato in session_destroy. NULL = nessuna sessione
 * kvmetal attiva -> auth mai pronto (fail-closed: si usa sempre l'eager concat). */
char* g_kvmetal_pop_ok = NULL;
bool llm_mlx_kvmetal_auth_layer_ready(int layer_idx) {
    return g_kvmetal_pop_ok != NULL && layer_idx >= 0 && layer_idx < g_kvmetal_n_layers
           && g_kvmetal_pop_ok[layer_idx] != 0;
}

/* GEMV Q4 Metal reale (2026-09-06, sperimentale, default OFF): kernel Metal custom per la
 * GEMV quantizzata Q4 affine (staging cooperativo + accesso coalescente + simd_sum, vedi
 * native/llmmetal_gemv/llmmetal_gemv_q4.h), usato in quantized_linear_cfg (il chokepoint
 * condiviso da TUTTE le 11 famiglie di modelli per ogni proiezione quantizzata: q/k/v/o,
 * gate/up/down, lm_head) al posto di mlx_quantized_matmul, SOLO quando: quantizzazione
 * affine 4-bit (non mxfp4/8-bit/2-bit — il kernel supporta solo questo formato), decode a
 * singolo token (non prefill: il kernel e' una GEMV, non una GEMM), H multiplo di 64.
 * Formato verificato byte-per-byte contro un tensore reale (vedi llm_gemvq4_bridge.h) --
 * a differenza di LLMMETAL_KVMETAL qui non c'era da inventare un layout, solo da confermarlo.
 * LLM_GEMVQ4_VERIFY=1: calcola ENTRAMBI i risultati, stampa il diff su stderr, USA SEMPRE
 * quello normale -- stesso schema di sicurezza di LLMMETAL_KVMETAL_VERIFY. LLM_GEMVQ4=1 (senza
 * _VERIFY): usa il risultato del kernel Metal. Cache dei contesti Metal per tensore di peso
 * (chiave: puntatore ai dati grezzi quantizzati, stabile finche' il modello e' caricato) --
 * stato ombra per lo stesso motivo di g_kvmetal_* (quantized_linear_cfg e' definita PRIMA
 * di LlmMlxSession). */
static int g_gemvq4_state = -1;
bool llm_mlx_gemvq4_enabled(void) {
    if (g_gemvq4_state < 0) g_gemvq4_state = (llm_ai_mlx_opt_env("LLM_GEMVQ4") != NULL) ? 1 : 0;
    return g_gemvq4_state == 1;
}
static int g_gemvq4_verify_state = -1;
bool llm_mlx_gemvq4_verify_enabled(void) {
    if (g_gemvq4_verify_state < 0) g_gemvq4_verify_state = (llm_ai_mlx_opt_env("LLM_GEMVQ4_VERIFY") != NULL) ? 1 : 0;
    return g_gemvq4_verify_state == 1;
}
void* g_gemvq4_id = NULL;
void** g_gemvq4_ctx = NULL;        /* array [g_gemvq4_cap], NULL se nessuna sessione attiva */
const void** g_gemvq4_key = NULL;  /* array parallelo: chiave (puntatore dati pesi), NULL = slot libero */
int g_gemvq4_cap = 0;
int g_gemvq4_disabled = 0;
#define LLM_GEMVQ4_MAX_PER_LAYER 8 /* q,k,v,o,gate,up,down + 1 margine -- vedi llm_mlx_session_create */

/* Flag di debug env cached (bug P3 2026-09-03: getenv era letto a ogni layer/token).
 * Lette UNA volta per processo: il debug non va nel percorso caldo. */
static int g_env_prof = -1, g_env_phase = -1, g_env_mem = -1, g_env_mixtral_time = -1;
/* Barrier reduction (Rapid-MLX #2165 §Track 3): permette di misurare l'effetto di NON
 * svuotare la cache allocator MLX (mlx_clear_cache). Default OFF = comportamento invariato
 * (replica mlx_lm). LLM_CLEAR_CACHE_EVERY=N spazia invece che rimuovere del tutto. */
bool llm_no_clear_cache(void) {
    static int v = -1;
    if (v < 0) v = (llm_ai_mlx_opt_env("LLM_NO_CLEAR_CACHE") != NULL) ? 1 : 0;
    return v == 1;
}
static int llm_clear_cache_every(void) {
    static int v = -2;
    if (v == -2) { const char* e = llm_ai_mlx_opt_env("LLM_CLEAR_CACHE_EVERY"); v = e ? atoi(e) : 0; }
    return v;
}
bool llm_env_prof(void)      { if (g_env_prof < 0)      g_env_prof = (llm_ai_mlx_opt_env("LLM_PROF") != NULL) ? 1 : 0; return g_env_prof == 1; }
bool llm_env_phase(void)     { if (g_env_phase < 0)     g_env_phase = (llm_ai_mlx_opt_env("LLM_PHASE") != NULL) ? 1 : 0; return g_env_phase == 1; }
bool llm_env_memprof(void)   { if (g_env_mem < 0)       g_env_mem = (llm_ai_mlx_opt_env("LLM_MEMPROF") != NULL) ? 1 : 0; return g_env_mem == 1; }
bool llm_env_mixtral_time(void) { if (g_env_mixtral_time < 0) g_env_mixtral_time = (getenv("LLM_MIXTRAL_TIME") != NULL) ? 1 : 0; return g_env_mixtral_time == 1; }

/* Cache GENERICA per le flag d'ambiente lette nel percorso caldo (per-layer / per-token):
 * getenv cammina `environ` a ogni chiamata (~50 strcmp) ed era visibile nel profilo `sample`
 * del decode (27 frame __findenv). Chiave = stringa: prima per pointer (i literal sono
 * internati dal compilatore), poi per strcmp. Letta UNA volta per processo, come le g_env_*.
 * NON usare per env il cui valore-stringa serve (path di debug): quelle restano getenv diretto. */
bool llm_env_isset(const char* name) {
    static const char* keys[128];
    static unsigned char vals[128];
    static int n = 0;
    for (int i = 0; i < n; i++) if (keys[i] == name || strcmp(keys[i], name) == 0) return vals[i] != 0;
    bool v = (getenv(name) != NULL);
    if (n < 128) { keys[n] = name; vals[n] = (unsigned char)v; n++; }
    return v;
}

/* Come llm_env_isset ma ritorna il VALORE stringa (path di debug letti per-layer). */
const char* llm_env_get(const char* name) {
    static const char* keys[64];
    static const char* vals[64];
    static int n = 0;
    for (int i = 0; i < n; i++) if (keys[i] == name || strcmp(keys[i], name) == 0) return vals[i];
    const char* v = getenv(name);
    if (n < 64) { keys[n] = name; vals[n] = v; n++; }
    return v;
}

/* Capacita' del buffer KV statico per il decode: multipli di 256 >= needed (min 16).
 * Il buffer non va mai sotto il numero di token da tenere, e cresce a passi di
 * 256 come mlx_lm KV (cache.py) invece che per doubling: con SliceUpdate la
 * copia piena del pool a ogni token costa O(cap), quindi tenere cap ~ pos
 * (invece di cap <= 2*pos) dimezza la scrittura KV per token nel path
 * LLM_KVSLICE a contesto lungo (dense-long). */
int llm_mlx_kv_cap_for(int needed) {
    if (needed <= 16) return 16;
    int c = 256;
    while (c < needed && c < (1 << 28)) c += 256;
    return c;
}

/* RMLX10 Fase A: normalizza la cache di sessione da buffer statici PADDED
 * (shape2 > cache_offset) a cache eager compatta (lunghezza = cache_offset)
 * PRIMA di un forward batch (prefill/append/chunked). Senza questo, un append
 * dopo N step di decode sotto LLM_FIXKV concatenerebbe SOPRA il padding e
 * romperebbe causalita'/parita'. Nessun effetto se il path fixkv non e' mai
 * stato usato (shape2 == cache_offset, o cache non 4D). */
void llm_mlx_kv_compact_prefix(int n_layers, int cache_offset, mlx_stream stream,
                                      mlx_array* k_cache, mlx_array* v_cache) {
    if (llm_ai_mlx_opt_env("LLM_NO_KVCOMPACT")) return;   /* A/B diagnostica (bug allineamento W8A8) */
    /* FIX 2026-09-19 (bug multi-append / multi-turn): il padding della KV al multiplo di 256
     * a fine append_impl (riga ~19408) e' INCONDIZIONATO, ma questa compaction era gated dietro
     * LLM_FIXKV/LLM_KVSLICE. Con i flag OFF (default) un SECONDO session_append operava su una
     * cache di shape2=256 con soli `cache_offset` token reali: il concat della cache padded e il
     * mask causale di MLX (offset = kL-qL) facevano attendere il modello su slot di ZERO padding
     * -> risposta degenere (es. Llama-3.2 "**.ai**" invece di "Roma."). Il C fa un solo append
     * (full prompt) e non era affetto; il C# a sessione appende system+user in due chiamate.
     * La compaction e' chiamata SOLO da forward_tokens_batch (mai dal decode, che usa
     * forward_one_token): rimuovere il gate non aggiunge costo per-token; il loop sotto e'
     * comunque un no-op quando non c'e' padding (shape2 <= cache_offset). */
    if (cache_offset < 0) return;
    for (int l = 0; l < n_layers; l++) {
        /* ctx==NULL = slot di cache mai riempito (mlx_array_new() iniziale):
         * NON chiamare mlx_array_ndim su array vuoti (mlx-c lancia "expected a
         * non-empty mlx_array" dentro mlx_array_get_). */
        if (k_cache[l].ctx == NULL || v_cache[l].ctx == NULL) continue;
        if (mlx_array_ndim(k_cache[l]) != 4 || mlx_array_ndim(v_cache[l]) != 4) continue;
        const int* ksh = mlx_array_shape(k_cache[l]);
        const int* vsh = mlx_array_shape(v_cache[l]);
        if (ksh == NULL || vsh == NULL) continue;
        if (ksh[2] <= cache_offset || vsh[2] <= cache_offset) continue;
        int start[4] = {0, 0, 0, 0};
        int kstop[4] = {ksh[0], ksh[1], cache_offset, ksh[3]};
        int vstop[4] = {vsh[0], vsh[1], cache_offset, vsh[3]};
        int stride[4] = {1, 1, 1, 1};
        mlx_array kt = mlx_array_new(), vt = mlx_array_new();
        bool okc = mlx_slice(&kt, k_cache[l], start, 4, kstop, 4, stride, 4, stream) == 0;
        okc = okc && mlx_slice(&vt, v_cache[l], start, 4, vstop, 4, stride, 4, stream) == 0;
        if (okc) {
            mlx_array_set(&k_cache[l], kt);
            mlx_array_set(&v_cache[l], vt);
        }
        mlx_array_free(kt);
        mlx_array_free(vt);
    }
}

/* LLM_GEMVQ4 (sperimentale, default OFF): tenta la GEMV Q4 Metal per un passo di decode a
 * singolo token, con fallback (return false, *res non toccato) su qualunque condizione non
 * soddisfatta -- mai un errore. Scope: SOLO quantizzazione affine 4-bit (non mxfp4/8-bit/
 * 2-bit), SOLO decode (un token: mlx_array_size(x)==H, non prefill multi-token -- il kernel
 * e' una GEMV, non una GEMM), H multiplo di 64. Cache dei contesti Metal per tensore di
 * peso, chiave = puntatore ai dati grezzi di `w` dopo eval (stabile finche' il tensore
 * resta caricato -- niente ricarica dei pesi ad ogni token). */
static bool llm_mlx_gemvq4_try(mlx_array* res, mlx_array x, mlx_array w, mlx_array scales, mlx_array biases,
                                int e_group, int e_bits, const char* e_mode, mlx_stream stream) {
    if (!(llm_mlx_gemvq4_enabled() || llm_mlx_gemvq4_verify_enabled())) return false;
    /* LLM_GEMVQ4_DEBUG=1: stampa UNA volta i valori che decidono l'eleggibilita'. Serve a
     * distinguere "il kernel ha perso" da "il kernel non e' mai partito" -- senza questo non
     * c'e' modo di saperlo (un A/B identico al bit puo' voler dire entrambe le cose). */
    {
        static int dbg = -1, printed = 0;
        if (dbg < 0) dbg = (getenv("LLM_GEMVQ4_DEBUG") != NULL) ? 1 : 0;
        if (dbg == 1 && !printed) {
            printed = 1;
            const int xnd0 = mlx_array_ndim(x);
            const int* xs0 = xnd0 >= 1 ? mlx_array_shape(x) : NULL;
            fprintf(stderr, "[gemvq4-debug] ctx=%p disabled=%d bits=%d mode=%s scales=%p biases=%p "
                            "xnd=%d xlast=%d xsize=%ld wnd=%d wdt=%d M=%d gsz=%d\n",
                    (void*)g_gemvq4_ctx, g_gemvq4_disabled, e_bits, e_mode ? e_mode : "(null)",
                    (void*)scales.ctx, (void*)biases.ctx, xnd0, xs0 ? xs0[xnd0 - 1] : -1,
                    (long)mlx_array_size(x), mlx_array_ndim(w),
                    mlx_array_ndim(w) >= 1 ? (int)mlx_array_dtype(w) : -1,
                    mlx_array_ndim(w) >= 1 ? (int)mlx_array_shape(w)[0] : -1, e_group);
        }
    }
    if (g_gemvq4_disabled || !g_gemvq4_ctx) return false;
    if (e_bits != 4 || !e_mode || strcmp(e_mode, "affine") != 0) return false;
    if (scales.ctx == NULL || biases.ctx == NULL) return false; /* affine reale: presenti sempre */
    int xnd = mlx_array_ndim(x);
    if (xnd < 1) return false;
    const int* xs = mlx_array_shape(x);
    uint32_t H = (uint32_t)xs[xnd - 1];
    if (H == 0 || H % 64 != 0) return false;
    if ((uint32_t)mlx_array_size(x) != H) return false; /* solo decode a 1 token, non prefill */
    if (mlx_array_ndim(w) < 1 || mlx_array_dtype(w) != MLX_UINT32) return false;
    uint32_t M = (uint32_t)mlx_array_shape(w)[0];
    if (M == 0) return false;
    uint32_t group_size = (uint32_t)e_group;
    if (group_size == 0 || H % group_size != 0) return false;

    if (mlx_array_eval(w) != 0) return false;
    const void* key = (const void*)mlx_array_data_uint32(w);
    if (!key) return false;

    LlmGemvQ4Ctx* ctx = NULL;
    for (int i = 0; i < g_gemvq4_cap; i++) if (g_gemvq4_key[i] == key) { ctx = (LlmGemvQ4Ctx*)g_gemvq4_ctx[i]; break; }
    if (!ctx) {
        /* prima volta per questo tensore: converte scale/bias (bf16->fp16 se serve -- i
         * modelli reali le hanno spesso in bf16, non fp16) e crea il contesto Metal. Le
         * chiamate successive per lo STESSO tensore saltano questo blocco (lookup sopra). */
        mlx_array scf = mlx_array_new(), bcf = mlx_array_new();
        bool ok = (mlx_array_dtype(scales) == MLX_FLOAT16) ? mlx_array_set(&scf, scales) == 0
                                                            : mlx_astype(&scf, scales, MLX_FLOAT16, stream) == 0;
        ok = ok && ((mlx_array_dtype(biases) == MLX_FLOAT16) ? mlx_array_set(&bcf, biases) == 0
                                                              : mlx_astype(&bcf, biases, MLX_FLOAT16, stream) == 0);
        if (ok) ok = mlx_array_eval(scf) == 0 && mlx_array_eval(bcf) == 0;
        const uint16_t* scd = ok ? (const uint16_t*)mlx_array_data_float16(scf) : NULL;
        const uint16_t* bid = ok ? (const uint16_t*)mlx_array_data_float16(bcf) : NULL;
        if (ok && scd && bid) {
            int free_slot = -1;
            for (int i = 0; i < g_gemvq4_cap; i++) if (g_gemvq4_key[i] == NULL) { free_slot = i; break; }
            if (free_slot >= 0) {
                LlmGemvQ4Ctx* nc = llm_gemvq4_create();
                if (nc && llm_gemvq4_load_weights(nc, (const uint8_t*)key, scd, bid, H, M, group_size) == 0) {
                    g_gemvq4_ctx[free_slot] = nc; g_gemvq4_key[free_slot] = key; ctx = nc;
                } else if (nc) { llm_gemvq4_destroy(nc); }
            }
        }
        mlx_array_free(scf); mlx_array_free(bcf);
    }
    if (!ctx) return false;

    mlx_array xf = mlx_array_new(), xc = mlx_array_new();
    bool ok = (mlx_array_dtype(x) == MLX_FLOAT16) ? mlx_array_set(&xf, x) == 0
                                                   : mlx_astype(&xf, x, MLX_FLOAT16, stream) == 0;
    ok = ok && mlx_contiguous(&xc, xf, false, stream) == 0;
    if (ok) ok = mlx_array_eval(xc) == 0;
    const uint16_t* xd = ok ? (const uint16_t*)mlx_array_data_float16(xc) : NULL;
    bool result = false;
    uint16_t* out_raw = NULL;
    if (ok && xd) {
        out_raw = (uint16_t*)malloc((size_t)M * sizeof(uint16_t));
        if (out_raw && llm_gemvq4_run(ctx, xd, out_raw) == 0) {
            int oshape[8];
            for (int i = 0; i < xnd && i < 8; i++) oshape[i] = xs[i];
            oshape[xnd - 1] = (int)M;
            mlx_array raw = mlx_array_new_data(out_raw, oshape, xnd, MLX_FLOAT16);
            mlx_dtype xdt0 = mlx_array_dtype(x);
            result = (xdt0 == MLX_FLOAT16) ? mlx_array_set(res, raw) == 0
                                            : mlx_astype(res, raw, xdt0, stream) == 0;
            mlx_array_free(raw);
        }
    }
    free(out_raw);
    mlx_array_free(xf); mlx_array_free(xc);
    return result;
}




/* ── P10 (oMLX oQ A8): GEMM int8-activation × Q4/Q5 sui TENSOR UNIT M5 (NAX) nel PREFILL ──
 * Intercetta il chokepoint quantized_linear_cfg quando: mode affine, bits 4/5, group 64,
 * prefill (M>1, M%16==0), N%64==0, K%64==0. Kernel: Stage-A (quant. attivazione) + GEMM NAX
 * (port verbatim di oMLX, validato; vedi native/llmmetal_oq_a8/ e Knowledge/oQ-A8-...).
 * LLMMETAL_OQ_A8=1 usa il kernel; LLMMETAL_OQ_A8_VERIFY=1 calcola ENTRAMBI e usa SEMPRE il normale.
 * Le metallib si indicano con LLMMETAL_OQ_A8_QMM_LIB / LLMMETAL_OQ_A8_QUANT_LIB (build_metallibs.sh).
 * Default OFF; qualunque ineleggibilita' -> fallback mlx_quantized_matmul (nessun rischio). */
static int g_oq_a8_state = -1, g_oq_a8_verify_state = -1, g_oq_a8_failed = 0;
static int g_oq_a8_prim_state = -1;   /* primitive MLX in-graph (default): UNA submit, zero sync, come oMLX */
static bool llm_mlx_oq_a8_prim_enabled(void) {
    /* DEFAULT ON (come oMLX): il prim encoda nel command encoder del GRAFO
     * (metal::get_command_encoder) -> una sola submit, nessuna sync per matmul.
     * Il ramo EAGER (llmmetal_oq_a8_forward) ha invece una coda Metal PROPRIA con
     * commit+waitUntilCompleted PER PROIEZIONE: centinaia di sync GPU per prefill.
     * Misurato (prefill 2048 512, bf16, quiet): 27B eager 642/-28% vs prim 1095/+22%;
     * 8B eager 2046/-37% vs prim 4554/+40%. Kill-switch: LLMMETAL_OQ_A8_OFF_PRIM=1 (torna eager). */
    if (g_oq_a8_prim_state < 0) g_oq_a8_prim_state = (llm_ai_mlx_opt_env("LLMMETAL_OQ_A8_OFF_PRIM") != NULL) ? 0 : 1;
    return g_oq_a8_prim_state == 1;
}
static LlmMetalOqA8Ctx* g_oq_a8_ctx = NULL;
static bool llm_mlx_oq_a8_enabled(void) {
    /* REVERT a OPT-IN (2026-09-16, dopo la RETTIFICA della misura). Era stato promosso a
     * default ON su un A/B fatto col vecchio llm_prefill_bench che appendeva piu' volte nella
     * STESSA sessione (primo tocco dei pesi + contesto che cresce) -> misura CONTAMINATA.
     * Con `batch_scale_probe` (sessione nuova per N + warm-up che inizializza anche il path
     * oQ A8 lazy) il quadro e' MODEL-DEPENDENT:
     *   Qwen3-8B-4bit  N=1024/2048/4096: ON/OFF = 0,72x/0,84x/0,77x -> **ON VINCE +16..28%**
     *   Llama-1B-4bit  N=16..2048:       ON/OFF = 3,4x/3,1x/3,0x/5,9x/1,9x -> **ON PERDE**
     * (l'overhead per-call del bridge eager non si ammortizza sui GEMM piu' piccoli di llama:
     * hidden 2048 vs 4096; non e' il packed-qkv, che anzi aiuta). Un default ON quindi
     * REGREDIVA i modelli tipo llama di 3-6x -> **opt-in** (LLMMETAL_OQ_A8=1) finche' non c'e' un
     * gate per-famiglia misurato. Kill-switch non piu' necessario.
     * QUALITA' (invariata, misurata): 3,52% argmax divergenti (gate teacher-forcing su 256
     * posizioni; controllo negativo deterministico). FAIL-OPEN garantito. */
    if (g_oq_a8_state < 0) {
        const char* e = llm_ai_mlx_opt_env("LLMMETAL_OQ_A8");
        g_oq_a8_state = (e != NULL && strcmp(e, "0") != 0) ? 1 : 0;   /* =0 = kill-switch */
    }
    return g_oq_a8_state == 1;
}
/* Gate PER-FAMIGLIA (2026-09-17): l'A8 era opt-in perche' un default ON puro regrediva i
 * modelli piccoli (Llama-1B hidden 2048: 3-6x piu' lento; l'overhead per-call del bridge non
 * si ammortizza sui GEMM piccoli). Il discriminante misurato e' la dimensione della proiezione:
 * K>=4096 vince (Qwen3-8B hidden 4096: +16..28%; Qwen3.6-27B hidden 5120/h17408: +10%),
 * K<4096 perde (Llama-1B K=2048). Quindi: AUTO quando la env NON e' impostata -> A8 se K>=4096
 * (il gate e' applicato al chiamante, che conosce K). LLMMETAL_OQ_A8=1 forza su ogni K; =0 disabilita. */
static bool llm_mlx_oq_a8_auto_enabled(void) { return llm_ai_mlx_opt_env("LLMMETAL_OQ_A8") == NULL; }
/* A8 AUTO cablato (2026-09-18, richiesta utente: "attivato automaticamente in base al modello").
 * Regola MISURATA (K = dim di input della proiezione):
 *   K >= 4096 -> vince (Qwen3-8B K=4096 +16..28%; Qwen3.6-27B K=5120 +10%; gemma-4-31b +58%)
 *   K <  4096 -> perde (Llama-1B K=2048: 3-6x piu' lento)
 * Quindi: LLMMETAL_OQ_A8=1 forza su ogni K; LLMMETAL_OQ_A8=0 disabilita; env assente -> auto (K>=4096).
 * NOTA (2026-09-18): la GUARDIA multi-chunk e' stata RIMOSSA — il difetto multi-chunk era la cache
 * prep scT/biT riallocata ad ogni call dentro il primitive (fix in llmmetal_oq_a8_bridge.mm), non il
 * chunking in se'. Verificato unit-first (probe: call2 == call1) e sul modello (prompt 8848 -> OK). */
static bool llm_mlx_oq_a8_active_for(int K) {
    (void)K;
    /* GUARD (2026-09-19): la combo bf16-mode OFF + W8A8 e' ROTTA (il modello resta un token
     * indietro: score arm=off 8-16/24 vs arm=both 16/16). Con bf16 ON (il default reale) la
     * qualita' e' identica a MLX (score 256 pos: 89,8% vs 90,6%). Qui si disabilita l'A8
     * quando il modo bf16 e' spento: fail-open a MLX, mai output sbagliato. */
    if (!g_bf16_all) return false;
    if (llm_mlx_oq_a8_enabled()) return true;          /* LLMMETAL_OQ_A8=1 */
    if (!llm_mlx_oq_a8_auto_enabled()) return false;   /* LLMMETAL_OQ_A8=0 */
    /* AUTO DISATTIVATO (2026-09-19 sera): l'indagine qualita' ha trovato un BUG REALE nel
     * path W8A8 del prefill — il modello resta UN TOKEN INDIETRO (score su conteggio:
     * argmax(i) == ref(i-1); MLX 24/24 vs W8A8 17/24 con prefill-only e 0/24 con decode
     * incluso; riproducibile anche con prefill MONO-chunk e con lm_head escluso). Non e'
     * precisione int8: e' disallineamento di stato. Finche' non e' risolto il default e'
     * QUALITY-SAFE (MLX); W8A8 resta opt-in esplicito con LLMMETAL_OQ_A8=1 (per chi indaga).
     * Storia: AUTO era ON dal 2026-09-18 (guadagni +44..+69% sul prefill) ma la qualita' non
     * era mai stata validata su testo reale — vedi Knowledge/oQ-A8-...-M5-NAX.md §6quinquies.4bis. */
    return false;
}
static bool llm_mlx_oq_a8_verify_enabled(void) {
    if (g_oq_a8_verify_state < 0) g_oq_a8_verify_state = (llm_ai_mlx_opt_env("LLMMETAL_OQ_A8_VERIFY") != NULL) ? 1 : 0;
    return g_oq_a8_verify_state == 1;
}
/* Directory della dylib che contiene questo codice: le risorse oQ A8 (.metallib, versionate in
 * Resources/ e copiate nell'output della build) atterrano accanto alla dylib. */
static const char* llm_mlx_self_dir(void) {
    static char dir[1024]; static int done = 0;
    if (!done) {
        done = 1; dir[0] = '\0';
        Dl_info info;
        if (dladdr((void*)&llm_ai_mlx_free_model, &info) && info.dli_fname != NULL) {
            const char* p = info.dli_fname;
            const char* slash = strrchr(p, '/');
            if (slash != NULL) {
                size_t n = (size_t)(slash - p);
                if (n < sizeof(dir)) { memcpy(dir, p, n); dir[n] = '\0'; }
            }
        }
    }
    return dir[0] != '\0' ? dir : NULL;
}
LlmMetalOqA8Ctx* llm_mlx_oq_a8_ctx(void) {
    if (g_oq_a8_ctx) return g_oq_a8_ctx;
    if (g_oq_a8_failed) return NULL;
    const char* q = llm_ai_mlx_opt_env("LLMMETAL_OQ_A8_QMM_LIB");
    const char* a = llm_ai_mlx_opt_env("LLMMETAL_OQ_A8_QUANT_LIB");
    char qbuf[1200], abuf[1200];
    if (q == NULL || a == NULL) {
        /* Default (nessuna env): le metallib accanto alla dylib. Era il blocco alla promozione
         * (prima erano solo artefatti non versionati, trovabili unicamente via env). */
        const char* d = llm_mlx_self_dir();
        if (d == NULL) { g_oq_a8_failed = 1; return NULL; }
        if (q == NULL) { snprintf(qbuf, sizeof(qbuf), "%s/oq_a8_qmm_nax.metallib", d); q = qbuf; }
        if (a == NULL) { snprintf(abuf, sizeof(abuf), "%s/oq_a8_quantize.metallib", d); a = abuf; }
    }
    g_oq_a8_ctx = llmmetal_oq_a8_create(q, a);
    if (!g_oq_a8_ctx) g_oq_a8_failed = 1;
    return g_oq_a8_ctx;
}

/* Override RUNTIME del gate M dell'OQ-A8 (>=0 vince sull'env LLMMETAL_OQ_A8_MINTOK).
 * Impostato dal runtime DFlash2 alla creazione: nel path speculativo AR e verify DEVONO
 * usare la stessa numerica (altrimenti l'acceptance crolla), e la configurazione misurata
 * migliore e' W8A8 a M>=1 (verify 27B-8bit 114,0 -> 97,3 ms/round; AR -10%; spec netto +). */
int g_oq_a8_minm_override = -1;

static bool llm_mlx_oq_a8_try(mlx_array* res, mlx_array x, mlx_array w, mlx_array scales, mlx_array biases,
                              int e_group, int e_bits, const char* e_mode, mlx_stream stream) {
    if (!(llm_mlx_oq_a8_enabled() || llm_mlx_oq_a8_verify_enabled())) return false;
    if (e_group != 64 || (e_bits != 4 && e_bits != 5 && e_bits != 8)) return false;
    if (!e_mode || strcmp(e_mode, "affine") != 0) return false;
    if (scales.ctx == NULL || biases.ctx == NULL) return false;
    if (mlx_array_ndim(w) < 2 || mlx_array_dtype(w) != MLX_UINT32) return false;
    int xnd = mlx_array_ndim(x); if (xnd < 2) return false;
    const int* xs = mlx_array_shape(x);
    int K = xs[xnd - 1]; if (K <= 0 || (K % 64) != 0) return false;
    size_t total = mlx_array_size(x); if (total == 0 || total % (size_t)K != 0) return false;
    int M = (int)(total / (size_t)K);
    if (M < 16) return false;   /* SOLO PREFILL: tile MMA 16x32x16; il decode (M=1) resta su MLX */
    int N = (int)mlx_array_shape(w)[0]; if (N <= 0 || (N % 64) != 0) return false;
    if (N > 16384) return false;   /* esclude lm_head/embed (vocab ~128k) */
    LlmMetalOqA8Ctx* ctx = llm_mlx_oq_a8_ctx(); if (!ctx) return false;

    /* Path normale: **primitive MLX** (in-graph, una sola submit, zero sync per matmul). */
    if (!llm_mlx_oq_a8_verify_enabled()) {
        mlx_array p = llmmetal_oq_a8_prim(x, w, scales, biases, stream, e_bits);
        if (p.ctx != NULL) {
            mlx_dtype xdt0 = mlx_array_dtype(x);
            bool r2 = (xdt0 == MLX_FLOAT16) ? mlx_array_set(res, p) == 0
                                            : mlx_astype(res, p, xdt0, stream) == 0;
            mlx_array_free(p);
            return r2;
        }
        mlx_array_free(p);
        return false;
    }
    /* VERIFY: eager + confronto (sotto) */
    /* x -> f16 contiguous */
    mlx_array xf = mlx_array_new(), xc = mlx_array_new();
    bool ok = (mlx_array_dtype(x) == MLX_FLOAT16) ? mlx_array_set(&xf, x) == 0
                                                   : mlx_astype(&xf, x, MLX_FLOAT16, stream) == 0;
    ok = ok && mlx_contiguous(&xc, xf, false, stream) == 0;
    /* out f16 [.., N] (zero-copy: il bridge scrive nel suo buffer Metal) */
    mlx_array out = mlx_array_new();
    if (ok) { int osh[8]; for (int i2 = 0; i2 < xnd && i2 < 8; i2++) osh[i2] = xs[i2]; osh[xnd - 1] = N;
              ok = mlx_zeros(&out, osh, xnd, MLX_FLOAT16, stream) == 0; }
    bool result = false;
    if (ok) {
        int rc = llmmetal_oq_a8_forward(xc, w, scales, biases, out, stream, e_bits);
        if (rc == 0) {
            mlx_dtype xdt0 = mlx_array_dtype(x);
            result = (xdt0 == MLX_FLOAT16) ? mlx_array_set(res, out) == 0
                                            : mlx_astype(res, out, xdt0, stream) == 0;
        }
    }
    mlx_array_free(out); mlx_array_free(xf); mlx_array_free(xc);
    return result;
}


/* MPP q4 via BRIDGE (metallib build-time): il kernel MPP NON funziona via mlx_fast_metal_kernel
 * (output ~0 silenzioso). x [1,T,K] -> pad a [M,K] (M minimo valido >= T), out [M,N] bf16. */
static LlmMetalMppQ4Ctx* g_mpp_q4_bridge = NULL;
static bool llm_mpp_q4_bridge_try(mlx_array* res, mlx_array x, mlx_array w, mlx_array sc, mlx_array bi,
                                  int group, int bits, int K, int N, mlx_stream s) {
    static int s_tried = 0;
    if (!s_tried) { s_tried = 1; const char* lib = llm_ai_mlx_opt_env("LLMMETAL_MPP_Q4_LIB"); if (lib) g_mpp_q4_bridge = llmmetal_mpp_q4_create(lib); }
    if (g_mpp_q4_bridge == NULL) return false;
    if ((group != 64 && group != 32) || bits != 4) return false;
    if (mlx_array_dtype(sc) != MLX_BFLOAT16 || mlx_array_dtype(bi) != MLX_BFLOAT16) return false;
    if (mlx_array_ndim(x) != 3 || mlx_array_dtype(w) != MLX_UINT32) return false;
    int T = (int)mlx_array_shape(x)[1];
    if (T < 1 || T > 64 || K % group != 0 || N % 128 != 0) return false;
    int M = (T <= 8) ? 8 : (T <= 16) ? 16 : (T <= 32) ? 32 : 64;
    mlx_array x2 = mlx_array_new();
    if (T < M) { int zsh[3]={1,M-T,K}; mlx_array z=mlx_array_new();
        if (mlx_zeros(&z,zsh,3,mlx_array_dtype(x),s)!=0){mlx_array_free(z);return false;}
        mlx_vector_array cv=mlx_vector_array_new_data((mlx_array[]){x,z},2);
        int rc=mlx_concatenate_axis(&x2,cv,1,s); mlx_vector_array_free(cv); mlx_array_free(z);
        if (rc!=0){mlx_array_free(x2);return false;} }
    else mlx_array_set(&x2,x);
    mlx_array x3=mlx_array_new(); int sh2[2]={M,K};
    if (mlx_reshape(&x3,x2,sh2,2,s)!=0){mlx_array_free(x2);return false;}
    mlx_array_free(x2);
    /* In-graph: il nodo MPP entra nel grafo MLX (nessuna sync per chiamata -> e' la via veloce,
     * come OqA8QmmPrim). L'array e' lazy: l'eval lo esegue insieme al resto del forward. */
    mlx_array out2 = llmmetal_mpp_q4_prim(x3, w, sc, bi, s, M, group);
    mlx_array_free(x3);
    if (out2.ctx == NULL) return false;
    mlx_array o3=mlx_array_new(); int sh3[3]={1,M,N};
    if (mlx_reshape(&o3,out2,sh3,3,s)!=0){mlx_array_free(out2);return false;}
    mlx_array_free(out2);
    if (T<M){ int st[3]={0,0,0},sp[3]={1,T,N},ss[3]={1,1,1}; mlx_array yt=mlx_array_new();
        if (mlx_slice(&yt,o3,st,3,sp,3,ss,3,s)==0) mlx_array_set(res,yt); else return false;
        mlx_array_free(yt); }
    else mlx_array_set(res,o3);
    mlx_array_free(o3);
    return true;
}

/* MPP q4 variante TILE (schema Splash, q4_mpp_tiles.h): somme d'ingresso condivise per il
 * termine bias + lettura pesi a stride dal layout MLX (nessun repack). E' la variante che BATTE
 * il qmm MLX (probe tools/mlx_mpp_probe/q4tile: 1.2-2.7x a M=16/24/32, GS=32 e 64, parity OK).
 * Attiva per T in [9,32] -> M=16/24/32; fuori intervallo si ricade su qmm. */
static long g_mpp_tile_miss = 0;
static bool llm_mpp_q4_tile_try(mlx_array* res, mlx_array x, mlx_array w, mlx_array sc, mlx_array bi,
                                int group, int bits, int K, int N, mlx_stream s, int T) {
    static int s_tried = 0;
    if (!s_tried) { s_tried = 1; const char* lib = llm_ai_mlx_opt_env("LLMMETAL_MPP_Q4_LIB"); if (lib) g_mpp_q4_bridge = llmmetal_mpp_q4_create(lib); }
    if (g_mpp_q4_bridge == NULL) return false;
    if ((group != 64 && group != 32) || bits != 4) return false;
    if (mlx_array_dtype(sc) != MLX_BFLOAT16 || mlx_array_dtype(bi) != MLX_BFLOAT16) return false;
    if (mlx_array_ndim(x) != 3 || mlx_array_dtype(w) != MLX_UINT32) return false;
    if (T < 9 || T > 32 || K % group != 0 || N % 128 != 0) return false;
    int M = (T <= 16) ? 16 : (T <= 24) ? 24 : 32;
    mlx_array x2 = mlx_array_new();
    if (T < M) { int zsh[3] = {1, M - T, K}; mlx_array z = mlx_array_new();
        if (mlx_zeros(&z, zsh, 3, mlx_array_dtype(x), s) != 0) { mlx_array_free(z); return false; }
        mlx_vector_array cv = mlx_vector_array_new_data((mlx_array[]){x, z}, 2);
        int rc = mlx_concatenate_axis(&x2, cv, 1, s); mlx_vector_array_free(cv); mlx_array_free(z);
        if (rc != 0) { mlx_array_free(x2); return false; } }
    else mlx_array_set(&x2, x);
    mlx_array x3 = mlx_array_new(); int sh2[2] = {M, K};
    if (mlx_reshape(&x3, x2, sh2, 2, s) != 0) { mlx_array_free(x2); return false; }
    mlx_array_free(x2);
    mlx_array out2 = llmmetal_mpp_q4_tile_prim(x3, w, sc, bi, s, M, group);
    mlx_array_free(x3);
    if (out2.ctx == NULL) return false;
    mlx_array o3 = mlx_array_new(); int sh3[3] = {1, M, N};
    if (mlx_reshape(&o3, out2, sh3, 3, s) != 0) { mlx_array_free(out2); return false; }
    mlx_array_free(out2);
    if (T < M) { int st[3] = {0, 0, 0}, sp[3] = {1, T, N}, ss[3] = {1, 1, 1}; mlx_array yt = mlx_array_new();
        if (mlx_slice(&yt, o3, st, 3, sp, 3, ss, 3, s) == 0) mlx_array_set(res, yt); else return false;
        mlx_array_free(yt); }
    else mlx_array_set(res, o3);
    mlx_array_free(o3);
    return true;
}

#include "llm_mlx_tq_qmv_kernel.h"

/* Codebook TurboQuant GLOBALE (identico in tutti i tensori tq): `quantization.codebook` del config. */
static float g_tq_cb[16];
static int g_tq_cb_n = 0;
static mlx_array g_tq_cb_arr;   /* codebook come array MLX, costruito una volta (evita 1 array/call) */
void llm_mlx_tq_set_cb(const float* cb, int n) {
    if (n < 0) n = 0; if (n > 16) n = 16;
    for (int i = 0; i < n; i++) g_tq_cb[i] = cb[i];
    g_tq_cb_n = n;
    if (g_tq_cb_arr.ctx != NULL) { mlx_array_free(g_tq_cb_arr); g_tq_cb_arr = mlx_array_new(); }
    if (n > 0) { int sh[1] = { n }; g_tq_cb_arr = mlx_array_new_data(g_tq_cb, sh, 1, MLX_FLOAT32); }
}
int llm_mlx_tq_get_cb(float* out, int cap) {
    int n = g_tq_cb_n < cap ? g_tq_cb_n : cap;
    for (int i = 0; i < n; i++) out[i] = g_tq_cb[i];
    return g_tq_cb_n;
}
int llm_mlx_tq_enabled(void) { return g_tq_cb_n > 0; }

bool quantized_linear_cfg(mlx_array* res, mlx_array x, mlx_array w, mlx_array scales, mlx_array biases, int group_size, int bits, const char* mode, mlx_stream stream) {
    /* TQ dense (opzione 2, LLM_TQ_DENSE=1): w UINT32 packed + biases=signs (già forniti dal fetch),
     * codebook GLOBALE. Decode: kernel TQ che legge packed (meno traffico memoria del bf16 plain).
     * Prefill (T>1): loop per riga (una volta, correttezza prima). */
    if (llm_mlx_tq_enabled() && mlx_array_dtype(w) == MLX_UINT32 && biases.ctx != NULL) {
        float cb[16]; const int cbn = llm_mlx_tq_get_cb(cb, 16);
        const int xnd = (int)mlx_array_ndim(x);
        const int K = (xnd >= 1) ? (int)mlx_array_shape(x)[xnd - 1] : 0;
        const int cw = mlx_array_ndim(w) >= 2 ? (int)mlx_array_shape(w)[1] : 0;
        const int O = mlx_array_ndim(w) >= 2 ? (int)mlx_array_shape(w)[0] : 0;
        const int NG = mlx_array_ndim(scales) >= 2 ? (int)mlx_array_shape(scales)[mlx_array_ndim(scales) - 1] : 0;
        const int grp = (NG > 0 && K > 0 && K % NG == 0) ? K / NG : 64;
        const size_t tot = mlx_array_size(x);
        if (cbn >= 3 && K > 0 && O > 0 && NG > 0 && mlx_array_shape(biases)[mlx_array_ndim(biases) - 1] == K) {
            /* signs = biases; codebook globale [cbn] (array cachato); trit = (cbn==3). */
            int trit = (cbn == 3), bits3 = trit ? 3 : (cbn == 8 ? 3 : (cbn == 4 ? 2 : 4));
            mlx_array CB = g_tq_cb_arr;
            const int rows = (K > 0) ? (int)(tot / (size_t)K) : 1;
            mlx_array X2 = mlx_array_new(); int s2[2] = { rows, K };
            if (mlx_reshape(&X2, x, s2, 2, stream) == 0) {
                mlx_array Xr = mlx_array_new();
                if (mlx_multiply(&Xr, X2, biases, stream) == 0) {
                    mlx_array Xh = mlx_array_new();
                    mlx_optional_float hsc = { 1.0f / sqrtf((float)K), true };
                    if (mlx_hadamard_transform(&Xh, Xr, hsc, stream) == 0) {
                        if (rows == 1) {
                            /* DECODE (1 vettore): niente take_axis/stack — 1 kernel, reshape out. */
                            int s1[1] = { K };
                            mlx_array rv = mlx_array_new(), y = mlx_array_new();
                            if (mlx_reshape(&rv, Xh, s1, 1, stream) == 0 &&
                                llm_mlx_tq_qmv(&y, w, scales, CB, rv, 0, K, O, NG, cw, bits3, grp, trit, stream) == 0) {
                                mlx_array yc = mlx_array_new();
                                bool aokc = (mlx_array_dtype(y) == mlx_array_dtype(x)) ? (mlx_array_set(&yc, y) == 0) : (mlx_astype(&yc, y, mlx_array_dtype(x), stream) == 0);
                                if (aokc) {
                                    int shp[8]; int nd = xnd;
                                    for (int i = 0; i < nd && i < 8; i++) shp[i] = mlx_array_shape(x)[i];
                                    shp[nd - 1] = O;
                                    mlx_array yr = mlx_array_new();
                                    if (mlx_reshape(&yr, yc, shp, nd, stream) == 0) {
                                        mlx_array_set(res, yr); mlx_array_free(yr);
                                        mlx_array_free(yc);
                                        mlx_array_free(y); mlx_array_free(rv);
                                        mlx_array_free(Xh); mlx_array_free(Xr); mlx_array_free(X2);
                                        return true;
                                    }
                                    mlx_array_free(yr);
                                }
                                mlx_array_free(yc);
                            }
                            mlx_array_free(y); mlx_array_free(rv);
                        } else {
                        /* PREFILL (T>1): UN solo launch batched — mode kg con indici=0 (esperto 0),
                         * x [T,K] -> y [T,O]. Prima era 1 launch PER RIGA (~440×T per prefill). */
                        mlx_array idxz = mlx_array_new(), y = mlx_array_new();
                        int iz[1] = { rows };
                        if (mlx_zeros(&idxz, iz, 1, MLX_UINT32, stream) == 0 &&
                            llm_mlx_tq_qmv_kg(&y, w, scales, CB, Xh, idxz, rows, K, O, NG, cw, bits3, grp, trit, stream) == 0) {
                            mlx_array yc = mlx_array_new();
                            bool aokc = (mlx_array_dtype(y) == mlx_array_dtype(x)) ? (mlx_array_set(&yc, y) == 0) : (mlx_astype(&yc, y, mlx_array_dtype(x), stream) == 0);
                            if (aokc) {
                                int shp[8]; int nd = xnd;
                                for (int i = 0; i < nd && i < 8; i++) shp[i] = mlx_array_shape(x)[i];
                                shp[nd - 1] = O;
                                mlx_array yr = mlx_array_new();
                                if (mlx_reshape(&yr, yc, shp, nd, stream) == 0) {
                                    mlx_array_set(res, yr); mlx_array_free(yr);
                                    mlx_array_free(yc); mlx_array_free(y); mlx_array_free(idxz);
                                    mlx_array_free(Xh); mlx_array_free(Xr); mlx_array_free(X2);
                                    return true;
                                }
                                mlx_array_free(yr);
                            }
                            mlx_array_free(yc);
                        }
                        mlx_array_free(y); mlx_array_free(idxz);
                        }
                    }
                    mlx_array_free(Xh);
                }
                mlx_array_free(Xr);
            }
            mlx_array_free(X2);
        }
    }
    /* MPP q4 (tensor unit Apple) — LLMMETAL_MPP_Q4=1: x [1,T,K] affine g64 b4 -> tensor-op MPP.
     * `=0` e' un vero kill-switch. Group/bits/mode sono derivati dalle shape del tensore:
     * un trunk affine g32-b4 deve ricadere sempre nella qmm generica. */
    const char* mpp_q4_env = llm_ai_mlx_opt_env("LLMMETAL_MPP_Q4");
    if (mpp_q4_env != NULL && strcmp(mpp_q4_env, "0") != 0 &&
        mlx_array_ndim(x) == 3 && mlx_array_dtype(w) == MLX_UINT32 &&
        scales.ctx != NULL && mlx_array_size(scales) > 0 && mlx_array_ndim(w) == 2) {
        int mg, mb; char mmode[16];
        llm_mlx_quant_derive(w, scales, group_size, bits, mode, &mg, &mb, mmode, sizeof(mmode));
        const int* xs = mlx_array_shape(x);
        int K = xs[2], N = mlx_array_shape(w)[0];
        if (llm_mpp_q4_tile_try(res, x, w, scales, biases, mg, mb, K, N, stream, xs[1])) {
            static long mpp_hits = 0;
            if (++mpp_hits == 1 && getenv("LLM_WHICH_PATH") != NULL)
                fprintf(stderr, "[which-path] quantized_linear: MPP q4 TILE g%d b4 T=%d K=%d N=%d\n",
                        mg, xs[1], K, N);
            if (llm_ai_mlx_opt_env("LLMMETAL_MPP_Q4_COUNT") != NULL) {
                static long last = 0;
                if (mpp_hits - last >= 500) { last = mpp_hits;
                    fprintf(stderr, "[mpp-count] tile_hits=%ld tile_miss=%ld\n", mpp_hits, g_mpp_tile_miss); }
            }
            return true;
        }
        g_mpp_tile_miss++;
        /* Seconda via: kernel MPP via mlx_fast_metal_kernel (llm_mlx_mpp_q4_kernel.h), ripristinata
         * dopo la FIX del set_grid (= thread): era stata tolta quando si e' integrata la variante
         * "tile". Richiede sc/bi bf16 (con lo scales-hoist e' f32 -> non entra: usare
         * LLM_F32_SCALES=0 per provarla). */
        if (llm_mlx_mpp_q4_linear(res, x, w, scales, biases, mg, mb, K, N, stream)) {
            static int mpp2_hit = 0;
            if (!mpp2_hit && getenv("LLM_WHICH_PATH") != NULL) {
                fprintf(stderr, "[which-path] quantized_linear: MPP q4 FM g%d b4 T=%d K=%d N=%d\n", mg, xs[1], K, N);
                mpp2_hit = 1;
            }
            return true;
        }
        static int mpp_fallback_logged = 0;
        if (!mpp_fallback_logged && getenv("LLM_WHICH_PATH") != NULL) {
            fprintf(stderr, "[which-path] quantized_linear: MPP q4 fallback mode=%s group=%d bits=%d K=%d N=%d\n",
                    mmode, mg, mb, K, N);
            mpp_fallback_logged = 1;
        }
    }
    /* P10 — GEMM int8-activation × Q4/Q5 (tensor unit M5) per il PREFILL — PRIMITIVE MLX.
     * LLMMETAL_OQ_A8=1 usa il primitive (in-graph, una submit, zero sync). LLMMETAL_OQ_A8_VERIFY=1 confronta
     * il primitive con mlx_quantized_matmul e usa SEMPRE il normale. Default OFF. */
    if ((llm_mlx_oq_a8_enabled() || llm_mlx_oq_a8_verify_enabled() || llm_mlx_oq_a8_auto_enabled()) &&
        mode != NULL && strcmp(mode, "none") != 0 &&
        scales.ctx != NULL && mlx_array_size(scales) > 0) {
        int og_group, og_bits; char og_mode[16];
        llm_mlx_quant_derive(w, scales, group_size, bits, mode, &og_group, &og_bits, og_mode, sizeof(og_mode));
        if (strcmp(og_mode, "affine") == 0 && og_group == 64 && (og_bits == 4 || og_bits == 5 || og_bits == 8)) {
            int xnd = mlx_array_ndim(x);
            const int* xs = (xnd > 0) ? mlx_array_shape(x) : NULL;
            int K = (xnd >= 2) ? xs[xnd - 1] : 0;
            size_t tot = mlx_array_size(x);
            int M = (K > 0) ? (int)(tot / (size_t)K) : 0;
            int N = (mlx_array_ndim(w) >= 2) ? (int)mlx_array_shape(w)[0] : 0;
            /* FAIL-OPEN (2026-09-16, promozione a default): il ctx e' lazy (metallib/pipeline).
             * Il vecchio `if (ctx == NULL) return false;` faceva FALLIRE il prefill quando le
             * metallib mancavano -- incompatibile con un default ON. Ora l'assenza del ctx rende
             * semplicemente NON eleggibile (si ricade sul path normale, come ogni altra leva). */
            /* FEDELTA' oMLX: (a) min_tokens=128 -- sotto questa soglia la Stage-A costa piu' di
             * quanto il GEMM risparmi (_MIN_TOKENS_DEFAULT in qwen35_oq_a8.py); (b) NESSUN tetto N
             * -- oMLX richiede solo N % tile_bn == 0 (BN=64 per la variante 806), quindi anche
             * lm_head/FFN grandi sono eleggibili. Il vecchio `M>=16 && N<=16384` escludeva per
             * esempio le gate/up del 27B (inter=17408). Override per A/B: LLMMETAL_OQ_A8_MINTOK,
             * LLMMETAL_OQ_A8_MAXN (0 = illimitato, default). */
            static int s_oq_a8_minm = -1, s_oq_a8_maxn = -1;
            if (s_oq_a8_minm < 0) { const char* e = llm_ai_mlx_opt_env("LLMMETAL_OQ_A8_MINTOK"); s_oq_a8_minm = e ? atoi(e) : 128; }
            /* Tetto N di DEFAULT 16384 (come il gate del path verify): ESCLUDE l'lm_head
             * (vocab ~151k) dal W8A8. BUG 2026-09-19: con LLMMETAL_OQ_A8_MINTOK=1 l'lm_head andava
             * in W8A8 a M=1 e le probabilita' collassavano (score: max prob ~0,1 e 0/256 hit
             * con MLX a 89,8%): la distribuzione dei logits si appiattisce. Il path di
             * generate escludeva gia' l'lm_head (M<16 nel try), ma greedy_prob/verify_token
             * passano da qui. Override: LLMMETAL_OQ_A8_MAXN=0 (illimitato) per riprodurre. */
            if (s_oq_a8_maxn < 0) { const char* e = llm_ai_mlx_opt_env("LLMMETAL_OQ_A8_MAXN"); s_oq_a8_maxn = e ? atoi(e) : 16384; }
            const int minm_eff = (g_oq_a8_minm_override >= 0) ? g_oq_a8_minm_override : s_oq_a8_minm;
            if ((llm_mlx_oq_a8_verify_enabled() || llm_mlx_oq_a8_active_for(K)) &&
                M >= minm_eff && N > 0 && (s_oq_a8_maxn == 0 || N <= s_oq_a8_maxn) &&
                (N % 64) == 0 && (K % 64) == 0 &&
                llm_mlx_oq_a8_ctx() != NULL) {
                mlx_array kres = mlx_array_new();
                if (llm_mlx_oq_a8_prim_enabled()) {
                    /* DIAGNOSTICA (§6ter, "prossimo passo #1"): LLMMETAL_OQ_A8_SYNC_X forza la
                     * materializzazione di x/w/scales/biases SEPARATAMENTE, PRIMA che il
                     * primitive li catturi nel grafo — diverso da LLMMETAL_OQ_A8_SYNC (che forza
                     * l'INTERO output del primitive, quindi anche il costoso matmul, a girare
                     * subito). Se SOLO questo basta a correggere l'output, isola quale input è
                     * "stantio" quando letto lazy (x è il sospetto principale: è l'unico input
                     * per-layer, non statico come pesi/scale/bias). Il tentativo precedente
                     * (`copy(x)` DENTRO eval_gpu, quindi ALL'EVAL, non prima) non risolveva —
                     * questo testa l'ipotesi diversa che il dato sia già sbagliato PRIMA che il
                     * primitive lo catturi, non che venga sporcato dopo.
                     * Default OFF, zero overhead quando non impostata (getenv una volta pesa poco
                     * ma qui e' un chokepoint hot: TODO cache come le altre llm_mlx_oq_a8_*). */
                    if (llm_ai_mlx_opt_env("LLMMETAL_OQ_A8_SYNC_X")) {
                        (void)mlx_array_eval(x); (void)mlx_array_eval(w);
                        (void)mlx_array_eval(scales); (void)mlx_array_eval(biases);
                    }
                    kres = llmmetal_oq_a8_prim(x, w, scales, biases, stream, og_bits);   /* in-graph (sperimentale) */
                    if (kres.ctx != NULL && llm_ai_mlx_opt_env("LLMMETAL_OQ_A8_SYNC")) (void)mlx_array_eval(kres);
                } else {
                    /* EAGER (corretto, +7-9%): Stage-A + layout GPU + GEMM via bridge */
                    mlx_array xf = mlx_array_new(), xc = mlx_array_new(), o8 = mlx_array_new();
                    bool eok = (mlx_array_dtype(x) == MLX_FLOAT16) ? mlx_array_set(&xf, x) == 0
                                                                   : mlx_astype(&xf, x, MLX_FLOAT16, stream) == 0;
                    eok = eok && mlx_contiguous(&xc, xf, false, stream) == 0;
                    int osh[8]; for (int i2 = 0; i2 < xnd && i2 < 8; i2++) osh[i2] = xs[i2]; osh[xnd - 1] = N;
                    eok = eok && mlx_zeros(&o8, osh, xnd, MLX_FLOAT16, stream) == 0;
                    if (eok && llmmetal_oq_a8_forward(xc, w, scales, biases, o8, stream, og_bits) == 0)
                        mlx_array_set(&kres, o8);
                    mlx_array_free(xf); mlx_array_free(xc); mlx_array_free(o8);
                }
                if (kres.ctx != NULL) {
                    if (!llm_mlx_oq_a8_verify_enabled()) {
                        mlx_dtype xdt0 = mlx_array_dtype(x);
                        bool r2 = (xdt0 == MLX_FLOAT16) ? mlx_array_set(res, kres) == 0
                                                        : mlx_astype(res, kres, xdt0, stream) == 0;
                        mlx_array_free(kres);
                        return r2;
                    }
                    mlx_optional_int gs2 = {og_group, true}, bt2 = {og_bits, true};
                    mlx_array nres = mlx_array_new();
                    if (mlx_quantized_matmul(&nres, x, w, scales, biases, true, gs2, bt2, og_mode, stream) == 0) {
                        mlx_array kf = mlx_array_new(), nf = mlx_array_new();
                        if (mlx_astype(&kf, kres, MLX_FLOAT32, stream) == 0 &&
                            mlx_astype(&nf, nres, MLX_FLOAT32, stream) == 0 &&
                            mlx_array_eval(kf) == 0 && mlx_array_eval(nf) == 0) {
                            const float* kd = mlx_array_data_float32(kf);
                            const float* nd = mlx_array_data_float32(nf);
                            if (kd && nd) {
                                size_t nel = mlx_array_size(nres); double mx = 0, su = 0;
                                for (size_t vi = 0; vi < nel; vi++) { double d = fabs((double)kd[vi] - (double)nd[vi]); if (d > mx) mx = d; su += d; }
                                fprintf(stderr, "[oq_a8-verify] M=%d N=%d K=%d max_abs_diff=%.6g mean=%.3g (nel=%zu)\n",
                                        M, N, K, mx, nel ? su / nel : 0.0, nel);
                            }
                        }
                        mlx_array_free(kf); mlx_array_free(nf);
                    }
                    mlx_array_set(res, nres);
                    mlx_array_free(nres);
                    mlx_array_free(kres);
                    return true;
                }
                mlx_array_free(kres);
            }
        }
    }

    /* §13.6: mode "none" = modello NON quantizzato (draft BF16/FP32 pieni, es.
     * Qwen3.8-27B-DFlash2). w e' il .weight completo [out,in]; scales/biases sono
     * vuoti (non esistono nel safetensors BF16) -- delega a plain_linear (x @ w.T). */
    if (g_bf16_all) {
        /* Nessun alias quando `x` e' gia' nel dtype di attivazione: niente handle `xb`
         * allocato quando il cast non serve (evita un `new mlx::core::array` per matmul). */
        mlx_array xb = {0};
        mlx_array xuse = x;
        bool need_cast = (mlx_array_dtype(x) != g_act16);
        bool ok = true;
        if (need_cast) {
            ok = mlx_astype(&xb, x, g_act16, stream) == 0;
            if (ok) xuse = xb;
        }
        /* Tensore PLAIN (scale assenti/vuote — attenzione/embed bf16 di gpt-oss-120b): sotto
         * bf16 la branca quantizzata qui sotto chiamerebbe mlx_quantized_matmul con scales vuote
         * -> "expected a non-empty mlx_array" (array.cpp:311, ndim sulle scales). Il 20b non ci
         * cade perche' ha OGNI proiezione quantizzata (per-tensor); il 120b ha l'attenzione plain.
         * Stessa guardia del path non-bf16 (:463), replicata qui. mode=="none" ricade in plain. */
        bool plain = (mode != NULL && strcmp(mode, "none") == 0)
                  || scales.ctx == NULL || mlx_array_size(scales) == 0;
        if (plain) {
            if (ok) ok = plain_linear(res, xuse, w, stream);
        } else {
            int e_group, e_bits; char e_mode[16];
            llm_mlx_quant_derive(w, scales, group_size, bits, mode, &e_group, &e_bits, e_mode, sizeof(e_mode));
            mlx_optional_int gsz = {e_group, true};
            mlx_optional_int bts = {e_bits, true};
            if (ok) ok = mlx_quantized_matmul(res, xuse, w, scales, biases, /*transpose=*/true, gsz, bts, e_mode, stream) == 0;
        }
        mlx_array_free(xb);
        return ok;
    }
    if (mode != NULL && strcmp(mode, "none") == 0) {
        return plain_linear(res, x, w, stream);
    }
    /* Tensore NON quantizzato (scales assente, es. lm_head/embed bf16 plain di
     * gpt-oss-120b-mxfp4-bf16): instrada a plain_linear a prescindere dal quant_mode
     * di config (altrimenti mlx_quantized_matmul con scales vuote fallirebbe). */
    if (scales.ctx == NULL || mlx_array_size(scales) == 0) {
        return plain_linear(res, x, w, stream);
    }
    /* §quant per-tensore (Qwen3.6-35B-A3B, GPT-OSS): derive group/bits/mode dalle
     * shape (es. GPT-OSS: attention/embed/lm_head affine 64/8, SOLO gli esperti
     * mxfp4 32/4 -- config quantization.<tensore> per-tensore, 122 override). */
    int e_group, e_bits; char e_mode[16];
    llm_mlx_quant_derive(w, scales, group_size, bits, mode, &e_group, &e_bits, e_mode, sizeof(e_mode));
    mlx_optional_int gsz = {e_group, true};
    mlx_optional_int bts = {e_bits, true};


    /* LLM_GEMVQ4 / LLM_GEMVQ4_VERIFY (sperimentale, default OFF, vedi llm_mlx_gemvq4_try):
     * intercetta la GEMV affine 4-bit a singolo token col kernel Metal Q4 (chokepoint di
     * tutte le proiezioni dense quantizzate: q/k/v/o, gate/up/down, shared, lm_head).
     * In VERIFY calcola ENTRAMBI e usa SEMPRE il path normale (zero rischio); in modalita'
     * pura usa il kernel se disponibile. Fallback silenzioso su qualunque ineleggibilita'
     * (try=false) -> si continua col path normale sotto. */
    /* P0-ter FIX (2026-09-16): i parametri ARGOMENTO possono essere i fallback ("none"/0) per i
     * checkpoint la cui config non dichiara mode/bits: il blocco oQ A8 sopra lo sa gia' e DERIVA
     * con llm_mlx_quant_derive. Il guard di GEMVQ4 testava invece gli argomenti grezzi -> il
     * kernel NON PARTIVA MAI, e l'A/B identico al bit sembrava un pareggio invece di un no-op.
     * Qui si deriva (senza toccare e_mode/e_bits: servono al path normale, dove "none" instrada
     * a plain_linear) e si passa ai valori reali al try(). */
    int g4_group = e_group, g4_bits = e_bits;
    char g4_mode[16]; g4_mode[0] = '\0';
    bool g4_ok = false;
    if ((llm_mlx_gemvq4_enabled() || llm_mlx_gemvq4_verify_enabled()) &&
        scales.ctx != NULL && mlx_array_size(scales) > 0) {
        llm_mlx_quant_derive(w, scales, e_group, e_bits, e_mode, &g4_group, &g4_bits, g4_mode, sizeof(g4_mode));
        g4_ok = (g4_bits == 4 && strcmp(g4_mode, "affine") == 0);
    }
    if (g4_ok) {
        mlx_array kres = mlx_array_new();
        if (llm_mlx_gemvq4_try(&kres, x, w, scales, biases, g4_group, g4_bits, g4_mode, stream)) {
            if (!llm_mlx_gemvq4_verify_enabled()) {
                mlx_array_set(res, kres);
                mlx_array_free(kres);
                return true;
            }
            /* VERIFY: path normale + confronto; il risultato usato resta quello normale. */
            mlx_array nres = mlx_array_new();
            bool okn = mlx_quantized_matmul(&nres, x, w, scales, biases, /*transpose=*/true, gsz, bts, e_mode, stream) == 0;
            if (okn) {
                mlx_array kf = mlx_array_new(), nf = mlx_array_new();
                if (mlx_astype(&kf, kres, MLX_FLOAT32, stream) == 0 &&
                    mlx_astype(&nf, nres, MLX_FLOAT32, stream) == 0 &&
                    mlx_array_eval(kf) == 0 && mlx_array_eval(nf) == 0) {
                    const float* kd = mlx_array_data_float32(kf);
                    const float* nd = mlx_array_data_float32(nf);
                    if (kd && nd) {
                        size_t nel = (size_t)mlx_array_size(nres);
                        double maxd = 0; size_t widx = 0;
                        for (size_t vi = 0; vi < nel; vi++) {
                            double d = fabs((double)kd[vi] - (double)nd[vi]);
                            if (d > maxd) { maxd = d; widx = vi; }
                        }
                        fprintf(stderr, "[gemvq4-verify] M=%u max_abs_diff=%.6g (nel=%zu)\n",
                                (unsigned)mlx_array_shape(w)[0], maxd, nel);
                    }
                }
                mlx_array_free(kf); mlx_array_free(nf);
            }
            mlx_array_set(res, nres);
            mlx_array_free(nres);
            mlx_array_free(kres);
            return true;
        }
        mlx_array_free(kres);
        /* try fallita: continua col path normale sotto (mai un errore). */
    }
    return mlx_quantized_matmul(res, x, w, scales, biases, /*transpose=*/true, gsz, bts, e_mode, stream) == 0;
}

/* ── FEDELTA' oMLX: Stage A CONDIVISA (qwen35_oq_a8.py: stage_a(x) + apply_plan) ──
 * oMLX quantizza l'attivazione UNA volta e la riusa per ogni proiezione che la consuma
 * (MLP gate+up; qkv/z/a/b). Il fused llmmetal_oq_a8_prim ricalcola Stage A ad ogni proiezione.
 * Solo col prim in-graph (l'eager non ha Stage A separabile). */
bool llm_mlx_oq_a8_stage_try(mlx_array x, mlx_array wref, mlx_array scref, mlx_array biref,
                                    int e_group, int e_bits, const char* e_mode, mlx_stream stream,
                                    mlx_array* qa, mlx_array* sa, mlx_array* ra) {
    qa->ctx = NULL; sa->ctx = NULL; ra->ctx = NULL;
    if (!llm_mlx_oq_a8_prim_enabled()) return false;
    /* POLICY + GUARD (2026-09-19): lo stage path non controllava ne' il gate globale ne' il
     * modo bf16. La combo bf16-OFF + W8A8 e' ROTTA (un token indietro) -> fail-open a MLX. */
    if (!g_bf16_all) return false;
    if (!llm_mlx_oq_a8_enabled() && !llm_mlx_oq_a8_verify_enabled() && !llm_mlx_oq_a8_auto_enabled()) return false;
    if (llm_ai_mlx_opt_env("LLMMETAL_OQ_A8_NO_SHARE")) return false;   /* A/B: disattiva Stage A condivisa */
    if (e_group != 64 || (e_bits != 4 && e_bits != 5 && e_bits != 8)) return false;
    if (!e_mode || strcmp(e_mode, "affine") != 0) return false;
    if (mlx_array_ndim(x) < 2) return false;
    int xnd = mlx_array_ndim(x);
    const int* xs = mlx_array_shape(x);
    int K = xs[xnd - 1];
    if (K <= 0 || (K % 64) != 0) return false;
    size_t total = mlx_array_size(x);
    if (total == 0 || total % (size_t)K != 0) return false;
    int M = (int)(total / (size_t)K);
    /* Gate M: default 128 (Stage-A costa piu' di quanto il GEMM risparmi sotto); per il
     * VERIFY DFlash2 (M=8) si abbassa con LLMMETAL_OQ_A8_MINTOK (stesso env del gate esterno).
     * Il kernel OQ-A8 e' M-safe (store guardato `if (m < M)`, clamp delle righe oltre M). */
    static int s_stage_minm = -2;
    if (s_stage_minm == -2) { const char* e = llm_ai_mlx_opt_env("LLMMETAL_OQ_A8_MINTOK"); s_stage_minm = e ? atoi(e) : 128; }
    const int minm_eff = (g_oq_a8_minm_override >= 0) ? g_oq_a8_minm_override : s_stage_minm;
    if (M < minm_eff) return false;
    if (mlx_array_ndim(wref) < 2 || mlx_array_dtype(wref) != MLX_UINT32) return false;
    int N = (int)mlx_array_shape(wref)[0];
    if (N <= 0 || (N % 64) != 0) return false;
    if (scref.ctx == NULL || biref.ctx == NULL) return false;
    if (llm_mlx_oq_a8_ctx() == NULL) return false;
    return llmmetal_oq_a8_stage_a(x, qa, sa, ra, stream) == 0;
}

/* GEMM su Stage A gia' calcolata (fedelta' oMLX). */
bool llm_mlx_oq_a8_qmm_stage_try(mlx_array* res, mlx_array qa, mlx_array sa, mlx_array ra,
                                        mlx_array xref, mlx_array w, mlx_array scales, mlx_array biases,
                                        int e_group, int e_bits, const char* e_mode, mlx_stream stream) {
    if (qa.ctx == NULL) return false;
    if (!(llm_mlx_oq_a8_enabled() || llm_mlx_oq_a8_verify_enabled())) return false;
    if (e_group != 64 || (e_bits != 4 && e_bits != 5 && e_bits != 8)) return false;
    if (!e_mode || strcmp(e_mode, "affine") != 0) return false;
    if (scales.ctx == NULL || biases.ctx == NULL) return false;
    if (mlx_array_ndim(w) < 2 || mlx_array_dtype(w) != MLX_UINT32) return false;
    int N = (int)mlx_array_shape(w)[0];
    if (N <= 0 || (N % 64) != 0) return false;
    mlx_dtype xdt = mlx_array_dtype(xref);
    mlx_array p = llmmetal_oq_a8_qmm(qa, sa, ra, w, scales, biases, stream, e_bits, xdt == MLX_BFLOAT16);
    if (p.ctx == NULL) return false;
    bool r2 = (xdt == MLX_FLOAT16) ? mlx_array_set(res, p) == 0
                                   : mlx_astype(res, p, xdt, stream) == 0;
    mlx_array_free(p);
    return r2;
}

/* Deriva (group_size, bits, mode) per-tensore dalle shape di w/scales quantizzati,
 * per i modelli con quantizzazione PER-TENSORE eterogenea (GPT-OSS, Qwen3.6-35B):
 * ratio = wcols / scols = group/bpp (bpp=32/bits). Combo standard:
 *   ratio 16 -> (64, 8)  affine  (8-bit, 4 valori per uint32)
 *   ratio 4  -> (32, 4)  mxfp4   (4-bit, 8 valori per uint32)
 * Altrimenti fallback ai valori globali del config. */
/* Copia il nome-mode SENZA snprintf (parsing del formato = colla chiamata 17+ volte/token). */
void llm_mlx_mode_set(char* dst, size_t cap, const char* s) {
    size_t n = 0; while (s[n] != '\0' && n + 1 < cap) { dst[n] = s[n]; ++n; } dst[n] = '\0';
}
void llm_mlx_quant_derive(mlx_array w, mlx_array scales,
                                 int fb_group, int fb_bits, const char* fb_mode,
                                 int* out_group, int* out_bits, char* out_mode, size_t mode_cap) {
    if (w.ctx == NULL || scales.ctx == NULL) {
        /* Pesi non quantizzati (F16, nessuno scales): nessuna derivazione affine ->
         * mode "" (il percorso MoE usa gather_mm non quantizzato). */
        *out_group = 0; *out_bits = 0;
        llm_mlx_mode_set(out_mode, mode_cap, "");
        return;
    }
    int wcols = 0, scols = 0;
    if (mlx_array_ndim(w) >= 2 && mlx_array_ndim(scales) >= 2) {
        const int* ws = mlx_array_shape(w);
        const int* ss = mlx_array_shape(scales);
        wcols = ws[mlx_array_ndim(w) - 1];
        scols = ss[mlx_array_ndim(scales) - 1];
    }
    *out_group = fb_group; *out_bits = fb_bits;
    llm_mlx_mode_set(out_mode, mode_cap, fb_mode != NULL ? fb_mode : "affine");
    if (wcols > 0 && scols > 0 && (wcols % scols) == 0) {
        int ratio = wcols / scols;
        /* 2-bit affine (es. Ternary-Bonsai g128, bits 2): ratio 8 e' AMBIGUO con affine 4-bit g64
         * (128/16=8 e 64/8=8). Quando la config dichiara bits 2 si tiene fb_group (128) e bits 2,
         * NON la derivazione da ratio (che darebbe (64,4) e corromperebbe il layout 2-bit). */
        if (fb_bits == 2) {
            *out_group = fb_group > 0 ? fb_group : 128;
            *out_bits = 2;
            llm_mlx_mode_set(out_mode, mode_cap, "affine");
        } else if (ratio == 16) { *out_group = 64; *out_bits = 8; llm_mlx_mode_set(out_mode, mode_cap, "affine"); }
        else if (ratio == 8) { *out_group = 64; *out_bits = 4; llm_mlx_mode_set(out_mode, mode_cap, "affine"); }
        /* BUGFIX (2026-09-19): ratio 4 e' AMBIGUO tra affine-g32-b4 e mxfp4 (entrambi
         * wcols/scols=4). Prima si forzava mxfp4 -> i modelli AFFINE g32 4-bit (es. MTPLX
         * Qwen3.8-27B-Optimized-Speed) fallivano con "[quantized_matmul] Scale type must be
         * uint8 but received bfloat16". Il discriminante e' il DTYPE delle scale: mxfp4 usa
         * scale uint8 (e8m0), affine usa lo stesso dtype del compute (bf16/f16/f32). */
        else if (ratio == 4) {
            if (mlx_array_dtype(scales) == MLX_UINT8) {
                *out_group = 32; *out_bits = 4; llm_mlx_mode_set(out_mode, mode_cap, "mxfp4");
            } else {
                *out_group = 32; *out_bits = 4; llm_mlx_mode_set(out_mode, mode_cap, "affine");
            }
        }
    }
}

/* FASE 1b.3 (Mixtral): proiezione lineare NON quantizzata (x @ w.T, senza bias) --
 * capacità nuova: finora ogni proiezione di questo motore passava per
 * quantized_linear_cfg (pesi sempre in formato affine-quantizzato). Mixtral-8x22B-4bit
 * tiene invece `block_sparse_moe.gate` (il router MoE) in float pieno -- verificato
 * sull'elenco tensori reale: solo ".weight", nessuna ".scales"/".biases" accanto, a
 * differenza di q/k/v/o_proj e degli esperti che restano quantizzati -- pratica comune
 * per i router, dove la precisione della decisione di instradamento pesa più del
 * risparmio di memoria. w ha forma [out_features, in_features] (convenzione
 * nn.Linear), quindi serve la trasposta prima del matmul.
 * IMPORTANTE (bug reale trovato dal vivo): i tensori non quantizzati di questo
 * modello sono salvati in F16 (verificato sull'header safetensors reale:
 * dtype "F16" sia per block_sparse_moe.gate.weight sia per embed_tokens.weight,
 * contro "U32" per i tensori quantizzati) -- SENZA un cast esplicito a float32,
 * mlx_array_data_float32 sui dati risultanti reinterpreta i bit float16 come se
 * fossero float32 (stride/formato sbagliati), producendo valori quasi-zero
 * plausibili ma completamente sbagliati (osservato: tutti gli 8 valori letti
 * ~0.000000). Serve mlx_astype esplicito PRIMA di ogni lettura host o operazione
 * che assuma float32 a valle. */
bool plain_linear(mlx_array* res, mlx_array x, mlx_array w, mlx_stream stream) {
    /* Allinea il dtype del PESO a quello dell'ATTIVAZIONE, non forzare f32. Forzare f32
     * (comportamento precedente, per evitare disallineamenti x_f16/w_f32) su modelli con pesi
     * bf16 e attivazione bf16 (attenzione/embed/lm_head plain del gpt-oss-120b) materializza ad
     * OGNI call una copia f32 del peso e fa un matmul f32: 2x banda + compute f32. Misurato sul
     * 120b: il blocco attention per-layer era 3,2x quello del 20b (pesi 8-bit) e l'attention era
     * il 73% del decode. Con x bf16 il peso resta bf16 (matmul bf16, accumulo f32 in MLX) come
     * mlx_lm. Su x f32 il comportamento e' identico a prima (w->f32). */
    mlx_dtype xd = mlx_array_dtype(x);
    mlx_array wc = mlx_array_new(), wt = mlx_array_new();
    bool ok = (mlx_array_dtype(w) == xd) ? (mlx_array_set(&wc, w) == 0)
                                         : (mlx_astype(&wc, w, xd, stream) == 0);
    if (ok) ok = mlx_transpose(&wt, wc, stream) == 0;
    if (ok) ok = mlx_matmul(res, x, wt, stream) == 0;
    mlx_array_free(wc);
    mlx_array_free(wt);
    return ok;
}

/* Logit in float32 PRIMA di ogni lettura host o softmax. Non e' difensivo, e' un difetto
 * misurato: con g_bf16_all acceso la proiezione lm_head produce bf16, e
 *   (a) mlx_array_data_float32 nella mlx-c e' dichiarato "cast to float32*" -- un REINTERPRET
 *       cast, non una conversione -- quindi leggere vocab*sizeof(float) byte su un buffer da
 *       vocab*sizeof(bf16) legge fuori dal buffer e SIGSEGA (misurato 2026-09-08:
 *       Qwen3.6-27B-4bit, bf16=both, temperature=0.8 -> exit 139, nessun token emesso);
 *   (b) stessa causa sul lato softmax: mlx_softmax_axis(precise=true) NON promote il dtype, quindi
 *       con input bf16 l'output e' bf16 e leggerlo come float32* dà valori >1 (misurato: P(argmax)
 *       = 1,0019 nel 97,3% delle posizioni). Non era la distribuzione a collassare, era la lettura.
 *       Dopo il cast: P(argmax) media 0,5873 (bf16) vs 0,5872 (fp32) e loss 0,6705 vs 0,6711 nats
 *       su 600 posizioni di testo reale -> bf16 e fp32 sono indistinguibili anche in qualita'.
 * Il cast bf16->fp32 e' ESATTO (bf16 e' un sottinsieme di fp32), quindi non muove nessun argmax:
 * la via greedy resta bit-identica a prima. Costa un vettore [vocab] per token, e il percorso
 * gia' fp32 non paga nulla (check sul dtype). Stessa classe di bug gia' documentata sopra in
 * plain_linear per i pesi F16. */
bool llm_mlx_logits_f32(mlx_array* out, mlx_array logits, mlx_stream stream) {
    if (mlx_array_dtype(logits) == MLX_FLOAT32) return mlx_array_set(out, logits) == 0;
    return mlx_astype(out, logits, MLX_FLOAT32, stream) == 0;
}

/* FASE 1b.4.3 (Gemma4): RMSNorm SENZA peso apprendibile (RMSNormNoScale in
 * gemma4_text.py), applicata ai valori (V) dell'attenzione -- verificato che è
 * applicata su TUTTI i layer (sliding e full), non solo quelli full/global come
 * ipotizzato inizialmente nel piano (corretto dopo aver riletto Attention.__call__
 * per intero: `if self.has_kv:` include entrambi i tipi di layer in questo modello,
 * dato che num_kv_shared_layers=0). Non serve una nuova primitiva MLX: un peso
 * costante a 1.0 passato a mlx_fast_rms_norm produce esattamente la stessa
 * normalizzazione senza riscalatura, dato che moltiplicare per 1 è un no-op. */
bool mlx_fast_rms_norm_no_scale(mlx_array* res, mlx_array x, int dims, float eps, mlx_stream stream) {
    /* Cache del peso ones (per (dims, dtype)): prima veniva ricreato+castato a OGNI chiamata
     * -> 1 AsType/token per norm (colla dispatch). Il valore non dipende dai dati. */
    static mlx_array g_ones_cache = {0};
    static int g_ones_dims = -1;
    static int g_ones_dtype = -1;
    mlx_dtype want = g_bf16_all ? g_act16 : MLX_FLOAT32;
    if (g_ones_cache.ctx != NULL && (g_ones_dims != dims || g_ones_dtype != (int)want)) {
        mlx_array_free(g_ones_cache); g_ones_cache = mlx_array_new();
    }
    if (g_ones_cache.ctx == NULL) {
        float* ones = (float*)malloc((size_t)dims * sizeof(float));
        for (int i = 0; i < dims; i++) ones[i] = 1.0f;
        const int shape[1] = {dims};
        mlx_array ones_f = mlx_array_new_data(ones, shape, 1, MLX_FLOAT32);
        free(ones);
        if (mlx_astype(&g_ones_cache, ones_f, want, stream) == 0) { g_ones_dims = dims; g_ones_dtype = (int)want; }
        mlx_array_free(ones_f);
    }
    mlx_array ones_w = mlx_array_new();
    bool ok = g_ones_cache.ctx != NULL;
    if (ok) ok = mlx_array_set(&ones_w, g_ones_cache) == 0;
    mlx_array xc = mlx_array_new();
    if (ok) ok = (g_bf16_all && mlx_array_dtype(x) != g_act16)
                     ? (mlx_astype(&xc, x, g_act16, stream) == 0)
                     : (mlx_array_set(&xc, x) == 0);
    int rc = ok ? mlx_fast_rms_norm(res, xc, ones_w, eps, stream) : 1;
    mlx_array_free(ones_w);
    mlx_array_free(xc);
    return rc == 0;
}




/* FASE 1b.0: un'architettura di modello = attenzione + FFN per un layer.
 * `attn` fa input_layernorm->QKV->QK-norm->RoPE->cache->SDPA->o_proj->residual,
 * ritorna l'hidden state POST-attenzione (h) in *out_h. `ffn` fa
 * post_attention_layernorm->(denso o MoE)->residual, ritorna l'hidden state
 * finale del layer in *out. Stesso principio che llama.cpp stesso usa (un
 * build_graph per famiglia, selezionato per nome) -- vedi
 * Fase1b-MLX-Multi-Architecture-Plan.md §2. */

/* Dichiarazioni in avanti: le implementazioni sono la Fase 1b.0 (qwen3_moe,
 * estratta identica da quella che prima era l'unico layer_forward monolitico)
 * -- vedi le definizioni dopo layer_forward. */

int llm_ai_mlx_model_validate(llm_ai_mlx_model_handle handle);
bool qwen3_moe_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                      mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool qwen2_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                           mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool gemma4_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                    mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool mixtral_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                     mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool llama_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                   mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool gdn_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool qwen3_next_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                        mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool qwen3_5_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                    mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool qwen3_next_linear_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                     mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool qwen3_next_attention_dispatch(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                            mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool qwen3_5_moe_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out); /* FASE 1b.6: MoE + shared_expert */
bool dense_swiglu_ffn(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);
bool dense_geglu_ffn(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);
bool mixtral_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);
bool gpt_oss_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                    mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h); /* GPT-OSS: q/k/v/o bias + sinks + sliding 128 */
bool gpt_oss_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out); /* GPT-OSS: MoE 32x top4 SwiGLU speciale + bias */
bool glm4_moe_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                     mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h); /* GLM-4.5: bias + partial rotary 0.5 */
bool glm4_moe_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out); /* GLM-4.5: denso (primi N) / MoE 128 + shared + gate sigmoid */

/* DeepSeek-V2 (MLA, Multi-head Latent Attention): attenzione latente + MoE group-limited-greedy. */
bool deepseek_v2_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                        mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool deepseek_v2_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);

/* Kimi Linear: dispatch ibrido MLA-assorbita (full) / KDA Gated DeltaNet (linear). */
bool kimi_linear_attention_dispatch(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                           mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool kimi_linear_ffn_dispatch(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);
bool kimi_mla_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                     mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);

/* Kimi Linear: i layer KDA (kda_layers) riusano la cache ausiliaria m->aux_cache; la KDA
 * stessa resta da implementare (errore ESPLICITO, mai silent-wrong). Definizione di
 * kimi_linear_ffn_dispatch dopo la struct (usa m->cfg). Vedi
 * `Todo/(open) - MLX-Engine-Lavoro-Residuo.md` §C. */

bool qwen4_exp_attention_dispatch(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                          mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool qwen4_exp_ffn_dispatch(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);



/* FASE 1b (Qwen2, es. DeepSeek-R1-Distill-Qwen-32B): stessa attenzione GQA di
 * qwen3_moe_attention_block MA senza QK-norm (Qwen2 non ha q_norm/k_norm --
 * verificato: il lookup fallirebbe, i tensori non esistono nei pesi reali) e
 * CON bias reale additivo su q/k/v_proj (Qwen2: `bias=True` in mlx_lm/models/
 * qwen2.py, un tensore self_attn.{q,k,v}_proj.bias distinto dalle "biases" di
 * quantizzazione sempre presenti -- verificato sul modello reale, entrambi i
 * tensori coesistono con nomi diversi: .bias singolare vs .biases plurale).
 * o_proj non ha bias (bias=False in mlx_lm), FFN riusa dense_swiglu_ffn
 * invariata (stessa formula MLP, stesso naming tensori). */
/* Forward decl: definite più sotto (introvicate da llama_attention_block) -- servono al
 * wiring LLMMETAL_KVMETAL esteso a qwen2 (DeepSeek). Ri-dichiarare populate_batch/step e' innocuo
 * (prototipi identici); llm_mlx_max_abs_diff non aveva dichiarazione precedente. */
bool llm_mlx_kvmetal_populate_batch(void** kvmetal_ctx_arr, int layer_idx, LlmMlxModel* m,
                                            int cache_offset, int n_tokens,
                                            mlx_array k_rope, mlx_array v_t);
bool llm_mlx_kvmetal_step(void** kvmetal_ctx_arr, int layer_idx, LlmMlxModel* m, int cache_offset,
                                  mlx_array q_rope, mlx_array k_rope, mlx_array v_t,
                                  mlx_array* attn_out);
/* LLM_KVSEG (#61): cache segmentata main+segmento — dichiarazioni avanti (def. ~riga 200). */
bool llm_mlx_kvseg_step(mlx_stream stream, int nkv, int hd, int layer_idx, int cache_offset, int n_tokens,
                               mlx_array k_rope, mlx_array v_t,
                               mlx_array* k_cache_out, mlx_array* v_cache_out,
                               mlx_array* k_piece1, mlx_array* v_piece1,
                               mlx_array* k_piece2, mlx_array* v_piece2,
                               int* n_pieces);
bool llm_mlx_kvseg_merge_two(LlmMlxModel* m, mlx_array q_rope,
                                    mlx_array k1, mlx_array v1,
                                    mlx_array k2, mlx_array v2,
                                    mlx_array* out);
void llm_mlx_kvseg_reset_all(void);


/* GPT-OSS attention (gpt_oss, es. gpt-oss-20b-MXFP4-Q8): attenzione GQA standard
 * con BIAS lineari su q/k/v/o, RoPE full head_dim (rope_theta 150000, nessun partial),
 * attention SINKS appresi (self_attn.sinks, per-head) e sliding window sui layer
 * sliding (alternati). Reference: mlx_lm/models/gpt_oss.py AttentionBlock. I sinks
 * NON sono un no-op: MLX li prepende ai key (fast.cpp:765-783), softmax + slice. */


/* Forward decl: definite più sotto (vicino a llama_attention_block, che le ha introdotte
 * per prima) -- servono qui per il wiring LLMMETAL_KVMETAL esteso a GLM-4.5-Air. */
bool llm_mlx_kvmetal_populate_batch(void** kvmetal_ctx_arr, int layer_idx, LlmMlxModel* m,
                                            int cache_offset, int n_tokens,
                                            mlx_array k_rope, mlx_array v_t);
bool llm_mlx_kvmetal_step(void** kvmetal_ctx_arr, int layer_idx, LlmMlxModel* m, int cache_offset,
                                  mlx_array q_rope, mlx_array k_rope, mlx_array v_t,
                                  mlx_array* attn_out);






/* G1 (piano antigravity 2026-09-08): tenta il kernel Metal che fonde reshape +
 * transpose + RoPE di Q/K e transpose di V (8 dispatch -> 1) nel decode a UN solo
 * token. Produce q_rope/k_rope/v_t gia' in layout cache [1,H,1,D], quindi e'
 * trasparente sia per il ramo concat sia per LLM_KVSLICE. RoPE: stesso branch del
 * path originale -- llama3 usa la tabella freqs (llm_mlx_llama3_rope_freqs, riusata
 * SENZA duplicare la formula), altrimenti la formula a base con rope_theta.
 * Ritorna false SENZA toccare nulla di visibile al chiamante quando la geometria
 * non e' supportata o il launch fallisce: il chiamante ripiega sull'originale.
 * NON puo' essere usato per prefill (n_tokens>1): il kernel calcola la posizione
 * come t+cache_offset, che e' il contratto del decode mono-token. */
/* Cache della tabella RoPE "llama3" (3 fasce). Ritorna un handle RETAINED (il chiamante fa
 * mlx_array_free come prima); la copia del modello resta viva fino a llm_mlx_model_destroy. */

/* FASE 1b.5 (Qwen3.8 / qwen3_5): GatedDeltaNet linear-attention block.
 * Implementa esattamente il forward verificato per parità (MAX DIFF ~1e-6) in
 * native/llmengine_mlx/gdn_parity.py. Il layer lineare fa:
 *   h = input_layernorm(x)
 *   qkv = in_proj_qkv(h)                 # [1,S,conv_dim]
 *   z   = in_proj_z(h).reshape(1,S,Hv,Dv)
 *   b   = in_proj_b(h); a = in_proj_a(h)
 *   conv_out = silu(conv1d(concat([conv_state, qkv])))   # depthwise, groups=conv_dim
 *   q,k,v = split(conv_out, [key_dim, 2*key_dim]) reshaped a teste
 *   inv = Dk**-0.5; q = inv^2 * rms_norm(q); k = inv * rms_norm(k)
 *   beta = sigmoid(b); g = exp(-exp(A_log) * softplus(a + dt_bias))
 *   y,state = gated_delta_update(q,k,v,g,beta,state)     # ricorrente su S
 *   out = out_proj(RMSNormGated(y, z))
 * NOTA stato: per ora il blocco parte da stato zero (equivalente a un prefill da
 * zero); l'integrazione dello stato ricorrente persistente per il decode
 * multi-step è il passo successivo (v. commento di testa).
 */




/* Un layer completo (attention + FFN) per uno o più token, per l'architettura
 * risolta in m->arch (Fase 1b.0: sola "qwen3_moe" in ARCH_TABLE, comportamento
 * IDENTICO a prima -- attn+ffn erano un'unica funzione monolitica, ora due,
 * ma stesso identico calcolo). x/out: [1, n_tokens, hidden]. */
/* FASE 2b: persistent per-session KV-cache (see llm_ai_mlx_create_session).
 * k_cache/v_cache/has_cache sono array a dimensione fissa (n_layers,
 * allocati una volta alla creazione della sessione, mai ridimensionati). */
/* §G-ter: checkpoint/rewind su disco -- definite nella sezione prefix cache (piu' sotto). */


/* Embedding lookup per un singolo token: stesso identico schema di
 * llm_ai_mlx_embedding_checkpoint (gather della riga PRIMA di
 * dequantizzare).
 * FASE 1b.3 (Mixtral): `embed_w` può non essere quantizzato -- verificato:
 * Mixtral-8x22B-4bit ha solo ".weight" per embed_tokens, nessuna ".scales"/
 * ".biases" accanto (a differenza di ogni altro tensore del layer e di
 * embed_tokens in tutti gli altri modelli reali verificati finora, tutti
 * quantizzati) -- pratica comune per l'embedding, dove la precisione della
 * lookup pesa più del risparmio di memoria. `embed_s.ctx == NULL` segnala
 * questo caso (load_output_tensors lascia embed_s/embed_b come mlx_array_new()
 * mai riassegnati quando non trova ".scales"/".biases") -- stesso schema
 * "may be null" già usato altrove in questo file per mlx_array opzionali
 * (es. no_lhs in *_ffn_block). Quando non quantizzato, si salta interamente
 * il passaggio mlx_dequantize: la riga presa da embed_w è già il vettore
 * float finale. */


llm_ai_mlx_session_handle llm_ai_mlx_create_session(llm_ai_mlx_model_handle model_handle) {
    if (model_handle == NULL) return NULL;
    LlmMlxModel* m = (LlmMlxModel*)model_handle;
    LlmMlxSession* s = llm_mlx_session_create(m);
    /* P4: registra la sessione attiva (conto multi-sessione del budget). */
    if (s != NULL && m->p4_nsessions < 64) m->p4_sessions[m->p4_nsessions++] = s;
    return s;
}

void llm_ai_mlx_destroy_session(llm_ai_mlx_session_handle handle) {
    if (handle == NULL) return;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    /* P4: deregistra (swap-remove) PRIMA di liberare. */
    LlmMlxModel* m = s->model;
    if (m != NULL) {
        for (int i = 0; i < m->p4_nsessions; i++) {
            if (m->p4_sessions[i] == s) {
                m->p4_sessions[i] = m->p4_sessions[m->p4_nsessions - 1];
                m->p4_nsessions--;
                break;
            }
        }
    }
    llm_mlx_session_destroy(s);
}



/* D16bis (2026-09-06, vedi piano Proposta-614GBs-Banda-Metal-DeepDive.md §9 "RapidMLX --
 * analisi codice reale"): replica lo schema async_eval+doppio buffering di mlx_lm/rapidMLX
 * (mai mx.compile) -- verificato: parita' greedy IDENTICA su gemma-4-31b-it-4bit (16/64 token),
 * +5.5% pulito a macchina scarica (18.7->19.8 tok/s, 3 run A/B interleaved). Tocca SOLO il loop
 * esterno di generate (mai le funzioni *_attention_block per-famiglia), quindi si applica a
 * tutte le architetture -- nessuna ragione documentata per tenerlo spento. DEFAULT ON dal
 * 2026-09-07; LLM_NO_ASYNC_DECODE=1 = fallback esplicito al path eager sincrono se serve
 * isolare un sospetto di regressione. */


/* Prefix cache (paragrafo 8.4/8.10 del piano): salva lo stato di prefill della sessione

 * (cache KV + stato GDN, history, pending) sotto un nome sul modello, per riusarlo tra

 * sessioni ed evitare di ricalcolare un prefisso stabile (es. system prompt). Le cache sono

 * alias MLX (refcount): la sessione che le carica non le possiede, il prefisso le libera.

 * Ritorna 0 ok, -1 errore.

 */

/* §G — TIER SSD DELLA KV CACHE. Abilitazione lazy via opzione di registro `kvdisk`
 * (off|on|<path>, default OFF); `kvdisk_cap_mb` limita i byte VIVI su disco (LRU).
 * Vedi llm_mlx_kvdisk.h. */





/* ============================================================================
 * CORE MICRO-BENCH (model-free) — 2026-09-13.
 *
 * Misura OGNI wrapper delle primitive core del host contro la CHIAMATA MLX DIRETTA
 * equivalente, con input sintetici DETERMINISTICI (stessa forma/dtype dei layer reali).
 * Obiettivo: isolare il core — se il wrapper e' lento (ratio llm/raw >> 1) o scorretto
 * (maxdiff non ~0), il problema e' QUI, senza montare un modello.
 * Output TSV (una riga per op):
 *   OP <name> shape=<...> T=<...> raw_ms=<x> llm_ms=<x> ratio=<g/r> maxdiff=<x>
 * ratio = llm/raw (>1 = wrapper piu' lento del MLX diretto); maxdiff = max|llm-raw|.
 * `op` = "rms_norm" | "qlinear" | "sdpa" | "all". iters<1 => 200.
 * ========================================================================== */
