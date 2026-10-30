/* llm_mlx_internal.h — header INTERNO condiviso del motore MLX (engine LLMEnv).
 *
 * Nato dallo split di llmengine_mlx.c (30k righe) in domini (AGENTS §0quater: un .c per
 * dominio + header interno condiviso, modello llm_parser_internal.h). Qui vivono i TIPI
 * condivisi fra i domini; le funzioni passano da static a extern man mano che si estraggono
 * i domini. NON e' l'ABI pubblica (quella e' llmengine.h / llmengine.h).
 */
#ifndef LLM_MLX_INTERNAL_H
#define LLM_MLX_INTERNAL_H

#include "mlx/c/mlx.h"
#include "llmengine.h"
extern void* g_mlx_release_dummy;
extern char g_qmm_tag[96];  /* diagnostic: ctx del dummy di release */             /* ABI comune: engine esterni DS4/DS4.1 (deleg) */
#include "llm_mlx_granite_cfg.h"
#include "llm_mlx_mamba_cfg.h"
#include "llm_mlx_hybrid_cfg.h"
#include "llm_mlx_radix.h"
#include "llm_mlx_kvpage.h"
#include "llm_mlx_kvdisk.h"
#include "llm_mlx_ngram.h"         /* NGramTable */
#include "llmcodec_rolemap.h"           /* LlmCodecRoleMap (codec) */
#include "llmdesc.h"              /* LlmDesc (llm_mlx_resolve_arch) */
#include "llm_mlx_packed_qkv.h"    /* LlmMlxPackedQkv */
#include "llm_mlx_qwen4exp.h"      /* LlmQwen4ExpQSACache */
#include "../../LLMMetal/oq_a8/llmmetal_oq_a8_bridge.h" /* LlmMetalOqA8Ctx (self-test eval) */
#include "oss/llm_oss.h"           /* dominio GPT-OSS: contratto (struct memo + prototipi) */
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#ifndef LLM_SPEC_MAXL
#define LLM_SPEC_MAXL 256
#endif
/* Soglia S per la cattura spec GDN per-posizione (state_seq [B,S,Hv,Dv,Dk]).
 * Va catturata SOLO per il blocco verify (S = draft+1, piccolo): su S grande
 * (es. forward di cattura del ctx DFlash2 sull'intero contesto) lo state_seq
 * misura 4*S*Hv*Dv*Dk byte -> 48*S*... a 71k token = ~223GB (OOM 2026-10-01).
 * Oltre la soglia si usa il percorso non-capture (stato finale, senza sseq). */
#ifndef LLM_SPEC_CAP_MAXS
#define LLM_SPEC_CAP_MAXS 128
#endif
typedef struct LlmBonsaiHadamard LlmBonsaiHadamard;  /* fwd: il monolite include il completo */

typedef struct {
    int hidden_size;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    /* MLA (DeepSeek-V2/V3, Kimi, GLM-5): attenzione multi-head latente. q_proj -> [n_heads,
     * qk_nope+qk_rope]; kv_a_proj_with_mqa -> [kv_lora_rank + qk_rope]; kv_b_proj -> [n_heads,
     * qk_nope+v_head]. RoPE (yarn) solo sui qk_rope_head_dim. Cache: keys [n_heads,
     * qk_nope+qk_rope], values [n_heads, v_head]. V2-Lite: q_lora_rank=0 (nessuna q-lora). */
    int kv_lora_rank;             /* rango latente KV (512) */
    int q_lora_rank;             /* rango latente Q (0 = nessuna, V2-Lite) */
    int qk_nope_head_dim;        /* dim per-head K/Q non-ruotato (128) */
    int qk_rope_head_dim;        /* dim per-head K/Q ruotato (64) */
    int v_head_dim;              /* dim per-head di V (128) */
    bool mla_use_nope;           /* MLA "nope": senza il ramo RoPE decoupled (Kimi Linear) */
    /* Kimi Linear (kimi_linear): layer KDA (Gated DeltaNet) una volta i full_attn_layers. */
    int linear_attn_head_dim;    /* head_dim del KDA (linear_attn_config.head_dim) */
    int linear_attn_num_heads;   /* num_heads del KDA */
    int linear_attn_conv_kernel; /* short_conv_kernel_size (default 4) */
    char moe_router_activation[16]; /* sigmoid|softmax (default sigmoid) */
    bool moe_renormalize;        /* rinormalizza i top-k (default true) */
    int moe_layer_freq;          /* ogni quanti layer c'e' un MoE (default 1) */
    int n_experts;
    int top_k;
    int moe_intermediate_size;
    int intermediate_size;        /* Q2: FFN denso (dense_swiglu_ffn/dense_geglu_ffn); per le MoE
                                   * coincide con moe_intermediate_size, ma per i densi/ibridi
                                   * e' l'UNICA fonte della larghezza del mid (letto a parte). */
    int n_shared_experts;         /* GLM-4.5: esperti condivisi nel MoE (1) */
    int first_k_dense_replace;    /* GLM-4.5: primi N layer con MLP denso (1) */
    int n_group;                  /* GLM-4.5: group-pruning router (n_group>1 = attivo) */
    int topk_group;               /* GLM-4.5: gruppi tenuti dopo il pruning */
    float routed_scaling_factor;  /* GLM-4.5: scaling dei pesi router (default 1.0) */
    int vocab_size;
    int quant_group_size;
    int quant_bits;
    char quant_mode[16]; /* affine | mxfp4 | mxfp8 | nvfp4 (mlx quantized_matmul mode) */
    char layer_prefix[48]; /* prefisso dei layer ("model.layers." default; DFlash2 usa "layers.") */
    /* DFlash2 DRAFT: l'attenzione/MLP operano a dim ridotta (es. 640) mentre hidden/norm
     * sono 5120 (down-proiezione). 0 = usa hidden_size (nessuna down-proiezione). */
    int attn_dim;
    /* DFlash2 DRAFT (config.json["dflash_config"]; 0 = non dflash): block-diffusion drafter
     * (Qwen3.8-27B-DFlash2). La presenza di dflash_config setta gia' layer_prefix="layers."
     * (vedi load_config). Campi usati dal loader/forward dflash2 dedicato. */
    int dflash_block_size;
    int dflash_conv_kernel;
    int dflash_conv_group;
    int dflash_selector_rank;
    int dflash_selector_top_k;
    int dflash_mask_token;
    int dflash_n_ctx_layers;      /* numero di layer target da catturare (5) */
    int dflash_ctx_layers[8];     /* id layer target (es. [5,19,33,47,61]) */
    float rms_norm_eps;
    float rope_theta;
    bool norm_topk_prob;
    bool use_expert_bias;         /* LFM2-MoE: expert_bias applicato SOLO alla selezione top-k (non ai pesi) */
    /* MoE v2 (percorso parametrizzato): selezione layer densi/MoE + shared-expert + gate act. */
    int moe_layer_interval;       /* >0: layer MoE se (i+1)%interval==0 && start<=i<=end; 0: usa first_k_dense_replace */
    int moe_layer_start;          /* ernie: moe_layer_start_index */
    int moe_layer_end;            /* ernie: moe_layer_end_index (-1 = n_layers-1) */
    char moe_shared_base[24];     /* base name shared expert (default "mlp.shared_expert"; ernie "mlp.shared_experts") */
    int moe_shared_gate;          /* 1 = esiste <base>_gate (qwen); 0 = assente (ernie) */
    int moe_gate_sigmoid;         /* 1 = gate_act sigmoid; 0 = softmax */
    /* MoE v2 — indirezione nomi + router plain + expert_bias di SELEZIONE (Stadio 1, 2026-10-17).
     * Tutti i campi sono VUOTI/0 di default (memset in load_config) => comportamento attuale
     * invariato. Servono per assorbire le famiglie con naming/semantica diversi (mixtral/lfm2...). */
    char moe_norm_name[48];       /* suffix post-norm layer, dopo layer_prefix; vuoto = "post_attention_layernorm.weight" */
    char moe_router_name[48];     /* nome del router (mlp.gate); vuoto = "mlp.gate" */
    char moe_experts_base[48];    /* base degli esperti stacked (mlp.switch_mlp); vuoto = "mlp.switch_mlp" */
    char moe_dense_base[48];      /* base del MLP denso dei layer dense-first (mlp); vuoto = "mlp" */
    int moe_router_plain;         /* 1 = router NON quantizzato (plain_linear, es. mixtral); 0 = quantizzato */
    char moe_expert_bias_name[48];/* suffix del tensore expert_bias (es. "feed_forward.expert_bias"); vuoto = assente.
                                   * Attivo solo se anche use_expert_bias e' true: il bias entra SOLO nella
                                   * selezione top-k, mai nei pesi (come mlx_lm LFM2-MoE). */
    char moe_expert_bias_alt[48]; /* fallback del nome expert_bias (GLM-4.5: "mlp.gate.e_score_correction_bias") */
    int moe_group_prune;          /* 1 = group-pruning router (n_group/topk_group), come GLM-4.5 */
    int moe_gate_f32;             /* 1 = attivazione del router calcolata in f32 (glm: sigmoid(gates.astype(f32))) */
    int moe_quant_derive;         /* 1 = deriva group/bits/mode PER-TENSORE da scales (router ed esperti),
                                   * come LFM2-MoE (router 8-bit in modello 4-bit); 0 = usa il quant globale */
    float moe_residual_multiplier;/* scala del residuo dell'FFN (es. granitemoe 0.22); 0.0 = 1.0 (nessuna scala) */
    int parallel_residual;        /* 1 = GPT-NeoX parallel residual: out = attn(ln1(x)) + mlp(ln2(x)) + x */
    int learned_pos_emb;          /* 1 = position embeddings APPRESE (wpe) sommate agli embed (gpt2/gpt_bigcode) */
    float query_pre_attn_scalar;  /* gemma3: scale SDPA = query_pre_attn_scalar^-0.5 (0 = 1/sqrt(head_dim)) */
    /* openelm: n_heads/n_kv_heads PER-LAYER (liste nel config). NULL = uniforme (cfg.n_heads/n_kv_heads). */
    int* layer_n_heads;
    int* layer_n_kv_heads;
    bool tie_word_embeddings;
    /* FASE 1b.0: da config.json["model_type"] -- seleziona la entry di ARCH_TABLE
     * (vedi LlmMlxArchOps sotto). Buffer fisso: i valori reali osservati
     * ("qwen3_moe", "qwen3", "llama", "mistral", "gemma2", ...) sono corti. */
    char model_type[64];

    /* FASE 1b.4.0 (Gemma4): attenzione con parametri DIVERSI per tipo di layer
     * (sliding vs full/global) -- ignorati (restano a 0/false/NULL) per
     * qualunque altra architettura. Letti da config.json["text_config"] (annidato,
     * non dalla radice come tutti gli altri campi sopra -- vedi load_config). */
    int global_head_dim;          /* head_dim dei layer "full_attention" */
    int num_global_kv_heads;      /* n_kv_heads dei layer "full_attention" quando attention_k_eq_v */
    bool attention_k_eq_v;        /* K=V condiviso sui layer full (non sliding) */
    float rope_theta_sliding;     /* rope_parameters.sliding_attention.rope_theta */
    float rope_theta_full;        /* rope_parameters.full_attention.rope_theta */
    float partial_rotary_factor;  /* rope_parameters.full_attention.partial_rotary_factor (0 = intero, nessun parziale) */
    float final_logit_softcap;    /* 0 = nessun soft-cap sui logit finali */
    float embed_scale;            /* scala l'embedding in ingresso (gemma4: sqrt(hidden_size), 0 = nessuna) */
    int sliding_window;           /* finestra dei layer sliding (gemma4: 1024); 0 = nessun window */
    /* is_sliding_layer[i] = true se layer i è "sliding_attention", false se "full_attention".
     * Letto dall'array layer_types di config.json -- MAI assunto un periodo fisso, anche se il
     * modello reale osservato ne ha uno (5 sliding + 1 full ripetuto). Alloc dinamica (n_layers
     * non noto a compile-time), liberata in llm_mlx_model_destroy. NULL per architetture senza
     * questa distinzione (tutti i layer si comportano uguale). */
    bool* is_sliding_layer;
    /* gemma4 kv-sharing: gli ULTIMI kv_shared_layers layer NON hanno k_proj/v_proj/k_norm e
     * riusano il K/V (post-norm+rope) del primo layer precedente dello stesso tipo. 0 = off. */
    int kv_shared_layers;
    /* gemma4 per-layer inputs (gemma4_text.py: hidden_size_per_layer_input). 0 = off. */
    int gemma4_pli;
    /* ministral3/mistral4: attn-scale llama-4 (rope_parameters.llama_4_scaling_beta) —
     * moltiplicatore per-posizione su Q: 1 + beta*log(1+floor((pos)/mpe_orig)). 0 = off. */
    float llama4_scaling_beta;
    int llama4_mpe_orig;

    /* FASE 1b.2 (Llama 3.x): RoPE con scaling a 3 fasce di frequenza
     * (config.json["rope_scaling"]["rope_type"] == "llama3") -- ignorati (0/false)
     * per qualunque altra architettura, che continua a usare `rope_theta` piatto
     * come base semplice. */
    bool use_llama3_rope;
    float llama3_rope_factor;       /* rope_scaling.factor (tipicamente 32.0) */
    float llama3_low_freq_factor;   /* rope_scaling.low_freq_factor (default 1.0) */
    float llama3_high_freq_factor;  /* rope_scaling.high_freq_factor (default 4.0) */
    float llama3_old_context_len;   /* rope_scaling.original_max_position_embeddings (default 8192) */

    /* RoPE "traditional" (stile Meta/Llama: rotate_half, rotazione a META' testa) vs
     * default (stile GPT-NeoX: coppie (2i,2i+1)). Default false (tutte le arch verificate
     * finora usano il default). ERNIE-4.5 lo usa ESPLICITAMENTE (mlx_lm/models/
     * ernie4_5.py: initialize_rope(..., traditional=True)). */
    bool rope_traditional;

    /* FASE 1b.7 (YaRN, 2026-09-06): rope_scaling.rope_type == "yarn" (replica 1:1
     * YarnRoPE in mlx_lm rope_utils.py). Usato da gpt-oss-20b/120b (factor 32, orig 4096,
     * beta_fast 32/beta_slow 1, base=rope_theta 150000) e Ternary-Bonsai-4B (qwen3, factor 4,
     * orig 8192, base=rope_theta 5000000). Default 0/false per ogni altra architettura. */
    bool use_yarn_rope;
    float yarn_factor;              /* rope_scaling.factor */
    int yarn_orig_max;              /* rope_scaling.original_max_position_embeddings */
    float yarn_beta_fast;           /* rope_scaling.beta_fast (default 32) */
    float yarn_beta_slow;           /* rope_scaling.beta_slow (default 1) */
    float yarn_mscale;              /* rope_scaling.mscale (default 1; mscale_all_dim non usato, default 0) */
    float yarn_mscale_all_dim;      /* rope_scaling.mscale_all_dim (DeepSeek: il mscale effettivo e' il
                                     * rapporto mscale/mscale_all_dim; V2-Lite = 0.707/0.707 = 1.0) */

    /* FASE 1b.5 (Qwen3.8 / qwen3_5 hybrid GDN): attenzione lineare GatedDeltaNet
     * su layer lineari + self-attention standard su layer full. Parametri letti da
     * config.json["text_config"] (annidato). Tutto a 0/NULL per architetture senza GDN. */
    int full_attention_interval;      /* ogni N layer e' full (3 lineari : 1 full con 4) */
    int linear_num_key_heads;         /* linear_num_key_heads (es. 16) */
    int linear_num_value_heads;       /* linear_num_value_heads (es. 48) */
    int linear_key_head_dim;          /* linear_key_head_dim (es. 128) */
    int linear_value_head_dim;        /* linear_value_head_dim (es. 128) */
    int linear_conv_kernel_dim;       /* linear_conv_kernel_dim (es. 4) */
    LlmCfgGranite granite;   /* famiglia granite (vedi llm_mlx_granite_cfg.h) */
    LlmCfgMamba mamba;   /* famiglia SSM/mamba (vedi llm_mlx_mamba_cfg.h) */
    LlmCfgHybrid hybrid; /* jamba/olmo_hybrid (vedi llm_mlx_hybrid_cfg.h) */
    float falcon_h1_out_scale;        /* falcon_h1: lm_head_multiplier/embedding_multiplier (uscita tied) */
    /* is_linear_layer[i]=true se layer i e' linear_attn (GDN), false se full_attention. */
    bool* is_linear_layer;
    /* FASE 1b.5 (Qwen3.8 / qwen3_5 hybrid GDN): i layer full usano Qwen3NextAttention
     * con output gate -- q_proj produce n_heads*head_dim*2 (queries + gate), poi
     * o_proj(sdpa(...) * sigmoid(gate)). attn_output_gate=true dal config. */
    bool attn_output_gate;
    /* qwen4_exp GDN output gate: activation = output_gate_type or hidden_act; il reference usa
     * sigmoid SE activation=="sigmoid", altrimenti silu (Qwen4Exp SigmoidRMSNormGated). */
    bool q4_gate_silu;
    /* E3 KV quant (opzionale, default OFF): quando la KV-cache supera la soglia,
     * quantizza a Q8/Q4 (policy Recent FP16 / Cold Q8-Q4). Campi letti da config
     * provider (AI MODEL LOAD {kvQuantize:true, kvQuantizeBits:8, kvQuantizeThreshold:N}). */
    bool kv_quantize;
    int kv_quantize_bits;
    int kv_quantize_threshold; /* token oltre cui quantizza la parte cold */
    /* AUTO (2026-09-29): attiva la KV int8 Splash da sola quando il contesto supera
     * `kv_quantize_auto_trigger` (il crossover misurato e' ~8-10k token: sotto conviene fp,
     * sopra l'int8 e' ~4x piu' veloce e ~1.7x meno memoria). Solo per modelli compatibili. */
    bool kv_quantize_auto;
    int kv_quantize_auto_trigger;
    int kv_max_tokens;         /* Rolling-Window KV (fase 3 paging): max token nella KV-cache full; 0=off */
    bool bfloat16_input;       /* matcha mlx_lm: embed/rms_norm in bfloat16 */
    int per_tensor_quant;      /* config quantization contiene override per-tensore (es. gpt-oss-20b MXFP4-Q8: embed/attention affine, esperti mxfp4) */
    /* NSA sparse (opzionale, default OFF): attenzione top-k sul decode dei layer full,
     * riduce il costo da O(N) a O(topk+window) a contesto lungo. Campi letti da config
     * provider (AI MODEL LOAD {nsaSparse:true, nsaTopk:N, nsaWindow:M}). */
    bool nsa_sparse;
    int nsa_topk;
    int nsa_window;
    /* §G-bis leva 1 — sparse page-level attention: config in LlmMlxModel (m->pagesel_on, ...). */
    /* Qwen4-Exp (Flash-Next, qwen4_exp): hyper-connection (4 stream) + QSA (Qwen Sparse
     * Attention, indexer dedicato). Nomi presi 1:1 da TextModelArgs del reference rapidMLX
     * (vllm_mlx/models/qwen4_exp.py) -- 0 per ogni altra architettura. is_linear_layer (gia'
     * esistente sopra, riusato) distingue "linear_attention" (GDN, true) da
     * "qwen_sparse_attention" (QSA, false) invece di lineare/full generico. */
    int hc_count;                /* hc_count (4): stream della hyper-connection */
    int hc_lowrank;              /* hc_lowrank (320): bottleneck di GatedResidual */
    int indexer_n_heads;         /* indexer_n_heads: teste query dell'indexer QSA */
    int indexer_kv_heads;        /* indexer_kv_heads: SEMPRE 1 nella reference */
    int indexer_head_dim;        /* indexer_head_dim */
    int indexer_budget;          /* indexer_budget: token oltre cui l'attenzione si sparsifica */
    int indexer_compress_ratio;  /* indexer_compress_ratio: token per blocco compresso */
    /* N-gram/PLE (qwen4_exp) -- WIRED 2026-09-06. ple_layer_ids e' 1-indexed come il
     * reference (layer_index+1). eos_token_id qui e' SOLO il primo se il config ne
     * elenca piu' d'uno (`_shift_right_ignore_eos` nella reference usa comunque un solo
     * id, `self.eos_token_ids[0]`). */
    int* ple_layer_ids;
    int ple_layer_ids_n;
    int ple_embed_dim;                    /* ple_embed_dim (default hidden_size se assente) */
    int ple_conv_kernel_size;             /* ple_conv_kernel_size */
    int ngram_size;                       /* ngram_size (bigram+trigram = 3 -> context_len=2) */
    int heads_per_ngram;                  /* heads_per_ngram */
    int64_t ngram_vocab_size_base;        /* ngram_vocab_size_base (~20M) */
    int make_ngram_vocab_size_divisible_by;
    int split_ngram_parts;                /* riferimento storico (reference usa 128 shard;
                                            * LLMEnv carica una tabella unica, vedi
                                            * llm_mlx_qwen4exp_ngram_table_path) */
    int64_t ngram_seed;                   /* seed (per ricostruire layer_multipliers se il
                                            * checkpoint non li fornisce gia' come tensori --
                                            * QUESTO checkpoint reale li fornisce, vedi loader) */
    int64_t eos_token_id;                 /* primo eos_token_id (puo' essere lista nel config) */
struct LlmMlxSession;   /* P4: forward (registro sessioni nel modello) */
} ModelConfig;

struct LlmMlxModel;   /* fwd: definito nel monolite (in fase di split) */
typedef struct LlmMlxModel LlmMlxModel;

typedef struct LlmMlxSession {
    LlmMlxModel* model; /* borrowed, not owned -- caller keeps the model alive */
    mlx_array* k_cache;
    mlx_array* v_cache;
    /* Cache AUSILIARIA per-layer (2026-09-12): stato per architetture che NON usano solo K/V.
     * - KDA (Kimi Linear): 3 stati conv + 1 stato ricorrente, impacchettati per layer.
     * - DSA indexer (DeepSeek-V3.2/V4): la cache K dell'indexer.
     * Layout opaco, interpretato dal blocco attention che la usa. Inizializzata vuota come
     * k/v_cache; nessun costo se non usata. */
    mlx_array* aux_cache;
    bool* has_cache;
    int n_layers;
    /* KV quant (E3, risparmio memoria a contesto lungo): quando la cache di un layer supera la
     * soglia, la parte COLD (piu' vecchia) viene mantenuta QUANTIZZATA (w/sc/bi) invece di FP16.
     * La SDPA dequantizza la parte cold e concatena con i recenti FP16 al volo. kvq_len[layer]
     * = numero di token cold quantizzati (0 = cache non quantizzata). Default OFF (no-alloc). */
    int* kvq_len;        /* per layer: token cold quantizzati (0 = non quantizzato) */
    mlx_array* kvq_w;    /* per layer: weight packed quantizzato della parte cold */
    mlx_array* kvq_sc;   /* per layer: scales */
    mlx_array* kvq_bi;   /* per layer: biases */
    int kvq_bits;        /* 4 o 8 (K) */
    /* HETERO-KV (LeanKV, arXiv 2412.03131 §3.1): K piu' preciso di V (le key influenzano
     * TUTTI gli attention score; le value solo il proprio token). Quando attivo (env
     * LLM_KV_HETERO=K8V4|K8V2|K4V2), V viene quantizzato a `kvq_v_bits` (< kvq_bits).
     * Default OFF: kvq_v_bits = kvq_bits e la cache V resta FP16 se `kvq_v_w` non allocato
     * (comportamento storico K-only). LLM_KV_HETERO_VERIFY=1: calcola entrambi e stampa il diff. */
    int kvq_v_bits;
    int* kvq_v_len;      /* per layer: token cold quantizzati di V (0 = non quantizzato) */
    mlx_array* kvq_v_w;  /* per layer: weight packed quantizzato della parte cold di V */
    mlx_array* kvq_v_sc; /* per layer: scales V */
    mlx_array* kvq_v_bi; /* per layer: biases V */
    int cache_offset;
    int pinned_tokens; /* F1 GDN — pin in-memory dei primi n_tokens (non evictabili) */
    int32_t pending_next_id; /* predicted continuation after the last append/generate call */
    bool has_pending;        /* false until at least one append has run -- generate() refuses before that */
    /* FASE 3: hidden state whose final_step produced pending_next_id -- kept alive (not just
     * the resulting id) so llm_ai_mlx_session_generate_constrained can re-decide the SAME
     * undecided position with a restricted argmax instead of the one final_step already
     * committed to. Always updated in lockstep with pending_next_id/has_pending, at every site
     * that sets them (session_append's tail, session_generate's loop body, and this function's
     * own tail after accepting a constrained choice). */
    mlx_array pending_h;
    /* Per-session sampling. sampling_set==false keeps historical greedy-argmax
     * behavior. temperature<=0 means greedy (deterministic) even when set. */
    bool sampling_set;
    float sampling_temperature;
    float sampling_top_p;
    int sampling_top_k;
    float sampling_min_p;    /* min-p (0 = off). Applicato DOPO top_p e prima della temperatura. */
    int64_t sampling_seed;
    float sampling_repetition_penalty;   /* >1.0 penalizza i token gia' visti (default 1.0) */
    float sampling_frequency_penalty;    /* OpenAI frequency_penalty (default 0 = off) */
    float sampling_presence_penalty;     /* OpenAI presence_penalty  (default 0 = off) */
    /* Chunked prefill: dimensione massima dei token processati in un singolo
     * grafo da llm_ai_mlx_session_append. Default sessione = 8192 (misurato 2026-09-01, vedi
     * commento nel create). 0 = adattivo via llm_ai_mlx_session_set_prefill_chunk (2048 dense /
     * 4096 MoE). Override env LLM_PREFILL_CHUNK=N. Prompt piu'
     * grandi vengono processati in sequenza di chunk (KV-cache che cresce),
     * evitando grafi enormi che esplodono la memoria del grafo Metal. */
    int prefill_chunk_size;
    /* N-Gram self-speculative (Â§8.4/Â§8.10): history dei token visti (prefill + generati)
     * + tabella n-gram costruita da essa. Il draft proposto viene verificato dal modello
     * con la Cache ABI (checkpoint/rollback) quando il draft batch sara' integrato. */
    int32_t* history;
    int history_len;
    int history_cap;
    NGramTable ngram;
    bool ngram_built;
    int ngram_order;
    /* Cascade: tabella a ordine BASSO di fallback (il primario a ordine alto spesso non predice;
     * il basso predice di piu'. Il gate `d[0]==pending` di spec_can filtra i falsi positivi). */
    NGramTable ngram_lo;
    bool ngram_lo_built;
    int ngram_lo_order;
    /* ── Speculative streaming (spec_ngram, §8.4) ──────────────────────────────
     * Il consumer streaming (FastLLM CLI/server) chiama generate(max_new_tokens=1) per
     * token: lo spec self-speculativo (draft n-gram verificato con UN forward batch) non
     * si ingaggerebbe. Qui il motore genera in ANTICIPO un chunk di token con
     * llm_ai_mlx_session_generate_ngram e li bufferizza; ogni generate(1) del consumer
     * pesca dal buffer. L'output resta IDENTICO al greedy AR (i draft sono verificati).
     * Se il consumer fa un append esterno (nuovo turno) con token non consumati, la cache
     * K/V viene riportata al chunk-start via checkpoint (alias O(1), nessuna copia) e
     * ri-avanzata dei soli consumati, poi la history viene troncata (spec_discard). */
    int32_t spec_buf[64];
    int spec_buf_len;   /* token validi in spec_buf */
    int spec_buf_pos;   /* prossimo da consumare */
    int spec_ctx_len;   /* history_len all'inizio del chunk corrente */
    int spec_ctx_off;   /* cache_offset all'inizio del chunk corrente */
    mlx_array* spec_ckpt_k;  /* per-layer checkpoint K del chunk-start (alias O(1)) */
    mlx_array* spec_ckpt_v;  /* per-layer checkpoint V del chunk-start */
    bool spec_ckpt_valid;
    /* Larghezza adattiva del draft (da Gufo DFlashLengthController): EWMA dell'acceptance e streak
     * di full-accept per sondare più largo. Deterministico (nessuna timing della richiesta). */
    float spec_acc_ewma;
    bool df2_off;   /* latch di sessione: DFlash2 non conviene (abbinamento sbagliato) -> AR/n-gram */
    int spec_full_streak;
    /* F20: acceptance CUMULATIVA per sessione (metrica di primo livello: draft proposti vs
     * accettati dal target). Esposta da llm_ai_mlx_session_spec_stats(). */
    long long spec_drafts, spec_accepted;
    /* RMLX10 Fase B (2026-09-03): decode-step compilato (mlx_compile). La
     * closure e' creata lazy al primo step compilato e VIVE per tutta la
     * sessione (la cache di compile di mlx core e' keyed per fun_id: riusare
     * la stessa closure = cache-hit a ogni step con le stesse shape). */
    mlx_closure compiled_decode;
    bool compiled_decode_valid;
    /* LLMMETAL_KVMETAL (sperimentale, default OFF, vedi llm_mlx_kvmetal_enabled): pool Metal
     * reale per layer. Allocato SOLO se LLMMETAL_KVMETAL o LLMMETAL_KVMETAL_VERIFY sono attive
     * (altrimenti NULL, zero overhead) -- stessa disciplina di kvq_*. */
    void** kvmetal_ctx; /* per layer: LlmKvMetalCtx* (NULL finche' non creato lazy al primo uso) */
    /* LLM_GEMVQ4 (sperimentale, default OFF, vedi llm_mlx_gemvq4_try): cache dei contesti
     * Metal per tensore di peso affine4 (chiave = puntatore dati grezzi). Allocata SOLO se
     * LLM_GEMVQ4 o LLM_GEMVQ4_VERIFY sono attive (altrimenti NULL, zero overhead). */
    void** gemvq4_ctx;        /* per slot: LlmGemvQ4Ctx* (NULL finche' non creato lazy) */
    const void** gemvq4_key;  /* per slot: chiave (puntatore dati pesi); NULL = libero */
    int gemvq4_cap;           /* n. slot allocati (n_layers*MAX_PER_LAYER + margine) */
    bool parked;              /* §G-ter evict: KV su disco, array RAM liberati (restore on-demand) */
    uint64_t park_key;        /* chiave del blob di park su llm_kv_disk */
    mlx_array* ps_mean;       /* §G-bis pagesel: per-layer [1,nkv,cap,hd] FP32 (media di pagina) */
    int* ps_done;             /* §G-bis pagesel: per-layer token gia' riassunti (multiplo di ps_page) */
    int ps_cap;               /* §G-bis pagesel: capacita' pagine allocata */
    /* Sessione dell'engine ESTERNO (modello delegato: DS4/DS4.1). */
    void* deleg_sess;
} LlmMlxSession;

typedef struct {
    bool active;       /* segmentazione in corso per questo layer */
    int base_len;      /* token nel main (prefill + segmenti chiusi) */
    int seg_len;       /* token nel segmento corrente */
    int seg_cap;       /* capacita' del segmento (llm_mlx_kvseg_size) */
    mlx_array k_main, v_main;  /* [1,nkv,base_len,hd] — stabile tra i flush */
    mlx_array k_seg, v_seg;    /* [1,nkv,seg_cap,hd] — buffer fisso del segmento */
} LlmKvSegState;

typedef struct {
    mlx_map_string_to_array* shards;
    int count;
    int cap;
    /* Non-NULL SOLO per i modelli caricati da GGUF: i pesi sono chiavati col NOME CANONICO GGUF
     * (`blk.N.attn_q.weight`); il forward chiede nomi HF -> multishard_get li traduce via questa
     * role map `.llmarch` (unica mappa HF<->GGUF). NULL per i modelli da dir MLX (chiavi HF). */
    const LlmCodecRoleMap* gguf_rm;
} MultiShardWeights;

typedef struct {
    char** names;
    int count;
    int cap;
} StrSet;

typedef struct {
    bool active;
    int n_layers, n_experts, top_k, gate_cols, down_cols;
    float* gate;   /* [layers*experts*gate_cols] sum x[col]^2 */
    float* down;   /* [layers*experts*down_cols] sum mid[col]^2 */
    int*   cnt;    /* [layers*experts] volte che l'expert e' stato routato */
    /* Q2-b: ATTENTION densa, riga dedicata con expert = -1 (i routed experts sono >= 0):
     * attn_qkv = input normalizzato condiviso da q/k/v (colonne = gate_cols = hidden),
     * attn_o = input di o_proj (colonne = n_heads*head_dim). Vuoti se il modello non li espone. */
    float* attn_qkv;   /* [layers*gate_cols] */
    float* attn_o;     /* [layers*attn_o_cols] */
    int    attn_o_cols;
    int*   attn_cnt;   /* [layers] quante volte il blocco attention ha accumulato */
} ImatrixState;

typedef struct {
    mlx_array ln2;
    mlx_array gate_w, gate_s, gate_b;
    mlx_array up_w, up_s, up_b;
    mlx_array down_w, down_s, down_b;
    bool valid;
    int gemma4;   /* 0 = tname/post_attention_layernorm (llama/qwen3), 1 = tname_gemma4/pre_feedforward */
} GdenseFfnQuant;

typedef struct {
    mlx_array ln1;                 /* input_layernorm.weight */
    mlx_array qw, qs, qb;
    mlx_array kw, ks, kb;
    mlx_array vw, vs, vb;
    mlx_array ow, os, ob;
    mlx_array qn_w, kn_w;          /* QK-norm (se presenti) */
    bool has_qn;
    bool has_v;                    /* v_proj presente (assente sui layer K=V gemma4) */
    LlmMlxPackedQkv packed_qkv;
    bool valid;
} GattnQuant;

typedef struct {
    mlx_array ln2;
    mlx_array rw, rs, rb;
    mlx_array gw, gs, gb;   /* routed experts gate */
    mlx_array uw, us, ub;   /* up */
    mlx_array dw, ds, db;   /* down */
    mlx_array se_w, se_s, se_b;  /* shared_expert gate */
    mlx_array su_w, su_s, su_b;  /* shared_expert up */
    mlx_array sd_w, sd_s, sd_b;  /* shared_expert down */
    mlx_array sg_w, sg_s, sg_b;  /* shared_expert_gate */
    int gg, gbts; char gmode[8];
    int ug, ubts; char umode[8];
    int dg, dbts; char dmode[8];
    bool has_shared;       /* shared_expert presente e cachato */
    bool valid;
    /* LLMENGINE_MOE_GATEUP_FUSE (2026-09-06, vedi piano Proposta-614GBs-Banda-Metal-DeepDive.md §9
     * "RapidMLX -- analisi codice reale" punto 6 e Todo/(open) - Metal-Decode-Fusion-Slice.md):
     * gate_proj e up_proj concatenati UNA VOLTA lungo l'asse out_features (assiome di
     * mlx_gather_qmm con transpose=true) -- un gather_qmm invece di due per token/layer, split
     * del risultato in due meta' via mlx_split. Stessa tecnica di rapidMLX
     * (vllm_mlx/moe_fusion.py, +7.0% decode misurato DA LORO su Qwen3.6-35B-A3B). Popolato SOLO
     * se gate/up hanno la STESSA quantizzazione (group/bits/mode) e la STESSA presenza di bias
     * (altrimenti la concat non e' valida) -- has_gate_up_fused=false in quel caso, il chiamante
     * fa fallback ai due gather_qmm separati.
     * ATTENZIONE: le copie guw/gus/gub sono MATERIALIZZATE e trattenute dalla cache fino allo
     * spegnimento del modello (~955 MB/layer su Mixtral: 805 MB di peso impilato + scale/bias).
     * Costruirle quando il flag e' OFF significa sprecare ~51 GB: misurato 2026-09-09, su
     * Mixtral-8x22B portavano l'attivo Metal da 79,39 a 130,51 GB su un limite di 130,57 e la
     * VM MLX si serializzava su wait_for_one a ogni passo (decode 0,3-1,3 tok/s, processo anche
     * ucciso dal kernel con SIGKILL/exit 137 durante il prefill). Quindi la concat la fa SOLO
     * il ramo fuse; tutti i consumer passano da llm_mlx_moe_gateup_gather che gia' verifica
     * lo stesso flag. */
    mlx_array guw, gus, gub;
    bool has_gate_up_fused;
} GmoeFfnQuant;

typedef struct {

    char name[64];
    int32_t* tokens;
    int token_len;

    mlx_array* k_cache;
    mlx_array* v_cache;
    bool* has_cache;
    mlx_array* aux_cache;  /* per-layer ausiliari della sessione (KDA/MLA k_pe, stash gemma): NULL se la sessione non ne ha */
    bool recurrent;        /* modello con stato ricorrente (GDN/KDA/SSM/conv): lo stato vale SOLO a token_len esatto, quindi il riuso
                            * richiede che TUTTO il prefisso combaci (mai troncare) e non si spilla su disco */

    int cache_offset;
    int32_t* history;
    int history_len;
    int history_cap;

    int32_t pending_next_id;
    bool pending_valid;    /* pending_next_id e' noto: vero se salvato in RAM o ripristinato da un blob che lo porta; falso per un prefisso solo registrato da disco */
    int n_layers;
    uint64_t last_used; /* LRU: timestamp (lru_clock del modello) dell'ultimo uso */
    bool has_kv;        /* true = KV cache presente (salvato in-process); false = caricato
                         * da disco (solo keys): il riuso deve rifare il prefill del prefisso */
    int hits;           /* C5 (HiCache selettivo): quante volte il prefisso e' stato RIUSATO; l'eviction spilla su SSD solo
                         * i prefissi con hits >= opzione `prefix_spill_min_hits` (default 1 = dal 2o uso) */
    bool on_disk;       /* §G: KV spillata su llm_kv_disk (RAM liberata); tokens/name restano,
                         * il riuso la ripristina da disco invece di ri-prefillare */
} SavedPrefix;

typedef struct {
    mlx_array qkv_w,qkv_sc,qkv_b, z_w,z_sc,z_b, a_w,a_sc,a_b, b_w,b_sc,b_b;
    int gsz, bts; const char* mode;
} GdnInProjPayload;

typedef struct {
    mlx_array ln_w;
    mlx_array qkv_w, qkv_sc, qkv_b;
    mlx_array z_w, z_sc, z_b;
    mlx_array a_w, a_sc, a_b;
    mlx_array b_w, b_sc, b_b;
    mlx_array conv_w, A_log, dt_bias, eAlog, norm_w;
    mlx_array out_w, out_sc, out_b;
    int Hk, Hv, Dk, Dv, key_dim, value_dim, conv_dim, cstate_len, repeat_factor;
    int gsz, bts;
    const char* mode;
    float eps;
    mlx_dtype act_dt;
    int gate_metal;
    double inv_scale;
} GdnBlkPayload;

typedef struct {
    mlx_array kw, ks, kb, vw, vs, vb;   /* cold quantizzati [1,nkv,nc,l] / [1,nkv,nc,hd/64] */
    int nc;                             /* token cold (K == V) */
    int kbits, vbits;                   /* bit affine K / V (Hetero-KV) */
    mlx_array ptab;                     /* page table logical->physical (int32), per il kernel paged */
} LlmKvqCold;

typedef struct {
    bool active;
    float temperature;
    float top_p;
    int top_k;
    float min_p;                /* min-p (>0 = attivo): azzera p < min_p*max(p). Dopo top_p, prima di temp. */
    int64_t seed;
    float repetition_penalty;   /* >1.0 = penalizza i token gia' visti (CTRL); 1.0 = off */
    float frequency_penalty;    /* OpenAI: logits[t] -= fp * count(t)      (>=0; 0 = off) */
    float presence_penalty;     /* OpenAI: logits[t] -= pp se count(t) > 0 (>=0; 0 = off) */
    const int32_t* hist;        /* token gia' visti (prompt + generati) per la penalty */
    int hist_n;
} llm_mlx_sampling_t;

typedef struct {
    mlx_array embed_w, embed_s, embed_b;
    mlx_array final_norm_w;
    mlx_array out_w, out_s, out_b;
    bool have_lm_head;
} OutputTensors;

typedef struct { const void* key; char name[256]; mlx_array val; } LlmWCacheEnt;

typedef struct { mlx_array guw, gus, uw, us, dw, ds; bool ready; } LlmPad512;

typedef struct { mlx_array w, s, b; bool ready; } LlmGateUp;

typedef struct { LlmMlxModel* m; LlmMlxSession* s; int layer; } LlmPageselCtx;

typedef struct { float value; int index; } e9_topk_t;

typedef bool (*LlmMlxAttnFn)(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                              mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
typedef bool (*LlmMlxFfnFn)(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);

typedef struct {
    const char* model_type; /* da config.json["model_type"] */
    LlmMlxAttnFn attn;
    LlmMlxFfnFn ffn;
    /* Whole-model forward (DeepSeek V4.1): quando non-NULL il modello NON usa la coppia
     * attn/ffn per-layer ma un forward unico (carrier opaco in m->dsv41), preso in carico nei
     * punti di dispatch (load/session_append/session_generate) per model_type. */
    void* whole;
} LlmMlxArchOps;

typedef struct {
    int32_t cache_offset;
    int32_t pending_next_id;
    bool has_pending;
    int32_t n_layers;
    int32_t hidden_size;
} MlxSwapMeta;

typedef struct {
    bool loaded;
    int post_norm;   /* contract MTPLX hidden_variant: 1=hidden post-final-norm del target */
    mlx_array fc_w,fc_s,fc_b, pne, pnh, iln, pln, qn, kn, mnorm;
    mlx_array q_w,q_s,q_b, k_w,k_s,k_b, v_w,v_s,v_b, o_w,o_s,o_b;
    mlx_array g_w,g_s,g_b, u_w,u_s,u_b, d_w,d_s,d_b;
} LlmMlxMtp;

typedef struct {
    mlx_array x, idx, gw, gsc, uw, usc, dw, dsc, glb, ulb, dlb, sc;
    int din, dff, k; mlx_stream s;
} llm_cb_moe_ctx;

typedef struct {
    mlx_array x, ln1, ln2;
    mlx_array qw, qs, qb, kw, ks, kb, vw, vs, vb, ow, os, ob;
    mlx_array gw, gs, gb, uw, us, ub, dw, ds, db;
    mlx_array kcache, vcache;
    int H, nh, nkv, hd, dff, S, rope_off;
    float eps, rope_theta;
    mlx_stream s;
} llm_cb_layer_ctx;

typedef struct { LlmMlxModel* m; const int32_t* toks; int T; const int32_t* tgt; int li; const uint8_t* mask; } GtCtx;

typedef struct { LlmMlxModel* m; const int32_t *tw, *tgw, *tl, *tgl; const uint8_t *mw, *ml; int Tw, Tl, li; float beta; } GdpoCtx;

typedef struct { mlx_array x, w; float eps; mlx_stream s; } llm_cb_rms_ctx;

typedef struct { mlx_array x, wq, sc, bi; int gs, bits; mlx_stream s; } llm_cb_ql_ctx;

typedef struct { mlx_array q, k, v; float scale; mlx_stream s; } llm_cb_sd_ctx;

typedef struct { mlx_array q, Kc_w, Kc_s, Kc_b, Vc_w, Vc_s, Vc_b, Kr, Vr; float scale; int bits, vbits, gs; mlx_stream s; } llm_cb_kvq_ctx;

typedef struct { mlx_array gate, up; mlx_stream s; } llm_cb_sw_ctx;

typedef struct { mlx_array logits; float cap; mlx_stream s; } llm_cb_ls_ctx;

typedef struct { mlx_array emb_w, emb_s, emb_b, tid; int hidden, gs, bits; const char* mode; mlx_stream s; } llm_cb_emb_ctx;

typedef struct { mlx_array kc, vc, kn, vn; int off; mlx_stream s; } llm_cb_kv_ctx;

typedef struct { mlx_array x, w; mlx_stream s; } llm_cb_pl_ctx;

typedef struct { mlx_array xw, wsc, x; int gs, bits; mlx_stream s; } llm_cb_qm_ctx;

typedef struct { mlx_array x; mlx_stream s; } llm_cb_u_ctx;

typedef struct { mlx_array x, w; int gs; float eps; mlx_stream s; } llm_cb_zc_ctx;

typedef struct { mlx_array h, w, gate; int use_silu; float eps; mlx_stream s; } llm_cb_sg_ctx;

typedef struct { mlx_array x, A, B; float scale; mlx_stream s; } llm_cb_lo_ctx;

typedef struct { mlx_array q, k; float scale; mlx_stream s; } llm_cb_l2_ctx;

typedef struct {
    mlx_array* kcache;        /* ctx keys per layer: [1,n_kv_heads,T,head_dim] */
    mlx_array* vcache;
    int ctx_pos;              /* righe ctx accumulate nella cache (per layer) */
} Df2CtxState;

#include "llm_mlx_linear_packed_model.h"
struct LlmMlxModel {
    char model_dir[1024];   /* MTP: dove sta mtp.safetensors */
    LlmMlxPackedCtx packed; /* M3.2: descrittori packed (usati solo con weights_backend="packed") */
    /* Ternary-Bonsai 2 27B (PrismML): contratto Hadamard del pack (block, sign per width,
     * weight_names da ruotare, inverse_weight_names). Non NULL SOLO se model_type ==
     * prism_hadamard_qwen35. Gli hook per-proiezione che lo USANO sono in corso (vedi todo
     * "(open) - Ternary-Bonsai-2-27B-Supporto-e-Benchmark"): finche' non sono completi l'entry
     * ARCH_TABLE resta assente e il load fa fail-loud (mai output sbagliato in silenzio). */
    LlmBonsaiHadamard* bonsai_had;
    ModelConfig cfg;
    MultiShardWeights weights;
    mlx_device device;
    mlx_stream stream;
    /* Cache della tabella RoPE "llama3" (3 fasce): dipende SOLO da cfg (head_dim, theta,
     * factor, low/high freq, old_context_len), non dal token ne' dal layer. Prima era
     * ricostruita a OGNI chiamata (pow/exp host + upload GPU + nodo nel grafo): su
     * Llama-3.x = 16 ricostruzioni per token. La copia del modello e' l'unico proprietario;
     * i chiamanti ricevono un handle retained (mlx_array_set). memset(0) in model_init ->
     * handle vuoto valido. */
    mlx_array llama3_freqs;
    bool llama3_freqs_valid;
    const LlmMlxArchOps* arch; /* risolta in llm_ai_mlx_load_model da cfg.model_type */
    GmoeFfnQuant* ffn_quant;   /* cache pesi/quant FFN MoE per-layer (lazy, distrutta col modello) */
    int ffn_quant_n;
    GattnQuant* attn_quant;    /* cache pesi attenzione per-layer (lazy) */
    int attn_quant_n;
    GdenseFfnQuant* dense_ffn; /* cache pesi FFN denso per-layer (lazy, LLM_FFN_WCACHE) */
    int dense_ffn_n;
    GptossLayerWE* gptoss_we;  /* memo nome->array per-layer gpt-oss (LLM_GPTOSS_WCACHE) */
    int gptoss_we_n;
    mlx_array* gdn_cw;         /* cache per-layer del peso conv1d GDN gia' castato FP32 (P2 invarianti) */
    int gdn_cw_n;
    mlx_array* gdn_ae;         /* cache per-layer di exp(A_log) GDN (P2 invarianti) */
    int gdn_ae_n;
    mlx_array* gdn_dt;         /* cache per-layer di linear_attn.dt_bias */
    int gdn_dt_n;
    /* RWKV7: carry cross-layer `v_first` (prodotto dal layer 0, letto dai successivi nello
     * stesso forward). Transiente per-forward: sovrascritto dal layer 0 ad ogni chiamata. */
    mlx_array rwkv_v_first;
    /* GPT-2/GPT-BigCode: position embeddings APPRESE (wpe) [n_positions, hidden]. ctx==NULL
     * per tutte le altre arch (solo RoPE). Caricata in load_output_tensors, sommata agli embed
     * nel forward (forward_tokens_batch + forward_one_token_dev). */
    mlx_array pos_emb_w;
    /* gemma4 per-layer inputs (hidden_size_per_layer_input>0): pesi caricati UNA volta
     * (schema pos_emb_w) e il per-layer inputs PROIETTATO del forward corrente
     * [1,T,N,pli], preparato a ogni embed (forward_tokens_batch/forward_one_token_dev)
     * e consumato dal blocco ffn (gate chain). ctx==NULL quando la feature e' off. */
    mlx_array pli_embed_w, pli_embed_s, pli_proj_w, pli_proj_s, pli_norm_w;
    mlx_array pli_all;
    /* Qwen4-Exp (qwen4_exp): stato persistente dell'indexer QSA per-layer (ring raw +
     * chiavi compresse growable + offset) -- NON e' un mlx_array singolo come k_cache/
     * v_cache (schema diverso, vedi LlmQwen4ExpQSACache in llm_mlx_qwen4exp.h), quindi
     * serve un array dedicato invece di riusare k_cache/v_cache. Lazy (init al primo
     * forward del layer via .ring.ctx==NULL), distrutto col modello. */
    LlmQwen4ExpQSACache* qsa_indexer_cache;
    int qsa_indexer_cache_n;
    /* Qwen4-Exp N-gram/PLE: tabella di embedding hashata (`ngram_table.bin`, formato
     * safetensors standard con un marcatore "format":"mlx-serve-ngram" nei metadata -- NON
     * elencato in model.safetensors.index.json, troppo grande per starci insieme al resto:
     * 30 GB su un checkpoint reale, 320M righe quantizzate 4-bit).
     * NON caricata con mlx_load_safetensors+mlx_take_axis: quel percorso passa per
     * mlx/backend/common/load.cpp (Load::eval_cpu), che alloca e legge SEMPRE l'intero
     * tensore prima che un gather possa estrarne anche solo poche righe -- root cause
     * verificata dello swap thrashing/157s-per-token osservato sul checkpoint reale (97GB
     * modello + 30GB tabella su macchina condivisa). Vedi Piano-Correzioni-Modelli-MLX.md
     * §C17 "PIANO -- memoria" (2026-09-07). Sostituito con lettura mirata a righe via
     * pread() sul file grezzo (qwen4_exp_ngram_table_open/qwen4_exp_ngram_read_rows,
     * sopra qwen4_exp_ple_inject) -- l'array intero non esiste MAI in RAM, solo le poche
     * righe richieste per ogni chiamata. ngram_fd<0 se il modello non ha layer PLE
     * (cfg.ple_layer_ids_n==0) o se il file non esiste (i layer PLE si rifiutano
     * esplicitamente, invariato rispetto a prima). */
    int ngram_fd;
    int64_t ngram_data_start;                  /* 8 + lunghezza header JSON, offset assoluto */
    int64_t ngram_w_off, ngram_s_off, ngram_b_off; /* data_offsets[0] di weight/scales/biases */
    int ngram_w_cols, ngram_s_cols, ngram_b_cols;  /* shape[1] di ciascun tensore */
    int ngram_w_itemsize, ngram_s_itemsize, ngram_b_itemsize;
    mlx_dtype ngram_w_dtype, ngram_s_dtype, ngram_b_dtype;
    bool ngram_has_biases;
    bool ngram_table_loaded;
    int ngram_table_bits, ngram_table_group_size; /* da __metadata__ del file, vedi loader */
    int embed_q_group, embed_q_bits;             /* override per-modulo dell'embedding (0 = cfg) */
    int lmhead_q_group, lmhead_q_bits;           /* override per-modulo di lm_head (0 = cfg) */
    /* Qwen4-Exp N-gram/PLE SHARDED: sul checkpoint UFFICIALE mlx-community non esiste un
     * `ngram_table.bin` monolitico -- la stesssa tabella e' inline nei file safetensors
     * principali, divisa row-wise in 128 tensori `...ngram_embedding.shards.N.{weight,
     * scales,biases}` (shard = row // rows_per_shard, come la ShardedEmbedding di
     * riferimento). Quando `ngram_table.bin` manca, qwen4_exp_ngram_table_open_shards
     * compila questa mappa (file/offset per shard) e qwen4_exp_ngram_read_rows legge le
     * righe via pread dalla shard giusta -- STESSO API a valle, mai l'array intero in RAM. */
    int ngram_sharded;              /* true = tabella letta dagli shard inline */
    int ngram_nshards;              /* numero shard effettivi (128 sul checkpoint reale) */
    int64_t ngram_rows_per_shard;   /* righe per shard (shape[0] di un tensore shard) */
    int ngram_nfiles;               /* file safetensors distinti aperti (fd cache) */
    struct { char path[1024]; int fd; int64_t data_start; } ngram_file[16];
    /* Offsets (relativi a data_start del file) delle tre foglie di ogni shard. */
    struct { int w_file, s_file, b_file; int64_t w_off, s_off, b_off; } ngram_shard[256];
    /* Stato persistente del conv1d di PLELayer, per-layer (solo i layer in ple_layer_ids
     * lo usano davvero) -- stesso schema semplice di gdn_cw/gdn_dt (array [n_layers] di
     * mlx_array, ctx==NULL finche' non popolato al primo forward di quel layer). */
    mlx_array* ple_conv_cache;
    int ple_conv_cache_n;
    /* Side-channel per il contesto token durante il forward corrente (N-gram/PLE ha bisogno
     * della storia dei token, non solo dell'hidden state -- unico punto del motore che ne ha
     * bisogno). Impostato dal chiamante (forward_one_token/forward_tokens_batch) subito PRIMA
     * del loop sui layer, letto da qwen4_exp_attention_dispatch SOLO sui layer PLE. Puntatore
     * a un buffer di proprieta' del chiamante (mai posseduto qui, mai liberato in destroy) --
     * valido solo per la durata della chiamata che lo imposta. q4e_ctx_len deve coprire
     * ALMENO cache_offset+n_tokens token, altrimenti il layer PLE si rifiuta (side-channel
     * non impostato per questo percorso == mai fidarsi di dati stantii). */
    const int32_t* q4e_ctx_ids;
    int q4e_ctx_len;
    /* Cache ausiliaria per-layer della sessione corrente (side-channel come q4e_ctx_ids):
     * settata dai loop di forward PRIMA di forward_one_token; i blocchi attention che ne hanno
     * bisogno la leggono (KDA/DSA). NULL = non impostata / non usata. */
    mlx_array* aux_cache;
    /* Prefix cache (paragrafo 8.4/8.10 del piano): stato di prefill salvato di un prefisso
     * (es. system prompt stabile), riusabile tra sessioni. Le cache sono alias MLX (refcount):
     * la sessione che le carica non le possiede, il prefisso le libera. */
    SavedPrefix* prefixes;
    int n_prefixes;
    int prefix_cap;
    llm_radix_trie radix;        /* trie content-addressed dei prefissi (prefix_id = indice in prefixes) */
    KvPagePool kvpool;           /* paged cache CONTENT-ADDRESSED dei blocchi KV del prefisso */
    KvDisk kvdisk;               /* §G: tier SSD dei blocchi KV del prefisso (registro `kvdisk`) */
    int kvdisk_state;            /* -1 = non inizializzato, 0 = OFF, 1 = ON */
    long kvdisk_spills;          /* statistiche: prefissi spillati su disco */
    long kvdisk_restores;        /* statistiche: prefissi ripristinati da disco */
    int  kvckpt_block;           /* §G-ter: blocco (token) dei checkpoint per il rewind (0 = OFF) */
    /* P4 MEMORY PLAN: budget deterministico + admission. */
    uint64_t p4_budget_bytes, p4_weights_bytes, p4_kv_per_token, p4_kv_per_token_q, p4_state_bytes;
    uint64_t gguf_mmap_bytes;   /* dimensione mmap dei pesi (pesi zero-copy, non contati da MLX) */
    /* P4: registro delle sessioni ATTIVE (conto MULTI-sessione del budget). */
    struct LlmMlxSession* p4_sessions[64];
    int p4_nsessions;
    int p4_reserve_mb, p4_plan_done, p4_enforce;
    long kvckpt_writes;          /* statistiche: checkpoint global-layer scritti su disco */
    long kvckpt_rewinds;         /* statistiche: rewind ripristinati da disco */
    long kvpark_writes;          /* §G-ter evict: sessioni evitte (KV su disco, RAM liberata) */
    long kvpark_reads;           /* §G-ter evict: sessioni ripristinate on-demand da disco */
    /* §G-bis leva 1 — sparse page-level attention (default OFF, env LLM_PAGESEL=1). */
    bool pagesel_on;
    int ps_page;                 /* token per pagina (LLM_PAGESEL_PAGE, default 64) */
    int ps_topk;                 /* pagine top-k da attendere (LLM_PAGESEL_TOPK, default 8) */
    int ps_window;               /* pagine recenti sempre incluse (LLM_PAGESEL_WINDOW, default 2) */
    int ps_min;                  /* token minimi per attivare (LLM_PAGESEL_MIN, default 1024) */
    int ps_cap;                  /* capacita' pagine per layer (LLM_PAGESEL_CAP, default 512) */
    int ps_verify;               /* LLM_PAGESEL_VERIFY=1: confronta con l'attenzione piena */
    /* LoRA (fase 4): moduli per (layer, proj) con mlx_array A/B. Array fisso. */
    int lora_n;
    int lora_cap;          /* R4.1: capacita' DINAMICA (cresce a richiesta; >16 moduli) */
    mlx_array* lora_A;
    mlx_array* lora_B;
    int* lora_layer;
    char (*lora_proj)[32];
    float* lora_scale;
    int* lora_expert;      /* MoE: indice esperto (default -1 = proiezione normale) */
    bool lora_enabled;
    uint64_t lru_clock;   /* contatore monotono per eviction LRU dei prefissi */
    int prefix_max;       /* limite prefissi salvati (eviction LRU oltre; default 16) */
    ImatrixState im;      /* imatrix collector (attivo solo via API dedicata) */
    int mem_active_limit_mb; /* guardia saturazione: tetto su mlx active (MB; 0 = off = nessun
                              * check, costo zero in normale operazione) */
    /* Motore ESTERNO (DS4/DS4.1): se non-NULL il modello e' gestito da un engine AUTONOMO
     * (`llmengine_llm_engine_ds4/ds41`) tramite l'ABI a 9 metodi. Il forward MLX
     * generico NON e' usato per questo modello. `dsv4`/`dsv41` sono spariti: la logica vive
     * negli engine (ds4/, ds41-native/). */
    const struct LlmEngineLlmEngine* deleg;
    void* deleg_model;
};

/* Stato ctx della DRAFT, PER-SESSIONE: kcache/vcache per layer + righe accumulate. Un runtime
 * DFlash2 condiviso puo' servire piu' sessioni (uno slot ciascuna): il round-loop multi-sessione
 * seleziona lo slot. Il single-session usa sempre lo slot 0 -> comportamento invariato. */

typedef struct DFlash2Runtime {
    LlmMlxModel* target;             /* modello target gia' caricato (embed/lm_head) */
    LlmMlxModel* draft;              /* modello draft caricato da create (pesi bf16) */
    int n_layers;
    Df2CtxState* ctx;         /* slot per-sessione; [0] = la sessione single (default) */
    int n_ctx;                /* slot allocati */
    int cur_ctx;              /* slot corrente (single-session: 0) */
    /* A1 — contesto del draft PERSISTENTE tra chiamate di generate (streaming). retained_ctx =
     * l'ultimo `vctx` (feature target delle righe accettate, non ancora appese al ring);
     * last_session/last_offset identificano la CONTINUAZIONE (stessa sessione, cache_offset
     * invariato). Serve a NON ri-catturare il prompt ne' resettare il ring a ogni chiamata. */
    void* last_session;
    int last_offset;
    float* retained_ctx;
    int retained_n;
    double acc_ema;   /* media mobile dei token accettati per round (K adattivo); 0 = non inizializzata */
    double round_ms_ema;   /* durata media (ms) di un round propose+verify, dopo i primi 2 */
    int rounds_seen;      /* round eseguiti dall'ultimo reset (guardia di profittabilita') */
    /* pesi target referenziati (embed + lm_head), fetch in create */
    mlx_array embed_w, embed_s, embed_b;
    mlx_array lm_w, lm_s, lm_b;
    bool have_lm_head;
    /* Modo bf16 PER-MODELO (g_bf16_all/g_act16 sono process-globali e il load del draft li
     * clobbera con il modo del draft): salviamo i due stati in create e li scambiamo attorno
     * a propose/debug (draft) e al resto (target). Senza, il target dopo create gira col modo
     * del draft — e con il GDN in dtype di attivazione il kernel GDN non compila nemmeno
     * (q/k f32 vs v bf16: trovato col bench dflash2 il 2026-09-19). */
    bool tgt_bf16_all; mlx_dtype tgt_act16; bool tgt_bf16_input;
    bool drf_bf16_all; mlx_dtype drf_act16; bool drf_bf16_input;
    /* P1 (2026-09-19): cache HOST del predecessor_codebook [vocab,rank] per il greedy-walk su
     * CPU in df2_selector -> elimina le K sincronizzazioni GPU per round (una sola eval/round). */
    float* sel_cpw_host; int sel_cpw_rows; int sel_cpw_rank;
    /* P1: pesi del selector caricati UNA volta (prima multishard_get ad ogni round -> l'eval
     * materializzava ogni volta le copie, in particolare predecessor_codebook [vocab,rank]). */
    mlx_array sel_hp_w, sel_cs_w, sel_cp_w; bool sel_w_loaded;
} DFlash2Runtime;


#include "llm_mlx_protos.h"
#endif /* LLM_MLX_INTERNAL_H */
