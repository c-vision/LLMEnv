/* llm_dsv41_tensors.h — DeepSeek V4.1: config, mappa nomi tensori e struttura pesi MLX,
 * header-only. Nessun peso reale qui: il caricamento dal pack tiny e' in
 * `llm_dsv41_tensors_ref.h`, il loader safetensors usera' `llm_dsv41_expected_names`.
 *
 * NOMI CANONICI (autorevoli): il release HF `deepseek-ai/DeepSeek-V4.1-Flash` usa i nomi
 * dell'inference ufficiale SENZA prefisso `model.` (v. `inference/model.py`, classe
 * `Transformer`; confermato da Rapid-MLX `deepseek_v41_native/convert.py`: "The hub release
 * already uses inference-style names (layers.N.attn.wq_a.weight, ffn, .scale)"):
 *   embed.weight · norm.weight · head.weight
 *   layers.N.attn.{wq_a,q_norm,wq_b,wkv,kv_norm,wo_a,wo_b}.weight · layers.N.attn.attn_sink
 *   layers.N.attn.compressor.{wkv,norm,wgate}.weight   (solo layer kv_source)
 *   layers.N.attn.indexer.{wq_b,weights_proj,wk,wk_norm}.weight (solo layer index_source)
 *   layers.N.{attn_norm,ffn_norm}.weight
 *   layers.N.{hc_attn_fn,hc_ffn_fn,hc_attn_base,hc_ffn_base,hc_attn_scale,hc_ffn_scale}
 *   layers.N.ffn.gate.{weight,bias} · layers.N.ffn.shared_experts.{w1,w2,w3}.weight
 *   layers.N.ffn.experts.E.{w1,w2,w3}.weight
 *   layers.N.engram.{embed.weight,embed.scale,wkv.weight,q_weight,k_weight}
 *
 * `llm_dsv41_expected_names(cfg, sink, user)` enumera TUTTI i nomi attesi per una config: e'
 * la checklist del loader e va mantenuta allineata a `inference/model.py`.
 */
#ifndef LLM_DSV41_TENSORS_H
#define LLM_DSV41_TENSORS_H

#include <mlx/c/mlx.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define LLM_DSV41_MAX_LAYERS 64

/* score_func (config.py) */
#define LLM_DSV41_SCORE_SOFTMAX 0
#define LLM_DSV41_SCORE_SIGMOID 1
#define LLM_DSV41_SCORE_SQRTSOFTPLUS 2

typedef struct {
    /* shape */
    int dim, n_layers, n_heads, head_dim, rope_head_dim;
    int o_groups, o_lora_rank, q_lora_rank;
    int moe_inter_dim, n_routed_experts, n_activated_experts, n_shared_experts;
    int window_size, vocab_size, hc_mult, hc_sinkhorn_iters;
    int index_n_heads, index_head_dim, index_topk;
    int candidate_source_layer, candidate_block_size, candidate_topk_blocks;
    /* scala-indipendenti */
    float norm_eps, hc_eps, rope_theta, compress_rope_theta, rope_factor;
    float beta_fast, beta_slow, route_scale, gate_temp, swiglu_limit;
    int original_seq_len;
    int score_func;        /* LLM_DSV41_SCORE_* */
    int norm_topk_prob;
    /* layout per-layer (compress_ratios[i]>0 => CSA2; in_kv_source/in_index_source) */
    int compress_ratios[LLM_DSV41_MAX_LAYERS];
    int in_kv_source[LLM_DSV41_MAX_LAYERS];
    int in_index_source[LLM_DSV41_MAX_LAYERS];
    int engram_layer_ids[LLM_DSV41_MAX_LAYERS];  /* nessuno se n_engram_layers==0 */
    int n_engram_layers;
    int engram_num_embeddings[LLM_DSV41_MAX_LAYERS];
    int engram_max_ngram_size, engram_n_heads, engram_head_dim;
    int engram_vocab_size, engram_compressed_vocab_size, engram_pad_id;
    /* extra (config reali; non usati dal forward tiny) */
    int max_seq_len, n_mtp_layers;
    int image_token_id, vision_n_layers;
    int ced_prefill, preserve_mtp;
    int dspark_block_size, dspark_n_routed_experts, dspark_n_activated_experts;
    int dspark_noise_token_id, dspark_markov_rank;
    int dspark_target_layer_ids[LLM_DSV41_MAX_LAYERS], n_dspark_targets;
} LlmDsv41Config;

/* Engram di un layer (tabelle + parametrici di hash). `token_map` mappa id raw -> id compresso;
 * `primes`/`mult`/`offsets` sono i parametrici del NgramHash (per-layer). */
typedef struct {
    bool present;
    mlx_array embed;   /* [rows, hd] */
    mlx_array wkv;     /* [dim*(hc+1), ncols*hd] */
    mlx_array qw, kw;  /* [hc, dim] */
    int64_t* token_map; int token_map_n;
    int64_t* primes;   /* [depth*nheads] */
    int64_t* mult;     /* [depth+1] */
    int64_t* offsets;  /* [depth*nheads] */
    int depth, nheads, hd, rows, ncols;
    int64_t pad_id;
} LlmDsv41Engram;

/* Pesi di UN layer. Array MLX (float32 per il tiny; il loader reale potra' usare bf16/fp8).
 * wq_a [q_lora_rank,dim] · q_norm [q_lora_rank] · wq_b [n_heads*head_dim,q_lora_rank]
 * wkv [head_dim,dim] · kv_norm [head_dim] · wo_a [o_groups*o_lora_rank, n_heads*head_dim/o_groups]
 * wo_b [dim, o_groups*o_lora_rank] · attn_sink [n_heads]
 * experts w1/w3 [n_routed_experts,moe_inter_dim,dim] · w2 [n_routed_experts,dim,moe_inter_dim]
 * shared w1/w3 [moe_inter_dim,dim] · w2 [dim,moe_inter_dim] */
typedef struct {
    mlx_array hc_attn_fn, hc_attn_scale, hc_attn_base;
    mlx_array hc_ffn_fn,  hc_ffn_scale,  hc_ffn_base;
    mlx_array attn_norm, ffn_norm;
    mlx_array wq_a, q_norm, wq_b, wkv, kv_norm, wo_a, wo_b, attn_sink;
    mlx_array comp_wkv, comp_norm, comp_wgate;   /* Compressor, solo layer kv_source con ratio>1 */
    mlx_array idx_wq_b, idx_weights_proj, idx_wk, idx_k_norm; /* Indexer (layer index_source) */
    mlx_array gate_w, gate_b;
    mlx_array experts_w1, experts_w3, experts_w2;
    mlx_array shared_w1, shared_w3, shared_w2;
    LlmDsv41Engram engram;
} LlmDsv41LayerWeights;

typedef struct {
    LlmDsv41Config cfg;
    mlx_array embed, norm, head;
    LlmDsv41LayerWeights* layers;   /* [cfg.n_layers] */
    int n_layers;
    void* stream;                   /* opaco: LlmDsv41Stream* se lo weights e' in STREAMING */
} LlmDsv41Weights;

/* ── helper: array float32 da un buffer host (MLX copia i dati) ───────────────── */
static inline mlx_array llm_dsv41_array_f32(const float* data, int nd, const int* shape) {
    return mlx_array_new_data(data, shape, nd, MLX_FLOAT32);
}

/* Config del riferimento tiny (tools/gen_dsv41_fwd_ref.py / gen_dsv41_forward_reference.py):
 * dim=8, 2 layer, hc=4, n_heads=2, head_dim=32, rope_head_dim=4, o_groups=2, o_lora=8,
 * q_lora=8, moe_inter=8, 4 routed / 2 act / 1 shared, window=4, vocab=64, compress_ratios 0. */
static inline LlmDsv41Config llm_dsv41_config_tiny_ref(void) {
    LlmDsv41Config c;
    memset(&c, 0, sizeof c);
    c.dim = 8; c.n_layers = 2; c.n_heads = 2; c.head_dim = 32; c.rope_head_dim = 4;
    c.o_groups = 2; c.o_lora_rank = 8; c.q_lora_rank = 8;
    c.moe_inter_dim = 8; c.n_routed_experts = 4; c.n_activated_experts = 2; c.n_shared_experts = 1;
    c.window_size = 4; c.vocab_size = 64; c.hc_mult = 4; c.hc_sinkhorn_iters = 20;
    c.index_n_heads = 0; c.index_head_dim = 32; c.index_topk = 0;
    c.candidate_source_layer = -1;
    c.norm_eps = 1e-20f; c.hc_eps = 1e-6f;
    c.rope_theta = 10000.0f; c.compress_rope_theta = 40000.0f; c.rope_factor = 40.0f;
    c.beta_fast = 32.0f; c.beta_slow = 1.0f; c.original_seq_len = 0;
    c.route_scale = 1.0f; c.gate_temp = 1.0f; c.swiglu_limit = 0.0f;
    c.score_func = LLM_DSV41_SCORE_SQRTSOFTPLUS; c.norm_topk_prob = 1;
    for (int i = 0; i < c.n_layers; i++) c.compress_ratios[i] = 0;
    c.n_engram_layers = 0;
    return c;
}

/* Config del riferimento CSA2 tiny (tools/gen_dsv41_fwd_csa_ref.py): come il tiny ma con
 * layer1 ratio=2, kv_source+index_source, index_topk=64 (>= chiavi compresse). */
static inline LlmDsv41Config llm_dsv41_config_csa_ref(void) {
    LlmDsv41Config c = llm_dsv41_config_tiny_ref();
    c.compress_ratios[1] = 2;
    c.in_kv_source[1] = 1;
    c.in_index_source[1] = 1;
    c.index_n_heads = 2; c.index_head_dim = 32; c.index_topk = 64;
    c.candidate_source_layer = -1;
    return c;
}

/* Config del riferimento Engram tiny (tools/gen_dsv41_engram_fwd_ref.py). */
static inline LlmDsv41Config llm_dsv41_config_engram_ref(void) {
    LlmDsv41Config c = llm_dsv41_config_tiny_ref();
    c.engram_layer_ids[0] = 1; c.n_engram_layers = 1;
    c.engram_max_ngram_size = 2; c.engram_n_heads = 2; c.engram_head_dim = 8;
    c.engram_vocab_size = 20; c.engram_compressed_vocab_size = 64; c.engram_pad_id = 2;
    c.engram_num_embeddings[0] = 52;
    return c;
}

/* Config del riferimento Candidate-blocks tiny (tools/gen_dsv41_cand_ref.py). */
static inline LlmDsv41Config llm_dsv41_config_cand_ref(void) {
    LlmDsv41Config c = llm_dsv41_config_csa_ref();
    c.n_layers = 3;
    c.compress_ratios[1] = 1; c.compress_ratios[2] = 1;
    c.in_kv_source[1] = 1; c.in_kv_source[2] = 1;
    c.in_index_source[1] = 1; c.in_index_source[2] = 1;
    c.candidate_source_layer = 1;
    c.candidate_topk_blocks = 2; c.candidate_block_size = 2;
    c.index_topk = 2;
    return c;
}

/* Config del riferimento condivisione cross-layer tiny (tools/gen_dsv41_share_ref.py):
 * layer1 kv_source+index_source, layer2 index_source-only (riusa il KV compresso di layer1). */
static inline LlmDsv41Config llm_dsv41_config_share_ref(void) {
    LlmDsv41Config c = llm_dsv41_config_csa_ref();
    c.n_layers = 3;
    c.compress_ratios[2] = 2;
    c.in_kv_source[2] = 0;
    c.in_index_source[2] = 1;
    c.index_topk = 2;
    return c;
}

/* Config del riferimento pure-reuse tiny (tools/gen_dsv41_purereuse_ref.py): layer1
 * kv+index source; layer2 ratio>0 ma NESSUN source -> riusa KV compresso e top-k di layer1. */
static inline LlmDsv41Config llm_dsv41_config_purereuse_ref(void) {
    LlmDsv41Config c = llm_dsv41_config_share_ref();
    c.in_index_source[2] = 0;
    return c;
}


/* Config del riferimento CED prefill-skip tiny (tools/gen_dsv41_ced_ref.py). */
static inline LlmDsv41Config llm_dsv41_config_ced_ref(void) {
    LlmDsv41Config c = llm_dsv41_config_csa_ref();
    c.n_layers = 4;
    c.compress_ratios[1] = 2; c.compress_ratios[2] = 1; c.compress_ratios[3] = 1;
    c.in_kv_source[1] = 1; c.in_kv_source[2] = 1; c.in_kv_source[3] = 0;
    c.in_index_source[1] = 1; c.in_index_source[2] = 1; c.in_index_source[3] = 1;
    c.index_topk = 2;
    c.ced_prefill = 1;
    return c;
}

/* Config REALE del release `deepseek-ai/DeepSeek-V4.1-Flash` (root/inference config.json).
 * Usata per la copertura dei nomi (`llm_dsv41_expected_names`). I 3 layer MTP (mtp.0/1/2) e la
 * vision NON sono nello spazio `layers.` del backbone. */
static inline LlmDsv41Config llm_dsv41_config_real(void) {
    LlmDsv41Config c;
    memset(&c, 0, sizeof c);
    c.dim = 5120; c.n_layers = 40; c.n_heads = 64; c.head_dim = 512; c.rope_head_dim = 64;
    c.o_groups = 8; c.o_lora_rank = 1024; c.q_lora_rank = 1280;
    c.moe_inter_dim = 2304; c.n_routed_experts = 384; c.n_activated_experts = 6; c.n_shared_experts = 1;
    c.window_size = 128; c.vocab_size = 129280; c.hc_mult = 4; c.hc_sinkhorn_iters = 20;
    c.index_n_heads = 32; c.index_head_dim = 128; c.index_topk = 512;
    c.candidate_source_layer = 20; c.candidate_topk_blocks = 0; c.candidate_block_size = 0;
    c.norm_eps = 1e-20f; c.hc_eps = 1e-6f; c.rope_theta = 10000.0f;
    c.compress_rope_theta = 160000.0f; c.rope_factor = 16.0f;
    c.beta_fast = 32.0f; c.beta_slow = 1.0f; c.original_seq_len = 65536;
    c.route_scale = 1.5f; c.gate_temp = 1.0f; c.swiglu_limit = 10.0f;
    c.score_func = LLM_DSV41_SCORE_SQRTSOFTPLUS; c.norm_topk_prob = 1;
    for (int i = 0; i < 20; i++) c.compress_ratios[i] = (i < 2) ? 0 : 2;
    for (int i = 20; i < c.n_layers; i++) c.compress_ratios[i] = 1;
    int kv[] = { 2, 8, 14, 20 };
    for (size_t i = 0; i < sizeof kv / sizeof kv[0]; i++) c.in_kv_source[kv[i]] = 1;
    int ix[] = { 2, 8, 14, 20, 24, 28, 32, 36 };
    for (size_t i = 0; i < sizeof ix / sizeof ix[0]; i++) c.in_index_source[ix[i]] = 1;
    c.engram_layer_ids[0] = 1; c.engram_layer_ids[1] = 14; c.n_engram_layers = 2;
    c.engram_max_ngram_size = 4; c.engram_n_heads = 8; c.engram_head_dim = 256;
    c.engram_vocab_size = 16000000; c.engram_compressed_vocab_size = 99092;
    c.engram_pad_id = 2;
    c.engram_num_embeddings[0] = 384006168; c.engram_num_embeddings[1] = 384016682;
    /* DSpark/MTP (config reale: num_nextn_predict_layers=3, target [37,38,39]). */
    c.n_mtp_layers = 3; c.dspark_block_size = 5; c.dspark_noise_token_id = 128799;
    c.dspark_markov_rank = 256; c.dspark_n_routed_experts = 128; c.dspark_n_activated_experts = 3;
    c.dspark_target_layer_ids[0] = 37; c.dspark_target_layer_ids[1] = 38; c.dspark_target_layer_ids[2] = 39;
    c.n_dspark_targets = 3;
    c.vision_n_layers = 32;
    return c;
}

/* ── mappa nomi canonici ──────────────────────────────────────────────────────── */

/* costruttori di stringhe (root senza prefisso `model.`) */
static inline void llm_dsv41_name_layer(char* out, size_t n, int layer, const char* suffix) {
    snprintf(out, n, "layers.%d.%s", layer, suffix);
}
static inline void llm_dsv41_name_expert(char* out, size_t n, int layer, int expert, const char* w) {
    snprintf(out, n, "layers.%d.ffn.experts.%d.%s.weight", layer, expert, w);
}

/* callback di emissione: ritorna 0 per continuare, non-zero per interrompere. */
typedef int (*LlmDsv41NameSink)(void* user, const char* name);

/* Suffissi comuni a OGNI layer backbone (attenzione + mHC + FFN shared/gate/norm). */
static const char* const llm_dsv41_layer_suffixes[] = {
    "attn.wq_a.weight", "attn.q_norm.weight", "attn.wq_b.weight",
    "attn.wkv.weight", "attn.kv_norm.weight", "attn.wo_a.weight", "attn.wo_b.weight",
    "attn.attn_sink",
    "attn_norm.weight", "ffn_norm.weight",
    "hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale",
    "ffn.gate.weight", "ffn.gate.bias", "ffn.gate.bias_vl",
    "ffn.shared_experts.w1.weight", "ffn.shared_experts.w2.weight", "ffn.shared_experts.w3.weight",
};
#define LLM_DSV41_LAYER_SUFFIX_COUNT \
    (sizeof(llm_dsv41_layer_suffixes) / sizeof(llm_dsv41_layer_suffixes[0]))

/* Enumerazione COMPLETA dei nomi attesi per `cfg`. Ritorna 0 se completata, non-zero se il
 * sink ha interrotto. Include esperti routed, compressor/indexer per i layer sorgente ed
 * engram per i layer elencati. */
static inline int llm_dsv41_expected_names(const LlmDsv41Config* cfg,
                                           LlmDsv41NameSink sink, void* user) {
    char buf[160];
    const char* top[] = { "embed.weight", "norm.weight", "head.weight" };
    for (size_t i = 0; i < 3; i++) if (sink(user, top[i])) return 1;
    for (int l = 0; l < cfg->n_layers; l++) {
        for (size_t i = 0; i < LLM_DSV41_LAYER_SUFFIX_COUNT; i++) {
            llm_dsv41_name_layer(buf, sizeof buf, l, llm_dsv41_layer_suffixes[i]);
            if (sink(user, buf)) return 1;
        }
        for (int e = 0; e < cfg->n_routed_experts; e++) {
            const char* ws[] = { "w1", "w2", "w3" };
            for (size_t i = 0; i < 3; i++) {
                llm_dsv41_name_expert(buf, sizeof buf, l, e, ws[i]);
                if (sink(user, buf)) return 1;
            }
        }
        if (cfg->in_kv_source[l]) {
            const char* cs[] = { "attn.compressor.wkv.weight", "attn.compressor.norm.weight" };
            for (size_t i = 0; i < 2; i++) {
                llm_dsv41_name_layer(buf, sizeof buf, l, cs[i]);
                if (sink(user, buf)) return 1;
            }
            /* wgate esiste solo per ratio>1 (Compressor lo crea se ratio>1). */
            if (cfg->compress_ratios[l] > 1) {
                llm_dsv41_name_layer(buf, sizeof buf, l, "attn.compressor.wgate.weight");
                if (sink(user, buf)) return 1;
            }
        }
        if (cfg->in_index_source[l]) {
            const char* is[] = { "attn.indexer.wq_b.weight", "attn.indexer.weights_proj.weight" };
            for (size_t i = 0; i < 2; i++) {
                llm_dsv41_name_layer(buf, sizeof buf, l, is[i]);
                if (sink(user, buf)) return 1;
            }
            if (cfg->in_kv_source[l]) {
                const char* ik[] = { "attn.indexer.wk.weight", "attn.indexer.k_norm.weight" };
                for (size_t i = 0; i < 2; i++) {
                    llm_dsv41_name_layer(buf, sizeof buf, l, ik[i]);
                    if (sink(user, buf)) return 1;
                }
            }
        }
        for (int e = 0; e < cfg->n_engram_layers; e++) {
            if (cfg->engram_layer_ids[e] != l) continue;
            const char* es[] = { "engram.embed.weight", "engram.embed.scale", "engram.wkv.weight",
                                 "engram.q_weight", "engram.k_weight" };
            for (size_t i = 0; i < 5; i++) {
                llm_dsv41_name_layer(buf, sizeof buf, l, es[i]);
                if (sink(user, buf)) return 1;
            }
        }
    }
    return 0;
}

/* Nomi attesi dei pesi DSpark/MTP (`mtp.N.*`): stesso blocco di un layer + main_proj/main_norm
 * (stage 0) + norm/markov/confidence (stage finale) + esperti draft. */
static inline int llm_dsv41_expected_dspark_names(const LlmDsv41Config* cfg,
                                                  LlmDsv41NameSink sink, void* user) {
    char buf[160];
    for (int st = 0; st < cfg->n_mtp_layers; st++) {
        for (size_t i = 0; i < LLM_DSV41_LAYER_SUFFIX_COUNT; i++) {
            snprintf(buf, sizeof buf, "mtp.%d.%s", st, llm_dsv41_layer_suffixes[i]);
            if (sink(user, buf)) return 1;
        }
        for (int e = 0; e < cfg->dspark_n_routed_experts; e++) {
            const char* ws[] = { "w1", "w2", "w3" };
            for (size_t i = 0; i < 3; i++) {
                snprintf(buf, sizeof buf, "mtp.%d.ffn.experts.%d.%s.weight", st, e, ws[i]);
                if (sink(user, buf)) return 1;
            }
        }
        if (st == 0) {
            snprintf(buf, sizeof buf, "mtp.%d.main_proj.weight", st); if (sink(user, buf)) return 1;
            snprintf(buf, sizeof buf, "mtp.%d.main_norm.weight", st); if (sink(user, buf)) return 1;
        }
        if (st == cfg->n_mtp_layers - 1) {
            const char* fs[] = { "norm.weight", "markov_head.embed.weight",
                                 "markov_head.head.weight", "confidence_head.proj.weight" };
            for (size_t i = 0; i < 4; i++) {
                snprintf(buf, sizeof buf, "mtp.%d.%s", st, fs[i]);
                if (sink(user, buf)) return 1;
            }
        }
    }
    return 0;
}

/* Nomi attesi del VLM (solo se vision abilitata): ViT (32 blocchi) + aligner + token immagine. */
static inline int llm_dsv41_expected_vision_names(const LlmDsv41Config* cfg,
                                                  LlmDsv41NameSink sink, void* user) {
    if (cfg->vision_n_layers <= 0) return 0;
    char buf[160];
    const char* top[] = { "image_start", "image_end", "image_newline",
                          "vision.patch_embed.proj.weight", "vision.patch_embed.proj.bias",
                          "vision.norm.weight", "aligner.w1.weight", "aligner.w1.bias",
                          "aligner.w2.weight", "aligner.w2.bias" };
    for (size_t i = 0; i < sizeof top / sizeof top[0]; i++) if (sink(user, top[i])) return 1;
    for (int b = 0; b < cfg->vision_n_layers; b++) {
        const char* bs[] = { "attn.wqkv.weight", "attn.wqkv.bias", "attn.wo.weight", "attn.wo.bias",
                             "mlp.w1.weight", "mlp.w2.weight", "norm1.weight", "norm2.weight" };
        for (size_t i = 0; i < sizeof bs / sizeof bs[0]; i++) {
            snprintf(buf, sizeof buf, "vision.blocks.%d.%s", b, bs[i]);
            if (sink(user, buf)) return 1;
        }
    }
    return 0;
}

/* ── alloc/libera struttura pesi ──────────────────────────────────────────────── */
static inline bool llm_dsv41_weights_alloc(LlmDsv41Weights* w, const LlmDsv41Config* cfg) {
    memset(w, 0, sizeof *w);
    w->cfg = *cfg;
    w->n_layers = cfg->n_layers;
    w->embed = mlx_array_new(); w->norm = mlx_array_new(); w->head = mlx_array_new();
    w->layers = (LlmDsv41LayerWeights*)calloc((size_t)cfg->n_layers, sizeof(LlmDsv41LayerWeights));
    if (!w->layers) return false;
    for (int l = 0; l < cfg->n_layers; l++) {
        LlmDsv41LayerWeights* L = &w->layers[l];
        L->hc_attn_fn = mlx_array_new(); L->hc_attn_scale = mlx_array_new(); L->hc_attn_base = mlx_array_new();
        L->hc_ffn_fn  = mlx_array_new(); L->hc_ffn_scale  = mlx_array_new(); L->hc_ffn_base  = mlx_array_new();
        L->attn_norm = mlx_array_new(); L->ffn_norm = mlx_array_new();
        L->wq_a = mlx_array_new(); L->q_norm = mlx_array_new(); L->wq_b = mlx_array_new();
        L->wkv = mlx_array_new(); L->kv_norm = mlx_array_new(); L->wo_a = mlx_array_new();
        L->wo_b = mlx_array_new(); L->attn_sink = mlx_array_new();
        L->comp_wkv = mlx_array_new(); L->comp_norm = mlx_array_new(); L->comp_wgate = mlx_array_new();
        L->idx_wq_b = mlx_array_new(); L->idx_weights_proj = mlx_array_new();
        L->idx_wk = mlx_array_new(); L->idx_k_norm = mlx_array_new();
        L->gate_w = mlx_array_new(); L->gate_b = mlx_array_new();
        L->experts_w1 = mlx_array_new(); L->experts_w3 = mlx_array_new(); L->experts_w2 = mlx_array_new();
        L->shared_w1 = mlx_array_new(); L->shared_w3 = mlx_array_new(); L->shared_w2 = mlx_array_new();
        L->engram.present = false;
        L->engram.embed = mlx_array_new(); L->engram.wkv = mlx_array_new();
        L->engram.qw = mlx_array_new(); L->engram.kw = mlx_array_new();
        L->engram.token_map = NULL; L->engram.primes = NULL;
        L->engram.mult = NULL; L->engram.offsets = NULL;
    }
    return true;
}

/* Libera i campi di UN layer + azzera la struct (usato dallo stream per lo slot riusabile). */
static inline void llm_dsv41_layer_free(LlmDsv41LayerWeights* L) {
    if (!L) return;
    mlx_array_free(L->hc_attn_fn); mlx_array_free(L->hc_attn_scale); mlx_array_free(L->hc_attn_base);
    mlx_array_free(L->hc_ffn_fn);  mlx_array_free(L->hc_ffn_scale);  mlx_array_free(L->hc_ffn_base);
    mlx_array_free(L->attn_norm); mlx_array_free(L->ffn_norm);
    mlx_array_free(L->wq_a); mlx_array_free(L->q_norm); mlx_array_free(L->wq_b);
    mlx_array_free(L->wkv); mlx_array_free(L->kv_norm); mlx_array_free(L->wo_a);
    mlx_array_free(L->wo_b); mlx_array_free(L->attn_sink);
    mlx_array_free(L->comp_wkv); mlx_array_free(L->comp_norm); mlx_array_free(L->comp_wgate);
    mlx_array_free(L->idx_wq_b); mlx_array_free(L->idx_weights_proj);
    mlx_array_free(L->idx_wk); mlx_array_free(L->idx_k_norm);
    mlx_array_free(L->gate_w); mlx_array_free(L->gate_b);
    mlx_array_free(L->experts_w1); mlx_array_free(L->experts_w3); mlx_array_free(L->experts_w2);
    mlx_array_free(L->shared_w1); mlx_array_free(L->shared_w3); mlx_array_free(L->shared_w2);
    mlx_array_free(L->engram.embed); mlx_array_free(L->engram.wkv);
    mlx_array_free(L->engram.qw); mlx_array_free(L->engram.kw);
    free(L->engram.token_map); free(L->engram.primes);
    free(L->engram.mult); free(L->engram.offsets);
    memset(L, 0, sizeof *L);
}

static inline void llm_dsv41_weights_free(LlmDsv41Weights* w) {
    if (!w || !w->layers) return;
    mlx_array_free(w->embed); mlx_array_free(w->norm); mlx_array_free(w->head);
    for (int l = 0; l < w->n_layers; l++) llm_dsv41_layer_free(&w->layers[l]);
    free(w->layers);
    memset(w, 0, sizeof *w);
}

#endif /* LLM_DSV41_TENSORS_H */
