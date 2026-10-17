/* llm_mlx_protos.h — PROTOTIPI e dichiarazioni dei DOMINI, estratti da llm_mlx_internal.h
 * (refactor Fase A: internal.h = tipi+cfg+session+model; questo = decl dei domini).
 * Incluso SOLO da llm_mlx_internal.h (in coda, dopo i tipi): non usarlo standalone. */
#ifndef LLM_MLX_PROTOS_H
#define LLM_MLX_PROTOS_H

/* --- dominio P4 (llm_mlx_p4.c) --- */
uint64_t llm_p4_sysctl_u64(const char* name);
uint64_t llm_p4_kv_per_token(const ModelConfig* c);
uint64_t llm_p4_kv_per_token_int8(const ModelConfig* c);
uint64_t llm_p4_kv_state(const ModelConfig* c);
void     llm_p4_plan_init(LlmMlxModel* m);
uint64_t llm_p4_kv_needed(LlmMlxModel* m, int offset, int n);
uint64_t llm_p4_used_sessions(LlmMlxModel* m, const struct LlmMlxSession* exclude);

/* --- dominio DFlash2 (llm_mlx_dflash2.c) --- */
void df2_mode_enter_draft(DFlash2Runtime* rt, bool* sv_all, mlx_dtype* sv_act);
void df2_mode_leave_draft(DFlash2Runtime* rt, bool sv_all, mlx_dtype sv_act);
bool dflash_fetch_target_tensors(DFlash2Runtime* rt);
void df2_q8_reset(void);   /* libera la cache dei pesi draft quantizzati (opzione df2_q8) */
bool llm_mlx_df2_unprofitable(void* h, LlmMlxModel* m);   /* true se i round costano piu' dell'AR */
bool df2_backbone(DFlash2Runtime* rt, const int32_t* block_ids, int L, mlx_array x_ctx, int S, mlx_array* out_h, mlx_array* layer_out);
bool df2_selector(DFlash2Runtime* rt, mlx_array hidden_post, mlx_array logits, int K, int32_t anchor, int32_t* out_path);
double g_df2_now(void);
bool g_df2_time_on(void);
extern double g_df2_t_ckpt, g_df2_t_fwd, g_df2_t_arg, g_df2_t_roll;
extern double g_df2_t_prop, g_df2_t_sel, g_df2_t_bb, g_df2_t_wget, g_df2_t_argp;
extern double g_df2_ts_argp, g_df2_ts_unary, g_df2_ts_hproj, g_df2_ts_succ, g_df2_ts_ast;
extern int g_df2_n_rounds, g_df2_n_roll;

/* --- CORE shared (extern; usati dai domini .c) --- */
bool plain_linear(mlx_array* res, mlx_array x, mlx_array w, mlx_stream stream);
bool rms_norm_cfg(mlx_array* res, mlx_array x, mlx_array w, float eps, mlx_stream stream);
bool quantized_linear_cfg(mlx_array* res, mlx_array x, mlx_array w, mlx_array scales, mlx_array biases, int group_size, int bits, const char* mode, mlx_stream stream);
void llm_mlx_dbg_dump(const char* name, int layer, mlx_array a);   /* LLM_DBG_DUMP_DIR */
/* Codebook TurboQuant GLOBALE (identico in tutti i tensori tq, da config.quantization.codebook). */
void llm_mlx_tq_set_cb(const float* cb, int n);
int llm_mlx_tq_get_cb(float* out, int cap);
int llm_mlx_tq_enabled(void);   /* true se il modello e' TurboQuant (codebook dal config) */
int llm_mlx_derive_bits(mlx_array w, mlx_array scales, int group_size, int fallback);
/* --- dominio sensitivity/quantize (llm_mlx_sensitivity.c) --- */
/* Definito in llm_mlx_sensitivity.c, usato dal monolite (model_build). */
const char* llm_mlx_layer_proj(const ModelConfig* c, int layer);
void llm_mlx_mode_set(char* dst, size_t cap, const char* s);
bool llm_mlx_embed_row_from_idx(LlmMlxModel* m, mlx_array embed_w, mlx_array embed_s, mlx_array embed_b, mlx_array tid, mlx_array* out_x);
bool llm_kvdisk_enabled(LlmMlxModel* m);
const char* llm_prefix_cache_path(LlmMlxModel* m, char* buf, int cap);
int llm_session_ckpt_write_k(LlmMlxModel* m, LlmMlxSession* s, int pos, const int32_t* key_tok, int key_n);
int llm_session_unpark(LlmMlxModel* m, LlmMlxSession* s);
void session_history_append(LlmMlxSession* s, const int32_t* tok, int n);
llm_mlx_sampling_t llm_mlx_session_sampling(const LlmMlxSession* s);
int llm_mlx_model_lora_get(LlmMlxModel* m, int layer, const char* proj);
/* --- CORE: definiti nel monolite, usati dal dominio DFlash2 --- */
void llm_mlx_quant_derive(mlx_array w, mlx_array scales, int fb_group, int fb_bits, const char* fb_mode, int* out_group, int* out_bits, char* out_mode, size_t mode_cap);
bool multishard_get(const MultiShardWeights* mw, const char* name, mlx_array* out);
bool multishard_get_triple(const MultiShardWeights* mw, const char* base, mlx_array* w, mlx_array* scales, mlx_array* biases);
void tname(const char* layer_prefix, int layer_idx, const char* suffix, char* out, size_t out_cap);
extern mlx_array g_mlx_empty_arr;
extern bool g_bf16_all;
extern mlx_dtype g_act16;
static inline bool llm_mlx_is_act16(mlx_dtype dt) { return dt == g_act16; }
bool llm_mlx_logit_softcap(mlx_array* res, mlx_array logits, float softcap, mlx_stream stream);

/* --- dominio API esportata (llm_mlx_api.c) --- */
/* Definiti nel monolite (o in altri domini), usati da llm_mlx_api.c. */
bool llm_bonsai_rotate(mlx_array* x, LlmMlxModel* m, int layer_idx, const char* suffix, bool invert);
double llm_mlx_max_abs_diff(mlx_array a, mlx_array b, size_t n, mlx_stream s);
/* A/B bit-level GDN (LLM_GDN_AB): se non-NULL, i loop per-layer di df2_capture_batch e
 * forward_tokens_batch stampano il checksum di h con questo tag ("A"/"B") per diffarli. */
extern const char* g_layerh_run;
bool load_output_tensors(LlmMlxModel* m, OutputTensors* t);
void free_output_tensors(OutputTensors* t);
bool embed_tokens_batch(LlmMlxModel* m, mlx_array embed_w, mlx_array embed_s, mlx_array embed_b,
                        const int32_t* token_ids, int n_tokens, mlx_array* out_x);
bool layer_forward(LlmMlxModel* m, LlmMlxSession* kvq_sess, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                   mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out);
/* --- dominio layer_forward KV-quant/pagesel (llm_mlx_layer_forward.c) --- */
void llm_mlx_set_layer_ops(const char* const* ops, int n);
/* B7: accesso al registry dei pool int8 (definito in llm_mlx_layer_forward.c) dal lifecycle. */
void llm_mlx_kvq_pool_release_session(void* model, void* sess);
void llm_mlx_kvq_pool_reset_session(void* model, void* sess);
/* Definiti nel monolite, usati da layer_forward (llm_mlx_layer_forward.c). */
bool llm_env_prof(void);
bool llm_env_phase(void);
bool forward_tokens_batch_multi_cap(LlmMlxModel* m, LlmMlxSession** sess, const int* ks, int n_sess,
                                    const int32_t* toks, const int* offs,
                                    mlx_array embed_w, mlx_array embed_s, mlx_array embed_b,
                                    const int* caplayers, int ncap,
                                    float* capout, int capcap, mlx_array* out_h);
bool llm_mlx_final_norm(LlmMlxModel* m, mlx_array h, mlx_array norm_w, mlx_array* out_normed);
void llm_mlx_shards_to_gpu(MultiShardWeights* mw, mlx_stream gpu);
extern int g_oq_a8_minm_override;
extern int g_qmv_wide_cur;   /* ultimo valore passato a mlx_set_qmv_wide */
/* Definito in llm_mlx_api.c, usato dal monolite (qwen3_next_linear_block). */
bool q4_sigmoid_rmsnorm_gated(mlx_array* res, mlx_array hidden, mlx_array gate,
                              mlx_array w, float eps, bool use_silu, mlx_stream s);

/* --- dominio MTP (llm_mlx_mtp.c) --- */
LlmMlxMtp* llm_mlx_mtp_get(LlmMlxModel* m);

/* --- dominio SELFTEST (llm_mlx_selftest.c) --- */
/* Definiti nel monolite (top), usati da llm_mlx_selftest.c. */
bool quantized_linear(mlx_array* res, mlx_array x, mlx_array w, mlx_array scales, mlx_array biases, mlx_stream stream);
bool get_quantized_triple(mlx_map_string_to_array map, const char* base,
                          mlx_array* w, mlx_array* scales, mlx_array* biases);

/* --- dominio TRAIN (llm_mlx_train.c) --- */
/* Definito nel monolite, usato da llm_mlx_train.c. */
bool forward_tokens_batch(LlmMlxModel* m, LlmMlxSession* kvq_sess, mlx_array embed_w, mlx_array embed_s, mlx_array embed_b,
                          const int32_t* token_ids, int n_tokens, int cache_offset,
                          mlx_array* k_cache, mlx_array* v_cache, bool* has_cache, mlx_array* out_h);

/* --- dominio STATS (llm_mlx_stats.c) --- */
/* Definiti nel monolite, usati da llm_mlx_stats.c. */
int llm_mlx_imatrix_begin(LlmMlxModel* m);
int llm_mlx_imatrix_write(LlmMlxModel* m, const char* path);
/* Definito in llm_mlx_stats.c, usato dal monolite (session_append/generate). */
int llm_mlx_active_guard(LlmMlxModel* m);

/* --- TRAINING: modalità forward COMPOSABILE (def. in llm_mlx_train.c) ---
 * 1 = il forward non usa kernel Metal custom (che non hanno vjp): le funzioni *_enabled()
 * dei kernel fusi ritornano false e si usa il path MLX nativo differenziabile. */
extern int g_llm_mlx_train_composable;

/* --- dominio attivazioni compilate (llm_mlx_activation.c) --- */
bool llm_mlx_fuse_geglu_enabled(void);
bool llm_mlx_fuse_geglu_metal_enabled(void);
bool llm_mlx_geglu_compiled(mlx_array gate_out, mlx_array up_out, mlx_array* out_act, mlx_stream stream);
bool llm_mlx_fuse_swiglu_prefill_enabled(void);
bool llm_mlx_swiglu_compiled(mlx_array gate_out, mlx_array up_out, mlx_array* out_act, mlx_stream stream);

/* --- dominio GDN cache per-layer (llm_mlx_gdn_cache.c) --- */
/* llm_mlx_act_dt e' definita nel monolite (usata dal dominio). */
mlx_dtype llm_mlx_act_dt(bool on);
mlx_array llm_mlx_act_scalar(float v);
mlx_array* llm_mlx_gdn_conv_cached(LlmMlxModel* m, int layer_idx);
mlx_array* llm_mlx_gdn_dt_bias(LlmMlxModel* m, int layer_idx);
LlmQwen4ExpQSACache* llm_mlx_qwen4exp_model_qsa_cache(LlmMlxModel* m, int layer_idx);
mlx_array* llm_mlx_qwen4exp_ple_conv_cache(LlmMlxModel* m, int layer_idx, bool* out_has_state);
mlx_array* llm_mlx_gdn_alog_exp(LlmMlxModel* m, int layer_idx);

/* --- dominio BF16/dtype-policy (llm_mlx_bf16.c) --- */
/* llm_mlx_act16_name e' definita nel monolite (usata dal dominio). */
const char* llm_mlx_act16_name(void);
bool llm_mlx_bf16_mode_known(const char* mode);
bool llm_mlx_bf16_bits_on(const LlmMlxModel* m, const char* mode);
int  llm_mlx_apply_bf16_mode(LlmMlxModel* m, const char* mode, const char* origin);
void llm_mlx_auto_pair_act_dtype(LlmMlxModel* m, bool explicit_requested);
int  llm_mlx_promote_scales_f32(LlmMlxModel* m);

/* --- dominio GPT-OSS: contratto in oss/llm_oss.h (incluso in testa) --- */
/* llm_mlx_wcache_enabled e' definita nel monolite (usata dal dominio). */
bool llm_mlx_wcache_enabled(void);
/* cfg-branch GPT-OSS (implementato in oss/llm_oss_config.c). ModelConfig è anonimo -> dichiarato qui. */
struct cJSON;
bool llm_oss_parse_config(ModelConfig* cfg, struct cJSON* src);

/* --- dominio ROPE (llm_mlx_rope.c) --- */
mlx_array llm_mlx_proportional_rope_freqs(int dims, int rotated_dims, float base, float factor);
mlx_array llm_mlx_llama3_rope_freqs(int dims, float base, float factor,
                                    float low_freq_factor, float high_freq_factor,
                                    float old_context_len);
float llm_mlx_yarn_mscale(float factor, float mscale);
mlx_array llm_mlx_yarn_rope_freqs_cached(int dims, float base, float factor,
                                         int orig_max, float beta_fast, float beta_slow);
bool llm_mlx_yarn_rope_apply(mlx_array* q_out, mlx_array* k_out,
                             int head_dim, float rope_theta, float factor, int orig_max,
                             float beta_fast, float beta_slow, float mscale_param,
                             mlx_array q_t, mlx_array k_t, int offset, mlx_stream stream);

/* --- dominio MoE (llm_mlx_moe.c) --- */
/* Definiti in llm_mlx_moe.c, usati dal monolite/altri domini. */
bool llm_mlx_moe_gateup_gather(GmoeFfnQuant* fq, mlx_array x, mlx_array lhs_indices,
                               mlx_array rhs_indices, bool sorted_indices,
                               mlx_optional_int gsz_opt, mlx_optional_int bts_opt,
                               const char* mode,
                               mlx_array* out_gate, mlx_array* out_up, mlx_stream stream);
int llm_mlx_moe_gather_mm(const ModelConfig* cfg, mlx_array* res, mlx_array x, mlx_array w,
                          mlx_array scales, mlx_array biases, mlx_array lhs, mlx_array idxs,
                          bool transpose, mlx_optional_int gsz, mlx_optional_int bts,
                          const char* mode, bool sorted, mlx_stream s);
/* Wrapper streaming esperti (UNICO punto per tutte le ffn MoE): delega a llm_mlx_moe_gather_mm se
 * l'ingest non e' attivo, altrimenti rimappa gli indici + rebuild dall'ingest (layer+proj). */
int llm_mlx_moe_gather_streamed(const ModelConfig* cfg, int layer, int proj,
                                mlx_array* res, mlx_array x, mlx_array w, mlx_array s, mlx_array b,
                                mlx_array lhs, mlx_array idxs, bool transpose,
                                mlx_optional_int gsz, mlx_optional_int bts,
                                const char* mode, bool sorted, mlx_stream st);
/* Byte totali degli esperti (dall'ingest SWAP attivo), da sottrarre all'active memory. */
uint64_t llm_mlx_moe_swap_expert_bytes(void);
/* Offload esperti (streaming da disco): stato impostato via API `llm_ai_mlx_set_moe_swap_dir`. */
int llm_mlx_moe_swap_on(void);
/* TurboQuant: 1 = esperti packed (memoria), 0 = affine-2bit residenti (velocita'). */
int llm_mlx_tq_mem_opt(void);
/* true se l'ingest SWAP e' realmente aperto (streaming utilizzabile). Condizione per saltare la
 * materializzazione degli esperti nel synth (`turboquant_synthesize_weights`), a differenza di
 * `swap_on` che riflette solo la dir impostata. */
int llm_mlx_moe_swap_active(void);
const char* llm_mlx_moe_swap_dir(void);
/* Kernel MoE-fuso stream-aware: OFF usa i tensori completi, ON gli stacked degli esperti attivi
 * (indici locali) — stesso kernel, nessuna divergenza per-modello. */
bool llm_mlx_moe_fused_stream_aware(const ModelConfig* cfg, int layer, mlx_array x1,
    mlx_array top_inds_arr, mlx_array gw, mlx_array gs, mlx_array gb,
    mlx_array uw, mlx_array us, mlx_array ub, mlx_array dw, mlx_array ds, mlx_array db,
    int gu_bits, int gu_group, int d_bits, int d_group, int act,
    mlx_array scores_flat, mlx_array* out_summed, mlx_stream st);
bool llm_mlx_reshape_canonical_enabled(void);
GmoeFfnQuant* llm_mlx_moe_ffn_quant(LlmMlxModel* m, int layer_idx);
/* Definiti nel monolite, usati dal dominio MoE. */
bool llm_mlx_fuse_moe_decode_enabled(void);
bool llm_mlx_fuse_moe_decode_verify_enabled(void);
bool llm_mlx_fuse_swiglu_metal_enabled(void);
void llm_mlx_imatrix_accumulate_rows(LlmMlxModel* m, int layer_idx, mlx_array x3d,
                                     mlx_array act_rows, const uint32_t* row_expert,
                                     const uint32_t* row_src, int n_rows);
bool mlx_array_set_optional(mlx_array* out, mlx_array value);
const char* llm_env_get(const char* name);
/* --- dominio Kimi (llm_mlx_kimi.c) --- */
bool kimi_mla_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                              mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool kimi_kda_conv_weight(LlmMlxModel* m, int layer_idx, mlx_stream stream,
                          mlx_array cqw, mlx_array ckw, mlx_array cvw, mlx_dtype act_dt, mlx_array* out);
bool kimi_delta_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool kimi_linear_attention_dispatch(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                    mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool kimi_linear_ffn_dispatch(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);
/* Definiti nel monolite, usati dal dominio Kimi. */
bool dense_swiglu_ffn(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);
GdenseFfnQuant* llm_mlx_dense_ffn_quant(LlmMlxModel* m, int layer_idx, int gemma4);
bool dense_geglu_ffn(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);
bool glm4_moe_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);
bool mlx_fast_rms_norm_no_scale(mlx_array* res, mlx_array x, int dims, float eps, mlx_stream stream);
bool llm_env_isset(const char* name);
/* --- dominio Qwen4-Exp operativo (llm_mlx_qwen4exp_ops.c) --- */
bool qwen4_exp_gated_residual_q(LlmMlxModel* m, mlx_array hyper_input, mlx_array hc_norm_w,
                                mlx_array down_w, mlx_array down_s, mlx_array down_b,
                                mlx_array up_w, mlx_array up_s, mlx_array up_b,
                                mlx_array inject_w, mlx_array inject_s, mlx_array inject_b,
                                mlx_array* out_mixed, mlx_array* out_residual, mlx_array* out_injection);
bool qwen4_exp_ngram_table_open(const char* path, LlmMlxModel* m);
bool qwen4_exp_ngram_table_open_shards(const char* model_dir, LlmMlxModel* m);
bool qwen4_exp_attention_dispatch(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                  mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool qwen4_exp_ffn_dispatch(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);
/* Definiti nel monolite, usati dal dominio Qwen4-Exp operativo. */
bool qwen4_exp_moe_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);
bool qwen4_full_qsa_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                          mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool qwen4_linear_gdn_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                            mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
/* --- dominio Mixtral --- */
/* Definiti nel monolite, usati dal dominio Mixtral. */
bool llm_env_memprof(void);
bool llm_env_mixtral_time(void);
/* --- dominio GLM/DeepSeek MoE FFN (llm_mlx_glm_ds_ffn.c) --- */
bool llm_glm_sorted_enabled(void);
bool llm_ds_sorted_enabled(void);
bool deepseek_v2_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);
/* Definito nel monolite, usato dal dominio GLM/DeepSeek MoE FFN. */
void llm_mlx_imatrix_accumulate(LlmMlxModel* m, int layer_idx, mlx_array x3d,
                                mlx_array mid4d, const uint32_t* top_inds, int n_tokens);
/* --- dominio GPT-OSS FFN/attn: prototipi in oss/llm_oss.h --- */
/* Definiti nel monolite, usati dal dominio GPT-OSS. */
bool llm_mlx_fuse_moe_mxfp4_enabled(void);
bool llm_mlx_fuse_moe_mxfp4_verify_enabled(void);
void llm_mlx_imatrix_energy_add(LlmMlxModel* m, float* dst, int cols, mlx_array x3d, int n_tokens);
bool llm_mlx_kvmetal_populate_batch(void** kvmetal_ctx_arr, int layer_idx, LlmMlxModel* m,
                                    int cache_offset, int n_tokens, mlx_array k_rope, mlx_array v_t);
bool llm_mlx_kvmetal_step(void** kvmetal_ctx_arr, int layer_idx, LlmMlxModel* m, int cache_offset,
                          mlx_array q_rope, mlx_array k_rope, mlx_array v_t, mlx_array* attn_out);
/* Definiti nel monolite, usati dal dominio session (llm_mlx_session_lifecycle). */
bool llm_mlx_gemvq4_enabled(void);
bool llm_mlx_gemvq4_verify_enabled(void);
extern void* g_gemvq4_id;
extern void** g_gemvq4_ctx;
extern const void** g_gemvq4_key;
extern int g_gemvq4_cap;
extern int g_gemvq4_disabled;
#ifndef LLM_GEMVQ4_MAX_PER_LAYER
#define LLM_GEMVQ4_MAX_PER_LAYER 8
#endif
bool llm_mlx_mxfp4_act_f16_enabled(void);
mlx_array llm_mlx_scalar_act(float v, mlx_stream stream);
/* (llm_mlx_pad512_reset/gateup_reset: prototipi in oss/llm_oss.h) */
/* --- dominio self-test eval (llm_mlx_selftest_eval.c) --- */
/* Definito nel monolite, usato dal dominio self-test eval (oQ A8). */
LlmMetalOqA8Ctx* llm_mlx_oq_a8_ctx(void);
/* --- dominio session swap (llm_mlx_session_swap.c) --- */
/* Definiti nel monolite, usati dal dominio session swap. */
LlmMlxSession* llm_mlx_session_create(LlmMlxModel* m);
void llm_mlx_session_destroy(LlmMlxSession* s);
bool write_mlx_meta_file(const char* path, const MlxSwapMeta* meta);
bool read_mlx_meta_file(const char* path, MlxSwapMeta* out);
/* --- dominio pesi/shard/quant-synth (llm_mlx_weights_io.c) --- */
/* Definiti in llm_mlx_weights_io.c, usati dal monolite (load_model). */
void llm_mlx_wcache_clear(void);
void multishard_free_all(MultiShardWeights* mw);
bool load_all_shards(const char* model_dir, mlx_stream stream, MultiShardWeights* weights);
int awq_synthesize_weights(MultiShardWeights* weights, mlx_stream stream);
int gptq_nf4_synthesize_weights(MultiShardWeights* weights, mlx_stream stream);
int turboquant_synthesize_weights(MultiShardWeights* weights, mlx_stream stream);
bool llm_mlx_pretouch_weights_enabled(void);
bool kimi_linear_synthesize_absorbed_mla(const ModelConfig* cfg, MultiShardWeights* weights, mlx_stream stream);
void llm_mlx_pretouch_weights(const MultiShardWeights* weights);
bool stack_moe_experts(const ModelConfig* cfg, MultiShardWeights* weights, mlx_stream stream,
                       const char* base, const char* s0, const char* s1, const char* s2);
bool fuse_mixtral_gate_up(const ModelConfig* cfg, MultiShardWeights* weights, mlx_stream stream);
/* --- dominio GGUF/model IO (llm_mlx_gguf_io.c + monolite) --- */
const LlmMlxArchOps* find_arch_ops(const char* model_type);
/* Definiti in llm_mlx_gguf_io.c, usati dal monolite (load_model). */
size_t llm_mlx_dir_safetensors_bytes(const char* dir);
void llm_mlx_err_trace_handler(const char* msg, void* data);
bool llm_mlx_load_generic_gguf(const char* gguf_path, LlmMlxModel* m, const char* model_type);
int llm_mlx_resolve_arch(const char* path, LlmDesc* out);
/* Definiti nel monolite, usati dal dominio model IO (llm_mlx_model_io.c). */
bool load_config(const char* model_dir, ModelConfig* cfg);
void llm_mlx_model_init(LlmMlxModel* m);
void llm_mlx_model_destroy(LlmMlxModel* m);
void llm_mlx_kvseg_reset_all(void);
/* --- dominio session spec/verify (llm_mlx_session_spec.c) --- */
/* Definiti nel monolite, usati dal dominio session spec/verify. */
bool forward_one_token(LlmMlxModel* m, LlmMlxSession* kvq_sess, mlx_array embed_w, mlx_array embed_s, mlx_array embed_b,
                       int32_t token_id, int cache_offset,
                       mlx_array* k_cache, mlx_array* v_cache, bool* has_cache, mlx_array* out_h);
bool final_step(LlmMlxModel* m, mlx_array h, mlx_array norm_w,
                mlx_array out_w, mlx_array out_s, mlx_array out_b,
                llm_mlx_sampling_t sampling, int32_t* out_id);
/* Batch greedy (verify spec n-gram): h `[1,nd,H]` -> nd id, una norm + una matmul lm_head. */
bool final_step_greedy_batch(LlmMlxModel* m, mlx_array h, mlx_array norm_w,
                             mlx_array out_w, mlx_array out_s, mlx_array out_b,
                             int32_t* out_ids, int n_out);
extern mlx_array g_spec_snap_k[];
extern mlx_array g_spec_snap_v[];
extern int g_spec_capture;
/* Precarica il drafter DFlash2 PRIMA del prefill (evita l'OOM del picco prefill + drafter). */
void llm_mlx_df2_preload(LlmMlxModel* m);
/* 1 se la generazione usera' un drafter DFlash2 (opt `dflash_model` esplicita OPPURE auto-detect
 * del sibling <stem>-DFlash2). Usato per NON usare la radix prefix-cache (append_reuse_prefix non
 * fa il prefill -> cattura-ctx non armata -> drafter degenerato). Diagnosi 2026-10-03. */
int llm_mlx_df2_will_use(LlmMlxModel* m);
/* HOOK cattura-DFlash2-durante-prefill (in forward_tokens_batch): armato da llm_mlx_df2_preload. */
extern int g_df2_pf_active;
extern void (*g_df2_pf_hook)(int layer, mlx_array h, int n_tokens, int cache_offset);
bool llm_mlx_df2_pf_arm(DFlash2Runtime* rt, LlmMlxModel* m);
void llm_mlx_df2_pf_disarm(void);
bool llm_mlx_df2_pf_take(float** out, int* out_n);
bool llm_mlx_logits_f32(mlx_array* out, mlx_array logits, mlx_stream stream);
bool final_step_constrained(LlmMlxModel* m, mlx_array h, mlx_array norm_w,
                            mlx_array out_w, mlx_array out_s, mlx_array out_b,
                            const int32_t* allowed_ids, int n_allowed, int32_t* out_id);
/* Definiti nel monolite, usati dal dominio sampling/final_step (llm_mlx_final_step.c). */
bool llm_mlx_sampler_legacy_enabled(void);
bool llm_mlx_sampler_ondevice_enabled(void);
/* --- dominio session generate (llm_mlx_session_generate.c) --- */
/* Definiti nel monolite, usati dal dominio session generate. */
bool llm_mlx_fixkv_enabled(void);
bool llm_no_clear_cache(void);
bool llm_mlx_compile_enabled(void);
bool llm_mlx_compile_arch_supported(const LlmMlxModel* m);
bool llm_mlx_static_cache_ready(const LlmMlxSession* s);
bool forward_one_token_dev(LlmMlxModel* m, LlmMlxSession* kvq_sess, mlx_array embed_w, mlx_array embed_s, mlx_array embed_b,
                           mlx_array token_id_arr, int cache_offset,
                           mlx_array* k_cache, mlx_array* v_cache, bool* has_cache, mlx_array* out_h);
bool final_step_greedy_dev(LlmMlxModel* m, mlx_array h, mlx_array norm_w,
                           mlx_array out_w, mlx_array out_s, mlx_array out_b,
                           mlx_array* out_id_arr);
bool llm_mlx_session_run_compiled_step_arr(LlmMlxSession* s, mlx_array tid_in, int pos,
                                           mlx_array* out_h);
bool llm_mlx_session_run_compiled_step(LlmMlxSession* s, int32_t token_id, int pos,
                                       mlx_array* out_h);
bool forward_one_token_batch(LlmMlxModel* m, LlmMlxSession** sess, int n_sess,
                             const int32_t* token_ids,
                             mlx_array embed_w, mlx_array embed_s, mlx_array embed_b,
                             mlx_array* out_h);
/* --- dominio attn batch varlen (llm_mlx_attn_batch.c) --- */
/* Definiti nel monolite, usati dal dominio attn batch varlen. */
bool llm_mlx_kvseg_enabled(void);
bool llm_mlx_kvslice_enabled(void);
bool llm_mlx_kv_f16_enabled(void);
bool llm_mlx_kvmetal_enabled(void);
bool llm_mlx_kvmetal_verify_enabled(void);
bool llm_mlx_kvmetal_auth_enabled(void);
bool llm_mlx_kvmetal_auth_layer_ready(int layer_idx);
GattnQuant* llm_mlx_attn_quant(LlmMlxModel* m, int layer_idx);
mlx_array llm_mlx_llama3_freqs_cached(LlmMlxModel* m);
bool llm_mlx_flash_decode_enabled(void);
bool llm_mlx_flash_decode_verify_enabled(void);
bool llm_mlx_flash_prefill_enabled(void);
GattnQuant* llm_mlx_gemma4_attn_quant(LlmMlxModel* m, int layer_idx);
void tname_gemma4(int layer_idx, const char* suffix, char* out, size_t out_cap);
void llm_mlx_kv_compact_prefix(int n_layers, int cache_offset, mlx_stream stream,
                               mlx_array* k_cache, mlx_array* v_cache);
/* --- dominio Gemma4+Mixtral attenzione (llm_mlx_gemma_mixtral.c) --- */
bool gemma4_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                            mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool mixtral_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                             mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
/* --- dominio Qwen3-MoE + Qwen3.5/Next attenzione (llm_mlx_qwen3_attn.c) --- */
bool llm_bonsai_signs_right(void);
/* --- dominio MLA DeepSeek-V2 + GLM-4.5 attenzione (llm_mlx_mla_glm4.c) --- */
bool deepseek_v2_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                 mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool glm4_moe_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                              mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
/* --- dominio GDN (llm_mlx_gdn.c) --- */
/* Definito in llm_mlx_gdn.c, usato dal monolite (layer dispatch). */
bool gdn_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                         mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
/* Definiti nel monolite, usati dal dominio GDN. */
bool llm_mlx_gdn_eval_forced(void);
bool llm_mlx_gdn_kernel_dbg_enabled(void);
bool llm_mlx_gdn_inproj_compiled(int li, mlx_array h,
                                 mlx_array* qkv, mlx_array* z, mlx_array* av, mlx_array* bv,
                                 mlx_array qkv_w, mlx_array qkv_sc, mlx_array qkv_b,
                                 mlx_array z_w, mlx_array z_sc, mlx_array z_b,
                                 mlx_array a_w, mlx_array a_sc, mlx_array a_b,
                                 mlx_array b_w, mlx_array b_sc, mlx_array b_b,
                                 int gsz, int bts, const char* mode);
bool llm_mlx_gdn_gate_metal_enabled(void);
bool llm_mlx_m1_block_enabled(void);
bool llm_mlx_compile_qwen35_enabled(void);
bool llm_mlx_gdn_block_compiled(LlmMlxModel* m, int li, mlx_array x,
                                mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool llm_mlx_oq_a8_stage_try(mlx_array x, mlx_array wref, mlx_array scref, mlx_array biref,
                             int e_group, int e_bits, const char* e_mode, mlx_stream stream,
                             mlx_array* qa, mlx_array* sa, mlx_array* ra);
bool llm_mlx_oq_a8_qmm_stage_try(mlx_array* res, mlx_array qa, mlx_array sa, mlx_array ra,
                                 mlx_array xref, mlx_array w, mlx_array scales, mlx_array biases,
                                 int e_group, int e_bits, const char* e_mode, mlx_stream stream);
/* --- dominio Qwen3.5/Next (llm_mlx_qwen35.c) --- */
/* Definiti in llm_mlx_qwen35.c, usati dal monolite (ARCH_TABLE). */
bool qwen3_5_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                             mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool qwen3_next_linear_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                             mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool qwen3_next_attention_dispatch(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                   mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
/* Definiti nel monolite, usati dal dominio Qwen3.5/Next. */
bool qwen3_moe_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                               mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool qwen3_next_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
/* Definiti nel monolite, usati dal dominio MoE FFN Qwen3.5/4-Exp (llm_mlx_qwen35_moe_ffn.c). */
bool qwen3_5_moe_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);
bool llm_mlx_cast_f16_inplace(mlx_array* lhs, mlx_stream stream);
/* Definito nel monolite, usato dal dominio Qwen2 attenzione (llm_mlx_qwen2_attn.c). */
bool qwen2_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                           mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
/* --- dominio Llama (llm_mlx_llama.c) --- */
/* Definito in llm_mlx_llama.c, usato dal monolite (ARCH_TABLE). */
bool llama_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                           mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
/* LFM2 (Liquid): ibrido attn-full (GQA + q/k RMSNorm) / ShortConv. */
bool lfm2_attention_dispatch(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                    mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool lfm2_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);
/* OLMo-3 (AllenAI): dense, GQA + q/k RMSNorm FLAT, layer POST-NORM. */
bool olmo3_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                  mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool olmo3_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);
/* OLMo-Hybrid: dispatcher hybrid (linear_attention = GatedDeltaNet, full_attention = GQA senza RoPE). */
bool olmo_hybrid_attention_dispatch(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                    mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool olmo_hybrid_ffn_dispatch(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);
/* Granite-MoE-Hybrid (IBM): dispatcher hybrid (mamba = Mamba2 SSD, attention = GQA senza RoPE). */
bool granite_hybrid_attention_dispatch(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                       mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool granite_hybrid_ffn_dispatch(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);
/* GLM-4 denso (model_type "glm4"). */
bool glm4_attention_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                          mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool glm4_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);
/* MoE v2 (parametrizzato): FFN MoE copre piu' famiglie via config. */
bool moe_v2_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);
/* Mamba (SSM selettivo): layer ricorrente; ffn = identita'. */
bool mamba_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                        mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool mamba_ffn_identity(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);
bool mamba2_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                         mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool mamba2_ffn_identity(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);
bool falcon_h1_attn_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                                 mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool falcon_h1_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);
bool jamba_attn_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                            mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool jamba_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);
/* RWKV7: time-mixing ricorrente (WKV7) + channel-mixing. */
bool rwkv7_attn_block(LlmMlxModel* m, int layer_idx, int cache_offset, int n_tokens, mlx_array x,
                      mlx_array* k_cache, mlx_array* v_cache, bool has_cache, mlx_array* out_h);
bool rwkv7_ffn_block(LlmMlxModel* m, int layer_idx, int n_tokens, mlx_array h, mlx_array* out);
/* Definiti nel monolite, usati dal dominio Llama. */
bool llm_mlx_fuse_rope_enabled(void);
bool llm_mlx_fuse_rope_verify_enabled(void);
bool llm_mlx_kvseg_verify_enabled(void);
void llm_mlx_kvseg_reset_layer(int layer_idx);
bool llm_mlx_kvseg_step(mlx_stream stream, int nkv, int hd, int layer_idx,
                        int cache_offset, int n_tokens,
                        mlx_array k_rope, mlx_array v_t,
                        mlx_array* k_cache_out, mlx_array* v_cache_out,
                        mlx_array* k_piece1, mlx_array* v_piece1,
                        mlx_array* k_piece2, mlx_array* v_piece2,
                        int* n_pieces);
bool llm_mlx_kvseg_merge_two(LlmMlxModel* m, mlx_array q_rope,
                             mlx_array k1, mlx_array v1,
                             mlx_array k2, mlx_array v2,
                             mlx_array* out);
int llm_mlx_kv_cap_for(int needed);
extern void* g_kvmetal_id;
extern void** g_kvmetal_ctx;
extern int g_kvmetal_n_layers;
extern int g_kvmetal_disabled;
extern char* g_kvmetal_pop_ok;

#endif /* LLM_MLX_PROTOS_H */
