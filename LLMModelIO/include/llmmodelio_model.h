#include <stddef.h>
#include "llmengine_plugin.h"
#include <stdint.h>

uint16_t llmmodelio_fp32_to_fp16(float f);
void llmmodelio_model_policy_set(const char* role, const char* fmt);
void llmmodelio_model_policy_reset(void);
const char* llmmodelio_model_policy_get(const char* role);
int llmmodelio_model_quantize_auto(const char* in_path, const char* out_path, double target_bpw);
int llmmodelio_model_extract(const char* in_path, const char* out_path, int64_t layer, int64_t expert);
int llmmodelio_model_merge(const char* a_path, const char* b_path, const char* out_path);
int llmmodelio_arch_resolve(const char* hf_dir);
const char* llmmodelio_arch_name(void);
const char* llmmodelio_arch_path(void);
const char* llmmodelio_arch_role_of(const char* tensor);
const char* llmmodelio_arch_fmt_of(const char* role);
void llmmodelio_arch_show(void);
int llmmodelio_arch_check(const char* hf_dir);
int llmmodelio_arch_plan(const char* hf_dir, const char* out_path);
int llmmodelio_arch_support(const char* hf_dir, char* out, size_t cap);
int llmmodelio_codec_resolve(const char* hf_dir, const char* fmt, LlmCodec* out, char* plugin_path, size_t cap);
int llmmodelio_prim_resolve(const char* hf_dir, const char* kind, LlmEnginePluginPrim* out, char* plugin_path, size_t cap);
int llmmodelio_arch_pipeline(const char* hf_dir, const char* kind, char* out, size_t cap);
void llmmodelio_arch_list(void);
int llmmodelio_arch_cmd(const char* sub, const char* a, const char* b);
/* Come sopra ma cattura il testo in out (per i backend gestiti via P/Invoke). */
int llmmodelio_arch_cmd_to(const char* sub, const char* a, const char* b, char* out, size_t cap);
int llmmodelio_policy_cmd_to(const char* sub, const char* a, const char* b, char* out, size_t cap);
int llmmodelio_policy_cmd(const char* sub, const char* a, const char* b);
/* Model compiler LLM: verbi TENSOR/KERNEL/MODULE/KV (superficie .llmarch data-driven). */
int llmmodelio_tensor_cmd(const char* sub, const char* a, const char* b);
int llmmodelio_kernel_cmd(const char* sub, const char* a, const char* b);
int llmmodelio_module_cmd(const char* sub, const char* a, const char* b);
int llmmodelio_kv_cmd(const char* sub, const char* a, const char* b);
int llmmodelio_tensor_cmd_to(const char* sub, const char* a, const char* b, char* out, size_t cap);
int llmmodelio_kernel_cmd_to(const char* sub, const char* a, const char* b, char* out, size_t cap);
int llmmodelio_module_cmd_to(const char* sub, const char* a, const char* b, char* out, size_t cap);
int llmmodelio_kv_cmd_to(const char* sub, const char* a, const char* b, char* out, size_t cap);
/* LLM MODEL CONVERT <src>, <dst>[, <fmt>] — GGUF(-2 = percorso MoE) | GGUF | MLX (codec). */
int llmmodelio_model_convert(const char* src, const char* dst, const char* fmt);
int llmmodelio_model_profile_compute(const char* model_path, const char* ids_path, const char* out_path, int64_t chunk);
int llmmodelio_model_profile(const char* model_path, const char* out_path);
int llmmodelio_model_prune(const char* in_path, const char* out_path, int64_t keep);
void llmmodelio_model_policy_show(void);
/* llmmodelio_model.h — MODEL COMPILER (namespace LLM): operazioni sui file modello. */
#ifndef LLMMODELIO_MODEL_H
#define LLMMODELIO_MODEL_H

/* Round-trip GGUF: legge metadata KV + tensor-info da `in_path` e riscrive un GGUF nuovo
 * (stessi metadata, byte dei tensori) in `out_path`. 0 ok / -1 errore. */
int llmmodelio_model_save(const char* in_path, const char* out_path);

/* Round-trip con QUANTIZE: F32 (ultima dim %32==0) -> Q8_0; altri tensori copiati. 0/-1. */
int llmmodelio_model_quantize(const char* in_path, const char* out_path);
int llmmodelio_model_ablate(const char* in_path, const char* out_path, const char* pattern);

/* Q4_K weighted, stessa convenzione di llama.cpp: quant_weights contiene `weight_sets` vettori
 * consecutivi da n_per_row elementi; ogni vettore si applica a nrows/weight_sets righe. */
int llmmodelio_quantize_q4_k_weighted(const float* src, int64_t nrows, int64_t n_per_row,
                                   uint8_t* dst, const float* quant_weights, int64_t weight_sets);

/* Scrive il PIANO (per-tensore: ruolo + formato proposto) in out_path. 0/-1. */
int llmmodelio_model_plan(const char* in_path, const char* out_path);

#endif
