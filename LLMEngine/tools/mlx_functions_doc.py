#!/usr/bin/env python3
"""mlx_functions_doc.py — genera l'indice function-level (una riga per funzione) del motore MLX.
Uso: python3 tools/mlx_functions_doc.py > /tmp/mlx_functions_table.md
Fonte: LLMEnv/LLMEngine/mlx/{gmb_host_ai_mlx.c, gmb_host_ai_mlx.h, llm_mlx_ops.h,
llm_mlx_moe_fused_kernel.h, llm_mlx_packed_qkv.h, llm_mlx_radix.h}. Header dsv41 esclusi
(documentati in DeepSeek-V41-Funzioni-Implementate).
"""
import os
import re

BASE = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx")
FILES = ["gmb_host_ai_mlx.c", "gmb_host_ai_mlx.h", "llm_mlx_ops.h",
         "llm_mlx_moe_fused_kernel.h", "llm_mlx_packed_qkv.h", "llm_mlx_radix.h"]

FUNC_RE = re.compile(
    r'(?m)^([A-Za-z_][A-Za-z0-9_\s\*]*?)\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^;{]*?)\)\s*\{',
    re.S)
KW = {"if", "for", "while", "switch", "return", "sizeof", "do", "else"}


def classify(n):
    if n.endswith("_verify_enabled"):
        return "flag: verifica parità raw-vs-fuso"
    if n.endswith("_enabled"):
        return "flag opt-in (kill-switch)"
    if any(s in n for s in ("_selftest", "smoketest", "_bench", "_checkpoint", "_drift")):
        return "self-test/diagnostica (non percorso di produzione)"
    if n.startswith("gmb_cb_"):
        return "lab di confronto raw-MLX vs path GMBasic"
    if n.startswith(("df2_", "dflash_")):
        return "DFlash2 (speculazione block-diffusion)"
    if n.startswith(("gmb_train_",)) or "imatrix" in n or "dpo" in n or "loss_fn" in n:
        return "training / imatrix"
    if n.startswith(("gmb_radix_",)):
        return "radix prefix cache"
    if n.startswith("llm_mlx_qkv_"):
        return "QKV impacchettato"
    if n.startswith("llm_mlx_moe_") and "kernel_build" in n:
        return "kernel Metal MoE (build)"
    if n in ("llm_mlx_moe_fused_decode", "llm_mlx_moe_fused_decode_mxfp4"):
        return "MoE fused decode (kernel)"
    if n.startswith("llm_mlx_kv"):
        return "primitiva KV (VeloxQuant: quant/pack/select/retain/rot/...)"
    if n.startswith("llm_mlx_cache_"):
        return "cache checkpoint (commit/rollback/trim)"
    if "gqa_sdpa" in n or "decode_sdpa" in n or n.endswith("_sdpa"):
        return "SDPA (attenzione)"
    if n.startswith("llm_mlx_gated_delta_step"):
        return "GatedDeltaNet step"
    if n.startswith(("llm_mlx_gdn_",)):
        return "primitive GDN (conv/gate/norm)"
    if n.endswith("_attention_block") or n.endswith("_attention_dispatch") or "attention_block" in n:
        return "blocco attenzione per-architettura"
    if n.startswith("kimi_mla_attention_block") or n.startswith("kimi_delta_attention_block"):
        return "blocco attenzione per-architettura"
    if n.endswith("_ffn_block") or n.endswith("_ffn") or n.endswith("_ffn_dispatch") or "moe_block" in n:
        return "blocco FFN/MoE per-architettura"
    if n in ("dense_swiglu_ffn", "dense_geglu_ffn"):
        return "FFN denso (SwiGLU/GeGLU)"
    if n.startswith(("multishard_", "quantized_linear", "get_quantized", "plain_linear")):
        return "quantizzazione/pesi (load, dequant)"
    if any(s in n for s in ("scales_dtype", "act_dt", "derive_bits", "quant_derive", "_quant",
                            "bf16_mode", "f16_scales", "f32_scales", "bf16_scales", "promote_scales",
                            "auto_pair", "is_scales", "is_norm_weight")):
        return "policy dtype/quantizzazione"
    if n.startswith(("llm_ai_mlx_dflash2",)):
        return "DFlash2 ABI"
    if n.startswith("llm_ai_mlx_session_") or n.startswith("llm_ai_mlx_create_session") or \
       n.startswith("llm_ai_mlx_destroy_session"):
        return "ABI sessione"
    if n.startswith("llm_ai_mlx_"):
        return "ABI pubblica"
    if n.startswith("llm_mlx_session_"):
        return "sessione"
    if n.startswith("llm_mlx_prefix") or "kvdisk" in n or n.startswith("gmb_session_"):
        return "prefix/offload/kvdisk"
    if n.startswith("gmb_env_"):
        return "env/flag"
    if n.startswith(("llm_mlx_yarn", "llm_mlx_proportional_rope", "llm_mlx_llama3")):
        return "RoPE (frequenze/apply)"
    if n.startswith(("fuse_", "llm_mlx_fuse", "llm_mlx_swiglu", "llm_mlx_geglu",
                     "llm_mlx_moe_gateup", "llama_try_fused", "stack_mixtral", "llm_mlx_concat2",
                     "llm_mlx_pad", "gmb_kern_")):
        return "fusione/helper elementwise"
    if n.startswith(("llm_mlx_compile", "llm_mlx_static_cache", "llm_mlx_async",
                     "llm_mlx_batch_varlen", "llm_mlx_reshape", "llm_mlx_pretouch")):
        return "compile/scheduling"
    if n.startswith(("llm_mlx_oq_a8", "llm_mlx_gemvq4")):
        return "A8/oQ (quant runtime)"
    if n.startswith("final_step"):
        return "sampling / final step"
    if n.startswith("forward") or n == "layer_forward":
        return "forward"
    if n.startswith(("load_config", "llm_mlx_model_", "llm_ai_mlx_load_model", "llm_ai_mlx_free_model")):
        return "load/modello"
    if n.startswith(("embed_token", "embed_tokens")):
        return "embedding"
    if n.startswith(("e9_",)):
        return "helper top-k"
    if n.startswith(("q4_sigmoid", "qwen4_")):
        return "Qwen4-Exp (PLE/QSA/hc/GDN)"
    if n.startswith(("llm_mlx_gptoss",)):
        return "gpt-oss (memo/bias)"
    if n.startswith(("llm_mlx_imatrix",)):
        return "imatrix"
    if n.startswith(("llm_mlx_dir", "llm_mlx_self_dir", "llm_mlx_dp_ms", "llm_mlx_active_guard",
                     "llm_mlx_max_abs_diff", "llm_mlx_layer_proj", "llm_mlx_model_lora",
                     "tname", "strset_", "mlx_array_set_optional", "mlx_fast_rms_norm_no_scale",
                     "write_mlx_meta", "read_mlx_meta", "free_output_tensors", "llm_mlx_scalar_act",
                     "llm_mlx_quant_derive", "llm_mlx_derive_bits")):
        return "utility"
    return "helper del motore"


def main():
    rows = []
    for fn in FILES:
        path = os.path.join(BASE, fn)
        if not os.path.exists(path):
            continue
        src = open(path).read()
        seen = set()
        for m in FUNC_RE.finditer(src):
            name = m.group(2)
            if name in KW or name in seen:
                continue
            seen.add(name)
            sig = re.sub(r'\s+', ' ', (m.group(1) + name + "(" + m.group(3).strip() + ")").strip())
            sig = sig.replace("|", "\\|")
            rows.append((fn, name, sig, classify(name)))
    rows.sort(key=lambda r: (r[0], r[1]))
    print("| file | funzione | firma | scopo |")
    print("|---|---|---|---|")
    cur = None
    for fn, name, sig, purp in rows:
        print(f"| `{fn}` | `{name}` | `{sig}` | {purp} |")
    print(f"\n_({len(rows)} funzioni)_")


if __name__ == "__main__":
    main()
