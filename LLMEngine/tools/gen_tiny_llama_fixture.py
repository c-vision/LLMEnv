#!/usr/bin/env python3
# tools/gen_tiny_llama_fixture.py — genera un modello MLX tiny (llama) engine-loadable per i
# MiniTest della PPL reale (LLM REGISTRY MEASURE / LLM QUALITY). Deterministico (seed fisso).
#
# Uso: ~/ai/env-mlxlm/bin/python tools/gen_tiny_llama_fixture.py <out_dir>
# Richiede numpy + safetensors. NB: usa head_dim=8 (standard); head_dim molto piccolo (es. 4) era
# un problema del kernel Metal fused `fused_rope_qkv_layout` dell'engine, ora RISOLTO lato LLMEnv
# (skip del kernel per head_dim<8 -> fallback non-fuso): vedi llm_mlx_llama.c.
import json, sys, os, shutil
import numpy as np
from safetensors.numpy import save_file

out = sys.argv[1] if len(sys.argv) > 1 else "samples/Fixtures/models/tiny_llama"
h, nh, nkv, hd, inter, vocab = 16, 2, 1, 8, 32, 259   # vocab = tiny_tokenizer.json (259)
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(7)
f16 = lambda s: (rng.standard_normal(s) * 0.05).astype(np.float16)
save_file({
    "model.embed_tokens.weight": f16((vocab, h)),
    "model.layers.0.input_layernorm.weight": np.ones((h,), np.float16),
    "model.layers.0.self_attn.q_proj.weight": f16((nh*hd, h)),
    "model.layers.0.self_attn.k_proj.weight": f16((nkv*hd, h)),
    "model.layers.0.self_attn.v_proj.weight": f16((nkv*hd, h)),
    "model.layers.0.self_attn.o_proj.weight": f16((h, nh*hd)),
    "model.layers.0.post_attention_layernorm.weight": np.ones((h,), np.float16),
    "model.layers.0.mlp.gate_proj.weight": f16((inter, h)),
    "model.layers.0.mlp.up_proj.weight": f16((inter, h)),
    "model.layers.0.mlp.down_proj.weight": f16((h, inter)),
    "model.norm.weight": np.ones((h,), np.float16),
}, os.path.join(out, "model.safetensors"))
json.dump({"architectures": ["LlamaForCausalLM"], "hidden_act": "silu", "hidden_size": h,
           "intermediate_size": inter, "max_position_embeddings": 2048, "model_type": "llama",
           "num_attention_heads": nh, "num_hidden_layers": 1, "num_key_value_heads": nkv,
           "head_dim": hd, "rms_norm_eps": 1e-05, "rope_theta": 10000.0,
           "tie_word_embeddings": True, "vocab_size": vocab},
          open(os.path.join(out, "config.json"), "w"))
shutil.copy("samples/Fixtures/ai/tiny_tokenizer.json", os.path.join(out, "tokenizer.json"))
print("fixture scritto:", out)
