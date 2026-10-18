#!/usr/bin/env python3
"""gen_tiny_ckpt.py — genera un CHECKPOINT "tiny" sintetico per un'arch di mlx_lm.

PERCHE': implementare/validare una arch non deve dipendere dal download di modelli reali
(lenti, GB). Con un tiny (dim minime, pesi casuali deterministici) si valida la STESSA via di
codice dell'arch in pochi secondi, offline. Il modello tiny e' costruito con la classe
`mlx_lm.models.<arch>.Model` (l'ORACOLO) e salvato come config.json + model.safetensors:
l'engine lo carica e si confrontano i logits a parita' di token-ids. Niente tokenizer
(la parita' e' su token-ids espliciti).

Uso:
  gen_tiny_ckpt.py <arch> <out_dir> [--ids 10,20,30]
Stampa, oltre a creare il checkpoint, i riferimenti (top-k logprob) di mlx_lm per quegli ids.
"""
import sys, os, json
import numpy as np
import mlx.core as mx
from mlx.utils import tree_map, tree_flatten

# Config TINY per-arch (dim minime; i parametri arch-specifici con valori "reali" scalati).
# head_dim = hidden/n_heads deve restare sensato per RoPE/GQA.
TINY = {
    "ernie4_5_moe": dict(
        cls="ernie4_5_moe", model_type="ernie4_5_moe",
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5,
        vocab_size=128, max_position_embeddings=512, rope_theta=10000.0,
        use_bias=False, tie_word_embeddings=False,
        moe_num_experts=4, moe_k=2, moe_intermediate_size=64,
        moe_layer_start_index=0, moe_layer_interval=2, moe_num_shared_experts=1,
        moe_gate_act="softmax", head_dim=16,
    ),
    "qwen3_moe": dict(
        cls="qwen3_moe", model_type="qwen3_moe",
        hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        rms_norm_eps=1e-5, vocab_size=128, max_position_embeddings=512,
        rope_theta=10000.0, tie_word_embeddings=False,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=64,
        decoder_sparse_step=1, mlp_only_layers=[], norm_topk_prob=True,
    ),
    "glm4": dict(
        cls="glm4", model_type="glm4",
        hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, attention_bias=True,
        head_dim=16, rms_norm_eps=1e-5, vocab_size=128,
        partial_rotary_factor=0.5, rope_theta=10000.0, rope_traditional=True,
        max_position_embeddings=512,
    ),
    "granite": dict(
        cls="granite", model_type="granite",
        hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5,
        vocab_size=128, logits_scaling=8.0, attention_multiplier=0.015625,
        embedding_multiplier=12.0, residual_multiplier=0.22,
        max_position_embeddings=512, attention_bias=False, mlp_bias=False,
        rope_theta=5000000.0, tie_word_embeddings=True,
    ),
    # SSM/hybrid: esercita il mixer Mamba2 (granite_hybrid block) + i campi cfg granite.ssm_* e mamba.*
    "granitemoehybrid": dict(
        cls="granitemoehybrid", model_type="granitemoehybrid",
        hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5,
        vocab_size=128, logits_scaling=8.0, attention_multiplier=0.015625,
        embedding_multiplier=12.0, residual_multiplier=0.22,
        max_position_embeddings=512, attention_bias=False, mlp_bias=False,
        rope_theta=10000.0, tie_word_embeddings=True,
        layer_types=["mamba", "attention"],
        mamba_n_heads=2, mamba_d_head=32, mamba_d_state=8, mamba_d_conv=2, mamba_n_groups=1,
        mamba_proj_bias=False, mamba_conv_bias=True,
        position_embedding_type="nope", time_step_limit=[0.001, 100.0],
    ),
    # Migrati a moe_v2 (2026-10-17): richiedono il `sanitize` di mlx_lm (espertoni -> switch_mlp),
    # che e' la forma attesa dall'engine per queste due arch.
    "mixtral": dict(
        cls="mixtral", model_type="mixtral",
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5,
        vocab_size=128, rope_theta=10000.0, tie_word_embeddings=False,
        num_local_experts=4, num_experts_per_tok=2, sanitize=True,
    ),
    "lfm2_moe": dict(
        cls="lfm2_moe", model_type="lfm2_moe",
        hidden_size=64, intermediate_size=128, moe_intermediate_size=64,
        num_hidden_layers=4, num_experts=4, num_experts_per_tok=2,
        norm_topk_prob=True, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, use_expert_bias=True, num_dense_layers=1,
        norm_eps=1e-5, conv_bias=True, conv_L_cache=4, vocab_size=128,
        rope_theta=10000.0,
        layer_types=["conv", "full_attention", "conv", "full_attention"],
        sanitize=True,
    ),
    # glm4_moe: SwitchGLU gia' stacked (niente sanitize); n_group=2/topk_group=1 esercita il group-prune.
    "glm4_moe": dict(
        cls="glm4_moe", model_type="glm4_moe",
        hidden_size=64, intermediate_size=128, moe_intermediate_size=64,
        max_position_embeddings=512, norm_topk_prob=True,
        num_attention_heads=4, n_group=2, head_dim=16, topk_group=1,
        n_shared_experts=1, n_routed_experts=4, routed_scaling_factor=1.0,
        num_experts_per_tok=2, first_k_dense_replace=1, num_hidden_layers=4,
        num_key_value_heads=2, rms_norm_eps=1e-5, rope_theta=10000.0,
        use_qk_norm=True, tie_word_embeddings=False, attention_bias=True,
        partial_rotary_factor=0.5, vocab_size=128, rope_scaling=None,
    ),
    # olmoe: SwitchGLU gia' stacked (niente sanitize); router plain mlp.gate, softmax, norm_topk_prob=True/Var.
    "olmoe": dict(
        cls="olmoe", model_type="olmoe",
        hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        rms_norm_eps=1e-5, vocab_size=128, num_experts=4, num_experts_per_tok=2,
        norm_topk_prob=False, rope_theta=10000.0, tie_word_embeddings=False,
        attention_bias=False, mlp_bias=False,
    ),
    # granitemoe: attn granite + FFN MoE (block_sparse_moe.router.layer plain + switch_mlp), residuo scalato.
    "granitemoe": dict(
        cls="granitemoe", model_type="granitemoe",
        hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5,
        vocab_size=128, logits_scaling=8.0, attention_multiplier=0.015625,
        embedding_multiplier=12.0, residual_multiplier=0.22,
        max_position_embeddings=512, attention_bias=False, rope_theta=10000.0,
        num_local_experts=4, num_experts_per_tok=2, tie_word_embeddings=False,
    ),
    "lille-130m": dict(
        cls="lille-130m", model_type="lille-130m",
        block_size=512, layer_norm_eps=1e-5, n_embd=64, n_head=4, n_kv_heads=2,
        n_layer=2, rope_theta=10000.0, vocab_size=128, tie_word_embeddings=True,
    ),
    "gpt_neox": dict(
        cls="gpt_neox", model_type="gpt_neox",
        max_position_embeddings=512, hidden_size=64, num_attention_heads=4,
        num_hidden_layers=2, layer_norm_eps=1e-5, vocab_size=128,
        rotary_emb_base=10000.0, rotary_pct=0.25, use_parallel_residual=True,
    ),
    "qwen": dict(
        cls="qwen", model_type="qwen",
        hidden_size=64, num_attention_heads=4, num_hidden_layers=2, kv_channels=16,
        max_position_embeddings=512, layer_norm_epsilon=1e-6, intermediate_size=128,
        no_bias=True, vocab_size=128,
    ),
    "hunyuan_v1_dense": dict(
        cls="hunyuan_v1_dense", model_type="hunyuan_v1_dense",
        vocab_size=128, hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5,
        rope_theta=10000.0, max_position_embeddings=512, attention_bias=True,
        use_qk_norm=True, tie_word_embeddings=False,
    ),
    "gpt2": dict(
        cls="gpt2", model_type="gpt2",
        n_ctx=512, n_embd=64, n_head=4, n_layer=2, n_positions=512,
        layer_norm_epsilon=1e-5, vocab_size=128,
    ),
    "gpt_bigcode": dict(
        cls="gpt_bigcode", model_type="gpt_bigcode",
        n_embd=64, n_layer=2, n_inner=256, n_head=4, n_positions=512,
        layer_norm_epsilon=1e-5, vocab_size=128, multi_query=True,
        attention_bias=True, mlp_bias=True, tie_word_embeddings=True,
    ),
    "gemma3_text": dict(
        cls="gemma3_text", model_type="gemma3_text",
        hidden_size=64, num_hidden_layers=4, intermediate_size=128,
        num_attention_heads=4, head_dim=16, rms_norm_eps=1e-6, vocab_size=128,
        num_key_value_heads=2, rope_theta=1000000.0, rope_local_base_freq=10000.0,
        query_pre_attn_scalar=256, sliding_window=512, sliding_window_pattern=2,
        max_position_embeddings=512,
    ),
}


def build(arch):
    if arch not in TINY:
        raise SystemExit(f"arch '{arch}' non in TINY (aggiungila)")
    cfg = dict(TINY[arch])
    cls_name = cfg.pop("cls")
    do_sanitize = cfg.pop("sanitize", False)
    mod = __import__(f"mlx_lm.models.{cls_name}", fromlist=["Model", "ModelArgs"])
    args = mod.ModelArgs(**cfg)
    model = mod.Model(args)
    # pesi casuali deterministici
    mx.random.seed(0)
    model.update(tree_map(lambda p: mx.random.normal(p.shape, dtype=p.dtype) * 0.05,
                          model.parameters()))
    mx.eval(model.parameters())
    return model, cfg, do_sanitize


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    arch, out = sys.argv[1], sys.argv[2]
    ids = [10, 20, 30, 40, 50]
    for i, a in enumerate(sys.argv[3:]):
        if a == "--ids" and i + 1 < len(sys.argv):
            ids = [int(x) for x in sys.argv[4].split(",")]
    os.makedirs(out, exist_ok=True)
    model, cfg, do_sanitize = build(arch)
    flat = dict(tree_flatten(model.parameters()))
    # `sanitize` (mlx_lm) fonde gli expertoni in `switch_mlp`: forma attesa dall'engine per le
    # arch che lo richiedono (mixtral/lfm2_moe). Per le altre la map resta identica.
    if do_sanitize and hasattr(model, "sanitize"):
        flat = model.sanitize(flat)
    mx.save_safetensors(os.path.join(out, "model.safetensors"), flat)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    print(f"tiny '{arch}' -> {out}  ({len(flat)} tensori)")
    lg = model(mx.array([ids]))[0, -1]
    lp = mx.log(mx.softmax(lg))
    print(f"REFERENCE ids={ids}:")
    for t in mx.argsort(lg)[::-1][:8].tolist():
        print(f"  tok {t} logprob {float(lp[t]):.5f}")


if __name__ == "__main__":
    main()
