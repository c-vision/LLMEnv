#!/usr/bin/env python3
"""gen_synthetic_mixtral.py — genera un checkpoint Mixtral SINTETICO minimale,
caricabile da gmb_ai_mlx (model_type="mixtral"), per misurare il percorso
`stack_mixtral_moe_experts` + `mixtral_ffn_block` SENZA il modello da 74 GB.

Nato 2026-09-17 dalla diagnosi del bug MLX-Mixtral-8x22B (prefill 5 tok/s vs 272 di
mlx_lm): tutti gli altri MoE battono mlx_lm nel prefill, solo Mixtral no => il gap
è nel percorso Mixtral-specifico. Questo generatore lo rende riproducibile a scala
di millisecondi, load-free e deterministico (seed fisso).

Fedeltà al checkpoint reale (mlx-community/Mixtral-8x22B-4bit, verificato):
  - naming: model.layers.N.block_sparse_moe.experts.E.w1/w2/w3.{weight,scales,biases}
  - embed_tokens NON quantizzato (solo .weight, F16); lm_head quantizzato
  - block_sparse_moe.gate.weight NON quantizzato (F16), norms F16
  - quantizzazione affine group 64 / 4 bit, scales/biases F16
Uso:
  python3 tools/gen_synthetic_mixtral.py <outdir> [--layers 2 --hidden 512
      --heads 4 --kv-heads 2 --inter 1024 --experts 8 --top-k 2 --vocab 1024]
"""
import argparse
import json
import os

import mlx.core as mx


def quant(shape, group, bits, scale):
    w = (mx.random.normal(shape) * scale).astype(mx.float16)
    wq, sc, bi = mx.quantize(w, group_size=group, bits=bits)
    return wq, sc, bi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=None,
                    help="default hidden/heads")
    ap.add_argument("--inter", type=int, default=1024)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    mx.random.seed(a.seed)
    hd = a.head_dim or (a.hidden // a.heads)
    assert a.hidden % a.heads == 0 and a.hidden % a.group == 0
    assert a.inter % a.group == 0
    W = {}

    # embedding (NON quantizzato, come il reale) + lm_head (quantizzato)
    W["model.embed_tokens.weight"] = (
        mx.random.normal((a.vocab, a.hidden)) * 0.02).astype(mx.float16)
    W["lm_head.weight"], W["lm_head.scales"], W["lm_head.biases"] = quant(
        (a.vocab, a.hidden), a.group, a.bits, 0.02)

    for l in range(a.layers):
        p = f"model.layers.{l}."
        W[p + "input_layernorm.weight"] = mx.ones((a.hidden,), mx.float16)
        W[p + "post_attention_layernorm.weight"] = mx.ones((a.hidden,), mx.float16)
        W[p + "self_attn.q_proj.weight"], W[p + "self_attn.q_proj.scales"], W[p + "self_attn.q_proj.biases"] = quant(
            (a.heads * hd, a.hidden), a.group, a.bits, 0.02)
        W[p + "self_attn.k_proj.weight"], W[p + "self_attn.k_proj.scales"], W[p + "self_attn.k_proj.biases"] = quant(
            (a.kv_heads * hd, a.hidden), a.group, a.bits, 0.02)
        W[p + "self_attn.v_proj.weight"], W[p + "self_attn.v_proj.scales"], W[p + "self_attn.v_proj.biases"] = quant(
            (a.kv_heads * hd, a.hidden), a.group, a.bits, 0.02)
        W[p + "self_attn.o_proj.weight"], W[p + "self_attn.o_proj.scales"], W[p + "self_attn.o_proj.biases"] = quant(
            (a.hidden, a.heads * hd), a.group, a.bits, 0.02)
        # router NON quantizzato (come il reale)
        W[p + "block_sparse_moe.gate.weight"] = (
            mx.random.normal((a.experts, a.hidden)) * 0.02).astype(mx.float16)
        for e in range(a.experts):
            ep = p + f"block_sparse_moe.experts.{e}."
            for name, shape in (("w1", (a.inter, a.hidden)),      # gate_proj
                                ("w3", (a.inter, a.hidden)),      # up_proj
                                ("w2", (a.hidden, a.inter))):     # down_proj
                W[ep + name + ".weight"], W[ep + name + ".scales"], W[ep + name + ".biases"] = quant(
                    shape, a.group, a.bits, 0.02)

    W["model.norm.weight"] = mx.ones((a.hidden,), mx.float16)

    os.makedirs(a.outdir, exist_ok=True)
    mx.save_safetensors(os.path.join(a.outdir, "model.safetensors"), W)

    cfg = {
        "architectures": ["MixtralForCausalLM"],
        "model_type": "mixtral",
        "hidden_size": a.hidden,
        "intermediate_size": a.inter,
        "num_hidden_layers": a.layers,
        "num_attention_heads": a.heads,
        "num_key_value_heads": a.kv_heads,
        "head_dim": hd,
        "num_local_experts": a.experts,
        "num_experts_per_tok": a.top_k,
        "vocab_size": a.vocab,
        "rms_norm_eps": 1e-5,
        "rope_theta": 1000000,
        "max_position_embeddings": 65536,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "quantization": {"group_size": a.group, "bits": a.bits},
    }
    with open(os.path.join(a.outdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    nbytes = sum(v.nbytes for v in W.values())
    print(f"OK {a.outdir}: layers={a.layers} hidden={a.hidden} heads={a.heads}"
          f"/{a.kv_heads} hd={hd} inter={a.inter} experts={a.experts} top_k={a.top_k}"
          f" vocab={a.vocab} g{a.group}/{a.bits}bit tensori={len(W)} bytes={nbytes/1e6:.2f}MB")


if __name__ == "__main__":
    main()
