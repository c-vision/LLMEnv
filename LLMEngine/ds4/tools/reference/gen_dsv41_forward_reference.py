#!/usr/bin/env python3
"""gen_dsv41_forward_reference.py — riferimento del FORWARD DeepSeek V4.1 (senza scaricare il modello).

Istanzia `omlx.patches.deepseek_v41.language.LanguageModel` con una config tiny, inizializza i pesi
random (seed fisso) e salva config + TUTTI i pesi + input ids + logits attesi in un `.npz`.
E' il riferimento con cui verificare il forward C (`llm_dsv41_forward.h`): pesi piccoli, nessun
download, nessun device obbligatorio (MLX CPU/GPU basta).

Uso:  ~/ai/env-omlx/bin/python tools/gen_dsv41_forward_reference.py [out_dir]
Default out_dir: /tmp/dsv41-fwd-ref
"""
import os
import sys
import json
import numpy as np

def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dsv41-fwd-ref"
    os.makedirs(out_dir, exist_ok=True)

    import mlx.core as mx
    import mlx.nn as nn
    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.language import LanguageModel

    cfg = ModelConfig(
        dim=8, n_layers=2, n_heads=2, o_groups=2, head_dim=32, rope_head_dim=4,
        index_head_dim=32, q_lora_rank=8, o_lora_rank=8, moe_inter_dim=8,
        n_routed_experts=4, n_activated_experts=2, n_shared_experts=1,
        window_size=4, compress_ratios=(0, 0), kv_source_layers=(), index_source_layers=(),
        engram_layer_ids=(), engram_num_embeddings=(), vocab_size=64, hc_mult=4,
        n_mtp_layers=1, preserve_mtp=False, max_seq_len=16,
    )
    cfg.validate()
    model = LanguageModel(cfg)

    mx.random.seed(1234)

    def rnd(tree):
        if isinstance(tree, dict):
            return {k: rnd(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [rnd(v) for v in tree]
        if isinstance(tree, mx.array) and tree.dtype in (mx.float32, mx.float16, mx.bfloat16):
            return mx.random.uniform(-0.1, 0.1, tree.shape).astype(tree.dtype)
        return tree

    model.update(rnd(model.parameters()))

    ids = np.array([[3, 7]], np.int32)
    logits = model(mx.array(ids))
    mx.eval(logits)

    flat = nn.utils.tree_flatten(model.parameters())
    out = {"input_ids": ids, "logits": np.array(logits, dtype=np.float32),
           "_config": np.frombuffer(json.dumps({
               "dim": cfg.dim, "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
               "head_dim": cfg.head_dim, "rope_head_dim": cfg.rope_head_dim,
               "o_groups": cfg.o_groups, "o_lora_rank": cfg.o_lora_rank,
               "q_lora_rank": cfg.q_lora_rank, "moe_inter_dim": cfg.moe_inter_dim,
               "n_routed_experts": cfg.n_routed_experts,
               "n_activated_experts": cfg.n_activated_experts,
               "window_size": cfg.window_size, "compress_ratios": list(cfg.compress_ratios),
               "hc_mult": cfg.hc_mult, "vocab_size": cfg.vocab_size,
               "norm_eps": cfg.norm_eps, "rope_theta": cfg.rope_theta,
               "score_func": cfg.score_func, "gate_temp": cfg.gate_temp,
               "route_scale": cfg.route_scale, "swiglu_limit": cfg.swiglu_limit,
           }).encode(), dtype=np.uint8)}
    for name, arr in flat:
        out["w:" + name] = np.array(arr, dtype=np.float32)
    dst = os.path.join(out_dir, "dsv41_fwd_ref.npz")
    np.savez(dst, **out)
    print("config:", json.dumps({k: v for k, v in {
        "dim": cfg.dim, "n_layers": cfg.n_layers, "head_dim": cfg.head_dim}.items()}))
    print("tensori:", len(flat), "| logits shape:", tuple(np.array(logits).shape))
    print("logits[0,0,:4]:", np.array(logits)[0, 0, :4].tolist())
    print("checksum logits:", float(np.abs(np.array(logits)).sum()))
    print("scritto:", dst)

if __name__ == "__main__":
    main()
