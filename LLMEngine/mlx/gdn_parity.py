#!/usr/bin/env python3
"""GDN linear-attention forward — verified parity reference (L2 foundation).

Reconstructs the Qwen3.8-27B `qwen3_5` GatedDeltaNet forward and checks it
matches mlx-lm's own `linear_attn` output numerically (expect diff ~1e-6).

Usage:  python tools/mlx_reference/gdn_parity.py [model_dir]
Default model_dir: ~/ai/Models/Qwen3.8-27B-8bit

Run in the rapid-mlx env (has mlx + mlx_lm + safetensors).
This is the reference to port into native/llmengine_mlx (LLMEnv L2).
"""
import sys
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gated_delta import gated_delta_ops, compute_g


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "/Users/gaetano/ai/Models/Qwen3.8-27B-8bit"
    import mlx_lm
    model, tok = mlx_lm.load(model_dir, lazy=True)
    lm = model.language_model
    layer = lm.model.layers[1]          # layer 1 is linear (interval 4)
    la = layer.linear_attn

    B, S = 1, 4
    x = (mx.random.normal((1, S, 5120)).astype(mx.float32) * 0.05)
    h = layer.input_layernorm(x)

    # reference from mlx-lm
    caches = lm.make_cache()
    ref = la(h, cache=caches[1])
    mx.eval(ref)

    # --- GDN forward (must match GatedDeltaNet.__call__ exactly) ---
    qkv = la.in_proj_qkv(h)                                   # (B,S,conv_dim)
    z = la.in_proj_z(h).reshape(B, S, la.num_v_heads, la.head_v_dim)
    b = la.in_proj_b(h)
    a = la.in_proj_a(h)
    conv_input = mx.concatenate(
        [mx.zeros((B, la.conv_kernel_size - 1, la.conv_dim), dtype=h.dtype), qkv], axis=1
    )
    conv_out = nn.silu(la.conv1d(conv_input))
    q, k, v = [
        t.reshape(B, S, hd, dd)
        for t, hd, dd in zip(
            mx.split(conv_out, [la.key_dim, 2 * la.key_dim], -1),
            [la.num_k_heads, la.num_k_heads, la.num_v_heads],
            [la.head_k_dim, la.head_k_dim, la.head_v_dim],
        )
    ]
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    beta = mx.sigmoid(b)
    g = compute_g(la.A_log, a, la.dt_bias)
    state = mx.zeros((B, la.num_v_heads, la.head_v_dim, la.head_k_dim), dtype=mx.float32)
    out, _ = gated_delta_ops(q, k, v, g, beta, state, None)
    out = la.norm(out, z)
    fin = la.out_proj(out.reshape(B, S, -1))
    mx.eval(fin)

    diff = float(mx.max(mx.abs(fin - ref)))
    print("mlx-lm linear_attn:", tuple(ref.shape))
    print("reconstructed    :", tuple(fin.shape))
    print(f"MAX DIFF = {diff:.2e}")
    ok = diff < 1e-4
    print("PARITY", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
