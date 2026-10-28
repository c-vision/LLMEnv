#!/usr/bin/env python3
"""gen_dsv41_math_reference.py — genera i valori di riferimento per llm_dsv41_math.h.

Riferimento: MLX (env-omlx). Le formule replicano VERBATIM:
  - omlx/patches/deepseek_v41/hyper_connection.py : sinkhorn_reference
  - omlx/patches/deepseek_v41/language.py         : _hc_mix_weights, _hc_mixes, hc_pre, hc_post
  - omlx/patches/deepseek_v41/language.py         : Compressor.__call__ (pooling) + RMSNorm
  - omlx/patches/deepseek_v41/kernels.py          : packed_index_scores -> sum_h relu(dot)*w
Input deterministici via LCG (interi -> float): identici in Python e C (nessun trascendente in input).

Uso:  ~/ai/env-omlx/bin/python tools/gen_dsv41_math_reference.py
Scrive: LLMEnv/LLMEngine/mlx/llm_dsv41_math_ref.h
"""
import os
import sys
import mlx.core as mx

MASK = 0x7FFFFFFF

def lcg(seed, n, lo=-1000, hi=1000):
    out = []
    s = seed
    for _ in range(n):
        s = (s * 1103515245 + 12345) & MASK
        out.append((s % (hi - lo + 1) + lo) / 1000.0)
    return out

def sigmoid(x):
    return 1.0 / (1.0 + mx.exp(-x))

def sinkhorn_reference(comb, eps, iters):
    comb = comb / (mx.sum(comb, -2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (mx.sum(comb, -1, keepdims=True) + eps)
        comb = comb / (mx.sum(comb, -2, keepdims=True) + eps)
    return comb

def hc_mix_weights(mixes, scale, base, n, hc_eps, iters):
    pre = sigmoid(mixes[:n] * scale[0] + base[:n]) + hc_eps
    post = 2 * sigmoid(mixes[n:2 * n] * scale[1] + base[n:2 * n])
    comb = (mixes[2 * n:] * scale[2] + base[2 * n:]).reshape(n, n)
    comb = mx.softmax(comb, -1) + hc_eps
    return pre, post, sinkhorn_reference(comb, hc_eps, iters)

def hc_mixes(x, fn, scale, base, n, norm_eps, hc_eps, iters):
    flat = x.reshape(-1).astype(mx.float32)
    mixes = (flat @ fn.T) * mx.rsqrt(mx.mean(flat * flat, -1, keepdims=True) + norm_eps)
    return hc_mix_weights(mixes, scale, base, n, hc_eps, iters)

def hc_pre(x, pre):
    return mx.sum(x.astype(mx.float32) * pre[..., None], axis=-2)

def hc_post(x, residual, post, comb):
    return (post[..., None] * x[..., None, :] +
            mx.einsum("ij,id->jd", comb, residual.astype(mx.float32)))

def compressor_pool(kv, gate, r):
    # kv/gate: (L, H) con L multiplo di r; pooling su gruppi di r.
    L, H = kv.shape
    m = L // r
    pooled = mx.zeros((m, H))
    for i in range(m):
        seg_kv = kv[i * r:(i + 1) * r]          # (r,H)
        w = mx.softmax(gate[i * r:(i + 1) * r], axis=0)  # (r,H)
        pooled[i] = mx.sum(seg_kv * w, axis=0)
    return pooled

def rms_norm(x, w, eps):
    f = x.astype(mx.float32)
    return f * mx.rsqrt(mx.mean(f * f, -1, keepdims=True) + eps) * w

def index_score(q, key, weights):
    # q/key: (heads, dim); weights: (heads,) -> sum_h relu(dot_h)*w_h
    dots = mx.sum(q * key, axis=-1)             # (heads,)
    return mx.sum(mx.maximum(dots, 0) * weights)

def cvec(name, arr, per=8, fmt="%.7g"):
    def flat(a):
        for v in a:
            if isinstance(v, (list, tuple)):
                yield from flat(v)
            else:
                yield v
    vals = [fmt % float(v) for v in flat(arr)]
    lines = []
    for i in range(0, len(vals), per):
        lines.append("    " + ", ".join(vals[i:i + per]) + ",")
    return f"static const float {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"

def main():
    eps = 1e-6
    iters = 20
    n = 4

    out = []
    out.append("/* GENERATO da tools/gen_dsv41_math_reference.py — NON modificare a mano. */\n")
    out.append("#ifndef GMB_DSV41_MATH_REF_H\n#define GMB_DSV41_MATH_REF_H\n")

    # --- sinkhorn (n=4) ---
    comb0 = mx.array(lcg(1, 16, 0, 2000), mx.float32).reshape(4, 4)  # 0..2 (positivi)
    sink = sinkhorn_reference(comb0, eps, iters)
    out.append(cvec("REF_SINK_COMB", comb0.tolist()))
    out.append(cvec("REF_SINK_OUT", sink.tolist()))

    # --- hc_mix_weights (n=4, mixes len = (2+n)*n = 24) ---
    mixes = mx.array(lcg(2, (2 + n) * n), mx.float32)
    scale = [1.1, 0.7, 0.9]
    base = mx.array(lcg(3, (2 + n) * n), mx.float32) * 0.1
    pre, post, comb = hc_mix_weights(mixes, scale, base, n, eps, iters)
    out.append(cvec("REF_MIX_MIXES", mixes.tolist()))
    out.append(cvec("REF_MIX_SCALE", scale))
    out.append(cvec("REF_MIX_BASE", base.tolist()))
    out.append(cvec("REF_MIX_PRE", pre.tolist()))
    out.append(cvec("REF_MIX_POST", post.tolist()))
    out.append(cvec("REF_MIX_COMB", comb.tolist()))

    # --- hc_mixes (n=4, D=6) ---
    D = 6
    x = mx.array(lcg(4, n * D), mx.float32)
    fn = mx.array(lcg(5, (2 + n) * n * (n * D)), mx.float32).reshape((2 + n) * n, n * D)  # (size=24, n*D=24)
    scale2 = [0.8, 1.2, 1.0]
    base2 = mx.array(lcg(6, (2 + n) * n), mx.float32) * 0.05
    pre2, post2, comb2 = hc_mixes(x, fn, scale2, base2, n, 1e-20, eps, iters)
    out.append(cvec("REF_HCM_X", x.tolist()))
    out.append(cvec("REF_HCM_FN", fn.reshape(-1).tolist()))
    out.append(cvec("REF_HCM_SCALE", scale2))
    out.append(cvec("REF_HCM_BASE", base2.tolist()))
    out.append(cvec("REF_HCM_PRE", pre2.tolist()))
    out.append(cvec("REF_HCM_POST", post2.tolist()))
    out.append(cvec("REF_HCM_COMB", comb2.tolist()))

    # --- hc_pre ---
    xp = mx.array(lcg(7, n * D), mx.float32).reshape(n, D)
    pr = mx.array(lcg(8, n, 0, 2000), mx.float32)  # 0..2
    out.append(cvec("REF_PRE_X", xp.tolist()))
    out.append(cvec("REF_PRE_P", pr.tolist()))
    out.append(cvec("REF_PRE_OUT", hc_pre(xp, pr).tolist()))

    # --- hc_post ---
    xq = mx.array(lcg(9, D), mx.float32)
    res = mx.array(lcg(10, n * D), mx.float32).reshape(n, D)
    postq = mx.array(lcg(11, n, 0, 2000), mx.float32)
    combq = (mx.array(lcg(12, n * n, 0, 1000), mx.float32) / 1000.0).reshape(n, n)
    out.append(cvec("REF_POST_X", xq.tolist()))
    out.append(cvec("REF_POST_RES", res.tolist()))
    out.append(cvec("REF_POST_P", postq.tolist()))
    out.append(cvec("REF_POST_COMB", combq.tolist()))
    out.append(cvec("REF_POST_OUT", hc_post(xq, res, postq, combq).tolist()))

    # --- compressor pooling (r=2, H=3) ---
    r, H = 2, 3
    kv = mx.array(lcg(13, 4 * H), mx.float32).reshape(4, H)
    gate = mx.array(lcg(14, 4 * H), mx.float32).reshape(4, H)
    out.append(cvec("REF_CP_KV", kv.tolist()))
    out.append(cvec("REF_CP_GATE", gate.tolist()))
    out.append(cvec("REF_CP_OUT", compressor_pool(kv, gate, r).tolist()))

    # --- index score (heads=2, dim=3) ---
    q = mx.array(lcg(15, 2 * 3), mx.float32)
    k = mx.array(lcg(16, 2 * 3), mx.float32)
    w = mx.array(lcg(17, 2, 0, 1000), mx.float32) / 1000.0
    out.append(cvec("REF_IDX_Q", q.tolist()))
    out.append(cvec("REF_IDX_K", k.tolist()))
    out.append(cvec("REF_IDX_W", w.tolist()))
    out.append(cvec("REF_IDX_OUT", [index_score(q, k, w).item()]))

    out.append("#endif /* GMB_DSV41_MATH_REF_H */\n")

    dst = os.path.join(os.path.dirname(__file__), "..",
                       "native", "gmb_host_ai_mlx", "llm_dsv41_math_ref.h")
    with open(dst, "w") as f:
        f.write("".join(out))
    print("scritto:", os.path.realpath(dst))

if __name__ == "__main__":
    main()
