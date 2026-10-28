#!/usr/bin/env python3
"""gen_dsv41_dspark_attn_ref.py — riferimento tiny di DSparkContextCache + DSparkAttention.
Uso: ~/ai/env-omlx/bin/python tools/gen_dsv41_dspark_attn_ref.py
Scrive: LLMEnv/LLMEngine/mlx/llm_dsv41_dspark_attn_ref.h
"""
import os
import numpy as np


def cvec(name, arr, per=8):
    vals = ["%.7g" % float(v) for v in np.asarray(arr, np.float32).reshape(-1)] or ["0.0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const float {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"


def main():
    import mlx.core as mx
    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.dspark import DSparkAttention
    from omlx.patches.mlx_lm_mtp.deepseek_v4_dspark import DSparkContextCache

    cfg = ModelConfig(dim=8, n_heads=2, head_dim=32, rope_head_dim=4, o_groups=2, o_lora_rank=8,
                      q_lora_rank=8, vocab_size=16, norm_eps=1e-20, rope_theta=10000.0,
                      original_seq_len=0, rope_factor=1.0)
    mx.random.seed(1234)
    attn = DSparkAttention(cfg, 0)

    def rnd(t):
        if isinstance(t, dict):
            return {k: rnd(v) for k, v in t.items()}
        if isinstance(t, list):
            return [rnd(v) for v in t]
        if isinstance(t, mx.array) and t.dtype in (mx.float32, mx.float16, mx.bfloat16):
            return mx.random.uniform(-0.2, 0.2, t.shape).astype(t.dtype)
        return t

    attn.update(rnd(attn.parameters()))
    D = cfg.head_dim
    cache = DSparkContextCache(max_size=4)
    ctx = mx.random.uniform(-0.2, 0.2, (1, 1, 3, D))
    cache.append(ctx)
    x = mx.random.uniform(-0.2, 0.2, (1, 2, cfg.dim))
    from omlx.patches.deepseek_v41.language import rope
    from omlx.patches.deepseek_v41.quantization import quantize_activation
    positions = mx.arange(cache.offset, cache.offset + x.shape[1])
    qpre = attn.wq_b(attn.q_norm(attn.wq_a(x))).reshape(1, x.shape[1], cfg.n_heads, cfg.head_dim)
    q = rope(qpre, positions, cfg, False)
    dk = quantize_activation(rope(attn.kv_norm(attn.wkv(x)), positions, cfg, False))
    kv = mx.concatenate([cache.keys[:, 0], dk], axis=1)
    out = attn(x, cache)
    mx.eval(out, cache.keys, q, dk, kv, qpre)
    cache2 = DSparkContextCache(max_size=4)
    ck = []
    for _i in range(3):
        k = mx.random.uniform(-0.2, 0.2, (1, 1, 2, D))
        cache2.append(k)
        ck.append(k)
        mx.eval(cache2.keys)
    mx.eval(cache2.keys)

    out_h = ["/* GENERATO da tools/gen_dsv41_dspark_attn_ref.py — NON modificare a mano. */\n",
             "#ifndef GMB_DSV41_DSPARK_ATTN_REF_H\n#define GMB_DSV41_DSPARK_ATTN_REF_H\n",
             f"#define REF_DA_DIM {cfg.dim}\n#define REF_DA_H {cfg.n_heads}\n#define REF_DA_D {cfg.head_dim}\n"
             f"#define REF_DA_DR {cfg.rope_head_dim}\n#define REF_DA_QD {cfg.q_lora_rank}\n"
             f"#define REF_DA_OG {cfg.o_groups}\n#define REF_DA_OL {cfg.o_lora_rank}\n"
             f"#define REF_DA_CTX_N {int(ctx.shape[2])}\n#define REF_DA_L {int(x.shape[1])}\n"
             f"#define REF_DA_OFF {int(cache.offset)}\n#define REF_DA_CAP 4\n"]
    out_h.append(cvec("REF_DA_CTX", ctx))
    out_h.append(cvec("REF_DA_X", x))
    out_h.append(cvec("REF_DA_WQA", attn.wq_a.weight))
    out_h.append(cvec("REF_DA_QN", attn.q_norm.weight))
    out_h.append(cvec("REF_DA_WQB", attn.wq_b.weight))
    out_h.append(cvec("REF_DA_WKV", attn.wkv.weight))
    out_h.append(cvec("REF_DA_KVN", attn.kv_norm.weight))
    out_h.append(cvec("REF_DA_WOA", attn.wo_a.weight))
    out_h.append(cvec("REF_DA_WOB", attn.wo_b.weight))
    out_h.append(cvec("REF_DA_SINK", attn.attn_sink))
    out_h.append(cvec("REF_DA_QPRE", qpre))
    out_h.append(cvec("REF_DA_Q", q))
    out_h.append(cvec("REF_DA_DK", dk))
    out_h.append(cvec("REF_DA_KV", kv))
    out_h.append(cvec("REF_DA_OUT", out))
    out_h.append(cvec("REF_DA_CK0", ck[0]))
    out_h.append(cvec("REF_DA_CK1", ck[1]))
    out_h.append(cvec("REF_DA_CK2", ck[2]))
    out_h.append(cvec("REF_DA_CKEYS", cache2.keys))
    out_h.append(f"#define REF_DA_CKOFF {cache2.offset}\n")
    out_h.append("#endif /* GMB_DSV41_DSPARK_ATTN_REF_H */\n")
    dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx", "llm_dsv41_dspark_attn_ref.h")
    open(dst, "w").write("".join(out_h))
    print("scritto:", os.path.realpath(dst), "| out", out.shape, "| offset", cache.offset)


if __name__ == "__main__":
    main()
