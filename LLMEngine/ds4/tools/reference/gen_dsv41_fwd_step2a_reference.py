#!/usr/bin/env python3
"""gen_dsv41_fwd_step2a_reference.py — riferimento step 2a (mHC layer 0 + hc_pre_norm).
Costruisce il modello tiny (stessa config/seed dell'harness), calcola:
  ap, ao, ac = hc_mixes(h0, hc_attn_fn, hc_attn_scale, hc_attn_base, cfg)
  x          = hc_pre_norm(h0, pre0, attn_norm.weight, eps)
e scrive LLMEnv/LLMEngine/mlx/llm_dsv41_fwd_step2a_ref.h.
Uso: ~/ai/env-omlx/bin/python tools/gen_dsv41_fwd_step2a_reference.py
"""
import os
import numpy as np

def cvec(name, arr, per=8):
    vals = ["%.7g" % float(v) for v in np.asarray(arr, np.float32).reshape(-1)] or ["0.0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const float {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"

def main():
    import mlx.core as mx
    import mlx.nn as nn
    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.language import LanguageModel, hc_mixes, hc_pre_norm

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
        if isinstance(tree, dict): return {k: rnd(v) for k, v in tree.items()}
        if isinstance(tree, list): return [rnd(v) for v in tree]
        if isinstance(tree, mx.array) and tree.dtype in (mx.float32, mx.float16, mx.bfloat16):
            return mx.random.uniform(-0.1, 0.1, tree.shape).astype(tree.dtype)
        return tree
    model.update(rnd(model.parameters()))

    L = model.layers[0]
    c = model._config
    ids = mx.array([[3, 7]], mx.int32)
    h = model.embed(ids)
    h = mx.repeat(h[..., None, :], c.hc_mult, -2)
    pre = mx.broadcast_to((mx.arange(c.hc_mult) == 0).astype(mx.float32), h.shape[:-1])
    mx.eval(h, pre)

    ap, ao, ac = hc_mixes(h, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, c)
    x = hc_pre_norm(h, pre, L.attn_norm.weight, L.attn_norm.eps)
    mx.eval(ap, ao, ac, x)

    out = ["/* GENERATO da tools/gen_dsv41_fwd_step2a_reference.py — NON modificare a mano. */\n",
           "#ifndef GMB_DSV41_FWD_STEP2A_REF_H\n#define GMB_DSV41_FWD_STEP2A_REF_H\n",
           "#define REF_S2_L 2\n#define REF_S2_HC 4\n#define REF_S2_DIM 8\n#define REF_S2_SIZE 24\n"]
    out.append(cvec("REF_S2_H", h))
    out.append(cvec("REF_S2_PRE", pre))
    out.append(cvec("REF_S2_HCFN", L.hc_attn_fn))
    out.append(cvec("REF_S2_HCSCALE", L.hc_attn_scale))
    out.append(cvec("REF_S2_HCBASE", L.hc_attn_base))
    out.append(cvec("REF_S2_ANORM", L.attn_norm.weight))
    out.append(cvec("REF_S2_AP", ap))
    out.append(cvec("REF_S2_AO", ao))
    out.append(cvec("REF_S2_AC", ac))
    out.append(cvec("REF_S2_X", x))
    out.append("#endif /* GMB_DSV41_FWD_STEP2A_REF_H */\n")
    dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx", "llm_dsv41_fwd_step2a_ref.h")
    with open(dst, "w") as f:
        f.write("".join(out))
    print("scritto:", os.path.realpath(dst))

if __name__ == "__main__":
    main()
