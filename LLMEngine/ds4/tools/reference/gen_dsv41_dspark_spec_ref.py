#!/usr/bin/env python3
"""gen_dsv41_dspark_spec_ref.py — riferimento tiny di DSparkBlock/proposal_forward/forward_spec.
Uso: ~/ai/env-omlx/bin/python tools/gen_dsv41_dspark_spec_ref.py
Scrive: LLMEnv/LLMEngine/mlx/llm_dsv41_dspark_spec_ref.h
"""
import os
import numpy as np


def cvec(name, arr, per=8):
    vals = ["%.7g" % float(v) for v in np.asarray(arr, np.float32).reshape(-1)] or ["0.0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const float {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"


def cint(name, arr):
    vals = ["%d" % int(v) for v in np.asarray(arr, np.int32).reshape(-1)]
    return f"static const int {name}[{len(vals)}] = {{ " + ", ".join(vals) + " };\n"


def cshape(name, arr):
    sh = np.asarray(arr).shape
    d = ", ".join(str(int(x)) for x in sh) or "0"
    return f"#define {name}_ND {len(sh)}\n#define {name}_SH {{ {d} }}\n"


def cdump(out, name, arr):
    out.append(cvec(name, arr))
    out.append(cshape(name, arr))


def main():
    import mlx.core as mx
    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.language import LanguageModel
    from omlx.patches.deepseek_v41.dspark import forward_spec

    cfg = ModelConfig(
        dim=8, n_layers=1, n_heads=2, o_groups=2, head_dim=32, rope_head_dim=4,
        index_head_dim=32, q_lora_rank=8, o_lora_rank=8, moe_inter_dim=8,
        n_routed_experts=4, n_activated_experts=2, n_shared_experts=1,
        window_size=4, compress_ratios=(0, 0), kv_source_layers=(), index_source_layers=(),
        engram_layer_ids=(), engram_num_embeddings=(), vocab_size=64, hc_mult=4,
        n_mtp_layers=1, preserve_mtp=True, max_seq_len=16,
        dspark_block_size=3, dspark_noise_token_id=9, dspark_target_layer_ids=(0,),
        dspark_markov_rank=4, dspark_n_routed_experts=4, dspark_n_activated_experts=2,
    )
    cfg.validate()
    model = LanguageModel(cfg)
    mx.random.seed(2024)

    def rnd(t):
        if isinstance(t, dict): return {k: rnd(v) for k, v in t.items()}
        if isinstance(t, list): return [rnd(v) for v in t]
        if isinstance(t, mx.array) and t.dtype in (mx.float32, mx.float16, mx.bfloat16):
            return mx.random.uniform(-0.2, 0.2, t.shape).astype(t.dtype)
        return t
    model.update(rnd(model.parameters()))

    c = model._config
    cache = model.make_dspark_cache()
    mh_p = mx.random.uniform(-0.2, 0.2, (1, 2, c.dim * len(c.dspark_target_layer_ids)))
    ids_p = mx.array([[3]], mx.int32)
    r0 = forward_spec(model, ids_p, mh_p, cache)
    assert r0 is None, "priming deve ritornare None"
    mh_s = mx.random.uniform(-0.2, 0.2, (1, 1, c.dim * len(c.dspark_target_layer_ids)))
    ids_s = mx.array([[5]], mx.int32)
    prop, bias, conf = forward_spec(model, ids_s, mh_s, cache)
    mx.eval(prop, bias, conf, cache[0].keys)

    M = c.dim * len(c.dspark_target_layer_ids)
    out = ["/* GENERATO da tools/gen_dsv41_dspark_spec_ref.py — NON modificare a mano. */\n",
           "#ifndef GMB_DSV41_DSPARK_SPEC_REF_H\n#define GMB_DSV41_DSPARK_SPEC_REF_H\n",
           f"#define REF_DS_DIM {c.dim}\n#define REF_DS_HC {c.hc_mult}\n#define REF_DS_H {c.n_heads}\n"
           f"#define REF_DS_D {c.head_dim}\n#define REF_DS_DROPE {c.rope_head_dim}\n"
           f"#define REF_DS_QD {c.q_lora_rank}\n#define REF_DS_OG {c.o_groups}\n#define REF_DS_OL {c.o_lora_rank}\n"
           f"#define REF_DS_DI {c.moe_inter_dim}\n#define REF_DS_E {c.dspark_n_routed_experts}\n"
           f"#define REF_DS_ACT {c.dspark_n_activated_experts}\n#define REF_DS_VOCAB {c.vocab_size}\n"
           f"#define REF_DS_RANK {c.dspark_markov_rank}\n#define REF_DS_BLOCK {c.dspark_block_size}\n"
           f"#define REF_DS_NOISE {c.dspark_noise_token_id}\n#define REF_DS_WIN {c.window_size}\n"
           f"#define REF_DS_M {M}\n#define REF_DS_NL {c.n_layers}\n#define REF_DS_NT {len(c.dspark_target_layer_ids)}\n"
           f"#define REF_DS_P {int(mh_p.shape[1])}\n"
           f"#define REF_DS_S {int(mh_s.shape[1])}\n"
           f"#define REF_DS_CKOFF {int(cache[0].offset)}\n"]
    cdump(out, "REF_DS_EMBED", model.embed.weight)
    cdump(out, "REF_DS_HEAD", model.head.weight)
    out.append(cint("REF_DS_IDSP", ids_p))
    cdump(out, "REF_DS_MHP", mh_p)
    out.append(cint("REF_DS_IDSS", ids_s))
    cdump(out, "REF_DS_MHS", mh_s)

    L = model.mtp[0]; A = L.attn; E = L.ffn.experts; S = L.ffn.shared_experts
    for nm, v in [("HCAFN", L.hc_attn_fn), ("HCAS", L.hc_attn_scale), ("HCAB", L.hc_attn_base),
                  ("HCFFN", L.hc_ffn_fn), ("HCFFS", L.hc_ffn_scale), ("HCFFB", L.hc_ffn_base),
                  ("ANORM", L.attn_norm.weight), ("FNORM", L.ffn_norm.weight),
                  ("WQA", A.wq_a.weight), ("QN", A.q_norm.weight), ("WQB", A.wq_b.weight),
                  ("WKV", A.wkv.weight), ("KVN", A.kv_norm.weight), ("WOA", A.wo_a.weight),
                  ("WOB", A.wo_b.weight), ("SINK", A.attn_sink),
                  ("GW", L.ffn.gate.weight), ("GB", L.ffn.gate.bias),
                  ("EW1", E.w1.weight), ("EW3", E.w3.weight), ("EW2", E.w2.weight),
                  ("SW1", S.w1.weight), ("SW3", S.w3.weight), ("SW2", S.w2.weight),
                  ("MAINPROJ", L.main_proj.weight), ("MAINNORM", L.main_norm.weight),
                  ("OUTNORM", L.norm.weight), ("MKEMB", L.markov_head.embed.weight),
                  ("MKHEAD", L.markov_head.head.weight), ("CONFPROJ", L.confidence_head.proj.weight)]:
        cdump(out, "REF_DS_" + nm, v)

    out.append(cint("REF_DS_PROP", prop))
    cdump(out, "REF_DS_BIAS", bias)
    cdump(out, "REF_DS_CONF", conf)
    cdump(out, "REF_DS_CKEYS", cache[0].keys)
    out.append("#endif /* GMB_DSV41_DSPARK_SPEC_REF_H */\n")
    dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx", "llm_dsv41_dspark_spec_ref.h")
    open(dst, "w").write("".join(out))
    print("scritto:", os.path.realpath(dst))
    print("prop", np.asarray(prop).reshape(-1).tolist(), "| conf", np.asarray(conf).reshape(-1).tolist(),
          "| ckoff", int(cache[0].offset), "| ckeys", cache[0].keys.shape)


if __name__ == "__main__":
    main()
