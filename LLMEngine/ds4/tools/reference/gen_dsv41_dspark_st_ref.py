#!/usr/bin/env python3
"""gen_dsv41_dspark_st_ref.py — fixture per il loader DSpark (`mtp.*`) e riferimento forward_spec.
Scrive:
  /tmp/dsv41_st/mtp.safetensors — checkpoint tiny (embed/head + backbone layer + mtp.0) in F32.
  LLMEnv/LLMEngine/mlx/llm_dsv41_dspark_st_ref.h — config + input + proposte/confidence attese.
Uso: ~/ai/env-omlx/bin/python tools/gen_dsv41_dspark_st_ref.py
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


def main():
    import mlx.core as mx
    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.language import LanguageModel
    from omlx.patches.deepseek_v41.dspark import forward_spec

    outdir = "/tmp/dsv41_st"
    os.makedirs(outdir, exist_ok=True)
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

    arrs = {}
    def add(name, a):
        arrs[name] = mx.array(a).astype(mx.float32)

    add("embed.weight", model.embed.weight)
    add("head.weight", model.head.weight)
    add("norm.weight", model.norm.weight)

    def dump_block(prefix, L, A, E, S):
        add(prefix + "hc_attn_fn", L.hc_attn_fn); add(prefix + "hc_attn_scale", L.hc_attn_scale)
        add(prefix + "hc_attn_base", L.hc_attn_base)
        add(prefix + "hc_ffn_fn", L.hc_ffn_fn); add(prefix + "hc_ffn_scale", L.hc_ffn_scale)
        add(prefix + "hc_ffn_base", L.hc_ffn_base)
        add(prefix + "attn_norm.weight", L.attn_norm.weight)
        add(prefix + "ffn_norm.weight", L.ffn_norm.weight)
        add(prefix + "attn.wq_a.weight", A.wq_a.weight); add(prefix + "attn.q_norm.weight", A.q_norm.weight)
        add(prefix + "attn.wq_b.weight", A.wq_b.weight); add(prefix + "attn.wkv.weight", A.wkv.weight)
        add(prefix + "attn.kv_norm.weight", A.kv_norm.weight); add(prefix + "attn.wo_a.weight", A.wo_a.weight)
        add(prefix + "attn.wo_b.weight", A.wo_b.weight); add(prefix + "attn.attn_sink", A.attn_sink)
        add(prefix + "ffn.gate.weight", L.ffn.gate.weight); add(prefix + "ffn.gate.bias", L.ffn.gate.bias)
        add(prefix + "ffn.shared_experts.w1.weight", S.w1.weight)
        add(prefix + "ffn.shared_experts.w2.weight", S.w2.weight)
        add(prefix + "ffn.shared_experts.w3.weight", S.w3.weight)
        for e in range(cfg.dspark_n_routed_experts):
            add(prefix + f"ffn.experts.{e}.w1.weight", E.w1.weight[e])
            add(prefix + f"ffn.experts.{e}.w2.weight", E.w2.weight[e])
            add(prefix + f"ffn.experts.{e}.w3.weight", E.w3.weight[e])

    # backbone layer 0 (serve solo a completare l'index; il loader DSpark ignora `layers.*`)
    L0 = model.layers[0]
    dump_block("layers.0.", L0, L0.attn, L0.ffn.experts, L0.ffn.shared_experts)
    # stage MTP 0
    M = model.mtp[0]
    dump_block("mtp.0.", M, M.attn, M.ffn.experts, M.ffn.shared_experts)
    add("mtp.0.main_proj.weight", M.main_proj.weight)
    add("mtp.0.main_norm.weight", M.main_norm.weight)
    add("mtp.0.norm.weight", M.norm.weight)
    add("mtp.0.markov_head.embed.weight", M.markov_head.embed.weight)
    add("mtp.0.markov_head.head.weight", M.markov_head.head.weight)
    add("mtp.0.confidence_head.proj.weight", M.confidence_head.proj.weight)

    mx.eval(*arrs.values())
    st_path = os.path.join(outdir, "mtp.safetensors")
    mx.save_safetensors(st_path, arrs)
    print("scritto:", st_path, "|", len(arrs), "tensori")

    # riferimento forward_spec (stessa sequenza del test dspark_spec)
    c = model._config
    cache = model.make_dspark_cache()
    mh_p = mx.random.uniform(-0.2, 0.2, (1, 2, c.dim * len(c.dspark_target_layer_ids)))
    r0 = forward_spec(model, mx.array([[3]], mx.int32), mh_p, cache)
    assert r0 is None
    mh_s = mx.random.uniform(-0.2, 0.2, (1, 1, c.dim * len(c.dspark_target_layer_ids)))
    prop, bias, conf = forward_spec(model, mx.array([[5]], mx.int32), mh_s, cache)
    mx.eval(prop, bias, conf)

    D = c.head_dim
    out = ["/* GENERATO da tools/gen_dsv41_dspark_st_ref.py — NON modificare a mano. */\n",
           "#ifndef GMB_DSV41_DSPARK_ST_REF_H\n#define GMB_DSV41_DSPARK_ST_REF_H\n",
           f"#define DSR_DIM {c.dim}\n#define DSR_HC {c.hc_mult}\n#define DSR_H {c.n_heads}\n"
           f"#define DSR_D {c.head_dim}\n#define DSR_DROPE {c.rope_head_dim}\n"
           f"#define DSR_QD {c.q_lora_rank}\n#define DSR_OG {c.o_groups}\n#define DSR_OL {c.o_lora_rank}\n"
           f"#define DSR_DI {c.moe_inter_dim}\n#define DSR_E {c.dspark_n_routed_experts}\n"
           f"#define DSR_ACT {c.dspark_n_activated_experts}\n#define DSR_VOCAB {c.vocab_size}\n"
           f"#define DSR_RANK {c.dspark_markov_rank}\n#define DSR_BLOCK {c.dspark_block_size}\n"
           f"#define DSR_NOISE {c.dspark_noise_token_id}\n#define DSR_WIN {c.window_size}\n"
           f"#define DSR_M {c.dim * len(c.dspark_target_layer_ids)}\n#define DSR_NT {len(c.dspark_target_layer_ids)}\n"
           f"#define DSR_P {int(mh_p.shape[1])}\n#define DSR_S {int(mh_s.shape[1])}\n"]
    out.append(cint("DSR_IDS_P", mx.array([[3]], mx.int32)))
    out.append(cvec("DSR_MHP", mh_p))
    out.append(cint("DSR_IDS_S", mx.array([[5]], mx.int32)))
    out.append(cvec("DSR_MHS", mh_s))
    out.append(cint("DSR_PROP", prop))
    out.append(cvec("DSR_BIAS", bias))
    out.append(cvec("DSR_CONF", conf))
    out.append("#endif /* GMB_DSV41_DSPARK_ST_REF_H */\n")
    dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx", "llm_dsv41_dspark_st_ref.h")
    open(dst, "w").write("".join(out))
    print("scritto:", os.path.realpath(dst), "| prop", np.asarray(prop).reshape(-1).tolist())


if __name__ == "__main__":
    main()
