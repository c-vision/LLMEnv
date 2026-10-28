#!/usr/bin/env python3
"""gen_dsv41_fwd_step2c_reference.py — riferimento step 2c (MoE layer 0 + hc_post -> Block 0 out).
Ricostruisce il forward del Block 0 (mHC + attn + MoE) col modello tiny e dumpa pesi + intermedi.
Uso: ~/ai/env-omlx/bin/python tools/gen_dsv41_fwd_step2c_reference.py
"""
import os
import numpy as np

def cvec(name, arr, per=8):
    vals = ["%.7g" % float(v) for v in np.asarray(arr, np.float32).reshape(-1)] or ["0.0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const float {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"

def ivec(name, arr, per=16):
    vals = [str(int(v)) for v in np.asarray(arr).reshape(-1)] or ["0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const int {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"

def main():
    import mlx.core as mx
    from omlx.patches.deepseek_v41.config import ModelConfig
    from omlx.patches.deepseek_v41.language import LanguageModel, hc_mixes, hc_pre_norm, hc_post

    cfg = ModelConfig(dim=8, n_layers=2, n_heads=2, o_groups=2, head_dim=32, rope_head_dim=4,
        index_head_dim=32, q_lora_rank=8, o_lora_rank=8, moe_inter_dim=8, n_routed_experts=4,
        n_activated_experts=2, n_shared_experts=1, window_size=4, compress_ratios=(0, 0),
        kv_source_layers=(), index_source_layers=(), engram_layer_ids=(), engram_num_embeddings=(),
        vocab_size=64, hc_mult=4, n_mtp_layers=1, preserve_mtp=False, max_seq_len=16)
    cfg.validate()
    model = LanguageModel(cfg)
    mx.random.seed(1234)
    def rnd(t):
        if isinstance(t, dict): return {k: rnd(v) for k, v in t.items()}
        if isinstance(t, list): return [rnd(v) for v in t]
        if isinstance(t, mx.array) and t.dtype in (mx.float32, mx.float16, mx.bfloat16):
            return mx.random.uniform(-0.1, 0.1, t.shape).astype(t.dtype)
        return t
    model.update(rnd(model.parameters()))

    c = model._config
    ids = mx.array([[3, 7]], mx.int32)
    h = mx.repeat(model.embed(ids)[..., None, :], c.hc_mult, -2)
    pre = mx.broadcast_to((mx.arange(c.hc_mult) == 0).astype(mx.float32), h.shape[:-1])
    # layer 0 via il Block stesso (con una cache fresca) -> h1, pre1
    L0 = model.layers[0]
    cache = model.make_cache()
    h1, pre1 = L0(h, pre, cache[0], {}, 0, None)
    # intermedi: attn_out e h_attn (per isolare errori)
    ap, ao, ac = hc_mixes(h, L0.hc_attn_fn, L0.hc_attn_scale, L0.hc_attn_base, c)
    x = hc_pre_norm(h, pre, L0.attn_norm.weight, L0.attn_norm.eps)
    attn_out = L0.attn(x, cache[0], {}, 0)
    h_attn = hc_post(attn_out, h, ao, ac)
    fp, fo, fc = hc_mixes(h_attn, L0.hc_ffn_fn, L0.hc_ffn_scale, L0.hc_ffn_base, c)
    ffn_in = hc_pre_norm(h_attn, ap, L0.ffn_norm.weight, L0.ffn_norm.eps)
    ffn_out = L0.ffn(ffn_in, None)
    mx.eval(h1, pre1, attn_out, h_attn, fp, fo, fc, ffn_in, ffn_out)

    out = ["/* GENERATO da tools/gen_dsv41_fwd_step2c_reference.py — NON modificare a mano. */\n",
           "#ifndef GMB_DSV41_FWD_STEP2C_REF_H\n#define GMB_DSV41_FWD_STEP2C_REF_H\n",
           "#define REF_S2C_L 2\n#define REF_S2C_DIM 8\n#define REF_S2C_HC 4\n#define REF_S2C_SIZE 24\n"
           "#define REF_S2C_E 4\n#define REF_S2C_ACT 2\n#define REF_S2C_DI 8\n"]
    out.append(cvec("REF_S2C_FFNIN", ffn_in))
    out.append(cvec("REF_S2C_FFNOUT", ffn_out))
    out.append(cvec("REF_S2C_HATTN", h_attn))
    out.append(cvec("REF_S2C_FP", fp))
    out.append(cvec("REF_S2C_FO", fo))
    out.append(cvec("REF_S2C_FC", fc))
    out.append(cvec("REF_S2C_H1", h1))
    out.append(cvec("REF_S2C_PRE1", pre1))
    out.append(cvec("REF_S2C_AP", ap))
    out.append(cvec("REF_S2C_FFNFN", L0.hc_ffn_fn))
    out.append(cvec("REF_S2C_FFNSCALE", L0.hc_ffn_scale))
    out.append(cvec("REF_S2C_FFNBASE", L0.hc_ffn_base))
    out.append(cvec("REF_S2C_FFNNORM", L0.ffn_norm.weight))
    out.append(cvec("REF_S2C_GW", L0.ffn.gate.weight))
    out.append(cvec("REF_S2C_GB", L0.ffn.gate.bias))
    E = L0.ffn.experts
    out.append(cvec("REF_S2C_E_W1", E.w1.weight))
    out.append(cvec("REF_S2C_E_W3", E.w3.weight))
    out.append(cvec("REF_S2C_E_W2", E.w2.weight))
    S = L0.ffn.shared_experts
    out.append(cvec("REF_S2C_S_W1", S.w1.weight))
    out.append(cvec("REF_S2C_S_W3", S.w3.weight))
    out.append(cvec("REF_S2C_S_W2", S.w2.weight))
    out.append("#endif /* GMB_DSV41_FWD_STEP2C_REF_H */\n")
    dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx",
                       "llm_dsv41_fwd_step2c_ref.h")
    with open(dst, "w") as f:
        f.write("".join(out))
    print("scritto:", os.path.realpath(dst), "| ffnout", ffn_out.shape, "h1", h1.shape)

if __name__ == "__main__":
    main()
