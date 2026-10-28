#!/usr/bin/env python3
"""gen_dsv41_fwd_ref.py — riferimento CONSOLIDATO del forward V4.1 (modello tiny, no download).
Dumpa pesi di entrambi i layer + norm + head + embed + ids + logits + window dequant per layer.
Uso: ~/ai/env-omlx/bin/python tools/gen_dsv41_fwd_ref.py
Scrive: LLMEnv/LLMEngine/mlx/llm_dsv41_fwd_ref.h
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
    from omlx.patches.deepseek_v41.language import LanguageModel, hc_mixes, hc_pre_norm, hc_post, rope
    from omlx.patches.deepseek_v41.quantization import pack_activation, unpack_activation
    from omlx.patches.deepseek_v41.head import project_logits

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
    logits = model(ids)
    mx.eval(logits)

    def win_deq(A, x):
        length = x.shape[1]; positions = mx.arange(0, length)
        _, kv_input = A._input_projections(x)
        new = pack_activation(rope(A.kv_norm(kv_input), positions, c, False))
        return unpack_activation(new, 8, 32, False, mx.float32)

    out = ["/* GENERATO da tools/gen_dsv41_fwd_ref.py — NON modificare a mano. */\n",
           "#ifndef GMB_DSV41_FWD_REF_H\n#define GMB_DSV41_FWD_REF_H\n",
           "#define REF_FW_NL 2\n#define REF_FW_L 2\n#define REF_FW_DIM 8\n#define REF_FW_HC 4\n"
           "#define REF_FW_SIZE 24\n#define REF_FW_H 2\n#define REF_FW_D 32\n#define REF_FW_DROPE 4\n"
           "#define REF_FW_QD 8\n#define REF_FW_OG 2\n#define REF_FW_OLORA 8\n#define REF_FW_WIN 4\n"
           "#define REF_FW_E 4\n#define REF_FW_ACT 2\n#define REF_FW_DI 8\n#define REF_FW_VOCAB 64\n"]
    out.append(cvec("REF_FW_EMBED", model.embed.weight))
    out.append(("static const int REF_FW_IDS[2] = {%d, %d};\n" % (int(ids[0,0]), int(ids[0,1]))))
    out.append(cvec("REF_FW_LOGITS", logits))
    out.append(cvec("REF_FW_FNORM", model.norm.weight))
    out.append(cvec("REF_FW_HEAD", model.head.weight))

    for i in range(2):
        L = model.layers[i]; A = L.attn; E = L.ffn.experts; S = L.ffn.shared_experts
        s = str(i)
        # ricostruisci l'input del layer per il win_deq
        if i == 0:
            h = mx.repeat(model.embed(ids)[..., None, :], c.hc_mult, -2)
            pre = mx.broadcast_to((mx.arange(c.hc_mult) == 0).astype(mx.float32), h.shape[:-1])
        # (per il dump del win_deq serve x = hc_pre_norm(h, pre, attn_norm))
        ap, ao, ac = hc_mixes(h, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base, c)
        x = hc_pre_norm(h, pre, L.attn_norm.weight, L.attn_norm.eps)
        wd = win_deq(A, x)
        out.append(cvec("REF_FW_WIN" + s, wd))
        for nm, v in [("HCAFN", L.hc_attn_fn), ("HCAS", L.hc_attn_scale), ("HCAB", L.hc_attn_base),
                      ("ANORM", L.attn_norm.weight), ("WQA", A.wq_a.weight), ("QN", A.q_norm.weight),
                      ("WQB", A.wq_b.weight), ("WKV", A.wkv.weight), ("KVN", A.kv_norm.weight),
                      ("WOA", A.wo_a.weight), ("WOB", A.wo_b.weight), ("SINK", A.attn_sink),
                      ("HCFFN", L.hc_ffn_fn), ("HCFFS", L.hc_ffn_scale), ("HCFFB", L.hc_ffn_base),
                      ("FNORM", L.ffn_norm.weight), ("GW", L.ffn.gate.weight), ("GB", L.ffn.gate.bias),
                      ("EW1", E.w1.weight), ("EW3", E.w3.weight), ("EW2", E.w2.weight),
                      ("SW1", S.w1.weight), ("SW3", S.w3.weight), ("SW2", S.w2.weight)]:
            out.append(cvec("REF_FW_" + nm + s, v))
        # avanza al layer successivo (replica Block) per l'input del layer 1
        attn_out = A(x, model.make_cache()[i], {}, 0)
        h = hc_post(attn_out, h, ao, ac)
        fp, fo, fc = hc_mixes(h, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base, c)
        ffn_in = hc_pre_norm(h, ap, L.ffn_norm.weight, L.ffn_norm.eps)
        h = hc_post(L.ffn(ffn_in, None), h, fo, fc)
        pre = fp
        mx.eval(h, pre)

    out.append("#endif /* GMB_DSV41_FWD_REF_H */\n")
    dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx", "llm_dsv41_fwd_ref.h")
    with open(dst, "w") as f:
        f.write("".join(out))
    print("scritto:", os.path.realpath(dst), "| logits", logits.shape)

if __name__ == "__main__":
    main()
