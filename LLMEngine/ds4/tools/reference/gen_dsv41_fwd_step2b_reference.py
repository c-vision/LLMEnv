#!/usr/bin/env python3
"""gen_dsv41_fwd_step2b_reference.py — riferimento step 2b (attention layer 0).
Nota: la attention del riferimento legge KV **impacchettati fp8**; qui si usa il window
DEQUANTIZZATO (unpack) come input per il C, quindi si verifica rope+sdpa+wo:
  q = rope(wq_b(q_norm(wq_a(x)))); win = unpack(pack(rope(kv_norm(wkv(x)))))
  out = attn(x) [riferimento, packed]; ci si aspetta che il C con win dequantizzato lo riproduca.
Il PACKING/DEQUANT esatto e' nel todo KV-Packed-Layout.
Uso: ~/ai/env-omlx/bin/python tools/gen_dsv41_fwd_step2b_reference.py
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
    from omlx.patches.deepseek_v41.language import LanguageModel, hc_mixes, hc_pre_norm, rope
    from omlx.patches.deepseek_v41.quantization import pack_activation, unpack_activation

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

    c = model._config; L0 = model.layers[0]; A = L0.attn
    ids = mx.array([[3, 7]], mx.int32)
    h = mx.repeat(model.embed(ids)[..., None, :], c.hc_mult, -2)
    pre = mx.broadcast_to((mx.arange(c.hc_mult) == 0).astype(mx.float32), h.shape[:-1])
    ap, ao, ac = hc_mixes(h, L0.hc_attn_fn, L0.hc_attn_scale, L0.hc_attn_base, c)
    x = hc_pre_norm(h, pre, L0.attn_norm.weight, L0.attn_norm.eps)

    length = x.shape[1]; start = 0
    positions = mx.arange(start, start + length)
    query, kv_input = A._input_projections(x)
    qr = A.q_norm(query)
    q = rope(A.wq_b(qr).reshape(1, length, c.n_heads, c.head_dim), positions, c, False)
    new = pack_activation(rope(A.kv_norm(kv_input), positions, c, False))
    win_deq = unpack_activation(new, 8, 32, False, mx.float32)
    cache = model.make_cache()
    out = A(x, cache[0], {}, start)
    # intermedi: wi (indici finestra), o = attn output dopo rope inverso (pre-wo)
    from omlx.patches.deepseek_v41.kernels import packed_sparse_attention
    local = mx.maximum(positions[:, None] - c.window_size + 1, 0) + mx.arange(min(length, c.window_size))
    valid = (local >= 0) & (local <= positions[:, None])
    wi = mx.where(valid, local, -1)[None].astype(mx.int32)
    empty_pooled = mx.zeros((1, 0, c.head_dim // 2 + c.head_dim // 16), mx.uint8)
    ci = mx.zeros((1, length, 0), mx.int32)
    o = packed_sparse_attention(q, new, empty_pooled, wi, ci, A.attn_sink, c.head_dim ** -0.5)
    o = rope(o, positions, c, bool(0), inverse=True)
    mx.eval(out, q, win_deq, wi, o)

    outd = ["/* GENERATO da tools/gen_dsv41_fwd_step2b_reference.py — NON modificare a mano. */\n",
            "#ifndef GMB_DSV41_FWD_STEP2B_REF_H\n#define GMB_DSV41_FWD_STEP2B_REF_H\n",
            "#define REF_S2B_L 2\n#define REF_S2B_DIM 8\n#define REF_S2B_H 2\n#define REF_S2B_D 32\n#define REF_S2B_DROPE 4\n"
            "#define REF_S2B_QD 8\n#define REF_S2B_OG 2\n#define REF_S2B_OLORA 8\n#define REF_S2B_WIN 4\n"]
    outd.append(cvec("REF_S2B_X", x))
    outd.append(cvec("REF_S2B_WQA", A.wq_a.weight))
    outd.append(cvec("REF_S2B_WQAX", A.wq_a(x)))
    outd.append(cvec("REF_S2B_QRN", A.q_norm(A.wq_a(x))))
    outd.append(cvec("REF_S2B_QNORM", A.q_norm.weight))
    outd.append(cvec("REF_S2B_WQB", A.wq_b.weight))
    outd.append(cvec("REF_S2B_WKV", A.wkv.weight))
    outd.append(cvec("REF_S2B_KVNORM", A.kv_norm.weight))
    outd.append(cvec("REF_S2B_WOA", A.wo_a.weight))
    outd.append(cvec("REF_S2B_WOB", A.wo_b.weight))
    outd.append(cvec("REF_S2B_SINK", A.attn_sink))
    outd.append(cvec("REF_S2B_WIN_DEQ", win_deq))
    outd.append(cvec("REF_S2B_Q", q))
    outd.append(("static const int REF_S2B_WI[%d] = {%s};\n"
                 % (wi.size, ", ".join(str(int(v)) for v in np.array(wi).reshape(-1)))))
    outd.append(cvec("REF_S2B_O", o))
    outd.append(cvec("REF_S2B_OUT", out))
    outd.append("#endif /* GMB_DSV41_FWD_STEP2B_REF_H */\n")
    dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx",
                       "llm_dsv41_fwd_step2b_ref.h")
    with open(dst, "w") as f:
        f.write("".join(outd))
    print("scritto:", os.path.realpath(dst), "| win_deq", win_deq.shape, "out", out.shape)

if __name__ == "__main__":
    main()
