#!/usr/bin/env python3
"""gen_dsv41_fp8_reference.py — riferimento packing FP8 e4m3 (window layer 0) per llm_dsv41_fp8.h."""
import os
import numpy as np

def cvec(name, arr, per=8):
    vals = ["%.7g" % float(v) for v in np.asarray(arr, np.float32).reshape(-1)] or ["0.0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const float {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"

def u8vec(name, arr, per=32):
    vals = [str(int(v)) for v in np.asarray(arr, np.uint8).reshape(-1)] or ["0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const unsigned char {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"

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
    cfg.validate(); model = LanguageModel(cfg)
    mx.random.seed(1234)
    def rnd(t):
        if isinstance(t, dict): return {k: rnd(v) for k, v in t.items()}
        if isinstance(t, list): return [rnd(v) for v in t]
        if isinstance(t, mx.array) and t.dtype in (mx.float32, mx.float16, mx.bfloat16):
            return mx.random.uniform(-0.1, 0.1, t.shape).astype(t.dtype)
        return t
    model.update(rnd(model.parameters()))
    c = model._config; A = model.layers[0].attn
    ids = mx.array([[3, 7]], mx.int32)
    h = mx.repeat(model.embed(ids)[..., None, :], c.hc_mult, -2)
    pre = mx.broadcast_to((mx.arange(c.hc_mult) == 0).astype(mx.float32), h.shape[:-1])
    x = hc_pre_norm(h, pre, model.layers[0].attn_norm.weight, model.layers[0].attn_norm.eps)
    positions = mx.arange(0, x.shape[1])
    _, kv = A._input_projections(x)
    winp = rope(A.kv_norm(kv), positions, c, False)          # float pre-pack (L x 32)
    packed = pack_activation(winp, 8, 32, False)              # (L x 33) uint8
    un = unpack_activation(packed, 8, 32, False, mx.float32)  # (L x 32) float
    mx.eval(winp, packed, un)
    out = ["/* GENERATO da tools/gen_dsv41_fp8_reference.py — NON modificare a mano. */\n",
           "#ifndef GMB_DSV41_FP8_REF_H\n#define GMB_DSV41_FP8_REF_H\n",
           "#define REF_FP8_L 2\n#define REF_FP8_W 32\n#define REF_FP8_BYTES 33\n"]
    out.append(cvec("REF_FP8_PRE", winp))
    out.append(u8vec("REF_FP8_PACKED", np.array(packed, np.uint8)))
    out.append(cvec("REF_FP8_UNPACKED", un))
    out.append("#endif /* GMB_DSV41_FP8_REF_H */\n")
    dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx", "llm_dsv41_fp8_ref.h")
    open(dst, "w").write("".join(out)); print("scritto:", os.path.realpath(dst), "| winp", winp.shape, "packed", packed.shape)
if __name__ == "__main__":
    main()
