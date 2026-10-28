#!/usr/bin/env python3
"""gen_dsv41_real_dequant_ref.py — riferimento di dequant su DATI REALI (shard presenti).
Copre tutti i tipi del layer 0 reale: fp8 32x32, fp4 e2m1, bf16/f32 plain.
Uso: DSV41_MODEL_DIR=<dir> ~/ai/env-omlx/bin/python tools/gen_dsv41_real_dequant_ref.py
"""
import os
import numpy as np

_LUT4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)


def fp8_dequant(w, sc):
    import mlx.core as mx
    w = np.asarray(w, np.uint8); inn = w.shape[1]
    s = np.ldexp(1.0, (np.asarray(sc, np.uint8).astype(np.int32) - 127)).astype(np.float32)
    s2 = np.repeat(np.repeat(s, 32, 0), 32, 1)[:w.shape[0], :inn]
    return np.asarray(mx.from_fp8(mx.array(w), mx.float32), np.float32) * s2


def fp4_dequant(wp, sc):
    wp = np.asarray(wp, np.uint8); out, in2 = wp.shape; inn = in2 * 2
    def dec(c):
        v = _LUT4[c & 7]; return np.where(c & 8, -v, v)
    vals = np.empty((out, inn), np.float32)
    vals[:, 0::2] = dec(wp & 15); vals[:, 1::2] = dec((wp >> 4) & 15)
    s = np.ldexp(1.0, (np.asarray(sc, np.uint8).astype(np.int32) - 127)).astype(np.float32)
    return vals * np.repeat(s, 32, 1)[:, :inn]


# (nome, mode: 0 plain / 1 fp8 / 2 fp4)
TENSORS = [
    ("layers.0.attn.wq_a.weight", 1),
    ("layers.0.attn.wq_b.weight", 1),
    ("layers.0.attn.wkv.weight", 1),
    ("layers.0.attn.wo_a.weight", 1),
    ("layers.0.attn.wo_b.weight", 1),
    ("layers.0.ffn.shared_experts.w1.weight", 1),
    ("layers.0.ffn.shared_experts.w2.weight", 1),
    ("layers.0.ffn.shared_experts.w3.weight", 1),
    ("layers.0.ffn.experts.0.w1.weight", 2),
    ("layers.0.ffn.experts.0.w2.weight", 2),
    ("layers.0.ffn.experts.0.w3.weight", 2),
    ("layers.0.attn.q_norm.weight", 0),
    ("layers.0.attn.kv_norm.weight", 0),
    ("layers.0.attn.attn_sink", 0),
    ("layers.0.attn_norm.weight", 0),
    ("layers.0.ffn_norm.weight", 0),
    ("layers.0.hc_attn_fn", 0),
    ("layers.0.hc_attn_scale", 0),
    ("layers.0.hc_attn_base", 0),
    ("layers.0.hc_ffn_fn", 0),
    ("layers.0.ffn.gate.weight", 0),
    ("layers.0.ffn.gate.bias", 0),
    ("layers.0.ffn.gate.bias_vl", 0),
    # vision reale (shard 1, bf16 plain): patch embed, ViT blocco 0, norm finale, aligner
    ("vision.patch_embed.proj.weight", 0),
    ("vision.patch_embed.proj.bias", 0),
    ("vision.blocks.0.attn.wqkv.weight", 0),
    ("vision.blocks.0.attn.wo.weight", 0),
    ("vision.blocks.0.mlp.w1.weight", 0),
    ("vision.blocks.0.mlp.w2.weight", 0),
    ("vision.blocks.0.norm1.weight", 0),
    ("vision.norm.weight", 0),
    ("aligner.w1.weight", 0),
    ("aligner.w2.bias", 0),
    # layer 1 reale (shard 4, ratio 0): nomi/shape di un secondo layer
    ("layers.1.attn.wq_a.weight", 1),
    ("layers.1.attn.q_norm.weight", 0),
    ("layers.1.ffn.experts.0.w1.weight", 2),
    ("layers.1.ffn.shared_experts.w2.weight", 1),
    ("layers.1.ffn.gate.weight", 0),
    ("layers.1.hc_attn_fn", 0),
    # layer 2 reale (shard 5, CSA2): compressor + indexer (nuovo code path)
    ("layers.2.attn.compressor.wkv.weight", 0),
    ("layers.2.attn.compressor.norm.weight", 0),
    ("layers.2.attn.compressor.wgate.weight", 0),
    ("layers.2.attn.indexer.wq_b.weight", 1),
    ("layers.2.attn.indexer.weights_proj.weight", 0),
    ("layers.2.attn.indexer.wk.weight", 0),
    ("layers.2.attn.indexer.k_norm.weight", 0),
    ("layers.2.attn.wq_a.weight", 1),
    ("layers.2.ffn.experts.0.w1.weight", 2),
    # layer 3-4 (shard 6-7): plain (ratio>0 ma NON source -> riuso condiviso)
    ("layers.3.attn.wq_a.weight", 1),
    ("layers.3.attn.attn_sink", 0),
    ("layers.3.ffn.experts.0.w1.weight", 2),
    ("layers.4.attn.wkv.weight", 1),
    ("layers.4.attn.q_norm.weight", 0),
    ("layers.4.ffn.shared_experts.w2.weight", 1),
]


def main():
    dest = os.environ.get("DSV41_MODEL_DIR", "/Volumes/Armor5TB/LocalLLM/ai/models/DeepSeek-V4.1-Flash")
    import mlx.core as mx
    d = mx.load(os.path.join(dest, "model-00003-of-00048.safetensors"))
    dvis = [None]
    d1 = [None]
    d2 = [None]
    dm = {}
    def get(name):
        if name.startswith(("vision.", "aligner.", "image_")):
            if dvis[0] is None:
                dvis[0] = mx.load(os.path.join(dest, "model-00001-of-00048.safetensors"))
            return dvis[0][name]
        if name.startswith("layers.1."):
            if d1[0] is None:
                d1[0] = mx.load(os.path.join(dest, "model-00004-of-00048.safetensors"))
            return d1[0][name]
        if name.startswith("layers.2."):
            if d2[0] is None:
                d2[0] = mx.load(os.path.join(dest, "model-00005-of-00048.safetensors"))
            return d2[0][name]
        for lyr, shard in (("layers.3.", "model-00006-of-00048.safetensors"),
                           ("layers.4.", "model-00007-of-00048.safetensors")):
            if name.startswith(lyr):
                if shard not in dm:
                    dm[shard] = mx.load(os.path.join(dest, shard))
                return dm[shard][name]
        return d[name]
    names, modes, outs, ins, nsm, idxs, vals, sums = [], [], [], [], [], [], [], []
    rng = np.random.default_rng(11)
    for name, mode in TENSORS:
        if mode == 0:
            arr = np.asarray(mx.array(get(name)).astype(mx.float32), np.float32)
            f = arr.reshape(-1); out = arr.size; inn = 0
        else:
            w = np.asarray(get(name), np.uint8)
            sc = np.asarray(get(name.rsplit(".", 1)[0] + ".scale"), np.uint8)
            dq = fp8_dequant(w, sc) if mode == 1 else fp4_dequant(w, sc)
            f = dq.reshape(-1); out = dq.shape[0]; inn = dq.shape[1]
        N = 16
        nS = min(N, int(f.size))
        idx = sorted(rng.choice(f.size, size=nS, replace=(nS < f.size)).tolist()) if nS < N else list(range(nS))
        idx = sorted(set(idx))[:nS]
        while len(idx) < N: idx.append(idx[-1] if idx else 0)
        vals.append([float(f[i]) for i in idx])
        nsm.append(min(N, nS))
        idxs.append(idx); sums.append(float(f.astype(np.float64).sum()))
        names.append(name); modes.append(mode); outs.append(out); ins.append(inn)
        print(f"  {name} mode={mode} out={out} in={inn} sum={sums[-1]:.6g}")

    def carr(name, lst): return "static const char* %s[%d] = { %s };\n" % (name, len(lst), ", ".join('"%s"' % s for s in lst))
    def iarr(name, lst): return "static const int %s[%d] = { %s };\n" % (name, len(lst), ", ".join(str(int(v)) for v in lst))
    def darr(name, lst): return "static const double %s[%d] = { %s };\n" % (name, len(lst), ", ".join("%.12g" % v for v in lst))
    N = 16
    def i2d(name, rows):
        body = ",\n".join("    { " + ", ".join(str(int(v)) for v in r) + " }" for r in rows)
        return "static const long %s[%d][%d] = {\n%s\n};\n" % (name, len(rows), N, body)
    def f2d(name, rows):
        body = ",\n".join("    { " + ", ".join("%.9g" % v for v in r) + " }" for r in rows)
        return "static const float %s[%d][%d] = {\n%s\n};\n" % (name, len(rows), N, body)

    out = ["/* GENERATO da tools/gen_dsv41_real_dequant_ref.py — DATI REALI (layer 0). */\n",
           "#ifndef GMB_DSV41_REAL_REF_H\n#define GMB_DSV41_REAL_REF_H\n",
           f"#define REF_RT_COUNT {len(names)}\n#define REF_RT_N {N}\n",
           carr("REF_RT_NAME", names), iarr("REF_RT_MODE", modes), iarr("REF_RT_OUT", outs),
           iarr("REF_RT_IN", ins), iarr("REF_RT_NS", nsm),
           i2d("REF_RT_IDX", idxs), f2d("REF_RT_VAL", vals), darr("REF_RT_SUM", sums),
           "#endif /* GMB_DSV41_REAL_REF_H */\n"]
    dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx", "llm_dsv41_real_ref.h")
    open(dst, "w").write("".join(out))
    print("scritto:", os.path.realpath(dst), "|", len(names), "tensori")


if __name__ == "__main__":
    main()
