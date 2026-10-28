#!/usr/bin/env python3
"""mlx_quant_dequant_ref.py — reference dequantizer for MLX-engine HF 4-bit checkpoints.

Independent (pure numpy) implementation of the dequant formulas used by the MLX engine
(`LLMEnv/LLMCodec/src/gmb_quant_{awq,gptq,nf4}.c`). Use it to (a) cross-check the C
codec element-wise, and (b) materialize a plain F16 safetensors model to feed a separate
framework (`mlx_lm`) for token-parity verification.

Auto-detects the format from the tensor names:
  * GPTQ (AutoGPTQ/gptqmodel): `<base>.qweight` + `.qzeros` + `.scales` + `.g_idx`
  * AWQ  (AutoAWQ GEMM):       `<base>.qweight` + `.qzeros` + `.scales` (no `.g_idx`)
  * NF4  (bitsandbytes):       `<base>.weight` + `.weight.quant_state.bitsandbytes__nf4`

Formulas (sources):
  - GPTQ: AutoGPTQ `qlinear_cuda.py` pack/forward. qweight [in/(32/bits),out] packed along
    IN, qzeros [in/group,out/(32/bits)] packed along OUT, scales F16 [in/group,out],
    g_idx[i]=group of input row i. Pack stores `zeros-1`; dequant:
        w[o,i] = (q - ((z_raw)+1)) * scales[g_idx[i], o]
  - AWQ: AutoAWQ `packing_utils.dequantize_gemm`, reverse order [0,4,1,5,2,6,3,7].
  - NF4: bitsandbytes `functional.py` `get_4bit_type("nf4")` + `dequantize_4bit`; nibble ALTO
    = elemento pari; absmax doppio-quantizzato:
        absmax[b] = nested_quant_map[absmax_u8[b]] * nested_absmax[b//nested_blocksize]
                    + nested_offset
        w[j] = quant_map[nibble] * absmax[j//blocksize]

Usage:
  python3 tools/mlx_quant_dequant_ref.py <model_dir> --tensor <base> [--out f.npy]
  python3 tools/mlx_quant_dequant_ref.py <model_dir> --to-model <out_dir>
"""
import argparse, json, os, shutil, struct
import numpy as np

AWQ_REVERSE = [0, 4, 1, 5, 2, 6, 3, 7]


def load(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
    return hdr, base


def make_reader(path, hdr, base):
    def raw(name):
        t = hdr[name]
        o0, o1 = t["data_offsets"]
        with open(path, "rb") as f:
            f.seek(base + o0)
            return f.read(o1 - o0)
    return raw


def read(raw, name, hdr, force=None):
    t = hdr[name]
    dt = force or t["dtype"]
    buf = raw(name)
    if dt == "F16":
        return np.frombuffer(buf, "<f2").astype(np.float32).reshape(t["shape"])
    if dt == "BF16":
        u = np.frombuffer(buf, "<u2").astype(np.uint32)
        return (u << 16).view(np.float32).reshape(t["shape"])
    if dt == "F32":
        return np.frombuffer(buf, "<f4").reshape(t["shape"])
    if dt == "I32":
        return np.frombuffer(buf, "<u4").reshape(t["shape"])
    if dt == "U8":
        return np.frombuffer(buf, "<u1").reshape(t["shape"])
    raise ValueError(dt)


def dequant_gptq(raw, hdr, base):
    qw = read(raw, base + ".qweight", hdr, "I32")
    qz = read(raw, base + ".qzeros", hdr, "I32")
    sc = read(raw, base + ".scales", hdr, "F16")
    gi = read(raw, base + ".g_idx", hdr, "I32").astype(np.int64) if (base + ".g_idx") in hdr else None
    IN, OUT = qw.shape[0] * 8, qw.shape[1]
    g = sc.shape[0]
    if gi is None:
        gi = (np.arange(IN) // (IN // g)).astype(np.int64)
    qw = qw.reshape(IN // 8, OUT)
    qz = qz.reshape(g, OUT // 8)
    zmat = np.zeros((g, OUT), np.float32)
    for k in range(OUT // 8):
        zmat[:, k * 8:(k + 1) * 8] = (((qz[:, k:k + 1] >> (4 * np.arange(8))) & 0xF) + 1).astype(np.float32)
    W = np.zeros((OUT, IN), np.float32)
    for i in range(IN):
        q = ((qw[i // 8] >> (4 * (i % 8))) & 0xF).astype(np.float32)
        W[:, i] = (q - zmat[gi[i]]) * sc[gi[i]]
    return W


def dequant_awq(raw, hdr, base):
    qw = read(raw, base + ".qweight", hdr, "I32")   # [in, out/8]
    qz = read(raw, base + ".qzeros", hdr, "I32")    # [in/group, out/8]
    sc = read(raw, base + ".scales", hdr, "F16")    # [in/group, out]
    IN, OP = qw.shape
    OUT = OP * 8
    g = sc.shape[0]
    W = np.zeros((OUT, IN), np.float32)
    for o in range(OUT):
        c = o >> 3
        sh = 4 * AWQ_REVERSE[o & 7]
        q = ((qw[:, c] >> sh) & 0xF).astype(np.float32)
        z = ((qz[:, c] >> sh) & 0xF).astype(np.float32)
        zz = np.repeat(z, IN // g)[:IN]
        ss = np.repeat(sc[:, o], IN // g)[:IN]
        W[o, :] = (q - zz) * ss
    return W


def dequant_nf4(raw, hdr, base):
    suf = ".weight.quant_state.bitsandbytes__nf4"
    qs = json.loads(bytes(raw(base + suf)).decode())
    OUT, IN = qs["shape"]
    bs = qs["blocksize"]
    nbs = qs["nested_blocksize"]
    off = qs["nested_offset"]
    w = read(raw, base + ".weight", hdr, "U8").reshape(-1)
    ab = read(raw, base + ".weight.absmax", hdr, "U8").reshape(-1)
    nabs = read(raw, base + ".weight.nested_absmax", hdr, "F32")
    nmap = read(raw, base + ".weight.nested_quant_map", hdr, "F32")
    cmap = read(raw, base + ".weight.quant_map", hdr, "F32")
    N = OUT * IN
    nblocks = (N + bs - 1) // bs
    absmax = nmap[ab[:nblocks]] * nabs[np.arange(nblocks) // nbs] + off
    idx = np.arange(N)
    byte = w[idx // 2].astype(np.int32)
    nib = np.where(idx % 2 == 0, byte >> 4, byte & 0xF)
    return (cmap[nib] * absmax[idx // bs]).reshape(OUT, IN)


def detect(hdr, base):
    if (base + ".weight.quant_state.bitsandbytes__nf4") in hdr:
        return "nf4"
    if (base + ".qweight") in hdr:
        return "gptq" if (base + ".g_idx") in hdr else "awq"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--tensor", help="tensor base (e.g. model.layers.0.self_attn.q_proj)")
    ap.add_argument("--out", help="(.npy) for --tensor")
    ap.add_argument("--to-model", help="materialize a plain-F16 model dir")
    a = ap.parse_args()
    path = os.path.join(a.model_dir, "model.safetensors")
    hdr, base = load(path)
    raw = make_reader(path, hdr, base)
    fn = {"gptq": dequant_gptq, "awq": dequant_awq, "nf4": dequant_nf4}

    if a.tensor:
        fmt = detect(hdr, a.tensor)
        if fmt is None:
            raise SystemExit(f"{a.tensor}: formato non riconosciuto")
        W = fn[fmt](raw, hdr, a.tensor)
        print(f"{a.tensor}: fmt={fmt} shape={W.shape} mean|w|={np.abs(W).mean():.6g} max|w|={np.abs(W).max():.6g}")
        if a.out:
            np.save(a.out, W)
            print("saved", a.out)
        return

    if a.to_model:
        quant_w = {}
        for name in hdr:
            if name.endswith(".weight.quant_state.bitsandbytes__nf4"):
                b = name[: -len(".weight.quant_state.bitsandbytes__nf4")]
                quant_w[b + ".weight"] = "nf4"
            elif name.endswith(".qweight"):
                b = name[:-8]
                quant_w[b + ".weight"] = detect(hdr, b)
        out = {}
        for name in hdr:
            if name == "__metadata__":
                continue
            if name in quant_w:
                out[name] = fn[quant_w[name]](raw, hdr, name[: -len(".weight")]).astype(np.float16)
            elif name.endswith((".qzeros", ".scales", ".g_idx", ".weight.absmax",
                               ".weight.nested_absmax", ".weight.nested_quant_map",
                               ".weight.quant_map", ".weight.quant_state.bitsandbytes__nf4")):
                continue
            else:
                t = hdr[name]
                if t["dtype"] in ("F16", "BF16", "F32"):
                    out[name] = read(raw, name, hdr).astype(np.float16)
                else:
                    out[name] = read(raw, name, hdr)
        os.makedirs(a.to_model, exist_ok=True)
        h = {"__metadata__": {"format": "pt"}}
        off = 0
        blobs = []
        for k in sorted(out):
            arr = np.ascontiguousarray(out[k])
            b = arr.tobytes()
            h[k] = {"dtype": "F16", "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
            blobs.append(b)
            off += len(b)
        js = json.dumps(h, separators=(",", ":")).encode()
        with open(os.path.join(a.to_model, "model.safetensors"), "wb") as f:
            f.write(struct.pack("<Q", len(js)))
            f.write(js)
            for b in blobs:
                f.write(b)
        for f2 in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "generation_config.json", "added_tokens.json", "special_tokens_map.json", "LICENSE"):
            p = os.path.join(a.model_dir, f2)
            if os.path.exists(p):
                shutil.copy(p, os.path.join(a.to_model, f2))
        cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
        cfg.pop("quantization_config", None)
        cfg["torch_dtype"] = "float16"
        json.dump(cfg, open(os.path.join(a.to_model, "config.json"), "w"), indent=2)
        print(f"wrote {a.to_model} ({len(out)} tensori)")
        return
    ap.error("specificare --tensor oppure --to-model")


if __name__ == "__main__":
    main()
