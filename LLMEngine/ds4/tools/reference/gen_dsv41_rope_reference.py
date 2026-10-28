#!/usr/bin/env python3
"""gen_dsv41_rope_reference.py — valori di riferimento per llm_dsv41_rope.h.

Replica VERBATIM di language.py `_rope`/`rope` (float32). Uso:
  ~/ai/env-omlx/bin/python tools/gen_dsv41_rope_reference.py
Scrive: LLMEnv/LLMEngine/mlx/llm_dsv41_rope_ref.h
"""
import math
import os
import numpy as np

MASK = 0x7FFFFFFF
TWO_PI = 2.0 * math.pi

def lcg(seed, n, lo=-1000, hi=1000):
    out, s = [], seed
    for _ in range(n):
        s = (s * 1103515245 + 12345) & MASK
        out.append((s % (hi - lo + 1) + lo) / 1000.0)
    return out

def rope(x, head_dim, pos, d, base, original_seq_len, beta_fast, beta_slow, rope_factor, compressed, inverse):
    x = np.array(x, dtype=np.float32)
    hd2 = d // 2
    freq = np.array([1.0 / (base ** ((2.0 * j) / d)) for j in range(hd2)], dtype=np.float32)
    if compressed and original_seq_len > 0:
        lnb = math.log(base)
        cfast = d * math.log(original_seq_len / (beta_fast * TWO_PI)) / (2.0 * lnb)
        cslow = d * math.log(original_seq_len / (beta_slow * TWO_PI)) / (2.0 * lnb)
        low = max(math.floor(cfast), 0)
        high = min(math.ceil(cslow), d - 1)
        denom = max(high - low, 1e-3)
        for j in range(hd2):
            sm = 1.0 - min(max((j - low) / denom, 0.0), 1.0)
            freq[j] = freq[j] / rope_factor * (1.0 - sm) + freq[j] * sm
    out = x.copy()
    off = head_dim - d
    for j in range(hd2):
        ang = pos * float(freq[j])
        if inverse:
            ang = -ang
        c, s = math.cos(ang), math.sin(ang)
        a, b = float(x[off + 2 * j]), float(x[off + 2 * j + 1])
        out[off + 2 * j] = np.float32(a * c - b * s)
        out[off + 2 * j + 1] = np.float32(a * s + b * c)
    return out

def cvec(name, arr, per=8):
    vals = ["%.7g" % float(v) for v in np.asarray(arr).reshape(-1)] or ["0.0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const float {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"

def main():
    out = ["/* GENERATO da tools/gen_dsv41_rope_reference.py — NON modificare a mano. */\n",
           "#ifndef GMB_DSV41_ROPE_REF_H\n#define GMB_DSV41_ROPE_REF_H\n"]
    head_dim, d = 6, 4
    x = np.array(lcg(70, head_dim), np.float32)
    out.append("#define REF_RP_HEADDIM %d\n#define REF_RP_D %d\n" % (head_dim, d))
    out.append(cvec("REF_RP_X", x))

    # (pos, base, orig, bf, bs, factor, compressed, inverse)
    cases = [
        (3.0, 10000.0, 0, 32, 1, 4.0, 0, 0),
        (100.0, 10000.0, 4096, 32, 1, 4.0, 1, 0),
        (100.0, 40000.0, 4096, 32, 1, 4.0, 1, 1),
    ]
    out.append("#define REF_RP_NCASE %d\n" % len(cases))
    for ci, (pos, base, orig, bf, bs, fac, comp, inv) in enumerate(cases):
        y = rope(x, head_dim, pos, d, base, orig, bf, bs, fac, comp, inv)
        out.append("#define REF_RP_POS%d %g\n#define REF_RP_BASE%d %g\n#define REF_RP_ORIG%d %d\n"
                   % (ci, pos, ci, base, ci, orig))
        out.append("#define REF_RP_BF%d %g\n#define REF_RP_BS%d %g\n#define REF_RP_FAC%d %g\n"
                   % (ci, bf, ci, bs, ci, fac))
        out.append("#define REF_RP_COMP%d %d\n#define REF_RP_INV%d %d\n" % (ci, comp, ci, inv))
        out.append(cvec("REF_RP_OUT%d" % ci, y))
    out.append("#endif /* GMB_DSV41_ROPE_REF_H */\n")
    dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx", "llm_dsv41_rope_ref.h")
    with open(dst, "w") as f:
        f.write("".join(out))
    print("scritto:", os.path.realpath(dst))

if __name__ == "__main__":
    main()
