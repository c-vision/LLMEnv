#!/usr/bin/env python3
"""gen_dsv41_engram_table_ref.py — fixture safetensors sintetica per il loader a righe della
tabella Engram (fp8 e4m3 + ue8m0). Uso: ~/ai/env-omlx/bin/python tools/gen_dsv41_engram_table_ref.py
Scrive: /tmp/dsv41_engram_table.safetensors e LLMEnv/LLMEngine/mlx/llm_dsv41_engram_table_ref.h
"""
import os
import numpy as np


def cvec(name, arr, per=8):
    vals = ["%.7g" % float(v) for v in np.asarray(arr, np.float32).reshape(-1)] or ["0.0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const float {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"


def u8vec(name, arr, per=24):
    vals = [str(int(v)) for v in np.asarray(arr, np.uint8).reshape(-1)] or ["0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const unsigned char {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"


def main():
    import mlx.core as mx
    rows, hd, block = 8, 64, 32
    scols = hd // block
    rng = np.random.default_rng(7)
    w = rng.integers(0, 256, (rows, hd), dtype=np.uint8)
    w[(w >> 3 & 15) == 15] &= 0xF8  # evita NaN e4m3
    s = rng.integers(120, 136, (rows, scols), dtype=np.uint8)
    wm, sm = mx.array(w, mx.uint8), mx.array(s, mx.uint8)
    expected = mx.from_fp8(wm, mx.float32) * np.ldexp(1.0, s.astype(np.int32) - 127).repeat(block, 1)
    mx.eval(expected)

    path = "/tmp/dsv41_engram_table.safetensors"
    mx.save_safetensors(path, {"engram.embed.weight": wm, "engram.embed.scale": sm})
    print("scritto:", path, "| rows", rows, "hd", hd)

    out = ["/* GENERATO da tools/gen_dsv41_engram_table_ref.py — NON modificare a mano. */\n",
           "#ifndef GMB_DSV41_ENGRAM_TABLE_REF_H\n#define GMB_DSV41_ENGRAM_TABLE_REF_H\n",
           f"#define REF_ENT_ROWS {rows}\n#define REF_ENT_HD {hd}\n#define REF_ENT_BLOCK {block}\n"]
    out.append(cvec("REF_ENT_EXPECT", np.array(expected, np.float32)))
    out.append("#endif /* GMB_DSV41_ENGRAM_TABLE_REF_H */\n")
    dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx", "llm_dsv41_engram_table_ref.h")
    open(dst, "w").write("".join(out))
    print("scritto:", os.path.realpath(dst))


if __name__ == "__main__":
    main()
