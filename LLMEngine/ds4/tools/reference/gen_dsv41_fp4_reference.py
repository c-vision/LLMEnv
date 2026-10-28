#!/usr/bin/env python3
"""gen_dsv41_fp4_reference.py — riferimento packing FP4 E2M1 (e FP8 con scala e4m3) per
llm_dsv41_fp4.h. Copre i tre formati usati da CSA2:
  A) bits=4, group_size=32, e4m3_scale=False  (index q/key: pack_activation default)
  B) bits=4, group_size=16, e4m3_scale=True   (KV compresso)
  C) bits=8, group_size=16, e4m3_scale=True   (variante con scala e4m3)
Genera pre / packed / unpacked per ciascuno. Uso: ~/ai/env-omlx/bin/python tools/gen_dsv41_fp4_reference.py
Scrive: LLMEnv/LLMEngine/mlx/llm_dsv41_fp4_ref.h
"""
import os
import numpy as np


def cvec(name, arr, per=8):
    vals = ["%.7g" % float(v) for v in np.asarray(arr, np.float32).reshape(-1)] or ["0.0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const float {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"


def u8vec(name, arr, per=16):
    vals = [str(int(v)) for v in np.asarray(arr, np.uint8).reshape(-1)] or ["0"]
    lines = ["    " + ", ".join(vals[i:i + per]) + "," for i in range(0, len(vals), per)]
    return f"static const unsigned char {name}[{len(vals)}] = {{\n" + "\n".join(lines) + "\n};\n"


def main():
    import mlx.core as mx
    from omlx.patches.deepseek_v41.quantization import pack_activation, unpack_activation

    mx.random.seed(99)
    cases = [
        ("A", 4, 32, False, 64),
        ("B", 4, 16, True, 32),
        ("C", 8, 16, True, 32),
    ]
    out = ["/* GENERATO da tools/gen_dsv41_fp4_reference.py — NON modificare a mano. */\n",
           "#ifndef GMB_DSV41_FP4_REF_H\n#define GMB_DSV41_FP4_REF_H\n"]
    for name, bits, gs, e4, width in cases:
        x = mx.random.uniform(-1.5, 1.5, (3, width))
        packed = pack_activation(x, bits, gs, e4)
        un = unpack_activation(packed, bits, gs, e4, mx.float32)
        mx.eval(x, packed, un)
        p = np.array(packed, np.uint8)
        nbytes = 3 * (width * bits // 8 + width // gs)
        out.append(f"#define REF_{name}_BITS {bits}\n#define REF_{name}_GS {gs}\n"
                   f"#define REF_{name}_E4 {1 if e4 else 0}\n#define REF_{name}_L 3\n"
                   f"#define REF_{name}_W {width}\n#define REF_{name}_BYTES {nbytes}\n")
        out.append(cvec(f"REF_{name}_PRE", np.array(x, np.float32)))
        out.append(u8vec(f"REF_{name}_PACKED", p))
        out.append(cvec(f"REF_{name}_UNPACKED", np.array(un, np.float32)))
    out.append("#endif /* GMB_DSV41_FP4_REF_H */\n")
    dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx", "llm_dsv41_fp4_ref.h")
    open(dst, "w").write("".join(out))
    print("scritto:", os.path.realpath(dst))


if __name__ == "__main__":
    main()
