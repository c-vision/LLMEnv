#!/usr/bin/env python3
"""gen_dsv41_numpy_rng_ref.py — riferimento NumPy per il port C di `np.random.default_rng`
(SeedSequence + PCG64 + Lemire bounded). Verifica `llm_dsv41_numpy_rng.h`.

Uso: ~/ai/env-omlx/bin/python tools/gen_dsv41_numpy_rng_ref.py
Scrive: LLMEnv/LLMEngine/mlx/llm_dsv41_numpy_rng_ref.h
"""
import os
import numpy as np


def main():
    # seed, high, n
    cases = [
        (10007, 288230376151, 4),      # Engram layer 1 reale
        (140098, 288230376151, 4),     # Engram layer 14 reale
        (1, 288230376151, 4),
        (2, 1000, 8),
        (123456789, 7, 5),
        (0, 2**31 - 1, 6),
    ]
    rows = []
    for seed, high, n in cases:
        rng = np.random.default_rng(seed)
        vals = rng.integers(0, high, n, dtype=np.int64)
        rows.append((seed, high, n, np.asarray(vals, np.int64)))

    out = ["/* GENERATO da tools/gen_dsv41_numpy_rng_ref.py — NON modificare a mano. */\n",
           "#ifndef GMB_DSV41_NUMPY_RNG_REF_H\n#define GMB_DSV41_NUMPY_RNG_REF_H\n",
           "#define REF_RNG_NCASES %d\n" % len(rows)]
    seeds = ", ".join(str(s) for s, _, _, _ in rows)
    highs = ", ".join(str(h) for _, h, _, _ in rows)
    ns = ", ".join(str(n) for _, _, n, _ in rows)
    out.append(f"static const long long REF_RNG_SEED[{len(rows)}] = {{{seeds}}};\n")
    out.append(f"static const long long REF_RNG_HIGH[{len(rows)}] = {{{highs}}};\n")
    out.append(f"static const int REF_RNG_N[{len(rows)}] = {{{ns}}};\n")
    maxn = max(n for _, _, n, _ in rows)
    flat = []
    for _, _, n, v in rows:
        flat.extend(int(x) for x in v)
        flat.extend([0] * (maxn - n))
    out.append(f"static const long long REF_RNG_EXPECT[{len(rows)} * {maxn}] = {{\n    "
               + ", ".join(str(x) for x in flat) + "\n};\n")
    out.append(f"#define REF_RNG_MAXN {maxn}\n")
    out.append("#endif /* GMB_DSV41_NUMPY_RNG_REF_H */\n")
    dst = os.path.join(os.path.dirname(__file__), "..", "native", "gmb_host_ai_mlx", "llm_dsv41_numpy_rng_ref.h")
    open(dst, "w").write("".join(out))
    print("scritto:", os.path.realpath(dst), "| casi", len(rows))
    for s, h, n, v in rows:
        print("  seed", s, "high", h, "->", v.tolist())


if __name__ == "__main__":
    main()
