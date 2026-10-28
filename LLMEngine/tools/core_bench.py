#!/usr/bin/env python3
"""core_bench.py — micro-bench MODEL-FREE delle primitive del core MLX di GMBasic.

Carica il dylib `libgmb_ai_mlx.dylib` e chiama l'export `llm_ai_mlx_core_bench`:
misura OGNI wrapper delle primitive core contro la CHIAMATA MLX DIRETTA equivalente,
con input sintetici deterministici alle shape/dtype reali. Stampa TSV:
    OP <name> shape=<...> raw_ms= gmb_ms= ratio= enc_us=raw/gmb maxdiff=
- raw_ms/gmb_ms: latenza per chiamata con eval-sync (include ~0.15 ms di sync fisso).
- ratio = gmb/raw (<1 = wrapper piu' veloce del MLX diretto).
- enc_us: costo CPU di ENCODING del grafo per op (K build senza sync), separato dalla GPU.
- maxdiff: |gmb - raw| max (0 = fedele; ~1e-2 atteso solo per attivazioni bf16 fuse).

Uso:
    tools/core_bench.py [op|all] [iters] [dylib]
    op    in {rms_norm,qlinear,sdpa,swiglu,gptoss_swiglu,geglu,rms_norm_prefill,
              qlinear_prefill,logit_softcap,embed,kv_append,moe_mxfp4,all}
    iters default 200; dylib default test_mlx_bench/bin/Release/net10.0/.
"""
import ctypes
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DYLIB = os.path.join(HERE, "..", "test_mlx_bench", "bin", "Release", "net10.0", "libgmb_ai_mlx.dylib")


def main() -> int:
    op = sys.argv[1] if len(sys.argv) > 1 else "all"
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    dylib = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_DYLIB
    dylib = os.path.abspath(dylib)
    if not os.path.exists(dylib):
        print(f"dylib non trovato: {dylib}", file=sys.stderr)
        return 2
    lib = ctypes.CDLL(dylib)
    fn = lib.llm_ai_mlx_core_bench
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
    buf = ctypes.create_string_buffer(8192)
    n = fn(op.encode(), iters, buf, ctypes.sizeof(buf))
    sys.stdout.write(buf.value.decode())
    print(f"# corebench op={op} iters={iters} ops={n}")
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
