#!/usr/bin/env python
"""A/B GMBasic vs mlx_lm a contesti crescenti (prefill + decode), entrambi gli ordini.

GMB:     `test_mlx_bench <model> context <ctx,...>` (UN load per modello, sweep contesti).
mlx_lm:  `tools/bench_mlxlm.py --lengths <ctx,...>` (UN load per modello).

Protocollo: ogni ordine esegue le due fasi in sequenza nella stessa finestra quiete
(gmb-first = GMB poi mlx_lm; mlxlm-first = viceversa). Mediana su `--reps` per fase.
Output TSV: model<TAB>order<TAB>N<TAB>gmb_prefill<TAB>mlxlm_prefill<TAB>gmb_decode<TAB>mlxlm_decode

Uso:
  ab_ctx_mlxlm.py --model <dir> [--contexts 512,2048,4096,8192] [--gen 32] [--reps 2]
                  [--orders gmb-first,mlxlm-first] [--out file.tsv]
"""
import argparse
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "test_mlx_bench" / "bin" / "Release" / "net10.0"
DLL = BIN / "test_mlx_bench.dll"
PY = "/Users/gaetano/ai/env312/bin/python"


def dotnet():
    for c in ("dotnet", "/usr/local/share/dotnet/dotnet", "/usr/bin/dotnet"):
        from shutil import which
        p = which(c)
        if p:
            return p
    return "dotnet"


def run_gmb(model, contexts, reps):
    """Ritorna {N: (prefill_med, decode_med)}."""
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = str(BIN)
    acc = {}
    for _ in range(reps):
        p = subprocess.run([dotnet(), str(DLL), model, "context", contexts],
                           capture_output=True, text=True, env=env)
        for line in p.stdout.splitlines():
            m = re.search(r"CONTEXT N=\s*(\d+)\s+prefill=.*?\(([\d.]+) tok/s\)\s+decode32=([\d.]+) tok/s", line)
            if m:
                n = int(m.group(1))
                acc.setdefault(n, ([], []))
                acc[n][0].append(float(m.group(2)))
                acc[n][1].append(float(m.group(3)))
    return {n: (statistics.median(v[0]), statistics.median(v[1])) for n, v in acc.items()}


def run_mlxlm(model, contexts, gen, reps):
    p = subprocess.run([PY, str(ROOT / "tools" / "bench_mlxlm.py"), "--model", model,
                        "--lengths", contexts, "--max-tokens", str(gen), "--repeats", str(reps)],
                       capture_output=True, text=True)
    out = {}
    for line in p.stdout.splitlines():
        m = re.search(r"N=(\d+) prefill ([\d.]+) tok/s decode ([\d.]+) tok/s", line)
        if m:
            out[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--contexts", default="512,2048,4096,8192")
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--orders", default="gmb-first,mlxlm-first")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    name = Path(a.model).name
    orders = [o.strip() for o in a.orders.split(",") if o.strip()]

    rows = []
    for order in orders:
        t0 = time.time()
        if order == "gmb-first":
            g = run_gmb(a.model, a.contexts, a.reps)
            m = run_mlxlm(a.model, a.contexts, a.gen, a.reps)
        else:
            m = run_mlxlm(a.model, a.contexts, a.gen, a.reps)
            g = run_gmb(a.model, a.contexts, a.reps)
        dt = time.time() - t0
        for n in sorted(set(g) | set(m)):
            gp, gd = g.get(n, (0.0, 0.0))
            mp, md = m.get(n, (0.0, 0.0))
            rows.append((name, order, n, gp, mp, gd, md))
            print(f"  [{name} {order}] N={n:6d}  P gmb={gp:7.0f} mlx_lm={mp:8.0f} "
                  f"({gp/mp if mp else 0:.3f}x)  D gmb={gd:6.1f} mlx_lm={md:7.1f} "
                  f"({gd/md if md else 0:.3f}x)", flush=True)
        print(f"  [{name} {order}] fatto in {dt:.0f}s", flush=True)

    if a.out:
        with open(a.out, "a") as f:
            for r in rows:
                f.write("\t".join(str(x) for x in r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
