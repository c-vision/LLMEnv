#!/usr/bin/env python3
"""kv_append_probe.py — l'append della KV cache e' O(pos) per ALLOCAZIONE/COPIA?

Contesto (dense-long, tracker §2/§4/§8): il decode a contesto lungo e' dominato dallo
scaling O(pos) del KV. Fatti a sorgente:
  - `mx.concatenate` -> `concatenate_gpu` fa SEMPRE `out.set_data(allocator::malloc(...))`
    e copia ogni input (`mlx/backend/metal/slicing.cpp:26`): nessun path di donation.
  - `mx.slice_update` -> `SliceUpdate : UnaryPrimitive` (in-place-eligible) ma
    `DynamicSliceUpdate::eval_gpu` fa `copy_gpu(in, out)`: se `out` condivide il buffer
    di `in` (donation, refcount==1) la copia e' no-op -> O(1); altrimenti e' O(N).

Misura model-free (nessun modello): per una KV [B,H,N,D] bf16, il costo di UN append:
  A) CONCAT      : concat([k(N), nuovo(1)])            -> output N+1
  B) SLICE_PREALL: slice_update(kbuf(Nmax), nuovo, N)  -> output Nmax (buffer preallocato)
  C) SLICE_ALIAS : come B ma con un alias trattenuto di kbuf (refcount>1 -> no donation)
Metrica: ms/append e GB/s effettivi (byte logici / tempo) al crescere di N.

Lettura: se A cresce ~linearmente con N -> il concat e' O(pos) (alloc+copy). Se B e' ~
PIATTO in N -> slice_update dona il buffer (in-place, O(1)); se B cresce con Nmax -> e'
funzionale (ricopia Nmax). C isola l'effetto del refcount.
Model-free, load-light. Uso: <venv py> tools/kv_append_probe.py [--nmax 16384] [--reps 100]
"""
import argparse
import sys
import time

import mlx.core as mx

MB = 1024.0 * 1024.0


def med_time(fn, reps, warmup=10):
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def seq(N0, N1, step, B, H, D, dt, reps):
    """Append SEQUENZIALE realistico: cresce da N0 a N1, un token per passo.
    Cattura il churn di allocazione a dimensione crescente (che il probe a N fisso
    NON vede). Ritorna (ms/step_concat, ms/step_slice) misurati sull'ultima finestra."""
    def run_concat():
        k = mx.zeros([B, H, N0, D], dtype=dt)
        new = mx.zeros([B, H, 1, D], dtype=dt)
        mx.eval(k, new)
        t0 = time.perf_counter()
        for _ in range(N0, N1, step):
            k = mx.concatenate([k, new], axis=2)
            mx.eval(k)
        return (time.perf_counter() - t0)

    def run_slice():
        kbuf = mx.zeros([B, H, N1, D], dtype=dt)
        new = mx.zeros([B, H, 1, D], dtype=dt)
        mx.eval(kbuf, new)
        t0 = time.perf_counter()
        for n in range(N0, N1, step):
            kbuf = mx.slice_update(kbuf, new, mx.array([0, 0, n, 0], dtype=mx.int32),
                                   axes=[0, 1, 2, 3])
            mx.eval(kbuf)
        return (time.perf_counter() - t0)

    steps = len(range(N0, N1, step))
    tc = min(run_concat() for _ in range(reps)) / steps * 1e3
    ts = min(run_slice() for _ in range(reps)) / steps * 1e3
    return tc, ts


def step_batched(L, N, B, H, D, dt, reps, method):
    """UN passo di decode realistico: L append K/V LAZY + UN solo eval. Misura ms/passo.
    method='concat' | 'slice' (buffer preallocato Nmax)."""
    Nmax = N if method == "slice" else N
    ks = [mx.zeros([B, H, N, D], dtype=dt) for _ in range(L)]
    new = [mx.zeros([B, H, 1, D], dtype=dt) for _ in range(L)]
    if method == "slice":
        Nmax = max(N, 16384)
        ks = [mx.zeros([B, H, Nmax, D], dtype=dt) for _ in range(L)]
    start = mx.array([0, 0, N, 0], dtype=mx.int32)
    mx.eval(*ks, *new)
    if method == "concat":
        def f():
            return [mx.concatenate([ks[i], new[i]], axis=2) for i in range(L)]
    else:
        def f():
            return [mx.slice_update(ks[i], new[i], start, axes=[0, 1, 2, 3])
                    for i in range(L)]
    for _ in range(5):
        mx.eval(f())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(f())
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--h", type=int, default=8)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--nmax", type=int, default=16384)
    ap.add_argument("--ns", default="256,1024,4096,16384")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seq", default="", help="N0,N1,step: append sequenziale crescente")
    ap.add_argument("--layers", type=int, default=0, help="L: passo a L layer, 1 eval")
    args = ap.parse_args()

    B, H, D, Nmax = args.b, args.h, args.d, args.nmax

    if args.layers:
        L = args.layers
        print(f"# PASSO a L={L} layer, UN eval/passo (concat K di ogni layer) "
              f"[B={B},H={H},D={D}]")
        print(f"# {'N':>7} | {'CONCAT ms/step':>14} | {'SLICE_PRE ms/step':>17} | "
              f"{'slice/concat':>12}")
        for N in [int(s) for s in args.ns.split(",")]:
            tc = step_batched(L, N, B, H, D, mx.bfloat16, args.reps, "concat")
            ts = step_batched(L, N, B, H, D, mx.bfloat16, args.reps, "slice")
            print(f"  {N:>7} | {tc*1e3:14.4f} | {ts*1e3:17.4f} | {ts/tc:12.3f}")
        return 0

    if args.seq:
        N0, N1, step = (int(s) for s in args.seq.split(","))
        row = B * H * D * 2
        print(f"# SEQUENZIALE bf16 [B={B},H={H},D={D}] da N={N0} a N={N1} (step={step})")
        for reps in (1, 3):
            tc, ts = seq(N0, N1, step, B, H, D, mx.bfloat16, reps)
            print(f"  reps={reps}: CONCAT {tc:8.4f} ms/step | "
                  f"SLICE_PRE {ts:8.4f} ms/step | slice/concat={ts/tc:.3f}")
        print(f"  (byte/token={row} B; un append concat medio sposta "
              f"~2*{row}*N B, crescente)")
        return 0
    ns = [int(s) for s in args.ns.split(",")]
    dt = mx.bfloat16
    row = B * H * D * 2  # byte di UN token per K (o V)

    print(f"# KV append bf16 [B={B},H={H},N,D={D}] Nmax={Nmax}  (byte/token={row} B)")
    print(f"# {'N':>7} | {'CONCAT ms':>10} {'GB/s':>7} | "
          f"{'SLICE_PRE ms':>12} {'GB/s':>7} | {'SLICE_ALIAS ms':>14} {'GB/s':>7}")

    for N in ns:
        k = mx.random.normal([B, H, N, D], dtype=mx.float32).astype(dt)
        new = mx.random.normal([B, H, 1, D], dtype=mx.float32).astype(dt)
        kbuf = mx.zeros([B, H, Nmax, D], dtype=dt)
        mx.eval(k, new, kbuf)

        # A) concat: legge N+1, scrive N+1 (alloca output N+1)
        tc = med_time(lambda: mx.concatenate([k, new], axis=2), args.reps)
        gc = (2 * (N + 1) * row) / tc / 1e9

        # B) slice_update su buffer preallocato Nmax, offset N
        start = mx.array([0, 0, N, 0], dtype=mx.int32)
        tb = med_time(lambda: mx.slice_update(kbuf, new, start, axes=[0, 1, 2, 3]),
                      args.reps)
        gb = (2 * (N + 1) * row) / tb / 1e9  # byte UTILI (se in-place)

        # C) come B ma con un alias trattenuto (refcount>1 -> donation persa)
        alias = kbuf
        tcal = med_time(lambda: mx.slice_update(alias, new, start, axes=[0, 1, 2, 3]),
                        args.reps)
        gcal = (2 * (N + 1) * row) / tcal / 1e9
        assert alias is kbuf

        print(f"  {N:>7} | {tc*1e3:10.4f} {gc:7.1f} | "
              f"{tb*1e3:12.4f} {gb:7.1f} | {tcal*1e3:14.4f} {gcal:7.1f}")

    print("# Lettura: CONCAT che cresce ~N => O(pos) alloc+copy. "
          "SLICE_PRE piatto in N => in-place O(1); se cresce con Nmax => funzionale.")


if __name__ == "__main__":
    sys.exit(main())
