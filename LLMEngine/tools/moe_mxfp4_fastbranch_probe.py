#!/usr/bin/env python3
"""moe_mxfp4_fastbranch_probe.py — la selezione `gather_qmv_fast` dipende da K % 512.

Ipotesi (verificata a sorgente, `mlx/backend/metal/quantized.cpp:1078`):
    bool fast = N % bn == 0 && K % 512 == 0;   // bn=8
con `K = x.shape(-1)` = dimensione INTERNA della matrice (`quantized.cpp:1571`),
NON il numero di esperti attivi. Nel decode (M<limit) il MoE usa `gather_qmv`
(transpose=True). Config locali:
  - gpt-oss-120b : Din=Dff=2880 -> K=2880, 2880 % 512 = 320 != 0 -> ramo GENERICO
  - Qwen3.5-122B : K=3072 / 1024                    -> ramo FAST
  - Coder-Next   : K=2048 / 512                      -> ramo FAST

Test: STESSA matrice mxfp4 (K=2880) vs variante paddata a 3072 con ZERI, con i
blocchi di quantizzazione (group_size=32) preservati sui primi K/32 gruppi.
  1) verifica numerica PRIMA: out(2880) vs out(3072-paddata) sui N canali reali;
  2) timing interleaved generico vs fast; metrica = GB/s (bytes/tempo).
Kill: se il ramo fast recupera MOLTO piu' del 6,7% di byte extra del padding
allora la penalita' e' di shape -> specializzare la coda 2880 / layout a load.
Se il rapporto tempo ~= 1,067 (byte) o peggio, la pista e' morta.

Model-free (nessun checkpoint), RAM-safe (pochi MB), rapporti interleaved nello
stesso processo -> validi anche sotto carico (load>10). Uso:
  <venv py> tools/moe_mxfp4_fastbranch_probe.py [--n 2880] [--pad 192]
      [--experts 4] [--reps 50] [--mode mxfp4|affine]
"""
import argparse
import sys
import time

import mlx.core as mx

MB = 1024.0 * 1024.0


def q_bytes(N, K, mode):
    """Byte dei pesi + scale per UNA matrice [N, K]."""
    if mode == "mxfp4":
        return N * K * 0.5 + N * (K / 32.0) * 1.0
    return N * K * 0.5 + N * (K / 32.0) * 2.0  # affine: scale+biases


def make(N, K, mode):
    w = mx.random.normal([N, K], dtype=mx.float32)
    if mode == "mxfp4":
        wq, ws = mx.quantize(w, group_size=32, bits=4, mode="mxfp4")
    else:
        wq, ws, wb = mx.quantize(w, group_size=32, bits=4)
        ws = (ws, wb)
    mx.eval(wq, *(ws if isinstance(ws, tuple) else (ws,)))
    return wq, ws


def pad_weights(wq, ws, N, K, pad, mode):
    """Aggiunge `pad` colonne di zeri (gruppi interi, group_size=32) preservando i
    blocchi originali. Ritorna (wq_pad, ws_pad). La larghezza impacchettata dipende
    dal packing (4-bit -> 8 valori/word): la deduco da wq.shape[-1]/K."""
    pq = wq.shape[-1] * pad // K          # colonne packed da aggiungere
    g = pad // 32                         # gruppi di scala da aggiungere
    if mode == "mxfp4":
        # wq: [N, K/8] packed uint32 ; ws: [N, K/32] e8m0
        zq = mx.zeros([N, pq], dtype=wq.dtype)
        zs = mx.zeros([N, g], dtype=ws.dtype)
        return mx.concatenate([wq, zq], axis=-1), mx.concatenate([ws, zs], axis=-1)
    wq_pad = mx.concatenate([wq, mx.zeros([N, pq], dtype=wq.dtype)], axis=-1)
    s, b = ws
    s_pad = mx.concatenate([s, mx.zeros([N, g], dtype=s.dtype)], axis=-1)
    b_pad = mx.concatenate([b, mx.zeros([N, g], dtype=b.dtype)], axis=-1)
    return wq_pad, (s_pad, b_pad)


def gather(x, wq, ws, rhs, mode):
    if mode == "mxfp4":
        return mx.gather_qmm(x, wq, ws, rhs_indices=rhs, transpose=True,
                             group_size=32, bits=4, mode="mxfp4",
                             sorted_indices=False)
    s, b = ws
    return mx.gather_qmm(x, wq, s, b, rhs_indices=rhs, transpose=True,
                         group_size=32, bits=4, sorted_indices=False)


def timeit(fn, reps, warmup=10):
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def sweep(N, ks, E, reps, mode):
    """Sweep di K a N fisso: GB/s per K, marcando fast (K%512==0) vs generic."""
    print(f"# SWEEP N={N} mode={mode} B={E}  (fast <=> K%512==0)")
    x_by_k = {}
    for K in ks:
        wq, ws = make(N, K, mode)
        x = mx.random.normal([E, 1, K], dtype=mx.float32)
        rhs = mx.arange(E, dtype=mx.uint32)
        mx.eval(x, rhs)
        t = timeit(lambda: gather(x, wq, ws, rhs, mode), reps)
        b = q_bytes(N, K, mode)
        tag = "FAST" if K % 512 == 0 else "GEN "
        print(f"   K={K:>5} {tag}  {t*1e3:8.4f} ms  {b/t/1e9:7.1f} GB/s  "
              f"({b/MB:5.2f} MB)")
        del wq, ws, x
        mx.clear_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2880, help="dim out N (Dff)")
    ap.add_argument("--pad", type=int, default=192, help="padding di K (192 -> 3072)")
    ap.add_argument("--experts", type=int, default=4, help="esperti attivi (batch B)")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--mode", default="mxfp4", choices=["mxfp4", "affine"])
    ap.add_argument("--sweep", default="", help="lista K separati da virgola (sweep)")
    args = ap.parse_args()

    if args.sweep:
        ks = [int(s) for s in args.sweep.split(",")]
        sweep(args.n, ks, args.experts, args.reps, args.mode)
        return 0

    N = args.n
    K = N  # gpt-oss: Din=Dff=2880 -> K interna = 2880
    Kp = K + args.pad
    E = args.experts

    print(f"# mode={args.mode} N={N} K={K} (generic: K%512={K % 512}) "
          f"-> Kp={Kp} (fast: Kp%512={Kp % 512}) B(esperti)={E}")
    print(f"# byte/matrice: K={q_bytes(N, K, args.mode)/MB:.3f} MB  "
          f"Kp={q_bytes(N, Kp, args.mode)/MB:.3f} MB  "
          f"ratio={q_bytes(N, Kp, args.mode)/q_bytes(N, K, args.mode):.4f}")

    wq, ws = make(N, K, args.mode)
    wq_p, ws_p = pad_weights(wq, ws, N, K, args.pad, args.mode)

    x = mx.random.normal([E, 1, K], dtype=mx.float32)
    xp = mx.concatenate([x, mx.zeros([E, 1, args.pad], dtype=mx.float32)], axis=-1)
    rhs = mx.arange(E, dtype=mx.uint32)
    mx.eval(x, xp, rhs)

    out = gather(x, wq, ws, rhs, args.mode)
    outp = gather(xp, wq_p, ws_p, rhs, args.mode)
    d = mx.max(mx.abs(out.astype(mx.float32) - outp.astype(mx.float32)))
    mx.eval(d)
    print(f"# parita' out(2880) vs out(3072-paddata): maxabsdiff={float(d):.3e} "
          f"(atteso ~0: padding a zeri, blocchi preservati)")

    fa = lambda: gather(x, wq, ws, rhs, args.mode)
    fb = lambda: gather(xp, wq_p, ws_p, rhs, args.mode)
    for _ in range(10):
        mx.eval(fa(), fb())
    tsa, tsb = [], []
    for _ in range(args.reps):
        t0 = time.perf_counter(); mx.eval(fa()); t1 = time.perf_counter()
        mx.eval(fb()); t2 = time.perf_counter()
        tsa.append(t1 - t0); tsb.append(t2 - t1)
    tsa.sort(); tsb.sort()
    ta = tsa[len(tsa) // 2]; tb = tsb[len(tsb) // 2]
    ga = q_bytes(N, K, args.mode) / ta / 1e9
    gb = q_bytes(N, Kp, args.mode) / tb / 1e9
    print(f"   GENERIC K={K}: {ta*1e3:8.4f} ms  {ga:7.1f} GB/s  "
          f"[p10-p90 {tsa[len(tsa)//10]*1e3:.3f}-{tsa[9*len(tsa)//10]*1e3:.3f}]")
    print(f"   FAST    K={Kp}: {tb*1e3:8.4f} ms  {gb:7.1f} GB/s  "
          f"[p10-p90 {tsb[len(tsb)//10]*1e3:.3f}-{tsb[9*len(tsb)//10]*1e3:.3f}]")
    print(f"   tempo fast/generic = {tb/ta:.4f}  (byte ratio = "
          f"{q_bytes(N, Kp, args.mode)/q_bytes(N, K, args.mode):.4f})")
    print(f"   GB/s fast/generic  = {gb/ga:.4f}  (>1 = il fast rende di piu' "
          f"del byte extra)")
    if tb / ta < 1.0:
        print("   => il ramo FAST vince NETTO nonostante +6,7% byte: "
              "specializzare la coda 2880 / layout a load ha senso.")
    elif gb / ga > 1.05:
        print("   => il FAST rende piu' per byte, ma il padding non ripaga "
              "del tutto a K=3072; valutare layout che eviti i byte extra.")
    else:
        print("   => nessun vantaggio di shape oltre i byte: pista morta.")


if __name__ == "__main__":
    sys.exit(main())
