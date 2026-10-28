#!/usr/bin/env python3
"""moe_mxfp4_floor_probe.py — la "floor di dispatch per-call" della gather_qmm mxfp4
esiste DAVVERO nel path batched (lazy) o e' un artefatto del sync-per-chiamata?

Contesto: `tools/moe_mxfp4_gather_probe.py` misura ~0.14 ms/chiamata PIATTI in K e
dtype-invarianti, ma fa `mx.eval()` DENTRO il loop -> ogni iterazione paga un sync GPU
completo. L'engine reale costruisce un grafo lazy per token e fa UN eval: le ~144 gather
di Coder-Next (48 layer x 3) finiscono in pochi command buffer, senza sync per gather.
Se la floor sparisce in batched, un "kernel fuso mono-dispatch" non serve.

Misura, per la stessa forma:
  A) PER-CALL SYNC  : encode 1 gather + eval        -> ms/chiamata (metodo del probe vecchio)
  B) BATCHED        : encode R gather + UN eval     -> ms/chiamata (metodo dell'engine)
  C) TOKEN BATCHED  : encode R gather + 1 eval      -> ms/token (R = 48*3, come Coder-Next)
Stampa anche GB/s effettivi per giudicare bandwidth vs latenza.

Uso: <py> tools/moe_mxfp4_floor_probe.py [--dims coder] [--K 10] [--reps 50] [--R 144]
"""
import argparse
import time

import mlx.core as mx

PRESETS = {
    "gptoss": dict(E=128, Din=2880, Dff=2880, K=4),
    "coder":  dict(E=512, Din=2048, Dff=512,  K=10),
}


def quant(E, Din, Dff, dtype):
    w = mx.random.normal([E, Dff, Din], dtype=dtype)
    wq, ws = mx.quantize(w, group_size=32, bits=4, mode="mxfp4")
    mx.eval(wq, ws)
    return wq, ws


def per_call_sync(wq, ws, K, Din, dtype, reps, warmup=5):
    x = mx.random.normal([K, 1, Din], dtype=dtype)
    rhs = mx.arange(K, dtype=mx.uint32)
    mx.eval(x, rhs)
    def one():
        r = mx.gather_qmm(x, wq, ws, rhs_indices=rhs, transpose=True,
                          group_size=32, bits=4, mode="mxfp4", sorted_indices=False)
        mx.eval(r)
    for _ in range(warmup):
        one()
    t0 = time.perf_counter()
    for _ in range(reps):
        one()
    return (time.perf_counter() - t0) / reps


def batched(wq, ws, K, Din, dtype, R, warmup=3):
    """R gather encodate in UN grafo, un solo eval. Ritorna ms/totale e ms/call."""
    x = mx.random.normal([K, 1, Din], dtype=dtype)
    rhs = mx.arange(K, dtype=mx.uint32)
    mx.eval(x, rhs)
    def run():
        acc = None
        for _ in range(R):
            r = mx.gather_qmm(x, wq, ws, rhs_indices=rhs, transpose=True,
                              group_size=32, bits=4, mode="mxfp4", sorted_indices=False)
            acc = r if acc is None else acc + r
        mx.eval(acc)
    for _ in range(warmup):
        run()
    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        run()
    dt = (time.perf_counter() - t0) / reps
    return dt, dt / R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="coder", choices=list(PRESETS))
    ap.add_argument("--K", type=int, default=0, help="0 = top_k del preset")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--R", type=int, default=144, help="gather per token (48 layer x 3)")
    a = ap.parse_args()
    p = PRESETS[a.dims]
    K = a.K or p["K"]
    dt = mx.float32
    wq, ws = quant(p["E"], p["Din"], p["Dff"], dt)
    be = (p["Dff"] * p["Din"] * 0.5) + (p["Dff"] * p["Din"] / 32.0)  # 1 proiezione
    print(f"# preset={a.dims} E={p['E']} Din={p['Din']} Dff={p['Dff']} K={K} R={a.R}")
    print(f"# bytes/proiezione ~ {be/1e6:.2f} MB (pesi mxfp4 + scale e8m0)")
    per_sync = per_call_sync(wq, ws, K, p["Din"], dt, a.reps)
    tot_b, per_b = batched(wq, ws, K, p["Din"], dt, a.R)
    gbs_b = (K * be) / per_b / 1e9
    print(f"  A) PER-CALL SYNC  {per_sync*1e3:9.3f} ms/call")
    print(f"  B) BATCHED {a.R:>3} call {per_b*1e3:9.3f} ms/call   ({tot_b*1e3:.3f} ms/tot)")
    print(f"  A/B = {per_sync/per_b:6.1f}x  -> se >>1 la 'floor' e' il sync, non il kernel")
    print(f"  banda effettiva batched: {gbs_b:6.1f} GB/s (1 proiezione per call, K={K} esperti)")


if __name__ == "__main__":
    main()
