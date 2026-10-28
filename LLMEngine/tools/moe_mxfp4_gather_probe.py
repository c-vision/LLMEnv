#!/usr/bin/env python3
"""Probe DIAGNOSTICO (nonShipping): gather_qmm mxfp4 a T=1 -- e' bandwidth-bound o
latency/dispatch-bound?

⚠️ METODOLOGIA SUPERATA (2026-09-17) — LEGGERE PRIMA. Questo probe fa `mx.eval()` DENTRO
il loop di misura: ogni chiamata paga un SYNC GPU completo (~0,14 ms), quindi il "floor
piatto ~0,14 ms/call, dtype-invariante" che stampa e' in gran parte ARTEFATTO DEL SYNC, non
del kernel. Il path reale (engine e mlx_lm) e' lazy: ~144 gather/token in pochi command
buffer, UN eval. Misura corretta (batched): `tools/moe_mxfp4_floor_probe.py` -> ~0,021 ms/call
(7,5x meno), 144 gather = ~3,0 ms/token a ~264 GB/s, f16~=f32. => NON c'e' una floor di
dispatch mxfp4 da aggredire con un kernel fuso; il presente probe resta utile solo come
stima GROSSOLANA bandwidth-vs-K in regime sincrono.

Motivo (#36): il kernel MoE fuso GMB (llm_mlx_moe_fused_kernel.h) e' affine-only, quindi
tutti i grandi MoE mxfp4 (122B, gpt-oss, Coder-Next) girano su mlx_gather_qmm. Prima di
impegnare una sessione dedicata a scrivere un kernel fuso mxfp4-native (payoff ignoto),
misuriamo la FORMA di scalabilita' di gather_qmm mxfp4 al variare di K (numEsperti attivi)
a T=1 (decode):

  - se il tempo per chiamata e' ~PIATTO in K  -> c'e' un FLOOR di latenza/dispatch per
    chiamata che domina: un kernel fuso che fa tutti gli K esperti in UN dispatch paga GROSSO.
  - se il tempo scala ~LINEARE con K -> e' bandwidth-bound sui pesi (gli mxfp4 pesi sono il
    costo): un kernel fuso non puo' battere il ceiling di bandwidth -> payoff ~nullo.

Costruisce tensori mxfp4 SYNTHETIC con mx.quantize (layout garantito corretto da MLX), quindi
NON serve nessun checkpoint. RAM-safe (pochi MB). Eseguito sotto carico condivido, gli ASSOLUTI
sono gonfiati ma la FORMA (rapporto K=n/K=1, GB/s vs BW-macchina) regge perche' e' A/B entro lo
stesso processo.

Uso:  <venv py> tools/moe_mxfp4_gather_probe.py [--dims gptoss|122b|coder] [--dtypes f32,f16]
                                                    [--reps 50] [--ks 1,2,4,8,16]
"""
import argparse
import sys
import time

import mlx.core as mx


# (n_experts, hidden=Din, moe_intermediate=Dff, top_k_ref) -- approssimati dai config noti.
PRESETS = {
    "gptoss": dict(E=128, Din=2880, Dff=2880, K=4, tag="gpt-oss-120b"),
    "122b":   dict(E=256, Din=3072, Dff=1024, K=8, tag="Qwen3.5-122B-A10B"),
    "coder":  dict(E=512, Din=2048, Dff=512,  K=8, tag="Qwen3-Coder-Next"),
}

MB = 1024.0 * 1024.0


def mxfp4_bytes_per_expert(Din, Dff):
    """Pesi mxfp4 (0.5 B/valore) + scale e8m0 (1 B ogni 32 valori), per UNA proiezione
    [Dff, Din]. gate e up sono [Dff, Din], down e' [Din, Dff]: stesso numero di valori."""
    weights = Dff * Din * 0.5
    scales = Dff * (Din / 32.0) * 1.0
    return weights + scales


def quantize_stack(E, Din, Dff, dtype):
    w_dense = mx.random.normal([E, Dff, Din], dtype=dtype)
    wq, ws = mx.quantize(w_dense, group_size=32, bits=4, mode="mxfp4")
    mx.eval(wq, ws)
    return wq, ws


def time_gather(wq, ws, K, Din, Dff, dtype, reps, warmup=10):
    x = mx.random.normal([K, 1, Din], dtype=dtype)
    rhs = mx.arange(K, dtype=mx.uint32)  # esperti 0..K-1 (scelti), non-sortato
    mx.eval(x, rhs)
    for _ in range(warmup):
        r = mx.gather_qmm(
            x, wq, ws, rhs_indices=rhs, transpose=True,
            group_size=32, bits=4, mode="mxfp4", sorted_indices=False,
        )
        mx.eval(r)
    t0 = time.perf_counter()
    for _ in range(reps):
        r = mx.gather_qmm(
            x, wq, ws, rhs_indices=rhs, transpose=True,
            group_size=32, bits=4, mode="mxfp4", sorted_indices=False,
        )
        mx.eval(r)
    dt = (time.perf_counter() - t0) / reps
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="gptoss", choices=list(PRESETS))
    ap.add_argument("--dtypes", default="f32")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--ks", default="1,2,4,8,16")
    args = ap.parse_args()

    p = PRESETS[args.dims]
    ks = [int(s) for s in args.ks.split(",")]
    dtmap = {"f32": mx.float32, "f16": mx.float16}
    dts = [dtmap[s.strip()] for s in args.dtypes.split(",") if s.strip() in dtmap]

    print(f"# preset={p['tag']}  E={p['E']} Din={p['Din']} Dff={p['Dff']} top_k_ref={p['K']}")
    be = mxfp4_bytes_per_expert(p["Din"], p["Dff"])
    print(f"# mxfp4 bytes/expert/proiezione ~ {be/MB:.2f} MB (pesi+scales e8m0)")
    print("# Per-token gpt-oss sposta gate+up+down = 3 proiezioni x K esperti. K_ref tempo atteso:")
    print(f"#   3*{p['K']}*{be/MB:.2f}MB = {3*p['K']*be/MB:.1f} MB / token (soli pesi esperti)\n")

    for dtype in dts:
        print(f"== dtype x = {str(dtype)} ==")
        wq, ws = quantize_stack(p["E"], p["Din"], p["Dff"], dtype)
        base = None
        for K in ks:
            dt = time_gather(wq, ws, K, p["Din"], p["Dff"], dtype, args.reps)
            gbs = (K * be) / dt / 1e9  # GB/s effettivi (una sola proiezione)
            ratio = dt / base if base else 1.0
            if base is None:
                base = dt
            print(f"  K={K:>2}  {dt*1e3:8.3f} ms/call   {gbs:6.1f} GB/s   ratio_vs_K1={ratio:5.2f}")
        print()

    print("Lettura: se ratio_vs_K1 cresce ~linearmente con K e' bandwidth-bound (fuso ~inutile).")
    print("Se i ms sono ~piatti (ratio<<K) e GB/s << BW-macchina (~centinaia GB/s su M-series),")
    print("c'e' un floor di dispatch che un kernel fuso mono-dispatch puo' aggredire.")


if __name__ == "__main__":
    sys.exit(main())
