#!/usr/bin/env python
"""GMBasic-vs-mlx_lm benchmark — lato mlx_lm (reference eager).

Misura PREFILL (prompt_tps, dal forward sui token di contesto) e DECODE
(generation_tps, token dopo il primo) a contesto N, con warm-up separato
(kernel Metal + grafo) e ripetizioni (mediana). Stesso protocollo di
tools/ab_engine_mlxlm.sh (entrambi gli ordini, finestra quiete).

Output parseable (una riga per N):
    N=<n> prefill <pf> tok/s decode <dc> tok/s

Uso:
    bench_mlxlm.py --model <dir> [--lengths 4096,8192] [--max-tokens 64] [--repeats 2]
"""
import argparse
import os
import statistics
import time
from importlib import import_module

from mlx_lm import load

stream_generate = import_module("mlx_lm.generate").stream_generate


def run_once(model, tok, ids, ngen):
    last = None
    t0 = time.perf_counter()
    ttft = None
    for resp in stream_generate(model, tok, ids, max_tokens=ngen):
        if ttft is None:
            ttft = time.perf_counter() - t0
        last = resp
    total = time.perf_counter() - t0
    return last, ttft, total


def measure(model, tok, n, ngen, reps):
    ids = [(i % 1000) + 1 for i in range(n)]
    # Warm-up separato (metal kernel cache + grafo lazy) — NON contato.
    run_once(model, tok, ids, 2)
    pfs, dcs = [], []
    per_rep = os.environ.get("GMB_BENCH_PERREP")  # stampa i valori per-rep (spread robusto)
    for _ in range(reps):
        last, ttft, _total = run_once(model, tok, ids, ngen)
        pf = getattr(last, "prompt_tps", 0.0) or (n / ttft if ttft else 0.0)
        dc = getattr(last, "generation_tps", 0.0)
        pfs.append(float(pf))
        dcs.append(float(dc))
        if per_rep:
            print(f"REP n={n} prefill {pf:.1f} tok/s decode {dc:.1f} tok/s", flush=True)
    return statistics.median(pfs), statistics.median(dcs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="4096")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()
    lengths = [int(x) for x in a.lengths.split(",") if x.strip()]
    model, tok = load(a.model, tokenizer_config={"trust_remote_code": True})
    # EOS DISABILITATO: il canary GMB (`test_mlx_bench decodectx`) chiama
    # session_generate(..., eos_token_id=-1) -> genera SEMPRE tutti gli M token. Senza
    # questo, stream_generate si ferma al primo EOS e `generation_tps` e' calcolato su
    # meno token: baseline NON equivalente (fix allineato a `mlx_lm.benchmark`, che fa
    # `tokenizer._eos_token_ids = {}`).
    tok._eos_token_ids = {}
    print(f"# loaded {a.model}", flush=True)
    for n in lengths:
        pf, dc = measure(model, tok, n, a.max_tokens, a.repeats)
        print(f"N={n} prefill {pf:.1f} tok/s decode {dc:.1f} tok/s", flush=True)


if __name__ == "__main__":
    main()
