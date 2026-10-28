#!/usr/bin/env python3
"""mlx_decode_ref.py — riferimento mlx_lm per il DECODE (simmetrico a mlx_prefill_ref.py).

Misura, con lo stesso protocollo del nostro motore (greedy, prompt fisso):
  - TTFT (time to first token, = prefill+1 token) ed END-TO-END (P+N token);
  - DECODE derivato = (total - ttft) / (N - 1)  [ms/token, steady state].
Il decode derivato e' confrontabile col "wall" del nostro gmb_decode_bench (che misura
il solo loop di decode, escluso il prefill).

Uso:
  <venv>/bin/python mlx_decode_ref.py <model_dir> <prompt> <n_tokens> [reps]
Richiede mlx_lm (venv ~/ai/env-mlxbench per i modelli recenti).
"""
import sys, time
import mlx.core as mx
import mlx_lm


def main():
    if len(sys.argv) < 4:
        print("uso: mlx_decode_ref.py <model_dir> <prompt> <n_tokens> [reps]", file=sys.stderr)
        return 2
    path, prompt, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    model, tok = mlx_lm.load(path)
    # greedy: temp 0 (argmax) -> confrontabile col nostro decode greedy
    sampler = mlx_lm.sample_utils.make_sampler(temp=0.0)

    best = None
    for r in range(reps + 1):  # la prima e' warm-up (scartata)
        mx.synchronize()
        t0 = time.perf_counter()
        first_at = None
        nout = 0
        for resp in mlx_lm.stream_generate(model, tok, prompt=prompt, max_tokens=n, sampler=sampler):
            nout += 1
            if first_at is None:
                first_at = time.perf_counter()
            # consuma fino alla fine
        t1 = time.perf_counter()
        if r == 0:
            continue
        ttft = (first_at - t0) * 1000.0
        total = (t1 - t0) * 1000.0
        dec = (total - ttft) / max(1, (nout - 1))
        if best is None or total < best[0]:
            best = (total, ttft, dec, nout)
    total, ttft, dec, nout = best
    print(f"MLXLM_DECODE n={n} tok_out={nout} ttft={ttft:.1f} ms total={total:.1f} ms "
          f"decode={dec:.3f} ms/token ({1000.0/dec:.1f} tok/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
