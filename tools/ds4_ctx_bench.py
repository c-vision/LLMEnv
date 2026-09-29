#!/usr/bin/env python3
"""Bench DS4 (nostro engine via fastllm) per frontiere di contesto: prefill t/s e decode tok/s separati.
Metodo: A (prompt corto, 1 tok) = costo di load; B (prompt di ~N token, 1 tok) = load+prefill;
C (GEN tok): il `tok/s` stampato da fastllm e' il decode (senza prefill). prefill = ctx/(B-A).
Uso: ds4_ctx_bench.py [--gen 48] [--csv out.csv] N1 N2 ...   (N = parole del prompt, ~1.3 token/parola)
CSV confrontabile con `ds4-bench` (ctx_tokens,prefill_tps,gen_tps). Registra il load 1-min: il wall-time
sotto carico e' inaffidabile, per confronti fini usare i tempi GPU (cb.GPUEndTime-GPUStartTime) temporanei."""
import subprocess, sys, time, re, os, random, argparse
FL = '/Users/gaetano/Documents/Projects/FastLLM/build-release/fastllm'
W = "il gatto cane casa sole luna mare monte fiume strada libro tempo uomo donna città vento pioggia notte giorno amico storia lavoro mondo acqua fuoco terra cielo verde rosso grande piccolo veloce lento antico nuovo".split()
def run(prompt, mt):
    t = time.time()
    p = subprocess.run([FL, 'run', '0', prompt, '--max-tokens', str(mt), '--offload-experts'], capture_output=True, text=True)
    dt = time.time() - t
    o = p.stdout + p.stderr
    m = re.search(r'contesto (\d+) tok', o); r = re.search(r'([\d.]+) tok/s', o)
    return dt, int(m.group(1)) if m else -1, float(r.group(1)) if r else 0.0
ap = argparse.ArgumentParser(); ap.add_argument('--gen', type=int, default=48); ap.add_argument('--csv'); ap.add_argument('n', type=int, nargs='+')
a = ap.parse_args()
load, _, _ = run('Ciao', 1)
rows = []
for n in a.n:
    random.seed(7); prompt = ' '.join(random.choice(W) for _ in range(n))
    l1 = os.getloadavg()[0]
    tb, ctx, _ = run(prompt, 1)
    _, _, dec = run(prompt, a.gen)
    pre = ctx / max(tb - load, 1e-3)
    rows.append((ctx, round(pre, 2), a.gen, round(dec, 2), round(l1, 2)))
    print('ctx=%d prefill=%.1f t/s decode=%.2f tok/s load=%.2f' % (ctx, pre, dec, l1), flush=True)
if a.csv:
    open(a.csv, 'w').write('ctx_tokens,prefill_tps,gen_tokens,gen_tps,load1\n' + '\n'.join(','.join(map(str, r)) for r in rows) + '\n')
