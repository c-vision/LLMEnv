#!/usr/bin/env python3
"""ds4_ab_host.py — A/B alternato: HOST fastllm vs ds4 standalone (STESSO engine ds4.c/ds4engine).

Obbiettivo: isolare l'overhead dell'host `fastllm` (LlmEngine wrapper) dall'engine, confrontando
`fastllm run ds4` con il CLI standalone `ds4` (stesso modello, stesso ds4.c). Ordine BILANCIATO
(rimuove il bias di warm-up). Valido sotto qualsiasi carico (rapporto, non valore assoluto).

Uso: ds4_ab_host.py [N=3] [tokens=64]
"""
import subprocess, re, statistics, sys
FL   = "/Users/gaetano/Documents/Projects/FastLLM/build-release/fastllm"
DS4  = "/Users/gaetano/Documents/Projects/LLMEnv/backups/ds4engine/ds4"
MDL  = "/Users/gaetano/ds4/ds4flash.gguf"
P    = "What is the capital of France?"
TO   = 400

def run_fastllm(n):
    p = subprocess.run([FL, "run", "ds4", P, "--max-tokens", str(n), "--offload-experts",
                        "--opt", "temperature=0"], capture_output=True, text=True, timeout=TO)
    o = (p.stdout or "") + (p.stderr or "")
    m = re.findall(r"([\d.]+) tok/s", o)
    return float(m[-1]) if m else None

def run_ds4(n):
    p = subprocess.run([DS4, "-m", MDL, "-p", P, "-n", str(n), "--ssd-streaming"],
                       capture_output=True, text=True, timeout=TO)
    o = (p.stdout or "") + (p.stderr or "")
    m = re.search(r"generation:\s*([\d.]+)\s*t/s", o)
    return float(m.group(1)) if m else None

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NT = int(sys.argv[2]) if len(sys.argv) > 2 else 64
fl, ds = [], []
for i in range(N):
    if i % 2 == 0:
        a = run_fastllm(NT); b = run_ds4(NT)
    else:
        b = run_ds4(NT); a = run_fastllm(NT)
    print(f"rep{i+1} ({'fl,ds' if i%2==0 else 'ds,fl'}): fastllm={a}  ds4_standalone={b}", flush=True)
    if a: fl.append(a)
    if b: ds.append(b)
mf = statistics.median(fl) if fl else None
md = statistics.median(ds) if ds else None
print(f"\nfastllm   median = {mf} (n={len(fl)}) min={min(fl) if fl else None} max={max(fl) if fl else None}")
print(f"ds4 stand median = {md} (n={len(ds)}) min={min(ds) if ds else None} max={max(ds) if ds else None}")
print(f"rapporto ds_standalone/fastllm = {md/mf if (mf and md) else None}  (>1 = fastllm ha overhead)")
