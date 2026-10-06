#!/usr/bin/env python3
"""ds4_ab_decode.py — A/B ALTERNATO vendored vs ds4-slow (decode), valido sotto QUALSIASI carico.

Metodo (registro DS4-Decode): alternare vendored e slow NELLA STESSA finestra -> il carico colpisce
entrambi allo stesso modo -> il RAPPORTO e la mediana sono affidabili senza finestra quieta.
NON serve una macchina scarica; NON usare confronti fra run lontani nel tempo.

Uso: ds4_ab_decode.py [N=3] [max_tokens=32]
Stampa: tok/s per rep (vendored, slow), mediane, rapporto.
"""
import subprocess, re, statistics, sys, os
FL = "/Users/gaetano/Documents/Projects/FastLLM/build-release/fastllm"
# Prompt LUNGO (~50 token): il fix dell'arena rende ds4-slow eseguibile su prompt reali; cosi' i due
# engine fanno lo stesso lavoro sostanzioso (prima il slow crashava a >=16 token -> A/B monche).
P = ("Explain in detail the history of France, covering the Roman conquest of Gaul, "
     "the Merovingian and Carolingian dynasties, the French Revolution, and the modern republic.")

def run(slow, mt, to):
    cmd = [FL, "run", "ds4", P, "--max-tokens", str(mt), "--offload-experts", "--opt", "temperature=0"]
    if slow: cmd += ["--opt", "ds4_backend=slow"]   # default (f16 OFF, hc GPU ON): path batch corretto
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=to)
    except subprocess.TimeoutExpired as e:
        return None
    o = (p.stdout or "") + (p.stderr or "")
    m = re.findall(r"([\d.]+) tok/s", o)
    return float(m[-1]) if m else None

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MT = int(sys.argv[2]) if len(sys.argv) > 2 else 64
TO = int(os.environ.get("RUN_TIMEOUT", "420"))  # solo watchdog; non e' un parametro di comportamento
v, s = [], []
for i in range(N):
    # ORDINE BILANCIATO: rep pari -> vendored primo, rep dispari -> slow primo (rimuove il bias d'ordine)
    if i % 2 == 0:
        a = run(False, MT, TO); b = run(True, MT, TO)
    else:
        b = run(True, MT, TO); a = run(False, MT, TO)
    print(f"rep{i+1} ({'v,s' if i%2==0 else 's,v'}): vendored={a}  slow={b}", flush=True)
    if a: v.append(a)
    if b: s.append(b)
mv = statistics.median(v) if v else None
ms = statistics.median(s) if s else None
print(f"\nvendored median = {mv}  (n={len(v)})  min={min(v) if v else None} max={max(v) if v else None}")
print(f"slow     median = {ms}  (n={len(s)})  min={min(s) if s else None} max={max(s) if s else None}")
print(f"rapporto vendored/slow = {mv/ms if (mv and ms) else None}  (>1 = vendored piu' veloce)")
