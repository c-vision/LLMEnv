#!/usr/bin/env python3
"""dsmin_ab.py — A/B ALTERNATA engine-only (driver dsmin2): vendored vs ds-slow, stessa ABI/loop/prompt.

VALIDITA' (correzioni dal feedback di un'altra AI):
- l'eos e' ESPLICITO (default 1 = eos reale DS4). Con eos=-1 si forza N token fissi (stesso lavoro
  forzato); con eos=1 i due motori si FERMANO (e contano l'EOS diversamente: il vendored lo include,
  il slow no).
- NON si confronta solo il tok/s: si verificano anche CONTEGGIO e IDENTITA' dei token (prefisso comune).
  Un A/B di velocita' e' valido solo se i due motori fanno lo STESSO lavoro (stessi token).
Uso: dsmin_ab.py [N=3] [max_new=16] [eos=-1|1]
"""
import subprocess, re, statistics, sys
ROOT = "/Users/gaetano/Documents/Projects/LLMEnv/LLMEngine"
MDL = "/Users/gaetano/ds4/ds4flash.gguf"
IDS = "0,3476,477,260,11502,22896,16,9361,3898,278,891,603,128803,3085,344,270,6102,294,8760,33,128804"

def run(which, maxn, eos):
    if which == "vendored":
        exe, cwd = ROOT + "/ds-minimal/dsmin2", ROOT + "/ds4engine"   # kernel Metal da sorgente -> CWD ds4engine
    else:
        exe, cwd = ROOT + "/ds-minimal/dsmin2", ROOT + "/ds-minimal"
    p = subprocess.run([exe, which, MDL, str(maxn), IDS, f"eos={eos}"], cwd=cwd, capture_output=True, text=True, timeout=300)
    o = (p.stdout or "") + (p.stderr or "")
    ts = re.search(r"([\d.]+) tok/s", o)
    ct = re.search(r"\] (\d+) tok", o)
    idm = re.search(r"\[ids-\w+\] ([0-9,]*)", o)
    ids = [int(x) for x in idm.group(1).split(",") if x] if idm else []
    return (float(ts.group(1)) if ts else None, int(ct.group(1)) if ct else None, ids)

def prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
MT = int(sys.argv[2]) if len(sys.argv) > 2 else 16
EOS = int(sys.argv[3]) if len(sys.argv) > 3 else -1
v, s, vc, sc, vids, sids = [], [], [], [], [], []
for i in range(N):
    if i % 2 == 0: a = run("vendored", MT, EOS); b = run("slow", MT, EOS)
    else:          b = run("slow", MT, EOS);     a = run("vendored", MT, EOS)
    print(f"rep{i+1}: vendored={a[0]} tok_s, n={a[1]}, ids={a[2][:6]}  |  slow={b[0]} tok_s, n={b[1]}, ids={b[2][:6]}", flush=True)
    if a[0]: v.append(a[0]); vc.append(a[1]); vids.append(a[2])
    if b[0]: s.append(b[0]); sc.append(b[1]); sids.append(b[2])
mv = statistics.median(v) if v else None
ms = statistics.median(s) if s else None
print(f"\n(eos={EOS}) vendored median={mv}  slow median={ms}  rapporto={mv/ms if (mv and ms) else None}")
print(f"conteggi: vendored={vc}  slow={sc}  (uguali={vc==sc})")
if vids and sids:
    pl = min(prefix(vids[0], sids[0]), len(vids[0]), len(sids[0]))
    print(f"IDENTITA': prefisso comune vendored/slow = {pl} token su (v={len(vids[0])}, s={len(sids[0])})")
    print(f"  vendored[0..]: {vids[0][:pl+4]}")
    print(f"  slow[0..]:     {sids[0][:pl+4]}")
