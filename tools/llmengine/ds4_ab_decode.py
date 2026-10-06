#!/usr/bin/env python3
"""ds4_ab_decode.py — A/B ALTERNATA vendored vs ds4-slow via fastllm, CON MEMORY CAP OBBLIGATORIO.

SICUREZZA (obbligatoria — lezione 2026-10-06: un run DS4 senza cap ha fatto riavviare la macchina):
- PRE-CHECK: se la memoria RECLAMABILE di sistema (free+inactive+speculative) e' sotto FLOOR_GB, NON parte.
- Durante OGNI run: un monitor thread controlla (a) l'RSS del processo fastllm -> SIGKILL sopra HARD_GB;
  (b) la memoria reclamabile del sistema -> SIGKILL sotto FLOOR_GB. Non si scende mai sotto il floor.
- TRA le due gambe: ricontrolla il floor.
- TIMEOUT per-run.
Nessun run del modello parte senza questi controlli.

Uso: ds4_ab_decode.py [N=3] [max_new=48] [floor_gb=6] [hard_gb=96]
"""
import subprocess, re, statistics, sys, time, threading, os, signal

FL = "/Users/gaetano/Documents/Projects/FastLLM/build-release/fastllm"
P = ("Explain in detail the history of France, covering the Roman conquest of Gaul, "
     "the Merovingian and Carolingian dynasties, the French Revolution, and the modern republic.")

def sys_avail_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    m = re.search(r"page size of (\d+) bytes", out); pg = int(m.group(1)) if m else 4096
    def pages(tag):
        mm = re.search(rf"{tag}:\s+(\d+)", out); return int(mm.group(1)) if mm else 0
    return (pages("Pages free") + pages("Pages inactive") + pages("Pages speculative")) * pg / 1e9

def proc_rss_gb(pid):
    try:
        p = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout.strip()
        return (int(p) / 1048576.0) if p.isdigit() else 0.0   # rss e' in KB
    except Exception:
        return 0.0

def run(slow, mt, to, floor_gb, hard_gb):
    cmd = [FL, "run", "ds4", P, "--max-tokens", str(mt), "--offload-experts", "--opt", "temperature=0"]
    if slow: cmd += ["--opt", "ds4_backend=slow"]
    if sys_avail_gb() < floor_gb:
        print(f"  ABORT: memoria reclamabile {sys_avail_gb():.1f} GB < floor {floor_gb} GB. Non lancio.", flush=True)
        return None
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    killed = {"why": None}
    def monitor():
        t0 = time.time()
        while p.poll() is None:
            rss = proc_rss_gb(p.pid); av = sys_avail_gb()
            if rss > hard_gb: killed["why"] = f"RSS {rss:.1f}GB > hard {hard_gb}GB"; p.kill(); return
            if av < floor_gb: killed["why"] = f"sys-avail {av:.1f}GB < floor {floor_gb}GB"; p.kill(); return
            if time.time() - t0 > to: killed["why"] = f"timeout {to}s"; p.kill(); return
            time.sleep(1.0)
    th = threading.Thread(target=monitor, daemon=True); th.start()
    try: o, e = p.communicate(timeout=to + 60)
    except subprocess.TimeoutExpired: p.kill(); o, e = p.communicate()
    th.join(timeout=2)
    if killed["why"]:
        print(f"  KILL: {killed['why']}", flush=True); return None
    m = re.findall(r"([\d.]+) tok/s", (o or "") + (e or ""))
    return float(m[-1]) if m else None

N  = int(sys.argv[1]) if len(sys.argv) > 1 else 3
MT = int(sys.argv[2]) if len(sys.argv) > 2 else 48
FLOOR = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
HARD  = float(sys.argv[4]) if len(sys.argv) > 4 else 96.0
TO = 420
print(f"[ds4_ab_decode] N={N} max_new={MT} floor={FLOOR}GB hard={HARD}GB | avail={sys_avail_gb():.1f}GB")
if sys_avail_gb() < FLOOR + 2.0:
    print(f"ABORT: avail {sys_avail_gb():.1f}GB < floor+2 ({FLOOR+2}). Non lancio."); sys.exit(2)

v, s = [], []
for i in range(N):
    # ORDINE ALTERNATO (bilanciato): il carico colpisce entrambi nella stessa finestra
    if i % 2 == 0: a = run(False, MT, TO, FLOOR, HARD); b = run(True, MT, TO, FLOOR, HARD)
    else:          b = run(True, MT, TO, FLOOR, HARD); a = run(False, MT, TO, FLOOR, HARD)
    print(f"rep{i+1} ({'v,s' if i%2==0 else 's,v'}): vendored={a}  slow={b}  [avail={sys_avail_gb():.1f}GB]", flush=True)
    if a: v.append(a)
    if b: s.append(b)
mv = statistics.median(v) if v else None
ms = statistics.median(s) if s else None
print(f"\nvendored median = {mv}  slow median = {ms}  rapporto = {mv/ms if (mv and ms) else None}")
