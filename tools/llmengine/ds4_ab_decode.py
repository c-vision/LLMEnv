#!/usr/bin/env python3
"""ds4_ab_decode.py — A/B ALTERNATA vendored vs ds4-slow via fastllm, CON MEMORY CAP OBBLIGATORIO.

SICUREZZA (obbligatoria — lezione 2026-10-06: un run DS4 senza cap ha fatto riavviare la macchina).
Cap allineati al protocollo provato dall'altra AI (RSS 55 GB + soglia memoria libera 35%):
- PRE-CHECK: se la memoria LIBERA (reclamabile = free+inactive+speculative) < FREE_PCT% del totale, NON parte.
- Durante OGNI run: un monitor thread uccide (SIGKILL) se (a) l'RSS del processo fastllm > RSS_CAP_GB, oppure
  (b) la memoria libera scende sotto FREE_PCT% del totale.
- TRA le due gambe: ricontrolla.
- TIMEOUT per-run.
Nessun run del modello parte senza questi controlli.

Uso: ds4_ab_decode.py [N=3] [max_new=48] [free_pct=35] [rss_cap_gb=55]
"""
import subprocess, re, statistics, sys, time, threading

FL = "/Users/gaetano/Documents/Projects/FastLLM/build-release/fastllm"
P = ("Explain in detail the history of France, covering the Roman conquest of Gaul, "
     "the Merovingian and Carolingian dynasties, the French Revolution, and the modern republic.")

def sys_total_gb():
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout.strip()
        return int(out) / 1e9
    except Exception:
        return 128.0

def sys_avail_gb():
    """memoria LIBERA reclamabile: free + inactive + speculative."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    m = re.search(r"page size of (\d+) bytes", out); pg = int(m.group(1)) if m else 4096
    def pages(tag):
        mm = re.search(rf"{tag}:\s+(\d+)", out); return int(mm.group(1)) if mm else 0
    return (pages("Pages free") + pages("Pages inactive") + pages("Pages speculative")) * pg / 1e9

def avail_pct():
    return 100.0 * sys_avail_gb() / max(sys_total_gb(), 1e-9)

def proc_rss_gb(pid):
    try:
        p = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout.strip()
        return (int(p) / 1048576.0) if p.isdigit() else 0.0   # rss e' in KB
    except Exception:
        return 0.0

def run(slow, mt, to, free_pct, rss_cap):
    if avail_pct() < free_pct:
        print(f"  ABORT: memoria libera {avail_pct():.1f}% < {free_pct}%. Non lancio.", flush=True)
        return None
    cmd = [FL, "run", "ds4", P, "--max-tokens", str(mt), "--offload-experts", "--opt", "temperature=0"]
    if slow: cmd += ["--opt", "ds4_backend=slow"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    killed = {"why": None}
    def monitor():
        t0 = time.time()
        while p.poll() is None:
            rss = proc_rss_gb(p.pid); ap = avail_pct()
            if rss > rss_cap:  killed["why"] = f"RSS {rss:.1f}GB > {rss_cap}GB"; p.kill(); return
            if ap < free_pct:  killed["why"] = f"libera {ap:.1f}% < {free_pct}%";   p.kill(); return
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
FREE_PCT = float(sys.argv[3]) if len(sys.argv) > 3 else 35.0
RSS_CAP  = float(sys.argv[4]) if len(sys.argv) > 4 else 55.0
TO = 420
print(f"[ds4_ab_decode] N={N} max_new={MT} free>={FREE_PCT}% rss_cap={RSS_CAP}GB | libera {avail_pct():.1f}% ({sys_avail_gb():.1f}GB)")
if avail_pct() < FREE_PCT + 5.0:
    print(f"ABORT: libera {avail_pct():.1f}% < {FREE_PCT+5}%. Non lancio."); sys.exit(2)

v, s = [], []
for i in range(N):
    # ORDINE ALTERNATO (bilanciato): il carico colpisce entrambi nella stessa finestra
    if i % 2 == 0: a = run(False, MT, TO, FREE_PCT, RSS_CAP); b = run(True, MT, TO, FREE_PCT, RSS_CAP)
    else:          b = run(True, MT, TO, FREE_PCT, RSS_CAP); a = run(False, MT, TO, FREE_PCT, RSS_CAP)
    print(f"rep{i+1} ({'v,s' if i%2==0 else 's,v'}): vendored={a}  slow={b}  [libera={avail_pct():.1f}%]", flush=True)
    if a: v.append(a)
    if b: s.append(b)
mv = statistics.median(v) if v else None
ms = statistics.median(s) if s else None
print(f"\nvendored median = {mv}  slow median = {ms}  rapporto = {mv/ms if (mv and ms) else None}")
