#!/usr/bin/env python3
"""ds4_prefill_phase.py — UN run del motore **ds4** su un prompt LUNGO con `dsv4_profile=1`,
per localizzare il costo del PREFILL BATCH (fase1 mHC+attn / MoE io-vs-gemm / fase3) e contare cb/wait.

Riusa i cap di sicurezza di `ds4_ab_decode.py` (RSS albero, floor memoria libera, pre-check): nessuna
run del modello parte senza. Nessun cap viene allentato.

Uso: ds4_prefill_phase.py [ntok=900] [max_new=1]
L'output del motore va su stderr: le righe `[dsv4-prof ...]`, `[dsv4-decode...]` vengono inoltrate.
"""
import os, subprocess, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds4_ab_decode as H   # noqa: E402  (free_pct, tree_rss_gb, kill_tree, wait_recover, FL, SLOW_RESERVE_MB)

_BASE = ("Explain in detail the history of France, covering the Roman conquest of Gaul, the Merovingian "
         "and Carolingian dynasties, the French Revolution, the modern republic, its wars and its culture. ")


def long_prompt(ntok):
    out = []
    while len(out) < ntok:
        out += _BASE.split()
    return " ".join(out[:ntok])


def main():
    ntok = int(sys.argv[1]) if len(sys.argv) > 1 else 900
    mt   = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    reserve = int(sys.argv[3]) if len(sys.argv) > 3 else H.SLOW_RESERVE_MB
    mmap  = int(sys.argv[4]) if len(sys.argv) > 4 else 0   # 1 = --opt dsv4_experts_mmap=1 (opzione 1: esperti dal mmap, no cache bundle)
    floor = float(sys.argv[5]) if len(sys.argv) > 5 else 30.0   # override SOLO per misure concordate (default: cap 30%)
    batch_dev = int(sys.argv[6]) if len(sys.argv) > 6 else 0   # 1 = --opt dsv4_batch_dev=1 (mHC device-resident)
    to = int(sys.argv[7]) if len(sys.argv) > 7 else 600        # timeout per-run (s)
    extra = sys.argv[8] if len(sys.argv) > 8 else ""            # opt extra k=v (es. offload_cache_mb=28000)
    rsscap, pre = 62.0, 60.0
    P = long_prompt(ntok)
    f0 = H.free_pct()
    print(f"[ds4_prefill_phase] prompt~{ntok} tok max_new={mt} mmap={mmap} batch_dev={batch_dev} floor={floor} to={to}s | libera {f0:.0f}%", flush=True)
    if f0 < pre:
        print("ABORT: memoria libera bassa. Non lancio."); return
    if not H.wait_recover(pre):
        return
    cmd = [H.FL, "run", "ds4", P, "--max-tokens", str(mt), "--offload-experts",
           "--opt", "temperature=0",
           "--opt", f"memgov_reserve_mb={reserve}", "--opt", "dsv4_profile=1"]
    if mmap: cmd += ["--opt", "dsv4_experts_mmap=1"]
    if batch_dev: cmd += ["--opt", "dsv4_batch_dev=1"]
    if extra: cmd += ["--opt", extra]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    why, peak, minfree = {"v": None}, {"rss": 0.0}, {"f": 100.0}
    stop = threading.Event()
    err, outb = [], []

    def rd():
        for raw in iter(p.stderr.readline, b""):
            err.append(raw.decode("utf-8", "replace").rstrip("\n"))

    def ro():
        while True:
            chunk = p.stdout.read1(4096)
            if not chunk:
                break
            outb.append(chunk)

    def mon():
        t0 = time.time()
        while not stop.is_set() and p.poll() is None:
            rss, f = H.tree_rss_gb(p.pid), H.free_pct()
            peak["rss"] = max(peak["rss"], rss); minfree["f"] = min(minfree["f"], f)
            if rss > rsscap:     why["v"] = f"RSS {rss:.1f}>{rsscap}"
            elif f < floor:      why["v"] = f"libera {f:.0f}<{floor}"
            elif time.time() - t0 > to: why["v"] = "timeout"
            if why["v"]:
                H.kill_tree(p.pid); return
            stop.wait(0.5)

    ths = [threading.Thread(target=t, daemon=True) for t in (rd, ro, mon)]
    for t in ths: t.start()
    try:
        p.wait(timeout=to + 60)
    except subprocess.TimeoutExpired:
        H.kill_tree(p.pid); p.wait()
    finally:
        stop.set()
        for t in ths: t.join(timeout=5)
    H.kill_tree(p.pid)
    print(f"[rc={p.returncode} peakRSS={peak['rss']:.0f}GB minLibera={minfree['f']:.0f}% why={why['v']}]", flush=True)
    for ln in err:
        if ln.lstrip().startswith("[dsv4") or "dsv4-prof" in ln or "dsv4-decode" in ln:
            print(ln)
    txt = b"".join(outb).decode("utf-8", "replace")
    print(f"[output generato, {len(txt)} char]: {txt[:300]!r}")


if __name__ == "__main__":
    main()
