#!/usr/bin/env python3
"""capped_run.py — esegue un comando (es. un test/binario DS4) SOTTO i cap di sicurezza memoria,
riusando gli helper di ds4_ab_decode.py (RSS albero <=62GB, free >=floor, pre-check, kill-tree).
Serve per far girare binari che NON hanno un monitor di memoria proprio (es. i test ds4).

Uso: capped_run.py [floor=30.0] [rss_cap=62.0] [pre=60.0] -- <cmd> [args...]
Stampa stdout+stderr del comando, poi il riepilogo [rc=.. peakRSS=.. minLibera=.. why=..].
"""
import os, subprocess, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds4_ab_decode as H   # noqa: E402


def main():
    args = sys.argv[1:]
    if "--" in args:
        i = args.index("--"); head = args[:i]; cmd = args[i + 1:]
    else:
        head, cmd = [], args
    floor = float(head[0]) if len(head) > 0 else 30.0
    rsscap = float(head[1]) if len(head) > 1 else 62.0
    pre = float(head[2]) if len(head) > 2 else 60.0
    to = 2400
    if not cmd:
        print("uso: capped_run.py [floor] [rss_cap] [pre] -- <cmd> [args...]"); return 2
    f0 = H.free_pct()
    print(f"[capped_run] floor>={floor} rss_cap={rsscap} pre>={pre} | libera {f0:.0f}% | cmd={' '.join(cmd)}", flush=True)
    if f0 < pre:
        print("ABORT: memoria libera bassa. Non lancio."); return 2
    if not H.wait_recover(pre):
        return 2
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    why, peak, minfree = {"v": None}, {"rss": 0.0}, {"f": 100.0}
    stop = threading.Event()

    def rd():
        for raw in iter(p.stdout.readline, b""):
            print(raw.decode("utf-8", "replace").rstrip("\n"), flush=True)

    def mon():
        t0 = time.time()
        while not stop.is_set() and p.poll() is None:
            rss, f = H.tree_rss_gb(p.pid), H.free_pct()
            peak["rss"] = max(peak["rss"], rss); minfree["f"] = min(minfree["f"], f)
            if rss > rsscap:            why["v"] = f"RSS {rss:.1f}>{rsscap}"
            elif f < floor:             why["v"] = f"libera {f:.0f}<{floor}"
            elif time.time() - t0 > to: why["v"] = "timeout"
            if why["v"]:
                H.kill_tree(p.pid); return
            stop.wait(0.5)

    ths = [threading.Thread(target=t, daemon=True) for t in (rd, mon)]
    for t in ths: t.start()
    try:
        p.wait(timeout=to + 60)
    except subprocess.TimeoutExpired:
        H.kill_tree(p.pid); p.wait()
    finally:
        stop.set()
        for t in ths: t.join(timeout=5)
    H.kill_tree(p.pid)
    print(f"[capped_run rc={p.returncode} peakRSS={peak['rss']:.0f}GB minLibera={minfree['f']:.0f}% why={why['v']}]", flush=True)
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())
