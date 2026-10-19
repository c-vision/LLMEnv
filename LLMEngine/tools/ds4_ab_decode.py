#!/usr/bin/env python3
"""ds4_ab_decode.py — A/B ALTERNATA vendored vs ds4 via fastllm: PREFILL e DECODE separati, CON MEMORY CAP OBBLIGATORIO.

COSA MISURA (verificato sul sorgente fastllm, src/cli/fmlx_gen.c + src/engine/fmlx_engine_gen.c):
- DECODE: il `tok/s` stampato da fastllm = token_generati / tempo del loop `fmlx_engine_next`. Il PREFILL e' eseguito PRIMA
  (dentro `fmlx_engine_chat`: "START = crea sessione + prefill") e quindi NON e' compreso: e' decode puro, con la sola
  coda di avvio a cache esperti fredda dentro i primi passi.
- PREFILL: non e' stampato da fastllm. Lo si ricava cronometrando due eventi della STESSA run:
    t_ctx   = riga stderr `contesto: <N> token` (stampata dopo il caricamento del modello e PRIMA del prefill)
    t_first = primo byte su stdout (primo token generato, scaricato subito con fflush)
    prefill_s = (t_first - t_ctx) - 1/decode_tok_s            (si toglie il tempo del primo token)
    prefill_tok_s = token_del_prompt / prefill_s              (token_del_prompt = contesto - token_generati, dalla riga finale)
  E' il prefill di UN prompt corto (~39 token, percorso per-token del motore slow, a cache esperti fredda), NON il prefill a
  regime di prompt lunghi (>=192 token: percorso batch), che richiede un prompt lungo.

SICUREZZA (obbligatoria — lezione 2026-10-06: un run DS4 senza cap ha fatto riavviare la macchina).
Protocollo PROVATO (stesso di /tmp/fl.sh + wd2.sh usati nelle misure del 2026-10-06/07):
- Metrica memoria = "System-wide memory free percentage" di `memory_pressure -Q` (macOS).
  Se la metrica NON si legge, si assume 0% (fail-SAFE: non si parte / si uccide).
- PRE-CHECK a PRE_PCT (default 60%): sotto non si parte, si attende il recupero (max RECOVER_S) e poi si salta la run.
  Il carico del modello abbassa la metrica di 15-25 punti: partire gia' vicino al floor fa uccidere la run al
  primo campionamento (misurato: run avviate con ~40% libero sono morte a 34% con RSS 8 GB).
- Durante OGNI run un monitor (0.5 s) uccide l'INTERO ALBERO di processi (SIGKILL) se
  (a) la somma degli RSS dell'albero > RSS_CAP_GB (default 62; la run slow ha picchi 57-63 GB perche' la cache esperti si
      auto-dimensiona sulla memoria libera: per questo la slow usa `memgov_reserve_mb`, vedi SLOW_RESERVE_MB), oppure
  (b) la metrica libera scende sotto FLOOR_PCT (default 30%, concordato con l'utente), oppure (c) timeout per-run.
  NB: la gamba VENDORED (carica l'intero GGUF da 97 GB) porta la metrica a ~29-34% anche con RSS di soli 8 GB: con
  floor 35% veniva SEMPRE uccisa (misurato 2026-10-07, come le run f2/f3 del 2026-10-06). Se i cap la fermano per la sola
  metrica libera si ritenta (max 2 volte, dopo il recupero): nessun cap viene allentato.
- UNA run alla volta; tra le gambe si attende il recupero della memoria.
Nessun run del modello parte senza questi controlli.

Uso: ds4_ab_decode.py [N=3] [max_new=48] [floor_pct=30] [rss_cap_gb=62] [pre_pct=60]
"""
import os, re, signal, statistics, subprocess, sys, threading, time

FL = "/Users/gaetano/Documents/Projects/FastLLM/build-release/fastllm"
P = ("Explain in detail the history of France, covering the Roman conquest of Gaul, "
     "the Merovingian and Carolingian dynasties, the French Revolution, and the modern republic.")
RECOVER_S = 120          # attesa massima del recupero memoria tra una run e l'altra
# La cache esperti di ds4 si AUTO-DIMENSIONA sulla memoria libera (llm_dsv4_autoswap.h: 62-70 GB, picco RSS 61-63 GB
# variabile da run a run -> sfora il cap). `memgov_reserve_mb` (registro opzioni) lascia libera piu' memoria: cache minore,
# RSS ben sotto il cap. NON allenta alcun cap; riduce un poco la velocita' della slow. 0 = auto-dimensionamento originale.
SLOW_RESERVE_MB = 38000
TOK_RE = re.compile(r"([\d.]+) tok/s")
STAT_RE = re.compile(r"\[(\d+) tok\D+?([\d.]+) tok/s\D+?(\d+) tok\]")   # [48 tok · 7.2 tok/s · contesto 87 tok]
CTX_RE = re.compile(r"^contesto:\s*\d+\s*token", re.I)                    # marcatore: modello caricato, prefill non ancora iniziato


def free_pct():
    """'System-wide memory free percentage' (memory_pressure -Q). Errore di lettura -> 0.0 (fail-safe)."""
    try:
        out = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True, timeout=15).stdout
        m = re.search(r"free percentage:\s*(\d+)%", out)
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0


def tree_pids(root):
    """pid del processo e di TUTTI i discendenti + RSS (KB) di ciascuno, da un'unica `ps`."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return {root: 0}
    kids, rss = {}, {}
    for ln in out.splitlines():
        f = ln.split()
        if len(f) == 3 and all(x.isdigit() for x in f):
            pid, ppid, r = int(f[0]), int(f[1]), int(f[2])
            kids.setdefault(ppid, []).append(pid)
            rss[pid] = r
    seen, stack = {}, [root]
    while stack:
        p = stack.pop()
        if p in seen: continue
        seen[p] = rss.get(p, 0)
        stack.extend(kids.get(p, []))
    return seen


def tree_rss_gb(root):
    return sum(tree_pids(root).values()) / 1048576.0      # rss e' in KB


def kill_tree(root):
    for pid in tree_pids(root):
        try: os.kill(pid, signal.SIGKILL)
        except Exception: pass


def wait_recover(pre_pct):
    """Attende (max RECOVER_S) che la metrica torni >= pre_pct. True se ok."""
    t0 = time.time()
    while True:
        f = free_pct()
        if f >= pre_pct: return True
        if time.time() - t0 > RECOVER_S:
            print(f"  SKIP: memoria libera {f:.0f}% < {pre_pct:.0f}% dopo {RECOVER_S}s. Non lancio.", flush=True)
            return False
        time.sleep(3.0)


_FREE_KILL = object()


def run(slow, mt, to, floor_pct, rss_cap, pre_pct, retries=2):
    """Una gamba. Se i cap la fermano per la METRICA LIBERA (non per RSS/timeout) si ritenta (max `retries`), ogni volta
    dopo aver atteso il recupero a pre_pct. NESSUN cap viene allentato."""
    for attempt in range(retries + 1):
        r = _run_once(slow, mt, to, floor_pct, rss_cap, pre_pct)
        if r is not _FREE_KILL:
            return r
        print(f"  retry {attempt+1}/{retries} dopo kill per memoria libera", flush=True)
    return None


def _run_once(slow, mt, to, floor_pct, rss_cap, pre_pct):
    if not wait_recover(pre_pct):
        return None
    cmd = [FL, "run", "ds4", P, "--max-tokens", str(mt), "--offload-experts", "--opt", "temperature=0"]
    if slow:
        cmd += ["--opt", "ds4_backend=slow"]
        if SLOW_RESERVE_MB > 0: cmd += ["--opt", f"memgov_reserve_mb={SLOW_RESERVE_MB}"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)  # binario: timestamp esatti
    why, peak, minfree = {"v": None}, {"rss": 0.0}, {"f": 100.0}
    stop = threading.Event()
    ev = {"t_ctx": None, "t_first": None}
    out_buf, err_lines = [], []

    def read_out():                       # primo byte = primo token (fastllm fa fflush a ogni evento)
        while True:
            chunk = p.stdout.read1(4096)
            if not chunk: break
            if ev["t_first"] is None: ev["t_first"] = time.monotonic()
            out_buf.append(chunk)

    def read_err():                       # riga per riga, con timestamp del marcatore `contesto:`
        for raw in iter(p.stderr.readline, b""):
            ln = raw.decode("utf-8", "replace").rstrip("\n")
            if ev["t_ctx"] is None and CTX_RE.match(ln): ev["t_ctx"] = time.monotonic()
            err_lines.append(ln)

    def monitor():
        t0 = time.time()
        while not stop.is_set() and p.poll() is None:
            rss, f = tree_rss_gb(p.pid), free_pct()
            peak["rss"] = max(peak["rss"], rss); minfree["f"] = min(minfree["f"], f)
            if rss > rss_cap:           why["v"] = f"RSS albero {rss:.1f}GB > {rss_cap}GB"
            elif f < floor_pct:         why["v"] = f"libera {f:.0f}% < {floor_pct:.0f}%"
            elif time.time() - t0 > to: why["v"] = f"timeout {to}s"
            if why["v"]:
                kill_tree(p.pid); return
            stop.wait(0.5)

    ths = [threading.Thread(target=t, daemon=True) for t in (read_out, read_err, monitor)]
    for t in ths: t.start()
    try:
        p.wait(timeout=to + 60)
    except subprocess.TimeoutExpired:
        kill_tree(p.pid); p.wait()
    except KeyboardInterrupt:
        kill_tree(p.pid); raise
    finally:
        stop.set()
        for t in ths: t.join(timeout=5)
    kill_tree(p.pid)                    # nessun figlio orfano che tiene la memoria
    info = f"[rc={p.returncode} peakRSS={peak['rss']:.0f}GB minLibera={minfree['f']:.0f}%]"
    if why["v"]:
        print(f"  KILL: {why['v']} {info}", flush=True)
        return _FREE_KILL if why["v"].startswith("libera") else None
    txt = "\n".join(err_lines) + "\n" + b"".join(out_buf).decode("utf-8", "replace")
    sm = STAT_RE.findall(txt)
    m = TOK_RE.findall(txt)
    if not m:
        tail = "\n    ".join((err_lines or ["(stderr vuoto)"])[-4:])
        print(f"  NESSUN tok/s nell'output {info}\n    {tail}", flush=True); return None
    tps = float(sm[-1][1]) if sm else float(m[-1])
    ctok = int(sm[-1][0]) if sm else None
    prompt = (int(sm[-1][2]) - ctok) if sm else None
    pre_s = pre_tps = None
    if ev["t_ctx"] is not None and ev["t_first"] is not None and tps > 0:
        pre_s = (ev["t_first"] - ev["t_ctx"]) - 1.0 / tps          # tolto il tempo del primo token
        if pre_s > 0 and prompt: pre_tps = prompt / pre_s
    d = {"decode": tps, "ctok": ctok, "prompt": prompt, "prefill_s": pre_s, "prefill_tps": pre_tps}
    print(f"  ok {'slow' if slow else 'vendored'}: decode={tps} tok/s  prefill="
          f"{'n/d' if pre_s is None else f'{pre_s:.1f}s'} (prompt {prompt} tok) {info}", flush=True)
    return d


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(x, nd=2):
    return "n/d" if x is None else f"{x:.{nd}f}"


def main():
    N       = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    MT      = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    FLOOR   = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    RSS_CAP = float(sys.argv[4]) if len(sys.argv) > 4 else 62.0
    PRE     = float(sys.argv[5]) if len(sys.argv) > 5 else 60.0
    TO = 420
    if not os.access(FL, os.X_OK):
        print(f"ABORT: fastllm non trovato/eseguibile: {FL}"); sys.exit(2)
    f0 = free_pct()
    print(f"[ds4_ab_decode] N={N} max_new={MT} floor>={FLOOR:.0f}% pre>={PRE:.0f}% rss_cap={RSS_CAP}GB | libera {f0:.0f}%")
    if f0 < PRE:
        print(f"ABORT: libera {f0:.0f}% < {PRE:.0f}%. Non lancio."); sys.exit(2)

    V, S = [], []
    for i in range(N):
        # ORDINE ALTERNATO (bilanciato): il carico colpisce entrambi nella stessa finestra
        if i % 2 == 0: a = run(False, MT, TO, FLOOR, RSS_CAP, PRE); b = run(True, MT, TO, FLOOR, RSS_CAP, PRE)
        else:          b = run(True, MT, TO, FLOOR, RSS_CAP, PRE);  a = run(False, MT, TO, FLOOR, RSS_CAP, PRE)
        for tag, d in (("vendored", a), ("slow", b)):
            print(f"rep{i+1} {tag:8s}: " + ("n/d" if d is None else
                  f"decode={d['decode']:.2f} tok/s   prefill={fmt(d['prefill_tps'],1)} tok/s ({fmt(d['prefill_s'],1)} s, prompt {d['prompt']} tok)"), flush=True)
        print(f"        [libera={free_pct():.0f}%]", flush=True)
        if a: V.append(a)
        if b: S.append(b)

    def row(label, key, nd=2):
        mv, ms = med([d[key] for d in V]), med([d[key] for d in S])
        r = (ms / mv) if (mv and ms) else None
        print(f"  {label:30s} vendored={fmt(mv,nd):>8}   slow={fmt(ms,nd):>8}   slow/vendored={fmt(r,2):>6}")
    print(f"\n=== CONFRONTO (mediane su {len(V)} / {len(S)} rep, max_new={MT}) ===")
    row("DECODE (tok/s)", "decode")
    row("PREFILL prompt corto (tok/s)", "prefill_tps", 1)
    row("PREFILL prompt corto (s)", "prefill_s", 1)
    print("  (rapporto per le tok/s: >1 = slow piu' veloce; per i secondi: <1 = slow piu' veloce)")


if __name__ == "__main__":
    main()
