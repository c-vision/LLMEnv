#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_rapidmlx — misura ATTENDIBILE di rapidMLX (server vLLM-MLX) per il confronto GMBasic.

Semantica (allineata alle misure dirette "oneste" della doc GMBasic-vs-RapidMLX-Comparison.md):
  - prefill tok/s = N / TTFT   (TTFT = tempo fino al primo token, streaming; include ~1 tok decode)
  - decode  tok/s = M / (t_last - t_first)  (dopo il prefill di N, streaming di M token)
Solo numero per completions non-chat (/v1/completions), prompt deterministico di esattamente N token.

Uso:
  # avvio io del server (consigliato) e misuro:
  python3 tools/bench_rapidmlx.py --model <dir> --serve --port 8011 \
        --lengths 4096 --max-tokens 64 --repeats 3
  # server gia' avviato:
  python3 tools/bench_rapidmlx.py --url http://127.0.0.1:8011/v1 --lengths 4096 --max-tokens 64

Logga la memoria disponibile (free+inactive+speculare+purgeable) prima delle misure e registra
anche il server-model id. Fairness: usare gli stessi N di test_mlx_bench GMBasic, a macchina
fredda e interleaved (vedi protocollo §13 della doc).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse

try:
    import requests
except Exception:  # pragma: no cover
    sys.exit("serve requests: pip install requests (env di sviluppo, non runtime GMBasic)")


def free_mem_gb() -> float:
    """RAM disponibile in GB = (free+inactive+speculare+purgeable)*page/1e9 (lezione memoria)."""
    try:
        out = subprocess.check_output(["vm_stat"]).decode()
        m = __import__("re").search(r"page size of (\d+) bytes", out)
        page = int(m.group(1)) if m else 16384
        d = {}
        for line in out.splitlines():
            line = line.strip().rstrip(".")
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            if not k.startswith("Pages"):
                continue
            d[k.strip()] = int(v.replace(",", "").split()[0])
        tot = (d.get("Pages free", 0) + d.get("Pages inactive", 0)
               + d.get("Pages speculative", 0) + d.get("Pages purgeable", 0)) * page / 1e9
        return round(tot, 1)
    except Exception:
        return float("nan")


def build_prompt_n_tokens(tokenizer, n: int) -> str:
    """Testo deterministico che il tokenizer codifica in (circa) n token."""
    unit = "The quick brown fox jumps over the lazy dog. "
    ids = tokenizer(unit).input_ids
    ids = (ids * ((n // len(ids)) + 1))[:n]
    return tokenizer.decode(ids)


def main():
    ap = argparse.ArgumentParser(description="Benchmark prefill+decode rapido e onesto di rapidMLX")
    ap.add_argument("--model", help="cartella modello MLX (serve se --serve)")
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--serve", action="store_true", help="avvia 'rapid-mlx serve' io stesso")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--served-name", default=None, help="served-model-name (default = nome cartella)")
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096], help="N token di prefill")
    ap.add_argument("--max-tokens", type=int, default=64, help="M token per decode")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    proc = None
    if args.serve:
        if not args.model or not os.path.isdir(args.model):
            sys.exit("--serve richiede --model <dir>")
        rapid = shutil.which("rapid-mlx")
        if not rapid:
            sys.exit("rapid-mlx non trovato (attiva il venv di sviluppo, es. ~/ai/env312)")
        name = args.served_name or os.path.basename(args.model.rstrip("/"))
        cmd = [rapid, "serve", args.model, "--served-model-name", name, "--port", str(args.port),
                "--disable-prefix-cache"]
        print(f"[bench] avvio server: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        base = f"http://127.0.0.1:{args.port}/v1"
    else:
        base = args.url

    # wait ready
    ok = False
    for _ in range(120):
        try:
            r = requests.get(base + "/models", timeout=5)
            if r.ok:
                ok = True
                break
        except Exception:
            pass
        time.sleep(1)
    if not ok:
        if proc:
            proc.kill()
        sys.exit("[bench] server non pronto")
    models = [m["id"] for m in r.json().get("data", [])]
    model_id = args.served_name or models[0]
    print(f"[bench] server pronto; model id: {model_id}")

    # tokenizer
    try:
        from transformers import AutoTokenizer
    except Exception:
        if proc:
            proc.kill()
        sys.exit("serve transformers (env di sviluppo)")
    tok = AutoTokenizer.from_pretrained(args.model or models[0], trust_remote_code=True)

    mem0 = free_mem_gb()
    print(f"[bench] RAM disponibile all'inizio: {mem0} GB")
    results = {"model": model_id, "mem_gb": mem0, "lengths": {}, "notes": []}

    headers = {"Content-Type": "application/json"}
    for N in args.lengths:
        prompt = build_prompt_n_tokens(tok, N)
        n_tok = len(tok(prompt).input_ids)
        pref = []
        for rep in range(args.repeats):
            t0 = time.time()
            first_seen = None
            last_seen = None
            got_text = []
            with requests.post(base + "/completions", headers=headers, stream=True,
                               json={"model": model_id, "prompt": prompt,
                                     "max_tokens": args.max_tokens, "stream": True,
                                     "temperature": 0}, timeout=args.timeout) as resp:
                if not resp.ok:
                    results["notes"].append(f"N={N}: HTTP {resp.status_code} {resp.text[:200]}")
                    break
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        txt = chunk["choices"][0].get("text") or ""
                    except Exception:
                        continue
                    now = time.time()
                    if first_seen is None and txt:
                        first_seen = now
                    if txt:
                        last_seen = now
                        got_text.append(txt)
            if first_seen is None:
                results["notes"].append(f"N={N} rep{rep}: nessun token (stream vuoto)")
                break
            ttft = first_seen - t0
            if last_seen and len(got_text) > 1:
                dec = len(got_text) / max(last_seen - first_seen, 1e-9)
            else:
                dec = float("nan")
            pref.append({"prefill_tok_s": round(N / ttft, 1), "ttft_s": round(ttft, 4),
                         "decode_tok_s": round(dec, 1), "chunks": len(got_text)})
            print(f"[bench] N={N} rep{rep}: prefill {N/ttft:.0f} tok/s (TTFT {ttft:.2f}s) "
                  f"decode {dec:.1f} tok/s chunks={len(got_text)}")
        if pref:
            results["lengths"][N] = pref

    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    out = json.dumps(results, indent=1)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
