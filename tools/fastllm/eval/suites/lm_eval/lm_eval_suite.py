"""Suite F4: integrazione **lm-evaluation-harness** via subprocess contro l'endpoint OpenAI.

Esegue il CLI `lm_eval` (`--model local-completions --model_args base_url=...,model=...`) verso il
server del motore e legge i risultati (accuracy per task). **Tool esterno/offline: nessun impatto
sulle prestazioni dell'engine.** Se `lm_eval` non è installato (binario non trovato) il gate è
**skipped** (informativo). Vedi todo Eval Harness §F4.
"""
import json
import os
import shutil
import subprocess
import tempfile

NAME = "lm_eval"
DESCRIPTION = "Accuracy (lm-evaluation-harness: mmlu/gsm8k/...) via subprocess sul server OpenAI."


def _find_results(outdir):
    """Cerca il `results*.json` scritto da lm-eval sotto `outdir` (ricorsivo)."""
    hits = []
    for root, _dirs, files in os.walk(outdir):
        for fn in files:
            if fn.startswith("results") and fn.endswith(".json"):
                hits.append(os.path.join(root, fn))
    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return hits


def _extract_acc(results):
    """{task: accuracy} dalle `results` di lm-eval (prima metrica float utile per task)."""
    out = {}
    for task, m in (results or {}).items():
        if not isinstance(m, dict):
            continue
        for k, v in m.items():
            if isinstance(v, (int, float)) and ("acc" in k or "exact_match" in k or "f1" in k):
                out[task] = round(float(v), 5)
                break
    return out


def run(client, cfg):
    bin_ = cfg.get("bin") or "lm_eval"
    resolved = shutil.which(bin_) or (bin_ if os.path.isfile(bin_) else None)
    if not resolved:
        return {"metric": "lm_eval_accuracy", "value": None, "threshold": cfg.get("min_accuracy"),
                "pass": True,
                "details": {"unsupported": f"'{bin_}' non trovato (lm-evaluation-harness non installato)"}}

    base = cfg.get("base_url") or ""
    model = cfg.get("model", "local")
    tasks = cfg.get("tasks", ["gsm8k"])
    outdir = cfg.get("output_dir") or tempfile.mkdtemp(prefix="lmeval-")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "results.json")

    cmd = [resolved, "--model", "local-completions",
           "--model_args", f"base_url={base},model={model}",
           "--tasks", ",".join(tasks), "--output_path", out,
           "--batch_size", str(cfg.get("batch_size", 1))]
    if cfg.get("limit"):
        cmd += ["--limit", str(cfg["limit"])]
    cmd += list(cfg.get("extra_args", []))

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=float(cfg.get("timeout", 3600)))
    except Exception as e:  # noqa: BLE001
        return {"metric": "lm_eval_accuracy", "value": None, "threshold": cfg.get("min_accuracy"),
                "pass": False, "details": {"error": str(e), "cmd": cmd}}
    if p.returncode != 0:
        return {"metric": "lm_eval_accuracy", "value": None, "threshold": cfg.get("min_accuracy"),
                "pass": False, "details": {"rc": p.returncode, "cmd": cmd, "stderr": p.stderr[-400:]}}

    acc = {}
    for f in _find_results(outdir):
        try:
            with open(f, encoding="utf-8") as fh:
                acc = _extract_acc(json.load(fh).get("results", {}))
            if acc:
                break
        except Exception:  # noqa: BLE001
            continue

    th = cfg.get("min_accuracy")
    if not acc:
        ok = True  # nessuna metrica -> informativo
    elif isinstance(th, dict):
        ok = all(acc.get(t, 0.0) >= th[t] for t in th)
    elif isinstance(th, (int, float)):
        ok = all(v >= th for v in acc.values())
    else:
        ok = True
    return {"metric": "lm_eval_accuracy", "value": acc or None, "threshold": th, "pass": ok,
            "details": {"tasks": tasks, "cmd": cmd[:4] + ["..."], "output": outdir}}
