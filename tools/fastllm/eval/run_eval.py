#!/usr/bin/env python3
"""run_eval.py — runner unico dell'Eval Harness (qualita'/fedelta').

Uso:
  python3 eval/run_eval.py --suite local --config eval/config/eval.json --report eval/reports
  python3 eval/run_eval.py --suite local --backend mock           # test senza motore

Suite `local` = gate DETERMINISTICI (id_parity, needle). Non usa giudici LLM.
Nessuna dipendenza dal build CMake; nessun import del motore (solo HTTP o mock).
Doc: 01-Projects/LLMEnv/Todo/(open) - LLM-Eval-Harness-Qualita-e-Allucinazioni.md.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapters.engine_server import make_client, EngineError, OpenAIClient  # noqa: E402

LOCAL_GATES = ["id_parity", "needle", "ppl", "parity"]


def load_config(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _gate_summary(report):
    return {g["gate"]: {"metric": g["metric"], "value": g["value"]} for g in report["gates"]}


def save_golden(report, path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"suite": report["suite"], "engine": report["engine"], "gates": _gate_summary(report)},
                  f, indent=2)


def compare_golden(report, golden, tol):
    """Confronta i gate della run col golden. bool = esatto; numerico = |Δ|/max(1,|b|) <= tol."""
    cur = _gate_summary(report)
    rows, ok = [], True
    for name, g in golden.get("gates", {}).items():
        b = g.get("value")
        if name not in cur:
            rows.append({"gate": name, "golden": b, "current": None, "ok": False, "why": "gate assente"})
            ok = False
            continue
        a = cur[name]["value"]
        if isinstance(a, bool) and isinstance(b, bool):
            good = (a == b)
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and a is not None and b is not None:
            good = abs(float(a) - float(b)) / max(1.0, abs(float(b))) <= tol
        else:
            good = (a == b)
        rows.append({"gate": name, "golden": b, "current": a, "ok": good})
        ok = ok and good
    return {"ok": ok, "tolerance": tol, "rows": rows}


def run_gate(modname, client, cfg, reps, ref_client=None):
    mod = importlib.import_module(f"suites.local.{modname}")
    gcfg = dict(cfg.get(modname, {}))
    gcfg["_ref_client"] = ref_client
    results = []
    for _ in range(max(1, reps)):
        results.append(mod.run(client, gcfg))
    passes = [r["pass"] for r in results]
    ok = all(passes)
    return {"gate": modname, "pass": ok,
            "value": results[0]["value"], "metric": results[0]["metric"],
            "threshold": results[0]["threshold"],
            "reps": reps, "runs": results}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Eval harness (qualita'/fedelta').")
    ap.add_argument("--suite", default="local", choices=["local", "lm-eval", "halogen"])
    ap.add_argument("--verifier-backend", default=None, help="local|openai (default: da config)")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "config", "eval.json"))
    ap.add_argument("--report", default=None, help="cartella di output (default: <eval>/reports)")
    ap.add_argument("--backend", default=None, help="override: mock|http")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--golden", default=None, help="confronta i gate col report di riferimento (regression)")
    ap.add_argument("--save-golden", default=None, help="salva il report corrente come golden")
    ap.add_argument("--golden-tol", type=float, default=0.05, help="tolleranza relativa per i gate numerici")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.backend:
        cfg.setdefault("engine", {})["backend"] = args.backend
    if args.base_url:
        cfg.setdefault("engine", {})["base_url"] = args.base_url
    if args.model:
        cfg.setdefault("engine", {})["model"] = args.model
    le = cfg.setdefault("lm_eval", {})
    if args.base_url:
        le["base_url"] = args.base_url
    if args.model:
        le["model"] = args.model

    report_dir = args.report or os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(report_dir, exist_ok=True)

    try:
        client = make_client(cfg)
    except EngineError as e:
        print(f"[eval] ERRORE client: {e}", file=sys.stderr)
        return 2

    ref_cfg = cfg.get("reference") or {}
    ref_client = None
    if ref_cfg.get("base_url"):
        ref_client = OpenAIClient(ref_cfg["base_url"], ref_cfg.get("model", "local"), ref_cfg.get("api_key"))

    report = {"generated": _dt.datetime.now().isoformat(timespec="seconds"),
              "suite": args.suite, "engine": cfg.get("engine", {}), "reps": args.reps,
              "gates": [], "pass": True}

    def _emit(r):
        report["gates"].append(r)
        report["pass"] = report["pass"] and r["pass"]
        print(f"[eval] {r['gate']:12s} {'PASS' if r['pass'] else 'FAIL'}  ({r['metric']}={r['value']})")

    if args.suite == "local":
        for g in LOCAL_GATES:
            _emit(run_gate(g, client, cfg, args.reps, ref_client))
    elif args.suite == "lm-eval":
        from suites.lm_eval import lm_eval_suite
        rr = lm_eval_suite.run(client, cfg.get("lm_eval", {}))
        _emit({"gate": "lm_eval", "pass": rr["pass"], "value": rr["value"], "metric": rr["metric"],
               "threshold": rr["threshold"], "reps": 1, "runs": [rr]})
    elif args.suite == "halogen":
        from suites.halogen import halogen_suite
        if args.verifier_backend:
            cfg.setdefault("halogen", {}).setdefault("verifier", {})["backend"] = args.verifier_backend
        rr = halogen_suite.run(client, cfg.get("halogen", {}))
        _emit({"gate": "halogen", "pass": rr["pass"], "value": rr["value"], "metric": rr["metric"],
               "threshold": rr["threshold"], "reps": 1, "runs": [rr]})

    if args.golden:
        try:
            with open(args.golden, encoding="utf-8") as f:
                gres = compare_golden(report, json.load(f), args.golden_tol)
            report["golden"] = gres
            report["pass"] = report["pass"] and gres["ok"]
            bad = [r["gate"] for r in gres["rows"] if not r["ok"]]
            print(f"[eval] golden {'PASS' if gres['ok'] else 'FAIL'}"
                  + (f" (gate diversi: {bad})" if bad else ""))
        except Exception as e:  # noqa: BLE001
            print(f"[eval] golden errore: {e}", file=sys.stderr)
    if args.save_golden:
        save_golden(report, args.save_golden)
        print(f"[eval] golden salvato: {args.save_golden}")

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    jpath = os.path.join(report_dir, f"eval-{args.suite}-{stamp}.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    mpath = os.path.join(report_dir, f"eval-{args.suite}-{stamp}.md")
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(render_md(report))
    print(f"[eval] report: {jpath}")
    print(f"[eval] overall: {'PASS' if report['pass'] else 'FAIL'}")
    return 0 if report["pass"] else 1


def render_md(report):
    lines = [f"# Eval report — suite `{report['suite']}`",
             "", f"- generated: {report['generated']}",
             f"- engine: `{report['engine'].get('backend','?')}` "
             f"model=`{report['engine'].get('model','-')}`",
             f"- overall: **{'PASS' if report['pass'] else 'FAIL'}**", "",
             "| gate | esito | metrica | valore |", "|---|---|---|---|"]
    for g in report["gates"]:
        lines.append(f"| {g['gate']} | {'PASS' if g['pass'] else 'FAIL'} | {g['metric']} | {g['value']} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
