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
from adapters.engine_server import make_client, EngineError  # noqa: E402

LOCAL_GATES = ["id_parity", "needle"]


def load_config(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def run_gate(modname, client, cfg, reps):
    mod = importlib.import_module(f"suites.local.{modname}")
    results = []
    for _ in range(max(1, reps)):
        results.append(mod.run(client, cfg.get(modname, {})))
    passes = [r["pass"] for r in results]
    ok = all(passes)
    return {"gate": modname, "pass": ok,
            "value": results[0]["value"], "metric": results[0]["metric"],
            "threshold": results[0]["threshold"],
            "reps": reps, "runs": results}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Eval harness (qualita'/fedelta').")
    ap.add_argument("--suite", default="local", choices=["local"])
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "config", "eval.json"))
    ap.add_argument("--report", default=None, help="cartella di output (default: <eval>/reports)")
    ap.add_argument("--backend", default=None, help="override: mock|http")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--reps", type=int, default=1)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.backend:
        cfg.setdefault("engine", {})["backend"] = args.backend
    if args.base_url:
        cfg.setdefault("engine", {})["base_url"] = args.base_url
    if args.model:
        cfg.setdefault("engine", {})["model"] = args.model

    report_dir = args.report or os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(report_dir, exist_ok=True)

    try:
        client = make_client(cfg)
    except EngineError as e:
        print(f"[eval] ERRORE client: {e}", file=sys.stderr)
        return 2

    gates = LOCAL_GATES if args.suite == "local" else []
    report = {"generated": _dt.datetime.now().isoformat(timespec="seconds"),
              "suite": args.suite, "engine": cfg.get("engine", {}), "reps": args.reps,
              "gates": [], "pass": True}
    for g in gates:
        r = run_gate(g, client, cfg, args.reps)
        report["gates"].append(r)
        report["pass"] = report["pass"] and r["pass"]
        mark = "PASS" if r["pass"] else "FAIL"
        print(f"[eval] {r['gate']:12s} {mark}  ({r['metric']}={r['value']})")

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
