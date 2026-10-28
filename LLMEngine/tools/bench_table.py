#!/usr/bin/env python3
"""bench_table.py — W2: risultati di bench come JSON + tabella markdown GENERATA (mai a mano).

PERCHE' (todo W2, pattern llm.mojo: "nothing in this table is entered by hand"): le tabelle del
vault erano trascritte a mano dai log -> errori di trascrizione (2 commessi il 2026-09-17). Qui la
tabella si rigenera dallo script, e il JSON e' il dato di riferimento accanto al bench che lo produce.

Input: uno o piu' TSV prodotti da `tools/bench4_compare.sh` (BENCH4_OUT), formato:
    modello<TAB>pf_ours<TAB>dc_ours<TAB>pf_mlx<TAB>dc_mlx<TAB>pf_rapid<TAB>dc_rapid
Output: JSON (`--json`) + tabella markdown su stdout (delta calcolati DENTRO la stessa run).

Uso:
    tools/bench_table.py --tsv /tmp/bench4_a8.tsv --label "A8 on" [--tsv ... --label ...] [--json out.json]
"""
import argparse
import json
import sys
from datetime import datetime


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_tsv(path):
    rows = []
    with open(path) as f:
        for ln in f:
            f7 = ln.rstrip("\n").split("\t")
            if len(f7) < 7 or f7[0] == "modello":
                continue
            rows.append({
                "model": f7[0],
                "pf_ours": num(f7[1]), "dc_ours": num(f7[2]),
                "pf_mlx": num(f7[3]), "dc_mlx": num(f7[4]),
                "pf_rapid": num(f7[5]), "dc_rapid": num(f7[6]),
            })
    return rows


def pct(a, b):
    return None if (a is None or b in (None, 0)) else 100.0 * (a - b) / b


def fmt(x, dec=1):
    return "n/d" if x is None else f"{x:.{dec}f}".replace(".", ",")


def fmt_pct(x):
    return "—" if x is None else f"**{x:+.1f}%**".replace(".", ",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", action="append", required=True, help="TSV di una run (ripetibile)")
    ap.add_argument("--label", action="append", default=None, help="etichetta della run (stesso ordine dei --tsv)")
    ap.add_argument("--json", default=None, help="scrive il JSON qui")
    a = ap.parse_args()

    labels = a.label or [f"run{i+1}" for i in range(len(a.tsv))]
    runs = [{"label": labels[i] if i < len(labels) else f"run{i+1}",
             "tsv": a.tsv[i], "rows": load_tsv(a.tsv[i])} for i in range(len(a.tsv))]

    doc = {"generated": datetime.now().isoformat(timespec="seconds"), "runs": []}
    for r in runs:
        for row in r["rows"]:
            row["d_pf"] = pct(row["pf_ours"], row["pf_mlx"])   # ours vs mlx, within-run
            row["d_dc"] = pct(row["dc_ours"], row["dc_mlx"])
        doc["runs"].append({"label": r["label"], "tsv": r["tsv"], "rows": r["rows"]})

    if a.json:
        with open(a.json, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"# JSON scritto: {a.json}", file=sys.stderr)

    # Tabella markdown: una sezione per run (delta within-run -> confrontabile).
    for r in doc["runs"]:
        print(f"\n### bench4 — {r['label']}  ({r['tsv']})")
        print("| modello | pf ours | pf mlx_lm | Δ pf | dc ours | dc mlx_lm | Δ dc |")
        print("|---|---|---|---|---|---|---|")
        for row in r["rows"]:
            print("| {} | {} | {} | {} | {} | {} | {} |".format(
                row["model"], fmt(row["pf_ours"]), fmt(row["pf_mlx"]), fmt_pct(row["d_pf"]),
                fmt(row["dc_ours"]), fmt(row["dc_mlx"]), fmt_pct(row["d_dc"])))
        n_win = sum(1 for row in r["rows"] if row["d_pf"] is not None and row["d_pf"] > 0)
        n_cmp = sum(1 for row in r["rows"] if row["d_pf"] is not None)
        print(f"\n_Prefill: {n_win}/{n_cmp} modelli con mlx_lm vinti._")


if __name__ == "__main__":
    main()
