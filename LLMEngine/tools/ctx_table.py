#!/usr/bin/env python
"""Riassume il TSV di ab_ctx_mlxlm.py (mediana fra gli ordini) in una tabella.

Uso: ctx_table.py <tsv> [--tsv]
Default: tabella leggibile. --tsv: righe TSV pulite.
Δ% = (GMB/mlx_lm - 1) * 100  (positivo = GMB più veloce).
"""
import statistics
import sys
from collections import defaultdict


def main():
    path = sys.argv[1]
    out_tsv = "--tsv" in sys.argv
    raw = defaultdict(list)  # (model, N) -> [(gp, mp, gd, md), ...] per ordine
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) != 7:
                continue
            try:
                raw[(p[0], int(p[2]))].append(
                    (float(p[3]), float(p[4]), float(p[5]), float(p[6])))
            except ValueError:
                continue

    if out_tsv:
        print("model\tN\tgmb_pref\tmlx_pref\tdpref%\tgmb_dec\tmlx_dec\tddec%")
    elif "--md" in sys.argv:
        print("| Modello | ctx | P GMB | P mlx_lm | **Δ P** | D GMB | D mlx_lm | **Δ D** |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")

    def col(d):
        c = "#1a7f37" if d >= 0 else "#c0392b"
        return f'<span style="color:{c}"><b>{d:+.1f}%</b></span>'

    for (model, n) in sorted(raw, key=lambda k: (k[0], k[1])):
        rows = raw[(model, n)]
        gp = statistics.median(r[0] for r in rows)
        mp = statistics.median(r[1] for r in rows)
        gd = statistics.median(r[2] for r in rows)
        md = statistics.median(r[3] for r in rows)
        dpref = (gp / mp - 1) * 100 if mp else 0.0
        ddec = (gd / md - 1) * 100 if md else 0.0
        if out_tsv:
            print(f"{model}\t{n}\t{gp:.0f}\t{mp:.0f}\t{dpref:+.1f}\t{gd:.1f}\t{md:.1f}\t{ddec:+.1f}")
        elif "--md" in sys.argv:
            print(f"| {model} | {n} | {gp:,.0f} | {mp:,.0f} | {col(dpref)} | {gd:.1f} | {md:.1f} | {col(ddec)} |")
        else:
            print(f"{model:46s} N={n:6d}  P gmb={gp:8.0f} mlx={mp:8.0f} ({dpref:+6.1f}%)  "
                  f"D gmb={gd:7.1f} mlx={md:7.1f} ({ddec:+6.1f}%)")


if __name__ == "__main__":
    main()
