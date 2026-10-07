#!/usr/bin/env python3
"""ds4_ab_long.py — come ds4_ab_decode.py ma con un PROMPT LUNGO (N parole), per misurare il PREFILL
BATCH (>=192 token), che il prompt corto (~39 tok) NON copre. Riusa interamente i cap di sicurezza di
ds4_ab_decode.py (RSS albero, floor memoria libera, pre-check).

Uso: ds4_ab_long.py <words=900> [N=2] [max_new=1] [floor=30] [rss_cap=62] [pre=60]
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ds4_ab_decode as H  # noqa: E402

_BASE = ("Explain in detail the history of France, covering the Roman conquest of Gaul, the Merovingian "
         "and Carolingian dynasties, the French Revolution, the modern republic, its wars, its culture, "
         "its law and its cities. ")


def main():
    words = int(sys.argv[1]) if len(sys.argv) > 1 else 900
    rest = sys.argv[2:]
    out = []
    while len(out) < words:
        out += _BASE.split()
    H.P = " ".join(out[:words])
    sys.argv = ["ds4_ab_long"] + rest
    H.main()


if __name__ == "__main__":
    main()
