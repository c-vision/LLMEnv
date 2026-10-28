#!/usr/bin/env python3
"""Probe/verifier per il tokenizer tiktoken nativo di Kimi (kimi_linear).

Verifica due cose PRIMA di portare in C:
  (a) l'algoritmo BPE per RANK MINIMO (tiktoken/_educational.py ``bpe_encode``) con il pat_str
      ORIGINALE di Kimi riproduce ESATTAMENTE ``tiktoken.Encoding.encode``;
  (b) il pat_str ORIGINALE (con intersezione ``&&``, non supportata da PCRE2) e la sua
      RISCRITTURA compatibile PCRE2 (lookahead ``(?![\p{Han}])[\p{Lo}]``) producono gli STESSI
      split su un corpus di prova -> si puo' hardcodare la riscrittura nel modulo C.

Uso:  ~/ai/env312/bin/python tools/gmb_kimi_tiktoken_probe.py [modello_dir]
"""
from __future__ import annotations

import json
import os
import sys

MODEL = os.path.expanduser("~/ai/Models/Kimi-Linear-48B-A3B-Instruct-4bit")


def load_ranks(path: str) -> dict:
    from tiktoken.load import load_tiktoken_bpe

    return load_tiktoken_bpe(path)


def kimi_pat_str(model: str) -> str:
    sys.path.insert(0, model)
    import tokenization_kimi as tk  # type: ignore

    pat = getattr(tk.TikTokenTokenizer, "pat_str", None)
    if pat is None:
        inst = tk.TikTokenTokenizer(os.path.join(model, "tiktoken.model"))
        pat = inst.pat_str
    return pat


def bpe_encode(ranks: dict, data: bytes) -> list[int]:
    parts = [bytes([b]) for b in data]
    while True:
        best = None
        bi = -1
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and (best is None or r < best):
                best = r
                bi = i
        if best is None:
            break
        parts = parts[:bi] + [parts[bi] + parts[bi + 1]] + parts[bi + 2:]
    return [ranks[p] for p in parts]


# Riscrittura PCRE2 del pat_str Kimi: ``[A&&[^Han]]`` -> lookahead.
#   UPPER = [Lu Lt Lm Lo M]  meno Han  = (?:[Lu Lt Lm M] | (?![\p{Han}])[\p{Lo}])
#   LOWER = [Ll Lm Lo M]     meno Han  = (?:[Ll Lm M] | (?![\p{Han}])[\p{Lo}])
UPPER = r"(?:[\p{Lu}\p{Lt}\p{Lm}\p{M}]|(?![\p{Han}])[\p{Lo}])"
LOWER = r"(?:[\p{Ll}\p{Lm}\p{M}]|(?![\p{Han}])[\p{Lo}])"
PAT2 = "|".join(
    [
        r"[\p{Han}]+",
        r"[^\r\n\p{L}\p{N}]?" + UPPER + r"*" + LOWER + r"+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?" + UPPER + r"+" + LOWER + r"*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)

TESTS = [
    "Hello, my name is",
    "La capitale d'Italia è Roma.",
    "汉字测试 123",
    "2+2=4",
    "Once upon a time",
    "We're testing it's fine, they'll go.",
    "Привет мир",
    "日本語のテキストとEnglishが混在",
    "  spazi   e\ttab\nnewline ",
    "emoji 😀🚀 e simboli @#%&*()",
    "1234567890 42 1000000",
    "camelCase snake_case UPPER lower MiXeD",
    "a",
    "",
]


def main() -> int:
    import regex
    import tiktoken

    model = sys.argv[1] if len(sys.argv) > 1 else MODEL
    pat = kimi_pat_str(model)
    ranks = load_ranks(os.path.join(model, "tiktoken.model"))
    cfg = json.load(open(os.path.join(model, "tokenizer_config.json")))
    dec = cfg.get("added_tokens_decoder", {})
    specials = {}
    for sid, sobj in dec.items():
        name = sobj if isinstance(sobj, str) else sobj.get("content")
        specials[name] = int(sid)
    enc = tiktoken.Encoding(name="kimi", pat_str=pat, mergeable_ranks=ranks, special_tokens=specials)
    # NB: il pat_str ORIGINALE usa ``&&`` (intersezione di classi): in Python ``regex`` da' un
    # insieme VUOTO per le lettere -> inutilizzabile come riferimento di split. tiktoken (Rust,
    # fancy-regex) invece lo supporta: si usa ``enc.encode`` come riferimento e si valida la
    # RISCRITTURA PCRE2 (lookahead) + l'algoritmo min-rank contro di essa.
    re2 = regex.compile(PAT2)

    fail = 0
    for t in TESTS:
        ref = enc.encode(t)
        mine = []
        for w in re2.findall(t):
            mine.extend(bpe_encode(ranks, w.encode("utf-8")))
        ok = ref == mine
        if not ok:
            fail += 1
        print(f"{'OK  ' if ok else 'FAIL'} {t!r} ref={ref[:10]} mine={mine[:10]}")

    print(f"\nranks={len(ranks)} specials={len(specials)} tests={len(TESTS)} fail={fail}")
    if fail == 0:
        print("== PARITA' PIENA: algoritmo min-rank + pat_str riscritto OK ==")
        print("PAT2 (da hardcodare nel C):")
        print(PAT2)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
