#!/usr/bin/env python3
"""gen_tokenizer_golden.py — oracolo d'oro per BertWordPieceTokenizer (fase 1).

Genera `samples/Fixtures/tokenizer/tokenizer_golden.json` + i due `vocab.txt` minimi,
usando come ORACOLO la libreria HF `tokenizers` (WordPiece + BertNormalizer(lowercase,
strip_accents) + BertPreTokenizer + TemplateProcessing [CLS]/[SEP]) — il software di
riferimento del modello, NON una nostra implementazione (todo Tokenizer-WordPiece §3:
"l'oracolo è l'output del software di riferimento del modello, versionato come fixture").

Uso:
  pip install tokenizers        (>= 0.20; verificato con 0.23.2)
  python3 tools/gen_tokenizer_golden.py            # rigenera fixture + vocab
  python3 tools/gen_tokenizer_golden.py --check    # rigenera in memoria e diffa (exit 1 su drift)

Determinismo: vocab e corpus sono definiti qui; il JSON è ri-generabile identico a meno
del campo informativo `tokenizers_version` (escluso dal confronto --check).

Regole di troncamento (casi `truncation`): NON delegate a HF (la Truncation di `tokenizers`
taglia anche i marker); il golden applica la forma canonica BERT/transformers:
  - singola: [CLS] + primi (max_length-2) token + [SEP];
  - coppia:  il documento viene tagliato (contenuto), i marker [CLS]/[SEP]q/[SEP]d restano;
    token_type_ids = 0 sul segmento query (incluso il suo [SEP]), 1 sul documento (incluso
    il [SEP] finale) — assegnati DOPO il taglio.
(criterio di accettazione nel todo §2: tokens[0]==CLS, ultimo non-pad==SEP, segmento 1
dopo il [SEP] della query.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.normalizers import BertNormalizer
    from tokenizers.pre_tokenizers import BertPreTokenizer
    from tokenizers.processors import TemplateProcessing
except ImportError:
    sys.exit("serve la libreria HF `tokenizers`: pip install tokenizers")

import tokenizers as _tk

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "samples" / "Fixtures" / "tokenizer"

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

VOCABS = {
    # uncased English minimale (ASCII): copre le stringhe del corpus che hanno pezzi
    # inglesi; tutto il resto finisce [UNK] (ed è atteso: il golden registra anche quello).
    "en_uncased": SPECIALS + [
        "hello", "world", "the", "quick", "brown", "fox", "jump", "##s", "!",
        "res", "##ume", "naive", "cafe",
        "snake", "_", "case", "test", "mixed", "lower", "##ing",
        "and", "tab", "control", "chars", "rocket", "launch", "middle",
        "guillemets", "em", "dash", "ellipsis", "-", ".", ",", "?",
    ],
    # multilingue minimale (stile mBERT): CJK per ideogramma singolo, cirillico/arabo/
    # hindi a wordpiece, punteggiatura tipografica presente, emoji ASSENTE (→ [UNK]).
    "multilingual": SPECIALS + [
        "hello", "world", "the", "quick", "brown", "fox", "jump", "##s", "!",
        "res", "##ume", "naive", "cafe",
        "snake", "_", "case", "test", "mixed", "lower", "##ing",
        "and", "tab", "control", "chars", "rocket", "launch", "middle",
        "guillemets", "em", "dash", "ellipsis", "-", ".", ",", "?",
        "при", "##вет", "мир",
        "مر", "##حبا", "با", "##لعالم",
        "नम", "##स्ते", "दु", "##निया",
        "人", "工", "智", "能", "東", "京",
        "タ", "##ワ", "##ー",
        "서", "##울", "##특", "##별", "##시",
        "«", "»", "—", "…", "·", "†", "°",
    ],
}

# ~16 stringhe × 2 vocabolari: i 4 casi del todo §3 (latino accentato, cirillico, arabo,
# hindi, cinese, giapponese, coreano, punteggiatura tipografica, byte di controllo,
# parola >100 char, stringa vuota, emoji) + underscore (Pc) e casing.
CORPUS = [
    ("hello-world", "Hello world"),
    ("ascii-punct", "The quick brown fox jumps!"),
    ("latin-accents", "résumé naïve café"),
    ("cyrillic", "Привет мир"),
    ("arabic", "مرحبا بالعالم"),
    ("hindi", "नमस्ते दुनिया"),
    ("chinese-cjk", "人工智能"),
    ("japanese-mix", "東京タワー"),
    ("korean", "서울특별시"),
    ("typo-punct", "«Guillemets» — em-dash … ellipsis · middle"),
    ("control-chars", "tab\tand\u0001control\u001Fchars"),
    ("long-word-101", "a" * 101),
    ("empty", ""),
    ("emoji", "🚀 rocket launch"),
    ("underscore-pc", "snake_case_test"),
    ("mixed-case", "Mixed CASE Lowering"),
]

PAIRS = [
    # (id, vocab, query, document, max_length oppure None = nessun troncamento)
    ("pair-en-short", "en_uncased", "the quick brown fox", "hello world rocket launch", None),
    ("pair-mlt-cross-script", "multilingual", "Привет мир", "人工智能 서울특별시", None),
    ("pair-trunc-doc", "en_uncased", "the quick brown fox",
     " ".join(["hello", "world"] * 20), 16),
]

TRUNC_SINGLE = [
    # (id, vocab, text, max_length)
    ("trunc-single-en", "en_uncased", " ".join(["the", "quick", "brown", "fox"] * 5), 16),
    ("trunc-single-mlt", "multilingual", " ".join(["人", "工", "智", "能"] * 6), 12),
    ("trunc-single-min", "en_uncased", "hello world rocket launch", 4),
]


def build_tokenizer(vocab: list[str]) -> Tokenizer:
    tok = Tokenizer(WordPiece({t: i for i, t in enumerate(vocab)}, unk_token="[UNK]"))
    tok.normalizer = BertNormalizer(lowercase=True, strip_accents=True)
    tok.pre_tokenizer = BertPreTokenizer()
    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    return tok


def basic_words(tok: Tokenizer, text: str) -> list[str]:
    """Oracle per BasicTokenize: normalizer (lowercase+strip accents+clean) → pre-tokenizer."""
    normalized = tok.normalizer.normalize_str(text)
    return [w for w, _ in tok.pre_tokenizer.pre_tokenize_str(normalized)]


def truncate_single(ids: list[int], max_length: int) -> list[int]:
    if len(ids) <= max_length:
        return ids
    keep = max_length - 2
    return ids[: 1 + keep] + [ids[-1]]


def truncate_pair(q_ids: list[int], d_ids: list[int], max_length: int):
    """q_ids include [CLS]…[SEP]; d_ids SENZA speciali. Ritorna (ids, type_ids)."""
    budget = max_length - len(q_ids) - 1  # contenuto documento, [SEP] finale escluso
    doc = d_ids[: max(0, budget)]
    ids = q_ids + doc + [3]
    type_ids = [0] * len(q_ids) + [1] * (len(ids) - len(q_ids))
    return ids, type_ids


def generate() -> dict:
    import tokenizers

    out = {
        "generated_by": "tools/gen_tokenizer_golden.py",
        "oracle": "HF tokenizers (WordPiece + BertNormalizer(lowercase, strip_accents) + BertPreTokenizer + TemplateProcessing)",
        "tokenizers_version": tokenizers.__version__,
        "vocabs": {
            name: {
                "path": f"vocab_{name}.txt",
                "size": len(v),
                "cls_id": 2, "sep_id": 3, "unk_id": 1, "pad_id": 0,
            }
            for name, v in VOCABS.items()
        },
        "cases": [],
        "pairs": [],
        "truncation": [],
    }
    toks = {name: build_tokenizer(v) for name, v in VOCABS.items()}

    for vocab_name in ("en_uncased", "multilingual"):
        tok = toks[vocab_name]
        for cid, text in CORPUS:
            enc = tok.encode(text)
            out["cases"].append({
                "id": f"{vocab_name}:{cid}",
                "vocab": vocab_name,
                "text": text,
                "basic_tokens": basic_words(tok, text),
                "tokens": enc.tokens,
                "ids": enc.ids,
            })

    for pid, vocab_name, query, document, max_len in PAIRS:
        tok = toks[vocab_name]
        q_ids = tok.encode(query).ids                      # [CLS] q [SEP]
        d_raw = tok.encode(document, add_special_tokens=False).ids
        if max_len is None:
            enc = tok.encode(query, document)
            ids, type_ids, tokens = enc.ids, enc.type_ids, enc.tokens
        else:
            ids, type_ids = truncate_pair(q_ids, d_raw, max_len)
            tokens = None
        case = {
            "id": pid,
            "vocab": vocab_name,
            "query": query,
            "document": document,
            "max_length": max_len,
            "ids": ids,
            "token_type_ids": type_ids,
        }
        if tokens is not None:
            case["tokens"] = tokens
        out["pairs"].append(case)

    for tid, vocab_name, text, max_len in TRUNC_SINGLE:
        tok = toks[vocab_name]
        full = tok.encode(text).ids
        out["truncation"].append({
            "id": tid,
            "vocab": vocab_name,
            "text": text,
            "max_length": max_len,
            "untruncated_len": len(full),
            "ids": truncate_single(full, max_len),
        })
    return out


def write_vocab_files() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, vocab in VOCABS.items():
        (FIXTURE_DIR / f"vocab_{name}.txt").write_text(
            "\n".join(vocab) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="diffa il JSON versionato contro una rigenerazione (exit 1 su drift)")
    args = ap.parse_args()

    data = generate()
    if args.check:
        path = FIXTURE_DIR / "tokenizer_golden.json"
        if not path.exists():
            sys.exit(f"fixture mancante: {path} (generala: python3 tools/gen_tokenizer_golden.py)")
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        for k in ("tokenizers_version",):
            on_disk.pop(k, None)
            data.pop(k, None)
        if on_disk != data:
            sys.exit("DRIFT: tokenizer_golden.json diverso dalla rigenerazione (oracolo cambiato?)")
        print("OK: golden coerente con l'oracolo (casi "
              f"{len(data['cases'])}, coppie {len(data['pairs'])}, troncamenti {len(data['truncation'])})")
        return

    write_vocab_files()
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIR / "tokenizer_golden.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"scritti: {FIXTURE_DIR}/tokenizer_golden.json "
          f"({len(data['cases'])} casi, {len(data['pairs'])} coppie, {len(data['truncation'])} troncamenti) "
          f"+ {len(VOCABS)} vocab (tokenizers {_tk.__version__})")


if __name__ == "__main__":
    main()
