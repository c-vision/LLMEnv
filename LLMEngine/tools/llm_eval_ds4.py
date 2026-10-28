#!/usr/bin/env python3
"""llm_eval_ds4.py — runner del benchmark ds4 (antirez) per i modelli GMBasic.

Porta il benchmark `ds4-eval` (casi in tools/ds4_eval/cases.json, esportati da
`ds4_eval_cases.c`) e lo esegue su un modello MLX qualunque — incluso quello PRODOTTO da
GMBasic (`LLM MODEL CONVERT`, round-trip MLX->GGUF->MLX), così la qualità del modello ridotto
si MISURA e si confronta con l'originale.

Backend: `mlx_lm` (runtime MLX standard) — carica una dir MLX valida (originale o ridotta).
Grading: portato da ds4_eval.c (choice/integer/rational/exact/ordered/line_set).

Uso:
  python3 llm_eval_ds4.py --model <dir-mlx> [--cases cases.json] \
      [--suite core|hard|hard_smoke|all] [--limit N] [--max-tokens N] [--json out.json]
"""
import argparse, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASES_DEFAULT = os.path.join(HERE, "ds4_eval", "cases.json")

# --- suites (bitmask nel campo `suites`) ---
SUITE_BIT = {"core": 1, "hard": 2, "hard_smoke": 4}
KIND = {0: "auto", 1: "choice", 2: "integer", 3: "rational", 4: "exact_text", 5: "ordered", 6: "line_set"}


def _visible(text):
    """Il modello può emettere un blocco di ragionamento: il grading guarda dopo </think>."""
    i = text.rfind("</think>")
    return text[i + len("</think>"):] if i >= 0 else text


def _answer_marker(text):
    """Ultimo 'Answer:' (case-insensitive) con confine di parola; fallback: ultima occorrenza."""
    m = None
    for mm in re.finditer(r"(?<![A-Za-z])answer\s*:", text, re.IGNORECASE):
        m = mm
    if m:
        return text[m.end():]
    m = re.search(r"answer", text, re.IGNORECASE)
    return text[m.end():] if m else None


def _extract_choice(text, nchoices):
    if nchoices <= 0:
        return "?"
    vis = _visible(text)
    maxc = chr(ord("A") + nchoices - 1)
    payload = _answer_marker(vis)
    if payload is not None:
        for ch in payload[:96]:
            c = ch.upper()
            if "A" <= c <= maxc:
                return c
    # fallback: ultima lettera valida nella parte visibile
    for ch in reversed(vis):
        c = ch.upper()
        if "A" <= c <= maxc:
            return c
    return "?"


def _extract_integer(text):
    vis = _visible(text)
    payload = _answer_marker(vis)
    if payload is not None:
        line = payload.split("\n", 1)[0][:160]
        eq = line.rfind("=")
        tail = line[eq + 1:] if eq >= 0 else line
        m = re.search(r"-?\d+", tail) or re.search(r"-?\d+", line)
        if m:
            return str(int(m.group(0)))
    nums = re.findall(r"\d+", vis)
    return str(int(nums[-1])) if nums else "?"


def _strip_latex(s):
    for tok in ("\\boxed", "\\left", "\\right"):
        s = s.replace(tok, "")
    return s


def _norm_exact(s):
    s = _strip_latex(s)
    s = re.sub(r"[\s\$\*\{\}]", "", s)
    return re.sub(r"[.;]+$", "", s).lower()


def _norm_ordered(s):
    s = re.sub(r"<[^>]*>", "", s)
    s = re.sub(r"[^0-9a-zA-Z,\-+/.]", "", s)
    return re.sub(r"[.,]+$", "", s).lower()


def _norm_rational(s):
    m = re.search(r"\\frac\s*\{([^}]*)\}\s*\{([^}]*)\}", s)
    if m:
        return re.sub(r"\s", "", m.group(1)) + "/" + re.sub(r"\s", "", m.group(2))
    return re.sub(r"[^0-9\-+/.]+", "", s).rstrip(".")


def _norm_line_set(s):
    nums = re.findall(r"\d+", s)
    return ",".join(str(int(n)) for n in sorted(set(nums)))


def _norm(kind, s):
    if kind == "rational":
        return _norm_rational(s)
    if kind == "exact_text":
        return _norm_exact(s)
    if kind == "ordered":
        return _norm_ordered(s)
    if kind == "line_set":
        return _norm_line_set(s)
    if kind == "integer":
        return _extract_integer(s)
    return s.strip()


def extract(case, text):
    kind = KIND.get(case["answer_kind"], "auto")
    nch = len(case.get("choices") or [])
    if kind == "choice":
        return _extract_choice(text, nch)
    if kind == "integer":
        return _extract_integer(text)
    if kind == "line_set":
        return _norm_line_set(_visible(text))
    payload = _answer_marker(_visible(text))
    if payload is None:
        payload = _visible(text)
    got = _norm(kind, payload)
    return got if got else "?"


def matches(case, got):
    kind = KIND.get(case["answer_kind"], "auto")
    ans = case.get("answer") or ""
    if kind == "choice":
        return bool(got) and got[0] == ans[:1]
    exp = _norm(kind, ans)
    if got == exp:
        return True
    for a in (case.get("aliases") or []):
        if got == _norm(kind, a):
            return True
    return False


def build_prompt(case):
    q = case["question"]
    ch = case.get("choices") or []
    if ch:
        opts = "\n".join(f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(ch))
        return f"{q}\n\n{opts}\n\nAnswer with the letter of the correct choice, as \"Answer: X\"."
    return f"{q}\n\nReply with your final answer as \"Answer: <value>\"."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="dir MLX (originale o prodotta da GMBasic)")
    ap.add_argument("--cases", default=CASES_DEFAULT)
    ap.add_argument("--suite", default="hard_smoke")
    ap.add_argument("--limit", type=int, default=0, help="max casi (0=tutti)")
    ap.add_argument("--max-tokens", type=int, default=0, help="override max_tokens per caso")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    cases = json.load(open(args.cases))
    bit = SUITE_BIT.get(args.suite)
    if args.suite != "all" and bit is not None:
        cases = [c for c in cases if c["suites"] & bit]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        print("nessun caso per la suite richiesta", file=sys.stderr)
        return 2

    from mlx_lm import load, generate
    model, tok = load(args.model)

    results = []
    nok = 0
    for i, c in enumerate(cases):
        prompt = build_prompt(c)
        mt = args.max_tokens or c.get("max_tokens") or 512
        if hasattr(tok, "apply_chat_template"):
            try:
                prompt = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True, tokenize=False)
            except Exception:
                pass
        out = generate(model, tok, prompt=prompt, max_tokens=mt, verbose=False)
        got = extract(c, out)
        ok = matches(c, got)
        nok += ok
        results.append({"id": c["id"], "domain": c.get("domain"), "kind": KIND.get(c["answer_kind"]),
                        "expected": c.get("answer"), "got": got, "ok": bool(ok)})
        print(f"[{i+1}/{len(cases)}] id={c['id']:<8} {KIND.get(c['answer_kind']):<11} "
              f"exp={str(c.get('answer')):<12} got={str(got):<12} {'OK' if ok else 'x'}")

    acc = nok / len(cases)
    print(f"\n=== {os.path.basename(args.model.rstrip('/'))} · suite={args.suite} · "
          f"{nok}/{len(cases)} = {acc*100:.1f}% ===")
    if args.json:
        json.dump({"model": args.model, "suite": args.suite, "n": len(cases),
                   "correct": nok, "accuracy": acc, "results": results},
                  open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
