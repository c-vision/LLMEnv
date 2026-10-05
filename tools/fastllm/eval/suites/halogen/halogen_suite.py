"""F3 — HALoGEN-subset: FACTUALITÀ (recall / hallucination) con verifier LOCALE a regole.

Per ogni caso (`prompt` + `facts` = atomic unit attese), chiede la risposta greedy al modello e
verifica con il **verifier** (default `local`, regola deterministica; F5) quante unità sono
supportate. Metriche: `recall = verifyati/totali` (= 1 − hallucination grezza locale) e
`response_ratio` (risposte non vuote). Gate: `recall >= min_recall` (assente = informativo).

Deterministico col verifier `local` (nessun giudice LLM, nessun dato fuori). **Tool offline: nessun
impatto sulle prestazioni dell'engine.** Verifier esterni = opt-in (F5). Vedi todo Eval Harness §F3.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from verifiers.verifier import make_verifier  # noqa: E402

NAME = "halogen"
DESCRIPTION = "Factualità (recall/HALoGEN-subset) con verifier locale a regole."

_DEFAULT_CASES = [
    {"prompt": "What is the capital of France? Answer with just the city.",
     "facts": ["Paris"]},
    {"prompt": "How many days are in a leap year? Answer with just the number.",
     "facts": ["366"]},
    {"prompt": "What gas do plants absorb from the atmosphere? Answer with just the name.",
     "facts": ["carbon dioxide"]},
]


def run(client, cfg):
    cases = cfg.get("cases") or _DEFAULT_CASES
    mt = int(cfg.get("max_tokens", 40))
    vcfg = cfg.get("verifier", {})
    ver = make_verifier(vcfg)
    if not ver.supported():
        return {"metric": "halogen_recall", "value": None, "threshold": cfg.get("min_recall"),
                "pass": True, "details": {"unsupported": f"verifier '{ver.name}' non configurato"}}

    total = verified = answered = 0
    rows = []
    for c in cases:
        ans = client.chat(c["prompt"], temperature=0.0, max_tokens=mt).get("text", "")
        facts = c.get("facts", [])
        vf = [f for f in facts if ver.verify(f, ans)]
        total += len(facts)
        verified += len(vf)
        answered += 1 if ans.strip() else 0
        rows.append({"prompt": c["prompt"], "answer": ans.strip()[:100], "facts": facts, "verified": vf})

    recall = (verified / total) if total else 0.0
    rr = answered / len(cases) if cases else 0.0
    minr = cfg.get("min_recall")
    ok = True if minr is None else (recall >= minr)
    return {"metric": "halogen_recall", "value": round(recall, 4), "threshold": minr, "pass": ok,
            "details": {"verifier": ver.name, "cases": len(cases), "verified": verified, "total": total,
                        "response_ratio": round(rr, 4), "rows": rows}}
