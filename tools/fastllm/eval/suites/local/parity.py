"""Gate DETERMINISTICO: parità greedy motore vs riferimento.

Confronta il testo greedy (temp 0) tra il **motore sotto test** e una **reference** (es. `mlx_lm.server`),
sugli stessi prompt. Metrica: `match` (identico) oppure `prefix_ratio` (prefisso comune / min-len).
Gate: `match` **oppure** `prefix_ratio >= tolerance`. Se la reference non è configurata → **saltato**
(informativo). Deterministico a parità di modelli/opzioni.
"""
NAME = "parity"
DESCRIPTION = "Parità greedy engine vs riferimento (testo identico o prefix_ratio >= tolleranza)."

_DEFAULT_PROMPTS = ["The capital of France is", "1, 2, 3, 4, 5,", "def add(a, b):"]


def _lcp(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def run(client, cfg):
    ref = cfg.get("_ref_client")
    if ref is None:
        return {"metric": "parity", "value": None, "threshold": None, "pass": True,
                "details": {"skipped": "reference non configurata"}}
    prompts = cfg.get("prompts", _DEFAULT_PROMPTS)
    mt = int(cfg.get("max_tokens", 64))
    tol = float(cfg.get("tolerance", 1.0))
    rows, all_ok = [], True
    for p in prompts:
        a = client.chat(p, temperature=0.0, max_tokens=mt)["text"]
        b = ref.chat(p, temperature=0.0, max_tokens=mt)["text"]
        match = (a == b)
        lcp = _lcp(a, b)
        ratio = lcp / max(1, min(len(a), len(b)))
        ok = match or ratio >= tol
        all_ok = all_ok and ok
        rows.append({"prompt": p, "match": match, "prefix_ratio": round(ratio, 4), "ok": ok,
                     "lcp": lcp, "engine_len": len(a), "ref_len": len(b)})
    return {"metric": "parity", "value": all_ok, "threshold": tol, "pass": all_ok,
            "details": {"prompts": len(prompts), "rows": rows}}
