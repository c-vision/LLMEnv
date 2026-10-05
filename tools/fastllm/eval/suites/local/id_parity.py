"""Gate DETERMINISTICO: parità greedy.

Esegue lo stesso prompt due volte a temperature 0 (greedy) e verifica che il testo sia
IDENTICO. E' il gate piu' economico e affidabile (non dipende da un giudice LLM): rileva
non-determinismo del motore/runtime (es. atomic-float, ordini di riduzione, cache instabile).

Metrica: identical (bool) + len + sha dell'output. Pass = identico.
"""
NAME = "id_parity"
DESCRIPTION = "Due run greedy (temp 0) dello stesso prompt devono dare testo identico."

_DEFAULT_PROMPTS = [
    "The capital of France is",
    "1, 2, 3,",
    "def add(a, b):",
]


def _sha(s):
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def run(client, cfg):
    prompts = cfg.get("prompts", _DEFAULT_PROMPTS)
    max_tokens = int(cfg.get("max_tokens", 64))
    rows, all_ok = [], True
    for p in prompts:
        a = client.chat(p, temperature=0.0, max_tokens=max_tokens)["text"]
        b = client.chat(p, temperature=0.0, max_tokens=max_tokens)["text"]
        same = (a == b)
        all_ok = all_ok and same
        rows.append({"prompt": p, "identical": same, "sha_a": _sha(a), "sha_b": _sha(b),
                     "len_a": len(a), "len_b": len(b)})
    return {
        "metric": "id_parity",
        "value": all_ok,
        "threshold": True,
        "pass": all_ok,
        "details": {"prompts": len(prompts), "rows": rows},
    }
