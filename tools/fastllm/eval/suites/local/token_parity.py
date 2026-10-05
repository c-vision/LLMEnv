"""Gate DETERMINISTICO: parità TOKEN-level (GEN_IDS).

Esegue lo stesso prompt due volte a temperature 0 con `logprobs` e confronta la **lista dei token
emessi** (`logprobs.tokens`, che il server restituisce in `/v1/completions`) — parità a livello di
token (equivalente a `GEN_IDS`), più stretta della parità di testo. Se il server non espone i token
→ **informativo** (skipped). Harness-only: **nessuna modifica al motore, nessun impatto sulle prestazioni**.
"""
NAME = "token_parity"
DESCRIPTION = "Parità TOKEN-level (GEN_IDS): due run greedy → lista di token identica."

_DEFAULT_PROMPTS = ["The capital of France is", "def add(a, b):"]


def _first_div(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


def run(client, cfg):
    prompts = cfg.get("prompts", _DEFAULT_PROMPTS)
    mt = int(cfg.get("max_tokens", 64))
    rows, ok, have = [], True, False
    for p in prompts:
        a = client.complete(p, max_tokens=mt, logprobs=True).get("tokens") or []
        b = client.complete(p, max_tokens=mt, logprobs=True).get("tokens") or []
        if not a and not b:
            rows.append({"prompt": p, "identical": None, "note": "token non esposti dal server"})
            continue
        have = True
        same = (a == b)
        ok = ok and same
        rows.append({"prompt": p, "identical": same, "ntok": len(a), "first_div": _first_div(a, b)})
    if not have:
        return {"metric": "token_parity", "value": None, "threshold": True, "pass": True,
                "details": {"skipped": "il server non espone i token (logprobs.tokens)", "rows": rows}}
    return {"metric": "token_parity", "value": ok, "threshold": True, "pass": ok, "details": {"rows": rows}}
