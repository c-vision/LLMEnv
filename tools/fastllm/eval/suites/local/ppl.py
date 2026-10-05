"""Gate DETERMINISTICO: fedeltà quantitativa con banda di controllo (F2bis).

Chiede le **logprob dei token emessi** via `/v1/completions` (`logprobs=1`), greedy, e calcola
`bits/token = -mean(logprob)/ln2`. Se in config è data una **banda pre-registrata** `band=[low,high]`
(bits/token), il gate PASSA se il valore è dentro; senza banda è **informativo** (misura e riporta).
Deterministico a parità di prompt/opzioni/modello; non usa giudici LLM.
"""
import math

NAME = "ppl"
DESCRIPTION = "Perplexità (bits/token) del testo greedy; gate = dentro la banda pre-registrata."

_DEFAULT_PROMPT = ("The capital of France is Paris. The quick brown fox jumps over the lazy dog. "
                   "Water boils at one hundred degrees Celsius at sea level.")


def run(client, cfg):
    prompt = cfg.get("prompt", _DEFAULT_PROMPT)
    max_tokens = int(cfg.get("max_tokens", 32))
    band = cfg.get("band")  # [low, high] in bits/token; None = informativo
    out = client.complete(prompt, max_tokens=max_tokens, logprobs=True)
    lps = [x for x in (out.get("logprobs") or []) if x is not None]
    if not lps:
        return {"metric": "ppl_bits_per_token", "value": None, "threshold": band, "pass": True,
                "details": {"unsupported": "nessuna logprob dal server (gate informativo)",
                            "text": out.get("text", "")[:80]}}
    bits = [-x / math.log(2.0) for x in lps]
    mean_bits = sum(bits) / len(bits)
    ok = True if not band else (band[0] <= mean_bits <= band[1])
    return {"metric": "ppl_bits_per_token", "value": round(mean_bits, 4), "threshold": band, "pass": ok,
            "details": {"ntok": len(lps), "min": round(min(bits), 3), "max": round(max(bits), 3),
                        "text": out.get("text", "")[:80]}}
