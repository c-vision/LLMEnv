"""Gate DETERMINISTICO: needle-in-a-haystack (retrieval a contesto medio).

Costruisce un contesto di rumore deterministico con un fatto "ago" in mezzo e chiede di
estrarlo a temperature 0. Pass = la risposta contiene l'ago. Deterministico (greedy +
contesto fisso); misura la capacita' di retrieval, non le prestazioni.
"""
NAME = "needle"
DESCRIPTION = "Fatto sepolto in un contesto di rumore: il modello deve recuperarlo (greedy)."

_FILLER = ("The archive log entry records a routine system check with no anomalies. "
           "Operators reviewed the telemetry and documented the sequence for audit.")


def run(client, cfg):
    n_filler = int(cfg.get("n_filler", 40))
    needle_key = cfg.get("needle_key", "PLUMBUS-7741")
    needle_fact = cfg.get("needle_fact",
                          f"The secret access code is {needle_key}.")
    question = cfg.get("question", "What is the secret access code? Answer with only the code.")
    pos = int(cfg.get("needle_pos", n_filler // 2))

    blocks = [f"[{i}] {_FILLER}" for i in range(n_filler)]
    blocks[pos] = f"[{pos}] {needle_fact}"
    context = "\n".join(blocks)
    prompt = f"{context}\n\n{question}"

    out = client.chat(prompt, temperature=0.0, max_tokens=int(cfg.get("max_tokens", 16)))
    text = out["text"]
    hit = needle_key in text
    return {
        "metric": "needle",
        "value": hit,
        "threshold": True,
        "pass": hit,
        "details": {"n_filler": n_filler, "needle_key": needle_key,
                    "needle_pos": pos, "answer": text.strip()[:120]},
    }
