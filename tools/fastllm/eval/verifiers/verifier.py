"""F5 — backend VERIFIER + caching.

Un verifier decide se un' **atomic unit** (claim) è supportata da un contesto (la risposta del
modello). Backend:
- `local` (default): **regola deterministica** (parole-chiave del claim presenti nel contesto) →
  nessun dato fuori dalla macchina, offline.
- `openai`: **LLM-as-judge** via API (opt-in; richiede `api_key` in config NON committata).
- `together`: segnaposto opt-in (stesso schema) — per ora non implementato → `unsupported`.

**Caching** (hash backend+claim+contesto) per non ripagare verdetti identici. Cache in
`tools/fastllm/eval/config/verifier_cache.json` (gitignorata).
"""
import hashlib
import json
import os
import urllib.request


def _tokens(s):
    import re
    return [w for w in re.split(r"[^0-9a-zA-Zàèéìòù']+", (s or "").lower()) if w]


class Cache:
    def __init__(self, path):
        self.path = path
        self.d = {}
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.d = json.load(f)
            except Exception:  # noqa: BLE001
                self.d = {}

    def key(self, backend, claim, context):
        return hashlib.sha256(f"{backend}|{claim}|{context}".encode("utf-8")).hexdigest()[:24]

    def get(self, k):
        return self.d.get(k)

    def put(self, k, v):
        self.d[k] = v
        if self.path:
            d = os.path.dirname(os.path.abspath(self.path))
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.d, f)


class Verifier:
    name = "local"

    def __init__(self, cfg, cache):
        self.cfg = cfg or {}
        self.cache = cache

    def supported(self):
        return True

    def verify(self, claim, context):
        k = self.cache.key(self.name, claim, context)
        v = self.cache.get(k)
        if v is not None:
            return bool(v)
        r = bool(self._verify(claim, context))
        self.cache.put(k, r)
        return r

    def _verify(self, claim, context):
        raise NotImplementedError


class LocalKeywordVerifier(Verifier):
    """Regola locale deterministica: il claim è supportato se le sue parole-chiave (>2 char) sono
    tutte presenti nel contesto (case-insensitive). Offline, nessun dato fuori."""

    name = "local"

    def _verify(self, claim, context):
        words = [w for w in _tokens(claim) if len(w) > 2]
        if not words:
            return False
        ctx = (context or "").lower()
        return all(w in ctx for w in words)


class OpenAIVerifier(Verifier):
    """LLM-as-judge via API OpenAI (opt-in). Richiede `api_key` (config non committata)."""

    name = "openai"

    def supported(self):
        return bool(self.cfg.get("api_key"))

    def _verify(self, claim, context):
        body = {"model": self.cfg.get("judge_model", "gpt-4o-mini"),
                "messages": [{"role": "user", "content":
                              f"Contesto:\n{context}\n\nAffermazione: {claim}\n\n"
                              "L'affermazione è SUPPORTATA dal contesto? Rispondi solo 'yes' o 'no'."}],
                "temperature": 0, "max_tokens": 3}
        req = urllib.request.Request(self.cfg.get("base_url", "https://api.openai.com") + "/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={
                                         "Content-Type": "application/json",
                                         "Authorization": "Bearer " + self.cfg["api_key"]})
        with urllib.request.urlopen(req, timeout=float(self.cfg.get("timeout", 60))) as r:
            raw = json.loads(r.read().decode())
        ans = raw["choices"][0]["message"]["content"].strip().lower()
        return ans.startswith("y")


def make_verifier(cfg):
    """cfg: blocco `verifier` {backend, api_key, ...}, cache_il path è `cache`."""
    cfg = cfg or {}
    cache = Cache(cfg.get("cache") or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "..", "config", "verifier_cache.json"))
    backend = cfg.get("backend", "local")
    if backend == "local":
        return LocalKeywordVerifier(cfg, cache)
    if backend == "openai":
        return OpenAIVerifier(cfg, cache)
    return LocalKeywordVerifier(cfg, cache)  # fallback sicuro (non esce dati)
