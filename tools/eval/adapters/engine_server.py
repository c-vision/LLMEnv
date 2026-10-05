"""Adapter del motore sotto test — OpenAI-compatibile, con backend MOCK per i test.

Nessuna dipendenza dal codice del motore: si parla solo via HTTP (`/v1/chat/completions`).
Il backend `mock` e' deterministico e non usa rete, cosi' l'harness e' testabile senza modello.

Doc: 01-Projects/LLMEnv/Todo/(open) - LLM-Eval-Harness-Qualita-e-Allucinazioni.md (F1).
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error


class EngineError(Exception):
    """Errore di trasporto/risposta del motore."""


class Client:
    name = "base"

    def chat(self, prompt, *, system=None, temperature=0.0, max_tokens=64, seed=None):
        """Ritorna un dict {text, raw}. temperature=0 = greedy (deterministico atteso)."""
        raise NotImplementedError


class OpenAIClient(Client):
    """Client HTTP verso un server OpenAI-compatibile (FastLLM serve / mlx_lm.server)."""

    name = "http"

    def __init__(self, base_url, model, api_key=None, timeout=1200.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def chat(self, prompt, *, system=None, temperature=0.0, max_tokens=64, seed=None):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        body = {"model": self.model, "messages": msgs,
                "temperature": temperature, "max_tokens": max_tokens, "stream": False}
        if seed is not None:
            body["seed"] = seed
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.base_url + "/v1/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("Authorization", "Bearer " + self.api_key)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise EngineError(f"HTTP {e.code}: {e.read()[:400]!r}") from e
        except Exception as e:  # noqa: BLE001
            raise EngineError(str(e)) from e
        try:
            text = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise EngineError(f"risposta inattesa: {raw!r}") from e
        return {"text": text, "raw": raw}

    def models(self):
        try:
            with urllib.request.urlopen(self.base_url + "/v1/models", timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            raise EngineError(str(e)) from e


class MockClient(Client):
    """Backend deterministico: nessuna rete. `vary=True` simula un motore non deterministico."""

    name = "mock"

    def __init__(self, answer=None, vary=False):
        self.answer = answer
        self.vary = vary
        self._n = 0

    def chat(self, prompt, *, system=None, temperature=0.0, max_tokens=64, seed=None):
        if self.answer is not None:
            text = self.answer
        else:
            import re
            m = re.search(r"access code is ([A-Z0-9\-]+)", prompt)
            text = m.group(1) if m else ("MOCK|" + prompt.strip()[:40])
        if self.vary:
            self._n += 1
            text += f" #{self._n}"
        return {"text": text, "raw": {"mock": True}}


def make_client(cfg):
    """Costruisce il client dal blocco `engine` della config."""
    eng = cfg.get("engine", {})
    backend = eng.get("backend", "mock")
    if backend == "mock":
        return MockClient()
    if backend in ("http", "openai", "server"):
        return OpenAIClient(eng.get("base_url", "http://127.0.0.1:8080"),
                            eng.get("model", "local"), eng.get("api_key"), float(eng.get("timeout", 1200)))
    raise EngineError(f"backend sconosciuto: {backend}")
