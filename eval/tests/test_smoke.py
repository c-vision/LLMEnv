"""Smoke test dell'eval harness — deterministico, senza motore (backend mock).

Verifica:
  1. `run_eval --suite local --backend mock` gira end-to-end e produce un report (PASS).
  2. il gate `id_parity` FALLISCE se il motore e' non deterministico (mock vary=True).
  3. il gate `needle` recupera il fatto.

Esecuzione:  python3 eval/tests/test_smoke.py
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(HERE)
sys.path.insert(0, EVAL)

import run_eval  # noqa: E402
from adapters.engine_server import MockClient  # noqa: E402
from suites.local import id_parity, needle  # noqa: E402


class Smoke(unittest.TestCase):
    def test_runner_local_mock_passes(self):
        with tempfile.TemporaryDirectory() as td:
            rc = run_eval.main(["--suite", "local", "--backend", "mock", "--report", td])
            self.assertEqual(rc, 0, "la suite local con mock deve passare")
            js = [f for f in os.listdir(td) if f.endswith(".json")]
            self.assertEqual(len(js), 1)
            with open(os.path.join(td, js[0]), encoding="utf-8") as fh:
                rep = json.load(fh)
            self.assertTrue(rep["pass"])
            gates = {g["gate"]: g["pass"] for g in rep["gates"]}
            self.assertTrue(gates["id_parity"])
            self.assertTrue(gates["needle"])

    def test_id_parity_detects_nondeterminism(self):
        r = id_parity.run(MockClient(vary=True), {"max_tokens": 8, "prompts": ["x"]})
        self.assertFalse(r["pass"], "id_parity deve fallire con un motore non deterministico")

    def test_needle_retrieves_fact(self):
        r = needle.run(MockClient(), {"n_filler": 6, "needle_key": "ABC-42", "needle_pos": 3})
        self.assertTrue(r["pass"], f"needle non recuperato: {r['details']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
