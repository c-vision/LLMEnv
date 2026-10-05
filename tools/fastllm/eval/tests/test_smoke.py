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
from suites.local import id_parity, needle, ppl, parity, token_parity  # noqa: E402


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
            self.assertTrue(gates["ppl"])
            self.assertTrue(gates["parity"])  # saltato senza reference -> pass

    def test_id_parity_detects_nondeterminism(self):
        r = id_parity.run(MockClient(vary=True), {"max_tokens": 8, "prompts": ["x"]})
        self.assertFalse(r["pass"], "id_parity deve fallire con un motore non deterministico")

    def test_needle_retrieves_fact(self):
        r = needle.run(MockClient(), {"n_filler": 6, "needle_key": "ABC-42", "needle_pos": 3})
        self.assertTrue(r["pass"], f"needle non recuperato: {r['details']}")

    def test_ppl_band(self):
        r = ppl.run(MockClient(), {"max_tokens": 8, "band": [0.0, 3.0]})
        self.assertTrue(r["pass"], f"ppl fuori banda: {r}")
        # banda stretta attorno a 0 -> deve fallire (0.288 bits/token)
        r2 = ppl.run(MockClient(), {"max_tokens": 8, "band": [0.29, 0.30]})
        self.assertTrue(r2["pass"] or r2["value"] > 0.28, f"ppl atteso ~0.288: {r2}")

    def test_verifier_local(self):
        from verifiers.verifier import LocalKeywordVerifier, Cache
        v = LocalKeywordVerifier({}, Cache(None))
        self.assertTrue(v.verify("Paris", "the capital of france is Paris"))
        self.assertFalse(v.verify("London", "the capital of france is Paris"))

    def test_halogen_suite(self):
        from suites.halogen import halogen_suite
        r = halogen_suite.run(MockClient(answer="Paris"), {"cases": [{"prompt": "x", "facts": ["Paris"]}],
                                                           "min_recall": 1.0})
        self.assertTrue(r["pass"], r)
        self.assertEqual(r["value"], 1.0)
        r2 = halogen_suite.run(MockClient(answer="Paris"), {"cases": [{"prompt": "x", "facts": ["Berlin"]}],
                                                            "min_recall": 1.0})
        self.assertFalse(r2["pass"], r2)

    def test_lm_eval_wrapper_with_fake(self):
        from suites.lm_eval import lm_eval_suite
        with tempfile.TemporaryDirectory() as td:
            fake = os.path.join(td, "lm_eval")
            with open(fake, "w") as fh:
                fh.write("#!/usr/bin/env python3\n"
                         "import sys, os, json\n"
                         "a=sys.argv[1:]\n"
                         "op=a[a.index('--output_path')+1] if '--output_path' in a else '.'\n"
                         "os.makedirs(op, exist_ok=True)\n"
                         "open(os.path.join(op,'results.json'),'w').write("
                         "json.dumps({'results':{'gsm8k':{'acc,none':0.42}}}))\n")
            os.chmod(fake, 0o755)
            r = lm_eval_suite.run(None, {"bin": fake, "base_url": "http://x", "tasks": ["gsm8k"],
                                         "output_dir": os.path.join(td, "out"), "min_accuracy": 0.4})
            self.assertTrue(r["pass"], r)
            self.assertEqual(r["value"], {"gsm8k": 0.42})
            r2 = lm_eval_suite.run(None, {"bin": fake, "base_url": "http://x", "tasks": ["gsm8k"],
                                          "output_dir": os.path.join(td, "out2"), "min_accuracy": 0.5})
            self.assertFalse(r2["pass"], r2)
            r3 = lm_eval_suite.run(None, {"bin": os.path.join(td, "nope")})
            self.assertTrue(r3["pass"])
            self.assertIn("unsupported", r3["details"])

    def test_golden_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            g = os.path.join(td, "golden.json")
            rc = run_eval.main(["--suite", "local", "--backend", "mock", "--report", td, "--save-golden", g])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(g))
            # la stessa run confrontata col golden -> PASS
            rc2 = run_eval.main(["--suite", "local", "--backend", "mock", "--report", td, "--golden", g])
            self.assertEqual(rc2, 0, "la run identica al golden deve passare")

    def test_token_parity(self):
        r = token_parity.run(MockClient(), {"max_tokens": 8, "prompts": ["x"]})
        self.assertTrue(r["pass"], r)
        self.assertTrue(r["value"])
        r2 = token_parity.run(MockClient(vary=True), {"max_tokens": 8, "prompts": ["x"]})
        self.assertFalse(r2["pass"], r2)

    def test_parity_skip_and_match(self):
        # senza reference -> saltato (pass)
        self.assertTrue(parity.run(MockClient(), {"prompts": ["p"]})["pass"])
        # reference identica -> match
        r = parity.run(MockClient(answer="same"), {"_ref_client": MockClient(answer="same"), "prompts": ["p"]})
        self.assertTrue(r["pass"], r)
        # reference diversa -> fail (tolleranza 1.0)
        r2 = parity.run(MockClient(answer="AAA"), {"_ref_client": MockClient(answer="BBB"),
                                                    "prompts": ["p"], "tolerance": 1.0})
        self.assertFalse(r2["pass"], r2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
