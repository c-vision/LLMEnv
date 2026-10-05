# LLMEnv — Eval Harness (qualita'/fedelta')

Tool **separato** (fuori dal CMake, nessun import del motore): valuta la **qualita'/**fedelta' di un
modello/engine, asse **ortogonale** agli [[oracoli]] (che misurano le prestazioni).

- **Deterministico** dove possibile: i gate "locali" non usano giudici LLM.
- **stdlib-only** (urllib/json/argparse): nessuna venv/pacchetto richiesto.
- Parla col motore **solo** via HTTP OpenAI-compatibile, oppure col backend `mock` (test, no rete).

Doc/piano completo: `01-Projects/LLMEnv/Todo/(open) - LLM-Eval-Harness-Qualita-e-Allucinazioni.md`.

## Uso

```bash
# smoke / CI: nessun motore, backend mock, deterministico
python3 eval/run_eval.py --suite local --backend mock

# contro un server locale OpenAI-compatibile (FastLLM serve / mlx_lm.server)
python3 eval/run_eval.py --suite local --backend http \
    --base-url http://127.0.0.1:8080 --model local --reps 3

# test unitari del harness
python3 eval/tests/test_smoke.py
```

Report JSON + Markdown in `eval/reports/` (gitignorati). Esito = PASS/FAIL complessivo.

## Suite `local` (gate deterministici)

| gate | cosa verifica |
|---|---|
| `id_parity` | due run greedy (temp 0) dello stesso prompt → **testo identico** (non-determinismo/atomic-float/ordine riduzione). |
| `needle` | un fatto sepolto in contesto di rumore deve essere **recuperato** (greedy). |

## Stato (slice 1 — F1 del todo)
- [x] runner CLI + adapter engine (mock + HTTP OpenAI-compatibile).
- [x] 2 gate locali deterministici + report JSON/MD.
- [x] test con mock (id_parity NON-determinismo, needle).
- [ ] (todo) altri gate locali: KL/Δppl, parità `GEN_IDS`, teacher-forcing (F2).
- [ ] (todo) HALoGEN (subset, verifier locali) (F3); lm-evaluation-harness (F4).
- [ ] (todo) verifier esterni opt-in + caching (F5).

## Layout
```
eval/
├── run_eval.py            # CLI
├── config/eval.json       # modelli/suite/soglie (NESSUNA chiave qui)
├── adapters/engine_server.py  # mock + OpenAI-compatibile (stdlib)
├── suites/local/{id_parity,needle}.py
├── tests/test_smoke.py
└── reports/               # output (gitignorato)
```
