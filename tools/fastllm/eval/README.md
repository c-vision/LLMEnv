# LLMEnv — Eval Harness (qualita'/fedelta')

Tool **separato** in `LLMEnv/tools/fastllm/eval/` (fuori dal CMake, nessun import del motore): valuta la
**qualita'/fedelta'** di un modello/engine, asse **ortogonale** a `tools/BenchChronos/` e agli
[[oracoli]] (che misurano le prestazioni).

- **Deterministico** dove possibile: i gate "locali" non usano giudici LLM.
- **stdlib-only** (urllib/json/argparse): nessuna venv/pacchetto richiesto.
- Parla col motore **solo** via HTTP OpenAI-compatibile, oppure col backend `mock` (test, no rete).

Doc/piano completo: `01-Projects/LLMEnv/Todo/(open) - LLM-Eval-Harness-Qualita-e-Allucinazioni.md`.

## Uso

```bash
# smoke / CI: nessun motore, backend mock, deterministico
python3 tools/fastllm/eval/run_eval.py --suite local --backend mock

# contro un server locale OpenAI-compatibile (FastLLM serve / mlx_lm.server)
python3 tools/fastllm/eval/run_eval.py --suite local --backend http \
    --base-url http://127.0.0.1:8080 --model local --reps 3

# test unitari del harness
python3 tools/fastllm/eval/tests/test_smoke.py
```

Report JSON + Markdown in `tools/fastllm/eval/reports/` (gitignorati). Esito = PASS/FAIL complessivo.

> **Relazione con `tools/BenchChronos/`** (l'altro tool Python della repo): `BenchChronos` misura le
> **prestazioni** (TTFT/throughput/decode/RAM, protocollo mlx-chronos); questo harness misura la
> **qualita'/fedelta'** (gate deterministici +, in seguito, HALoGEN/lm-eval). Assi **ortogonali**.

## Suite `local` (gate deterministici)

| gate | cosa verifica |
|---|---|
| `id_parity` | due run greedy (temp 0) dello stesso prompt → **testo identico** (non-determinismo/atomic-float/ordine riduzione). |
| `needle` | un fatto sepolto in contesto di rumore deve essere **recuperato** (greedy). |
| `ppl` | **perplexità (bits/token)** dei token greedy via `/v1/completions`+logprobs; gate = dentro la **banda pre-registrata** (`band`, F2bis). |

## Stato (slice 1 F1 + slice 2 F2-parziale)
- [x] runner CLI + adapter engine (mock + HTTP OpenAI-compatibile: chat + completions/logprobs).
- [x] 3 gate locali deterministici + report JSON/MD: `id_parity`, `needle`, `ppl` (banda).
- [x] test con mock (id_parity NON-determinismo, needle, ppl banda).
- [ ] (todo F2) parità **ID** engine-interna (`GEN_IDS`) e **teacher-forcing** gate; KL engine-vs-reference.
- [ ] (todo) HALoGEN (subset, verifier locali) (F3); lm-evaluation-harness (F4).
- [ ] (todo) verifier esterni opt-in + caching (F5).

## Layout
```
tools/fastllm/eval/
├── run_eval.py            # CLI
├── config/eval.json       # modelli/suite/soglie (NESSUNA chiave qui)
├── adapters/engine_server.py  # mock + OpenAI-compatibile (stdlib)
├── suites/local/{id_parity,needle}.py
├── tests/test_smoke.py
└── reports/               # output (gitignorato)
```
