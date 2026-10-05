# LLMEnv — Eval Harness (qualita'/fedelta')

Tool **separato** in `LLMEnv/tools/fastllm/eval/` (fuori dal CMake, nessun import del motore): valuta la
**qualita'/fedelta'** di un modello/engine, asse **ortogonale** a `tools/BenchChronos/` e agli
[[oracoli]] (che misurano le prestazioni).

- **Deterministico** dove possibile: i gate "locali" non usano giudici LLM.
- **stdlib-only** (urllib/json/argparse): nessuna venv/pacchetto richiesto.
- Parla col motore **solo** via HTTP OpenAI-compatibile, oppure col backend `mock` (test, no rete).

Doc/piano completo: `01-Projects/LLMEnv/Todo/(solved) - LLM-Eval-Harness-Qualita-e-Allucinazioni.md`.

## Uso

```bash
# smoke / CI: nessun motore, backend mock, deterministico
python3 tools/fastllm/eval/run_eval.py --suite local --backend mock

# contro un server locale OpenAI-compatibile (FastLLM serve / mlx_lm.server)
python3 tools/fastllm/eval/run_eval.py --suite local --backend http \
    --base-url http://127.0.0.1:8080 --model local --reps 3

# suite lm-eval (lm-evaluation-harness via subprocess; se `lm_eval` manca -> skipped)
python3 tools/fastllm/eval/run_eval.py --suite lm-eval \
    --base-url http://127.0.0.1:8080 --model local

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
| `parity` | **parità greedy engine vs reference** (`reference.base_url`, es. `mlx_lm.server`): testo identico o `prefix_ratio >= tolerance`. Saltato se la reference non è configurata. |
| `token_parity` | **parità TOKEN-level (GEN_IDS)**: due run greedy → lista **token identica** (`logprobs.tokens` della risposta). Più stretta del testo; skipped se il server non espone i token. |

## Stato (slice 1 F1 + slice 2 F2-parziale)
- [x] runner CLI + adapter engine (mock + HTTP OpenAI-compatibile: chat + completions/logprobs).
- [x] 4 gate locali deterministici + report JSON/MD: `id_parity`, `needle`, `ppl` (banda), `parity` (vs reference).
- [x] **golden/regressione** (`--save-golden`/`--golden`/`--golden-tol`).
- [x] test con mock (id_parity NON-determinismo, needle, ppl banda, parity skip/match/fail, golden roundtrip).
- [ ] (todo F2) parità **`GEN_IDS`** engine-interna (token-level) e **teacher-forcing** (il server NON
  supporta `echo`/logprobs del prompt → serve un canale engine-side dedicato).
- [x] **F3 HALoGEN-subset** (factualità, verifier locale) e **F5 verifier+caching** (local/openai).
- [x] **F4 lm-evaluation-harness** (subprocess).
- [ ] (todo) verifier esterni opt-in + caching (F5).

## Suite `halogen` (F3) — factualità (verifier locale)
`--suite halogen` misura la **factualità**: per ogni caso (`prompt` + `facts` = atomic-unit attese) chiede
la risposta greedy e verifica con il **verifier** quante unità sono supportate → `recall`
(= 1 − hallucination grezza), `response_ratio`. Gate = `min_recall` (assente = informativo). Verifier
**default `local`** (regola deterministica, offline); `openai` = opt-in (F5) con caching dei verdetti.
Config `halogen`: `cases`, `max_tokens`, `min_recall`, `verifier.backend`.

## Suite `lm-eval` (F4) — lm-evaluation-harness (accuracy)
`--suite lm-eval` esegue il CLI **`lm_eval`** come **subprocess** contro l'endpoint OpenAI
(`--model local-completions --model_args base_url=…,model=…`), legge l'accuracy per task dai risultati.
Config `lm_eval`: `bin` (default `lm_eval`), `base_url`, `model`, `tasks` (es. `gsm8k`), `limit`,
`batch_size`, `min_accuracy` (soglia; dict per-task o float; assente = informativo). Se `lm_eval`
**non è installato** → gate **skipped**. **Tool esterno/offline: nessun impatto sulle prestazioni.**

## Golden / regressione (F6-lite)
`--save-golden <file>` salva i valori dei gate come **riferimento**; `--golden <file>` confronta la run
col riferimento (`bool` esatto; numerico `|Δ|/max(1,|b|) <= --golden-tol`, default `0.05`). Serve a
**rilevare regressioni di qualità** dopo un'ottimizzazione (stesso modello/opzioni): salvi il golden sul
baseline, poi confronti dopo il cambiamento.

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
