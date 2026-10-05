# BenchChronos — benchmark coerente di ds4 e FastLLM (via mlx-chronos)

Adatta [mlx-chronos](https://github.com/igurss/mlx-chronos) (suite Apache-2.0 per il benchmark
di motori MLX su Apple Silicon) a **due motori** non previsti:

| nome motore | progetto | endpoint OpenAI-compatibile |
|---|---|---|
| `ds4`  | motore ds4 vendorizzato (`LLMEngine/ds4engine`) | `ds4-server -m <gguf> --host --port` |
| `fmlx` | FastLLM (`fmlx_serve`) | `fastllm serve --host --port` |

Entrambi espongono `/v1/models` e `/v1/chat/completions`, quindi mlx-chronos li misura con lo
stesso protocollo: **TTFT cold/cached, request throughput, sustained, RAM/swap, thermal**,
parametri deterministici (`temperature=0`, `top_p=1`), trial ripetuti, cooldown.

## Perché esiste
Serve una misura **coerente e riproducibile** di ds4 vs FastLLM, per sostituire i numeri
ad-hoc (spesso attribuiti male o rumorosi). Metriche di mlx-chronos + server reali + stesso
modello = confronto onesto.

## Setup
```bash
# 1) mlx-chronos (clone pinnato) come dipendenza Python
git clone --depth 1 https://github.com/igurss/mlx-chronos.git vendor/mlx-chronos

# 2) un venv con le dipendenze (httpx, psutil, pydantic). Esiste gia' ~/ai/env-vllmlx (py3.12).
```

## Uso
```bash
cd LLMEnv/tools/BenchChronos
export PYTHONPATH="$(pwd)/vendor/mlx-chronos:$(pwd)"
PY=~/ai/env-vllmlx/bin/python

$PY -m benchchronos.cli engines                 # elenca i motori (ds4, fmlx inclusi)
$PY -m benchchronos.cli doctor                  # diagnostica hardware/server

# 3) avvia i server (terminali separati)
MODEL=/Users/gaetano/ds4/ds4flash.gguf DS4_BIN=/path/ds4-server bash serve.sh ds4 start
FMLX_BIN=/path/fastllm bash serve.sh fmlx start

# 4) benchmark (stesso modello; ripetuto)
$PY -m benchchronos.cli run --engine ds4  --model <id-modello> --profile baseline --repeat 5 --format all
$PY -m benchchronos.cli run --engine fmlx --model <id-modello> --profile baseline --repeat 5 --format all
$PY -m benchchronos.cli compare results/local/<ds4>.json results/local/<fmlx>.json
```

## Come funziona l'adattamento
I due motori sono registrati **direttamente nel fork** `vendor/mlx-chronos`:
- `mlx_chronos/constants.py`: `ENGINE_NAME_DS4`/`ENGINE_NAME_FMLX` + in `VALID_ENGINE_NAMES`;
- `mlx_chronos/schema.py`: `ds4`/`fmlx` nel `Literal` `EngineName` (mlx-chronos valida
  `Engine.name` contro un Literal hardcoded);
- `mlx_chronos/engines.py`: le classi `DS4ServerEngine` (`ds4-server`, porta 8101) e
  `FMLXEngine` (`fastllm serve`, porta 8102) + in `ENGINES`.

`benchchronos/cli.py` è solo un wrapper (`from mlx_chronos.cli import main`): i `choices` di
argparse (`list(ENGINES)`) vedono i due motori. Perché un **fork** e non un patch a runtime:
mlx-chronos valida `Engine.name` contro un `Literal` pydantic hardcoded, che non si estende
in modo affidabile a runtime.

## Stato: OPERATIVO
- Motori `ds4` e `fmlx` registrati nel fork (`vendor/mlx-chronos`, edit di
  `constants.py`/`schema.py`/`engines.py`).
- `ds4-server` costruito via CMake (`-DLLMENV_BUILD_DS4ENGINE=ON`, target `ds4-server`).
- `fastllm` costruito (target FastLLM `fastllm`).
- **Run ds4 verificato end-to-end** (server + modello + risultato sealato):
  ```
  engine=ds4 ds4-vendored  model=deepseek-v4-flash 4bit
  TTFT cold 2.1s | request throughput 3.41 t/s | decode 6.87 t/s
  ```
- **Confronto decode (coerente)**: ds4 **6.87 t/s** vs motore nativo LLMEngine **5.09 t/s**
  → ds4 ~35% piu' veloce anche nel decode (la precedente "parita'" era un artefatto).

### Confronto a parita' di modello (`ds4flash.gguf`, trial=1)
| engine | TTFT cold | TTFT cached | req t/s | decode t/s |
|---|---|---|---|---|
| ds4 (ds4-server) | 2.10 s | 3.42 s | 3.41 | **6.87** |
| fmlx (fastllm serve) | 1.49 s | **0.78 s** | 3.74 | 5.24 |

- **ds4 vince il decode** (6.87 vs 5.24; fmlx ≈ motore nativo LLMEngine 5.09).
- **fmlx vince il TTFT** e soprattutto **ha una prefix/KV cache funzionante**
  (cached 0.78 << cold 1.49), che **ds4-server NON ha** (cached 3.42 > cold 2.10).
- `fastllm serve` **serve GGUF** (incluso il ds4): il model id va dato come **path assoluto**
  (`/Users/gaetano/ds4/ds4flash.gguf`); il registro `fastllm add` non basta per `serve`.
- Caveat: trial=1 (rumoroso); servono `--trials 5..30` per mediane/variance.

### Nota
- `vendor/mlx-chronos` è **vendorizzato** (Apache-2.0, `.git` rimosso): è la sorgente del fork,
  tracciata nel repo. Riaggiornarlo = riclonare a monte e riapplicare le tre edit (constants/
  schema/engines) descritte sopra.
- Guide/dettaglio nel vault: `AiWiki/01-Projects/LLMEnv/Knowledge/BenchChronos-Benchmark-Coerente.md`.
