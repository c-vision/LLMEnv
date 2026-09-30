# Guida completa a `fastllm`

`fastllm` è la CLI che pilota il motore **LLMEnv** (`LLMEngine/mlx`, la dylib
`libllm_ai_mlx`). Gestisce modelli locali e da Hugging Face, la generazione, un server
OpenAI/Anthropic e l'ispezione del **registro opzioni** del motore.

> **Regola d'oro**: i parametri di funzionamento del motore **non** sono variabili
> d'ambiente. Si passano come `--opt <nome>=<valore>` (registro). Le uniche env sono quelle
> di servizio (token HF, home, prompt offload) elencate in fondo. L'elenco sempre aggiornato
> è `fastllm options`.

---

## 1. Build e installazione

```sh
# dalla radice di FastLLM (che include il motore LLMEnv)
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target fastllm -j8
./build-release/fastllm help
```

Il binario è `build-release/fastllm`.

---

## 2. Struttura dei comandi

```
fastllm <comando> [argomenti] [opzioni]
```

| Gruppo | Comandi |
|---|---|
| Gestione modelli | `pull`, `models`, `list` (`ls`), `add`, `rm`, `show` |
| Generazione | `run`, `chat` |
| Server / utility | `serve`, `ps`, `doctor`, `options`, `help` |

`fastllm help <comando>` (o `fastllm <comando> --help`) mostra le bandiere e gli esempi di
UN comando.

---

## 3. Gestione modelli

### `pull <repo>` — scarica un repo Hugging Face (con resume)
```sh
fastllm pull mlx-community/Qwen3-8B-4bit
fastllm pull org/modello --revision main --cache ~/.cache/huggingface \
    --endpoint https://huggingface.co --dest ~/ai/Models --max-bytes 5000000000 \
    --stall-timeout 60 --stall-retries 3 --repair --quiet
fastllm pull org/modello --offline
```
Bandiere: `--revision`, `--cache`, `--endpoint`, `--dest`, `--max-bytes`, `--stall-timeout`,
`--stall-retries`, `--offline`, `--repair`, `--quiet`.

### `models [<filtro>]` — cosa c'è già nel cache HF (senza rete)
```sh
fastllm models
fastllm models qwen
fastllm models mlx-community/* --cache ~/.cache/huggingface --json
fastllm models --quiet
```

### `list` (`ls`) — modelli disponibili, NUMERATI (registro locale + cache HF)
```sh
fastllm list
fastllm list --cache ~/.cache/huggingface --json
```

### `add <nome> <path>` — registra un modello locale
```sh
fastllm add llama1b ~/ai/Models/Llama-3.2-1B-Instruct-4bit
fastllm add ds4 ~/ds4/ds4flash.gguf --modality chat
fastllm add doodle-ocr ~/ai/Models/gguf/PaddleOCR-VL-1.6-GGUF.gguf \
    --mmproj ~/ai/Models/gguf/PaddleOCR-VL-1.6-GGUF-mmproj.gguf --modality ocr
```
Bandiere: `--mmproj`, `--modality chat|vlm|ocr|audio`.

### `rm <nome|#n>` — rimuove una voce dal registro
```sh
fastllm rm ds4            # solo la voce
fastllm rm 2
fastllm rm doodle-ocr --purge   # cancella anche i file
```

### `show <nome|#n>` — dettagli di una voce
```sh
fastllm show ds4
fastllm show 0
```

---

## 4. Generazione

### `run <nome|#n> [prompt]` — una generazione one-shot
```sh
fastllm run llama1b "Ciao, presentati in una frase."
fastllm run 0 --cache ~/.cache/huggingface --system "Sei conciso." \
    --max-tokens 256 --think --offload-experts
```
Bandiere: `--cache`, `--system`, `--max-tokens`, `--think`, `--offload-experts`.
Senza prompt legge una riga per volta da stdin.

### `chat <nome|#n>` — REPL multi-turno
```sh
fastllm chat llama1b
fastllm chat 0 --cache ~/.cache/huggingface --system "Sei un assistente." \
    --max-tokens 512 --think
```
Comandi REPL: `/exit /quit /clear /new /system /thinking /prompt /compact /help`.

---

## 5. Server e utility

### `serve` — server OpenAI/Anthropic
Endpoint: `/v1/models`, `/v1/chat/completions`, `/v1/messages`.
```sh
fastllm serve
fastllm serve --host 0.0.0.0 --port 8000 --cache ~/.cache/huggingface \
    --max-body 64000000 --offload-experts --quiet
fastllm serve --once     # serve UNA richiesta e esce (smoke test)
```
Bandiere: `--host`, `--port`, `--cache`, `--max-body`, `--once`, `--offload-experts`, `--quiet`.

### `ps` — un `serve` è in ascolto? modelli residenti?
```sh
fastllm ps
fastllm ps --host 127.0.0.1 --port 8000
```

### `doctor [<filtro>]` — verifica l'integrità dei blob nel cache HF
```sh
fastllm doctor
fastllm doctor qwen
fastllm doctor mlx-community/* --cache ~/.cache/huggingface --quiet   # solo problemi
```

### `options` — elenco di TUTTE le opzioni del motore, con esempi
```sh
fastllm options            # lista umana: nome, valore corrente, descrizione, esempio
fastllm options --json     # [{name,value,example,desc}, ...] per tooling
```
Non carica alcun modello. È la fonte di verità del registro (§6).

---

## 6. Opzioni del motore (registro `--opt`)

Si impostano con `--opt <nome>=<valore>` (ripetibile) **prima del load**, e valgono per
`run`, `chat`, `serve`:
```sh
fastllm --opt context=131072 --opt thinking=on run ds4
```

### 6.1 Contesto e memoria
| Opzione | Descrizione | Esempio |
|---|---|---|
| `context` | capienza contesto/KV in token (0 = default del modello) | `--opt context=131072` |
| `prefill` | chunk di prefill (0 = default) | `--opt prefill=2048` |
| `maxtokens` | tetto sui token generati | `--opt maxtokens=4096` |
| `memgov` | memory governor attivo (1/0): tiene ~2 GiB liberi per macOS | `--opt memgov=1` |
| `memgov_reserve_mb` | byte da lasciare liberi per il sistema (MB) | `--opt memgov_reserve_mb=2048` |
| `memgov_grow_min_free_mb` | libero minimo richiesto per crescere le cache (MB) | `--opt memgov_grow_min_free_mb=1024` |
| `memgov_timeout_ms` | attesa massima del governor per la memoria (ms) | `--opt memgov_timeout_ms=30000` |

### 6.2 KV cache (memoria a contesti lunghi)
| Opzione | Descrizione | Esempio |
|---|---|---|
| `kvquant` | KV quantizzata: `off` / `8[:finestra]` (int8) / `auto[:soglia]` / `16` (bf16) | `--opt kvquant=8:8192` |
| `kvq_split_tokens` | KV-quant attn fusa: token per split-K (piccolo = più split) | `--opt kvq_split_tokens=256` |
| `kvq_splits_max` | KV-quant attn fusa: n. massimo di split-K | `--opt kvq_splits_max=64` |
| `kvq_gqa` | KV-quant attn fusa: GQA-shared (1/0, sperimentale) | `--opt kvq_gqa=1` |
| `kvdisk` | tier SSD KV: `off` / `on` / `<path>` (file scratch) | `--opt kvdisk=on` |
| `kvdisk_cap_mb` | budget byte vivi del tier SSD KV in MB (0 = illimitato) | `--opt kvdisk_cap_mb=8192` |
| `prefix_cache` | prefix-cache **persistente**: `auto` (=`~/.cache/llm_engine`, per-modello) / `off` / `<path>` | `--opt prefix_cache=off` |
| `prefix_min_match` | token comuni minimi per riusare un prefisso | `--opt prefix_min_match=8` |
| `prefix_min_save_len` | lunghezza minima (token) del prompt per salvarne la KV | `--opt prefix_min_save_len=64` |

> **Prefix cache (riuso prefisso / TTFT cached)**. Dopo il prefill il prompt viene salvato come
> prefisso; una richiesta successiva che **inizia** con gli stessi token riusa la KV ed esegue il
> prefill **solo della coda**. Di default (`prefix_cache=auto`) la cache è **persistente su disco**
> (per-modello, in `~/.cache/llm_engine/prefix_<hash>.cache`: il file dei prefissi + il blob KV
> `.cache.kv`), quindi il riuso funziona **anche tra riavvii del server** (niente re-prefill).
> Misurato: 1B ~9× su 2ª richiesta identica; 27B/32K prefill ~39 s → restore ~1 s. `off` la tiene
> solo in RAM. Le due soglie (`prefix_min_match`, `prefix_min_save_len`) sono **opzioni di registro**
> (nessuna variabile d'ambiente).

> **KV-quant, trade misurato**: a 8192 token riduce la memoria GPU totale di ~44% ma il decode
> è ~1.7–2× più lento. Utile **memory-bound** / contesti molto lunghi. Default `auto`
> (attiva int8 oltre ~8192 token).

### 6.3 Esperti MoE
| Opzione | Descrizione | Esempio |
|---|---|---|
| `offload` | streaming/cache esperti MoE da disco | `--opt offload=on` |
| `offload_cache_mb` | RAM per la cache esperti (MB) | `--opt offload_cache_mb=16384` |
| `tqmemory` | TurboQuant esperti: 1=packed (poca RAM) / 0=affine-2bit (max velocità) | `--opt tqmemory=0` |

### 6.4 Speculazione / DFlash2
| Opzione | Descrizione | Esempio |
|---|---|---|
| `spec_ngram` | self-speculative n-gram (greedy): `0`=off, `N>=2`=ordine | `--opt spec_ngram=8` |
| `dflash_model` | path del draft DFlash2 (speculazione a contesto persistente) | `--opt dflash_model=~/ai/Models/Qwen3.8-27B-DFlash2` |
| `df2_seq_verify` | DFlash2: verify per-sessione (1 = forza; A/B diagnostico) | `--opt df2_seq_verify=1` |

### 6.5 Sampling e formato
| Opzione | Descrizione | Esempio |
|---|---|---|
| `temp` | temperatura di sampling | `--opt temp=0` |
| `topp` | top-p | `--opt topp=0.95` |
| `minp` | min-p | `--opt minp=0.05` |
| `seed` | seed (-1 = casuale) | `--opt seed=42` |
| `dtype` | `auto` / `f16` / `bf16` | `--opt dtype=bf16` |
| `thinking` | reasoning on/off | `--opt thinking=on` |

> Elenco sempre allineato al binario: `fastllm options` (o `fastllm options --json`).

---

## 7. Ricette comuni

```sh
# Contesto lungo, KV int8 oltre 8k, dtype bf16
fastllm --opt context=131072 --opt kvquant=8:8192 --opt dtype=bf16 run ds4

# Modello MoE più grande della RAM: stream esperti + cache 16 GB
fastllm --opt offload=on --opt offload_cache_mb=16384 run qwenmoe

# Decode speculativo con DFlash2 (greedy, parità col single)
fastllm --opt dflash_model=~/ai/Models/Qwen3.8-27B-DFlash2 run qwen38

# Prefix cache persistente con path esplicito (default: auto in ~/.cache/llm_engine)
fastllm --opt prefix_cache=/tmp/mia_prefix.cache run qwen38

# Greedy riproducibile
fastllm --opt temp=0 --opt seed=42 run llama1b "Due più due?"

# Server su rete locale
fastllm serve --host 0.0.0.0 --port 8000 --quiet
```

---

## 8. Variabili d'ambiente (solo servizio, NON parametri del motore)

| Env | Uso |
|---|---|
| `HF_TOKEN` | token Hugging Face (unico modo per autenticarsi) |
| `FASTLLM_HOME` | dove sta il registro locale (default `~/.fastllm/models.tsv`) |
| `FMLX_MODELS_DIR` | directory alternativa per risolvere i modelli |
| `FASTLLM_NO_OFFLOAD_PROMPT=1` | disabilita la domanda [Y/n] sull'offload automatico |

I parametri di funzionamento del motore **non** sono env: stanno nel registro (`--opt`,
`fastllm options`), regola del progetto LLMEnv.

---

## 9. Riferimenti

- CLI: `FastLLM/src/cli/fmlx_cli.c` (help, `--opt`, `options`).
- Motore / registro opzioni: `LLMEnv/LLMEngine/Options/llm_engine_opts.c`
  (`llm_ai_mlx_opt_count` / `llm_ai_mlx_opt_at` / `llm_ai_mlx_opt_example`).
- Guida opzioni (vault): `01-Projects/LLMEnv/Knowledge/Options-Infrastruttura.md`.
