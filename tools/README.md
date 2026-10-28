---
tags:
  - project/llmenv
  - benchmark
  - tools
status: active
created: 2026-10-04
---

# Catalogo dei tool — LLMEnv

> [!IMPORTANT] DUE COPIE IDENTICHE — AGGIORNARE SEMPRE **ENTRAMBE**
> 1. **vault**: `AiWiki/01-Projects/LLMEnv/Knowledge/Benchmark-Tools-Catalogo.md` (questa);
> 2. **repo**: `LLMEnv/tools/README.md` (accanto ai tool).
> Devono restare **identiche** (`diff` vuoto). Modificandone una, aggiornare **subito** anche l'altra.
> Scopo: TUTTI gli strumenti di **LLMEngine** e **FastLLM** (+ i condivisi in `LLMEnv`), cosa fa ognuno,
> per chi e quando usarlo.

> [!IMPORTANT] REGOLA (protocollo di startup) — dove va un tool nuovo (agg. 2026-10-07)
> **Ogni tool sta nella `tools/` del SUO progetto** (mai in root, mai in `/tmp`):
> - tool dell'**engine** → **`LLMEngine/tools/`**;
> - tool di **FastLLM** (server/CLI) → **`FastLLM/tools/`**;
> - tool **condivisi** (servono a LLMEngine **e** FastLLM) → **`LLMEnv/tools/`**.
> Alla creazione di un tool, **aggiornare il documento in ENTRAMBE le copie** (vault +
> `LLMEnv/tools/README.md`) — una voce: cosa fa, per chi, come si usa.
> **Eccezione** (unica): i tool **accoppiati a un modulo** — **build CMake/codegen/test di quel
> modulo** (es. `LLMEngine/moe/tools/`) — restano col modulo (spostarli rompe il build) e sono
> elencati sotto. Vedi AGENTS.md §Tool.

> [!IMPORTANT] Dove vivono i tool — consolidato 2026-10-07
> Per **progetto**:
> - **`LLMEngine/tools/`** = tool dell'**engine** (LLMEngine): `decode_profile.c`, `kv_restore_bench.c`,
>   `dflash_*.c`, `mlx_*`, `moe_*`, `llm_autotune.sh`, `llm_plan_gate.sh`, `scale_sweep.sh`, `oracoli/`,
>   `ds4_prefill_phase.py`, `capped_run.py`, `gdn_parity.py`, `dflash_oracle.py` (ex `LLMEnv/tools/llmengine/`
>   + i tool-modulo di `LLMEngine/mlx/`).
> - **`FastLLM/tools/`** = tool del **server/CLI FastLLM**: `ds4_ctx_bench.py`, `eval/`, `bundle.sh`,
>   `moe_gate.py`, `kernel_tune.py`, `json_oracle.py`, `hf_*_probe.c`, `spec_gate.py`, … (ex
>   `LLMEnv/tools/fastllm/` + i tool FastLLM).
> - **`LLMEnv/tools/`** = tool **condivisi** (LLMEngine **e** FastLLM): `build_ai_libs.sh`,
>   `mlx_parity_run.sh`, `gmb_license_gate.py`, `migrate_env.py`, `llm_env_audit.tsv`, `BenchChronos/`.
> Storia: fino al 2026-10-07 i tool engine/FastLLM stavano sotto `LLMEnv/tools/{llmengine,fastllm}/`;
> ora vivono nelle `tools/` del rispettivo progetto (i condivisi restano in `LLMEnv/tools/`).

## Regole d'uso (trasversali)
- **Unità-first / branch-first** (AGENTS §0sexies): misura l'**unità** toccata (kernel, una layer, un
  allocatore), in modo **load-free/deterministico**; il modello intero solo a valle.
- **Overhead-first** (§0septies): separa **OVERHEAD** (allocazioni, view, commit, I/O) da **COMPUTE**.
- **Test chirugico per default**: il SINGOLO file/gamba toccato. Gli **sweep** (dominio intero, tutti i
  modelli) richiedono **permesso esplicito**.
- **Misura assoluta** (wall-time) valida solo a macchina **quieta**; **contatori** (byte, hit/miss,
  dispatch) e **oracoli di parità** valgono sempre.

## Tabella di scelta rapida
| vuoi misurare… | tool |
|---|---|
| prefill/decode di UN modello, tok/s | `LLMEngine/tools/decode_profile.c` |
| TTFT cold vs warm (KV su disco) | `LLMEngine/tools/kv_restore_bench.c` |
| autotune griglia configurazioni | `LLMEngine/tools/llm_autotune.sh` |
| quality/plan gate (eager vs opzioni) | `LLMEngine/tools/llm_plan_gate.sh` |
| **qualità/fedeltà** (parità greedy, needle; poi HALoGEN/lm-eval) | `FastLLM/tools/eval/` (mock o server OpenAI) |
| **prestazioni coerenti** ds4 vs fmlx (TTFT cold/warm, throughput, RAM) | `LLMEnv/tools/BenchChronos/` (mlx-chronos) |
| scalabilità (batch/ampiezza, cliff) | `LLMEngine/tools/scale_sweep.sh` |
| streaming esperti MoE (I/O) | `LLMEngine/tools/moe_stream_bench.c` |
| cache esperti (LRU vs routing-aware) | `LLMEnv/LLMEngine/moe/tests/test_moe_cache_bench.c` |
| spec/decode DFlash2/n-gram | `LLMEngine/tools/dflash_*.c` |
| DS4 prefill/decode (nostro engine) | `LLMEnv/LLMEngine/mlx/ds4/tests/llm_dsv4_prefill_bench.c` |

---

## LLMEnv

### `tools/` (root — tool per **ENTRAMBI**: LLMEngine + FastLLM)
- **`build_ai_libs.sh`** — costruisce le librerie AI native (static/dylib) per i consumer. *(build, non misura)*
  confronto degli output (oracolo di parità cross-build).
- **`gmb_license_gate.py`** — gate **licenza** (perimetro Apache LLMEnv). *(gate, non misura)*
- **`migrate_env.py`** — migra `getenv` → registro opzioni. *(refactor, non misura)*
- **`llm_env_audit.tsv`** — dati di audit env → registro.
- **`BenchChronos/`** — benchmark **prestazioni** (mlx-chronos; motori ds4 + fmlx). *V. sotto.*

### `LLMEngine/tools/` — tool dell'**ENGINE** (LLMEngine)
- **`decode_profile.c`** — F18: profilo di decode riutilizzabile (**tok/s + ms/token + memoria MLX**) via ABI.
- **`kv_restore_bench.c`** — D15: **TTFT cold (prefill) vs warm (restore da cache)**.
- **`dflash_parity.c`** — parità **DFlash2 vs greedy AR** (unità-first, senza dispatch).
- **`dflash_stream.c`** — DFlash2 a **contesto persistente**: AR vs one-shot vs streaming (warm).
- **`dflash_multi.c`** — verifica `dflash2_generate_multi` (N sessioni batched) vs N single.
- **`mlx_parity_driver.c`** — driver di **parità** (dlopen dylib motore + stesso prompt).
- **`mlx_embed_check.c`** / **`mlx_topk_check.c`** — accessor ABI (`embed_ids`, `token_topk`).
- **`moe_ep_run.c`** / **`moe_ep_socket.c`** — **EP distribuito** multi-processo / cross-nodo (§4.6).
- **`moe_stream_bench.c`** — "truth harness" dello **streaming esperti MoE** (I/O, hit/miss).
- **`tq_moe_batch_repro.c`** — repro deterministico di `forward_tokens_batch` su n piccolo (MoE TurboQuant).
- **`llm_autotune.sh`** — **autotuner (I5)**: decode su una **griglia** di configurazioni (registro).
- **`llm_plan_gate.sh`** — **plan/quality gate (§4.5)**: decode **EAGER vs opzioni del registro**.
- **`scale_sweep.sh`** — **scaling**: batch/ampiezza crescente → **cliff non monotoni**.
- **`arch_lab.c`** / **`arch_lab.sh`** — **A3-lab (OpenArch)**: funzioni reali del motore (ds4/dsv41, C)
  vs **riferimento CPU double** — 9 moduli: `linear · rms_norm · qk-norm · rope · gqa · swiglu · router ·
  swa · mla`. Model-free, deterministico.
- **`oracoli/`** — sonde **model-free** dell'engine (non tracciate). *V. sotto.*

### Tool MLX importati da GMBasic (2026-10-28)

> [!NOTE] Origine: `GMBasicMonoRepo/migration/tools-unclassified/` (backup in
> `/tmp/llmengine-tools-moved-2026-10-28/`). Non hanno dipendenze da GMBasic: vivono qui
> perché misurano/verificano **MLX/LLMEngine**. Gli oracoli indipendenti restano Python
> (preferibile riscriverli in C quando il calcolo è puro); i probe/bench misurano l'engine,
> non GMBasic.

**`LLMEngine/tools/`** (40 Python): bench/prefill/decode `bench_mlxlm.py`, `bench_rapidmlx.py`,
`bench_table.py`, `core_bench.py`, `ctx_table.py`, `mlx_prefill_ref.py`, `mlx_decode_ref.py`,
`mlx_inplace_append_bench.py`, `ab_ctx_mlxlm.py`, `gmb-ctx-mlxlm.py`, `llm_eval_ds4.py`;
probe/kernel/KV `mlx_inplace_probe.py`, `mlx_kernel_inplace_probe.py`, `mlx_kvq_attn_probe.py`,
`mlx_quant_dequant_ref.py`, `mlx_shared_buffer_probe.py`, `kv_append_probe.py`,
`kv_evict_lab.py`, `kv_quant_lab.py`, `kvseg_refactor.py`, `df2_xctrace_agg.py`,
`ref_mlx_graph_diff.py`; MoE/quant/ref `moe_mxfp4_fastbranch_probe.py`,
`moe_mxfp4_floor_probe.py`, `moe_mxfp4_gather_probe.py`, `q4_build_ngram_combined.py`,
`q4_build_ple_manifest.py`, `q4_ple_ref.py`, `q4_ref_decode_official.py`,
`q4_ref_layers_official.py`, `q4_ref_lazy.py`, `q4_ref_nople.py`, `q4_ref_plevalid.py`,
`qwen4exp_ngram_ref.py`, `mixtral_parity_layers.py`, `spike_mlx_bert_sparse.py`,
`gmb_mlx_dtype_provenance.py`, `mlx_functions_doc.py`, `evaluate_consistency_auroc.py`,
`run_auroc_eval.py`.

**`LLMEngine/ds4/tools/reference/`** (33 Python): `gen_dsv41_*.py` — oracoli/reference DS4
(Python ammesso; C preferibile quando il calcolo è puro).

**Regola**: un tool MLX nuovo nasce in `LLMEngine/tools/` (o `ds4/tools/reference/`), non in
GMBasic; aggiornare **entrambe** le copie di questo catalogo (vault + `LLMEnv/tools/README.md`).
**`LLMEngine/tools/`** (12 script): `ab_all_ctx_mlxlm.sh`, `ab_engine_mlxlm.sh`,
`ab_engine_rapid.sh`, `ab_models_under30g_rapid.sh`, `ab_models_under30g.sh`,
`ab_order_control.sh`, `ab_rapid_sweep.sh`, `attn_decode_bench.sh`, `bench3_compare.sh`,
`bench4_compare.sh`, `bench4_paired.sh`, `bench4_stable.sh`.

**`LLMEngine/mlx/ds41-native/tools/`** (28 script): `dsv4_test.sh` + `dsv41_*_test.sh`
(test/parità DS4.1; restano accanto al modulo perché accoppiati al build del modulo).


### `FastLLM/tools/` — tool del **SERVER/CLI FastLLM**
- **`ds4_ctx_bench.py`** — bench **DS4** (nostro engine via `fastllm`): **prefill t/s e decode tok/s
  separati** per frontiere di contesto (metodo A/B).
- **`eval/`** — harness **qualità/fedeltà** (mock o server OpenAI). *V. sotto.*

### `LLMEngine/mlx/`
- **`ds4/tests/llm_dsv4_prefill_bench.c`** — **prefill puro a chunk** su N token sintetici (un solo
  caricamento), t/s. Supporta il profilo hotness (`dsv4_n_out/n_in/n_topk`).
- **`goldens/specbench.c`** — **bench dello speculative batching** multi-sessione (B) vs single, a parità
  di token totali.
- **`tests/`** — decine di unittest/probe: `llm_batch_scale_probe`, `llm_encode_cost_test`,
  `llm_qmm_smallm_probe`, `llm_retention_probe`, `llm_kvdisk_*`, `llm_mlx_pagesel_test`, `llm_gguf_test`, …
  (probe **per-unità**, load-free).

### `LLMEngine/moe/tests/`
- **`test_moe_cache_bench.c`** — **M7**: LRU vs routing-aware sullo **stesso trace** (hit rate + bytes).
- `test_moe_prof.c`, `test_moe_async.c`, `test_moe_prefetch.c`, `test_moe_mem.c`, … — probe per-unità dello
  streaming esperti (async, prefetch, memoria).

### Tool interni ai moduli (LLMEngine) — restano col modulo (NON in `tools/`)
> **Eccezione alla regola**: sono **accoppiati al modulo** (target CMake / codegen / test di quel modulo);
> spostarli in `tools/` **romperebbe il build**. Documentati qui per completezza.
- `LLMEngine/moe/tools/{moe_sim.c, moe_iobw.c}` — replay offline cache + I/O bandwidth (target CMake `moe_sim`/`moe_iobw`).
- `LLMEngine/mlx/ds4/tools/` — `gen_dsv4_mmid_metal.py`/`gen_dsv4_metal_tables.py` (**codegen** dei header `.h` generati) e `llm_dsv4_store_policy_test.sh` (oracolo policy store).
- `LLMEngine/mlx/ds41-native/tools/dsv41_dspark_st_test.sh` — build/test DSpark (`mtp.*` + `forward_spec`).
- `LLMEngine/mlx/goldens/dflash2/{dflash_oracle.py, make_round0_bin.py}` — goldens/oracolo **DFlash2** (round 0).
- `LLMEngine/mlx/gdn_parity.py` — **parità GDN** (nostro kernel vs riferimento MLX).

### `tools/BenchChronos/` (era `LLMEnv/BenchChronos/`)
- Harness Python (fork di **mlx-chronos**, Apache) per **benchmark coerenti e ripetibili** (schema, stats,
  integrity, compare, energy profile) con i motori `ds4` (8101) e `fmlx` (8102) registrati.
  `benchchronos/cli.py` + `serve.sh`. Vedi nota `BenchChronos-Benchmark-Coerente.md`.
  **Asse: prestazioni** (TTFT cold/warm, throughput, decode, RAM/swap).

### `FastLLM/tools/eval/` — qualità/fedeltà (NUOVO 2026-10-05; F1–F5)
- Harness Python **stdlib-only** (nessun import del motore; HTTP OpenAI-compatibile o backend `mock`).
  `run_eval.py` — suite: **`local`** (5 gate **deterministici**: `id_parity` testo greedy identico,
  `needle` retrieval, `ppl` perplexità bits/token con **banda**, `parity` greedy vs **reference**,
  `token_parity` parità **token-level**/GEN_IDS),
  **`lm-eval`** (lm-evaluation-harness via subprocess, accuracy, F4), **`halogen`** (factualità:
  recall atomic-unit con **verifier** locale a regole; `openai` opt-in + caching, F3/F5).
  **Golden/regressione** (`--save-golden`/`--golden`). Report JSON+MD, `tests/test_smoke.py`.
  Guida: `Guide/Eval-Harness-Guida.md`. **Asse ortogonale** a `tools/BenchChronos/`.
  **Nessun impatto sulle prestazioni dell'engine.** Rif.: todo `(solved) - LLM-Eval-Harness-...`.

### `LLMEngine/tools/oracoli/` — sonde model-free (era `LLMEnv/oracoli/`; NON tracciata)

> **Cos'è.** La raccolta di **sonde-oracolo** dell'engine: programmi **self-contained** che costruiscono
> grafi sintetici e **non caricano alcun modello** (nessun peso, nessuna run).
> **A cosa serve.** Misurare e **validare** kernel/scelte di dispatch in modo **deterministico e sicuro**,
> quando la macchina è carica e i tok/s **non sono interpretabili**. Produce i soli numeri spendibili:
> conteggi (nodi/byte/token), **parità ID**, picco RSS, coerenza di un kernel contro un riferimento.
> **Come si usa.** Build **una tantum**: `bash LLMEngine/tools/oracoli/build_probes.sh [build_dir]`
> (default `FastLLM/build-release`; richiede `_deps/mlx_c-build/libmlxc.a` + la dylib in `llmengine_mlx/`),
> poi si esegue la singola sonda. **Regola**: guardare sempre il **rapporto A/B** tra due misure,
> mai il valore assoluto isolato. Metodo completo, principi, watchdog e catalogo delle sonde: **[[:oracoli]]**.
- Contenuto tipico: parità/algoritmi di kernel (MMA/FA2, gemv/slots, GDN), banda/dispatch, probe di
  coerenza e **self-test** delle unità; `build_probes.sh` le compila con le variabili del build.
- **Deterministiche, load-free.** Non tracciata in git ().

---

## FastLLM

> Repo **separato** (`/Users/gaetano/Documents/Projects/FastLLM`). I suoi tool vivono nel **suo**
> `FastLLM/tools/` (non in `LLMEnv/tools/`). Documentati qui per il **quadro completo** (FastLLM è il
> server/CLI degli engine).

### `FastLLM/tools/` — tool del server/CLI FastLLM
- **`spec_gate.py`** — Promotion Gate del decode **speculativo** (generale, metodo Gufo).
- **`moe_gate.py`** — Promotion Gate del fix **forward batch MoE** (n=2..7).
- **`mut_prove.py`** — prova che un'asserzione **mutata (ROSSA)** lo sia davvero (mutation prove).
- **`kernel_tune.py`** — harness di **autotuning dei kernel** (model-free, Gufo #6).
- **`gen_codebooks.py`** — genera/verifica le **tabelle codebook ggml** di FastLLM.
- **`longctx_bench.sh`** — bench **long-context decode** con accelerazioni.
- **`df2_k_ab.sh`** — A/B del **K adattivo DFlash2** + wall-time (solo finestra idle).
- **`fuse_pointwise_ab.sh`** — A/B della fusione **pointwise** via `compiled_decode` whole-step.
- **`bundle.sh`** — bundle **rilocabile** + update atomico + rollback dell'engine.
- **`fastllm_smoke_models.sh`** — **smoke** dei modelli.
- **`json_oracle.py`** + **`json_diff.c`** — oracolo lane **JSON** (codifica canonica vs nostro parser).
- **`hf_probe.c`** / **`hf_fetch_probe.c`** / **`hf_manifest_probe.c`** / **`hf_stall_probe.c`** — probe lane **HF** (cache locale, download reale, manifesto, watchdog di stall).
- **`vision_ocr.c`** — tool lane **VISION/OCR** (VLM/OCR multimodale).
- **`gmb_seq_unbad.py`** — refactor `bad |= A | B;` → early-return negli adapter.

---


## Debug/profilo in-process (gated, per-unità)
- **`ds4`**: opts registro **`dsv4_profile`** → profiler per-fase + **tempo GPU reale** (timestamp cb) +
  sotto-timer host MoE (`route`/`store`/`metal`) + **dispatch count**. *(no-op se 0 — zero overhead)*
- **Motore MLX/MoE**: opzioni `dsv4_prefill_mm`, `dsv4_gpu_hc`, `dsv4_prefill_chunk`, `dsv4_hotlist_topk` …

> [!NOTE] Vedi anche: `DS4-slow-Dove-va-il-tempo-Decode.md` (ripartizione CPU/GPU/decode),
> `BenchChronos-Benchmark-Coerente.md`, AGENTS §0sexies/§0septies.
