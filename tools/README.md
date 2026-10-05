---
tags:
  - project/llmenv
  - project/gmbasic
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
> (La vecchia copia in `GMBasic/Knowledge/` è stata rimossa.)
> Scopo: TUTTI gli strumenti della repo `LLMEnv`, cosa fa ognuno, per chi (LLMEngine / FastLLM / entrambi)
> e quando usarlo.

> [!IMPORTANT] REGOLA (protocollo di startup) — tool nuovi
> **Ogni nuovo tool va SEMPRE in `LLMEnv/tools/`** (mai nella root, mai in `/tmp`, mai dentro
> `LLMEngine/`/`FastLLM/`). Posizione: `tools/` per quelli che servono a **entrambi** (LLMEngine e
> FastLLM); `tools/llmengine/` per quelli dell'**engine**; `tools/fastllm/` per quelli di **FastLLM**.
> Alla creazione di un tool, **aggiornare il documento in ENTRAMBE le copie** (vault +
> `LLMEnv/tools/README.md`) — una voce: cosa fa, per chi, come si usa.
> **Eccezione** (unica): i tool **accoppiati a un modulo** — **build CMake/codegen/test di quel
> modulo** (es. `LLMEngine/moe/tools/`, `LLMEngine/mlx/ds4-slow/tools/`) — restano col modulo
> (spostarli rompe il build) e sono elencati sotto. Vedi AGENTS.md §Tool.

> [!IMPORTANT] Dove vivono i tool (LLMEnv) — consolidato 2026-10-05
> **TUTTI i tool di `LLMEnv` stanno sotto `LLMEnv/tools/`**, ripartiti **per motore**:
> - **`tools/`** (root) = tool per **ENTRAMBI** (LLMEngine **e** FastLLM): `build_ai_libs.sh`,
>   `mlx_parity_run.sh`, `gmb_license_gate.py`, `migrate_env.py`, `llm_env_audit.tsv`, `BenchChronos/`.
> - **`tools/llmengine/`** = tool dell'**engine**: `decode_profile.c`, `kv_restore_bench.c`,
>   `dflash_*.c`, `mlx_*`, `moe_*`, `llm_autotune.sh`, `llm_plan_gate.sh`, `scale_sweep.sh`, `oracoli/`.
> - **`tools/fastllm/`** = tool del **server/CLI FastLLM**: `ds4_ctx_bench.py`, `eval/`.
> Storia: prima `BenchChronos/` e `oracoli/` erano nella root; `oracoli/` è stato spostato sotto
> `tools/` poi in `tools/llmengine/` (LLMEnv commit `fb053d7` e successivo).

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
| prefill/decode di UN modello, tok/s | `GMBasic/test_mlx_bench` (prefill/decode/ctxdecode), `LLMEnv/tools/llmengine/decode_profile.c` |
| TTFT cold vs warm (KV su disco) | `LLMEnv/tools/llmengine/kv_restore_bench.c` |
| parità motore LLMEnv vs GMBasic | `LLMEnv/tools/mlx_parity_run.sh` + `mlx_parity_driver.c` |
| A/B GMBasic vs mlx_lm / rapidMLX | `GMBasic/tools/ab_*.sh`, `bench*.sh`, `bench_mlxlm.py`, `bench_rapidmlx.py` |
| autotune griglia configurazioni | `LLMEnv/tools/llmengine/llm_autotune.sh`, `GMBasic/tools/gmb_defaults_sweep.sh` |
| quality/plan gate (eager vs opzioni) | `LLMEnv/tools/llmengine/llm_plan_gate.sh` |
| **qualità/fedeltà** (parità greedy, needle; poi HALoGEN/lm-eval) | `LLMEnv/tools/fastllm/eval/` (mock o server OpenAI) |
| **prestazioni coerenti** ds4 vs fmlx (TTFT cold/warm, throughput, RAM) | `LLMEnv/tools/BenchChronos/` (mlx-chronos) |
| scalabilità (batch/ampiezza, cliff) | `LLMEnv/tools/llmengine/scale_sweep.sh`, `GMBasic/tools/mlx_batch_test.sh` |
| streaming esperti MoE (I/O) | `LLMEnv/tools/llmengine/moe_stream_bench.c`, `GMBasic/tools/moe_stream_bench` |
| cache esperti (LRU vs routing-aware) | `LLMEnv/LLMEngine/moe/tests/test_moe_cache_bench.c` |
| spec/decode DFlash2/n-gram | `LLMEnv/tools/llmengine/dflash_*.c`, `GMBasic/tools/gmb_mlx_dflash2_bench.c`, `gmb_mlx_ngram_bench.c` |
| DS4 prefill/decode (nostro engine) | `LLMEnv/LLMEngine/mlx/ds4-slow/tests/llm_dsv4_prefill_bench.c`, `GMBasic/tools/ds4_ctx_bench.py` |
| DS4 vendored (bridge) | `LLMEnv/LLMEngine/ds4engine/ds4_bench.c`, `ds4_bridge_bench.c` |
| op singole (kernel) load-free | `GMBasic/tools/decode_bench.c`, `gdn_bench.c`, `gemv_m1_bench.c`, `gmb_mlx_*_bench.c` |
| statement coverage C99Native | `GMBasic/tools/gmb_stmt_gate.sh` |

---

## LLMEnv

### `tools/` (root — tool per **ENTRAMBI**: LLMEngine + FastLLM)
- **`build_ai_libs.sh`** — costruisce le librerie AI native (static/dylib) per i consumer. *(build, non misura)*
- **`mlx_parity_run.sh`** — **parità MLX** (LLMEnv vs GMBasic): stesso modello, **stessi prompt ids**,
  confronto degli output (oracolo di parità cross-build).
- **`gmb_license_gate.py`** — gate **licenza** (perimetro Apache LLMEnv). *(gate, non misura)*
- **`migrate_env.py`** — migra `getenv` → registro opzioni. *(refactor, non misura)*
- **`llm_env_audit.tsv`** — dati di audit env → registro.
- **`BenchChronos/`** — benchmark **prestazioni** (mlx-chronos; motori ds4 + fmlx). *V. sotto.*

### `tools/llmengine/` — tool dell'**ENGINE** (LLMEngine)
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
- **`arch_lab.c`** / **`arch_lab.sh`** — **A3-lab (OpenArch)**: funzioni reali del motore (ds4-slow/dsv41, C)
  vs **riferimento CPU double** — 9 moduli: `linear · rms_norm · qk-norm · rope · gqa · swiglu · router ·
  swa · mla`. Model-free, deterministico.
- **`oracoli/`** — sonde **model-free** dell'engine (non tracciate). *V. sotto.*

### `tools/fastllm/` — tool del **SERVER/CLI FastLLM**
- **`ds4_ctx_bench.py`** — bench **DS4** (nostro engine via `fastllm`): **prefill t/s e decode tok/s
  separati** per frontiere di contesto (metodo A/B).
- **`eval/`** — harness **qualità/fedeltà** (mock o server OpenAI). *V. sotto.*

### `LLMEngine/ds4engine/` (vendored)
- **`ds4_bench.c`** — bench del motore **ds4 vendorizzato** (profili interni del `ds4`).
- **`ds4_bridge_bench.c`** — misura **rigorosa via il bridge token-level**: `<gguf> <n_prefill> <ctx> <stream>`.

### `LLMEngine/mlx/`
- **`ds4-slow/tests/llm_dsv4_prefill_bench.c`** — **prefill puro a chunk** su N token sintetici (un solo
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
- `LLMEngine/mlx/ds4-slow/tools/` — `gen_dsv4_mmid_metal.py`/`gen_dsv4_metal_tables.py` (**codegen** dei header `.h` generati) e `llm_dsv4_store_policy_test.sh` (oracolo policy store).
- `LLMEngine/mlx/ds41-native/tools/dsv41_dspark_st_test.sh` — build/test DSpark (`mtp.*` + `forward_spec`).
- `LLMEngine/ds4engine/metal/generate_mxfp4_half_lut.py` — codegen della **LUT MXFP4** Metal (vendored).
- `LLMEngine/mlx/goldens/dflash2/{dflash_oracle.py, make_round0_bin.py}` — goldens/oracolo **DFlash2** (round 0).
- `LLMEngine/mlx/gdn_parity.py` — **parità GDN** (nostro kernel vs riferimento MLX).

### `tools/BenchChronos/` (era `LLMEnv/BenchChronos/`)
- Harness Python (fork di **mlx-chronos**, Apache) per **benchmark coerenti e ripetibili** (schema, stats,
  integrity, compare, energy profile) con i motori `ds4` (8101) e `fmlx` (8102) registrati.
  `benchchronos/cli.py` + `serve.sh`. Vedi nota `BenchChronos-Benchmark-Coerente.md`.
  **Asse: prestazioni** (TTFT cold/warm, throughput, decode, RAM/swap).

### `tools/fastllm/eval/` — qualità/fedeltà (NUOVO 2026-10-05; F1–F5)
- Harness Python **stdlib-only** (nessun import del motore; HTTP OpenAI-compatibile o backend `mock`).
  `run_eval.py` — suite: **`local`** (5 gate **deterministici**: `id_parity` testo greedy identico,
  `needle` retrieval, `ppl` perplexità bits/token con **banda**, `parity` greedy vs **reference**,
  `token_parity` parità **token-level**/GEN_IDS),
  **`lm-eval`** (lm-evaluation-harness via subprocess, accuracy, F4), **`halogen`** (factualità:
  recall atomic-unit con **verifier** locale a regole; `openai` opt-in + caching, F3/F5).
  **Golden/regressione** (`--save-golden`/`--golden`). Report JSON+MD, `tests/test_smoke.py`.
  Guida: `Guide/Eval-Harness-Guida.md`. **Asse ortogonale** a `tools/BenchChronos/`.
  **Nessun impatto sulle prestazioni dell'engine.** Rif.: todo `(solved) - LLM-Eval-Harness-...`.

### `tools/llmengine/oracoli/` — sonde model-free (era `LLMEnv/oracoli/`; NON tracciata)

> **Cos'è.** La raccolta di **sonde-oracolo** dell'engine: programmi **self-contained** che costruiscono
> grafi sintetici e **non caricano alcun modello** (nessun peso, nessuna run).
> **A cosa serve.** Misurare e **validare** kernel/scelte di dispatch in modo **deterministico e sicuro**,
> quando la macchina è carica e i tok/s **non sono interpretabili**. Produce i soli numeri spendibili:
> conteggi (nodi/byte/token), **parità ID**, picco RSS, coerenza di un kernel contro un riferimento.
> **Come si usa.** Build **una tantum**: `bash tools/llmengine/oracoli/build_probes.sh [build_dir]`
> (default `FastLLM/build-release`; richiede `_deps/mlx_c-build/libmlxc.a` + la dylib in `llmengine_mlx/`),
> poi si esegue la singola sonda. **Regola**: guardare sempre il **rapporto A/B** tra due misure,
> mai il valore assoluto isolato. Metodo completo, principi, watchdog e catalogo delle sonde: **[[:oracoli]]**.
- Contenuto tipico: parità/algoritmi di kernel (MMA/FA2, gemv/slots, GDN), banda/dispatch, probe di
  coerenza e **self-test** delle unità; `build_probes.sh` le compila con le variabili del build.
- **Deterministiche, load-free.** Non tracciata in git (come le sonde di GMBasic).

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

## GMBasic

### `tools/` — A/B e sweep (script)
- **`ab_all_models.sh`** / **`ab_models_under30g.sh`** — **A/B GMBasic vs rapidMLX** sui modelli locali
  (entrambi gli ordini, PREFILL+DECODE).
- **`ab_all_models_mlxlm.sh`** / **`ab_all_ctx_mlxlm.sh`** / **`ab_ctx_mlxlm.py`** — A/B **vs mlx_lm**
  (per modello / per contesto).
- **`ab_engine_rapid.sh`** / **`ab_engine_mlxlm.sh`** — A/B **del motore** GMBasic vs rapid / mlx_lm.
- **`ab_rapid_sweep.sh`** / **`ab_order_control.sh`** / **`ab_models_under30g_rapid.sh`** — sweep A/B
  appaiato, controllo dell'**ordine** (bias termico), modelli <30GB.
- **`bench3_compare.sh`** / **`bench4_compare.sh`** — tabella **GMBasic vs mlx_lm vs Rapid-MLX** (prefill+decode).
- **`bench4_stable.sh`** — confronto **appaiato** stabile (⚠ `bench4_paired.sh` è **SUPERATO**, non usare).
- **`bench_mlxlm.py` / `bench_rapidmlx.py` / `bench_table.py`** — lati reference (mlx_lm, rapidMLX) +
  **tabella markdown generata** (mai a mano) + JSON.
- **`mlx_decode_sweep.sh`** — regression **decode** (solo GMBasic) su tutti i modelli della matrice.
- **`gmb_defaults_sweep.sh`** — verifica i **DEFAULT** dell'engine (`gmb_host_ai_mlx`).
- **`decode_bench.c`** — bench **per-funzione del decode**, isolato, **load-free, batched** (NBUF op indip.).
- **`gdn_bench.c`** — OP del bench per-funzione: **GDN** (nostro kernel vs MLX).
- **`gemv_m1_bench.c`** — **GEMV 4-bit affine g64 M=1** (decode) vs `gather_qmm` di MLX.
- **`gmb_mlx_ar_bench.c`** — bench **AR (decode)** parametrizzato sul modello.
- **`gmb_mlx_dflash2_bench.c`** — **DFlash2 speculativo vs greedy AR**.
- **`gmb_mlx_ngram_bench.c`** — speedup del path **n-gram self-speculative** vs greedy.
- **`gmb_qwen4exp_func_bench.c`** — bench del path Qwen4-EXP.
- **`attn_decode_bench.sh`** — (`GMB_DECODE_ATTENTION_BENCH`, load-free): banda/costo **attenzione+KV** del
  decode al crescere del contesto.
- **`mlx_batch_test.sh`** (+ `.c`) — wrapper del probe **continuous batching** (Fase 3).
- **`minitest_batch.sh`** — harness **MiniTest** batch **senza ricompilazione**.
- **`gmb_stmt_gate.sh`** — **gate statement-coverage** dell'emitter C99Native (AST C# vs `GMB_STMT_*`).
- **`gmb_dsv4_engine_real.sh`** / **`gmb_dsv4_parity_test.sh`** — DS4: run reale + **parità**.
- **`dsv41_engine_test.sh`** / `dsv41_engine_mtp_test.sh` / `dsv41_engine_mtp_spec_test.sh` /
  `dsv41_mlx_*_test.sh` — DS4.1: integrazione engine + **MTP spec** + CSA.
- **`engine_smoke_models.sh`**, **`mlx_domain_split.py`**, **`minitest_census.py`** — smoke / censimenti.
- **`arch_registry.py`** — **OpenArch (A1–A3)**: `build` (spec+contratto da `config.json`), `validate`
  (config vs registry), `selftest`, `checks <registry.json>` (A3↔A1: golden del lab per contratto),
  `invariants`. Registry in `GMBasic/common/Registry/architectures/` (17).

### Progetti test (`test_*/`)
- **`test_mlx_bench`** — **harness MLX principale**: `<modelDir> prefill|decode|ctxdecode|rprefill|rdecode|
  rdecodectx|score|sampletext|chunkparity|specparity|specbench|rewindtest|budgettest|leaktest|corebench`.
  Legge via l'ABI (`llm_ai_mlx_*`), stampa tok/s + `IDS=` (oracolo).
- **`test_ilrun`** / **`test_vmrun`** / **`test_pcode`** / **`test_run_all`** — harness dei backend
  (IL / VM-C / PCode) su file `.bas`.
- **`test_ilrun-sweep`** — sweep su molti `.bas`.
- **`test_gfx_c`** / **`test_graphics_headless`** / **`test_graphics_visible`** — harness grafico (parità).
- **`test_llm_server`**, **`test_keyboard`**, **`test_editor`**, **`test_astdump`** — harness dedicati.

### Gate canonici (verbatim AGENTS)
- C99: `dotnet run --project tests/GMBASIC.Backend.C99.Tests -p:GmbEdition=Commercial -p:GmbBackendC99=true [-- <filtro>|--file=<f>]`
- MiniTest parity: `tools/test_minitests_parity "<dominio>/"`
- MiniTest C99Native: `GMB_MINI_ONLY="<dominio>/<file>.bas" tools/test_minitests_c99native`
- Parità 5 backend: `tools/parity5 <sample>.bas` (equivalenti legacy `mlx_parity_*`/`test_ilrun`/`test_vmrun`)

---

## Debug/profilo in-process (gated, per-unità)
- **`ds4-slow`**: opts registro **`dsv4_profile`** → profiler per-fase + **tempo GPU reale** (timestamp cb) +
  sotto-timer host MoE (`route`/`store`/`metal`) + **dispatch count**. *(no-op se 0 — zero overhead)*
- **Motore MLX/MoE**: opzioni `dsv4_prefill_mm`, `dsv4_gpu_hc`, `dsv4_prefill_chunk`, `dsv4_hotlist_topk` …
- **GMBasic**: `GMB_DECODE_ATTENTION_BENCH`, harness `mlx_batch_test`, `decode_bench`.

> [!NOTE] Vedi anche: `DS4-slow-Dove-va-il-tempo-Decode.md` (ripartizione CPU/GPU/decode),
> `BenchChronos-Benchmark-Coerente.md`, AGENTS §0sexies/§0septies.
