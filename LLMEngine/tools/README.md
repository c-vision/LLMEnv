# tools

Benchmarks, probes and test drivers for the engine. They are for development, not for users of the
library. Most are small C programs or shell scripts that link against `libllm_ai_mlx`.

| Group | Files | Purpose |
|---|---|---|
| Parity | `mlx_parity_driver.c`, `llm_tok_ids.c`, `mlx_embed_check.c`, `mlx_topk_check.c`, `llm_spec_check.c` | Load a model through the common interface, generate tokens and compare with a reference. |
| Speed | `llm_decode_time.c`, `decode_profile.c`, `llm_perop_profile.c`, `llm_qmm_bench_4096.c`, `kv_restore_bench.c` | Time decoding, break it down per operation, time matrix products and KV restore. |
| A/B comparison | `ab_*.sh`, `bench3_compare.sh`, `bench4_*.sh`, `scale_sweep.sh`, `attn_decode_bench.sh` | Compare the engine with other engines or with itself under different settings. |
| Speculative decoding | `dflash_multi.c`, `dflash_parity.c`, `dflash_stream.c` | DFlash drafting tests. |
| Mixture of experts | `moe_stream_bench.c`, `moe_ep_run.c`, `moe_ep_socket.c`, `tq_moe_batch_repro.c` | Expert streaming and expert-parallel experiments. |
| Probes | `llm_kernelcache_probe.c`, `llm_qgemv_fast_probe.c`, `arch_lab.c` | One-question experiments about kernels and architecture. |
| Other | `llm_autotune.sh`, `llm_plan_gate.sh` | Automatic tuning and plan checks. |
| `oracoli/` | see its README | Reference ("oracle") programs and measurement helpers. |

Benchmarks load real models, which use a lot of memory. Run one at a time and put a memory limit on
the process (see `oracoli/wd.sh`).

## Probe Python (locali, non versionati)

> `.gitignore` di questo repo ignora `*.py`: i probe Python sono strumenti **locali di sviluppo**,
> non versionati (come tutti gli altri `.py` del repo). Il codice canonico del motore resta in C.

Gruppo **migrato dal repo `GMBasic`** (deprecato) il **2026-10-30**, insieme ai suoi 80 tool Python
già presenti qui. Sede: questa cartella (o `mlx_bench/`, `../../LLMCodec/tools/` per i GGUF).

| file | cosa fa |
|---|---|
| `gmb_requant_mlx.py` | ri-quantizza un modello MLX (safetensors) da un formato di pesi a un altro |
| `gmb_mlx_reshape_census.py` | census dei call-site `mlx_reshape` e classificazione dell'array sorgente |
| `moe_occupancy_h1_probe.py` | verifica se il decode MoE single-stream è occupancy-bound |
| `mx_region_fold_probe.py` | la fusione che paga è a REGIONE o solo ad attivazione |
| `mxcompile_fold_dispatch_probe.py` | risolve la contraddizione del puntone #85 (dispatch/fusione) |
| `mxfuse_parity_probe.py` | bit-fedeltà della regione fusa (mx.compile) di SwiGLU |
| `mxswiglu_parity_probe.py` | arbitra il claim di bit-fedeltà del kernel Metal SwiGLU |
| `mlx_domain_split.py` | estrae un run contiguo VERBATIM da `llm_engine_mlx.c` in un file di dominio |
| `phase_growth_probe.py` | decomposizione op-level LOAD-INDEPENDENT del decode denso |
| `emb_test.py` | confronto embedding dei chunk (db) con la query embeddata dal modello |
| `mlx_bench/compare_hidden.py` | confronto degli stati nascosti fra run |
| `mlx_bench/profile_parse.py` | parsing dei profili di decode |
| `mlx_bench/run_baseline.py` | baseline tok/s del motore (`REPO` = questo repo; la parte "ours" richiede il harness C# `test_mlx_bench` del monorepo, non presente qui) |
| `../../LLMCodec/tools/gguf_tensors.py` | elenca i tensori (nome, dims, tipo) di un GGUF leggendo solo l'header |
| `../../LLMCodec/tools/gmb_gguf_header_probe.py` | ispettore di HEADER GGUF (nessun caricamento di pesi) |

Adattamenti applicati allo spostamento: `REPO` → questo repo, `dylib` → `LLMEngine/mlx/build/…`,
riferimenti a `gmb_host_ai_mlx.c` → `llm_engine_mlx.c` (nome attuale del motore MLX).
