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
