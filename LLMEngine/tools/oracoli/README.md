# oracoli

"Oracle" programs: small tools that give a trustworthy answer to compare the engine against, and
helpers to measure memory and speed. The sources and scripts are tracked; compiled binaries, logs and
caches are ignored.

## Helper scripts

| Script | What it does |
|---|---|
| `wd.sh` | Watchdog: kills a command if its memory use goes over a limit in GB. |
| `rss_ab.sh` | Interleaved A/B runs that record peak memory (`/usr/bin/time -l`). |
| `diag_run.sh` | Runs a model with deterministic diagnostic settings. |
| `repeat_ids.sh` | Repeats one request N times and compares the token ids, to catch state leaks. |
| `spec_stats.sh` | Extracts pass time and acceptance from a speculative-decoding log. |
| `build_probes.sh` | Builds the probes below from the local sources. |

## Probes (no model needed)

- `llm_gemv_bench_test.c`: synthetic quantized matrix-vector product, effective GB/s.
- `llm_bw_test.c`: raw read, write and triad memory bandwidth.
- `llm_mlx_compile_cost_probe.c`: break-even of compiling per layer.
- `llm_mlx_attn_roof_probe.c` (+ `_alias.cpp`): roofline of the attention step.
- `llm_gather_qmm_sorted_probe.c`: checks `gather_qmm` on sorted input against a per-expert reference.

Caveat: a benchmark that repeats one shape can fit in the GPU cache and overstate bandwidth by several
times. Use it for A/B ratios, not as an absolute number, and size the working set like the real model.

## Using them

```sh
bash LLMEngine/tools/oracoli/build_probes.sh
# dtype arguments are numbers: 0 = f32, 1 = f16, 2 = bf16
LLMEngine/tools/oracoli/llm_gemv_bench_test 1 8192 8192 4 64 2 2 50 0
LLMEngine/tools/oracoli/llm_bw_test 512 0 10
LLMEngine/tools/oracoli/wd.sh <limit_GB> <prompt|-> <command...>
```
