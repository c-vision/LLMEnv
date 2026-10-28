# metal

Experimental Metal kernels and their benchmarks, written to replace MLX for the hottest operations.
The kernels that the engine actually uses today live in `LLMMetal/`; this folder is where new ones are
tried and measured.

## 4-bit matrix-vector product (token-by-token generation)

`gemb_kernel.metal` has four kernels for a 4-bit affine product (group size 64) against a single vector:

| Kernel | Notes |
|---|---|
| `qgemv4s` | The best one: one simdgroup per row, 8 rows per threadgroup. |
| `qgemv4` | One threadgroup per row. |
| `qgemv4v` | Vectorized loads. Fast, but gives **wrong results** (known bug). Do not use. |
| `bcopy` | A plain copy, to measure the machine's real memory bandwidth. |

`qgemv4s` needs a grid of `(N+7)/8` threadgroups; the others need `N`. A wrong grid computes fewer rows
and inflates the measured speed.

**How to measure.** Time many dispatches in one command buffer and wait once. Real generation queues
work without waiting per operation, and a per-call wait adds 0.1-0.3 ms that hides the real cost.

On a 4096 x 4096 4-bit matrix, `qgemv4s` takes about 0.018 ms (roughly 460 GB/s, about 88% of the
measured memory bandwidth), against about 0.24 ms for the MLX path used before.

```sh
bash LLMEngine/metal/build_bench.sh                 # builds build/gemb_bench
build/gemb_bench [kernel] [N] [K] [reps] [BATCH]    # default: qgemv4 4096 4096 30 50
build/gemb_bench bcopy 4096 4096 30 50              # the machine's real bandwidth
```

## Other experiments

- `gemv_gguf*`: matrix-vector kernels for GGUF block formats.
- `mpp_*`: matrix products on the Apple GPU tensor units.
- `steel_gemm.metal`, `qmv_multi.metal`: reference and multi-output variants.

These are work in progress; the 4-bit matrix product for long prompts is not final.
