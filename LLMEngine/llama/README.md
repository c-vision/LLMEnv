# llama

The llama.cpp back end. It runs GGUF models through [llama.cpp](https://github.com/ggml-org/llama.cpp),
which is built in statically. This back end works with text: llama.cpp keeps its own tokenizer and chat
template, and a model can be shared by many sessions, each with its own memory.

- `llmengine_llama.{h,c}`: model and session handling.
- `llmengine_llama_ocr.{h,c}`: reading text from images.
- `llm_engine_llama.c`: the adapter to the common interface in `../include/llmengine.h`.
- `apple_vision/`: a small Objective-C library, `libllm_vision_apple`, that uses Apple's Vision
  framework to recognize text in an image. It needs no model download.

The back end needs OpenSSL 3 at run time (`brew install openssl@3`).

## Load and session parameters (2026-10-30)

Two structs with a mandatory `size` field (a caller compiled against an older
version is rejected instead of being read past its own bytes):

- `llm_ai_model_params` + `llm_ai_load_model_ex`: `n_gpu_layers`, **`n_cpu_moe`**
  (`0` none / `N` first N layers / `-1` all / `-2` **AUTO** = all experts on CPU
  when the model file exceeds 60% of the RAM budget), **`load_mode`** (`-1` auto,
  `0` none, `1` mmap, `2` mlock, `3` mmap+mlock, `4` direct I/O), `ram_budget_mb`,
  `check_tensors`, `no_host`.
- `llm_ai_session_params` + `llm_ai_create_session_ex`: `n_ctx`, `n_batch`,
  `n_ubatch`, threads, KV types, `flash_attn`, `offload_kqv`.
- `llm_ai_model_stats` / `llm_ai_model_n_cpu_moe`: diagnostics after the load.

Experts moved to the CPU with `n_cpu_moe` stay **file-backed** (mmap): the OS can
reclaim them, which is what lets a model larger than RAM run at all. Measured on
OLMoE 4.2 GB: no large allocation (largest malloc zone 44 MB), weights served
from the mapping. Note that `footprint` and RSS report different numbers for the
same process, so state which one you measured. `n_batch` does **not** allocate
compute memory — `n_ubatch` does.

The same knobs are exposed through the options registry (never environment
variables) and are read by the token-id path FastLLM uses: `llama_ngl`,
`n_cpu_moe`, `llama_ram_budget_mb`, `llama_load_mode`, `llama_n_batch`,
`llama_n_ubatch`, `llama_threads`, `llama_threads_batch`, `llama_flash_attn`,
`llama_offload_kqv`, `llama_check_tensors`, `llama_no_host`.

Smoke tests (OFF by default): `-DLLMENGINE_LLAMA_BUILD_TESTS=ON` builds
`llama_params_smoke` and `llama_options_smoke`. Both must call `llm_ai_shutdown()`
before exiting; otherwise ggml-metal's teardown trips a residency-set assert
(known, pre-existing).
