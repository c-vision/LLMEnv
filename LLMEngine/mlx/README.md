# mlx

The main engine. It runs models in MLX format (safetensors) on Apple Silicon using Apple's
[MLX](https://github.com/ml-explore/mlx) framework, plus our own Metal kernels. It builds the shared
library `libllm_ai_mlx` (with `mlx.metallib`) and exposes the common interface of `../include/llmengine.h`.

## What it supports

- many model families, dense and mixture-of-experts, and the new hybrid designs (Qwen, Llama, Gemma,
  Mixtral, DeepSeek, GLM, gpt-oss, Kimi, LFM and others);
- quantized weights in several formats, through the codec plugins of `LLMCodec`;
- **packed weights** (`weights_backend=packed`): weights are read straight from the model files on disk
  instead of being copied into memory. It handles 2-bit and 4-bit MLX models of the Qwen3.8-Flash-Next
  family today; other formats and architectures are planned;
- a prefix cache that reuses the work done on a shared beginning of a prompt, also saved to disk;
- speculative decoding, constrained (JSON) output, KV-cache quantization, batching, embeddings, and
  expert swapping from disk for large mixture-of-experts models.

## Layout

| Folder | Contents |
|---|---|
| `*.c`, `*.h` (this folder) | The engine: model loading, layers, attention, mixture of experts, sessions, kernels. |
| `ds4/` | Native support for DeepSeek-V4 style models. |
| `ds41-native/` | Native DeepSeek-V4.1 engine and its reference tests. |
| `oss/` | gpt-oss support. |
| `tokenizers/` | Tokenizers used by the native DeepSeek paths. |
| `splash_kv_port/` | A paged KV cache: pages, storage tiers and paged attention (C++ and Metal). |
| `patches/` | Patches applied to the MLX sources at build time (see its README). |
| `tests/` | Tests and probes. |
| `goldens/` | Reference data and checks for speculative decoding. |

## Building

Use the top-level build with `-DLLMENV_BUILD_ENGINE=ON`, or `tools/build_ai_libs.sh` from the repository
root, which also produces `mlx.metallib`. MLX is fetched by the build (an offline cache can be used).

## Choosing options

Settings are passed by name through the registry in `../Options`, for example `weights_backend=packed`.
