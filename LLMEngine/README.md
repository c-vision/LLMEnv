# LLMEngine

The inference engine of LLMEnv: it loads a model, takes in text (as token ids), and generates new
tokens. All back ends sit behind **one common C interface**, so a program does not need to know which
engine is underneath.

## The common interface

`include/llmengine.h` defines it. A program loads a model, opens a **session** on it (a conversation
with its own memory), feeds in tokens, and asks for new ones. Each back end says which optional
features it supports (prefix cache, batching, embeddings, vision, log-probabilities, constrained
output, and others) through a capability flags field, so unsupported calls are simply absent.
`engine_select.c` picks a back end from the model path: a directory goes to MLX, a GGUF file goes to ds4.

## Back ends

| Folder | Back end | Runs |
|---|---|---|
| `mlx/` | MLX | Models in MLX format (safetensors), quantized or not. The main engine. |
| `ds4/` | ds4 | Large GGUF models, with a dedicated fast path. Based on antirez/ds4. |
| `llama/` | llama.cpp | GGUF models through llama.cpp, plus Apple Vision text recognition. |
| `moe/` | MoE runtime | Mixture-of-experts models that are too large for memory; experts are loaded from disk on demand. |
| `asr/` | Qwen3-ASR | Speech-to-text. |

## Supporting folders

| Folder | Purpose |
|---|---|
| `include/` | The public headers (`llmengine.h`, `llm_grammar.h`). |
| `Options/` | The option registry used to configure the engine. |
| `plugin/` | The plugin interface for new number formats and model building blocks. |
| `metal/` | Experimental Metal kernels and benchmarks. |
| `tools/` | Benchmarks, probes and test drivers. |

## Building

The engines are built together by the top-level project:

```sh
cmake -S . -B build -DLLMENV_BUILD_ENGINE=ON
cmake --build build
```

or, to produce all the shared libraries in one folder, `tools/build_ai_libs.sh` from the repository root.
The MLX and llama.cpp back ends need a Mac with Apple Silicon.
