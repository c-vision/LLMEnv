# LLMEnv

LLMEnv is a set of C libraries for running large language models (LLMs) on your own machine.
It reads model files, turns text into tokens and back, runs the model on the GPU, and generates
text. It is built mainly for Apple Silicon Macs.

It is a library, not an application: there is no chat window and no web server. You link it into
your own program and call it. If you just want to use the engine, start with [FastLLM](#fastllm).

## What it can do

- **Run models from several formats.** MLX models (safetensors), GGUF models, and large
  mixture-of-experts models whose parts can be loaded from disk only when needed.
- **Use quantized models.** Quantization shrinks a model so it fits in memory. LLMEnv reads the
  common schemes: 2, 4 and 8 bit, AWQ, GPTQ, bitsandbytes, MXFP4 and MXFP8.
- **Keep memory use low.** In its "packed" mode the engine reads weights straight from the model
  files on disk instead of copying them into memory, so large models can run
  with far less memory than their file size.
- **Reuse earlier work.** Prompts that share a beginning can reuse the computation already done
  (a "prefix cache"), which makes long conversations start faster.

## What is inside

| Folder | What it does |
|---|---|
| `LLMToken` | Splits text into tokens and joins them back. Reads `tokenizer.json`, tiktoken and WordPiece files. |
| `LLMCodec` | Reads model files (GGUF, safetensors) and decodes the different quantization formats. |
| `LLMDesc` | Describes how a model is laid out, so one engine can handle many model families. |
| `LLMMetal` | Small programs that run on the Apple GPU: matrix products, attention cache, quantized math. |
| `LLMJinja` | Fills in chat templates, the text patterns that turn a conversation into a prompt. |
| `LLMEngine` | The engine itself, with one common interface and several back ends (see below). |

The engine back ends:

- **MLX** runs MLX-format models on Apple's MLX framework.
- **llama.cpp** runs GGUF models through llama.cpp.
- **ds4** is a dedicated fast path for large GGUF mixture-of-experts models.
- **MoE** runs mixture-of-experts models, loading experts from disk on demand.

Your program talks to the common interface (`LLMEngine/include/llmengine.h`) and does not need to
know which back end is underneath.

## FastLLM

[FastLLM](https://github.com/c-vision/FastLLM) is the tool for using the LLMEngine engine from the
command line. It is built on top of LLMEnv and gives you, without writing any code:

- a **command-line interface**, in the style of Ollama, to download, list and run models;
- a **local server** that speaks the OpenAI and Anthropic APIs, so existing apps can talk to it;
- **downloads and caching** of models from Hugging Face.

In short: LLMEnv is the engine, FastLLM is how you drive it.

## Building

You need a C/C++ compiler and CMake. The GPU parts need a Mac with Apple Silicon.

Each folder builds on its own:

```
cmake -S LLMToken -B build/token
cmake --build build/token
```

To build everything together, including the engine back ends (which also need MLX and llama.cpp):

```
cmake -S . -B build -DLLMENV_BUILD_ENGINE=ON
cmake --build build
```

## Settings

Engine behaviour is controlled through named options that you set from your program, not through
environment variables. For example, `weights_backend=packed` turns on the low-memory mode
described above.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Contributions need a signed [CLA](CLA.md).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
