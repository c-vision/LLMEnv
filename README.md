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
| `LLMDesc` | Describes how a model is laid out, so one engine can handle many model families. The descriptions are the files in `ABI/`. |
| `ABI` | The model descriptions (`<architecture>.gmbarch`), the single source read by the engine and by other projects. Installed next to the libraries (`lib/llmenv/ABI`), so programs can find it wherever LLMEnv is installed. |
| `LLMMetal` | Small programs that run on the Apple GPU: matrix products, attention cache, quantized math. |
| `LLMJinja` | Fills in chat templates, the text patterns that turn a conversation into a prompt. |
| `LLMEngine` | The engine itself, with one common interface and several back ends (see below). |

The engine back ends:

- **MLX** runs MLX-format models on Apple's MLX framework.
- **llama.cpp** runs GGUF models through llama.cpp.
- **ds4** is a dedicated fast path for large GGUF mixture-of-experts models. It is based on
  [ds4](https://github.com/antirez/ds4) by Salvatore Sanfilippo (antirez), see
  [Acknowledgements](#acknowledgements).
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

## Installing (macOS, Apple Silicon)

LLMEnv comes in two builds of the same engine. The **compatible** build runs on macOS 14 or later. The **NAX** build needs macOS 26.2 or later and
uses the tensor units of the M5 for the prefill. The installer script and Homebrew pick the right one for your Mac; if you download a file
yourself, `macos14` is the compatible build and `macos26` the NAX one.

- **Installer script** (no sudo, installs into `~/.local`, safe to run again):

  ```sh
  curl -fsSL https://github.com/c-vision/LLMEnv/releases/latest/download/install-llmenv.sh | bash
  ```

  If a suitable LLMEnv is already on the machine (Homebrew, the `.pkg`, an earlier run), it does nothing and tells you where it is.
  Options: `--prefix DIR`, `--version X.Y.Z`, `--variant compat|nax`, `--force`, `--check`, `--uninstall`.
- **Homebrew:** `brew tap c-vision/fastllm`, then `brew trust c-vision/fastllm` (Homebrew 7 does not load a third-party tap until you trust it), then `brew install llmenv`. The formula lives in the [FastLLM tap](https://github.com/c-vision/homebrew-fastllm), together with `fastllm`.
- **Installer package:** open `llmenv-<version>-macos14-arm64.pkg` (or `macos26`). It puts the libraries in `/usr/local/lib/llmenv`
  and the headers in `/usr/local/include/llmenv`. The package is not signed with an Apple Developer ID,
  so macOS will ask you to confirm (right-click, Open, or System Settings > Privacy & Security > Open Anyway).
- **Archive:** unpack `llmenv-<version>-macos14-arm64.tar.gz` (or `macos26`) anywhere. The libraries find each other,
  so the folder can be moved freely.

All of them put the model descriptions in `lib/llmenv/ABI`, next to the libraries. A program that needs them can look the folder up with `llmdesc_abi_locate()` (see [ABI](ABI/README.md)).
The file `lib/llmenv/LLMENV_VERSION` says which version and variant is installed.

There are no external dependencies: from 0.3.0 the llama.cpp back end no longer links OpenSSL.

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

## Acknowledgements

The ds4 engine in `LLMEngine/ds4` is a copy of [ds4](https://github.com/antirez/ds4), created by
Salvatore Sanfilippo (antirez) and released under the MIT license, with small changes of our own
(kept as separate patches and a bridge layer). Many thanks to him and to the ds4 contributors. It
also builds on the work of the [ggml](https://github.com/ggml-org/ggml) authors, Apple's
[MLX](https://github.com/ml-explore/mlx) and DeepSeek. The speech-to-text code in `LLMEngine/asr` is a port of [qwen-asr](https://github.com/antirez/qwen-asr) by the same author (MIT), and the `oq_a8` kernels in `LLMMetal` come from [oMLX](https://github.com/jundot/omlx) (Apache-2.0). Full notices are in [NOTICE](NOTICE).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
