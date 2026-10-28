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
