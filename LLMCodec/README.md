# LLMCodec

Reads model files and decodes quantized weights. Quantization stores weights with fewer bits (2, 4, 8
and so on) so a model takes less memory. There are many schemes, and each needs its own decoder.
LLMCodec provides them.

## What it does

- **File formats**: reads and writes GGUF, and reads safetensors.
- **Quantization formats** (one plugin each): MLX affine, GGML (K-quants and IQ types), AWQ, GPTQ,
  bitsandbytes, MXFP4 and MXFP8, NVFP, FP8/FP4, HQQ, SpQR, VPTQ, AQLM, BitNet, TurboQuant, Higgs.
- **Plugins**: each format lives in its own shared library (`libllmcodec_<name>.dylib`) that is loaded
  at run time, so a new format does not require rebuilding the rest.

The library has no external dependencies beyond the C math library.

## Main headers (`include/`)

| Header | Purpose |
|---|---|
| `llmcodec.h` | Plugin registry and the `LlmCodec` type |
| `llmcodec_gguf_read.h`, `llmcodec_gguf_writer.h`, `llmcodec_gguf_meta.h` | GGUF reading and writing |
| `llmcodec_gguf_status.h` | One error code per failure cause |
| `llmcodec_safetensors.h` | safetensors reader |
| `llmcodec_quant_*.h`, `llmcodec_ggml_quants.h` | Dequantization routines |

## Building and testing

```sh
cmake -S LLMCodec -B build/codec
cmake --build build/codec -j
ctest --test-dir build/codec --output-on-failure
```

The tests compare decoded output with reference data from real models, stored in `tools/Fixtures/codec/`.
A manual differential check against ggml and gguflib is in `tools/quant_diff*.c`.
