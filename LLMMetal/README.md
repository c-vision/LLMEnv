# LLMMetal

Metal kernels for Apple GPUs: the small programs that do the heavy arithmetic of a language model on
the GPU. They use only system frameworks (Metal, Foundation) and no third-party libraries.

| Folder | What it does |
|---|---|
| `gemv/` | Matrix-vector product on quantized weights (4-bit affine and MXFP4). This is the core operation when generating one token at a time. |
| `kv/` | Paged key-value cache and the attention step for generation. |
| `mpp_q4/` | 4-bit matrix product that uses the Apple GPU tensor units. |
| `oq_a8/` | 8-bit-activation by 4/5-bit-weight matrix product for long prompts (a port of kernels from oMLX, Apache-2.0). See its own README. |

The kernels live as Metal source inside headers (and `.metal` files) and are compiled and used by
LLMEngine. This project builds the correctness checks for them.

## Requirements

A Mac with Apple Silicon and the Xcode command line tools. Some kernels (`oq_a8`, `mpp_q4`) need a
recent macOS and Metal 4.

## Building

```sh
cmake -S LLMMetal -B build/metal
cmake --build build/metal
```
