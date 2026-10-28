# oq_a8: int8 activations with 4/5-bit weights

Matrix-product kernels for the **prefill** step (processing a long prompt), which quantize the
activations to 8 bits and multiply them by 4-bit or 5-bit weights on the Apple GPU tensor units (M5).
This is a port of kernels from [oMLX](https://github.com/jundot/omlx) (Apache-2.0). The three kernel
files are byte-identical to the originals; see `THIRD_PARTY_NOTICES.md` and
`LICENSE-oMLX-Apache-2.0.txt`.

## Files

- `csrc/oq_a8_qmm_nax.metal`: the matrix product.
- `csrc/oq_a8_quantize.metal`: activation quantization (stage A).
- `csrc/oq_a8_decode.h`: reference weight decoding.
- `llmmetal_oq_a8_bridge.{h,mm}`: the Objective-C bridge the engine calls.
- `build_metallibs.sh`: compiles the Metal libraries.

## Building the Metal libraries

The tensor-unit kernels need Metal 4 and macOS 26.2 or later as the deployment target. Without
`-mmacosx-version-min=26.2` the kernel can load and run but gives wrong results.

```sh
bash LLMMetal/oq_a8/build_metallibs.sh
```

## When it is used

The engine uses this path automatically for layers where it applies: affine 4/5-bit weights with group
size 64, at least 128 prompt rows, and sizes divisible by 64. Otherwise, or if the Metal libraries are
missing, it quietly falls back to the normal path.

The result depends on the model: on Qwen3-8B 4-bit it made prefill 16-28% faster, but on the much
smaller Llama 1B 4-bit it was slower, because the fixed cost per call is not repaid on small matrices.

## Accuracy

Quantizing activations is an approximation. On a teacher-forcing check over 256 positions of real text,
3.5% of the top choices differed, higher than the 1.8-2.1% seen with the other accepted approximations.
The activation quantization and the 4-bit weight packing match the reference exactly; the matrix
product matches an FP64 reference to within FP16 output noise.

## Tests

`tests/` holds manual checks (not part of continuous integration), including
`oq_a8_omlx_selfcheck.m` and `oq_a8_prep_cache_selfcheck.c`.
