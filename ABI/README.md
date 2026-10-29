# ABI

The model descriptions. Each `<architecture>.gmbarch` file (JSON) tells the engine how one model family is built:
which tensors have which role, how names map between Hugging Face / MLX and GGUF, which number formats and codecs
are used, and which building blocks (attention, mixture of experts, ...) make up a layer.

**This folder is the single source for these files.** LLMEnv owns it; other projects that need the same descriptions
read this folder instead of keeping a copy. Nothing is generated per model: the engine looks the architecture up here
(by the `model_type` of the model's `config.json`) and derives the numeric sizes (layers, heads, vocabulary, ...) from
`config.json` in memory.

## How the engine finds it

`llmdesc_abi_dir()` (in `LLMDesc`) returns the first of:

1. the folder named by `LLMDESC_ABI_DIR`, for installations outside the source tree;
2. an `ABI`, `LLMEnv/ABI` or `../LLMEnv/ABI` folder found by climbing from the current directory;
3. the same search climbing from the folder of the engine library (this is how the installed package finds
   `lib/llmenv/ABI`).

## Adding or changing a model family

Edit or add `<architecture>.gmbarch` here, then run the checks:

```sh
cmake -S LLMCodec -B build/codec && cmake --build build/codec && ctest --test-dir build/codec
cmake -S LLMDesc -B build/desc && cmake --build build/desc && ctest --test-dir build/desc
```

`llmcodec_abi_all_selfcheck` loads every file here and checks it is complete; `llmcodec_structs_selfcheck` and
`llmcodec_qwen4_rolemap_selfcheck` check the structure and name mapping of two architectures;
`llmdesc_geometry_selfcheck` checks, for the models found under `~/ai/Models`, that the geometry the engine derives
matches the model's `config.json`.
