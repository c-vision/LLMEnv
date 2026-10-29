# ABI

The model descriptions. Each `<architecture>.gmbarch` file (JSON) tells the engine how one model family is built:
which tensors have which role, how names map between Hugging Face / MLX and GGUF, which number formats and codecs
are used, and which building blocks (attention, mixture of experts, ...) make up a layer.

**This folder is the single source for these files.** LLMEnv owns it; other projects that need the same descriptions
read this folder instead of keeping a copy. Nothing is generated per model: the engine looks the architecture up here
(by the `model_type` of the model's `config.json`) and derives the numeric sizes (layers, heads, vocabulary, ...) from
`config.json` in memory.

## Where the folder is installed

The folder does not live at a fixed path: it depends on how LLMEnv was installed, and it can be somewhere else on
another machine.

| Install | Location |
|---|---|
| Installer (`.pkg`) | `/usr/local/lib/llmenv/ABI` |
| Archive (`.tar.gz`) | `lib/llmenv/ABI` inside the unpacked folder, next to the libraries |
| Homebrew | `<prefix>/opt/llmenv/lib/llmenv/ABI` |
| Source tree | `ABI/` at the root of the repository |

`ABI_VERSION` (one number) says which file format the folder uses. A folder counts as valid when it holds at least one
`.gmbarch` and its `ABI_VERSION`, if present, is not newer than the reader understands (`LLMDESC_ABI_SCHEMA`).

## How the engine finds it

`llmdesc_abi_dir()` (in `LLMDesc`) returns the first of:

1. the folder named by `LLMDESC_ABI_DIR`;
2. the application's own resolver, if it registered one with `llmdesc_set_abi_policy()` (a program can keep the path in
   its settings and look it up only when the saved one is missing or stale);
3. `llmdesc_abi_locate()`, which tries, in order: next to the engine library and next to the executable
   (`ABI`, `llmenv/ABI`, `../lib/llmenv/ABI`); the system installs (`/usr/local/lib/llmenv/ABI`, the Homebrew `opt` link);
   an `ABI`, `LLMEnv/ABI` or `../LLMEnv/ABI` folder found by climbing from the current directory, the executable and the
   library; and finally `brew --prefix llmenv`.

`llmdesc_set_abi_policy(resolver, 1)` makes a missing ABI folder an error (`llmdesc_resolve()` returns -1) instead of
silently falling back to a generic description. FastLLM and the other plain users of the library do not set it.

## Adding or changing a model family

Edit or add `<architecture>.gmbarch` here, then run the checks:

```sh
cmake -S LLMCodec -B build/codec && cmake --build build/codec && ctest --test-dir build/codec
cmake -S LLMDesc -B build/desc && cmake --build build/desc && ctest --test-dir build/desc
```

`llmcodec_abi_all_selfcheck` loads every file here and checks it is complete; `llmcodec_structs_selfcheck` and
`llmcodec_qwen4_rolemap_selfcheck` check the structure and name mapping of two architectures;
`llmdesc_abi_selfcheck` checks the validity rules, the lookup and the required-folder policy;
`llmdesc_geometry_selfcheck` checks, for the models found under `~/ai/Models`, that the geometry the engine derives
matches the model's `config.json`.
