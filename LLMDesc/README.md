# LLMDesc

Describes a model family in a small JSON file so that one engine can run many models without being
recompiled. The file has the extension `.gmbarch` ("architecture description"; the older `.llmarch`
name is still read).

A description says:

- **roles**: how tensor names map to roles (for example "this tensor is an attention query weight"),
  first match wins, and how those names map between Hugging Face / MLX and GGUF;
- **quantization**: which number format each role uses and how the calculation is done at run time;
- **structures**: the building blocks of the model (attention, mixture of experts, compression,
  indexer, vision, and so on), and how steps combine into a pipeline;
- **codecs**: which plugin decodes each number format.

Adding a new model should therefore be a new description file, or at most a new plugin, not a change
to the engine core.

## Where the descriptions live

In the [`ABI`](../ABI) folder at the root of this repository, one file per architecture. It is the
single source: nothing is written per model. `llmdesc_resolve()` reads the architecture from the model's
`config.json`, loads `ABI/<architecture>.gmbarch`, and fills in the numeric sizes (layers, heads,
vocabulary, context, ...) **in memory** from `config.json` (for nested configs, from `text_config`).
`llmdesc_abi_dir()` finds the folder; see the [ABI README](../ABI/README.md) for the search order.

## Files

- `llmdesc.h`, `llmdesc.c`: parser, data structures and the lookup. A static library, `llmdesc`, with no
  dependencies.
- `llmdesc_geometry_selfcheck.c`: for every model found under `~/ai/Models`, checks that the resolved
  description has the sizes of its `config.json`.

## Building and testing

```sh
cmake -S LLMDesc -B build/desc
cmake --build build/desc
ctest --test-dir build/desc
```
