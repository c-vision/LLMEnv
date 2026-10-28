# LLMDesc

Describes a model family in a small JSON file so that one engine can run many models without being
recompiled. The file has the extension `.llmarch` ("architecture description").

A `.llmarch` file says:

- **config map**: which keys of the model's `config.json` hold sizes such as the number of layers;
- **roles**: how tensor names map to roles (for example "this tensor is an attention query weight"),
  first match wins;
- **quantization**: which number format each role uses and how the calculation is done at run time;
- **structures**: the building blocks of the model (attention, mixture of experts, compression,
  indexer, vision, and so on), and how steps combine into a pipeline;
- **codecs**: which plugin decodes each number format.

Adding a new model should therefore be a new description file, or at most a new plugin, not a change
to the engine core.

## Files

- `llmdesc.h`, `llmdesc.c`: parser and data structures. A static library, `llmdesc`, with no
  dependencies.

## Building

```sh
cmake -S LLMDesc -B build/desc
cmake --build build/desc
```
