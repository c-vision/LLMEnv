# ds41-native

A native engine for DeepSeek-V4.1 style models, written as header-mostly C. It covers the model
configuration and checkpoint reader, the attention variants, the FP8/FP4 formats, the engram tables
and the draft-model (DSpark) speculative decoding path.

- `llm_dsv41_*.h`: the engine, split by component.
- `llm_dsv41_*_ref.h`: reference implementations used to check each component.
- `llm_ds41_abi.c`: adapter that exposes the engine through the common interface
  (`llmengine_llm_engine_ds41`).
- `tools/`: shell scripts that run the component tests.
