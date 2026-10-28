# include

The public headers of LLMEngine.

- `llmengine.h`: the common interface shared by every back end. Handles are opaque. Calls return 0 or
  -1 (or a count where documented). The header carries an ABI version number so a program can check at
  run time which optional calls exist before using them.
- `llm_grammar.h`: constrained generation (header only, plain C). It forces the model to produce valid
  JSON, or JSON that matches a JSON Schema, by telling the engine at each step which tokens are allowed.
