# gguf_spm

A tiny vocabulary-only GGUF with a SentencePiece (`tokenizer.ggml.model = llama`) tokenizer, plus expected token ids.

- `spm.gguf`: 900 tokens trained with SentencePiece (BPE, byte fallback, two user-defined symbols), no tensors.
- `cases.tsv`: `text<TAB>ids`, with `\n`, `\t` and `\\` escaped. The ids come from llama.cpp (`llama-tokenize --ids --no-bos`), which is what runs these models.

Used by `LLMEngine/mlx/tests/llm_gguf_spm_tokenizer_test.c`, which builds the tokenizer from the GGUF with `llm_ai_mlx_gguf_tokenizer_json` and checks every case.
