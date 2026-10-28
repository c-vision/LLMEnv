# LLMToken

Turns text into token ids and back. A language model never sees text: it sees a list of numbers, and
the tokenizer is the dictionary that converts between the two. LLMToken reads the tokenizer files that
ship with models and does this conversion in plain C, with no Python and no external tokenizer library.

## Supported tokenizer families

| Family | Typical models | Input file |
|---|---|---|
| Byte-level BPE (GPT-2 / GPT-4 style) | Llama 3, Qwen, GLM | `tokenizer.json` |
| SentencePiece / metaspace | Llama 1 and 2, Mistral, Mixtral | `tokenizer.json` |
| tiktoken | OpenAI-style vocabularies | `tiktoken.model` |
| WordPiece | BERT-style encoders | `vocab.txt` |

The right family is chosen from what the vocabulary contains, not from the model name. Output was
checked token by token against the Hugging Face `tokenizers` library on real checkpoints, including
accented text, emoji and newlines.

## Main files

- `llmtoken.h`, `llmtoken.c`: the public interface and the BPE encoder.
- `llmtoken_sp.c`: SentencePiece.
- `llmtoken_wordpiece.h`, `llmtoken_wordpiece.c`: WordPiece (with generated Unicode tables).

## Building and testing

```sh
cmake -S LLMToken -B build/token
cmake --build build/token
ctest --test-dir build/token --output-on-failure
```

The library is `llmtoken` (a shared library). It needs cJSON, PCRE2 and sds, which the build fetches
for you. The reference data used by the WordPiece check lives in `tools/Fixtures/tokenizer/`.
