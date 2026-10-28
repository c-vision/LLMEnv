# Fixtures

Reference data used by the tests. These files are expected output and should not be edited by hand.

## `codec/`

Real tensors in several quantization formats, with the values they must decode to. The LLMCodec test
`llmcodec_real_fixtures_selfcheck` decodes each one and compares.

| Folder | Format |
|---|---|
| `awq_real/`, `gptq_real/` | AWQ and GPTQ |
| `bnb_real/` | bitsandbytes (including nested quantization) |
| `mxfp4/`, `mxfp8_real/` | MXFP4 and MXFP8 |
| `turboquant/` | TurboQuant |
| `dsv41/` | DeepSeek-V4.1 tensors |

## `tokenizer/`

A reference ("golden") file produced by the Hugging Face tokenizer and two small vocabularies. The
LLMToken check `llmtoken_wordpiece_selfcheck` compares our WordPiece tokenizer with it.
