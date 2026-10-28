# dflash2 goldens

Reference data and test programs for **DFlash2**, a speculative-decoding draft model (block diffusion)
used with the MLX engine. Speculative decoding lets a small draft model propose several tokens that the
main model then checks in one step; the output must stay identical to what the main model alone would say.

## The reference run

The reference data came from the original DFlash implementation, with the target model
`Qwen3.8-27B-8bit` and the draft model `Qwen3.8-27B-DFlash2`, on the prompt "The capital of France is"
with greedy decoding (8 rounds, block size 8). For every round it records the last verified token
(`anchor`), the target's hidden features (`ctx`), the block the draft proposed, and the final
`transcript` of emitted tokens.

## What is in this folder

| File | Purpose |
|---|---|
| `dflash_goldens.json` | Round metadata and the final transcript. |
| `round0.bin` | Binary data for round 0. |
| `dflash2_validate.c` | Checks that our draft step produces the recorded `draft` tokens. |
| `dflash2_transcript.c` | Checks that the full speculative decode emits the recorded `transcript`. |
| `dflash2_bench.c` | Compares plain greedy decoding with DFlash2. |

All three programs link against `libllm_ai_mlx`.

## What to expect

Speculative decoding is lossless: the output must equal the target model's own greedy output. On the
reference setup (120 tokens) plain greedy ran at about 10 tok/s and DFlash2 at about 14.8 tok/s (1.48x),
with identical output.
