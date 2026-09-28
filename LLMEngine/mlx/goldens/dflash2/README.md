# Golden DFlash2 — oracolo reference (port nativo motore MLX)

Golden prodotti dall'oracolo REFERENCE reale (non sintetico) per il port del
drafter DFlash2 (`Qwen3.8-27B-DFlash2`, BF16, block-diffusion) nel motore MLX.

## Come sono stati prodotti

`dflash_oracle.py` (questo file) esegue il round-loop DFlash REALE di mlx-vlm
0.6.3 (`generate_step` + drafter vendored in `Rapid-MLX-NextGen`, 
`vllm_mlx/speculative/dflash/drafters/qwen3_dflash.py`, che implementa two-tap
conv + CandidateSelector) con:

- target: `~/ai/Models/Qwen3.8-27B-8bit` (multimodale MLX; embed/lm_head sotto
  `language_model.*`), caricato via `mlx_vlm.load`
- draft: `~/ai/Models/Qwen3.8-27B-DFlash2` (bind di embed+lm_head dal target)
- prompt fisso `"The capital of France is"`, greedy (temperature 0),
  max_tokens 20, block_size 8

Esecuzione:
```bash
source ~/ai/env312/bin/activate
cd native/llmengine_mlx/goldens/dflash2 && python3 dflash_oracle.py
```

## Contenuto (`dflash_goldens.npz` + `.json`)

Per OGNI `draft_block` invocato dal reference (8 round):
- `round{i}_anchor`  — ultimo token verificato (int)
- `round{i}_ctx`     — ctx feature in ingresso `[S, 25600]` fp32 = concat degli
                       hidden del target ai layer [5,19,33,47,61] (S posizioni)
- `round{i}_draft`   — blocco proposto dal draft (token int, len = block_size-1)
- `round{i}_block_size` (json)
- `transcript`       — token emessi (greedy del target, incl. bonus): il golden
                       finale di parità per l'intero decode speculativo

I ctx hidden catturano anche lo stato di cache del draft tra round: il replay
C deve mantenere la stessa cache rotante per layer e riprodurre round-per-round
gli stessi `draft` dati (`anchor`, `ctx`).

## Uso

1. **Draft forward C**: dato (anchor, ctx) e la cache al round i, il loader
   nativo DFlash2 (conv + selector lattice, embed/lm_head bind-ati al target)
   deve produrre `draft_tokens == round{i}_draft`.
2. **Parità end-to-end**: il decode speculativo dflash in LLMEnv deve emettere
   `transcript` identico (output == greedy del target, lossless per costruzione).

Deviazioni da DEBUGGARE con:
- `round{i}_ctx` + `round{i}_draft`: se il blocco differisce, confrontare con
  l'hidden post-backbone e gli input selector (rigenerando golden intermedi con
  `dflash_oracle.py` esteso) per isolare conv / attenzione-kv-injection / head.

## Benchmark (120 token greedy, target Qwen3.8-27B-8bit, 2026-09-02)

- plain greedy : 120 tok in 11.99s -> 10.0 tok/s
- dflash2      : 120 tok in 8.08s  -> 14.8 tok/s  (~1.48x, PARITY==OK)

Harness: `dflash2_bench.c` (compila linkando `libllm_ai_mlx`, `DF2_STATS=1` per l'accettanza per-round).
