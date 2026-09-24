# LLMEnv

**Soluzione C** con due progetti **separati**, consumati come **librerie statiche** da GMBasic e FastLLM:

| progetto | ruolo | sorgente attuale |
|---|---|---|
| **LLMEngine** | **motore di inferenza** — MLX (mlx-c + kernel Metal), GGUF e futuri formati/backend | da estrarre da `GMBasic/native/gmb_host_ai_mlx/` (94 `.c` + 128 `.h`, monolite `gmb_host_ai_mlx.c` da 29.003 righe) |
| **LLMCodec** | **codec di formati e quantizzazione** — GGUF (parse/write), safetensors, 34 tipi quant, `rolemap`/`.gmbarch`, converter GGUF↔MLX, `moe_quant`, `fp8/fp4`, plugin | da estrarre da `GMBasic/native/gmb_ggml_codec/` (già autonomo: solo `libm`) |

> **Perché non "MLX" nel nome.** Il motore non è solo MLX: legge GGUF e gestirà altri formati → un
> nome MLX-centrico sarebbe riduttivo e falso.

## Struttura

```
LLMEnv/
├── LLMEngine/     # solo ciò che è specifico del motore
├── LLMCodec/      # solo ciò che è specifico del codec
└── common/        # include/file comuni ai due (una o più cartelle)
```

Nella **root** vanno solo i comuni (include condivisi, build); il resto nelle due sottocartelle.

## Confini (vincolanti)

- `LLMEngine` e `LLMCodec` **non citano mai GMBasic** (i wrapper verso il compilatore/GPM restano in
  GMBasic) né **FastLLM**.
- Dipendenza a senso unico: **GMBasic → LLMEnv**, **FastLLM → LLMEnv**. Mai il contrario.
- ABI pubbliche: `llmengine.h`, `llmcodec.h` (handle opachi, errori a codici enum, `*_free_str`).

## Stato

**Scheletro.** Il codice non è ancora estratto: un'altra sessione sta completando DeepSeek 4.1 Flash
nel motore+codec di GMBasic. L'estrazione parte **solo dopo** il congelamento di quella versione.

Piano e prove: `AiWiki/01-Projects/LLMEnv/` (`Piani/(open) - Piano-LLMEnv-LLMEngine-LLMCodec.md`,
`Knowledge/Inventario-Recupero-da-MLXEngine.md`).
