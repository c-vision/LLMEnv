# LLMEnv

**Soluzione C** con **progetti CMake autonomi** (uno per responsabilità, **grafo aciclico**), consumati
come **librerie statiche** da **host** come FastLLM. C# resta **adapter via P/Invoke**.

> **Ambito = `mlx-lm`** (una *libreria* di inferenza): tokenizer + codec/quant + modelli/runner +
> sampling. **NON** server HTTP, **NON** agenti, **NON** runtime BASIC — quelli restano nei consumatori.

## Layer (basso → alto)

```
LLMToken(1)   LLMCodec(2)   LLMDesc(3)   LLMMetal(4)
        └──────── LLMEngine(5, ABI) ────────┘
                     ├── LLMEngineMLX  (mlx/)
                     ├── LLMEngineLlama (llama/)
                     └── LLMEngineMoE  (moe/)
                 LLMModelIO(6)  ← 1,2,3
```

| # | progetto | responsabilità | dip. |
|---|---|---|---|
| 1 | **LLMToken** | lettore tokenizer.json/tiktoken/WordPiece; encode/decode + export | cJSON, PCRE2, sds |
| 2 | **LLMCodec** | formati GGUF/safetensors + codec quant (plugin). Non legge tokenizer. | solo formato |
| 3 | **LLMDesc** | role-map `.llmarch` + descrizione modello | — |
| 4 | **LLMMetal** | kernel Metal (GEMV/KV/MPP/OQ) | Metal |
| 5 | **LLMEngine** (ABI) | interfaccia common runner: load/prefill/decode/KV/sampling | 2,3,4 |
| 5a/b/c | **LLMEngineMLX / Llama / MoE** | i runner | 5 |
| 6 | **LLMModelIO** | conversione/quant (MLX↔GGUF) | 1,2,3 |

## Confini (vincolanti)

- Nomi del motore in `llm`/`LLM`; role-map `.llmarch`; env del motore `LLM_*`.
- `LLMEngine`/`LLMCodec` **non dipendono** dai consumatori. Dipendenza a senso unico:
  **FastLLM → LLMEnv**.

## Stato

Layer rinominati ed esternalizzati: `LLMCodec` (`llmcodec`), `LLMDesc` (`llmdesc`), `LLMToken`
(`llmtoken`), `LLMMetal` (`llmmetal`), `LLMEngine` (`llmengine`/`llmengine_mlx`/`llmengine_moe`/
`llmengine_llama`/`llmengine_plugin`), `LLMModelIO` (`llmmodelio`).

Ogni progetto compila **standalone** (`cmake -S LLMxxx -B build`). Il superbuild root aggancia i layer
sempre costruibili e, con `-DLLMENV_BUILD_ENGINE=ON`, anche i runner (richiedono MLX/llama.cpp).

Piano e prove: `AiWiki/01-Projects/LLMEnv/` (`Piani/`, `Knowledge/`, `Todo/(open) - Decisioni-Aperte-LLMEnv.md`).
