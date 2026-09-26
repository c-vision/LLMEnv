# LLMEnv

**Soluzione C** con **progetti CMake autonomi** (uno per responsabilità, **grafo aciclico**), consumati
come **librerie statiche** da **GMBasic** e **FastLLM**. C# resta **adapter via P/Invoke**.

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
common/  → include/build/helper CONDIVISI fra i layer
```

| # | progetto | responsabilità | dip. |
|---|---|---|---|
| 1 | **LLMToken** | lettore tokenizer.json/tiktoken/WordPiece; encode/decode + export | cJSON, PCRE2, sds |
| 2 | **LLMCodec** | formati GGUF/safetensors + codec quant (plugin). Non legge tokenizer. | solo formato |
| 3 | **LLMDesc** | role-map `.llmarch` (era `.gmbarch`) + descrizione modello | — |
| 4 | **LLMMetal** | kernel Metal (GEMV/KV/MPP/OQ) | Metal |
| 5 | **LLMEngine** (ABI) | interfaccia common runner: load/prefill/decode/KV/sampling | 2,3,4 |
| 5a/b/c | **LLMEngineMLX / Llama / MoE** | i runner | 5 |
| 6 | **LLMModelIO** | conversione/quant (MLX↔GGUF) | 1,2,3 |

## Confini (vincolanti)

- **Mai `gmb_`** nei nomi del motore (usare `llm`/`LLM`); `.gmbarch` → `.llmarch`; env del motore `LLM_*`.
- `LLMEngine`/`LLMCodec` **non citano** GMBasic né FastLLM. Dipendenza a senso unico:
  **GMBasic → LLMEnv**, **FastLLM → LLMEnv**.
- **GMBasic** possiede il runtime BASIC + agenti + compilatore; **FastLLM** possiede tutto il suo serving.

## Stato

**Scheletro.** La scomposizione è **già stata implementata dentro GMBasic** (con nomi `gmb_*`):
`gmb_model_desc`, `gmb_metal`, `gmb_llm_engine`, `gmb_model_io`, `gmb_plugin`. Il residuo è:
**rename `gmb_*`→`llm*`**, **esternalizzazione** qui, **split del runner MLX** (`gmb_host_ai_mlx`, 30.883
righe).

Piano e prove: `AiWiki/01-Projects/LLMEnv/` (`Piani/`, `Knowledge/`, `Todo/(open) - Decisioni-Aperte-LLMEnv.md`).
