# LLMEngine

Progetto C di LLMEnv: ABI comune (`include/llmengine.h`) + runner (`mlx/`, `llama/`, `moe/`)
+ registry plugin (`plugin/`).

- Dipendenza a senso unico: i consumatori usano LLMEnv, mai il contrario.
- Sub-target: LLMEngineMLX (`mlx/`), LLMEngineLlama (`llama/`), LLMEngineMoE (`moe/`).
- La `.llmarch` descrive i modelli; il motore carica i plugin a runtime (`dlopen`).
