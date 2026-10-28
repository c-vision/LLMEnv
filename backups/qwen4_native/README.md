# backups/qwen4_native — implementazione GGUF nativa di qwen4exp (rimossa da LLMEngine il 2026-10-29)

Backup dei sorgenti di `qwen4_native`, una re-implementazione sperimentale di `qwen4exp` (Qwen3.8-Flash-Next) su GGUF con kernel Metal propri
e oracoli di parità. **Rimossa dal motore** perché non serve in produzione:

- Il percorso GGUF di LLMEngine è il **ds4 integrato** (`LLMEngine/ds4`, vtable `llmengine_llm_engine_ds4n`, selezionato da `engine_select.c`
  per i file GGUF). Misurato sullo stesso `Qwen3.8-Flash-Next-Q2.gguf` (137 GiB, 255 token di prompt): **prefill 728 tok/s, decode 52.1 tok/s,
  memoria pianificata 50.5 GiB** (pesi residenti 41.7 GiB, n-gram da disco).
- `qwen4_native` mappa l'intero file (137 GiB) e lo tocca per intero: nel test ha superato **90 GB di RSS in 25 s** durante il caricamento (fermato dal cap),
  senza residenza limitata né streaming degli esperti. Nessun numero di prefill/decode ottenibile entro il cap.
- Non era referenziato da `engine_select`, CMake, FastLLM: solo dai propri test e oracoli (compilati a mano).

## Contenuto
- `src/` — `LLMEngine/mlx/qwen4_native/*` (16 header: gguf, layer, trunk, gdn/attn/indexer/router/hc/ple/moe kernel, modello)
- `tests/` — `LLMEngine/mlx/tests/llm_qwen4_native_*.c` (17)
- `parity_oracles/` — erano `LLMEngine/tools/oracoli/qwen4_native_*.c` (15 sorgenti; NON erano tracciati da git perche' `.gitignore` ignora le cartelle `oracoli/`; qui rinominata per versionarli; binari esclusi)

Ultimo stato: commit `26db008` (WIP di un'altra AI, 2026-10-28: rilevamento dei layer GDN da `attn_qkv`, fallback CPU di `hc down/up`, PLE/MoE quant).
Storia completa in git: `git log -- LLMEngine/mlx/qwen4_native` (35 commit dal 2026-10-03).

## Ripristino
```bash
cp -R backups/qwen4_native/src/*   LLMEngine/mlx/qwen4_native/     # (creare la cartella)
cp backups/qwen4_native/tests/*    LLMEngine/mlx/tests/
cp backups/qwen4_native/parity_oracles/*  LLMEngine/tools/oracoli/
```
Compilazione (manuale, linkando la libreria del motore): `clang -O2 -I libs/mlx-deps/mlx-c-src -I LLMEngine/mlx -I LLMCodec/include <file>.c -L LLMEngine/mlx/build-gmbasic -lllm_ai_mlx -lllmcodec_ggml -framework Accelerate`.
