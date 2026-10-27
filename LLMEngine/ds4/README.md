# LLMEngine/ds4 — motore nativo GGUF (ds4) + kernel

Layout pensato per aggiornare ds4 senza toccare il nostro codice:

- `vendor/`  : sorgenti ds4 **non modificati** (MIT) + `metal/` + `licenses/` + MANIFEST.sha256
- `patches/` : nostre modifiche a ds4, in serie (`0001-*.patch`, …), applicate dal sync
- `bridge/`  : glue nostra (token-level `ds4_bridge`, adattatore ABI `LlmEngine`) — non è vendor
- `tools/`   : `ds4_sync.sh` (aggiorna da upstream), `ds4_check.sh` (verifica manifest)

## Aggiornare ds4
```bash
LLMEngine/ds4/tools/ds4_check.sh                 # deve essere pulito
LLMEngine/ds4/tools/ds4_sync.sh /path/ds4        # o un URL git + ref
```
Se serve una modifica al codice upstream: si crea un file in `patches/` (mai editare `vendor/`),
così il prossimo sync la riapplica e il conflitto è esplicito.

## Build
`CMakeLists.txt` definisce il target `ds4engine` (opzione `DS4ENGINE_STANDALONE=ON`), **non**
ancora collegato a `llm_ai_mlx`: l'integrazione avviene nella fase F3 del piano.
