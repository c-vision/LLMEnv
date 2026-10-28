# LLMCodec

Progetto C autonomo di LLMEnv: formati GGUF/safetensors + codec quant (plugin `dlopen`).

- ABI pubblica: `include/llmcodec.h` (registry plugin + tipo `LlmCodec`).
- Tassonomia d'errore GGUF: `include/llmcodec_gguf_status.h` (`LlmCodecGgufStatus`, un codice per
  causa) + le varianti `llmcodec_gguf_parse_mem_status` / `llmcodec_gguf_read_{meta,info}_status`
  in `llmcodec_gguf_read.h`.
- Gate differenziale del dequant (manuale, fuori CMake): `tools/quant_diff.c` con i testimoni
  `tools/quant_diff_ggml.c` (llama.cpp/ggml) e `tools/quant_diff_gguflib.c` (gguflib).
- Confini: nessuna dipendenza esterna oltre `libm` e i propri plugin (caricati a runtime).
- Il CLI `gguf_convert` (conversione MLX/safetensors → GGUF) richiede LLMToken ed è opzionale:
  `-DLLMCODEC_BUILD_CONVERTER=ON`.

## Build standalone

```sh
cmake -S LLMCodec -B /tmp/b-codec
cmake --build /tmp/b-codec -j
ctest --test-dir /tmp/b-codec --output-on-failure
```

Le `.llmarch` usate dai test sono in `ABI/`, le fixture reali in `tools/Fixtures/codec/`.
