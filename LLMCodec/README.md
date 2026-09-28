# LLMCodec

Progetto C autonomo di LLMEnv: formati GGUF/safetensors + codec quant (plugin `dlopen`).

- ABI pubblica: `include/llmcodec.h` (registry plugin + tipo `LlmCodec`).
- Confini: nessuna dipendenza esterna oltre `libm` e i propri plugin (caricati a runtime).
- Il CLI `gguf_convert` (conversione MLX/safetensors → GGUF) richiede LLMToken ed è opzionale:
  `-DLLMCODEC_BUILD_CONVERTER=ON`.

## Build standalone

```sh
cmake -S LLMCodec -B /tmp/b-codec
cmake --build /tmp/b-codec -j
ctest --test-dir /tmp/b-codec --output-on-failure
```

Le `.llmarch` usate dai test sono in `ABI/`, le fixture reali in `samples/Fixtures/codec/`.
