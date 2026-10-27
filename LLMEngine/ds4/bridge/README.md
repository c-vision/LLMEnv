# bridge/ — glue nostra (non upstream)
- `ds4_bridge.{c,h}`: API token-level (open/sync/argmax/logits) sopra `ds4_engine`/`ds4_session`.
- Adattatore alla vtable `LlmEngine` (`llmengine.h`): **fase F3** del piano, non ancora presente.
Qui si può modificare liberamente: non è codice vendorizzato (upstream resta in `vendor/`).
