/* ds4_shim.h — API minimale del motore ds4 vendorizzato (LLMEngine/ds4engine).
 * Testo->testo: load del modello + generate con callback di testo. Vedi ds4_shim.c. */
#ifndef DS4_SHIM_H
#define DS4_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

/* Callback di emissione testo (ud = user data, text non NUL-terminated, len byte). */
typedef void (*ds4_vendor_cb)(void* ud, const char* text, int len);

/* Carica il modello. stream_experts=1 -> streaming esperti da disco (SSD). Ritorna 0 su successo. */
int  ds4_vendor_load(const char* path, int stream_experts, int ctx_size);

/* Genera da un prompt testuale, max_tokens, emettendo il testo via cb. Ritorna 0 su successo. */
int  ds4_vendor_generate(const char* prompt, int max_tokens, int ctx_size, ds4_vendor_cb cb, void* ud);

/* Chiude e libera il modello caricato. */
void ds4_vendor_close(void);

#ifdef __cplusplus
}
#endif

#endif /* DS4_SHIM_H */
