/* ds4_bridge.h — bridge TOKEN-LEVEL verso il motore ds4 (ds4_engine + ds4_session).
 * Espone le primitive che servono a LLMEngine: sync (prefill/extend), argmax, logits. */
#ifndef DS4_BRIDGE_H
#define DS4_BRIDGE_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4_bridge ds4_bridge;

/* Apre il motore e crea una sessione. prefill_chunk=0 -> default del motore. */
ds4_bridge* ds4_bridge_open(const char* model, int ctx, int stream, int prefill_chunk);
void        ds4_bridge_close(ds4_bridge* b);

/* Sincronizza la sessione al prefisso `toks[0..n)`: prefill se nuovo, altrimenti solo il suffisso. */
int  ds4_bridge_sync(ds4_bridge* b, const int* toks, int n);
/* Prossimo token greedy. */
int  ds4_bridge_argmax(ds4_bridge* b);
/* Copia i logits dell'ultimo sync (cap = n_vocab). Ritorna #copiati o -1. */
int  ds4_bridge_logits(ds4_bridge* b, float* out, int cap);
int  ds4_bridge_vocab(ds4_bridge* b);

#ifdef __cplusplus
}
#endif
#endif
