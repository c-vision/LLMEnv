/* ds4_bridge.h — bridge TOKEN-LEVEL verso il motore ds4 (ds4_engine + ds4_session).
 * Espone le primitive che servono a LLMEngine: sync (prefill/extend), argmax, logits. */
#ifndef DS4_BRIDGE_H
#define DS4_BRIDGE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct ds4_bridge ds4_bridge;

/* Apre il MOTORE (pesi; nessuna sessione: le sessioni si creano con `_session_new`).
 * `prefill_chunk` = 0 -> default del motore. */
ds4_bridge* ds4_bridge_open(const char* model, int ctx, int stream, int prefill_chunk);
/* inspect_only=1: valida solo configurazione/metadati (nessun peso caricato). */
ds4_bridge* ds4_bridge_open_ex(const char* model, int ctx, int stream, int prefill_chunk, int inspect_only);
/* Come `_ex`, con il tetto della cache esperti in byte per lo streaming (`cache_bytes = 0`:
 * auto-sizing del motore, che usa la RAM raccomandata meno i pesi non-esperto). */
ds4_bridge* ds4_bridge_open_opts(const char* model, int ctx, int stream, int prefill_chunk,
                                 uint64_t cache_bytes, int inspect_only);

/* Opzioni complete (MTP/DSpark, offload, chunk). Le altre `open` sono scorciatoie. */
typedef struct {
    int ctx;                /* 0 = default del bridge (32768) */
    int stream;             /* 1 = esperti in streaming da disco */
    int prefill_chunk;      /* 0 = default del motore */
    uint64_t cache_bytes;   /* 0 = auto-sizing del motore */
    const char* mtp_path;   /* NULL/"" = nessun draft MTP */
    int dspark;             /* 1 = proposta speculativa DSpark (richiede mtp_path) */
    int dspark_strict;      /* 1 = accettazione strict */
    int inspect_only;       /* 1 = valida solo i metadati */
} ds4_bridge_options;

ds4_bridge* ds4_bridge_open2(const char* model, const ds4_bridge_options* o);
/* 1 se il motore aperto ha un draft MTP/DSpark utilizzabile. */
int ds4_bridge_has_mtp(ds4_bridge* b);
void        ds4_bridge_close(ds4_bridge* b);

/* ---- sessioni esplicite ---------------------------------------------------------------------
 *
 * `ds4_bridge` = il MOTORE (pesi, KV dimensionata all'apertura). `ds4_bridge_session` = una
 * conversazione con la sua KV. L'upstream permette piu' sessioni sullo stesso motore
 * (`ds4_session_create` per ognuna): qui si espone quella possibilita', perche' il server la usa
 * (`serve --batch`). Le funzioni "senza session" piu' sotto restano e usano una sessione di
 * default creata al primo uso — compatibilita' per i chiamanti vecchi, non per il nuovo codice. */
typedef struct ds4_bridge_session ds4_bridge_session;

/* Nuova sessione sul motore. NULL se il motore non e' aperto o la creazione fallisce.
 * Il chiamante la libera con `ds4_bridge_session_free` PRIMA di chiudere il bridge. */
ds4_bridge_session* ds4_bridge_session_new(ds4_bridge* b);
void ds4_bridge_session_free(ds4_bridge_session* s);
/* Svuota la KV della sessione: butta la `ds4_session` e ne crea una nuova sullo stesso motore
 * (ds4 non espone un reset). 0 = ok. */
int  ds4_bridge_session_reset(ds4_bridge_session* s);

int  ds4_bridge_session_sync(ds4_bridge_session* s, const int* toks, int n);
int  ds4_bridge_session_argmax(ds4_bridge_session* s);
int  ds4_bridge_session_sample(ds4_bridge_session* s, float temperature, int top_k, float top_p,
                               float min_p, uint64_t* rng);
int  ds4_bridge_session_logits(ds4_bridge_session* s, float* out, int cap);

/* Compatibilita': sync sulla sessione di DEFAULT (creata al primo uso). Per il codice nuovo
 * usare `ds4_bridge_session_sync`. */
int  ds4_bridge_sync(ds4_bridge* b, const int* toks, int n);

/* Prossimo token greedy (sessione di DEFAULT, creata al primo uso). */
int  ds4_bridge_argmax(ds4_bridge* b);
/* Prossimo token con SAMPLING (temp<=0 => greedy). `rng` e' lo STATO del generatore e viene
 * avanzato: lo possiede il chiamante (una sessione = uno stato). `min_p` <= 0 disattiva il
 * filtro. Ritorna il token, o -1. Usa `ds4_session_sample` dell'upstream: nessun sampler
 * duplicato qui. */
int  ds4_bridge_sample(ds4_bridge* b, float temperature, int top_k, float top_p, float min_p,
                       uint64_t* rng);
/* Copia i logits dell'ultimo sync (cap = n_vocab). Ritorna #copiati o -1. */
int  ds4_bridge_logits(ds4_bridge* b, float* out, int cap);
int  ds4_bridge_vocab(ds4_bridge* b);

#ifdef __cplusplus
}
#endif
#endif
