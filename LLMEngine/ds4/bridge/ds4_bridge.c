/* ds4_bridge.c — implementazione del bridge token-level (prototipo). */
#include "ds4.h"
#include "ds4_bridge.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct ds4_bridge { ds4_engine* e; ds4_session* s; int ctx; };

ds4_bridge* ds4_bridge_open_ex(const char* model, int ctx, int stream, int prefill_chunk, int inspect_only);

ds4_bridge* ds4_bridge_open(const char* model, int ctx, int stream, int prefill_chunk) {
    return ds4_bridge_open_ex(model, ctx, stream, prefill_chunk, 0);
}

ds4_bridge* ds4_bridge_open_ex(const char* model, int ctx, int stream, int prefill_chunk, int inspect_only) {
    if (ctx <= 0) ctx = 32768;            /* default motore (context_size=0 non crea la sessione) */
    ds4_engine_options opt; memset(&opt, 0, sizeof opt);
    opt.inspect_only = inspect_only ? true : false;
    opt.model_path = model;
    opt.context_size = ctx;
    opt.ssd_streaming = stream ? true : false;
    if (prefill_chunk > 0) opt.prefill_chunk = (uint32_t)prefill_chunk;
    ds4_engine* e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) { fprintf(stderr, "bridge: open fallita\n"); return NULL; }
    ds4_session* s = NULL;
    if (ds4_session_create(&s, e, ctx) != 0) { fprintf(stderr, "bridge: session create fallita\n"); ds4_engine_close(e); return NULL; }
    ds4_bridge* b = (ds4_bridge*)calloc(1, sizeof *b);
    b->e = e; b->s = s; b->ctx = ctx;
    return b;
}

void ds4_bridge_close(ds4_bridge* b) {
    if (!b) return;
    if (b->s) ds4_session_free(b->s);
    if (b->e) ds4_engine_close(b->e);
    free(b);
}

int ds4_bridge_sync(ds4_bridge* b, const int* toks, int n) {
    if (!b || n < 0) return -1;
    int* v = (int*)malloc(sizeof(int) * (size_t)(n > 0 ? n : 1));
    if (!v) return -1;
    if (n > 0) memcpy(v, toks, sizeof(int) * (size_t)n);
    ds4_tokens t; memset(&t, 0, sizeof t); t.v = v; t.len = n; t.cap = n;
    char err[256]; err[0] = 0;
    int rc = ds4_session_sync(b->s, &t, err, sizeof err);
    if (rc != 0) fprintf(stderr, "bridge: sync rc=%d err=%s\n", rc, err);
    free(v);
    return rc;
}

int ds4_bridge_argmax(ds4_bridge* b) { return b ? ds4_session_argmax(b->s) : -1; }
int ds4_bridge_logits(ds4_bridge* b, float* out, int cap) { return b ? ds4_session_copy_logits(b->s, out, cap) : -1; }
int ds4_bridge_vocab(ds4_bridge* b) { return b ? ds4_engine_vocab_size(b->e) : -1; }
