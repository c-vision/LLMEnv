/* ds4_shim.c — API minimale attorno al motore ds4 (vendorizzato) per usarlo come backend.
 * ds4 e' COMPLETO: qui solo load + generate(testo)->callback di testo. */
#include "ds4.h"
#include <stdlib.h>
#include <string.h>

static ds4_engine* g_e = NULL;

typedef void (*ds4_vendor_cb)(void* ud, const char* text, int len);
typedef struct { ds4_engine* e; ds4_vendor_cb cb; void* ud; } shim_ctx;

static void shim_emit(void* ud, int token) {
    shim_ctx* c = (shim_ctx*)ud; size_t len = 0;
    char* s = ds4_token_text(c->e, token, &len);
    if (s) { c->cb(c->ud, s, (int)len); free(s); }
}

int ds4_vendor_load(const char* path, int stream_experts, int ctx_size) {
    ds4_engine_options opt; memset(&opt, 0, sizeof opt);
    opt.model_path = path;
    opt.context_size = ctx_size;
    opt.ssd_streaming = stream_experts ? true : false;
    return ds4_engine_open(&g_e, &opt);
}

int ds4_vendor_generate(const char* prompt, int max_tokens, int ctx_size, ds4_vendor_cb cb, void* ud) {
    if (!g_e) return -1;
    ds4_tokens t; memset(&t, 0, sizeof t);
    ds4_chat_begin(g_e, &t);
    ds4_chat_append_message(g_e, &t, "user", prompt);
    ds4_chat_append_assistant_prefix(g_e, &t, DS4_THINK_NONE);
    shim_ctx c = { g_e, cb, ud };
    int rc = ds4_engine_generate_argmax(g_e, &t, max_tokens, ctx_size, shim_emit, NULL, &c, NULL, NULL);
    ds4_tokens_free(&t);
    return rc;
}

void ds4_vendor_close(void) { if (g_e) { ds4_engine_close(g_e); g_e = NULL; } }
