/* llmcodec.c — registry plugin CODEC (dlopen + cache). */
#include "llmcodec.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

#define LLMCODEC_CODEC_MAX 32
static const LlmCodec* g_cache[LLMCODEC_CODEC_MAX];
static void*           g_handle[LLMCODEC_CODEC_MAX];
static char            g_name[LLMCODEC_CODEC_MAX][32];
static int             g_n = 0;

static const char* g_dir_candidates[] = {
    NULL,                                   /* $LLMCODEC_PLUGIN_DIR (gestito a parte) */
    "Resources/plugins",
    "../Resources/plugins",
    "../../Resources/plugins",
    "../../../Resources/plugins",
    ".",
};

static const char* g_suffix[] = { ".dylib", ".so", "" };

static void* try_open(const char* dir, const char* name, const char* suf) {
    char p[1400];
    if (dir && dir[0]) snprintf(p, sizeof p, "%s/libllmcodec_%s%s", dir, name, suf);
    else               snprintf(p, sizeof p, "libllmcodec_%s%s", name, suf);
    void* h = dlopen(p, RTLD_NOW | RTLD_LOCAL);
    return h;
}

const LlmCodec* llmcodec_resolve(const char* name) {
    if (name == NULL || name[0] == 0) return NULL;
    for (int i = 0; i < g_n; ++i) if (strcmp(g_name[i], name) == 0) return g_cache[i];

    void* h = NULL;
    const char* env = getenv("LLMCODEC_PLUGIN_DIR");
    for (size_t s = 0; s < sizeof g_suffix / sizeof g_suffix[0] && h == NULL; ++s) {
        if (env && env[0]) h = try_open(env, name, g_suffix[s]);
        for (size_t d = 1; d < sizeof g_dir_candidates / sizeof g_dir_candidates[0] && h == NULL; ++d)
            h = try_open(g_dir_candidates[d], name, g_suffix[s]);
    }
    if (h == NULL) return NULL;

    typedef const LlmCodec* (*desc_fn)(void);
    desc_fn fn = (desc_fn)dlsym(h, "llmcodec_describe");
    if (fn == NULL) { dlclose(h); return NULL; }
    const LlmCodec* c = fn();
    if (c == NULL || c->abi_version != LLMCODEC_CODEC_ABI_VERSION) { dlclose(h); return NULL; }

    if (g_n < LLMCODEC_CODEC_MAX) {
        snprintf(g_name[g_n], sizeof g_name[g_n], "%s", name);
        g_cache[g_n] = c; g_handle[g_n] = h; g_n++;
    }
    return c;
}

void* llmcodec_symbol(const char* name, const char* symbol) {
    if (llmcodec_resolve(name) == NULL) return NULL;   /* carica + cache */
    for (int i = 0; i < g_n; ++i)
        if (strcmp(g_name[i], name) == 0) return dlsym(g_handle[i], symbol);
    return NULL;
}
