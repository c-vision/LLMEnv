/* llmengine_plugin.c — registry dei PLUGIN nativi (dlopen). Vedi llmengine_plugin.h.
 *
 * Un plugin = una dylib `libllmengine_plugin_<kind>_<name>.dylib` che esporta UN descrittore
 * (`llmcodec_describe` / `llmengine_plugin_prim_describe`). Ricerca in: opzione `plugin_dir`,
 * <cwd risalendo>/Resources/plugins/<rid>/, poi la ricerca di sistema. Cache per (kind,name). */
#include "llmengine_plugin.h"
#include "../Options/llm_engine_opts.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <dirent.h>

static const char* plugin_rid(void)
{
#if defined(__aarch64__)
    return "osx-arm64";
#else
    return "osx-x64";
#endif
}

static const char* find_plugin(const char* kind, const char* name)
{
    static char buf[4096];
    char fname[256];
    snprintf(fname, sizeof fname, "libllmengine_plugin_%s_%s.dylib", kind, name);
    char pdir[2048];
    const char* env = (llm_ai_mlx_opt_get_str("plugin_dir", pdir, sizeof pdir) == 0 && pdir[0]) ? pdir : NULL;
    if (env && *env) { snprintf(buf, sizeof buf, "%s/%s", env, fname); if (access(buf, R_OK) == 0) return buf; }
    char cwd[2048];
    if (getcwd(cwd, sizeof cwd) != NULL) {
        char dir[2048]; snprintf(dir, sizeof dir, "%s", cwd);
        for (int up = 0; up < 12; ++up) {
            snprintf(buf, sizeof buf, "%s/Resources/plugins/%s/%s", dir, plugin_rid(), fname);
            if (access(buf, R_OK) == 0) return buf;
            char* s = strrchr(dir, '/');
            if (s == NULL || s == dir) break;
            *s = 0;
        }
    }
    snprintf(buf, sizeof buf, "%s", fname);
    return buf;
}

const char* llmengine_plugin_path(const char* kind, const char* name)
{
    if (!kind || !name) return NULL;
    const char* p = find_plugin(kind, name);
    return (p && access(p, R_OK) == 0) ? p : NULL;
}

/* cache (kind,name) -> descrittore; tiene la dylib aperta. */
typedef struct { char kind[16]; char name[64]; void* h; const void* desc; } PluginCache;
static PluginCache g_cache[128];
static int g_cache_n = 0;

static const void* cache_get(const char* kind, const char* name)
{
    for (int i = 0; i < g_cache_n; i++)
        if (strcmp(g_cache[i].kind, kind) == 0 && strcmp(g_cache[i].name, name) == 0) return g_cache[i].desc;
    return NULL;
}
static void cache_put(const char* kind, const char* name, void* h, const void* desc)
{
    if (g_cache_n >= 128) { dlclose(h); return; }
    snprintf(g_cache[g_cache_n].kind, sizeof g_cache[0].kind, "%s", kind);
    snprintf(g_cache[g_cache_n].name, sizeof g_cache[0].name, "%s", name);
    g_cache[g_cache_n].h = h; g_cache[g_cache_n].desc = desc; g_cache_n++;
}

static void* open_plugin(const char* kind, const char* name)
{
    const char* p = find_plugin(kind, name);
    void* h = dlopen(p, RTLD_NOW | RTLD_LOCAL);
    if (h == NULL && strchr(p, '/') == NULL) h = dlopen(name, RTLD_NOW | RTLD_LOCAL);
    return h;
}

const LlmCodec* llmengine_plugin_codec_get(const char* name)
{
    if (name == NULL || *name == 0) return NULL;
    const void* d = cache_get("codec", name);
    if (d != NULL) return (const LlmCodec*)d;
    void* h = open_plugin("codec", name);
    if (h == NULL) return NULL;
    void* sym = dlsym(h, "llmcodec_describe");
    if (sym == NULL) { dlclose(h); return NULL; }
    const LlmCodec* c = ((const LlmCodec* (*)(void))sym)();
    if (c == NULL || c->name == NULL) { dlclose(h); return NULL; }
    cache_put("codec", name, h, (const void*)c);
    return c;
}

const LlmEnginePluginPrim* llmengine_plugin_prim_get(const char* name)
{
    if (name == NULL || *name == 0) return NULL;
    const void* d = cache_get("prim", name);
    if (d != NULL) return (const LlmEnginePluginPrim*)d;
    void* h = open_plugin("prim", name);
    if (h == NULL) return NULL;
    void* sym = dlsym(h, "llmengine_plugin_prim_describe");
    if (sym == NULL) { dlclose(h); return NULL; }
    const LlmEnginePluginPrim* p = ((const LlmEnginePluginPrim* (*)(void))sym)();
    if (p == NULL || p->name == NULL || p->abi_version != LLMENGINE_PLUGIN_ABI_VERSION) { dlclose(h); return NULL; }
    cache_put("prim", name, h, (const void*)p);
    return p;
}

const LlmEnginePluginPrimMLX* llmengine_plugin_prim_mlx_get(const char* name)
{
    if (name == NULL || *name == 0) return NULL;
    const void* d = cache_get("primmlx", name);
    if (d != NULL) return (const LlmEnginePluginPrimMLX*)d;
    void* h = open_plugin("primmlx", name);
    if (h == NULL) return NULL;
    void* sym = dlsym(h, "llmengine_plugin_prim_mlx_describe");
    if (sym == NULL) { dlclose(h); return NULL; }
    const LlmEnginePluginPrimMLX* p = ((const LlmEnginePluginPrimMLX* (*)(void))sym)();
    if (p == NULL || p->name == NULL || p->abi_version != LLMENGINE_PLUGIN_ABI_VERSION) { dlclose(h); return NULL; }
    cache_put("primmlx", name, h, (const void*)p);
    return p;
}

int llmengine_plugin_list(void)
{
    char pdir[2048];
    const char* pdenv = (llm_ai_mlx_opt_get_str("plugin_dir", pdir, sizeof pdir) == 0 && pdir[0]) ? pdir : NULL;
    const char* dirs[2] = { pdenv, NULL };
    static char dflt[2048];
    snprintf(dflt, sizeof dflt, "Resources/plugins/%s", plugin_rid());
    dirs[1] = dflt;
    int found = 0;
    for (int i = 0; i < 2; i++) {
        if (dirs[i] == NULL) continue;
        printf("LLM PLUGIN: dir '%s':\n", dirs[i]);
        /* scansione semplice: nomi che iniziano con libllmcodec_ / libllmengine_plugin_prim_ */
        DIR* dp = opendir(dirs[i]);
        if (dp == NULL) { printf("  (assente)\n"); continue; }
        struct dirent* e; int n = 0;
        while ((e = readdir(dp)) != NULL) {
            if (strncmp(e->d_name, "libllmcodec_", 13) == 0 || strncmp(e->d_name, "libllmengine_plugin_prim_", 12) == 0) {
                printf("  %s\n", e->d_name); n++;
            }
        }
        closedir(dp);
        if (n == 0) printf("  (nessun plugin)\n");
        found += n;
    }
    return found;
}
