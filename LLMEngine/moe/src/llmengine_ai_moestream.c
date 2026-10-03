/* llmengine_moe_stream.c — API C stabile del provider AI embedded "moestream" (M14).
 *
 * Esposta come dylib `libllmengine_moe_stream` e consumata dal C# via P/Invoke (eccezione AI
 * embedded). NON e' un bridge generico C<->C#: e' la libreria del motore LLM, come
 * `libllm_ai_mlx`. ABI piatta (struct blittable, interi) per un marshalling semplice. */
#include "llmengine_moe_provider.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int32_t  mode;       /* 0 = SWAP (default), 1 = RESIDENT */
    int32_t  device;     /* 0 = AUTO, 1 = CPU, 2 = METAL */
    uint64_t cache_mb;
    uint32_t workers;    /* async I/O */
    uint32_t topn;       /* prefetch top-n per layer (0 = off) */
    uint32_t cap;        /* bound coda prefetch */
} LlmEngineMoeAiMoestreamConfig;

typedef struct {
    int32_t  loaded;
    int32_t  mode;
    int32_t  device;
    uint32_t layers;
    uint32_t experts;
    uint64_t bytes_read;
    uint64_t expert_bytes;
    uint64_t resident_bytes;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t cache_evictions;
    uint64_t cache_resident_hits;   /* K2 */
    uint64_t prefetch_predictions;
    uint64_t prefetch_hits;
    uint64_t prefetch_wasted;
} LlmEngineMoeAiMoestreamStats;

static void to_runtime_config(const LlmEngineMoeAiMoestreamConfig* in, LlmEngineMoeRuntimeConfig* out) {
    if (in == NULL) { llmengine_moe_config_defaults(out); return; }
    memset(out, 0, sizeof(*out));
    out->mode = (in->mode == 1) ? LLMENGINE_MOE_MODE_RESIDENT : LLMENGINE_MOE_MODE_SWAP;
    out->device = (in->device == 2) ? LLMENGINE_MOE_DEVICE_METAL
               : (in->device == 1) ? LLMENGINE_MOE_DEVICE_CPU : LLMENGINE_MOE_DEVICE_AUTO;
    out->cache_mb = in->cache_mb;
    out->async_workers = in->workers;
    out->prefetch_topn = in->topn;
    out->prefetch_cap = in->cap;
}

LlmEngineMoeProvider* llmengine_ai_moestream_create(void) { return llmengine_moe_provider_create(); }

void llmengine_ai_moestream_destroy(LlmEngineMoeProvider* p) { llmengine_moe_provider_destroy(p); }

int llmengine_ai_moestream_load(LlmEngineMoeProvider* p, const char* name, const char* path,
                          const LlmEngineMoeAiMoestreamConfig* cfg) {
    if (p == NULL || name == NULL || path == NULL) return LLMENGINE_MOE_ERR_INVALID;
    LlmEngineMoeRuntimeConfig c;
    to_runtime_config(cfg, &c);
    LlmEngineMoeStatus s = llmengine_moe_provider_load(p, name, path, &c);
    return (int)s;
}

int llmengine_ai_moestream_unload(LlmEngineMoeProvider* p, const char* name) {
    return (int)llmengine_moe_provider_unload(p, name);
}

int llmengine_ai_moestream_reconfigure(LlmEngineMoeProvider* p, const char* name,
                                 const LlmEngineMoeAiMoestreamConfig* cfg) {
    if (p == NULL || name == NULL || cfg == NULL) return LLMENGINE_MOE_ERR_INVALID;
    LlmEngineMoeRuntimeConfig c;
    to_runtime_config(cfg, &c);
    return (int)llmengine_moe_provider_reconfigure(p, name, &c);
}

/* Setter per-attributo (i comandi AI MODEL CACHE/PREFETCH/DEVICE cambiano UNA cosa sola). */
int llmengine_ai_moestream_set_cache(LlmEngineMoeProvider* p, const char* name, uint64_t cache_mb) {
    LlmEngineMoeRuntime* rt = llmengine_moe_provider_get(p, name);
    if (rt == NULL) return LLMENGINE_MOE_ERR_MODEL_NOT_FOUND;
    return (int)llmengine_moe_set_cache_mb(rt, cache_mb);
}

int llmengine_ai_moestream_set_prefetch(LlmEngineMoeProvider* p, const char* name, uint32_t topn, uint32_t cap) {
    LlmEngineMoeRuntime* rt = llmengine_moe_provider_get(p, name);
    if (rt == NULL) return LLMENGINE_MOE_ERR_MODEL_NOT_FOUND;
    if (cap > 0) { (void)llmengine_moe_set_async(rt, 4, cap); } /* prefetch senza async non ha senso */
    return (int)llmengine_moe_set_prefetch(rt, topn, 0);
}

int llmengine_ai_moestream_set_device(LlmEngineMoeProvider* p, const char* name, int device) {
    LlmEngineMoeRuntime* rt = llmengine_moe_provider_get(p, name);
    if (rt == NULL) return LLMENGINE_MOE_ERR_MODEL_NOT_FOUND;
    LlmEngineMoeDevice d = (device == 2) ? LLMENGINE_MOE_DEVICE_METAL
                   : (device == 1) ? LLMENGINE_MOE_DEVICE_CPU : LLMENGINE_MOE_DEVICE_AUTO;
    return (int)llmengine_moe_set_device(rt, d);
}

/* M21: residenza — PROMOTE porta tutti gli esperti in RAM (ACTIVE), DEMOTE torna on-demand (WARM).
 * Usa il provider (scheduler LRU-aware), non il runtime da solo. */
int llmengine_ai_moestream_promote(LlmEngineMoeProvider* p, const char* name) {
    return (int)llmengine_moe_provider_promote(p, name);
}
int llmengine_ai_moestream_demote(LlmEngineMoeProvider* p, const char* name) {
    return (int)llmengine_moe_provider_demote(p, name);
}
/* M21 scheduler: il chiamante (C#/C) inietta la memoria disponibile PRIMA di load/promote. */
void llmengine_ai_moestream_set_mem_available(LlmEngineMoeProvider* p, uint64_t bytes) {
    llmengine_moe_provider_set_mem_available(p, bytes);
}
/* M21 orchestrazione: priorità del modello (guidata da AI SET PRIORITY). */
void llmengine_ai_moestream_set_priority(LlmEngineMoeProvider* p, const char* name, int priority) {
    llmengine_moe_provider_set_priority(p, name, priority);
}

int llmengine_ai_moestream_count(const LlmEngineMoeProvider* p) {
    return (int)llmengine_moe_provider_count(p);
}

const char* llmengine_ai_moestream_name_at(const LlmEngineMoeProvider* p, uint32_t i) {
    return llmengine_moe_provider_name_at(p, i);
}

int llmengine_ai_moestream_stats(const LlmEngineMoeProvider* p, const char* name, LlmEngineMoeAiMoestreamStats* out) {
    if (out == NULL) return LLMENGINE_MOE_ERR_INVALID;
    LlmEngineMoeStreamStats s;
    LlmEngineMoeStatus st = llmengine_moe_provider_stats((LlmEngineMoeProvider*)p, name, &s);
    if (st != LLMENGINE_MOE_OK) return (int)st;
    memset(out, 0, sizeof(*out));
    out->loaded = s.loaded;
    out->mode = (int32_t)s.mode;
    out->device = (int32_t)s.device;
    out->layers = s.layers;
    out->experts = s.experts;
    out->bytes_read = s.bytes_read;
    out->expert_bytes = s.expert_bytes;
    out->resident_bytes = s.resident_bytes;
    out->cache_hits = s.cache_hits;
    out->cache_misses = s.cache_misses;
    out->cache_evictions = s.cache_evictions;
    out->cache_resident_hits = s.cache_resident_hits;
    out->prefetch_predictions = s.prefetch_predictions;
    out->prefetch_hits = s.prefetch_hits;
    out->prefetch_wasted = s.prefetch_wasted;
    return 0;
}

const char* llmengine_ai_moestream_status_str(int status) {
    return llmengine_moe_status_str((LlmEngineMoeStatus)status);
}

static uint32_t ms_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* Generazione greedy (token-id → token-id) via l'adapter reale del modello (M15.5).
 * Ritorna il numero di token generati, o <0 (LlmEngineMoeStatus). */
int llmengine_ai_moestream_generate(LlmEngineMoeProvider* p, const char* name,
                              const int32_t* prompt_ids, uint32_t n_prompt,
                              int32_t* out_ids, uint32_t n_out) {
    if (p == NULL || name == NULL || prompt_ids == NULL || out_ids == NULL) return LLMENGINE_MOE_ERR_INVALID;
    LlmEngineMoeRuntime* rt = llmengine_moe_provider_get(p, name);
    if (rt == NULL) return LLMENGINE_MOE_ERR_MODEL_NOT_FOUND;
    llmengine_moe_provider_mark_used(p, name);   /* M21: uso → LRU */
    uint8_t cb[96]; size_t got = 0;
    if (llmengine_moe_get_tensor(rt, "__config__", cb, sizeof(cb), &got) != LLMENGINE_MOE_OK || got < 48)
        return LLMENGINE_MOE_ERR_FORMAT;
    if (ms_u32(cb) != 0x44414D47u) return LLMENGINE_MOE_ERR_FORMAT;  /* 'GMAD' */
    uint32_t vocab = ms_u32(cb + 44);
    if (vocab == 0) return LLMENGINE_MOE_ERR_FORMAT;

    float* in = (float*)malloc((size_t)(n_prompt ? n_prompt : 1) * sizeof(float));
    float* logits = (float*)malloc((size_t)vocab * sizeof(float));
    if (!in || !logits) { free(in); free(logits); return LLMENGINE_MOE_ERR_NO_MEMORY; }
    for (uint32_t i = 0; i < n_prompt; ++i) in[i] = (float)prompt_ids[i];
    LlmEngineMoeStatus s = llmengine_moe_forward(rt, in, n_prompt, logits);
    if (s != LLMENGINE_MOE_OK) { free(in); free(logits); return (int)s; }
    for (uint32_t g = 0; g < n_out; ++g) {
        int best = 0; float bv = logits[0];
        for (uint32_t v = 1; v < vocab; ++v) if (logits[v] > bv) { bv = logits[v]; best = (int)v; }
        out_ids[g] = best;
        if (g + 1 < n_out) { float one = (float)best; s = llmengine_moe_forward(rt, &one, 1, logits); if (s != LLMENGINE_MOE_OK) { free(in); free(logits); return (int)s; } }
    }
    free(in); free(logits);
    return (int)n_out;
}

/* Estrae un blob del container (es. "__tokenizer__", "__model_type__") su file.
 * Ritorna i byte scritti, o <0 (LlmEngineMoeStatus). Serve al completer C# per il tokenizer
 * EMBEDDATO: un solo file .gmoe, niente tokenizer.json esterno. */
long llmengine_ai_moestream_extract(LlmEngineMoeProvider* p, const char* name, const char* tensor, const char* out_path) {
    if (p == NULL || name == NULL || tensor == NULL || out_path == NULL) return LLMENGINE_MOE_ERR_INVALID;
    LlmEngineMoeRuntime* rt = llmengine_moe_provider_get(p, name);
    if (rt == NULL) return LLMENGINE_MOE_ERR_MODEL_NOT_FOUND;
    size_t cap = 96u << 20;   /* tokenizer.json noto ~11 MB; margine ampio */
    uint8_t* buf = (uint8_t*)malloc(cap);
    if (buf == NULL) return LLMENGINE_MOE_ERR_NO_MEMORY;
    size_t got = 0;
    LlmEngineMoeStatus s = llmengine_moe_get_tensor(rt, tensor, buf, cap, &got);
    if (s != LLMENGINE_MOE_OK) { free(buf); return (int)s; }
    FILE* f = fopen(out_path, "wb");
    if (f == NULL) { free(buf); return LLMENGINE_MOE_ERR_IO; }
    size_t w = fwrite(buf, 1, got, f);
    fclose(f); free(buf);
    return (w == got) ? (long)got : (long)LLMENGINE_MOE_ERR_IO;
}
