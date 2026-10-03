/* llmengine_moe.c — implementazione del MoERuntime (M1: skeleton funzionante).
 *
 * ZERO codice Qwen/DeepSeek/GLM qui: il runtime conosce solo l'adapter e gli esperti. */
#include "llm_engine_opts.h"   /* registro opzioni condiviso (parametri senza env) */
#include "llmengine_moe.h"
#include "llmengine_moe_cache.h"
#include "llmengine_moe_async.h"
#include "llmengine_moe_prefetch.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef __APPLE__
#include "llmengine_moe_backend_metal.h"
#endif

#define LLMENGINE_MOE_MAX_ADAPTERS 8

struct LlmEngineMoeRuntime {
    LlmEngineMoeModelAdapter adapters[LLMENGINE_MOE_MAX_ADAPTERS];
    uint32_t adapter_count;
    uint32_t active;            /* indice dell'adapter attivo */
    int loaded;
    LlmEngineMoeModelMeta meta;
    uint8_t* scratch;           /* buffer di fetch esperto (riusato) */
    size_t scratch_cap;
    uint64_t bytes_read;        /* byte letti da disco per gli esperti */
    LlmEngineMoeExpertCache* cache;   /* expert cache opzionale (LLMENGINE_MOE_CACHE_MB) */
    LlmEngineMoeAsync* async;         /* async I/O opzionale (LLMENGINE_MOE_ASYNC_WORKERS) */
    LlmEngineMoePredictor* predictor; /* prefetch cross-layer opzionale (LLMENGINE_MOE_PREFETCH_TOPN) */
    LlmEngineMoeMemMgr* mem;          /* memory manager opzionale (M10) */
    LlmEngineMoeKvCache* kv;          /* KV cache separata (M11) */
    LlmEngineMoeProfiler* prof;       /* profiler opzionale (M12), NON posseduto */
    LlmEngineMoeLoadMode mode;        /* SWAP (default) | RESIDENT (M14) */
    LlmEngineMoeDevice device;        /* AUTO | CPU | METAL (M14) */
    uint64_t cfg_cache_mb;      /* cache_mb configurata (M14: usata dallo store GGUF dell'adapter) */
    /* M18: frequenza d'uso per (layer, expert) — persistita in `<container>.stats` (advisory). */
    uint32_t* freq;             /* [freq_layers * freq_experts] */
    uint32_t freq_layers, freq_experts;
    uint64_t expert_fetches;    /* M20: fetch totali di esperti (hit+miss) in questa sessione */
    char stats_path[1200];
    /* M17: controller ADATTIVO (opt-in LLMENGINE_MOE_ADAPTIVE=1) — prefetch depth + cache budget. */
    int adaptive;
    int hotness;                /* D19: expert cache adattiva (hotness+profilo) */
    uint32_t adapt_topn, adapt_pf_min, adapt_pf_max;
    uint32_t adapt_cache_mb, adapt_cache_min, adapt_cache_max;
    uint32_t adapt_steps, adapt_interval;
    uint64_t adapt_last_pred, adapt_last_hits, adapt_last_actual;
    LlmEngineMoePressure adapt_last_press;
    /* Bridge metriche store dell'adapter (es. GGUF stile DS4): pubblicate dall'adapter via
     * llmengine_moe_set_stream_metrics e usate da llmengine_moe_stream_stats quando rt->cache e' assente. */
    LlmEngineMoeStreamStats ext;
    int ext_valid;
};

static LlmEngineMoeStatus llmengine_moe_rt_load_expert(void* ctx, LlmEngineMoeExpertKey key,
                                           void* buf, size_t cap, size_t* got);
static LlmEngineMoeStatus llmengine_moe_preload_resident(LlmEngineMoeRuntime* rt);
static void llmengine_moe_adaptive_step(LlmEngineMoeRuntime* rt);   /* M17 */

const char* llmengine_moe_status_str(LlmEngineMoeStatus s) {
    switch (s) {
        case LLMENGINE_MOE_OK:                  return "OK";
        case LLMENGINE_MOE_ERR_INVALID:         return "invalid argument";
        case LLMENGINE_MOE_ERR_NO_MEMORY:       return "out of memory";
        case LLMENGINE_MOE_ERR_MODEL_NOT_FOUND: return "model not found";
        case LLMENGINE_MOE_ERR_IO:              return "I/O error";
        case LLMENGINE_MOE_ERR_NOT_IMPLEMENTED: return "not implemented";
        case LLMENGINE_MOE_ERR_ADAPTER:         return "adapter error";
        case LLMENGINE_MOE_ERR_FORMAT:          return "format error";
        case LLMENGINE_MOE_ERR_DEVICE:          return "device not available";
        default:                          return "unknown status";
    }
}

LlmEngineMoeRuntime* llmengine_moe_create(void) {
    LlmEngineMoeRuntime* rt = (LlmEngineMoeRuntime*)calloc(1, sizeof(*rt));
    return rt; /* NULL su OOM */
}

void llmengine_moe_destroy(LlmEngineMoeRuntime* rt) {
    if (rt == NULL) return;
    if (rt->loaded && rt->adapter_count > 0) {
        const LlmEngineMoeModelAdapter* a = &rt->adapters[rt->active];
        if (a->destroy != NULL && a->self != NULL) a->destroy(a->self);
    }
    llmengine_moe_mem_destroy(rt->mem);
    llmengine_moe_kv_destroy(rt->kv);
    llmengine_moe_predictor_destroy(rt->predictor);
    llmengine_moe_async_destroy(rt->async);
    llmengine_moe_cache_destroy(rt->cache);
    free(rt->freq);
    free(rt->scratch);
    free(rt);
}

LlmEngineMoeStatus llmengine_moe_register_adapter(LlmEngineMoeRuntime* rt, const LlmEngineMoeModelAdapter* adapter) {
    if (rt == NULL || adapter == NULL) return LLMENGINE_MOE_ERR_INVALID;
    if (adapter->load == NULL || adapter->forward == NULL) return LLMENGINE_MOE_ERR_INVALID;
    if (rt->adapter_count >= LLMENGINE_MOE_MAX_ADAPTERS) return LLMENGINE_MOE_ERR_NO_MEMORY;
    rt->adapters[rt->adapter_count++] = *adapter; /* copia della vtable */
    return LLMENGINE_MOE_OK;
}

void llmengine_moe_config_defaults(LlmEngineMoeRuntimeConfig* out) {
    if (out == NULL) return;
    memset(out, 0, sizeof(*out));
    out->mode = LLMENGINE_MOE_MODE_SWAP;
    out->device = LLMENGINE_MOE_DEVICE_AUTO;
    const char* cm = llm_ai_mlx_opt_env("LLMENGINE_MOE_CACHE_MB");
    out->cache_mb = (cm != NULL && atoi(cm) > 0) ? (uint64_t)atoi(cm) : 0;
    const char* aw = llm_ai_mlx_opt_env("LLMENGINE_MOE_ASYNC_WORKERS");
    out->async_workers = (aw != NULL && atoi(aw) > 0) ? (uint32_t)atoi(aw) : 0;
    const char* pt = llm_ai_mlx_opt_env("LLMENGINE_MOE_PREFETCH_TOPN");
    out->prefetch_topn = (pt != NULL && atoi(pt) > 0) ? (uint32_t)atoi(pt) : 0;
    const char* pc = llm_ai_mlx_opt_env("LLMENGINE_MOE_PREFETCH_CAP");
    out->prefetch_cap = (pc != NULL && atoi(pc) > 0) ? (uint32_t)atoi(pc) : 0;
    const char* ht = llm_ai_mlx_opt_env("LLMENGINE_MOE_HOTNESS");
    out->hotness = (ht != NULL && atoi(ht) > 0) ? 1 : 0;
}

/* D19: configura una cache appena creata (dims + hotness + profilo `<container>.stats`). */
/* K1: hook di eviction (definizione piu' sotto, usata alla creazione della cache). */
static void llmengine_moe_rt_evict_hook(void* ctx, LlmEngineMoeExpertKey key);

static void moe_cache_configure(LlmEngineMoeRuntime* rt, LlmEngineMoeExpertCache* cache, int hotness) {
    if (rt == NULL || cache == NULL) return;
    (void)llmengine_moe_cache_set_dims(cache, rt->meta.layer_count, rt->meta.expert_count);
    if (hotness) {
        llmengine_moe_cache_set_hotness(cache, 1);
        (void)llmengine_moe_cache_profile_load(cache, rt->stats_path);   /* advisory: file assente ok */
    }
}

LlmEngineMoeStatus llmengine_moe_load_model_ex(LlmEngineMoeRuntime* rt, const char* path, const LlmEngineMoeRuntimeConfig* cfg) {
    if (rt == NULL || path == NULL) return LLMENGINE_MOE_ERR_INVALID;
    LlmEngineMoeRuntimeConfig c;
    if (cfg != NULL) c = *cfg; else llmengine_moe_config_defaults(&c);

#ifdef __APPLE__
    if (c.device == LLMENGINE_MOE_DEVICE_METAL && !llmengine_moe_metal_available()) return LLMENGINE_MOE_ERR_DEVICE;
#else
    if (c.device == LLMENGINE_MOE_DEVICE_METAL) return LLMENGINE_MOE_ERR_DEVICE;
#endif

    if (rt->adapter_count == 0) return LLMENGINE_MOE_ERR_ADAPTER;

    /* Il file deve esistere (presenza/dimensione); il FORMATO lo apre l'adapter (GGUF). */
    FILE* fsz = fopen(path, "rb");
    if (fsz == NULL) return LLMENGINE_MOE_ERR_MODEL_NOT_FOUND;
    fseeko(fsz, 0, SEEK_END); uint64_t size = (uint64_t)ftello(fsz); fclose(fsz);
    if (size == 0) return LLMENGINE_MOE_ERR_FORMAT;

    memset(&rt->meta, 0, sizeof(rt->meta));
    rt->meta.total_bytes = size;

    snprintf(rt->stats_path, sizeof(rt->stats_path), "%s.stats", path);
    rt->mode = c.mode;
    rt->device = c.device;
    rt->cfg_cache_mb = c.cache_mb;
    rt->hotness = c.hotness;

    if (rt->meta.layer_count > 0 && rt->meta.expert_count > 0) {
        if (c.mode == LLMENGINE_MOE_MODE_RESIDENT) {
            /* Nessuno swap: cache >= expert_bytes (nessun async/prefetch). Preload dopo l'adapter. */
            uint64_t bytes = rt->meta.routed_bytes;
            uint64_t want_mb = (bytes + (1u << 20) - 1) >> 20;
            if (c.cache_mb > want_mb) want_mb = c.cache_mb;
            if (want_mb == 0) want_mb = 1;
            LlmEngineMoeExpertCache* cache = NULL;
            if (llmengine_moe_cache_create(&cache, want_mb * 1024ull * 1024ull) == LLMENGINE_MOE_OK) {
                llmengine_moe_cache_set_evict_hook(cache, llmengine_moe_rt_evict_hook, rt);   /* K1 */
                rt->cache = cache;
            }
        } else {
            if (c.cache_mb > 0) {
                LlmEngineMoeExpertCache* cache = NULL;
                if (llmengine_moe_cache_create(&cache, c.cache_mb * 1024ull * 1024ull) == LLMENGINE_MOE_OK) {
                    llmengine_moe_cache_set_evict_hook(cache, llmengine_moe_rt_evict_hook, rt);   /* K1 */
                    rt->cache = cache;
                }
            }
            if (c.async_workers > 0 && rt->cache != NULL) {
                LlmEngineMoeAsync* async = NULL;
                if (llmengine_moe_async_create(&async, c.async_workers, llmengine_moe_rt_load_expert, rt,
                                         rt->cache, c.prefetch_cap) == LLMENGINE_MOE_OK) {
                    llmengine_moe_async_set_profiler(async, rt->prof);   /* K1: trace richieste */
                    rt->async = async;
                }
            }
            if (c.prefetch_topn > 0 && rt->meta.layer_count > 0 && rt->meta.expert_count > 0) {
                LlmEngineMoePredictor* pred = NULL;
                if (llmengine_moe_predictor_create(&pred, rt->meta.layer_count, rt->meta.expert_count,
                                             c.prefetch_topn, 4) == LLMENGINE_MOE_OK)
                    rt->predictor = pred;
            }
        }
    }

    /* M17: controller adattivo (opt-in) — richiede prefetch attivo. */
    if (rt->predictor != NULL) {
        const char* ad = llm_ai_mlx_opt_env("LLMENGINE_MOE_ADAPTIVE");
        if (ad != NULL && *ad && strcmp(ad, "0") != 0) {
            rt->adaptive = 1;
            rt->adapt_topn = c.prefetch_topn;
            rt->adapt_pf_min = 1;
            uint32_t pfmax = c.prefetch_topn * 2;
            if (pfmax > rt->meta.expert_count) pfmax = rt->meta.expert_count;
            if (pfmax <= rt->adapt_topn) pfmax = rt->adapt_topn + 1;
            rt->adapt_pf_max = pfmax;
            rt->adapt_cache_mb = c.cache_mb;
            rt->adapt_cache_min = c.cache_mb / 4;
            rt->adapt_interval = 16;
            rt->adapt_last_press = LLMENGINE_MOE_PRESS_NORMAL;
        }
    }

    /* Primo adapter registrato che sa caricarlo. */
    for (uint32_t i = 0; i < rt->adapter_count; ++i) {
        LlmEngineMoeModelAdapter* a = &rt->adapters[i];
        LlmEngineMoeStatus ls = a->load(a->self, path, &rt->meta);
        if (ls == LLMENGINE_MOE_OK) {
            rt->active = i;
            rt->loaded = 1;
            if (rt->meta.total_bytes == 0) rt->meta.total_bytes = size;
            /* D19: hotness opt-in — la cache va creata ORA (meta noto: l'adapter lo riempie qui). */
            if (rt->hotness && rt->cache == NULL && c.cache_mb > 0) {
                LlmEngineMoeExpertCache* hc = NULL;
                if (llmengine_moe_cache_create(&hc, c.cache_mb * 1024ull * 1024ull) == LLMENGINE_MOE_OK) {
                    rt->cache = hc; moe_cache_configure(rt, hc, 1);
                }
            }
            if (c.mode == LLMENGINE_MOE_MODE_RESIDENT) llmengine_moe_preload_resident(rt);
            return LLMENGINE_MOE_OK;
        }
    }
    return LLMENGINE_MOE_ERR_ADAPTER;
}

LlmEngineMoeStatus llmengine_moe_load_model(LlmEngineMoeRuntime* rt, const char* path) {
    LlmEngineMoeRuntimeConfig c;
    llmengine_moe_config_defaults(&c);
    return llmengine_moe_load_model_ex(rt, path, &c);
}

int llmengine_moe_is_loaded(const LlmEngineMoeRuntime* rt) {
    return (rt != NULL) && rt->loaded;
}

const LlmEngineMoeModelMeta* llmengine_moe_meta(const LlmEngineMoeRuntime* rt) {
    if (rt == NULL || !rt->loaded) return NULL;
    return &rt->meta;
}

/* K1: eviction nel trace (il summary e il replay contano le stesse eviction). */
static void llmengine_moe_rt_evict_hook(void* ctx, LlmEngineMoeExpertKey key) {
    LlmEngineMoeRuntime* rt = (LlmEngineMoeRuntime*)ctx;
    if (rt != NULL && rt->prof != NULL)
        llmengine_moe_prof_event(rt->prof, LLMENGINE_MOE_TRACE_EVICT, key.layer, key.expert, 0, 1);
}

static LlmEngineMoeStatus llmengine_moe_rt_load_expert(void* ctx, LlmEngineMoeExpertKey key,
                                           void* buf, size_t cap, size_t* got) {
    LlmEngineMoeRuntime* rt = (LlmEngineMoeRuntime*)ctx;
    if (rt->prof != NULL) {
        llmengine_moe_prof_event(rt->prof, LLMENGINE_MOE_TRACE_LOAD_BEGIN, key.layer, key.expert, 0, 0);
        llmengine_moe_prof_begin(rt->prof, LLMENGINE_MOE_PH_DISK);
    }
    (void)key; (void)buf; (void)cap; (void)got;
    LlmEngineMoeStatus s = LLMENGINE_MOE_ERR_NOT_IMPLEMENTED;   /* esperti letti dal GGUF nell'adapter */
    if (rt->prof != NULL) {
        llmengine_moe_prof_end(rt->prof, LLMENGINE_MOE_PH_DISK);
        llmengine_moe_prof_event(rt->prof, LLMENGINE_MOE_TRACE_LOAD_END, key.layer, key.expert, 0,
                           s == LLMENGINE_MOE_OK ? (uint64_t)*got : 0);
    }
    if (s == LLMENGINE_MOE_OK) rt->bytes_read += *got;   /* metrica: solo i MISS toccano il disco */
    return s;
}

LlmEngineMoeStatus llmengine_moe_get_expert(LlmEngineMoeRuntime* rt, LlmEngineMoeExpertKey key,
                                const void** blob_out, size_t* size_out) {
    (void)rt; (void)key;
    if (blob_out) *blob_out = NULL; if (size_out) *size_out = 0;
    return LLMENGINE_MOE_ERR_NOT_IMPLEMENTED;   /* gli esperti si leggono dal GGUF (llmengine_moe_gguf_expert_f32) */
}

LlmEngineMoeStatus llmengine_moe_prefetch_expert(LlmEngineMoeRuntime* rt, LlmEngineMoeExpertKey key) {
    (void)rt; (void)key;
    return LLMENGINE_MOE_OK;   /* no-op: nessun container (gli esperti GGUF li gestisce l'adapter) */
}

LlmEngineMoeStatus llmengine_moe_notify_routing(LlmEngineMoeRuntime* rt, uint32_t layer,
                                    const uint32_t* experts, uint32_t n) {
    if (rt == NULL) return LLMENGINE_MOE_ERR_INVALID;
    if (rt->prof != NULL)
        llmengine_moe_prof_event(rt->prof, LLMENGINE_MOE_TRACE_ROUTER, layer, n > 0 ? experts[0] : 0, 0, n);
    /* D19: registra la hotness dei routing OSSERVATI (anche senza predictor/prefetch). */
    if (rt->cache != NULL) {
        for (uint32_t i = 0; i < n; ++i) { LlmEngineMoeExpertKey k = { layer, experts[i] }; llmengine_moe_cache_note_route(rt->cache, k, 1); }
    } else if (rt->freq != NULL && layer < rt->freq_layers) {
        for (uint32_t i = 0; i < n; ++i) if (experts[i] < rt->freq_experts) rt->freq[(size_t)layer * rt->freq_experts + experts[i]]++;
    }
    rt->expert_fetches += n;
    if (rt->predictor == NULL) return LLMENGINE_MOE_OK; /* no-op: prefetch speculativo */
    LlmEngineMoeStatus s = llmengine_moe_predictor_notify(rt->predictor, layer, experts, n);
    if (s != LLMENGINE_MOE_OK) return s;
    /* prefetch (async) degli esperti predetti per il layer successivo */
    if (rt->async == NULL) return LLMENGINE_MOE_OK;
    uint32_t pred[64];
    uint32_t np = 0;
    if (llmengine_moe_predictor_predict(rt->predictor, layer + 1, pred, 64, &np) != LLMENGINE_MOE_OK) {
        /* layer+1 fuori range (ultimo): nessun prefetch */
        np = 0;
    }
    for (uint32_t i = 0; i < np; ++i) {
        LlmEngineMoeExpertKey key = {layer + 1, pred[i]};
        (void)llmengine_moe_prefetch_expert(rt, key);
    }
    return LLMENGINE_MOE_OK;
}

LlmEngineMoeStatus llmengine_moe_prefetch_metrics(LlmEngineMoeRuntime* rt, LlmEngineMoePrefetchMetrics* out) {
    if (rt == NULL || out == NULL) return LLMENGINE_MOE_ERR_INVALID;
    if (rt->predictor == NULL) { memset(out, 0, sizeof(*out)); return LLMENGINE_MOE_OK; }
    llmengine_moe_predictor_metrics(rt->predictor, out);
    return LLMENGINE_MOE_OK;
}

void llmengine_moe_reset_prefetch(LlmEngineMoeRuntime* rt, int full) {
    if (rt != NULL && rt->predictor != NULL) llmengine_moe_predictor_reset(rt->predictor, full);
}

uint64_t llmengine_moe_bytes_read(const LlmEngineMoeRuntime* rt) {
    return (rt == NULL) ? 0 : rt->bytes_read;
}

LlmEngineMoeStatus llmengine_moe_forward(LlmEngineMoeRuntime* rt, const float* in, uint32_t seq, float* out) {
    if (rt == NULL || in == NULL || out == NULL) return LLMENGINE_MOE_ERR_INVALID;
    if (!rt->loaded) return LLMENGINE_MOE_ERR_INVALID;
    const LlmEngineMoeModelAdapter* a = &rt->adapters[rt->active];
    if (a->forward == NULL) return LLMENGINE_MOE_ERR_ADAPTER;
    if (rt->prof != NULL) llmengine_moe_prof_begin(rt->prof, LLMENGINE_MOE_PH_COMPUTE);
    LlmEngineMoeStatus fs = a->forward(a->self, rt, in, seq, rt->meta.hidden_size, out);
    if (rt->prof != NULL) llmengine_moe_prof_end(rt->prof, LLMENGINE_MOE_PH_COMPUTE);
    if (fs == LLMENGINE_MOE_OK) llmengine_moe_adaptive_step(rt);   /* M17: adatta prestazioni (opt-in) */
    return fs;
}

int llmengine_moe_generate(LlmEngineMoeRuntime* rt, const int32_t* prompt, uint32_t n_prompt,
                     int32_t* out, uint32_t n_out) {
    if (rt == NULL || out == NULL || n_out == 0) return LLMENGINE_MOE_ERR_INVALID;
    if (!rt->loaded) return LLMENGINE_MOE_ERR_INVALID;
    if (rt->meta.hidden_size == 0) return LLMENGINE_MOE_ERR_ADAPTER;

    /* Stato di lavoro: l'ultimo hidden [1 x h]. Il prompt inizializza lo stato in modo
     * deterministico (l'adapter reale fara' embedding/lm_head; qui la scelta e' argmax). */
    uint32_t h = rt->meta.hidden_size;
    float* state = (float*)calloc(h, sizeof(float));
    if (state == NULL) return LLMENGINE_MOE_ERR_NO_MEMORY;

    if (prompt != NULL && n_prompt > 0) {
        for (uint32_t t = 0; t < n_prompt; ++t) {
            float v = (float)(prompt[t] % 1000) / 1000.0f;
            for (uint32_t j = 0; j < h; ++j) state[j] += v;
        }
        for (uint32_t j = 0; j < h; ++j) state[j] /= (float)n_prompt;
    }

    LlmEngineMoeStatus s = llmengine_moe_forward(rt, state, 1u, state);
    if (s != LLMENGINE_MOE_OK) { free(state); return s; }

    uint32_t produced = 0;
    for (uint32_t i = 0; i < n_out; ++i) {
        uint32_t best = 0;
        float bestv = state[0];
        for (uint32_t j = 1; j < h; ++j) {
            if (state[j] > bestv) { bestv = state[j]; best = j; }
        }
        out[i] = (int32_t)best;
        produced++;
        s = llmengine_moe_forward(rt, state, 1u, state); /* in==out: aggiornamento in place */
        if (s != LLMENGINE_MOE_OK) break;
    }
    free(state);
    return (int)produced;
}

LlmEngineMoeStatus llmengine_moe_set_memory_budget(LlmEngineMoeRuntime* rt, uint64_t total_bytes,
                                       double warn_frac, double crit_frac) {
    if (rt == NULL) return LLMENGINE_MOE_ERR_INVALID;
    if (total_bytes == 0) { llmengine_moe_mem_destroy(rt->mem); rt->mem = NULL; return LLMENGINE_MOE_OK; }
    LlmEngineMoeMemMgr* m = NULL;
    LlmEngineMoeStatus st = llmengine_moe_mem_create(&m, total_bytes, warn_frac, crit_frac);
    if (st != LLMENGINE_MOE_OK) return st;
    llmengine_moe_mem_destroy(rt->mem);
    rt->mem = m;
    return LLMENGINE_MOE_OK;
}

const LlmEngineMoeMemMgr* llmengine_moe_mem(const LlmEngineMoeRuntime* rt) {
    return rt == NULL ? NULL : rt->mem;
}

LlmEngineMoeStatus llmengine_moe_set_cache_budgets(LlmEngineMoeRuntime* rt, uint64_t expert_bytes, uint64_t kv_bytes) {
    if (rt == NULL) return LLMENGINE_MOE_ERR_INVALID;
    LlmEngineMoeExpertCache* ec = NULL;
    if (expert_bytes > 0) {
        LlmEngineMoeStatus st = llmengine_moe_cache_create(&ec, expert_bytes);
        if (st != LLMENGINE_MOE_OK) return st;
    }
    LlmEngineMoeKvCache* kv = NULL;
    if (kv_bytes > 0) {
        LlmEngineMoeStatus st = llmengine_moe_kv_create(&kv, kv_bytes);
        if (st != LLMENGINE_MOE_OK) { llmengine_moe_cache_destroy(ec); return st; }
    }
    llmengine_moe_cache_destroy(rt->cache);
    llmengine_moe_kv_destroy(rt->kv);
    rt->cache = ec;
    rt->kv = kv;
    if (ec != NULL) moe_cache_configure(rt, ec, rt->hotness);
    return LLMENGINE_MOE_OK;
}

const LlmEngineMoeExpertCache* llmengine_moe_expert_cache(const LlmEngineMoeRuntime* rt) {
    return rt == NULL ? NULL : rt->cache;
}

const LlmEngineMoeKvCache* llmengine_moe_kv(const LlmEngineMoeRuntime* rt) {
    return rt == NULL ? NULL : rt->kv;
}

void llmengine_moe_set_profiler(LlmEngineMoeRuntime* rt, LlmEngineMoeProfiler* prof) {
    if (rt != NULL) rt->prof = prof;
}

LlmEngineMoeProfiler* llmengine_moe_profiler(const LlmEngineMoeRuntime* rt) {
    return rt == NULL ? NULL : rt->prof;
}

/* RESIDENT (M14): precarica tutti gli esperti nella cache -> nessuno swap dopo il load.
 * Le chiavi assenti (container sparso) vengono ignorate. */
static LlmEngineMoeStatus llmengine_moe_preload_resident(LlmEngineMoeRuntime* rt) {
    if (rt->cache == NULL) return LLMENGINE_MOE_OK;
    uint32_t L = rt->meta.layer_count, E = rt->meta.expert_count;
    const void* blob = NULL; size_t bs = 0;
    for (uint32_t l = 0; l < L; ++l)
        for (uint32_t e = 0; e < E; ++e) {
            LlmEngineMoeExpertKey key = { l, e };
            (void)llmengine_moe_get_expert(rt, key, &blob, &bs);
        }
    return LLMENGINE_MOE_OK;
}

LlmEngineMoeStatus llmengine_moe_stream_stats(const LlmEngineMoeRuntime* rt, LlmEngineMoeStreamStats* out) {
    if (rt == NULL || out == NULL) return LLMENGINE_MOE_ERR_INVALID;
    memset(out, 0, sizeof(*out));
    out->loaded = rt->loaded;
    out->mode = rt->mode;
    out->device = rt->device;
    out->layers = rt->meta.layer_count;
    out->experts = rt->meta.expert_count;
    out->bytes_read = rt->bytes_read;
    out->expert_bytes = rt->meta.routed_bytes;
    if (rt->cache != NULL) {
        LlmEngineMoeCacheStats cs;
        llmengine_moe_cache_stats(rt->cache, &cs);
        out->cache_hits = cs.hits;
        out->cache_misses = cs.misses;
        out->cache_evictions = cs.evictions;
        out->cache_resident_hits = cs.resident_hits;   /* K2 */
        out->resident_bytes = llmengine_moe_cache_bytes_used(rt->cache);
    } else if (rt->ext_valid) {
        /* Adapter con store proprio (GGUF stile DS4): metriche pubblicate dall'adapter. */
        out->bytes_read = rt->ext.bytes_read;
        out->cache_hits = rt->ext.cache_hits;
        out->cache_misses = rt->ext.cache_misses;
        out->cache_evictions = rt->ext.cache_evictions;
        out->resident_bytes = rt->ext.resident_bytes;
    }
    if (rt->predictor != NULL) {
        LlmEngineMoePrefetchMetrics pm;
        llmengine_moe_predictor_metrics(rt->predictor, &pm);
        out->prefetch_predictions = pm.predictions;
        out->prefetch_hits = pm.hits;
        out->prefetch_wasted = pm.wasted;
    }
    return LLMENGINE_MOE_OK;
}

void llmengine_moe_set_stream_metrics(LlmEngineMoeRuntime* rt, uint64_t bytes_read, uint64_t hits,
                                uint64_t misses, uint64_t evictions, uint64_t resident_bytes,
                                uint64_t budget_bytes) {
    if (rt == NULL) return;
    memset(&rt->ext, 0, sizeof(rt->ext));
    rt->ext.bytes_read = bytes_read;
    rt->ext.cache_hits = hits;
    rt->ext.cache_misses = misses;
    rt->ext.cache_evictions = evictions;
    rt->ext.resident_bytes = resident_bytes;
    rt->ext.expert_bytes = budget_bytes;   /* budget dello store (per AI.STREAMSTATS) */
    rt->ext_valid = 1;
}

LlmEngineMoeStatus llmengine_moe_set_device(LlmEngineMoeRuntime* rt, LlmEngineMoeDevice device) {
    if (rt == NULL) return LLMENGINE_MOE_ERR_INVALID;
#ifdef __APPLE__
    if (device == LLMENGINE_MOE_DEVICE_METAL && !llmengine_moe_metal_available()) return LLMENGINE_MOE_ERR_DEVICE;
#else
    if (device == LLMENGINE_MOE_DEVICE_METAL) return LLMENGINE_MOE_ERR_DEVICE;
#endif
    rt->device = device;
    return LLMENGINE_MOE_OK;
}

LlmEngineMoeStatus llmengine_moe_set_cache_mb(LlmEngineMoeRuntime* rt, uint64_t cache_mb) {
    if (rt == NULL) return LLMENGINE_MOE_ERR_INVALID;
    if (rt->mode == LLMENGINE_MOE_MODE_RESIDENT) return LLMENGINE_MOE_ERR_INVALID; /* residente: fissa */
    rt->cfg_cache_mb = cache_mb;   /* l'adapter GGUF riapre lo store col nuovo budget al forward */
    LlmEngineMoeExpertCache* nc = NULL;
    if (cache_mb > 0 && llmengine_moe_cache_create(&nc, cache_mb * 1024ull * 1024ull) != LLMENGINE_MOE_OK)
        return LLMENGINE_MOE_ERR_NO_MEMORY;
    /* l'async tiene un puntatore alla cache: va ricreato dopo. */
    if (rt->async != NULL) { llmengine_moe_async_destroy(rt->async); rt->async = NULL; }
    llmengine_moe_cache_destroy(rt->cache);
    rt->cache = nc;
    if (nc != NULL) moe_cache_configure(rt, nc, rt->hotness);
    return LLMENGINE_MOE_OK;
}

/* ── M21: gerarchia di residenza (DISK/COLD/WARM/ACTIVE) + transizioni ── */
LlmEngineMoeResidency llmengine_moe_residency(const LlmEngineMoeRuntime* rt) {
    if (rt == NULL || !rt->loaded) return LLMENGINE_MOE_RESIDENCY_DISK;
    return rt->mode == LLMENGINE_MOE_MODE_RESIDENT ? LLMENGINE_MOE_RESIDENCY_ACTIVE : LLMENGINE_MOE_RESIDENCY_WARM;
}

LlmEngineMoeStatus llmengine_moe_promote(LlmEngineMoeRuntime* rt) {
    if (rt == NULL) return LLMENGINE_MOE_ERR_INVALID;
    if (rt->mode == LLMENGINE_MOE_MODE_RESIDENT) return LLMENGINE_MOE_OK;   /* gia' ACTIVE */
    uint64_t bytes = rt->meta.routed_bytes;
    uint64_t want_mb = (bytes + (1u << 20) - 1) >> 20;
    if (want_mb == 0) want_mb = 1;
    if (rt->async != NULL) { llmengine_moe_async_destroy(rt->async); rt->async = NULL; }  /* niente swap */
    LlmEngineMoeExpertCache* nc = NULL;
    if (llmengine_moe_cache_create(&nc, want_mb * 1024ull * 1024ull) != LLMENGINE_MOE_OK) return LLMENGINE_MOE_ERR_NO_MEMORY;
    llmengine_moe_cache_destroy(rt->cache);
    rt->cache = nc;
    rt->mode = LLMENGINE_MOE_MODE_RESIDENT;
    return llmengine_moe_preload_resident(rt);   /* porta tutti gli esperti in RAM (ACTIVE) */
}

LlmEngineMoeStatus llmengine_moe_demote(LlmEngineMoeRuntime* rt) {
    if (rt == NULL) return LLMENGINE_MOE_ERR_INVALID;
    if (rt->mode != LLMENGINE_MOE_MODE_RESIDENT) return LLMENGINE_MOE_OK;   /* gia' WARM */
    rt->mode = LLMENGINE_MOE_MODE_SWAP;
    llmengine_moe_cache_destroy(rt->cache);   /* rinuncia ai pesi residenti: torna on-demand */
    rt->cache = NULL;
    return LLMENGINE_MOE_OK;
}

LlmEngineMoeStatus llmengine_moe_set_async(LlmEngineMoeRuntime* rt, uint32_t workers, uint32_t cap) {
    if (rt == NULL) return LLMENGINE_MOE_ERR_INVALID;
    if (rt->async != NULL) { llmengine_moe_async_destroy(rt->async); rt->async = NULL; }
    if (workers == 0) return LLMENGINE_MOE_OK;
    if (rt->cache == NULL) return LLMENGINE_MOE_ERR_INVALID; /* l'async richiede la cache */
    LlmEngineMoeAsync* a = NULL;
    if (llmengine_moe_async_create(&a, workers, llmengine_moe_rt_load_expert, rt, rt->cache, cap) != LLMENGINE_MOE_OK)
        return LLMENGINE_MOE_ERR_NO_MEMORY;
    llmengine_moe_async_set_profiler(a, rt->prof);   /* K1: trace richieste */
    rt->async = a;
    return LLMENGINE_MOE_OK;
}

LlmEngineMoeStatus llmengine_moe_set_prefetch(LlmEngineMoeRuntime* rt, uint32_t topn, uint32_t coupling_k) {
    if (rt == NULL) return LLMENGINE_MOE_ERR_INVALID;
    llmengine_moe_predictor_destroy(rt->predictor);
    rt->predictor = NULL;
    if (topn == 0) return LLMENGINE_MOE_OK;
    if (rt->meta.layer_count == 0 || rt->meta.expert_count == 0) return LLMENGINE_MOE_ERR_INVALID;
    LlmEngineMoePredictor* p = NULL;
    if (llmengine_moe_predictor_create(&p, rt->meta.layer_count, rt->meta.expert_count, topn,
                                 coupling_k == 0 ? 4u : coupling_k) != LLMENGINE_MOE_OK)
        return LLMENGINE_MOE_ERR_NO_MEMORY;
    rt->predictor = p;
    return LLMENGINE_MOE_OK;
}

LlmEngineMoeLoadMode llmengine_moe_mode(const LlmEngineMoeRuntime* rt) {
    return rt == NULL ? LLMENGINE_MOE_MODE_SWAP : rt->mode;
}

uint64_t llmengine_moe_cache_mb(const LlmEngineMoeRuntime* rt) {
    return rt == NULL ? 0 : rt->cfg_cache_mb;
}

LlmEngineMoeStatus llmengine_moe_get_tensor(const LlmEngineMoeRuntime* rt, const char* name, void* buf, size_t cap, size_t* got) {
    (void)rt; (void)name; (void)buf; (void)cap; (void)got;
    return LLMENGINE_MOE_ERR_NOT_IMPLEMENTED;   /* dense letti dal GGUF nell'adapter */
}

/* ── M18: statistiche d'uso per-esperto, persistite in `<container>.stats` — ADVISORY ──
 * Un file mancante o corrotto NON blocca il load: il chiamante ignora l'errore. */
uint64_t llmengine_moe_expert_hits(const LlmEngineMoeRuntime* rt, LlmEngineMoeExpertKey key) {
    if (rt == NULL) return 0;
    if (rt->cache != NULL) return llmengine_moe_cache_freq(rt->cache, key);   /* D19: tabella della cache */
    if (rt->freq == NULL || key.layer >= rt->freq_layers || key.expert >= rt->freq_experts) return 0;
    return rt->freq[(size_t)key.layer * rt->freq_experts + key.expert];
}

/* D19: attiva/disattiva la expert cache adattiva. */
LlmEngineMoeStatus llmengine_moe_set_hotness(LlmEngineMoeRuntime* rt, int on) {
    if (rt == NULL) return LLMENGINE_MOE_ERR_INVALID;
    rt->hotness = on ? 1 : 0;
    if (rt->cache != NULL) {
        (void)llmengine_moe_cache_set_dims(rt->cache, rt->meta.layer_count, rt->meta.expert_count);
        llmengine_moe_cache_set_hotness(rt->cache, rt->hotness);
    }
    return LLMENGINE_MOE_OK;
}

/* M20: fetch totali di esperti (hit+miss) in questa sessione. */
uint64_t llmengine_moe_expert_fetches(const LlmEngineMoeRuntime* rt) {
    if (rt == NULL) return 0;
    if (rt->cache != NULL) return llmengine_moe_cache_fetches(rt->cache);   /* D19 */
    return rt->expert_fetches;
}

LlmEngineMoeStatus llmengine_moe_stats_save(const LlmEngineMoeRuntime* rt, const char* path) {
    if (rt == NULL) return LLMENGINE_MOE_ERR_INVALID;
    if (path == NULL || *path == 0) path = rt->stats_path;
    if (rt->cache != NULL) return llmengine_moe_cache_profile_save(rt->cache, path);   /* D19 */
    if (rt->freq == NULL) return LLMENGINE_MOE_ERR_INVALID;
    if (path == NULL || *path == 0) return LLMENGINE_MOE_ERR_INVALID;
    FILE* f = fopen(path, "wb");
    if (f == NULL) return LLMENGINE_MOE_ERR_IO;
    if (fprintf(f, "GMSTATS 1\n") < 0) { fclose(f); return LLMENGINE_MOE_ERR_IO; }
    for (uint32_t l = 0; l < rt->freq_layers; ++l)
        for (uint32_t e = 0; e < rt->freq_experts; ++e) {
            uint32_t v = rt->freq[(size_t)l * rt->freq_experts + e];
            if (v != 0) fprintf(f, "%u %u %u\n", l, e, v);
        }
    fclose(f);
    return LLMENGINE_MOE_OK;
}

LlmEngineMoeStatus llmengine_moe_stats_load(LlmEngineMoeRuntime* rt, const char* path) {
    if (rt == NULL) return LLMENGINE_MOE_ERR_INVALID;
    if (path == NULL || *path == 0) path = rt->stats_path;
    if (rt->cache != NULL) return llmengine_moe_cache_profile_load(rt->cache, path);   /* D19 */
    if (rt->freq == NULL) return LLMENGINE_MOE_ERR_INVALID;
    if (path == NULL || *path == 0) return LLMENGINE_MOE_ERR_INVALID;
    FILE* f = fopen(path, "rb");
    if (f == NULL) return LLMENGINE_MOE_ERR_IO;
    char magic[16] = {0};
    if (fscanf(f, "%15s", magic) != 1 || strcmp(magic, "GMSTATS") != 0) { fclose(f); return LLMENGINE_MOE_ERR_FORMAT; }
    int ver = 0;
    if (fscanf(f, "%d", &ver) != 1 || ver != 1) { fclose(f); return LLMENGINE_MOE_ERR_FORMAT; }
    unsigned l = 0, e = 0, v = 0; int nread = 0;
    while (fscanf(f, "%u %u %u", &l, &e, &v) == 3) {
        if (l < rt->freq_layers && e < rt->freq_experts)
            rt->freq[(size_t)l * rt->freq_experts + e] += v;   /* accumula */
        nread++;
    }
    fclose(f);
    return nread > 0 ? LLMENGINE_MOE_OK : LLMENGINE_MOE_ERR_FORMAT;
}

/* ── M17: controller adattivo (prefetch depth + cache budget) — opt-in LLMENGINE_MOE_ADAPTIVE=1 ──
 * Risultato INVARIATO (solo policy di I/O e memoria). Prefetch: recall alta → più profondità,
 * bassa → meno, pressione CRITICA → minimo. Cache: sotto CRITICA ridotta, a NORMAL ripristinata. */
int llmengine_moe_adaptive_enabled(const LlmEngineMoeRuntime* rt) { return rt != NULL && rt->adaptive; }
uint32_t llmengine_moe_adaptive_topn(const LlmEngineMoeRuntime* rt) { return rt == NULL ? 0 : rt->adapt_topn; }

static void llmengine_moe_adaptive_step(LlmEngineMoeRuntime* rt) {
    if (rt == NULL || !rt->adaptive) return;
    if (++rt->adapt_steps < rt->adapt_interval) return;
    rt->adapt_steps = 0;
    LlmEngineMoePressure pr = rt->mem != NULL ? llmengine_moe_mem_pressure(rt->mem) : LLMENGINE_MOE_PRESS_NORMAL;

    if (rt->predictor != NULL) {
        LlmEngineMoePrefetchMetrics m; memset(&m, 0, sizeof(m));
        llmengine_moe_predictor_metrics(rt->predictor, &m);
        uint64_t dh = m.hits - rt->adapt_last_hits, da = m.actual_total - rt->adapt_last_actual;
        rt->adapt_last_pred = m.predictions; rt->adapt_last_hits = m.hits; rt->adapt_last_actual = m.actual_total;
        double recall = da > 0 ? (double)dh / (double)da : -1.0;
        uint32_t t = rt->adapt_topn;
        if (pr == LLMENGINE_MOE_PRESS_CRITICAL) t = rt->adapt_pf_min;
        else if (recall >= 0.5) { if (t < rt->adapt_pf_max) t++; }
        else if (recall >= 0.0 && recall <= 0.2) { if (t > rt->adapt_pf_min) t--; }
        if (t != rt->adapt_topn && llmengine_moe_predictor_set_topn(rt->predictor, t) == LLMENGINE_MOE_OK)
            rt->adapt_topn = t;
    }

    if (rt->mode != LLMENGINE_MOE_MODE_RESIDENT && rt->adapt_cache_mb > 0) {
        if (pr == LLMENGINE_MOE_PRESS_CRITICAL && rt->adapt_last_press != LLMENGINE_MOE_PRESS_CRITICAL
            && rt->adapt_cache_min < rt->adapt_cache_mb)
            (void)llmengine_moe_set_cache_mb(rt, rt->adapt_cache_min);
        else if (pr == LLMENGINE_MOE_PRESS_NORMAL && rt->adapt_last_press != LLMENGINE_MOE_PRESS_NORMAL)
            (void)llmengine_moe_set_cache_mb(rt, rt->adapt_cache_mb);
    }
    rt->adapt_last_press = pr;
}
