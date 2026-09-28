/* llmengine_moe_discover.c — implementazione dell'auto-discovery (vedi include/llmengine_moe_discover.h). */
#include "llmengine_moe_discover.h"
#include "llmengine_moe_json.h"
#include "llmengine_moe_gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* ── iterazione chiavi di un oggetto JSON (solo livello 0) ── */
typedef void (*LlmKeyCb)(const char* name, void* ctx);

static int disc_is_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

static void foreach_top_key(const char* obj, LlmKeyCb cb, void* ctx) {
    if (obj == NULL || *obj != '{') return;
    const char* p = obj + 1;
    int depth = 0;
    while (*p) {
        char c = *p;
        if (c == '"') {
            const char* start = ++p;
            while (*p && *p != '"') ++p;
            size_t len = (size_t)(p - start);
            if (*p == '"') ++p;
            const char* q = p; while (disc_is_ws(*q)) ++q;
            if (depth == 0 && *q == ':') {
                char name[256];
                size_t n = len < sizeof(name) - 1 ? len : sizeof(name) - 1;
                memcpy(name, start, n); name[n] = 0;
                cb(name, ctx);
            }
            continue;
        }
        if (c == '{') { depth++; ++p; continue; }
        if (c == '}') { if (depth == 0) break; depth--; ++p; continue; }
        if (c == '[') {            /* salta l'array (shape) */
            int ad = 1; ++p;
            while (*p && ad > 0) { if (*p == '[') ad++; else if (*p == ']') ad--; ++p; }
            continue;
        }
        ++p;
    }
}

/* ── raccolta nomi tensori: index safetensors, oppure header di model.safetensors ── */
typedef struct {
    LlmEngineMoeDiscover* d;
    uint32_t max_expert, max_layer;
    int has_shared;                            /* nomi `shared_experts` (DeepSeek/GLM HF) */
} NameCtx;

static void name_cb(const char* name, void* vctx) {
    NameCtx* nc = (NameCtx*)vctx;
    LlmEngineMoeDiscover* d = nc->d;
    if (strcmp(name, "__metadata__") == 0) return;
    d->n_tensors++;
    int role = llmengine_moe_role_of_tensor(name);
    if (role > 0 && role < LLMENGINE_MOE_ROLE_COUNT) d->role_counts[role]++;

    /* layer ed expert impliciti nel nome: "layers.<N>." / "blk.<N>." / "experts.<N>." */
    const char* l = strstr(name, "layers.");
    if (l == NULL) l = strstr(name, "blk.");
    if (l != NULL) {
        const char* q = strchr(l, '.');
        if (q != NULL) {
            uint32_t v = (uint32_t)strtoul(q + 1, NULL, 10);
            if (v + 1 > nc->max_layer) nc->max_layer = v + 1;
        }
    }
    const char* e = strstr(name, "experts.");
    if (e != NULL) {
        const char* q = strchr(e, '.');
        if (q != NULL) {
            uint32_t v = (uint32_t)strtoul(q + 1, NULL, 10);
            if (v + 1 > nc->max_expert) nc->max_expert = v + 1;
        }
    }
    if (strstr(name, "shared_experts") != NULL) nc->has_shared = 1;
}

static int collect_names_from_index(const char* dir, NameCtx* nc) {
    char path[1200];
    snprintf(path, sizeof(path), "%s/model.safetensors.index.json", dir);
    char* js = llmengine_moe_slurp(path, NULL);
    if (js == NULL) return 0;
    const char* wm = llmengine_moe_json_find(js, "weight_map");
    if (wm != NULL && *wm == '{') foreach_top_key(wm, name_cb, nc);
    free(js);
    return 1;
}

static int collect_names_from_safetensors(const char* dir, NameCtx* nc) {
    char path[1200];
    snprintf(path, sizeof(path), "%s/model.safetensors", dir);
    FILE* f = fopen(path, "rb");
    if (f == NULL) return 0;
    unsigned char hdr[8];
    if (fread(hdr, 1, 8, f) != 8) { fclose(f); return 0; }
    uint64_t n = 0;
    for (int i = 7; i >= 0; --i) n = (n << 8) | hdr[i];
    if (n == 0 || n > (uint64_t)64 * 1024 * 1024) { fclose(f); return 0; }
    char* js = (char*)malloc((size_t)n + 1);
    if (js == NULL) { fclose(f); return 0; }
    size_t got = fread(js, 1, (size_t)n, f);
    fclose(f);
    js[got] = 0;
    foreach_top_key(js, name_cb, nc);
    free(js);
    return 1;
}

/* ── euristiche famiglia/varianti ── */
static uint32_t family_from_model_type(const char* mt) {
    if (strstr(mt, "qwen3_moe"))  return LLMENGINE_MOE_ARCH_QWEN3_MOE;
    if (strstr(mt, "qwen3_next")) return LLMENGINE_MOE_ARCH_QWEN3_NEXT;
    if (strstr(mt, "qwen2_moe"))  return LLMENGINE_MOE_ARCH_QWEN2_MOE;
    if (strstr(mt, "glm4_moe") || strstr(mt, "glm4moe")) return LLMENGINE_MOE_ARCH_GLM_MOE;
    if (strstr(mt, "deepseek"))   return LLMENGINE_MOE_ARCH_DEEPSEEK;
    if (strstr(mt, "gemma"))      return LLMENGINE_MOE_ARCH_GEMMA;
    if (strstr(mt, "llama"))      return LLMENGINE_MOE_ARCH_LLAMA;
    return LLMENGINE_MOE_ARCH_UNKNOWN;
}

LlmEngineMoeStatus llmengine_moe_discover_hf(const char* dir, LlmEngineMoeDiscover* out) {
    if (dir == NULL || out == NULL) return LLMENGINE_MOE_ERR_INVALID;
    memset(out, 0, sizeof(*out));
    LlmEngineMoeArchDesc* d = &out->desc;

    char path[1200];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    char* cfg = llmengine_moe_slurp(path, NULL);
    if (cfg == NULL) return LLMENGINE_MOE_ERR_IO;

    d->hidden   = (uint32_t)llmengine_moe_json_num(cfg, "hidden_size", 0);
    d->n_layers = (uint32_t)llmengine_moe_json_num(cfg, "num_hidden_layers", 0);
    d->n_heads  = (uint32_t)llmengine_moe_json_num(cfg, "num_attention_heads", 0);
    d->n_kv_heads = (uint32_t)llmengine_moe_json_num(cfg, "num_key_value_heads", d->n_heads);
    d->head_dim = (uint32_t)llmengine_moe_json_num(cfg, "head_dim", d->n_heads ? d->hidden / d->n_heads : 0);
    d->vocab    = (uint32_t)llmengine_moe_json_num(cfg, "vocab_size", 0);
    d->n_experts = (uint32_t)llmengine_moe_json_num(cfg, "num_experts",
                    llmengine_moe_json_num(cfg, "num_local_experts",
                    llmengine_moe_json_num(cfg, "n_routed_experts", 0)));
    d->topk     = (uint32_t)llmengine_moe_json_num(cfg, "num_experts_per_tok", d->n_experts ? 1 : 0);
    d->inter    = (uint32_t)llmengine_moe_json_num(cfg, "moe_intermediate_size",
                    llmengine_moe_json_num(cfg, "intermediate_size", 0));
    d->ctx      = (uint32_t)llmengine_moe_json_num(cfg, "max_position_embeddings", 0);
    int has_layer_norm_eps = llmengine_moe_json_find(cfg, "layer_norm_eps") != NULL;
    d->eps      = (float)llmengine_moe_json_flt(cfg, "rms_norm_eps",
                    llmengine_moe_json_flt(cfg, "layer_norm_eps", 1e-6));
    d->rope_theta = (float)llmengine_moe_json_flt(cfg, "rope_theta", 10000000.0);
    int tie   = llmengine_moe_json_flag(cfg, "tie_word_embeddings", 0);
    int bias  = llmengine_moe_json_flag(cfg, "attention_bias", 0);
    int interleaved = llmengine_moe_json_flag(cfg, "rope_interleaved", 0);
    llmengine_moe_json_str(llmengine_moe_json_find(cfg, "model_type"), out->model_type, sizeof(out->model_type));
    char scoring[32] = {0};
    llmengine_moe_json_str(llmengine_moe_json_find(cfg, "scoring_func"), scoring, sizeof(scoring));
    free(cfg);

    /* parametri MLA/MoE-struttura (DeepSeek-V2/V3 e simili). */
    {
        char cpath2[1200];
        snprintf(cpath2, sizeof(cpath2), "%s/config.json", dir);
        char* cj2 = llmengine_moe_slurp(cpath2, NULL);
        if (cj2 != NULL) {
            out->mla_kv_lora = (uint32_t)llmengine_moe_json_num(cj2, "kv_lora_rank", 0);
            out->mla_qk_nope = (uint32_t)llmengine_moe_json_num(cj2, "qk_nope_head_dim", 0);
            out->mla_qk_rope = (uint32_t)llmengine_moe_json_num(cj2, "qk_rope_head_dim", 0);
            out->mla_v_head  = (uint32_t)llmengine_moe_json_num(cj2, "v_head_dim", 0);
            out->mla_q_lora  = (uint32_t)llmengine_moe_json_num(cj2, "q_lora_rank", 0);
            out->first_k_dense = (uint32_t)llmengine_moe_json_num(cj2, "first_k_dense_replace", 0);
            out->dense_inter = (uint32_t)llmengine_moe_json_num(cj2, "intermediate_size", 0);
            out->n_shared_experts = (uint32_t)llmengine_moe_json_num(cj2, "n_shared_experts", 0);
            out->yarn_factor = (float)llmengine_moe_json_flt(cj2, "factor", 0.0);
            out->yarn_orig_max = (uint32_t)llmengine_moe_json_num(cj2, "original_max_position_embeddings", 4096);
            out->yarn_beta_fast = (float)llmengine_moe_json_flt(cj2, "beta_fast", 32.0);
            out->yarn_beta_slow = (float)llmengine_moe_json_flt(cj2, "beta_slow", 1.0);
            out->yarn_mscale = (float)llmengine_moe_json_flt(cj2, "mscale", 1.0);
            out->yarn_mscale_all_dim = (float)llmengine_moe_json_flt(cj2, "mscale_all_dim", 0.0);
            free(cj2);
        }
    }

    /* nome del campo modello */
    d->arch_family = family_from_model_type(out->model_type);

    /* nomi tensori */
    NameCtx nc = { out, 0, 0, 0 };
    if (!collect_names_from_index(dir, &nc)) (void)collect_names_from_safetensors(dir, &nc);
    out->max_layer_seen = nc.max_layer;
    out->experts_per_layer = nc.max_expert ? nc.max_expert : d->n_experts;
    if (d->n_layers == 0 && nc.max_layer > 0) d->n_layers = nc.max_layer;

    /* varianti derivate dai nomi */
    if (out->role_counts[LLMENGINE_MOE_ROLE_Q_NORM] || out->role_counts[LLMENGINE_MOE_ROLE_K_NORM])
        d->flags |= LLMENGINE_MOE_FLAG_QK_NORM;
    if (out->role_counts[LLMENGINE_MOE_ROLE_EXPERT_GATE] || out->role_counts[LLMENGINE_MOE_ROLE_ROUTER])
        d->flags |= LLMENGINE_MOE_FLAG_MOE;
    if (bias) d->flags |= LLMENGINE_MOE_FLAG_BIAS;
    if (tie)  d->flags |= LLMENGINE_MOE_FLAG_TIED;
    /* MLA (DeepSeek-V2/V3): kv_lora_rank>0 → MLA. shared-expert da `shared_experts`/famiglia. */
    if (d->arch_family == LLMENGINE_MOE_ARCH_DEEPSEEK || out->mla_kv_lora > 0) d->flags |= LLMENGINE_MOE_FLAG_MLA;
    if (nc.has_shared || d->arch_family == LLMENGINE_MOE_ARCH_GLM_MOE || d->arch_family == LLMENGINE_MOE_ARCH_QWEN2_MOE)
        d->flags |= LLMENGINE_MOE_FLAG_SHARED;

    d->rope_variant = (out->mla_kv_lora > 0) ? LLMENGINE_MOE_ROPE_INTERLEAVED
                     : (interleaved ? LLMENGINE_MOE_ROPE_INTERLEAVED : LLMENGINE_MOE_ROPE_HALF);
    /* router: da `scoring_func` se presente; altrimenti sigmoid solo per GLM, softmax altrove. */
    if (scoring[0] != 0)
        d->router_kind = (strcmp(scoring, "sigmoid") == 0) ? LLMENGINE_MOE_ROUTER_SIGMOID : LLMENGINE_MOE_ROUTER_SOFTMAX;
    else
        d->router_kind = (d->arch_family == LLMENGINE_MOE_ARCH_GLM_MOE) ? LLMENGINE_MOE_ROUTER_SIGMOID : LLMENGINE_MOE_ROUTER_SOFTMAX;
    d->norm_kind = (has_layer_norm_eps || d->arch_family == LLMENGINE_MOE_ARCH_GEMMA)
                   ? LLMENGINE_MOE_NORM_LAYER : LLMENGINE_MOE_NORM_RMS;

    /* confidenza: dimensione + famiglia + coerenza */
    int conf = 40;
    if (d->hidden && d->n_heads && d->head_dim) conf += 15;
    if (d->n_layers) conf += 10;
    if (d->vocab) conf += 10;
    if (d->arch_family != LLMENGINE_MOE_ARCH_UNKNOWN) conf += 15;
    if (d->n_heads && d->head_dim && d->hidden == d->n_heads * d->head_dim) conf += 10;
    if (out->n_tensors == 0) conf -= 30;
    if (conf > 100) conf = 100; if (conf < 0) conf = 0;
    out->confidence = conf;
    out->source_gguf = 0;
    return LLMENGINE_MOE_OK;
}

static double gguf_arch_num(const LlmEngineMoeGguf* g, const char* arch, const char* field, double def) {
    char k[160]; snprintf(k, sizeof(k), "%s.%s", arch, field);
    return llmengine_moe_gguf_num(g, k, def);
}

static uint32_t family_from_arch_str(const char* a) {
    if (strstr(a, "qwen3moe") || strstr(a, "qwen3_moe")) return LLMENGINE_MOE_ARCH_QWEN3_MOE;
    if (strstr(a, "qwen3next") || strstr(a, "qwen3_next")) return LLMENGINE_MOE_ARCH_QWEN3_NEXT;
    if (strstr(a, "qwen2moe") || strstr(a, "qwen2_moe")) return LLMENGINE_MOE_ARCH_QWEN2_MOE;
    if (strstr(a, "glm4")) return LLMENGINE_MOE_ARCH_GLM_MOE;
    if (strstr(a, "deepseek")) return LLMENGINE_MOE_ARCH_DEEPSEEK;
    if (strstr(a, "gemma")) return LLMENGINE_MOE_ARCH_GEMMA;
    if (strstr(a, "llama")) return LLMENGINE_MOE_ARCH_LLAMA;
    return LLMENGINE_MOE_ARCH_UNKNOWN;
}

LlmEngineMoeStatus llmengine_moe_discover_gguf(const char* path, LlmEngineMoeDiscover* out) {
    if (path == NULL || out == NULL) return LLMENGINE_MOE_ERR_INVALID;
    memset(out, 0, sizeof(*out));
    LlmEngineMoeGguf g;
    LlmEngineMoeStatus st = llmengine_moe_gguf_read(path, &g);
    if (st != LLMENGINE_MOE_OK) return st;
    LlmEngineMoeArchDesc* d = &out->desc;

    const LlmEngineMoeGgufKv* akv = llmengine_moe_gguf_find(&g, "general.architecture");
    const char* arch = (akv != NULL && akv->type == LLMCODEC_GGUF_STRING) ? akv->str : "";
    snprintf(out->model_type, sizeof(out->model_type), "%s", arch);
    d->arch_family = family_from_arch_str(arch);

    d->n_layers   = (uint32_t)gguf_arch_num(&g, arch, "block_count", 0);
    d->hidden     = (uint32_t)gguf_arch_num(&g, arch, "embedding_length", 0);
    d->n_heads    = (uint32_t)gguf_arch_num(&g, arch, "attention.head_count", 0);
    d->n_kv_heads = (uint32_t)gguf_arch_num(&g, arch, "attention.head_count_kv", d->n_heads);
    d->head_dim   = (uint32_t)gguf_arch_num(&g, arch, "attention.key_length", 0);
    d->ctx        = (uint32_t)gguf_arch_num(&g, arch, "context_length", 0);
    double eps = gguf_arch_num(&g, arch, "attention.layer_norm_rms_epsilon", 0);
    if (eps == 0) eps = gguf_arch_num(&g, arch, "attention.layer_norm_epsilon", 0);
    d->eps = (float)(eps == 0 ? 1e-6 : eps);
    d->rope_theta = (float)gguf_arch_num(&g, arch, "rope.freq_base", 10000.0);
    d->n_experts  = (uint32_t)gguf_arch_num(&g, arch, "expert_count", 0);
    d->topk       = (uint32_t)gguf_arch_num(&g, arch, "expert_used_count", 0);
    double inter = gguf_arch_num(&g, arch, "expert_feed_forward_length", 0);
    if (inter == 0) inter = gguf_arch_num(&g, arch, "feed_forward_length", 0);
    d->inter = (uint32_t)inter;
    if (d->n_kv_heads == 0) d->n_kv_heads = d->n_heads;
    if (d->head_dim == 0 && d->n_heads) d->head_dim = d->hidden / d->n_heads;
    if (d->topk == 0 && d->n_experts) d->topk = 1;

    int has_qk_norm = 0, has_exps = 0, has_lm_head = 0, has_embed = 0, has_shared = 0;
    for (uint64_t i = 0; i < g.n_tensors; ++i) {
        const LlmCodecGgufTensor* t = &g.tensors[i];
        out->n_tensors++;
        int role = llmengine_moe_role_of_tensor(t->name);
        if (role > 0 && role < LLMENGINE_MOE_ROLE_COUNT) out->role_counts[role]++;
        if (role == LLMENGINE_MOE_ROLE_Q_NORM || role == LLMENGINE_MOE_ROLE_K_NORM) has_qk_norm = 1;
        if (role == LLMENGINE_MOE_ROLE_LM_HEAD) has_lm_head = 1;
        if (role == LLMENGINE_MOE_ROLE_EXPERT_GATE) {
            has_exps = 1;
            if (t->n_dims >= 3 && t->dims[2] > out->experts_per_layer)
                out->experts_per_layer = (uint32_t)t->dims[2];
        }
        if (role == LLMENGINE_MOE_ROLE_EMBED) {
            has_embed = 1;
            if (t->n_dims >= 2 && d->vocab == 0) d->vocab = (uint32_t)t->dims[1];
        }
        if (strstr(t->name, "_shexp")) has_shared = 1;
    }
    if (!has_lm_head && has_embed) d->flags |= LLMENGINE_MOE_FLAG_TIED;
    if (has_qk_norm) d->flags |= LLMENGINE_MOE_FLAG_QK_NORM;
    if (has_exps || d->n_experts) d->flags |= LLMENGINE_MOE_FLAG_MOE;
    if (d->arch_family == LLMENGINE_MOE_ARCH_DEEPSEEK) d->flags |= LLMENGINE_MOE_FLAG_MLA;
    if (has_shared) d->flags |= LLMENGINE_MOE_FLAG_SHARED;
    if (out->experts_per_layer == 0) out->experts_per_layer = d->n_experts;
    d->router_kind = (d->arch_family == LLMENGINE_MOE_ARCH_GLM_MOE || d->arch_family == LLMENGINE_MOE_ARCH_DEEPSEEK)
                     ? LLMENGINE_MOE_ROUTER_SIGMOID : LLMENGINE_MOE_ROUTER_SOFTMAX;
    d->norm_kind = (d->arch_family == LLMENGINE_MOE_ARCH_GEMMA) ? LLMENGINE_MOE_NORM_LAYER : LLMENGINE_MOE_NORM_RMS;

    int conf = 40;
    if (d->hidden && d->n_heads && d->head_dim) conf += 15;
    if (d->n_layers) conf += 10;
    if (d->vocab) conf += 10;
    if (d->arch_family != LLMENGINE_MOE_ARCH_UNKNOWN) conf += 15;
    if (d->n_heads && d->head_dim && d->hidden == d->n_heads * d->head_dim) conf += 10;
    if (out->n_tensors == 0) conf -= 30;
    if (conf > 100) conf = 100; if (conf < 0) conf = 0;
    out->confidence = conf;
    out->source_gguf = 1;
    llmengine_moe_gguf_free(&g);
    return LLMENGINE_MOE_OK;
}

LlmEngineMoeStatus llmengine_moe_discover_auto(const char* path, LlmEngineMoeDiscover* out) {
    if (path == NULL || out == NULL) return LLMENGINE_MOE_ERR_INVALID;
    struct stat sb;
    if (stat(path, &sb) != 0) return LLMENGINE_MOE_ERR_IO;
    if (S_ISDIR(sb.st_mode)) return llmengine_moe_discover_hf(path, out);
    return llmengine_moe_discover_gguf(path, out);
}

LlmEngineMoeStatus llmengine_moe_discover_override(LlmEngineMoeArchDesc* d, const char* key, const char* value) {
    if (d == NULL || key == NULL || value == NULL) return LLMENGINE_MOE_ERR_INVALID;
    long long n = strtoll(value, NULL, 0);
    double f = strtod(value, NULL);
    if      (strcmp(key, "arch_family") == 0)  d->arch_family = (uint32_t)n;
    else if (strcmp(key, "hidden") == 0)       d->hidden = (uint32_t)n;
    else if (strcmp(key, "n_layers") == 0)     d->n_layers = (uint32_t)n;
    else if (strcmp(key, "n_heads") == 0)      d->n_heads = (uint32_t)n;
    else if (strcmp(key, "n_kv") == 0)         d->n_kv_heads = (uint32_t)n;
    else if (strcmp(key, "head_dim") == 0)     d->head_dim = (uint32_t)n;
    else if (strcmp(key, "n_experts") == 0)    d->n_experts = (uint32_t)n;
    else if (strcmp(key, "topk") == 0)         d->topk = (uint32_t)n;
    else if (strcmp(key, "inter") == 0)        d->inter = (uint32_t)n;
    else if (strcmp(key, "vocab") == 0)        d->vocab = (uint32_t)n;
    else if (strcmp(key, "ctx") == 0)          d->ctx = (uint32_t)n;
    else if (strcmp(key, "eps") == 0)          d->eps = (float)f;
    else if (strcmp(key, "rope_theta") == 0)   d->rope_theta = (float)f;
    else if (strcmp(key, "rope_variant") == 0) d->rope_variant = (uint32_t)n;
    else if (strcmp(key, "router_kind") == 0)  d->router_kind = (uint32_t)n;
    else if (strcmp(key, "norm_kind") == 0)    d->norm_kind = (uint32_t)n;
    else if (strcmp(key, "flags") == 0)        d->flags = (uint32_t)n;
    else return LLMENGINE_MOE_ERR_INVALID;
    return LLMENGINE_MOE_OK;
}

int llmengine_moe_discover_report(const LlmEngineMoeDiscover* x, char* out, size_t cap) {
    if (x == NULL || out == NULL || cap == 0) return -1;
    const LlmEngineMoeArchDesc* d = &x->desc;
    int n = snprintf(out, cap,
        "arch_family   = %u (%s)\n"
        "model_type    = %s\n"
        "source        = %s\n"
        "hidden        = %u\n"
        "n_layers      = %u (dal nome: %u)\n"
        "n_heads       = %u\n"
        "n_kv_heads    = %u\n"
        "head_dim      = %u\n"
        "n_experts     = %u (per-layer: %u)\n"
        "topk          = %u\n"
        "inter         = %u\n"
        "vocab         = %u\n"
        "eps           = %.6g\n"
        "rope_theta    = %.6g\n"
        "rope_variant  = %u (%s)\n"
        "router_kind   = %u (%s)\n"
        "norm_kind     = %u (%s)\n"
        "flags         = 0x%02x%s%s%s%s%s%s\n"
        "tensors       = %u\n"
        "confidence    = %d%%\n",
        d->arch_family, llmengine_moe_arch_family_name(d->arch_family),
        x->model_type[0] ? x->model_type : "(assente)",
        x->source_gguf ? "gguf" : "hf safetensors",
        d->hidden, d->n_layers, x->max_layer_seen, d->n_heads, d->n_kv_heads, d->head_dim,
        d->n_experts, x->experts_per_layer, d->topk, d->inter, d->vocab,
        (double)d->eps, (double)d->rope_theta,
        d->rope_variant, d->rope_variant == LLMENGINE_MOE_ROPE_INTERLEAVED ? "interleaved" : "half",
        d->router_kind, d->router_kind == LLMENGINE_MOE_ROUTER_SIGMOID ? "sigmoid" : "softmax",
        d->norm_kind, d->norm_kind == LLMENGINE_MOE_NORM_LAYER ? "layernorm" : "rmsnorm",
        d->flags,
        (d->flags & LLMENGINE_MOE_FLAG_TIED)    ? " tied"    : "",
        (d->flags & LLMENGINE_MOE_FLAG_QK_NORM) ? " qk_norm" : "",
        (d->flags & LLMENGINE_MOE_FLAG_BIAS)    ? " bias"    : "",
        (d->flags & LLMENGINE_MOE_FLAG_MOE)     ? " moe"     : "",
        (d->flags & LLMENGINE_MOE_FLAG_MLA)     ? " mla"     : "",
        (d->flags & LLMENGINE_MOE_FLAG_SHARED)  ? " shared"  : "",
        x->n_tensors, x->confidence);
    if (n < 0) return -1;
    return n;
}

/* ── M8: LLM locale (processo esterno) per gli override incerti ─────────────────────────────── */
int llmengine_moe_discover_prompt(const LlmEngineMoeDiscover* d, char* out, size_t cap) {
    if (d == NULL || out == NULL || cap == 0) return -1;
    const LlmEngineMoeArchDesc* c = &d->desc;
    int n = snprintf(out, cap,
        "Compila i campi INCERTI del descriptor di architettura di questo modello MoE, per un "
        "runtime di inferenza. NON inventare: se non sei sicuro di un campo, omettilo.\n"
        "Contesto (discovery automatico da config.json + index safetensors):\n"
        "  model_type=%s\n  arch_family=%u (%s)\n"
        "  hidden=%u n_layers=%u n_heads=%u head_count_kv=%u head_dim=%u\n"
        "  n_experts=%u topk=%u inter=%u vocab=%u eps=%.6g rope_theta=%.6g\n"
        "  flags=0x%02x %s%s%s%s%s%s\n"
        "Campi incerti da determinare (rispondi SOLO con un oggetto JSON, niente testo attorno):\n"
        "  arch_family : qwen3_moe|qwen2_moe|glm_moe|deepseek_mla|llama|gemma|qwen3_next|unknown\n"
        "  rope_variant: half|interleaved\n"
        "  router_kind : softmax|sigmoid\n"
        "  norm_kind   : rmsnorm|layernorm\n"
        "Formato risposta: {\"arch_family\":\"...\",\"rope_variant\":\"...\",\"router_kind\":\"...\",\"norm_kind\":\"...\"}\n",
        d->model_type[0] ? d->model_type : "(assente)", c->arch_family,
        llmengine_moe_arch_family_name(c->arch_family),
        c->hidden, c->n_layers, c->n_heads, c->n_kv_heads, c->head_dim,
        c->n_experts, c->topk, c->inter, c->vocab, (double)c->eps, (double)c->rope_theta, c->flags,
        (c->flags & LLMENGINE_MOE_FLAG_MOE) ? "moe " : "",
        (c->flags & LLMENGINE_MOE_FLAG_MLA) ? "mla " : "",
        (c->flags & LLMENGINE_MOE_FLAG_SHARED) ? "shared " : "",
        (c->flags & LLMENGINE_MOE_FLAG_QK_NORM) ? "qk_norm " : "",
        (c->flags & LLMENGINE_MOE_FLAG_NO_RENORM) ? "no_renorm " : "",
        (c->flags & LLMENGINE_MOE_FLAG_QK_NORM_FULL) ? "qk_norm_full " : "");
    return n;
}

/* Legge il valore di `key` da una risposta JSON tollerante (stringa quotata o token numerico). */
static int reply_read(const char* reply, const char* key, char* out, size_t cap) {
    if (reply == NULL || key == NULL || cap == 0) return 0;
    char pat[128]; snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char* p = strstr(reply, pat);
    if (p != NULL) p += strlen(pat);
    else { p = strstr(reply, key); if (p == NULL) return 0; p += strlen(key); }
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
    if (*p != ':' && *p != '=') return 0;
    ++p;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
    size_t i = 0;
    if (*p == '"') { ++p; while (*p && *p != '"' && i + 1 < cap) out[i++] = *p++; }
    else { while (*p && *p != ',' && *p != '}' && *p != '\n' && *p != '\r' && *p != ' ' && i + 1 < cap) out[i++] = *p++; }
    out[i] = 0;
    return i > 0;
}

/* Normalizza i valori "stringa-enum" negli interi attesi da llmengine_moe_discover_override. */
static void norm_enum(const char* key, char* val) {
    for (char* q = val; *q; ++q) if (*q >= 'A' && *q <= 'Z') *q = (char)(*q + 32);
    if (strcmp(key, "rope_variant") == 0) {
        if (strstr(val, "interleav") || strstr(val, "neox") || strcmp(val, "1") == 0) snprintf(val, 8, "1");
        else if (strstr(val, "half") || strstr(val, "split") || strcmp(val, "0") == 0) snprintf(val, 8, "0");
    } else if (strcmp(key, "router_kind") == 0) {
        if (strstr(val, "sigmoid") || strcmp(val, "1") == 0) snprintf(val, 8, "1");
        else if (strstr(val, "softmax") || strcmp(val, "0") == 0) snprintf(val, 8, "0");
    } else if (strcmp(key, "norm_kind") == 0) {
        if (strstr(val, "layer") || strcmp(val, "1") == 0) snprintf(val, 8, "1");
        else if (strstr(val, "rms") || strcmp(val, "0") == 0) snprintf(val, 8, "0");
    } else if (strcmp(key, "arch_family") == 0) {
        if (strstr(val, "qwen3_moe")) snprintf(val, 8, "1");
        else if (strstr(val, "qwen2_moe")) snprintf(val, 8, "2");
        else if (strstr(val, "glm")) snprintf(val, 8, "3");
        else if (strstr(val, "deepseek") || strstr(val, "mla")) snprintf(val, 8, "4");
        else if (strstr(val, "llama")) snprintf(val, 8, "5");
        else if (strstr(val, "gemma")) snprintf(val, 8, "6");
        else if (strstr(val, "qwen3_next") || strstr(val, "qwen3next")) snprintf(val, 8, "7");
        else if (strstr(val, "unknown")) snprintf(val, 8, "0");
    }
}

LlmEngineMoeStatus llmengine_moe_discover_apply_reply(LlmEngineMoeArchDesc* desc, const char* reply) {
    if (desc == NULL || reply == NULL) return LLMENGINE_MOE_ERR_INVALID;
    static const char* keys[] = {
        "arch_family", "rope_variant", "router_kind", "norm_kind",
        "hidden", "n_layers", "n_heads", "n_kv", "head_dim", "n_experts", "topk",
        "inter", "vocab", "ctx", "eps", "rope_theta"
    };
    int applied = 0;
    for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); ++i) {
        char val[64];
        if (!reply_read(reply, keys[i], val, sizeof(val))) continue;
        norm_enum(keys[i], val);
        if (llmengine_moe_discover_override(desc, keys[i], val) == LLMENGINE_MOE_OK) applied++;
    }
    return applied > 0 ? LLMENGINE_MOE_OK : LLMENGINE_MOE_ERR_INVALID;
}
