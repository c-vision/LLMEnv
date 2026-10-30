/* llmdesc_gen.c — crea il descrittore `<model_type>.gmbarch` di un'architettura a partire da una cartella modello HF
 * (config.json + nomi dei tensori da model.safetensors.index.json o dall'header di model.safetensors).
 *
 * Il descrittore e' per ARCHITETTURA (non per modello): se `<abi>/<slug>.gmbarch` esiste non si tocca mai (i descrittori
 * curati a mano restano). Quello generato contiene i ruoli dedotti dai nomi dei tensori, la mappa dei campi di config,
 * le strutture (attenzione gqa|mla, moe, vision) e la quantizzazione di default; e' una base valida che si puo' rifinire.
 * Regole portate dal vecchio strumento gen_gmbarch_abi di GMBasic. Nessuna dipendenza oltre alla libc. */
#include "llmdesc.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define GEN_MAXPAT 4000

static const struct { const char* match; const char* role; } k_rules[] = {
    { "vision_tower", "VISION" }, { "vision_model", "VISION" }, { "embed_vision", "VISION" }, { "visual.", "VISION" },
    { "aligner", "VISION_ALIGNER" }, { "multi_modal_projector", "VISION_ALIGNER" },
    { "embed_tokens", "EMBEDDING" }, { "tok_embeddings", "EMBEDDING" }, { "wte.", "EMBEDDING" }, { ".embed.", "EMBEDDING" },
    { "word_embeddings", "EMBEDDING" },
    { "lm_head", "OUTPUT" }, { "output.weight", "OUTPUT" }, { ".head.", "OUTPUT" }, { "head.weight", "OUTPUT" },
    { "attn_sink", "ATTN_SINK" }, { "self_attn.sinks", "ATTN_SINK" },
    { "switch_mlp", "ROUTED_EXPERT" }, { "mlp.experts", "ROUTED_EXPERT" }, { "experts.", "ROUTED_EXPERT" },
    { "shared_experts", "SHARED_EXPERT" }, { "shared_expert.", "SHARED_EXPERT" },
    { "shared_expert_gate", "ROUTER" }, { "mlp.gate.weight", "ROUTER" }, { "mlp.gate.e_score", "ROUTER" },
    { "e_score_correction_bias", "ROUTER" }, { "mlp.router", "ROUTER" }, { ".router.", "ROUTER" },
    { "block_sparse_moe.gate", "ROUTER" }, { "moe.gate", "ROUTER" }, { "expert_bias", "ROUTER" },
    { "feed_forward.gate.weight", "ROUTER" },
    { "q_a_proj", "ATTN_Q_A" }, { "q_b_proj", "ATTN_Q_B" }, { "q_lora", "ATTN_Q_A" },
    { "kv_a_proj", "ATTN_KV" }, { "kv_b_proj", "ATTN_KV" }, { "kv_a_layernorm", "NORM" },
    { "q_norm", "NORM" }, { "k_norm", "NORM" },
    { "q_proj", "ATTN_Q" }, { "k_proj", "ATTN_KV" }, { "v_proj", "ATTN_V" }, { "o_proj", "ATTN_WO" },
    { "linear_attn", "ATTN_LINEAR" },
    { "gate_proj", "FFN" }, { "up_proj", "FFN" }, { "down_proj", "FFN" }, { "mlp.", "FFN" }, { "feed_forward", "FFN" },
    { "ffn.", "FFN" }, { "norm", "NORM" }, { "layernorm", "NORM" }, { ".ln_", "NORM" },
};
#define N_RULES ((int)(sizeof k_rules / sizeof k_rules[0]))

static char* slurp(const char* p, size_t* n) {
    FILE* f = fopen(p, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len < 0 || len > 64L * 1024 * 1024) { fclose(f); return NULL; }
    char* b = (char*)malloc((size_t)len + 1);
    if (!b) { fclose(f); return NULL; }
    size_t got = fread(b, 1, (size_t)len, f);
    fclose(f);
    b[got] = '\0';
    if (n) *n = got;
    return b;
}

/* ---- scansione JSON minimale (stringhe e profondita') -------------------------------------------------------------- */

/* Salta una stringa JSON che inizia su s[i]=='"': ritorna l'indice dopo la chiusura. */
static size_t skip_str(const char* s, size_t i) {
    for (i++; s[i] && s[i] != '"'; i++) if (s[i] == '\\' && s[i + 1]) i++;
    return s[i] ? i + 1 : i;
}

/* Valore (posizione dopo i due punti, senza spazi) della chiave `key` figlia DIRETTA dell'oggetto [b,e). NULL se assente. */
static const char* direct_key(const char* b, const char* e, const char* key) {
    size_t kl = strlen(key);
    int depth = 0;
    for (const char* p = b; p < e && *p; p++) {
        if (*p == '"') {
            const char* q = p;
            const char* close = p + 1;
            while (close < e && *close != '"') { if (*close == '\\') close++; close++; }
            if (depth == 1 && (size_t)(close - q - 1) == kl && !strncmp(q + 1, key, kl)) {
                const char* v = close + 1;
                while (v < e && isspace((unsigned char)*v)) v++;
                if (v < e && *v == ':') { v++; while (v < e && isspace((unsigned char)*v)) v++; return v; }
            }
            p = close;
            continue;
        }
        if (*p == '{' || *p == '[') depth++;
        else if (*p == '}' || *p == ']') depth--;
    }
    return NULL;
}

/* Estremi [b,e) dell'oggetto che inizia a `v` (v[0]=='{'). */
static const char* obj_end(const char* v) {
    int depth = 0;
    for (const char* p = v; *p; p++) {
        if (*p == '"') { p = v + skip_str(v, (size_t)(p - v)) - 1; continue; }
        if (*p == '{' || *p == '[') depth++;
        else if (*p == '}' || *p == ']') { depth--; if (depth == 0) return p + 1; }
    }
    return v + strlen(v);
}

static int copy_str_value(const char* v, char* out, size_t cap) {
    if (!v || *v != '"') return 0;
    size_t k = 0;
    for (v++; *v && *v != '"' && k + 1 < cap; v++) out[k++] = *v;
    out[k] = '\0';
    return k > 0;
}

/* ---- nomi dei tensori ---------------------------------------------------------------------------------------------- */

typedef struct { char** v; int n; } PatSet;

static int ends_with(const char* s, const char* x) {
    size_t a = strlen(s), b = strlen(x);
    return a >= b && !strcmp(s + a - b, x);
}

/* "model.layers.12.mlp.experts.7.w1.weight" -> "model.layers.N.mlp.experts.N.w1.weight" */
static void norm_name(const char* in, char* out, size_t cap) {
    size_t o = 0;
    for (size_t i = 0; in[i] && o + 4 < cap;) {
        if (in[i] == '.' && isdigit((unsigned char)in[i + 1])) {
            out[o++] = '.'; out[o++] = 'N';
            i++;
            while (isdigit((unsigned char)in[i])) i++;
        } else out[o++] = in[i++];
    }
    out[o] = '\0';
}

static void add_pattern(PatSet* ps, const char* name) {
    if (ends_with(name, ".scales") || ends_with(name, ".biases") || ends_with(name, ".scale") || ends_with(name, ".bias")) return;
    char buf[512];
    norm_name(name, buf, sizeof buf);
    for (int i = 0; i < ps->n; i++) if (!strcmp(ps->v[i], buf)) return;
    if (ps->n >= GEN_MAXPAT) return;
    ps->v[ps->n++] = strdup(buf);
}

/* chiavi di primo livello di un oggetto JSON [b,e) -> add_pattern (salta __metadata__) */
static void keys_of(const char* b, const char* e, PatSet* ps) {
    int depth = 0;
    for (const char* p = b; p < e && *p; p++) {
        if (*p == '"') {
            const char* close = p + 1;
            while (close < e && *close != '"') { if (*close == '\\') close++; close++; }
            const char* v = close + 1;
            while (v < e && isspace((unsigned char)*v)) v++;
            if (depth == 1 && v < e && *v == ':') {
                char nm[512];
                size_t k = (size_t)(close - p - 1);
                if (k >= sizeof nm) k = sizeof nm - 1;
                memcpy(nm, p + 1, k);
                nm[k] = '\0';
                if (strcmp(nm, "__metadata__") != 0) add_pattern(ps, nm);
            }
            p = close;
            continue;
        }
        if (*p == '{' || *p == '[') depth++;
        else if (*p == '}' || *p == ']') depth--;
    }
}

static int load_tensor_names(const char* dir, PatSet* ps) {
    char path[4096];
    snprintf(path, sizeof path, "%s/model.safetensors.index.json", dir);
    size_t n = 0;
    char* js = slurp(path, &n);
    if (js) {
        const char* wm = direct_key(js, js + n, "weight_map");
        if (wm && *wm == '{') keys_of(wm, obj_end(wm), ps);
        free(js);
        return ps->n > 0;
    }
    snprintf(path, sizeof path, "%s/model.safetensors", dir);
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    unsigned char len8[8];
    if (fread(len8, 1, 8, f) != 8) { fclose(f); return 0; }
    unsigned long long hl = 0;
    for (int i = 7; i >= 0; i--) hl = hl * 256 + len8[i];
    if (hl == 0 || hl > 64ULL * 1024 * 1024) { fclose(f); return 0; }
    char* hdr = (char*)malloc((size_t)hl + 1);
    if (!hdr || fread(hdr, 1, (size_t)hl, f) != (size_t)hl) { free(hdr); fclose(f); return 0; }
    fclose(f);
    hdr[hl] = '\0';
    keys_of(hdr, hdr + hl, ps);
    free(hdr);
    return ps->n > 0;
}

/* ---- generazione --------------------------------------------------------------------------------------------------- */

static int has_pat(const PatSet* ps, const char* sub) {
    for (int i = 0; i < ps->n; i++) if (strstr(ps->v[i], sub)) return 1;
    return 0;
}

static void slug_of(const char* arch, char* out, size_t cap) {
    size_t k = 0;
    for (const char* p = arch; *p && k + 1 < cap; p++) out[k++] = (*p == '-' || *p == ' ' || *p == '.') ? '_' : (char)tolower((unsigned char)*p);
    out[k] = '\0';
    char* t = strstr(out, "_text");   if (t) *t = '\0';
    t = strstr(out, "_vision");       if (t) *t = '\0';
}

/* Etichetta leggibile della cartella modello per il commento del descrittore: "org/nome" se e' uno snapshot del cache HF
 * (models--org--nome/snapshots/<commit>), altrimenti il nome della cartella; vuota per le cartelle temporanee. */
static void model_label(const char* dir, char* out, size_t cap) {
    out[0] = '\0';
    const char* snap = strstr(dir, "/snapshots/");
    if (snap) {
        const char* b = snap;
        while (b > dir && b[-1] != '/') b--;
        if (!strncmp(b, "models--", 8)) {
            size_t k = 0;
            for (const char* p = b + 8; p < snap && k + 1 < cap; p++) {
                if (p[0] == '-' && p[1] == '-' && p + 2 < snap) { out[k++] = '/'; p++; } else out[k++] = *p;
            }
            out[k] = '\0';
            return;
        }
    }
    const char* base = strrchr(dir, '/');
    base = base ? base + 1 : dir;
    if (strncmp(base, "fastllm_gmbarch_", 16) != 0) snprintf(out, cap, "%s", base);
}

int llmdesc_ensure_for_dir(const char* hf_dir, const char* abi_dir, char* out_path, size_t cap, char* err, size_t errcap) {
#define FAIL(msg) do { if (err && errcap) snprintf(err, errcap, "%s", (msg)); return -1; } while (0)
    char path[4096], abi[2048], arch[96] = "", slug[96];
    if (!hf_dir || !*hf_dir) FAIL("no model directory");
    if (abi_dir && *abi_dir) snprintf(abi, sizeof abi, "%s", abi_dir);
    else if (!llmdesc_abi_dir(abi, sizeof abi)) FAIL("the ABI folder was not found (install LLMEnv or set LLMDESC_ABI_DIR)");
    snprintf(path, sizeof path, "%s/config.json", hf_dir);
    size_t cn = 0;
    char* cfg = slurp(path, &cn);
    if (!cfg) FAIL("config.json not found in the model directory");

    /* nome dell'architettura: come llmdesc_resolve (primo "model_type", poi architectures[0]) */
    {
        const char* v = direct_key(cfg, cfg + cn, "model_type");
        if (!copy_str_value(v, arch, sizeof arch)) {
            const char* a = direct_key(cfg, cfg + cn, "architectures");
            if (a && *a == '[') { const char* q = strchr(a, '"'); copy_str_value(q, arch, sizeof arch); }
        }
    }
    if (!arch[0]) { free(cfg); FAIL("no model_type or architectures in config.json"); }
    slug_of(arch, slug, sizeof slug);
    static const char* skip[] = { "silero_vad", "flux", "vae", "clip", "t5", "whisper", "bert", "unet" };
    for (size_t i = 0; i < sizeof skip / sizeof skip[0]; i++)
        if (!strcmp(slug, skip[i])) { free(cfg); FAIL("not a text/vision language model architecture (no descriptor needed)"); }

    snprintf(path, sizeof path, "%s/%s.gmbarch", abi, slug);
    if (out_path && cap) snprintf(out_path, cap, "%s", path);
    struct stat st;
    if (stat(path, &st) == 0) { free(cfg); return 0; }
    snprintf(path, sizeof path, "%s/%s.llmarch", abi, slug);
    if (stat(path, &st) == 0) { free(cfg); return 0; }

    PatSet ps = { (char**)calloc(GEN_MAXPAT, sizeof(char*)), 0 };
    if (!ps.v) { free(cfg); FAIL("out of memory"); }
    if (!load_tensor_names(hf_dir, &ps)) { free(cfg); free(ps.v); FAIL("no tensor names (model.safetensors.index.json / model.safetensors)"); }

    /* configurazione del testo: text_config / language_model se presenti, altrimenti la radice */
    const char* sb = cfg;
    const char* se = cfg + cn;
    {
        const char* t = direct_key(cfg, cfg + cn, "text_config");
        if (!t) t = direct_key(cfg, cfg + cn, "language_model");
        if (t && *t == '{') { sb = t; se = obj_end(t); }
    }
#define HAS(k) (direct_key(sb, se, (k)) != NULL)
    int qn = has_pat(&ps, "q_norm"), kn = has_pat(&ps, "k_norm");
    int moe = has_pat(&ps, "mlp.gate.weight") || has_pat(&ps, "switch_mlp") || has_pat(&ps, "mlp.experts") ||
              has_pat(&ps, "mlp.router") || has_pat(&ps, "block_sparse_moe") || has_pat(&ps, "moe.gate");
    int vis = has_pat(&ps, "vision_tower") || has_pat(&ps, "embed_vision") || has_pat(&ps, "visual.");
    int mla = HAS("q_lora_rank") || HAS("kv_lora_rank") || has_pat(&ps, "q_a_proj") || has_pat(&ps, "kv_a_proj");

    snprintf(path, sizeof path, "%s/%s.gmbarch", abi, slug);
    FILE* f = fopen(path, "wb");
    if (!f) { free(cfg); for (int i = 0; i < ps.n; i++) free(ps.v[i]); free(ps.v); FAIL("cannot write the descriptor in the ABI folder"); }
    char label[200];
    model_label(hf_dir, label, sizeof label);
    fprintf(f, "{\n  \"_comment\": \"Generato automaticamente da llmdesc_ensure_for_dir%s%s%s. Base dedotta dai nomi dei tensori: "
               "si puo' rifinire a mano; il file esistente non viene mai sovrascritto.\",\n",
            label[0] ? " (modello " : "", label, label[0] ? ")" : "");
    fprintf(f, "  \"arch\": \"%s\",\n", slug);
    fprintf(f, "  \"source\": {\n    \"format\": \"safetensors\",\n    \"config\": \"config.json\",\n    \"index\": \"model.safetensors.index.json\"\n  },\n");
    static const struct { const char* k; const char* field; } cmap[] = {
        { "n_layers", "num_hidden_layers" }, { "hidden", "hidden_size" }, { "n_heads", "num_attention_heads" },
        { "n_kv", "num_key_value_heads" }, { "head_dim", "head_dim" }, { "vocab", "vocab_size" }, { "rope_theta", "rope_theta" },
        { "dense_inter", "intermediate_size" }, { "experts", "num_experts" }, { "top_k", "num_experts_per_tok" },
        { "moe_inter", "moe_intermediate_size" },
    };
    int first = 1;
    fprintf(f, "  \"config_map\": {");
    for (size_t i = 0; i < sizeof cmap / sizeof cmap[0]; i++)
        if (HAS(cmap[i].field)) { fprintf(f, "%s\n    \"%s\": \"%s\"", first ? "" : ",", cmap[i].k, cmap[i].field); first = 0; }
    fprintf(f, "\n  },\n  \"roles\": [");
    first = 1;
    {
        char used[N_RULES] = { 0 };
        for (int n = 0; n < ps.n; n++)
            for (int r = 0; r < N_RULES; r++)
                if (strstr(ps.v[n], k_rules[r].match)) {
                    if (!used[r]) {
                        used[r] = 1;
                        fprintf(f, "%s\n    { \"match\": \"%s\", \"role\": \"%s\" }", first ? "" : ",", k_rules[r].match, k_rules[r].role);
                        first = 0;
                    }
                    break;
                }
    }
    fprintf(f, "\n  ],\n");
    fprintf(f, "  \"quant\": {\n    \"NORM\": \"f32\",\n    \"EMBEDDING\": \"f16\",\n    \"OUTPUT\": \"q6_k\",\n    \"recipe\": \"_M\",\n"
               "    \"recipe_boost\": [\"attn_v\", \"ffn_down\"],\n    \"recipe_boost_fmt\": \"q6_k\"\n  },\n");
    fprintf(f, "  \"structures\": {\n    \"norm\": { \"type\": \"rmsnorm\", \"prim\": \"rmsnorm\" },\n");
    if (mla)
        fprintf(f, "    \"attention\": { \"type\": \"mla\", \"steps\": [ {\"op\":\"q_lora\",\"prim\":\"\"}, {\"op\":\"kv_lora\",\"prim\":\"\"}, "
                   "{\"op\":\"rope\",\"prim\":\"\"}, {\"op\":\"sdpa\",\"prim\":\"\"}, {\"op\":\"sink\",\"prim\":\"\"} ] }");
    else
        fprintf(f, "    \"attention\": { \"type\": \"gqa\", \"steps\": [ %s%s{\"op\":\"rope\",\"prim\":\"\"}, {\"op\":\"sdpa\",\"prim\":\"\"} ] }",
                qn ? "{\"op\":\"q_norm\",\"prim\":\"rmsnorm\"}, " : "", kn ? "{\"op\":\"k_norm\",\"prim\":\"rmsnorm\"}, " : "");
    if (moe)
        fprintf(f, ",\n    \"moe\": { \"experts\": \"cfg:experts\", \"top_k\": \"cfg:top_k\", \"steps\": [ {\"op\":\"router_topk\",\"prim\":\"\"}, "
                   "{\"op\":\"experts\",\"prim\":\"\"}, {\"op\":\"shared\",\"prim\":\"\"} ] }");
    if (vis) fprintf(f, ",\n    \"vision\": { \"type\": \"vit\", \"present\": true }");
    fprintf(f, "\n  }\n}\n");
    int werr = ferror(f);
    fclose(f);
    free(cfg);
    for (int i = 0; i < ps.n; i++) free(ps.v[i]);
    free(ps.v);
    if (werr) FAIL("write error");
    return 1;
#undef HAS
#undef FAIL
}
