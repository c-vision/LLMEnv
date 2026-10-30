/* llm_engine_opts.c — implementazione del registro opzioni (parametri del motore, niente env). */
#include "llm_engine_opts.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

typedef enum { T_INT, T_FLOAT, T_BOOL, T_STR } OptType;

typedef struct {
    const char* name;
    const char* desc;
    OptType     type;
    int         i,  def_i;
    double      f,  def_f;
    int         b,  def_b;
    char        s[64];
    const char* def_s;
    int         set;      /* 1 = sovrascritto dall'utente */
    const char* ex;       /* valore d'esempio per l'help (--opt nome=ex) */
} Opt;

/* Parametri CURATI (nomi umani). Default compilati; nessuna env. */
static Opt g_opts[] = {
    { "context",          "capienza contesto/KV in token (0 = default del modello)", T_INT,   0, 0, 0, 0, 0, 0, {0}, NULL, 0, "131072" },
    { "maxtokens",        "tetto sui token generati",                                T_INT,   262144, 262144, 0, 0, 0, 0, {0}, NULL, 0, "4096" },
    { "thinking",         "reasoning on/off",                                        T_BOOL,  0, 0, 0, 0, 0, 0, {0}, NULL, 0, "on" },
    { "offload",          "streaming/cache esperti MoE da disco",                    T_BOOL,  0, 0, 0, 0, 0, 0, {0}, NULL, 0, "on" },
    { "offload_cache_mb", "RAM per la cache esperti (MB)",                           T_INT,   14336, 14336, 0, 0, 0, 0, {0}, NULL, 0, "16384" },
    { "dtype",            "auto|f16|bf16",                                           T_STR,   0, 0, 0, 0, 0, 0, "auto", "auto", 0, "bf16" },
    { "qos",              "default|adaptive (QoS UTILITY + prefill chunk/4: cede la macchina condivisa)", T_STR, 0, 0, 0, 0, 0, 0, "", "", 0, "default" },
    { "ds4_engine",       "GGUF DS4: auto|native|vendor (nativo o vendorizzato; per misurare e instradare)", T_STR, 0, 0, 0, 0, 0, 0, "auto", "auto", 0, "native" },
    { "ds4_mtp",          "GGUF draft MTP/DSpark per DS4 (path; vuoto = spento)",  T_STR, 0, 0, 0, 0, 0, 0, "", "", 0, "/percorso/DSpark-support.gguf" },
    { "ds4_dspark",       "DSpark: proposta speculativa (0/1; richiede ds4_mtp)",    T_BOOL, 0, 0, 0, 0, 0, 0, {0}, NULL, 0, "1" },
    { "ds4_dspark_strict","DSpark: accettazione strict (0/1)",                      T_BOOL, 0, 0, 0, 0, 0, 0, {0}, NULL, 0, "0" },
    { "kvquant",          "off|8[:finestra]|auto[:soglia]|16",                       T_STR,   0, 0, 0, 0, 0, 0, "auto", "auto", 0, "8:8192" },
    { "kvq_split_tokens", "KV-quant attn fusa: token per split-K (piccolo = piu' split)", T_INT, 0, 1024, 0, 0, 0, 0, {0}, NULL, 0, "256" },
    { "kvq_splits_max",   "KV-quant attn fusa: n. massimo di split-K",                 T_INT, 0, 64, 0, 0, 0, 0, {0}, NULL, 0, "64" },
    { "kvq_gqa",          "KV-quant attn fusa: GQA-shared (1/0, sperimentale)",         T_BOOL, 0, 0, 0, 0, 0, 0, {0}, NULL, 0, "1" },
    { "qmv_wide",          "kernel MLX qmv_wide (M>=2): -1 auto(=on per qwen3_5), 0 off, 1 on", T_INT, -1, -1, 0, 0, 0, 0, {0}, NULL, 0, "1" },
    { "df2_q8",            "DFlash2: pesi bf16 del draft quantizzati 8-bit g64 al primo uso (meta' dei byte, propose piu' veloce; solo le proposte cambiano, il verify resta esatto): 1 on, 0 off", T_BOOL, 1, 1, 0, 0, 1, 1, {0}, NULL, 0, "1" },
    { "kvdisk",           "tier SSD KV: off|on|<path> (file scratch)",               T_STR,   0, 0, 0, 0, 0, 0, "off", "off", 0, "on" },
    { "kvdisk_cap_mb",    "budget byte VIVI del tier SSD KV in MB (0 = illimitato)", T_INT,   0, 0, 0, 0, 0, 0, {0}, NULL, 0, "8192" },
    { "prefix_cache",     "prefix-cache persistente: auto (=~/.cache/llm_engine, per-modello) | off | <path>", T_STR, 0, 0, 0, 0, 0, 0, "auto", "auto", 0, "auto" },
    { "prefix_min_match",    "prefix-cache: token comuni minimi per riusare un prefisso", T_INT, 8, 8, 0, 0, 0, 0, {0}, NULL, 0, "8" },
    { "prefix_min_save_len", "prefix-cache: lunghezza minima (token) del prompt per salvarne la KV", T_INT, 64, 64, 0, 0, 0, 0, {0}, NULL, 0, "64" },
    { "memgov",           "memory governor attivo (1/0): tiene ~2 GiB liberi per macOS", T_INT, 1, 1, 0, 0, 0, 0, {0}, NULL, 0, "1" },
    { "memgov_reserve_mb", "byte da lasciare liberi per il sistema (MB)",            T_INT,   2048, 2048, 0, 0, 0, 0, {0}, NULL, 0, "2048" },
    { "memgov_grow_min_free_mb", "libero minimo richiesto per crescere le cache (MB)", T_INT, 1024, 1024, 0, 0, 0, 0, {0}, NULL, 0, "1024" },
    { "memgov_timeout_ms", "attesa massima del governor per la memoria (ms)",         T_INT,   30000, 30000, 0, 0, 0, 0, {0}, NULL, 0, "30000" },
    { "prefill",          "chunk di prefill (0 = default)",                          T_INT,   0, 0, 0, 0, 0, 0, {0}, NULL, 0, "2048" },
    { "temp",             "temperatura di sampling",                                 T_FLOAT, 0, 0, 1.0, 1.0, 0, 0, {0}, NULL, 0, "0" },
    { "topp",             "top-p",                                                   T_FLOAT, 0, 0, 1.0, 1.0, 0, 0, {0}, NULL, 0, "0.95" },
    { "minp",             "min-p",                                                   T_FLOAT, 0, 0, 0.0, 0.0, 0, 0, {0}, NULL, 0, "0.05" },
    { "seed",             "seed (-1 = casuale)",                                     T_INT,   -1, -1, 0, 0, 0, 0, {0}, NULL, 0, "42" },
    /* TurboQuant (API del motore llm_ai_mlx_set_tq_memory_opt, ABI v6): gestita qui come parametro. */
    { "tqmemory",         "TurboQuant esperti: 1=packed (poca RAM) / 0=affine-2bit (max velocita')", T_BOOL, 0, 0, 0, 0, 1, 1, {0}, NULL, 0, "0" },
    /* Self-speculative n-gram (generale, tutti i modelli, greedy): draft dalla tabella n-gram del
     * contesto, verificati dal modello. 0=off; N>=2=ordine. Output identico al greedy AR. */
    { "spec_ngram",       "self-speculative n-gram (greedy): 0=off, N>=2=ordine (2=bigram: piu' copertura)", T_INT, 2, 2, 0, 0, 0, 0, {0}, NULL, 0, "2" },
    /* DFlash2 (A1): se impostata a un path di draft DFlash2, il decode greedy usa il drafter a
     * contesto PERSISTENTE (spec streaming). "" = off. */
    { "dflash_model",     "draft DFlash2 (path) per la speculazione a contesto persistente", T_STR, 0, 0, 0, 0, 0, 0, "", "", 0, "~/ai/Models/Qwen3.8-27B-DFlash2" },
    { "df2_seq_verify",   "DFlash2: verify per-sessione (1 = forza; A/B diagnostico, default batched)", T_BOOL, 0, 0, 0, 0, 0, 0, {0}, NULL, 0, "1" },
    { "df2_ctx_span",     "DFlash2: righe di ctx del draft per span (prefill chunked; 0 = un solo span)", T_INT, 0, 1024, 0, 0, 0, 0, {0}, NULL, 0, "1024" },
    { "df2_k",          "DFlash2 A/B: forza K (0=adattivo golden; N=N)", T_INT, 0, 8, 0, 0, 0, 0, {0}, NULL, 0, "0" },
};
#define NOPTS ((int)(sizeof g_opts / sizeof g_opts[0]))

/* Engine setters applicati all'istante quando l'opzione viene impostata (niente env). */
/* TurboQuant setter: REGISTRATO dal modulo mlx (nessun riferimento diretto: i target che
 * linkano solo `llmengine_opts` non hanno così simboli irrisolti). */
static void (*g_tq_setter)(int) = NULL;
void llm_ai_mlx_opt_register_tq_setter(void (*fn)(int)) { g_tq_setter = fn; }

static Opt* find(const char* name) {
    if (!name) return NULL;
    for (int i = 0; i < NOPTS; i++) if (!strcmp(name, g_opts[i].name)) return &g_opts[i];
    return NULL;
}

/* ── Registro AVANZATO: qualunque nome di switch interno (non curato) può essere impostato via
 * API/CLI e letto con opt_get_*: è il ponte con cui si migrano i `getenv` di funzionamento. ── */
static struct { char name[64]; char val[64]; int set; } g_adv[256];
static int g_adv_n = 0;
static int adv_find(const char* name) {
    if (!name) return -1;
    for (int i = 0; i < g_adv_n; i++) if (!strcmp(name, g_adv[i].name)) return i;
    return -1;
}

static int truthy(const char* s) {
    if (!s) return 0;
    if (!strcasecmp(s, "on") || !strcasecmp(s, "true") || !strcasecmp(s, "yes") ||
        !strcasecmp(s, "1")) return 1;
    return 0;
}

int llm_ai_mlx_opt_set(const char* name, const char* value) {
    Opt* o = find(name);
    if (!o) {
        /* Nome non curato: registro avanzato (accetta qualunque switch interno). */
        if (!name || !*name || !value) return -2;
        int i = adv_find(name);
        if (i < 0) {
            if (g_adv_n >= (int)(sizeof g_adv / sizeof g_adv[0])) return -1;
            i = g_adv_n++;
            snprintf(g_adv[i].name, sizeof g_adv[i].name, "%s", name);
        }
        snprintf(g_adv[i].val, sizeof g_adv[i].val, "%s", value);
        g_adv[i].set = 1;
        return 0;
    }
    switch (o->type) {
        case T_BOOL:  o->b = truthy(value); break;
        case T_INT: {
            char* e = NULL; long v = strtol(value ? value : "", &e, 10);
            if (!value || !*value || !e || *e) return -2;
            o->i = (int)v; break;
        }
        case T_FLOAT: {
            char* e = NULL; double v = strtod(value ? value : "", &e);
            if (!value || !*value || !e || *e) return -2;
            o->f = v; break;
        }
        case T_STR: {
            if (!value) return -2;
            snprintf(o->s, sizeof o->s, "%s", value); break;
        }
    }
    o->set = 1;
    /* Opzioni che hanno un effetto IMMEDIATO su un'API del motore (coerenza col lavoro esistente). */
    if (!strcmp(o->name, "tqmemory") && g_tq_setter) g_tq_setter(o->b);
    return 0;
}

int llm_ai_mlx_opt_get_int(const char* name, int* out) {
    Opt* o = find(name);
    if (!o) {
        int i = adv_find(name);
        if (i < 0 || !out) return -1;
        *out = atoi(g_adv[i].val);
        return 0;
    }
    *out = (o->type == T_INT) ? o->i : (o->type == T_BOOL) ? o->b : o->i;
    return 0;
}
int llm_ai_mlx_opt_get_float(const char* name, double* out) {
    Opt* o = find(name);
    if (!o) {
        int i = adv_find(name);
        if (i < 0 || !out) return -1;
        *out = atof(g_adv[i].val);
        return 0;
    }
    *out = o->f; return 0;
}
int llm_ai_mlx_opt_get_bool(const char* name, int* out) {
    Opt* o = find(name);
    if (!o) {
        int i = adv_find(name);
        if (i < 0 || !out) return -1;
        *out = truthy(g_adv[i].val);
        return 0;
    }
    *out = o->b; return 0;
}
int llm_ai_mlx_opt_get_str(const char* name, char* out, int cap) {
    Opt* o = find(name);
    if (!o) {
        int i = adv_find(name);
        if (i < 0 || !out) return -1;
        snprintf(out, (size_t)cap, "%s", g_adv[i].val);
        return 0;
    }
    snprintf(out, (size_t)cap, "%s", o->s); return 0;
}
int llm_ai_mlx_opt_is_set(const char* name) {
    Opt* o = find(name);
    if (o) return o->set;
    { int i = adv_find(name); return i >= 0 ? g_adv[i].set : -1; }
}

static void fmt(const Opt* o, char* out, int cap) {
    switch (o->type) {
        case T_BOOL:  snprintf(out, (size_t)cap, "%s", o->b ? "on" : "off"); break;
        case T_INT:   snprintf(out, (size_t)cap, "%d", o->i); break;
        case T_FLOAT: snprintf(out, (size_t)cap, "%g", o->f); break;
        case T_STR:   snprintf(out, (size_t)cap, "%s", o->s); break;
    }
}

const char* llm_ai_mlx_opt_env(const char* name) {
    static char buf[128];
    Opt* o = find(name);
    if (o) {
        if (!o->set) return NULL;          /* non impostato: il codice usa il suo default */
        fmt(o, buf, sizeof buf);
        return buf;
    }
    { int i = adv_find(name); if (i >= 0) return g_adv[i].val; }
    return getenv(name);                   /* fallback di transizione (poi rimosso) */
}

int llm_ai_mlx_opt_count(void) { return NOPTS; }
int llm_ai_mlx_opt_at(int i, const char** name, const char** desc, char* value, int cap) {
    if (i < 0 || i >= NOPTS) return -1;
    if (name) *name = g_opts[i].name;
    if (desc) *desc = g_opts[i].desc;
    if (value) fmt(&g_opts[i], value, cap);
    return 0;
}
/* Esempio d'uso del valore per l'help: `--opt <nome>=<ex>`. */
int llm_ai_mlx_opt_example(int i, char* buf, int cap) {
    if (i < 0 || i >= NOPTS || buf == NULL || cap <= 0) return -1;
    snprintf(buf, (size_t)cap, "%s", g_opts[i].ex ? g_opts[i].ex : "");
    return 0;
}
