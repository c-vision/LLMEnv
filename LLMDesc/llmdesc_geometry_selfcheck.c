/* llmdesc_geometry_selfcheck.c — per ogni modello locale (<radice>/<modello>/config.json) il descrittore risolto da llmdesc_resolve
 * deve avere una geometria uguale a quella di config.json. Il descrittore viene dalla cartella ABI/ di LLMEnv (unica sede);
 * la geometria si ricava in memoria: non esistono piu' file per modello. L'oracolo e' config.json: valgono i figli diretti di
 * `text_config` se esiste (modelli annidati), altrimenti della radice.
 * Uso: llmdesc_geometry_selfcheck [radice dei modelli] [cartella ABI]   (default: $HOME/ai/Models, ABI trovata da llmdesc_abi_dir).
 * Nessun modello trovato -> exit 77 (SKIP).
 * Compilazione: clang -I LLMDesc LLMDesc/llmdesc_geometry_selfcheck.c LLMDesc/llmdesc.c -o /tmp/llmdesc_geometry_selfcheck */
#include "llmdesc.h"
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static char* slurp_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* b = (char*)malloc((size_t)n + 1);
    if (!b) { fclose(f); return NULL; }
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return NULL; }
    fclose(f); b[n] = 0;
    return b;
}

/* Valore numerico di `"key": <numero>` tra i FIGLI DIRETTI dell'oggetto che inizia a `obj` (puntatore a '{'); 1 se trovato.
 * Non si guarda negli oggetti annidati: in Qwen4 `text_config.mtp.num_hidden_layers` (1) precede `text_config.num_hidden_layers` (48). */
static int direct_num(const char* obj, const char* key, double* out) {
    int depth = 0;
    size_t kl = strlen(key);
    for (const char* p = obj; *p; p++) {
        if (*p == '"') {
            const char* s = ++p;
            while (*p && *p != '"') { if (*p == '\\' && p[1]) p++; p++; }
            if (!*p) return 0;
            const char* q = p + 1;
            while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
            if (depth == 1 && *q == ':' && (size_t)(p - s) == kl && memcmp(s, key, kl) == 0) {
                q++;
                while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
                char* e = NULL; double v = strtod(q, &e);
                if (e != q) { *out = v; return 1; }
                return 0;
            }
            continue;
        }
        if (*p == '{' || *p == '[') depth++;
        else if (*p == '}' || *p == ']') { if (--depth == 0) return 0; }
    }
    return 0;
}
/* Inizio dell'oggetto da interrogare: `text_config` se il config ne ha uno (modelli annidati), altrimenti la radice. */
static int cfg_num(const char* cfg, const char* key, double* out) {
    const char* tc = strstr(cfg, "\"text_config\"");
    if (tc) {
        const char* b = strchr(tc, '{');
        if (b && direct_num(b, key, out)) return 1;
    }
    const char* r = strchr(cfg, '{');
    return r ? direct_num(r, key, out) : 0;
}

static int compare(const char* tag, const char* cfg, const LlmDesc* d) {
    static const struct { const char* key; size_t off; } F[] = {
        { "hidden_size", 0 }, { "num_hidden_layers", 1 }, { "num_attention_heads", 2 },
        { "vocab_size", 3 }, { "max_position_embeddings", 4 },
    };
    const unsigned got[] = { d->geometry.hidden, d->geometry.n_layers, d->geometry.n_heads, d->geometry.vocab, d->geometry.ctx };
    int bad = 0;
    for (size_t i = 0; i < sizeof F / sizeof F[0]; i++) {
        double want;
        if (!cfg_num(cfg, F[i].key, &want)) continue;   /* chiave assente nel config: nessun oracolo */
        if ((unsigned)want != got[F[i].off]) { printf("    %s: %s atteso %.0f, nel .gmbarch %u\n", tag, F[i].key, want, got[F[i].off]); bad++; }
    }
    double nh = 0, nkv = 0;
    if (cfg_num(cfg, "num_attention_heads", &nh)) {
        if (!cfg_num(cfg, "num_key_value_heads", &nkv)) nkv = nh;
        if ((unsigned)nkv != d->geometry.n_kv_heads) { printf("    %s: num_key_value_heads atteso %.0f, nel .gmbarch %u\n", tag, nkv, d->geometry.n_kv_heads); bad++; }
    }
    return bad;
}

int main(int argc, char** argv) {
    char root[1024];
    if (argc > 1) snprintf(root, sizeof root, "%s", argv[1]);
    else { const char* h = getenv("HOME"); snprintf(root, sizeof root, "%s/ai/Models", h ? h : "."); }
    if (argc > 2 && argv[2][0]) setenv("LLMDESC_ABI_DIR", argv[2], 1);
    char abi[2048] = "";
    if (!llmdesc_abi_dir(abi, sizeof abi)) { printf("FAIL: cartella ABI/ non trovata (LLMDESC_ABI_DIR o argv[2])\n"); return 1; }
    DIR* rd = opendir(root);
    if (!rd) { printf("SKIP: cartella dei modelli assente: %s\n", root); return 77; }
    int models = 0, from_abi = 0, generic = 0, bad = 0;
    struct dirent* e;
    while ((e = readdir(rd)) != NULL) {
        if (e->d_name[0] == '.') continue;
        char mdir[2048], cpath[2300];
        snprintf(mdir, sizeof mdir, "%s/%s", root, e->d_name);
        snprintf(cpath, sizeof cpath, "%s/config.json", mdir);
        char* cfg = slurp_file(cpath);
        if (!cfg) continue;
        LlmDesc d; memset(&d, 0, sizeof d);
        char dp[2600] = "";
        if (llmdesc_resolve(mdir, &d, dp, sizeof dp) != 0) { printf("  %s: nessun descrittore risolto\n", e->d_name); bad++; free(cfg); models++; continue; }
        models++;
        if (dp[0] && strncmp(dp, abi, strlen(abi)) == 0) from_abi++; else generic++;
        if (!d.geometry.present) { printf("  %s: geometria assente\n", e->d_name); bad++; }
        else {
            int mm = compare(e->d_name, cfg, &d);
            if (mm) { printf("  %s: la geometria differisce da config.json (%d campi)\n", e->d_name, mm); bad++; }
        }
        free(cfg);
    }
    closedir(rd);
    if (models == 0) { printf("SKIP: nessun modello sotto %s\n", root); return 77; }
    printf("[geometry] ABI=%s modelli=%d con descrittore ABI=%d generici=%d errori=%d\n", abi, models, from_abi, generic, bad);
    printf(bad ? "FAIL\n" : "PASS\n");
    return bad ? 1 : 0;
}
