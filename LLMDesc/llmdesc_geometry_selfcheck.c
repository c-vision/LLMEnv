/* llmdesc_geometry_selfcheck.c — i `.gmbarch` generati nelle cartelle dei modelli (<radice>/<modello>/arch/NOME.gmbarch) devono
 * descrivere lo stesso modello di config.json. Due controlli per ogni modello trovato:
 *   1. il file GIA' generato si carica, ha il nome dell'architettura e la geometria coincide con config.json;
 *   2. la geometria RIGENERATA da llmdesc_autobuild in una cartella temporanea coincide con config.json
 *      (prova il generatore, indipendentemente dai file vecchi).
 * L'oracolo e' config.json; se contiene `text_config` (modelli annidati: Qwen3.5/Qwen4, Gemma 4...) valgono i valori annidati.
 * Uso: llmdesc_geometry_selfcheck [radice dei modelli]   (default: $HOME/ai/Models). Nessun modello trovato -> exit 77 (SKIP).
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
    DIR* rd = opendir(root);
    if (!rd) { printf("SKIP: cartella dei modelli assente: %s\n", root); return 77; }
    int models = 0, files = 0, stale = 0, regen_bad = 0;
    struct dirent* e;
    while ((e = readdir(rd)) != NULL) {
        if (e->d_name[0] == '.') continue;
        char mdir[2048], cpath[2300], adir[2300];
        snprintf(mdir, sizeof mdir, "%s/%s", root, e->d_name);
        snprintf(cpath, sizeof cpath, "%s/config.json", mdir);
        snprintf(adir, sizeof adir, "%s/arch", mdir);
        char* cfg = slurp_file(cpath);
        if (!cfg) continue;
        int counted = 0;
        /* 1. file generati */
        DIR* ad = opendir(adir);
        struct dirent* a;
        while (ad && (a = readdir(ad)) != NULL) {
            size_t l = strlen(a->d_name);
            if (l <= 8 || strcmp(a->d_name + l - 8, ".gmbarch") != 0) continue;
            char fp[2500]; snprintf(fp, sizeof fp, "%s/%s", adir, a->d_name);
            LlmDesc d; memset(&d, 0, sizeof d);
            int bad = 0;
            if (llmdesc_load(fp, &d) != 0 || !d.geometry.present) { printf("  %s/%s: non si carica o senza geometry\n", e->d_name, a->d_name); bad = 1; }
            else {
                char stem[128]; snprintf(stem, sizeof stem, "%.*s", (int)(l - 8), a->d_name);
                if (strcmp(d.arch, stem) != 0) { printf("  %s/%s: arch '%s' != nome file\n", e->d_name, a->d_name, d.arch); bad++; }
                int mm = compare("generato", cfg, &d);
                if (mm) printf("  %s/%s: OBSOLETO o errato (%d campi)\n", e->d_name, a->d_name, mm);
                bad += mm;
            }
            files++; if (bad) stale++;
            counted = 1;
        }
        if (ad) closedir(ad);
        /* 2. rigenerazione in cartella temporanea */
        if (counted) {
            char tmpl[] = "/tmp/llmdesc_geom_XXXXXX";
            if (mkdtemp(tmpl)) {
                char tc[2400]; snprintf(tc, sizeof tc, "%s/config.json", tmpl);
                FILE* f = fopen(tc, "wb");
                if (f) {
                    fwrite(cfg, 1, strlen(cfg), f); fclose(f);
                    char outp[2600] = "";
                    LlmDesc d; memset(&d, 0, sizeof d);
                    if (llmdesc_autobuild(tmpl, outp, sizeof outp) != 1 || llmdesc_load(outp, &d) != 0 || !d.geometry.present) {
                        printf("  %s: rigenerazione fallita\n", e->d_name); regen_bad++;
                    } else {
                        int mm = compare("rigenerato", cfg, &d);
                        if (mm) { printf("  %s: il generatore produce una geometria diversa da config.json (%d campi)\n", e->d_name, mm); regen_bad++; }
                    }
                    remove(outp);
                    char ad2[2500]; snprintf(ad2, sizeof ad2, "%s/arch", tmpl); rmdir(ad2); remove(tc);
                }
                rmdir(tmpl);
            }
            models++;
        }
        free(cfg);
    }
    closedir(rd);
    if (models == 0) { printf("SKIP: nessun .gmbarch generato sotto %s\n", root); return 77; }
    printf("[geometry] modelli=%d file=%d obsoleti/errati=%d rigenerazioni errate=%d\n", models, files, stale, regen_bad);
    int ok = (stale == 0 && regen_bad == 0);
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
