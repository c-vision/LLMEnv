/* llmdesc_abi_selfcheck.c — validita', localizzazione e politica della cartella ABI (nessun modello necessario).
 * Uso: llmdesc_abi_selfcheck <ABI dir del repo> */
#include "llmdesc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); g_fail++; } } while (0)

static void put(const char* dir, const char* name, const char* body) {
    char p[2048]; snprintf(p, sizeof p, "%s/%s", dir, name);
    FILE* f = fopen(p, "wb"); if (f) { fputs(body, f); fclose(f); }
}
static int g_called = 0;
static char g_resolver_dir[1024] = "";
static int resolver(char* out, size_t cap) { g_called++; if (!g_resolver_dir[0]) return 0; snprintf(out, cap, "%s", g_resolver_dir); return 1; }

int main(int argc, char** argv) {
    const char* repo_abi = argc > 1 ? argv[1] : NULL;
    char tmpl[] = "/tmp/llmdesc_abi_XXXXXX"; char* root = mkdtemp(tmpl);
    if (root == NULL) { printf("SKIP mkdtemp\n"); return 77; }
    char good[1024], empty[1024], newer[1024], notdir[1024];
    snprintf(good, sizeof good, "%s/good", root);   mkdir(good, 0755);
    snprintf(empty, sizeof empty, "%s/empty", root); mkdir(empty, 0755);
    snprintf(newer, sizeof newer, "%s/newer", root); mkdir(newer, 0755);
    snprintf(notdir, sizeof notdir, "%s/nope", root);
    put(good, "x.gmbarch", "{}"); put(good, "ABI_VERSION", "1\n");
    put(empty, "README", "no descriptors");
    put(newer, "x.gmbarch", "{}"); put(newer, "ABI_VERSION", "99\n");

    CHECK(llmdesc_abi_valid(good) == 1);
    CHECK(llmdesc_abi_valid(empty) == 0);       /* cartella senza .gmbarch */
    CHECK(llmdesc_abi_valid(newer) == 0);       /* schema piu' recente di quello leggibile */
    CHECK(llmdesc_abi_valid(notdir) == 0);
    CHECK(llmdesc_abi_valid(NULL) == 0 && llmdesc_abi_valid("") == 0);

    char out[2048], how[128];
    unsetenv("LLMDESC_ABI_DIR");
    setenv("LLMDESC_ABI_DIR", good, 1);
    CHECK(llmdesc_abi_locate(out, sizeof out, how, sizeof how) == 1 && strcmp(out, good) == 0 && strcmp(how, "LLMDESC_ABI_DIR") == 0);
    setenv("LLMDESC_ABI_DIR", empty, 1);        /* env non valido: non deve essere restituito da locate */
    if (llmdesc_abi_locate(out, sizeof out, how, sizeof how)) CHECK(strcmp(out, empty) != 0);

    /* risolutore: chiamato da llmdesc_abi_dir quando l'env non c'e' */
    unsetenv("LLMDESC_ABI_DIR");
    snprintf(g_resolver_dir, sizeof g_resolver_dir, "%s", good);
    llmdesc_set_abi_policy(resolver, 1);
    CHECK(llmdesc_abi_dir(out, sizeof out) == 1 && strcmp(out, good) == 0 && g_called == 1);
    /* required + risolutore che non trova: nessuna localizzazione di ripiego, nessun descrittore generico */
    snprintf(g_resolver_dir, sizeof g_resolver_dir, "%s", notdir);
    CHECK(llmdesc_abi_dir(out, sizeof out) == 0);
    {   /* un modello minimale: con required fallisce, senza required ripiega sul descrittore generico */
        char mdir[1024]; snprintf(mdir, sizeof mdir, "%s/model", root); mkdir(mdir, 0755);
        put(mdir, "config.json", "{\"model_type\":\"llama\",\"num_attention_heads\":4,\"num_key_value_heads\":4}");
        LlmDesc d; memset(&d, 0, sizeof d);
        CHECK(llmdesc_resolve(mdir, &d, NULL, 0) == -1);
        llmdesc_set_abi_policy(resolver, 0);
        snprintf(g_resolver_dir, sizeof g_resolver_dir, "%s", good);
        memset(&d, 0, sizeof d);
        CHECK(llmdesc_resolve(mdir, &d, NULL, 0) == 0 && strcmp(d.arch, "llama") == 0);
    }
    llmdesc_set_abi_policy(NULL, 0);

    if (repo_abi != NULL) {
        CHECK(llmdesc_abi_valid(repo_abi) == 1);
        setenv("LLMDESC_ABI_DIR", repo_abi, 1);
        char want[4096]; if (realpath(repo_abi, want) == NULL) snprintf(want, sizeof want, "%s", repo_abi);   /* locate normalizza i ".." */
        CHECK(llmdesc_abi_locate(out, sizeof out, how, sizeof how) == 1 && strcmp(out, want) == 0);
    }
    char cmd[2048]; snprintf(cmd, sizeof cmd, "rm -rf '%s'", root); if (system(cmd) != 0) { }
    if (g_fail == 0) printf("llmdesc_abi_selfcheck: OK\n");
    return g_fail == 0 ? 0 : 1;
}
