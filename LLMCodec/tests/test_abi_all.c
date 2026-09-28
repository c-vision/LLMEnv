/* Carica TUTTE le .llmarch di ABI/ e verifica che siano complete (ruoli+codecs+weight_source). */
#include "llmcodec_rolemap.h"
#include <stdio.h>
#include <string.h>
#include <dirent.h>

#ifndef LLMCODEC_ABI_DIR
#define LLMCODEC_ABI_DIR "../../ABI"
#endif

int main(void) {
    DIR* d = opendir(LLMCODEC_ABI_DIR);
    if (d == NULL) { printf("FAIL opendir %s\n", LLMCODEC_ABI_DIR); return 1; }
    struct dirent* e; int n = 0, bad = 0;
    while ((e = readdir(d)) != NULL) {
        size_t l = strlen(e->d_name);
        if (l < 9 || strcmp(e->d_name + l - 8, ".llmarch") != 0) continue;
        char p[600]; snprintf(p, sizeof p, "%s/%s", LLMCODEC_ABI_DIR, e->d_name);
        LlmCodecRoleMap rm;
        int rc = llmcodec_rolemap_load(p, &rm);
        int ok = (rc == 0 && rm.n > 0 && rm.n_codec > 0 && rm.n_ws > 0 && llmcodec_rolemap_group(&rm) > 0 && llmcodec_rolemap_bits(&rm) > 0);
        if (!ok) bad++;
        printf("  %-34s arch=%-20s ruoli=%-3d codecs=%-2d ws=%-2d struct=%d grp=%d bits=%d %s\n",
               e->d_name, rm.arch, rm.n, rm.n_codec, rm.n_ws, rm.n_st, llmcodec_rolemap_group(&rm), llmcodec_rolemap_bits(&rm), ok ? "" : "<== INCOMPLETA");
        n++;
    }
    closedir(d);
    printf("[abi] file=%d incompleti=%d\n", n, bad);
    int ok = (n >= 22 && bad == 0);
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
