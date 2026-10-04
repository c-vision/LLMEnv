/* llm_dsv41_engram_table_test.c — verifica del loader a righe della tabella Engram (fp8+ue8m0).
 * Build: tools/dsv41_engram_table_test.sh */
#include <stdio.h>
#include "llm_dsv41_engram_table.h"
#include "llm_dsv41_engram_table_ref.h"
int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/tmp/dsv41_engram_table.safetensors";
    printf("llm_dsv41_engram_table — loader righe (fp8 e4m3 + ue8m0)\n");
    LlmDsv41EngramTable t;
    if (!llm_dsv41_engram_table_open(&t, path, "engram.embed.weight", "engram.embed.scale", REF_ENT_BLOCK)) {
        printf("SKIP: %s non apribile\n", path); return 0;
    }
    int fail = 0;
    for (int r = 0; r < REF_ENT_ROWS; r++) {
        float out[REF_ENT_HD];
        if (!llm_dsv41_engram_table_row(&t, r, out)) { printf("FAIL row %d read\n", r); fail = 1; continue; }
        int ok = 1;
        for (int i = 0; i < REF_ENT_HD; i++) {
            float d = out[i] - REF_ENT_EXPECT[r * REF_ENT_HD + i]; if (d < 0) d = -d;
            if (d > 1e-6f) { ok = 0; break; }
        }
        if (!ok) { printf("FAIL row %d dequant\n", r); fail = 1; }
    }
    llm_dsv41_engram_table_close(&t);
    if (!fail) printf("PASS %-46s\n", "engram table righe");
    printf(fail ? "FAILURES: %d\n" : "ALL PASS\n", fail);
    return fail;
}
