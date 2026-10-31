/* llm_dsv41_abi_test.c — contratto ABI di ds41-native, SENZA checkpoint (model-free):
 * caps dichiarate = capacita' vere, rifiuti espliciti (path inesistente, .gguf senza config.json),
 * kv_set_max con argomenti non validi. Build: tools/dsv41_abi_test.sh */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "llmengine.h"

static int fail = 0;
#define CHECK(c, name) do { if (c) printf("PASS %s\n", name); else { printf("FAIL %s\n", name); fail = 1; } } while (0)

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const LlmEngineLlmEngine* e = llmengine_llm_engine_ds41();
    CHECK(e != NULL, "accessor ds41");
    if (!e) return 1;
    CHECK((e->caps & LLMMODELIO_CAP_IDS) != 0, "cap IDS dichiarata");
    CHECK((e->caps & LLMMODELIO_CAP_KV_MAX) != 0, "cap KV_MAX dichiarata (kv_set_max e' reale)");
    CHECK((e->caps & (LLMMODELIO_CAP_VISION | LLMMODELIO_CAP_PREFIX | LLMMODELIO_CAP_LORA)) == 0,
          "nessuna cap non implementata dichiarata");

    CHECK(e->load_model(NULL) == NULL, "load NULL rifiutato");
    CHECK(e->load_model("/nonexistent/ds41_dir") == NULL, "load dir inesistente rifiutato");

    /* .gguf senza config.json accanto: rifiuto esplicito (non un crash ne' un caricamento muto) */
    char d[] = "/tmp/ds41_abi_XXXXXX";
    if (mkdtemp(d)) {
        char p[256];
        snprintf(p, sizeof p, "%s/m.gguf", d);
        FILE* f = fopen(p, "wb"); if (f) { fputs("GGUF", f); fclose(f); }
        CHECK(e->load_model(p) == NULL, "container .gguf senza config.json rifiutato");
        unlink(p); rmdir(d);
    }

    CHECK(e->kv_set_max(NULL, 4096) == -1, "kv_set_max su modello NULL rifiutato");
    e->set_expert_swap(NULL);   /* non deve crashare */
    printf(fail ? "SOME FAIL\n" : "ALL PASS\n");
    return fail;
}
