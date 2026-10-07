/* llm_mlx_xs_pool_test.c — test MODEL-FREE (solo libc) dello streaming esperti:
 *   1) pool riusabile dei bundle (`llm_mlx_xs_pool.h`): riciclo, capacità, esaurimento;
 *   2) ingest safetensors (`llmengine_moe_safetensors_ingest.h`) su un file sintetico: indicizzazione
 *      delle slice, offset corretti, e i nuovi hint `madvise` (prefetch/evict) non crashano.
 * Non carica modelli. Esce 0 = tutti PASS.
 * Build (dalla dir mlx):  cc -I. -Imoe/include tests/llm_mlx_xs_pool_test.c -o /tmp/t && /tmp/t */
#include "../llm_mlx_xs_pool.h"
#include "../../moe/include/llmengine_moe_safetensors_ingest.h"
#include <assert.h>
#include <string.h>
#include <stdio.h>

static int g_fail = 0;
#define CHECK(cond, msg) do { if (cond) printf("PASS %s\n", msg); else { printf("FAIL %s\n", msg); g_fail++; } } while (0)

/* ── mini-generatore safetensors: layer 0, switch_mlp, gate/up/down × weight/scales/biases ── */
static void build_safetensors(const char* path, int E) {
    const char* proj[3] = {"gate_proj", "up_proj", "down_proj"};
    long long off = 0;
    char body[4096]; body[0] = 0;
    int realk = 0;
    for (int p = 0; p < 3; p++)
        for (int kind = 0; kind < 3; kind++) {
            const char* suf = (kind == 0 ? "weight" : kind == 1 ? "scales" : "biases");
            int d1 = 4, d2 = (kind == 0 ? 4 : 2), esz = (kind == 0 ? 4 : 2);
            long long sz = (long long)E * d1 * d2 * esz;
            char name[128]; sprintf(name, "model.layers.0.mlp.switch_mlp.%s.%s", proj[p], suf);
            if (realk) strcat(body, ",");
            char ent[512]; sprintf(ent, "\"%s\":{\"dtype\":\"%s\",\"shape\":[%d,%d,%d],\"data_offsets\":[%lld,%lld]}",
                                   name, kind == 0 ? "U32" : "BF16", E, d1, d2, off, off + sz);
            strcat(body, ent); off += sz; realk++;
        }
    char hdr[4096]; strcpy(hdr, "{"); strcat(hdr, body); strcat(hdr, "}");
    size_t pad = (8 - (strlen(hdr) & 7)) & 7;
    for (size_t i = 0; i < pad; i++) strcat(hdr, " ");
    uint64_t hlen = (uint64_t)strlen(hdr);
    FILE* f = fopen(path, "wb");
    fwrite(&hlen, 8, 1, f); fwrite(hdr, 1, hlen, f);
    for (uint64_t i = 0; i < (uint64_t)off; i++) { unsigned char b = (unsigned char)(i & 0xff); fwrite(&b, 1, 1, f); }
    fclose(f);
}

int main(void) {
    /* 1. POOL: riciclo / capacità / esaurimento */
    LlmXsSlot* a = NULL; LlmXsSlot* b = NULL; LlmXsSlot* c = NULL;
    uint8_t* p1 = llm_xs_pool_acquire(100, &a);
    CHECK(p1 != NULL && a != NULL, "pool: acquire");
    memset(p1, 0xAB, 100);
    uint8_t* p2 = llm_xs_pool_acquire(200, &b);
    CHECK(p2 != NULL && b != a, "pool: slot distinto finche' in uso");
    llm_xs_slot_dtor(a);
    uint8_t* p3 = llm_xs_pool_acquire(100, &c);
    CHECK(p3 == p1 && c == a, "pool: buffer RICICLATO");
    CHECK(((uint8_t*)p3)[0] == 0xAB, "pool: riciclo senza copia (contenuto intatto)");
    llm_xs_slot_dtor(c); llm_xs_slot_dtor(b);
    LlmXsSlot* hold[LLM_XS_POOL_MAX];
    int all = 1;
    for (int i = 0; i < LLM_XS_POOL_MAX; i++) { hold[i] = NULL; if (llm_xs_pool_acquire(64, &hold[i]) == NULL) all = 0; }
    CHECK(all, "pool: acquisisce tutti gli slot");
    LlmXsSlot* nud = NULL;
    CHECK(llm_xs_pool_acquire(64, &nud) == NULL && nud == NULL, "pool: esaurito -> NULL (fallback chiamante)");
    for (int i = 0; i < LLM_XS_POOL_MAX; i++) llm_xs_slot_dtor(hold[i]);

    /* 2. INGEST + ADVISE su safetensors sintetico */
    char dir[] = "/tmp/gmb_xs_test_XXXXXX";
    if (mkdtemp(dir) == NULL) { printf("FAIL mkdtemp\n"); return 1; }
    char path[512]; snprintf(path, sizeof path, "%s/model.safetensors", dir);
    build_safetensors(path, 2);

    LlmEngineMoeTsIngest* ing = (LlmEngineMoeTsIngest*)calloc(1, sizeof *ing);
    CHECK(llmengine_moe_ts_open(ing, dir) >= 1 && ing->present[0] == 1, "ingest: open + present[0]");
    const LlmEngineMoeTsSlice* w = llmengine_moe_ts_slice(ing, 0, 0, 0);
    CHECK(w != NULL && w->n == 2 && w->d1 == 4 && w->d2 == 4 && w->stride == 64, "ingest: slice weight");
    const LlmEngineMoeTsSlice* sc = llmengine_moe_ts_slice(ing, 0, 1, 1);
    CHECK(sc != NULL && sc->n == 2 && sc->stride == 16 && sc->d1 == 4 && sc->d2 == 2, "ingest: slice up/scales");
    uint32_t nb = 0;
    const uint8_t* pe1 = llmengine_moe_ts_get(ing, 0, 0, 0, 1, &nb);
    CHECK(pe1 != NULL && nb == 64 && pe1[0] == (uint8_t)(64 & 0xff), "ingest: offset esperto 1 corretto");
    llmengine_moe_ts_advise_expert(ing, 0, 0, 0, MADV_WILLNEED);
    llmengine_moe_ts_advise_expert(ing, 0, 0, 1, MADV_DONTNEED);
    llmengine_moe_ts_advise_expert(ing, 0, 0, 9, MADV_WILLNEED);   /* fuori range: no-op */
    llmengine_moe_ts_advise_expert(ing, 0, 5, 0, MADV_WILLNEED);   /* proj fuori range: no-op */
    CHECK(1, "ingest: advise (prefetch/evict) senza crash");

    llmengine_moe_ts_close(ing); free(ing);
    remove(path); rmdir(dir);
    printf(g_fail ? "\n%d FALLITI\n" : "\nTUTTI PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
