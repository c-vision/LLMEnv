/* llm_mlx_kvpagesel_test.c — test del core sparse page-level (§G-bis leva 1). Nessuna dipendenza MLX. */
#include "llm_mlx_kvpagesel.h"

#include <stdio.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)

int main(void) {
    const int nkv = 2, hd = 4, page = 4, np = 8, nqh = 2;
    KvPageSum s = llm_kvps_new(nkv, hd, page, np);
    CHECK(s.sum != NULL && s.count != NULL);

    /* ogni pagina: 4 token; kv0=[1,0,0,0], kv1=[0,1,0,0]; la pagina 3 ha key "forti" x10 */
    for (int p = 0; p < np; p++) {
        float buf[4 * 2 * 4];
        for (int t = 0; t < page; t++) {
            float strong = (p == 3) ? 10.0f : 1.0f;
            float* k0 = buf + (t * nkv + 0) * hd; for (int d = 0; d < hd; d++) k0[d] = 0.0f; k0[0] = strong;
            float* k1 = buf + (t * nkv + 1) * hd; for (int d = 0; d < hd; d++) k1[d] = 0.0f; k1[1] = strong;
        }
        CHECK(llm_kvps_add(&s, p, buf, page) == 0);
        CHECK(s.count[p] == page);
    }
    /* il cap per pagina vale: aggiungere 5 token a una pagina piena non incrementa */
    { float one[1 * 2 * 4] = {0}; CHECK(llm_kvps_add(&s, 0, one, 5) == 0); CHECK(s.count[0] == page); }

    float q[2 * 4] = {1, 0, 0, 0,  0, 1, 0, 0};
    float sc[8];
    CHECK(llm_kvps_score(&s, q, nqh, sc) == 0);
    /* top score = pagina 3 (10), le altre 1; pagine vuote (oltre used) = -1e30 */
    CHECK(sc[3] > sc[0] && sc[0] == sc[7]);
    CHECK(sc[3] > 9.0f && sc[0] == 1.0f);

    /* selezione: topk=1 -> pagina 3; + sink(0) + recent(6,7) -> {0,3,6,7} */
    int out[16];
    int n = llm_kvps_select(sc, np, /*topk=*/1, /*window=*/2, out, 16);
    CHECK(n == 4);
    CHECK(out[0] == 0 && out[1] == 3 && out[2] == 6 && out[3] == 7);

    /* topk=3 -> pagine {3,0/…}; con window=0 resta il sink */
    n = llm_kvps_select(sc, np, /*topk=*/3, /*window=*/0, out, 16);
    CHECK(n >= 3 && out[0] == 0);   /* 0 e' sempre incluso */
    int has3 = 0; for (int i = 0; i < n; i++) if (out[i] == 3) has3 = 1;
    CHECK(has3);

    llm_kvps_free(&s);
    if (fails == 0) printf("llm_mlx_kvpagesel_test: OK\n");
    return fails ? 1 : 0;
}
