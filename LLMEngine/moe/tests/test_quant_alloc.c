/* test_quant_alloc.c — §E3: misura di sensibilità per-tensore + allocatore bit greedy.
 * Gate: (a) l'errore relativo scende coi bit ed è in (0,1); (b) il peso imatrix cambia la misura;
 * (c) l'allocatore rispetta il budget medio, promuove prima i più sensibili, non spreca budget. */
#include "llmengine_moe_quant_alloc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } } while (0)

static float rnd(unsigned* s) { *s = *s * 1103515245u + 12345u; return (float)((*s >> 8) & 0xFFFF) / 32768.0f - 1.0f; }

int main(void) {
    /* (a) errore relativo monotòno: 2 > 4 > 8 bit, in (0,1). */
    {
        enum { R = 8, C = 128, G = 64 };
        float x[R * C]; unsigned s = 7;
        for (uint32_t i = 0; i < R * C; ++i) x[i] = rnd(&s);
        double e2 = -1, e4 = -1, e8 = -1;
        CHECK(llmengine_moe_quant_rel_error(x, NULL, R, C, G, 2, &e2) == 0);
        CHECK(llmengine_moe_quant_rel_error(x, NULL, R, C, G, 4, &e4) == 0);
        CHECK(llmengine_moe_quant_rel_error(x, NULL, R, C, G, 8, &e8) == 0);
        printf("[alloc] rel 2/4/8 bit = %.4e / %.4e / %.4e\n", e2, e4, e8);
        CHECK(e2 > e4 && e4 > e8);
        CHECK(e2 > 0.0 && e2 < 1.0 && e8 > 0.0 && e8 < 1.0);

        /* (b) il peso imatrix (colonne a basso peso quasi ignorate) cambia la misura. */
        float imp[C];
        for (uint32_t i = 0; i < C; ++i) imp[i] = (i < C / 2) ? 1e-4f : 1.0f;
        double ew = -1;
        CHECK(llmengine_moe_quant_rel_error(x, imp, R, C, G, 4, &ew) == 0);
        printf("[alloc] rel 4 bit pesata = %.4e (uniforme %.4e)\n", ew, e4);
        CHECK(fabs(ew - e4) > 1e-6);
    }

    /* parametri invalidi */
    {
        float x[8] = {0}; double e;
        CHECK(llmengine_moe_quant_rel_error(NULL, NULL, 1, 8, 8, 4, &e) < 0);
        CHECK(llmengine_moe_quant_rel_error(x, NULL, 1, 8, 8, 3, &e) < 0);   /* bits 3 non impaccabile */
        CHECK(llmengine_moe_quant_rel_error(x, NULL, 1, 8, 8, 4, NULL) < 0);
    }

    /* (c) allocatore: candidati {4,8}. */
    {
        const int cand[2] = {4, 8};
        /* budget sotto il minimo → tutti al candidato 0 */
        {
            uint64_t np[3] = {100, 100, 100};
            double rel[6] = {1e-2, 1e-3, 1e-2, 1e-3, 1e-2, 1e-3};
            int oc[3];
            CHECK(llmengine_moe_quant_alloc_bits(np, rel, 3, 2, cand, 3.0, oc) == 0);
            CHECK(oc[0] == 0 && oc[1] == 0 && oc[2] == 0);
        }
        /* budget alto → tutti promossi */
        {
            uint64_t np[3] = {100, 100, 100};
            double rel[6] = {1e-2, 1e-3, 1e-2, 1e-3, 1e-2, 1e-3};
            int oc[3];
            CHECK(llmengine_moe_quant_alloc_bits(np, rel, 3, 2, cand, 8.0, oc) == 0);
            CHECK(oc[0] == 1 && oc[1] == 1 && oc[2] == 1);
        }
        /* budget per UN solo upgrade (avg target 5.34 < 6) → va al più sensibile (indice 1) */
        {
            uint64_t np[3] = {100, 100, 100};
            double rel[6] = {1e-2, 1e-2,   1e-2, 1e-5,   1e-2, 1e-2};
            int oc[3];
            CHECK(llmengine_moe_quant_alloc_bits(np, rel, 3, 2, cand, 5.34, oc) == 0);
            CHECK(oc[0] == 0 && oc[1] == 1 && oc[2] == 0);
            double avg = (4.0 * 100 + 8.0 * 100 + 4.0 * 100) / 300.0;
            CHECK(avg <= 5.34 + 1e-9);
        }
        /* guadagno nullo (errori uguali) → nessuna promozione anche con budget */
        {
            uint64_t np[2] = {100, 100};
            double rel[4] = {1e-2, 1e-2, 1e-2, 1e-2};
            int oc[2];
            CHECK(llmengine_moe_quant_alloc_bits(np, rel, 2, 2, cand, 8.0, oc) == 0);
            CHECK(oc[0] == 0 && oc[1] == 0);
        }
        /* input invalidi */
        {
            uint64_t np[2] = {100, 100};
            double rel[4] = {1, 2, 3, 4};
            int oc[2];
            const int bad[2] = {8, 4};
            CHECK(llmengine_moe_quant_alloc_bits(np, rel, 2, 2, bad, 6.0, oc) < 0);  /* non crescenti */
            CHECK(llmengine_moe_quant_alloc_bits(NULL, rel, 2, 2, cand, 6.0, oc) < 0);
            CHECK(llmengine_moe_quant_alloc_bits(np, rel, 0, 2, cand, 6.0, oc) < 0);
        }
    }

    if (fails) { fprintf(stderr, "test_quant_alloc: %d FAIL\n", fails); return 1; }
    printf("test_quant_alloc: OK\n");
    return 0;
}
