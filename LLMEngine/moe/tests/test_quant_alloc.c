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

    /* (d) allocatore ANCORATO (S3): parte dal FLOOR per-tensore e promuove solo sopra, entro il
     * budget. Verifica: (1) budget=pavimento -> nessuna promozione; (2) budget un po' sopra ->
     * promuove il piu' sensibile; (3) pavimento>budget -> -2 e out=floor. */
    {
        const int cand[3] = {2, 4, 8};   /* bit crescenti */
        /* (1) floor {1,1,1} (4 bit) e budget 4.0 = esattamente il pavimento -> resta il floor */
        { uint64_t np[3] = {100,100,100};
          double rel[9] = {0.10,0.05,0.01, 0.10,0.05,0.01, 0.10,0.05,0.01};
          const int mn[3] = {1,1,1}; int oc[3];
          CHECK(llmengine_moe_quant_alloc_bits_min(np, rel, 3, 3, cand, mn, 4.0, oc) == 0);
          CHECK(oc[0] == 1 && oc[1] == 1 && oc[2] == 1); }
        /* (2) floor {1,1,1}, budget 5.34 -> promuove UNO, il piu' sensibile (indice 2) a 8 bit
         *     (avg risultante (4+4+8)/3 = 5.333 <= 5.34) */
        { uint64_t np[3] = {100,100,100};
          double rel[9] = {0.10,0.05,0.045, 0.10,0.05,0.045, 0.10,0.05,0.001};
          const int mn[3] = {1,1,1}; int oc[3];
          CHECK(llmengine_moe_quant_alloc_bits_min(np, rel, 3, 3, cand, mn, 5.34, oc) == 0);
          CHECK(oc[0] == 1 && oc[1] == 1 && oc[2] == 2);
          double avg = (4.0*100 + 4.0*100 + 8.0*100) / 300.0; CHECK(avg <= 5.34 + 1e-9); }
        /* (3) floor {2,2,2} (8 bit) ma budget 4.0 -> infeasible, out resta il floor */
        { uint64_t np[3] = {100,100,100};
          double rel[9] = {0.10,0.05,0.01, 0.10,0.05,0.01, 0.10,0.05,0.01};
          const int mn[3] = {2,2,2}; int oc[3];
          CHECK(llmengine_moe_quant_alloc_bits_min(np, rel, 3, 3, cand, mn, 4.0, oc) == -2);
          CHECK(oc[0] == 2 && oc[1] == 2 && oc[2] == 2); }
        /* input invalidi */
        { uint64_t np[2] = {100,100}; double rel[6] = {0.1,0.05,0.01,0.1,0.05,0.01}; int oc[2];
          const int mn[2] = {0,0}; const int badmn[2] = {5,0};
          CHECK(llmengine_moe_quant_alloc_bits_min(np, rel, 2, 3, cand, NULL, 5.0, oc) < 0);
          CHECK(llmengine_moe_quant_alloc_bits_min(np, rel, 2, 3, cand, badmn, 5.0, oc) < 0); }
    }

    /* (e) SWAP / deviazione verso il basso (S3): ancora = floor (6 bit per tutti, target 6.0 =
     * il pavimento, nessuno slack). Il tensore 0 e' insensibile, il 2 sensibilissimo: lo swap
     * SCENDE il 0 (6->4, libera 200) e SALE il 2 (6->8, costa 200) -> guadagno netto > 0 A PARI
     * bpw. Verifica: oc = {1,2,3} e media bit ancora 6.0. */
    {
        const int cand[4] = {2, 4, 6, 8};
        uint64_t np[3] = {100,100,100};
        double rel[12] = {
            0.100, 0.099, 0.0980, 0.0975,   /* tensore 0: insensibile (guadagni minimi) */
            0.100, 0.050, 0.0200, 0.0150,   /* tensore 1: media */
            0.100, 0.050, 0.0200, 0.0010    /* tensore 2: sensibilissimo in alto */
        };
        const int mn[3] = {2,2,2};          /* ancora = 6 bit per tutti */
        int oc[3];
        CHECK(llmengine_moe_quant_alloc_bits_min(np, rel, 3, 4, cand, mn, 6.0, oc) == 0);
        /* lo swap promuove il SENSIBILE (2) e retrocede l'INSENSIBILE (0), a pari bpw; puo'
         * cascatare (piu' di un gradino), quindi si verifica la direzione, non il valore esatto. */
        CHECK(oc[2] == 3);          /* il sensibile e' salito al massimo */
        CHECK(oc[0] <= 1);          /* l'insensibile e' sceso sotto l'ancora */
        double avg = (cand[oc[0]] + cand[oc[1]] + cand[oc[2]]) / 3.0; CHECK(avg <= 6.0 + 1e-9);
    }

    if (fails) { fprintf(stderr, "test_quant_alloc: %d FAIL\n", fails); return 1; }
    printf("test_quant_alloc: OK\n");
    return 0;
}
