/* llm_dsv41_dspark_test.c — verifica matematica DSpark vs riferimento tiny. Build: tools/dsv41_dspark_test.sh */
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "llm_dsv41_dspark.h"
#include "llm_dsv41_dspark_ref.h"
static int nclose(const float* a, const float* b, int n, float tol) {
    for (int i = 0; i < n; i++) { float d = a[i]-b[i]; if (d<0) d=-d; float m=b[i]<0?-b[i]:b[i]; if (m<1.0f) m=1.0f;
        if (d > tol*m) { printf("  [%d] got=%.6g exp=%.6g\n", i, a[i], b[i]); return 0; } }
    return 1;
}
int main(void) {
    printf("llm_dsv41_dspark — topk/markov/confidence/proposta vs riferimento\n");
    int fail = 0;
    int width = (REF_DS_WIN < REF_DS_START+1 ? REF_DS_WIN : REF_DS_START+1) + REF_DS_BLK;
    int topk[REF_DS_BSZ*REF_DS_BLK*32];
    llm_dsv41_dspark_topk_idxs(REF_DS_WIN, REF_DS_BSZ, REF_DS_BLK, REF_DS_START, topk);
    int tok = 1; for (int i = 0; i < REF_DS_BSZ*REF_DS_BLK*width; i++) if (topk[i] != REF_DS_TOPK[i]) tok = 0;
    if (tok) printf("PASS %-46s\n", "topk_idxs"); else { printf("FAIL %-46s\n", "topk_idxs"); fail = 1; }

    float bias[REF_DS_VOCAB];
    llm_dsv41_dspark_markov_bias(REF_DS_EMBED + (size_t)REF_DS_ID[0]*REF_DS_RANK, REF_DS_HEAD, REF_DS_RANK, REF_DS_VOCAB, bias);
    if (nclose(bias, REF_DS_BIAS, REF_DS_VOCAB, 1e-4f)) printf("PASS %-46s\n", "MarkovHead bias");
    else { printf("FAIL %-46s\n", "MarkovHead bias"); fail = 1; }

    float c = llm_dsv41_dspark_confidence(REF_DS_H, REF_DS_MARKOV_EMB, REF_DS_DIM, REF_DS_RANK, REF_DS_PROJ);
    if (fabsf(c - REF_DS_CONF[0]) <= 1e-4f) printf("PASS %-46s\n", "ConfidenceHead");
    else { printf("FAIL %-46s got=%.6g exp=%.6g\n", "ConfidenceHead", c, REF_DS_CONF[0]); fail = 1; }

    int32_t prop[8];
    llm_dsv41_dspark_proposal(NULL, REF_DS_EMBED, REF_DS_HEAD, REF_DS_RANK, REF_DS_VOCAB, REF_DS_BLK, REF_DS_ID[0], prop);
    if (prop[0] == REF_DS_PROPOSAL) printf("PASS %-46s\n", "proposta (argmax bias)");
    else { printf("FAIL %-46s got=%d exp=%d\n", "proposta (argmax bias)", prop[0], REF_DS_PROPOSAL); fail = 1; }

    printf(fail ? "FAILURES: %d\n" : "ALL PASS\n", fail);
    return fail;
}
