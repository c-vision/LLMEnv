/* llm_dsv4_engine_real_test.c — engine reale: load packed + generate greedy su pochi token. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "llm_dsv4_engine.h"
int main(int argc, char** argv){
    const char* path = argc > 1 ? argv[1] : "/Users/gaetano/ds4/ds4flash.gguf";
    int nctx = argc > 2 ? atoi(argv[2]) : 128;
    clock_t t0 = clock();
    LlmDsv4Engine* e = llm_dsv4_engine_load(path, nctx);
    if(!e){ fprintf(stderr,"engine load fallita\n"); return 1; }
    double ld = (double)(clock()-t0)/CLOCKS_PER_SEC;
    LlmDsv4Config* c = &e->w.cfg;
    printf("caricato in %.1fs: layer=%d embd=%d vocab=%d exps=%d/%d hc=%d swa=%d nctx=%d (packed=%s)\n",
           ld,c->n_layer,c->n_embd,c->n_vocab,c->n_expert,c->n_expert_used,c->n_hc,c->n_swa,nctx,
           e->w.exps?"si":"no");
    int toks[16]; toks[0]=4000; toks[1]=221; toks[2]=900;
    t0 = clock();
    llm_dsv4_engine_generate(e, toks, 3, 4);
    double gt = (double)(clock()-t0)/CLOCKS_PER_SEC;
    printf("generate %.2fs ->", gt);
    for(int i=0;i<7;i++) printf(" %d", toks[i]);
    printf("\n");
    int bad=0; for(int i=0;i<7;i++) if(toks[i]<0||toks[i]>=c->n_vocab) bad++;
    llm_dsv4_engine_free(e);
    printf(bad?"FAIL\n":"PASS engine reale\n");
    return bad?1:0;
}
