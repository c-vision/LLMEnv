/* llm_dsv4_parity_test.c — parità col riferimento ds4 su un prompt raw (token noti). */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "llm_dsv4_engine.h"
int main(int argc, char** argv){
    const char* path = argc > 1 ? argv[1] : "/Users/gaetano/ds4/ds4flash.gguf";
    LlmDsv4Engine* e = llm_dsv4_engine_load(path, 512);
    if(!e){ fprintf(stderr,"load fallita\n"); return 1; }
    LlmDsv4Config* c = &e->w.cfg;
    int toks[16], np=0;
    if(argc>2){ char* q=argv[2]; while(*q && np<16){ toks[np++]=atoi(q); while(*q && *q!=',') q++; if(*q==',') q++; } }
    else { int def[5]={671,6102,294,8760,344}; np=5; for(int i=0;i<5;i++) toks[i]=def[i]; }
    int last=toks[np-1];
    float* hc = (float*)malloc(sizeof(float)*c->n_hc*c->n_embd);
    int pos=0;
    for(int i=0;i<np-1;i++,pos++) llm_dsv4_forward_step(c,&e->w,&e->caches,toks[i],pos,hc);
    llm_dsv4_forward_step(c,&e->w,&e->caches,last,pos,hc);
    float* logits=(float*)malloc(sizeof(float)*c->n_vocab);
    llm_dsv4_output_logits(c,&e->w,hc,logits);
    const char* dump = getenv("LLM_DSV4_DUMP_LOGITS");
    if (dump && dump[0]) { FILE* f=fopen(dump,"wb"); if(f){ fwrite(logits,sizeof(float),(size_t)c->n_vocab,f); fclose(f);} }
    int am=0; for(int i=1;i<c->n_vocab;i++) if(logits[i]>logits[am]) am=i;
    printf("mio argmax=%d logit=%.4f\n", am, logits[am]);
    int top[8]; for(int i=0;i<8;i++) top[i]=-1;
    for(int i=0;i<c->n_vocab;i++) for(int j=0;j<8;j++){ if(top[j]<0||logits[i]>logits[top[j]]){ for(int m=7;m>j;m--) top[m]=top[m-1]; top[j]=i; break; } }
    for(int i=0;i<8;i++) printf("  top%d: %d = %.4f\n", i, top[i], top[i]>=0?logits[top[i]]:0);
    printf("  logits[22]=%.4f logits[13]=%.4f\n", logits[22], logits[13]);
    free(logits); free(hc); llm_dsv4_engine_free(e);
    return 0;
}
