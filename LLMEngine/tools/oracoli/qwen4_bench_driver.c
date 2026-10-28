/* qwen4_bench_driver.c — MISURA: carica qwen4_exp via ABI (lib worktree) e misura tok/s decode. */
#include "llmengine.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
int main(int argc, char** argv){
    if(argc<3){ fprintf(stderr,"uso: %s <lib.dylib> <model_dir> [max_new]\n",argv[0]); return 2; }
    void* h = dlopen(argv[1], RTLD_NOW|RTLD_GLOBAL);
    if(!h){ fprintf(stderr,"dlopen: %s\n", dlerror()); return 3; }
    const LlmEngineLlmEngine* (*get)(void) = (const LlmEngineLlmEngine*(*)(void))dlsym(h,"llmengine_llm_engine_mlx");
    if(!get){ fprintf(stderr,"dlsym: %s\n", dlerror()); return 3; }
    const LlmEngineLlmEngine* e = get();
    fprintf(stderr,"engine=%s caps=0x%x\n", e->name, e->caps);
    if(e->set_expert_swap && !(argc>4)) e->set_expert_swap(argv[2]);
    LlmEngineLlmModel m = e->load_model(argv[2]);
    if(!m){ fprintf(stderr,"load_model FAIL\n"); return 4; }
    fprintf(stderr,"model loaded\n");
    LlmEngineLlmSession s = e->create_session(m);
    if(!s){ fprintf(stderr,"create_session FAIL\n"); return 4; }
    const char* prompt="The capital of France is";
    int ids[64]; int n = e->tokenize ? e->tokenize(m, prompt, ids, 64) : 0;
    if(n<=0){ ids[0]=785; ids[1]=6722; ids[2]=315; ids[3]=9625; ids[4]=374; n=5; }
    fprintf(stderr,"prompt ids n=%d\n", n);
    e->append_ids(s, ids, n);
    int maxn = argc>3?atoi(argv[3]):24;
    int out[256];
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    int g = e->generate_ids(s, maxn, -1, out);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double dt = (t1.tv_sec-t0.tv_sec)+ (t1.tv_nsec-t0.tv_nsec)/1e9;
    printf("generated=%d  time=%.3f s  TPS=%.3f tok/s\n", g, dt, g>0? g/dt : 0.0);
    printf("ids:"); for(int i=0;i<g && i<16;i++) printf(" %d", out[i]); printf("\n");
    return 0;
}
