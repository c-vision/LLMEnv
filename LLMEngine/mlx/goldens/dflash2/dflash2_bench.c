/* Benchmark DFlash2: 120 token greedy — acceptance (DF2_STATS) e tempo vs plain. */
#include <stdio.h>
#include "llm_engine_opts.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "llmengine_mlx.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

int main(int argc, char** argv) {
    const char* TARGET = llm_ai_mlx_opt_env("LLM_DF2_TARGET");
    if (!TARGET) TARGET = "/Volumes/Armor5TB/LocalLLM/ai/models/Qwen3.8-27B-8bit";
    const char* DRAFT = llm_ai_mlx_opt_env("LLM_DF2_DRAFT");
    if (!DRAFT) DRAFT = "/Volumes/Armor5TB/LocalLLM/ai/models/Qwen3.8-27B-DFlash2-mxfp4";
    const int32_t prompt5[]={760,6511,314,9338,369};
    const int N = argc>1 ? atoi(argv[1]) : 120;
    /* argv[2] = lunghezza contesto (prefill sintetico periodico); 0 = prompt 5-token. */
    const int ctx = argc>2 ? atoi(argv[2]) : 0;
    const int np = ctx > 5 ? ctx : 5;
    int32_t* prompt = (int32_t*)malloc(sizeof(int32_t)*(size_t)np);
    if (ctx > 5) { for (int i=0;i<np;i++) prompt[i]=(i%1000)+1; }
    else { memcpy(prompt, prompt5, 5*sizeof(int32_t)); }
    int32_t eos = -1;

    llm_ai_mlx_model_handle target=llm_ai_mlx_load_model(TARGET);
    if(!target){fprintf(stderr,"target fail\n");return 1;}
    llm_ai_mlx_dflash2_handle dfl=llm_ai_mlx_dflash2_create(target,DRAFT);
    if(!dfl){fprintf(stderr,"create fail\n");return 1;}

    /* plain greedy */
    llm_ai_mlx_session_handle s1=llm_ai_mlx_create_session(target);
    llm_ai_mlx_session_append(s1,prompt,np);
    int32_t* o1=(int32_t*)malloc(sizeof(int32_t)*N);
    double t0=now(); int n1=llm_ai_mlx_session_generate(s1,N,eos,o1); double dt1=now()-t0;

    /* dflash */
    llm_ai_mlx_session_handle s2=llm_ai_mlx_create_session(target);
    llm_ai_mlx_session_append(s2,prompt,np);
    int32_t* o2=(int32_t*)malloc(sizeof(int32_t)*N);
    t0=now(); int n2=llm_ai_mlx_dflash2_generate(dfl,s2,eos,N,o2); double dt2=now()-t0;

    /* parita' */
    int eq = (n1==n2);
    for(int i=0; eq && i<n1 && i<n2; i++) if(o1[i]!=o2[i]) eq=0;
    printf("plain   : n=%d  %.2fs (%.2f tok/s)\n", n1, dt1, n1/dt1);
    printf("dflash2 : n=%d  %.2fs (%.2f tok/s)  %s\n", n2, dt2, n2/dt2, eq?"PARITY==OK":"PARITY MISMATCH");
    if(!eq){ int d=0; for(int i=0;i<n1&&i<n2;i++) if(o1[i]!=o2[i]){d=i;break;} printf("  first diff at %d: %d vs %d\n",d,o1[d],o2[d]); }

    free(o1); free(o2); free(prompt);
    llm_ai_mlx_dflash2_free(dfl);
    llm_ai_mlx_destroy_session(s1); llm_ai_mlx_destroy_session(s2);
    llm_ai_mlx_free_model(target);
    return eq?0:2;
}
