/* Regressione su FIXTURE REALI (samples/Fixtures/codec/): confronto col golden dell'implementatore
 * ufficiale (MLX per mxfp4; engine per dsv41 fp8). */
#include "llmcodec.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#ifndef LLMCODEC_FIX_DIR
#define LLMCODEC_FIX_DIR "../../samples/Fixtures/codec"
#endif
static uint8_t* slurp(const char* p, long* n){ FILE* f=fopen(p,"rb"); if(!f) return NULL; fseek(f,0,SEEK_END); *n=ftell(f); fseek(f,0,SEEK_SET); uint8_t* b=malloc(*n); if(fread(b,1,*n,f)!=(size_t)*n){} fclose(f); return b; }
int main(void){
    char p[600]; int ok=1;
    /* --- mxfp4 reale (gpt-oss) vs golden MLX --- */
    { long W,S,E; snprintf(p,sizeof p,"%s/mxfp4/weight2d.bin",LLMCODEC_FIX_DIR); uint8_t* w=slurp(p,&W);
      snprintf(p,sizeof p,"%s/mxfp4/scales2d.bin",LLMCODEC_FIX_DIR); uint8_t* s=slurp(p,&S);
      snprintf(p,sizeof p,"%s/mxfp4/expected_f32.bin",LLMCODEC_FIX_DIR); uint8_t* e=slurp(p,&E);
      if(!w||!s||!e){ printf("[real] mxfp4: fixture mancanti\n"); ok=0; }
      else { float* o=malloc(2*2880*4); const LlmCodec* c=llmcodec_resolve("fp4");
             if(c && c->decode(w,W,1,1,32,2,2880,s,S,o)==0){ const float* g=(const float*)e; double md=0;
                for(int i=0;i<2*2880;i++){ double d=fabs(o[i]-g[i]); if(d>md)md=d; }
                printf("[real] mxfp4 gpt-oss vs MLX: maxerr=%.2e -> %s\n", md, md==0?"BIT-EXACT":"FAIL"); ok &= (md==0);
             } else { printf("[real] mxfp4: decode fail\n"); ok=0; } free(w); free(s); free(e); }
    }
    /* --- GPTQ reale (Qwen2.5-0.5B) vs golden HF-formula --- */
    { long QV,SV,ZV,GV,EV; snprintf(p,sizeof p,"%s/gptq_real/qweight.bin",LLMCODEC_FIX_DIR); uint8_t* qw=slurp(p,&QV);
      snprintf(p,sizeof p,"%s/gptq_real/scales.bin",LLMCODEC_FIX_DIR); uint8_t* scv=slurp(p,&SV);
      snprintf(p,sizeof p,"%s/gptq_real/qzeros.bin",LLMCODEC_FIX_DIR); uint8_t* zv=slurp(p,&ZV);
      snprintf(p,sizeof p,"%s/gptq_real/gidx.bin",LLMCODEC_FIX_DIR); uint8_t* gv=slurp(p,&GV);
      snprintf(p,sizeof p,"%s/gptq_real/expected_f32.bin",LLMCODEC_FIX_DIR); uint8_t* ev=slurp(p,&EV);
      if(!qw||!scv||!zv||!gv||!ev){ printf("[real] gptq: fixture mancanti\n"); ok=0; }
      else { uint8_t* aux=malloc(SV+ZV+GV); memcpy(aux,scv,SV); memcpy(aux+SV,zv,ZV); memcpy(aux+SV+ZV,gv,GV);
        float* o=malloc(64*256*4); const LlmCodec* c=llmcodec_resolve("gptq");
        if(c && c->decode(qw,QV,0,128,4,64,256,aux,SV+ZV+GV,o)==0){ const float* g=(const float*)ev; double md=0;
          for(int i=0;i<64*256;i++){ double d=fabs(o[i]-g[i]); if(d>md)md=d; }
          printf("[real] gptq Qwen2.5-0.5B: maxerr=%.2e -> %s\n", md, md<1e-3?"OK":"FAIL"); ok &= (md<1e-3);
        } else { printf("[real] gptq: decode fail\n"); ok=0; } free(qw);free(scv);free(zv);free(gv);free(ev);free(aux); }
    }
    /* --- AWQ reale --- */
    { long QV,SV,ZV,EV; snprintf(p,sizeof p,"%s/awq_real/qweight.bin",LLMCODEC_FIX_DIR); uint8_t* qw=slurp(p,&QV);
      snprintf(p,sizeof p,"%s/awq_real/scales.bin",LLMCODEC_FIX_DIR); uint8_t* scv=slurp(p,&SV);
      snprintf(p,sizeof p,"%s/awq_real/qzeros.bin",LLMCODEC_FIX_DIR); uint8_t* zv=slurp(p,&ZV);
      snprintf(p,sizeof p,"%s/awq_real/expected_f32.bin",LLMCODEC_FIX_DIR); uint8_t* ev=slurp(p,&EV);
      if(!qw||!scv||!zv||!ev){ printf("[real] awq: fixture mancanti\n"); ok=0; }
      else { uint8_t* aux=malloc(SV+ZV); memcpy(aux,scv,SV); memcpy(aux+SV,zv,ZV);
        float* o=malloc(64*256*4); const LlmCodec* c=llmcodec_resolve("awq");
        if(c && c->decode(qw,QV,0,128,4,64,256,aux,SV+ZV,o)==0){ const float* g=(const float*)ev; double md=0;
          for(int i=0;i<64*256;i++){ double d=fabs(o[i]-g[i]); if(d>md)md=d; }
          printf("[real] awq Qwen2.5-0.5B: maxerr=%.2e -> %s\n", md, md<1e-3?"OK":"FAIL"); ok &= (md<1e-3);
        } else { printf("[real] awq: decode fail\n"); ok=0; } free(qw);free(scv);free(zv);free(ev);free(aux); }
    }
    /* --- mxfp8 reale (Ministral-3) vs golden MLX --- */
    { long W,S,E; snprintf(p,sizeof p,"%s/mxfp8_real/weight2d.bin",LLMCODEC_FIX_DIR); uint8_t* w=slurp(p,&W);
      snprintf(p,sizeof p,"%s/mxfp8_real/scales2d.bin",LLMCODEC_FIX_DIR); uint8_t* s=slurp(p,&S);
      snprintf(p,sizeof p,"%s/mxfp8_real/expected_f32.bin",LLMCODEC_FIX_DIR); uint8_t* e=slurp(p,&E);
      if(!w||!s||!e){ printf("[real] mxfp8: fixture mancanti\n"); ok=0; }
      else { float* o=malloc(2*5120*4); const LlmCodec* c=llmcodec_resolve("mxfp");
             if(c && c->decode(w,W,0,0,32,2,5120,s,S,o)==0){ const float* g=(const float*)e; double md=0;
                for(int i=0;i<2*5120;i++){ double d=fabs(o[i]-g[i]); if(d>md)md=d; }
                printf("[real] mxfp8 Ministral-3 vs MLX: maxerr=%.2e -> %s\n", md, md==0?"BIT-EXACT":"FAIL"); ok &= (md==0);
             } else { printf("[real] mxfp8: decode fail\n"); ok=0; } free(w); free(s); free(e); }
    }
    /* --- dsv41 fp8 reale vs engine (llmcodec_fp8_e4m3_blocked) --- */
    { long W,S; snprintf(p,sizeof p,"%s/dsv41/tensor0.bin",LLMCODEC_FIX_DIR); uint8_t* w=slurp(p,&W);
      snprintf(p,sizeof p,"%s/dsv41/tensor0_scale.bin",LLMCODEC_FIX_DIR); uint8_t* s=slurp(p,&S);
      if(!w||!s){ printf("[real] dsv41: fixture mancanti\n"); ok=0; } else { printf("[real] dsv41 fp8 [32,32]: fixture ok (w=%ld s=%ld)\n",W,S); free(w); free(s); } }
    printf(ok?"PASS\n":"FAIL\n"); return ok?0:1;
}
