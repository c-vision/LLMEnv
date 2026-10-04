/* llm_dsv4_quant_test.c — fp8 KV e QAT indexer: proprieta' del round-trip. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "llm_dsv4_quant.h"
int main(void){
    int fails=0;
    /* fp8: la parte RoPE (ultimi n_rot) resta invariata; la nope cambia ma resta vicina */
    { int hd=512, nr=64; float x[512], y[512];
      for(int i=0;i<hd;i++){ x[i]=0.4f*sinf((float)i*0.37f)-0.1f; y[i]=x[i]; }
      llm_dsv4_fp8_kv_quantize_row(y,hd,nr);
      int bad=0; for(int i=hd-nr;i<hd;i++) if(y[i]!=x[i]) bad++;   /* RoPE intatta */
      double err=0, mag=0; for(int i=0;i<hd-nr;i++){ err+=(y[i]-x[i])*(y[i]-x[i]); mag+=x[i]*x[i]; }
      double rel=sqrt(err/(mag+1e-12));
      printf("%s %-40s (rel=%.3e)\n", (bad||rel>0.2)?"FAIL":"PASS", "fp8 KV: RoPE intatta + nope vicina", rel);
      if(bad||rel>0.2) fails++; }
    /* QAT: deterministico e bounded */
    { float x[128], y[128]; for(int i=0;i<128;i++){ x[i]=0.5f*sinf((float)i*0.21f); y[i]=x[i]; }
      llm_dsv4_indexer_qat_row(y,128);
      float y2[128]; memcpy(y2,x,sizeof x); llm_dsv4_indexer_qat_row(y2,128);
      int det=1; for(int i=0;i<128;i++) if(y[i]!=y2[i]) det=0;
      double mag=0, mo=0; for(int i=0;i<128;i++){ mag+=x[i]*x[i]; mo+=y[i]*y[i]; }
      double ratio=sqrt(mo/(mag+1e-12));
      printf("%s %-40s (ratio=%.3f)\n", (det&&ratio>0.5&&ratio<1.5)?"PASS":"FAIL", "QAT indexer: deterministico/bounded", ratio);
      if(!det||ratio<0.5||ratio>1.5) fails++; }
    printf(fails?"FAILURES\n":"ALL PASS\n");
    return fails?1:0;
}
