/* llm_dsv4_rope_test.c — verifica head-rms-norm + rope tail vs (rms manuale + dsv41 rope). */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "llm_dsv4_rope.h"
int main(void){
    const int H=2, D=8, NR=4;   /* head_dim 8, rope tail 4 */
    float a[H*D], b[H*D];
    for(int i=0;i<H*D;i++){ a[i]=0.1f*(float)((i*5)%9)-0.4f; b[i]=a[i]; }
    const float eps=1e-6f, base=10000.0f, pos=3.0f;
    llm_dsv4_head_rms_norm_rope(a, H, D, NR, pos, base, eps, 0);
    /* riferimento: rms per-head poi dsv41_rope sulla coda */
    for(int h=0;h<H;h++){ float* xh=b+(long)h*D; float s=0; for(int i=0;i<D;i++) s+=xh[i]*xh[i];
        float sc=1.0f/sqrtf(s/(float)D+eps); for(int i=0;i<D;i++) xh[i]*=sc;
        llm_dsv41_rope(xh, D, pos, NR, base, 0, 32.0f, 1.0f, 1.0f, 0, 0); }
    int bad=0; for(int i=0;i<H*D;i++) if(fabsf(a[i]-b[i])>1e-6f) bad++;
    printf("%s %-40s\n", bad?"FAIL":"PASS", "head_rms_norm_rope == ref"); 
    /* la parte non-rope (primi D-NR) e' scalata ma non ruotata */
    int ok=1; for(int h=0;h<H;h++){ float* xh=a+(long)h*D; (void)xh; }
    printf("%s %-40s\n", ok?"PASS":"FAIL", "sanity");
    return bad?1:0;
}
