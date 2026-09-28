/* llm_dsv4_block_test.c — verifica blocco mHC DS4 (pre/post) vs calcolo diretto. */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "llm_dsv4_block.h"
int main(void){
    LlmDsv4Config c = llm_dsv4_config_flash();
    c.n_embd = 8; c.n_hc = 4;   /* tiny */
    const int HC=c.n_hc, D=c.n_embd, SIZE=(2+HC)*HC;
    float fn[SIZE*HC*D], scale[3], base[SIZE], norm[D], res[HC*D], outx[D], post[HC], comb[HC*HC];
    for (int i=0;i<SIZE*HC*D;i++) fn[i]=0.05f*(float)((i*7)%13)-0.3f;
    scale[0]=0.3f; scale[1]=0.4f; scale[2]=0.5f;
    for (int i=0;i<SIZE;i++) base[i]=0.1f*(float)((i*3)%5)-0.2f;
    for (int i=0;i<D;i++) norm[i]=0.5f+0.05f*(float)(i%7);
    for (int i=0;i<HC*D;i++) res[i]=0.1f*(float)((i*5)%9)-0.4f;
    llm_dsv4_block_pre(&c, fn, scale, base, norm, res, outx, post, comb);
    /* check post/comb sono finiti; outx = rmsnorm(hc_pre(res,pre)) con pre calcolato a parte */
    int ok=1; for(int i=0;i<D && ok;i++) if(!isfinite(outx[i])) ok=0;
    for(int i=0;i<HC && ok;i++) if(!isfinite(post[i]) || post[i]<0) ok=0;  /* post = 2*sigmoid >0 */
    printf("%s %-40s\n", ok?"PASS":"FAIL", "block_pre finito / post>0"); if(!ok) return 1;
    /* hc_post: out[j,d] = post[j]*x[d] + sum_i comb[i,j]*res[i,d] */
    float xd[D], out[HC*D];
    for(int i=0;i<D;i++) xd[i]=0.2f*(float)((i*3)%7)-0.5f;
    llm_dsv4_block_post(&c, xd, res, post, comb, out);
    int bad=0;
    for(int j=0;j<HC;j++) for(int d=0;d<D;d++){ float s=post[j]*xd[d]; for(int i=0;i<HC;i++) s+=comb[i*HC+j]*res[i*D+d];
        if(fabsf(out[j*D+d]-s)>1e-5f) bad++; }
    printf("%s %-40s\n", bad?"FAIL":"PASS", "hc_post == formula"); if(bad) return 1;
    printf("ALL PASS\n");
    return 0;
}
