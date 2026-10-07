/* llm_dsv4_attn2_test.c — verifica llm_dsv4_attn_sink vs softmax diretta. */
#include <stdio.h>
#include <math.h>
#include "llm_dsv4_attn.h"
int main(void){
    const int H=2, D=4, M=5; float q[H*D], kv[M*D], sink[H], out[H*D]; int idx[M];
    for(int i=0;i<H*D;i++) q[i]=0.2f*(float)((i*3)%7)-0.6f;
    for(int i=0;i<M*D;i++) kv[i]=0.3f*(float)((i*5)%9)-1.2f;
    for(int i=0;i<H;i++) sink[i]=0.1f*(float)i;
    idx[0]=0; idx[1]=1; idx[2]=2; idx[3]=-1; idx[4]=4;   /* riga 3 saltata */
    float scale=1.0f/sqrtf((float)D);
    llm_dsv4_attn_sink(q,H,D,kv,idx,M,sink,scale,out);
    int bad=0;
    for(int h=0;h<H;h++){ float sc[M], mx=sink[h];
        for(int m=0;m<M;m++){ if(idx[m]<0){sc[m]=-1e30f;continue;} float d=0; for(int e=0;e<D;e++) d+=q[h*D+e]*kv[m*D+e]; sc[m]=d*scale; if(sc[m]>mx)mx=sc[m]; }
        float s=expf(sink[h]-mx); for(int m=0;m<M;m++) if(idx[m]>=0) s+=expf(sc[m]-mx);
        for(int e=0;e<D;e++){ float a=0; for(int m=0;m<M;m++) if(idx[m]>=0) a+=expf(sc[m]-mx)/s*kv[m*D+e];
            if(fabsf(a-out[h*D+e])>1e-5f) bad++; } }
    printf("%s %-40s\n", bad?"FAIL":"PASS", "attn_sink == softmax diretta");
    return bad?1:0;
}
