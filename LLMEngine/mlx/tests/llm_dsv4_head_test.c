/* llm_dsv4_head_test.c — testa finale DS4 vs riferimento indipendente. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "llm_dsv4_head.h"
#define EMB 8
#define HC 2
#define NV 6
static float rnd(int i){ return 0.3f*sinf((float)i*1.23f)-0.1f; }
int main(void){
    LlmDsv4Config c; memset(&c,0,sizeof(c));
    c.n_embd=EMB; c.n_hc=HC; c.n_vocab=NV; c.rms_eps=1e-6f; c.hc_eps=1e-6f;
    float ohf[HC*(HC*EMB)], ohb[HC], ohs[1]={0.7f}, on[EMB], onorm[EMB], ouf[NV*EMB], hc[HC*EMB];
    for(int i=0;i<HC*(HC*EMB);i++) ohf[i]=rnd(i+3);
    for(int i=0;i<HC;i++) ohb[i]=0.1f*(float)i-0.05f;
    for(int i=0;i<EMB;i++){ on[i]=0.9f+0.01f*(float)i; onorm[i]=0.8f+0.02f*(float)i; }
    for(int i=0;i<NV*EMB;i++) ouf[i]=rnd(i+17);
    for(int i=0;i<HC*EMB;i++) hc[i]=rnd(i+29);
    LlmDsv4Weights w; memset(&w,0,sizeof(w));
    w.output_hc_fn=ohf; w.output_hc_base=ohb; w.output_hc_scale=ohs; w.output_norm=onorm; w.output=ouf;
    float logits[NV]; llm_dsv4_output_logits(&c,&w,hc,logits);

    /* riferimento */
    int hcd=HC*EMB; float flat[HC*EMB]; { double ss=0; for(int i=0;i<hcd;i++) ss+=(double)hc[i]*hc[i];
        float sc=1.0f/sqrtf((float)(ss/hcd)+c.rms_eps); for(int i=0;i<hcd;i++) flat[i]=hc[i]*sc; }
    float pre[HC]; for(int o=0;o<HC;o++){ double a=0; for(int i=0;i<hcd;i++) a+=(double)ohf[o*hcd+i]*flat[i]; pre[o]=(float)a; }
    float wt[HC]; for(int i=0;i<HC;i++){ float z=pre[i]*ohs[0]+ohb[i]; float s=z>=0?1.0f/(1.0f+expf(-z)):expf(z)/(1.0f+expf(z)); wt[i]=s+c.hc_eps; }
    float emb[EMB]; for(int d=0;d<EMB;d++){ double a=0; for(int h=0;h<HC;h++) a+=(double)wt[h]*hc[h*EMB+d]; emb[d]=(float)a; }
    float nrm[EMB]; { double ss=0; for(int i=0;i<EMB;i++) ss+=(double)emb[i]*emb[i];
        float sc=1.0f/sqrtf((float)(ss/EMB)+c.rms_eps); for(int i=0;i<EMB;i++) nrm[i]=emb[i]*sc*onorm[i]; }
    float ref[NV]; for(int o=0;o<NV;o++){ double a=0; for(int i=0;i<EMB;i++) a+=(double)ouf[o*EMB+i]*nrm[i]; ref[o]=(float)a; }
    int bad=0; for(int o=0;o<NV;o++) if(fabsf(logits[o]-ref[o])>1e-5f) bad++;
    printf("%s %-40s\n", bad?"FAIL":"PASS", "output_logits == riferimento");
    int am=llm_dsv4_argmax(logits,NV); int amr=0; for(int o=1;o<NV;o++) if(ref[o]>ref[amr]) amr=o;
    printf("%s %-40s (argmax %d)\n", (am==amr)?"PASS":"FAIL", "argmax", am);
    printf((bad||am!=amr)?"FAILURES\n":"ALL PASS\n");
    return (bad||am!=amr)?1:0;
}
