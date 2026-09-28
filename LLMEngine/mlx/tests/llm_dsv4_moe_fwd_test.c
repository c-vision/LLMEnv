/* llm_dsv4_moe_fwd_test.c — MoE DS4 vs riferimento indipendente (top-k e hash). */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "llm_dsv4_moe_fwd.h"

#define EMB 8
#define NE 4
#define NUSED 2
#define FF 6
static float rnd(int i){ return 0.4f*sinf((float)i*0.91f)-0.2f; }
static float sp(float x){ if(x>20)return x; if(x<-20)return expf(x); return log1pf(expf(x)); }
static float sg(float x){ return x>=0?1.0f/(1.0f+expf(-x)):expf(x)/(1.0f+expf(x)); }

int main(void){
    LlmDsv4Config c; memset(&c,0,sizeof(c));
    c.n_embd=EMB; c.n_expert=NE; c.n_expert_used=NUSED; c.n_ff_exp=FF; c.n_hash_layer=1;
    c.expert_weight_scale=1.5f; c.swiglu_clamp_exp=10.0f;
    float gi[NE*EMB], bias[NE], ge[NE*FF*EMB], ue[NE*FF*EMB], de[NE*EMB*FF];
    float gs[FF*EMB], us[FF*EMB], ds[EMB*FF], tid[NUSED*4], x[EMB];
    for(int i=0;i<NE*EMB;i++) gi[i]=rnd(i);
    for(int i=0;i<NE;i++) bias[i]=0.1f*(float)i-0.15f;
    for(int i=0;i<NE*FF*EMB;i++){ ge[i]=rnd(i+31); ue[i]=rnd(i+77); }
    for(int i=0;i<NE*EMB*FF;i++) de[i]=rnd(i+113);
    for(int i=0;i<FF*EMB;i++){ gs[i]=rnd(i+201); us[i]=rnd(i+233); }
    for(int i=0;i<EMB*FF;i++) ds[i]=rnd(i+271);
    tid[0]=1; tid[1]=3; tid[2]=2; tid[3]=0;   /* riga token 0; token 1 -> riga 2 */
    for(int i=0;i<EMB;i++) x[i]=rnd(i+9);
    LlmDsv4LayerWeights L; memset(&L,0,sizeof(L));
    L.ffn_gate_inp=gi; L.exp_probs_b=bias; L.ffn_gate_exps=ge; L.ffn_up_exps=ue; L.ffn_down_exps=de;
    L.ffn_gate_shexp=gs; L.ffn_up_shexp=us; L.ffn_down_shexp=ds; L.ffn_gate_tid2eid=tid;

    /* -- top-k layer (il=1) -- */
    float out[EMB]; llm_dsv4_moe_forward(&c,&L,x,1,0,out);
    float probs[NE]; for(int e=0;e<NE;e++){ double a=0; for(int i=0;i<EMB;i++) a+=(double)gi[e*EMB+i]*x[i]; probs[e]=sqrtf(sp((float)a)); }
    float sel[NE]; for(int e=0;e<NE;e++) sel[e]=probs[e]+bias[e];
    int idx[NUSED]; float sv[NE]; memcpy(sv,sel,sizeof sv);
    for(int e=0;e<NE;e++){ int best=e; for(int f=e+1;f<NE;f++) if(sv[f]>sv[best]) best=f; float t=sv[e];sv[e]=sv[best];sv[best]=t; }
    /* topk: primo=indice di max */
    int first=0; for(int e=1;e<NE;e++) if(sel[e]>sel[first]) first=e;
    int second=0; for(int e=1;e<NE;e++) if(e!=first && (second==first||sel[e]>sel[second])) second=e;
    idx[0]=first; idx[1]=second;
    float w[NUSED]; float sm=0; for(int i=0;i<NUSED;i++){ w[i]=probs[idx[i]]; sm+=w[i]; } if(sm<6.103515625e-5f)sm=6.103515625e-5f;
    for(int i=0;i<NUSED;i++) w[i]=w[i]/sm*c.expert_weight_scale;
    float ref[EMB]; for(int o=0;o<EMB;o++) ref[o]=0;
    for(int i=0;i<NUSED;i++){ int e=idx[i]; float g[FF],u[FF],m[FF];
        for(int o=0;o<FF;o++){ double a=0; for(int j=0;j<EMB;j++) a+=(double)ge[(e*FF+o)*EMB+j]*x[j]; g[o]=(float)a; }
        for(int o=0;o<FF;o++){ double a=0; for(int j=0;j<EMB;j++) a+=(double)ue[(e*FF+o)*EMB+j]*x[j]; u[o]=(float)a; }
        for(int j=0;j<FF;j++){ float gg=g[j],uu=u[j]; if(gg>10)gg=10; if(uu>10)uu=10; if(uu<-10)uu=-10; m[j]=(gg*sg(gg))*uu; }
        for(int o=0;o<EMB;o++){ double a=0; for(int j=0;j<FF;j++) a+=(double)de[(e*EMB+o)*FF+j]*m[j]; ref[o]+=w[i]*(float)a; } }
    { float g[FF],u[FF],m[FF];
      for(int o=0;o<FF;o++){ double a=0; for(int j=0;j<EMB;j++) a+=(double)gs[o*EMB+j]*x[j]; g[o]=(float)a; }
      for(int o=0;o<FF;o++){ double a=0; for(int j=0;j<EMB;j++) a+=(double)us[o*EMB+j]*x[j]; u[o]=(float)a; }
      for(int j=0;j<FF;j++){ float gg=g[j],uu=u[j]; if(gg>10)gg=10; if(uu>10)uu=10; if(uu<-10)uu=-10; m[j]=(gg*sg(gg))*uu; }
      for(int o=0;o<EMB;o++){ double a=0; for(int j=0;j<FF;j++) a+=(double)ds[o*FF+j]*m[j]; ref[o]+=(float)a; } }
    int bad=0; for(int o=0;o<EMB;o++) if(fabsf(out[o]-ref[o])>2e-5f){bad++; printf("  topk out[%d] %.8f vs %.8f\n",o,out[o],ref[o]);}
    printf("%s %-40s\n", bad?"FAIL":"PASS", "moe top-k == riferimento");

    /* -- hash layer (il=0), token 1 -> selected = riga 1 (tid[2],tid[3]) -- */
    float outh[EMB]; llm_dsv4_moe_forward(&c,&L,x,0,1,outh);
    int hs[2]={ (int)tid[2], (int)tid[3] };
    float hw[2]; float hsm=0; for(int i=0;i<2;i++){ hw[i]=probs[hs[i]]; hsm+=hw[i]; } if(hsm<6.103515625e-5f)hsm=6.103515625e-5f;
    for(int i=0;i<2;i++) hw[i]=hw[i]/hsm*c.expert_weight_scale;
    float href[EMB]; for(int o=0;o<EMB;o++) href[o]=0;
    for(int i=0;i<2;i++){ int e=hs[i]; float g[FF],u[FF],m[FF];
        for(int o=0;o<FF;o++){ double a=0; for(int j=0;j<EMB;j++) a+=(double)ge[(e*FF+o)*EMB+j]*x[j]; g[o]=(float)a; }
        for(int o=0;o<FF;o++){ double a=0; for(int j=0;j<EMB;j++) a+=(double)ue[(e*FF+o)*EMB+j]*x[j]; u[o]=(float)a; }
        for(int j=0;j<FF;j++){ float gg=g[j],uu=u[j]; if(gg>10)gg=10; if(uu>10)uu=10; if(uu<-10)uu=-10; m[j]=(gg*sg(gg))*uu; }
        for(int o=0;o<EMB;o++){ double a=0; for(int j=0;j<FF;j++) a+=(double)de[(e*EMB+o)*FF+j]*m[j]; href[o]+=hw[i]*(float)a; } }
    { float g[FF],u[FF],m[FF];
      for(int o=0;o<FF;o++){ double a=0; for(int j=0;j<EMB;j++) a+=(double)gs[o*EMB+j]*x[j]; g[o]=(float)a; }
      for(int o=0;o<FF;o++){ double a=0; for(int j=0;j<EMB;j++) a+=(double)us[o*EMB+j]*x[j]; u[o]=(float)a; }
      for(int j=0;j<FF;j++){ float gg=g[j],uu=u[j]; if(gg>10)gg=10; if(uu>10)uu=10; if(uu<-10)uu=-10; m[j]=(gg*sg(gg))*uu; }
      for(int o=0;o<EMB;o++){ double a=0; for(int j=0;j<FF;j++) a+=(double)ds[o*FF+j]*m[j]; href[o]+=(float)a; } }
    int bad2=0; for(int o=0;o<EMB;o++) if(fabsf(outh[o]-href[o])>2e-5f){bad2++; printf("  hash out[%d] %.8f vs %.8f\n",o,outh[o],href[o]);}
    printf("%s %-40s\n", bad2?"FAIL":"PASS", "moe hash == riferimento");
    printf((bad||bad2)?"FAILURES\n":"ALL PASS\n");
    return (bad||bad2)?1:0;
}
