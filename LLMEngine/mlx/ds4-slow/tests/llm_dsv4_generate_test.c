/* llm_dsv4_generate_test.c — smoke end-to-end: modello tiny (2 layer), genera 3 token. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "llm_dsv4_generate.h"
#define EMB 8
#define NH 2
#define HD 8
#define NROT 4
#define NG 2
#define LQ 4
#define LO 4
#define FF 6
#define NE 4
#define NUSED 2
#define HC 2
#define QD (NH*HD)
#define GD (HD*(NH/NG))
#define OL (NG*LO)
#define NV 6
static float rnd(int i){ return 0.35f*sinf((float)i*1.11f)-0.15f; }
typedef struct { float an[EMB],qa[LQ*EMB],qan[LQ],qb[QD*LQ],wk[HD*EMB],kvn[HD],sk[NH],oa[OL*GD],ob[EMB*OL],
    hcfn[(2+HC)*HC*(HC*EMB)],hcfs[(2+HC)*HC],hcfb[(2+HC)*HC],
    hffn[(2+HC)*HC*(HC*EMB)],hffs[(2+HC)*HC],hffb[(2+HC)*HC],
    fn[EMB],gi[NE*EMB],bias[NE],ge[NE*FF*EMB],ue[NE*FF*EMB],de[NE*EMB*FF],gs[FF*EMB],us[FF*EMB],ds[EMB*FF];
} LW;
static LW L[2];
static void fill(int il){ int b=il*1000; LW* p=&L[il];
    for(int i=0;i<EMB;i++){ p->an[i]=0.9f+0.01f*(float)i; p->fn[i]=0.9f+0.01f*(float)i; }
    for(int i=0;i<LQ*EMB;i++) p->qa[i]=rnd(b+i);
    for(int i=0;i<LQ;i++) p->qan[i]=0.5f+0.1f*(float)i;
    for(int i=0;i<QD*LQ;i++) p->qb[i]=rnd(b+11+i);
    for(int i=0;i<HD*EMB;i++) p->wk[i]=rnd(b+23+i);
    for(int i=0;i<HD;i++) p->kvn[i]=0.7f+0.05f*(float)i;
    for(int i=0;i<NH;i++) p->sk[i]=0.2f*(float)i-0.1f;
    for(int i=0;i<OL*GD;i++) p->oa[i]=rnd(b+41+i);
    for(int i=0;i<EMB*OL;i++) p->ob[i]=rnd(b+53+i);
    int mix=(2+HC)*HC;
    for(int i=0;i<mix*(HC*EMB);i++){ p->hcfn[i]=rnd(b+61+i); p->hffn[i]=rnd(b+71+i); }
    for(int i=0;i<mix;i++){ p->hcfs[i]=0.3f+0.02f*(float)(i%5); p->hffs[i]=0.3f+0.02f*(float)(i%5);
        p->hcfb[i]=0.1f*(float)(i%3)-0.1f; p->hffb[i]=0.1f*(float)(i%3)-0.1f; }
    for(int i=0;i<NE*EMB;i++) p->gi[i]=rnd(b+83+i);
    for(int i=0;i<NE;i++) p->bias[i]=0.1f*(float)i-0.15f;
    for(int i=0;i<NE*FF*EMB;i++){ p->ge[i]=rnd(b+97+i); p->ue[i]=rnd(b+127+i); }
    for(int i=0;i<NE*EMB*FF;i++) p->de[i]=rnd(b+157+i);
    for(int i=0;i<FF*EMB;i++){ p->gs[i]=rnd(b+191+i); p->us[i]=rnd(b+211+i); }
    for(int i=0;i<EMB*FF;i++) p->ds[i]=rnd(b+233+i);
}
int main(void){
    LlmDsv4Config c; memset(&c,0,sizeof(c));
    c.n_layer=2; c.n_embd=EMB; c.n_head=NH; c.n_head_kv=1; c.n_head_dim=HD; c.n_value_dim=HD;
    c.n_rot=NROT; c.n_out_group=NG; c.n_lora_q=LQ; c.n_lora_o=LO;
    c.n_expert=NE; c.n_expert_used=NUSED; c.n_ff_exp=FF; c.n_hash_layer=0; c.n_hc=HC;
    c.n_vocab=NV; c.n_hc_sinkhorn_iter=4; c.rms_eps=1e-6f; c.hc_eps=1e-6f; c.expert_weight_scale=1.5f;
    c.swiglu_clamp_exp=10.0f; c.rope_freq_base=10000.0f; c.rope_scale_factor=16.0f;
    c.compress_rope_freq_base=160000.0f; c.rope_orig_ctx=65536;
    c.rope_yarn_beta_fast=32.0f; c.rope_yarn_beta_slow=1.0f;

    LlmDsv4LayerWeights lw[2]; memset(lw,0,sizeof lw);
    for(int il=0;il<2;il++){ fill(il); LW* p=&L[il];
        lw[il].attn_norm=p->an; lw[il].attn_q_a=p->qa; lw[il].attn_q_a_norm=p->qan; lw[il].attn_q_b=p->qb;
        lw[il].attn_kv=p->wk; lw[il].attn_kv_a_norm=p->kvn; lw[il].attn_sinks=p->sk;
        lw[il].attn_output_a=p->oa; lw[il].attn_output_b=p->ob;
        lw[il].hc_attn_fn=p->hcfn; lw[il].hc_attn_scale=p->hcfs; lw[il].hc_attn_base=p->hcfb;
        lw[il].hc_ffn_fn=p->hffn; lw[il].hc_ffn_scale=p->hffs; lw[il].hc_ffn_base=p->hffb;
        lw[il].ffn_norm=p->fn; lw[il].ffn_gate_inp=p->gi; lw[il].exp_probs_b=p->bias;
        lw[il].ffn_gate_exps=p->ge; lw[il].ffn_up_exps=p->ue; lw[il].ffn_down_exps=p->de;
        lw[il].ffn_gate_shexp=p->gs; lw[il].ffn_up_shexp=p->us; lw[il].ffn_down_shexp=p->ds; }
    float temb[NV*EMB], onorm[EMB], ouf[NV*EMB], ohf[HC*(HC*EMB)], ohb[HC], ohs[1]={0.7f};
    for(int i=0;i<NV*EMB;i++) temb[i]=rnd(i+301);
    for(int i=0;i<EMB;i++) onorm[i]=0.9f+0.01f*(float)i;
    for(int i=0;i<NV*EMB;i++) ouf[i]=rnd(i+401);
    for(int i=0;i<HC*(HC*EMB);i++) ohf[i]=rnd(i+501);
    for(int i=0;i<HC;i++) ohb[i]=0.1f*(float)i-0.05f;
    LlmDsv4Weights w; memset(&w,0,sizeof(w)); w.cfg=c; w.layers=lw; w.token_embd=temb;
    w.output_norm=onorm; w.output=ouf; w.output_hc_fn=ohf; w.output_hc_base=ohb; w.output_hc_scale=ohs;

    LlmDsv4Caches cs; llm_dsv4_caches_init(&cs,&c,128,256);
    int toks[8]={3,1,4,0,0,0,0,0};
    llm_dsv4_generate(&c,&w,&cs,toks,3,3,NULL);
    int bad=0; for(int i=0;i<6;i++){ if(toks[i]<0||toks[i]>=NV) bad++; }
    printf("%s %-40s [%d %d %d -> %d %d %d]\n", bad?"FAIL":"PASS", "generate greedy (range)",
           toks[0],toks[1],toks[2],toks[3],toks[4],toks[5]);
    /* determinismo */
    LlmDsv4Caches cs2; llm_dsv4_caches_init(&cs2,&c,128,256);
    int t2[8]={3,1,4,0,0,0,0,0};
    llm_dsv4_generate(&c,&w,&cs2,t2,3,3,NULL);
    int bad2=0; for(int i=0;i<6;i++) if(toks[i]!=t2[i]) bad2++;
    printf("%s %-40s\n", bad2?"FAIL":"PASS", "generate deterministico");
    llm_dsv4_caches_free(&cs); llm_dsv4_caches_free(&cs2);
    printf((bad||bad2)?"FAILURES\n":"ALL PASS\n");
    return (bad||bad2)?1:0;
}
