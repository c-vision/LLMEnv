/* llm_dsv4_layer_test.c — layer loop DS4: finitezza/determinismo + coerenza refactor attn. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "llm_dsv4_layer.h"

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
static float rnd(int i){ return 0.35f*sinf((float)i*1.11f)-0.15f; }
static int fails=0;

static void fill_layer(LlmDsv4LayerWeights* L, float** pool, int* pn,
        float* an,float* qa,float* qan,float* qb,float* wk,float* kvn,float* sk,float* oa,float* ob,
        float* hcfn,float* hcfs,float* hcfb,float* hffn,float* hffs,float* hffb,
        float* fn,float* gi,float* bias,float* ge,float* ue,float* de,float* gs,float* us,float* ds){
    L->attn_norm=an; L->attn_q_a=qa; L->attn_q_a_norm=qan; L->attn_q_b=qb; L->attn_kv=wk;
    L->attn_kv_a_norm=kvn; L->attn_sinks=sk; L->attn_output_a=oa; L->attn_output_b=ob;
    L->hc_attn_fn=hcfn; L->hc_attn_scale=hcfs; L->hc_attn_base=hcfb;
    L->hc_ffn_fn=hffn; L->hc_ffn_scale=hffs; L->hc_ffn_base=hffb;
    L->ffn_norm=fn; L->ffn_gate_inp=gi; L->exp_probs_b=bias;
    L->ffn_gate_exps=ge; L->ffn_up_exps=ue; L->ffn_down_exps=de;
    L->ffn_gate_shexp=gs; L->ffn_up_shexp=us; L->ffn_down_shexp=ds;
    (void)pool;(void)pn;
}
int main(void){
    LlmDsv4Config c; memset(&c,0,sizeof(c));
    c.n_layer=2; c.n_embd=EMB; c.n_head=NH; c.n_head_kv=1; c.n_head_dim=HD; c.n_value_dim=HD;
    c.n_rot=NROT; c.n_out_group=NG; c.n_lora_q=LQ; c.n_lora_o=LO;
    c.n_expert=NE; c.n_expert_used=NUSED; c.n_ff_exp=FF; c.n_hash_layer=0; c.n_hc=HC;
    c.n_hc_sinkhorn_iter=4; c.rms_eps=1e-6f; c.hc_eps=1e-6f; c.expert_weight_scale=1.5f;
    c.swiglu_clamp_exp=10.0f; c.rope_freq_base=10000.0f; c.rope_scale_factor=16.0f;
    c.compress_rope_freq_base=160000.0f; c.rope_orig_ctx=65536;
    c.rope_yarn_beta_fast=32.0f; c.rope_yarn_beta_slow=1.0f;

    LlmDsv4Weights w; memset(&w,0,sizeof(w)); w.cfg=c;
    LlmDsv4LayerWeights L[2]; memset(L,0,sizeof(L));
    float bufs[2][64][128]; /* unused */
    (void)bufs;
    static float an[2][EMB], qa[2][LQ*EMB], qan[2][LQ], qb[2][QD*LQ], wk[2][HD*EMB], kvn[2][HD], sk[2][NH],
        oa[2][OL*GD], ob[2][EMB*OL], hcfn[2][(2+HC)*HC*(HC*EMB)], hcfs[2][(2+HC)*HC], hcfb[2][(2+HC)*HC],
        hffn[2][(2+HC)*HC*(HC*EMB)], hffs[2][(2+HC)*HC], hffb[2][(2+HC)*HC],
        fn[2][EMB], gi[2][NE*EMB], bias[2][NE], ge[2][NE*FF*EMB], ue[2][NE*FF*EMB], de[2][NE*EMB*FF],
        gs[2][FF*EMB], us[2][FF*EMB], ds[2][EMB*FF];
    for(int il=0; il<2; il++){
        int base=il*1000;
        for(int i=0;i<EMB;i++){ an[il][i]=0.9f+0.01f*(float)i; fn[il][i]=0.9f+0.01f*(float)i; }
        for(int i=0;i<LQ*EMB;i++) qa[il][i]=rnd(base+i);
        for(int i=0;i<LQ;i++) qan[il][i]=0.5f+0.1f*(float)i;
        for(int i=0;i<QD*LQ;i++) qb[il][i]=rnd(base+11+i);
        for(int i=0;i<HD*EMB;i++) wk[il][i]=rnd(base+23+i);
        for(int i=0;i<HD;i++) kvn[il][i]=0.7f+0.05f*(float)i;
        for(int i=0;i<NH;i++) sk[il][i]=0.2f*(float)i-0.1f;
        for(int i=0;i<OL*GD;i++) oa[il][i]=rnd(base+41+i);
        for(int i=0;i<EMB*OL;i++) ob[il][i]=rnd(base+53+i);
        int mix=(2+HC)*HC;
        for(int i=0;i<mix*(HC*EMB);i++){ hcfn[il][i]=rnd(base+61+i); hffn[il][i]=rnd(base+71+i); }
        for(int i=0;i<mix;i++){ hcfs[il][i]=0.3f+0.02f*(float)(i%5); hcfs[il][i+0]=hcfs[il][i]; (void)0;
            hffs[il][i]=0.3f+0.02f*(float)(i%5); }
        for(int i=0;i<mix;i++){ hcfb[il][i]=0.1f*(float)(i%3)-0.1f; hffb[il][i]=0.1f*(float)(i%3)-0.1f; }
        for(int i=0;i<NE*EMB;i++) gi[il][i]=rnd(base+83+i);
        for(int i=0;i<NE;i++) bias[il][i]=0.1f*(float)i-0.15f;
        for(int i=0;i<NE*FF*EMB;i++){ ge[il][i]=rnd(base+97+i); ue[il][i]=rnd(base+127+i); }
        for(int i=0;i<NE*EMB*FF;i++) de[il][i]=rnd(base+157+i);
        for(int i=0;i<FF*EMB;i++){ gs[il][i]=rnd(base+191+i); us[il][i]=rnd(base+211+i); }
        for(int i=0;i<EMB*FF;i++) ds[il][i]=rnd(base+233+i);
        fill_layer(&L[il], NULL,NULL, an[il],qa[il],qan[il],qb[il],wk[il],kvn[il],sk[il],oa[il],ob[il],
                   hcfn[il],hcfs[il],hcfb[il],hffn[il],hffs[il],hffb[il],
                   fn[il],gi[il],bias[il],ge[il],ue[il],de[il],gs[il],us[il],ds[il]);
    }
    w.layers=L;

    float emb[EMB]; for(int i=0;i<EMB;i++) emb[i]=rnd(i+7);
    float hc1[HC*EMB], hc2[HC*EMB];
    llm_dsv4_forward_token(&c,&w,emb,3,2,hc1);
    llm_dsv4_forward_token(&c,&w,emb,3,2,hc2);
    int bad=0; for(int i=0;i<HC*EMB;i++){ if(!isfinite(hc1[i])) bad++; if(fabsf(hc1[i]-hc2[i])>1e-9f) bad++; }
    printf("%s %-40s\n", bad?"FAIL":"PASS", "forward_token finito/deterministico");

    /* coerenza refactor: attn_from_normed == attn_layer_forward */
    float xn[EMB]; llm_dsv4_rms_norm_weight(xn, emb, L[0].attn_norm, EMB, c.rms_eps);
    float a1[EMB], a2[EMB];
    llm_dsv4_attn_from_normed(&c,&L[0],xn,2,0,NULL,0,a1);
    llm_dsv4_attn_layer_forward(&c,&L[0],emb,2,0,NULL,0,a2);
    int bad2=0; for(int i=0;i<EMB;i++) if(fabsf(a1[i]-a2[i])>1e-9f) bad2++;
    printf("%s %-40s\n", bad2?"FAIL":"PASS", "attn_from_normed == attn_layer_forward");
    printf((bad||bad2)?"FAILURES\n":"ALL PASS\n");
    return (bad||bad2)?1:0;
}
