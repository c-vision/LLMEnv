/* llm_dsv4_attn_fwd_test.c — confronta llm_dsv4_attn_layer_forward(pos=0) con un riferimento
 * indipendente scritto a loop espliciti (RoPE a pos=0 e' identita'). */
#include <stdio.h>
#include <math.h>
#include "llm_dsv4_attn_fwd.h"

#define EMB 16
#define NH 2
#define HD 8
#define NROT 4
#define NG 2
#define LQ 4
#define LO 4
#define QD (NH*HD)
#define GH (NH/NG)
#define GD (HD*GH)
#define OL (NG*LO)

static float rnd(int i){ return 0.3f*sinf((float)i*1.37f)-0.1f; }

int main(void){
    LlmDsv4Config c; memset(&c,0,sizeof(c));
    c.n_layer=5; c.n_embd=EMB; c.n_head=NH; c.n_head_kv=1; c.n_head_dim=HD; c.n_value_dim=HD;
    c.n_rot=NROT; c.n_out_group=NG; c.n_lora_q=LQ; c.n_lora_o=LO;
    c.n_hash_layer=3; c.n_swa=128;
    c.rms_eps=1e-6f; c.rope_freq_base=10000.0f; c.rope_scale_factor=16.0f;
    c.compress_rope_freq_base=160000.0f; c.rope_orig_ctx=65536;
    c.rope_yarn_beta_fast=32.0f; c.rope_yarn_beta_slow=1.0f;
    int rc=c.n_layer; (void)rc;

    LlmDsv4LayerWeights L; memset(&L,0,sizeof(L));
    float wa_q[LQ*EMB], wa_n[LQ], wq_b[QD*LQ], wkv[HD*EMB], kv_n[HD], sinks[NH], an[EMB], Wa[OL*GD], Wb[EMB*OL];
    for(int i=0;i<LQ*EMB;i++) wa_q[i]=rnd(i);
    for(int i=0;i<LQ;i++) wa_n[i]=0.5f+0.1f*(float)i;
    for(int i=0;i<QD*LQ;i++) wq_b[i]=rnd(i+11);
    for(int i=0;i<HD*EMB;i++) wkv[i]=rnd(i+23);
    for(int i=0;i<HD;i++) kv_n[i]=0.7f+0.05f*(float)i;
    for(int i=0;i<NH;i++) sinks[i]=0.2f*(float)i-0.1f;
    for(int i=0;i<EMB;i++) an[i]=0.9f+0.02f*(float)i;
    for(int i=0;i<OL*GD;i++) Wa[i]=rnd(i+41);
    for(int i=0;i<EMB*OL;i++) Wb[i]=rnd(i+53);
    L.attn_norm=an; L.attn_q_a=wa_q; L.attn_q_a_norm=wa_n; L.attn_q_b=wq_b;
    L.attn_kv=wkv; L.attn_kv_a_norm=kv_n; L.attn_sinks=sinks; L.attn_output_a=Wa; L.attn_output_b=Wb;

    float x[EMB]; for(int i=0;i<EMB;i++) x[i]=rnd(i+7);

    float out1[EMB]; llm_dsv4_attn_layer_forward(&c,&L,x,0,0,NULL,0,out1);

    /* riferimento indipendente a loop espliciti */
    float xn[EMB]; { double ss=0; for(int i=0;i<EMB;i++) ss+=(double)x[i]*x[i];
        float sc=1.0f/sqrtf((float)(ss/EMB)+c.rms_eps); for(int i=0;i<EMB;i++) xn[i]=x[i]*sc*an[i]; }
    float qr[LQ]; for(int o=0;o<LQ;o++){ double a=0; for(int i=0;i<EMB;i++) a+=(double)wa_q[o*EMB+i]*xn[i]; qr[o]=(float)a; }
    float qrn[LQ]; { double ss=0; for(int i=0;i<LQ;i++) ss+=(double)qr[i]*qr[i];
        float sc=1.0f/sqrtf((float)(ss/LQ)+c.rms_eps); for(int i=0;i<LQ;i++) qrn[i]=qr[i]*sc*wa_n[i]; }
    float q[QD]; for(int o=0;o<QD;o++){ double a=0; for(int i=0;i<LQ;i++) a+=(double)wq_b[o*LQ+i]*qrn[i]; q[o]=(float)a; }
    for(int h=0;h<NH;h++){ double ss=0; for(int i=0;i<HD;i++) ss+=(double)q[h*HD+i]*q[h*HD+i];
        float sc=1.0f/sqrtf((float)(ss/HD)+c.rms_eps); for(int i=0;i<HD;i++) q[h*HD+i]*=sc; }
    float kvr[HD]; for(int o=0;o<HD;o++){ double a=0; for(int i=0;i<EMB;i++) a+=(double)wkv[o*EMB+i]*xn[i]; kvr[o]=(float)a; }
    float kvn[HD]; { double ss=0; for(int i=0;i<HD;i++) ss+=(double)kvr[i]*kvr[i];
        float sc=1.0f/sqrtf((float)(ss/HD)+c.rms_eps); for(int i=0;i<HD;i++) kvn[i]=kvr[i]*sc*kv_n[i]; }
    float heads[QD]; for(int h=0;h<NH;h++){ float sc=1.0f/sqrtf((float)HD); double d=0;
        for(int e=0;e<HD;e++) d+=(double)q[h*HD+e]*kvn[e]; float s=(float)d*sc;
        float mx=fmaxf(sinks[h],s); float de=expf(sinks[h]-mx)+expf(s-mx);
        for(int e=0;e<HD;e++) heads[h*HD+e]=expf(s-mx)/de*kvn[e]; }
    float low[OL]; for(int g=0;g<NG;g++) for(int r=0;r<LO;r++){ double a=0;
        for(int i=0;i<GD;i++) a+=(double)Wa[(g*LO+r)*GD+i]*heads[g*GD+i]; low[g*LO+r]=(float)a; }
    float ref[EMB]; for(int o=0;o<EMB;o++){ double a=0; for(int i=0;i<OL;i++) a+=(double)Wb[o*OL+i]*low[i]; ref[o]=(float)a; }

    int bad=0; for(int i=0;i<EMB;i++) if(fabsf(out1[i]-ref[i])>1e-5f){ bad++; printf("  out[%d] %.8f vs %.8f\n",i,out1[i],ref[i]); }
    printf("%s %-40s\n", bad?"FAIL":"PASS", "attn_layer(pos=0) == riferimento indipendente");

    /* pos=1: la parte nope dei q non cambia, la coda si' (rope tail-only) */
    LlmDsv4LayerWeights L2=L; (void)L2;
    float xqd[QD]; /* ricostruisco q senza rope e con rope per confronto usando q_a/q_b */
    /* uso direttamente rope_tail_ext per stimare l'effetto: coda diversa, nope uguale */
    int bad2=0; {
        float a[HD], b[HD]; for(int i=0;i<HD;i++){ a[i]=rnd(i+3); b[i]=a[i]; }
        llm_dsv4_rope_tail_ext_inplace(b,1,HD,NROT,1,0,c.rope_freq_base,1.0f,0.0f,1.0f,
                                       c.rope_yarn_beta_fast,c.rope_yarn_beta_slow,0);
        for(int i=0;i<HD-NROT;i++) if(fabsf(a[i]-b[i])>1e-6f) bad2++;
        int tail_changed=0; for(int i=HD-NROT;i<HD;i++) if(fabsf(a[i]-b[i])>1e-6f) tail_changed++;
        if(!tail_changed) bad2++;
    }
    printf("%s %-40s\n", bad2?"FAIL":"PASS", "rope tail-only (nope invariato, coda ruota)");
    printf((bad||bad2)?"FAILURES\n":"ALL PASS\n");
    return (bad||bad2)?1:0;
}
