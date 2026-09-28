/* llm_dsv4_attn_cached_test.c — smoke attn_cached (cache colma, compresse accumulate) + rope inversa. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "llm_dsv4_cache.h"
static float rnd(int i){ return 0.4f*sinf((float)i*1.3f)-0.15f; }
int main(void){
    int fails=0;
    /* rope: forward poi inverse == identita' */
    { LlmDsv4Config c; memset(&c,0,sizeof(c)); c.n_layer=4; c.n_head_dim=8; c.n_rot=4;
      c.rope_freq_base=10000.0f; c.rope_scale_factor=1.0f; c.compress_rope_freq_base=160000.0f;
      float a[2*8], b[2*8]; for(int i=0;i<16;i++) a[i]=rnd(i);
      memcpy(b,a,sizeof a);
      llm_dsv4_rope_tail_layer_inplace(&c,b,2,8,4,5,0,0);
      llm_dsv4_rope_tail_layer_inplace(&c,b,2,8,4,5,0,1);
      int bad=0; for(int i=0;i<16;i++) if(fabsf(a[i]-b[i])>1e-4f) bad++;
      printf("%s %-40s\n", bad?"FAIL":"PASS", "rope forward+inverse == identita'"); if(bad) fails++; }

    /* attn_cached smoke su layer compresso (il=2, ratio 4) */
    LlmDsv4Config c; memset(&c,0,sizeof(c));
    c.n_layer=4; c.n_embd=8; c.n_head=2; c.n_head_kv=1; c.n_head_dim=8; c.n_rot=4;
    c.n_out_group=2; c.n_lora_q=4; c.n_lora_o=4; c.n_indexer_head=2; c.n_indexer_head_dim=4;
    c.n_indexer_top_k=2; c.n_hc=2; c.rms_eps=1e-6f; c.hc_eps=1e-6f;
    c.rope_freq_base=10000.0f; c.rope_scale_factor=1.0f; c.compress_rope_freq_base=160000.0f;
    int qd=16, hd=8, lq=4, ihd=4, ih=2, idxq=ihd*ih, cw=2*hd, iw=2*ihd, ratio=4;
    float aqa[lq*8], qan[lq], aqb[qd*lq], akv[hd*8], kvan[hd], sink[2], oa[8*8], ob[8*8];
    float ckv[cw*8], cgt[cw*8], cape[ratio*cw], cnm[hd];
    float iqb[idxq*lq], ipr[ih*8], ickv[iw*8], icgt[iw*8], icape[ratio*iw], icnm[ihd];
    for(int i=0;i<lq*8;i++) aqa[i]=rnd(i);
    for(int i=0;i<lq;i++) qan[i]=0.5f+0.1f*(float)i;
    for(int i=0;i<qd*lq;i++) aqb[i]=rnd(i+11);
    for(int i=0;i<hd*8;i++) akv[i]=rnd(i+23);
    for(int i=0;i<hd;i++) kvan[i]=0.7f+0.05f*(float)i;
    for(int i=0;i<2;i++) sink[i]=0.1f*(float)i;
    for(int i=0;i<64;i++){ oa[i]=rnd(i+41); ob[i]=rnd(i+53); }
    for(int i=0;i<cw*8;i++){ ckv[i]=rnd(i+61); cgt[i]=rnd(i+71); }
    for(int i=0;i<ratio*cw;i++) cape[i]=0.1f*(float)(i%5);
    for(int i=0;i<hd;i++) cnm[i]=0.8f+0.02f*(float)i;
    for(int i=0;i<idxq*lq;i++) iqb[i]=rnd(i+83);
    for(int i=0;i<ih*8;i++) ipr[i]=rnd(i+97);
    for(int i=0;i<iw*8;i++){ ickv[i]=rnd(i+101); icgt[i]=rnd(i+113); }
    for(int i=0;i<ratio*iw;i++) icape[i]=0.1f*(float)(i%3);
    for(int i=0;i<ihd;i++) icnm[i]=0.8f+0.02f*(float)i;
    LlmDsv4LayerWeights L; memset(&L,0,sizeof(L));
    L.attn_q_a=aqa; L.attn_q_a_norm=qan; L.attn_q_b=aqb; L.attn_kv=akv; L.attn_kv_a_norm=kvan;
    L.attn_sinks=sink; L.attn_output_a=oa; L.attn_output_b=ob;
    L.comp_kv=ckv; L.comp_gate=cgt; L.comp_ape=cape; L.comp_norm=cnm;
    L.idx_q_b=iqb; L.idx_proj=ipr; L.idx_c_kv=ickv; L.idx_c_gate=icgt; L.idx_c_ape=icape; L.idx_c_norm=icnm;
    LlmDsv4LayerCache k; llm_dsv4_cache_init(&k,&c,2,16,16);
    int bad=0, want_comp=0;
    for(int pos=0; pos<10; pos++){
        float xn[8]; for(int i=0;i<8;i++) xn[i]=rnd(pos*7+i);
        float out[8]; llm_dsv4_attn_cached(&c,&L,NULL,&k,xn,pos,2,out);
        for(int i=0;i<8;i++) if(!isfinite(out[i])) bad++;
        want_comp = (pos+1)/ratio;
        if(k.n_comp != want_comp) bad++;
    }
    printf("%s %-40s (n_comp=%d)\n", bad?"FAIL":"PASS", "attn_cached: finite + compresse accumulate", k.n_comp);
    if(bad) fails++;
    llm_dsv4_cache_free(&k);
    printf(fails?"FAILURES: %d\n":"ALL PASS\n", fails);
    return fails?1:0;
}
