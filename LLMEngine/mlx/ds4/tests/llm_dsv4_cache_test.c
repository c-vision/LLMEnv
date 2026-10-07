/* llm_dsv4_cache_test.c — pool compressor, decode (timing + valore), attenzione mixed, indexer top-k. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include "llm_dsv4_cache.h"
static float rnd(int i){ return 0.4f*sinf((float)i*1.7f)-0.15f; }
int main(void){
    LlmDsv4Config c; memset(&c,0,sizeof(c));
    c.n_layer=4; c.n_embd=8; c.n_head=2; c.n_head_dim=8; c.n_rot=4; c.n_lora_q=4; c.n_out_group=2; c.n_lora_o=4;
    c.n_indexer_head=2; c.n_indexer_head_dim=4; c.n_indexer_top_k=2; c.n_hc=2;
    c.rms_eps=1e-6f; c.rope_freq_base=10000.0f; c.rope_scale_factor=1.0f; c.compress_rope_freq_base=160000.0f; c.rope_orig_ctx=0;
    int fails=0;

    /* pool ratio4: solo piano corrente (righe 4..7, colonne [hd,2hd)) */
    { int ratio=4, hd=8, width=2*hd; float sk[8*width], ss[8*width], out[hd];
      memset(sk,0,sizeof sk); for(int i=0;i<8*width;i++) ss[i]=-1e30f;
      for(int r=0;r<4;r++) for(int j=0;j<hd;j++){ ss[(ratio+r)*width+hd+j]=0.1f*(float)((r*3+j)%5); sk[(ratio+r)*width+hd+j]=rnd(r*13+j); }
      llm_dsv4_compressor_pool_state(out,sk,ss,hd,ratio);
      int bad=0;
      for(int j=0;j<hd;j++){ float mx=-1e30f; for(int r=0;r<4;r++){ float s=ss[(ratio+r)*width+hd+j]; if(s>mx)mx=s; }
          float den=0,sum=0; for(int r=0;r<4;r++){ float w=expf(ss[(ratio+r)*width+hd+j]-mx); den+=w; sum+=w*sk[(ratio+r)*width+hd+j]; }
          if(fabsf(sum/den-out[j])>1e-5f) bad++; }
      printf("%s %-44s\n", bad?"FAIL":"PASS", "pool ratio4 (piano corrente)"); if(bad) fails++; }

    /* decode: riga prodotta solo quando (pos+1)%ratio==0 + valore vs pool manuale */
    { int ratio=4, hd=8, width=2*hd, embd=8; float wkv[width*embd], wgate[width*embd], ape[ratio*width], nrm[hd];
      for(int i=0;i<width*embd;i++){ wkv[i]=rnd(i); wgate[i]=rnd(i+50); }
      for(int i=0;i<ratio*width;i++) ape[i]=0.1f*(float)(i%7);
      for(int i=0;i<hd;i++) nrm[i]=0.8f+0.02f*(float)i;
      LlmDsv4LayerCache k; llm_dsv4_cache_init(&k,&c,2,16,16);
      int produced=0, badv=0;
      float kvc[4][width], scc[4][width];
      for(int pos=0;pos<ratio;pos++){
          float x[embd]; for(int i=0;i<embd;i++) x[i]=rnd(pos*11+i);
          float comp[hd];
          int p=llm_dsv4_compressor_decode(&c,comp,wkv,wgate,ape,nrm,x,k.attn_state_kv,k.attn_state_score,embd,hd,ratio,0,pos);
          int want=((pos+1)%ratio)==0;
          if(p!=want) produced++;
          /* shadow riferimento */
          for(int o=0;o<width;o++){ double a=0; for(int i=0;i<embd;i++) a+=(double)wkv[o*embd+i]*x[i]; kvc[pos][o]=(float)a; }
          for(int o=0;o<width;o++){ double a=0; for(int i=0;i<embd;i++) a+=(double)wgate[o*embd+i]*x[i]; scc[pos][o]=(float)a+ape[(pos%ratio)*width+o]; }
          if(p){
              /* a pos==ratio-1 il piano precedente e' -inf -> pool sui 4 correnti (colonna hd+j), rope a comp_pos=0 = identita' */
              float pooled[hd];
              for(int j=0;j<hd;j++){
                  float scv[4]; for(int r=0;r<4;r++) scv[r]=scc[r][hd+j];
                  float mx=scv[0]; for(int r=1;r<4;r++) if(scv[r]>mx) mx=scv[r];
                  float den=0,sum=0; for(int r=0;r<4;r++){ float w=expf(scv[r]-mx); den+=w; sum+=w*kvc[r][hd+j]; }
                  pooled[j]=sum/den;
              }
              double ss=0; for(int j=0;j<hd;j++) ss+=(double)pooled[j]*pooled[j];
              float rms=1.0f/sqrtf((float)(ss/hd)+c.rms_eps);
              for(int j=0;j<hd;j++) if(fabsf(pooled[j]*rms*nrm[j]-comp[j])>1e-4f) badv++;
          }
      }
      printf("%s %-44s\n", (produced||badv)?"FAIL":"PASS", "decode: timing + valore (pos=ratio-1)"); if(produced||badv) fails++;
      llm_dsv4_cache_free(&k); }

    /* attenzione mixed vs softmax diretta */
    { int nh=2, hd=8, nr=3, nc=2; float q[nh*hd], raw[nr*hd], comp[nc*hd], sinks[nh], out[nh*hd]; int allow[2]={1,0};
      for(int i=0;i<nh*hd;i++) q[i]=rnd(i+3);
      for(int i=0;i<nr*hd;i++) raw[i]=rnd(i+21);
      for(int i=0;i<nc*hd;i++) comp[i]=rnd(i+39);
      for(int i=0;i<nh;i++) sinks[i]=0.1f*(float)i;
      llm_dsv4_attention_mixed(out,&c,q,raw,nr,comp,nc,allow,sinks,nh,hd);
      int bad=0;
      for(int h=0;h<nh;h++){ float sc[5]; float mx=sinks[h]; int idx=0;
          for(int r=0;r<nr;r++,idx++){ double d=0; for(int e=0;e<hd;e++) d+=(double)q[h*hd+e]*raw[r*hd+e]; sc[idx]=(float)d/sqrtf((float)hd); if(sc[idx]>mx)mx=sc[idx]; }
          for(int r=0;r<nc;r++,idx++){ if(!allow[r]){sc[idx]=-1e30f;continue;} double d=0; for(int e=0;e<hd;e++) d+=(double)q[h*hd+e]*comp[r*hd+e]; sc[idx]=(float)d/sqrtf((float)hd); if(sc[idx]>mx)mx=sc[idx]; }
          float den=expf(sinks[h]-mx); float acc[8]={0}; idx=0;
          for(int r=0;r<nr;r++,idx++){ float w=expf(sc[idx]-mx); den+=w; for(int e=0;e<hd;e++) acc[e]+=w*raw[r*hd+e]; }
          for(int r=0;r<nc;r++,idx++){ if(sc[idx]<-1e29f)continue; float w=expf(sc[idx]-mx); den+=w; for(int e=0;e<hd;e++) acc[e]+=w*comp[r*hd+e]; }
          for(int e=0;e<hd;e++) if(fabsf(acc[e]/den-out[h*hd+e])>1e-5f) bad++; }
      printf("%s %-44s\n", bad?"FAIL":"PASS", "attention_mixed == softmax diretta"); if(bad) fails++; }

    /* indexer top-k: n_comp=6, k=2 */
    { int nh=2, hd=4, nc=6; float idx_qb[nh*hd*c.n_lora_q], idx_proj[nh*c.n_embd], qrn[c.n_lora_q], cur[c.n_embd], idxc[nc*hd];
      for(int i=0;i<nh*hd*c.n_lora_q;i++) idx_qb[i]=rnd(i+5);
      for(int i=0;i<nh*c.n_embd;i++) idx_proj[i]=rnd(i+61);
      for(int i=0;i<c.n_lora_q;i++) qrn[i]=rnd(i+77);
      for(int i=0;i<c.n_embd;i++) cur[i]=rnd(i+91);
      for(int i=0;i<nc*hd;i++) idxc[i]=rnd(i+111);
      LlmDsv4LayerWeights L; memset(&L,0,sizeof(L)); L.idx_q_b=idx_qb; L.idx_proj=idx_proj;
      int* allow=llm_dsv4_indexer_allowed(&c,&L,cur,qrn,idxc,nc,0,3);
      int cnt=0; for(int i=0;i<nc;i++) cnt+=allow[i];
      /* punteggi indipendenti */
      float q[nh*hd]; for(int o=0;o<nh*hd;o++){ double a=0; for(int i=0;i<c.n_lora_q;i++) a+=(double)idx_qb[o*c.n_lora_q+i]*qrn[i]; q[o]=(float)a; }
      llm_dsv4_rope_tail_layer_inplace(&c,q,nh,hd,c.n_rot,3,0,0);
      float w[nh]; for(int o=0;o<nh;o++){ double a=0; for(int i=0;i<c.n_embd;i++) a+=(double)idx_proj[o*c.n_embd+i]*cur[i]; w[o]=(float)a; }
      float scv[nc]; for(int cc=0;cc<nc;cc++){ float s=0; for(int h=0;h<nh;h++){ double d=0; for(int e=0;e<hd;e++) d+=(double)q[h*hd+e]*idxc[cc*hd+e]; float dot=(float)d; if(dot<0)dot=0; s+=dot*w[h]; } scv[cc]=s; }
      int b1=0,b2=1; for(int i=1;i<nc;i++){ if(scv[i]>scv[b1]){b2=b1;b1=i;} else if(scv[i]>scv[b2]) b2=i; }
      int ok = (cnt==2) && allow[b1] && allow[b2];
      printf("%s %-44s (k=%d)\n", ok?"PASS":"FAIL", "indexer top-k seleziona i 2 migliori (scala assente)", cnt);
      free(allow); if(!ok) fails++; }

    printf(fails?"FAILURES: %d\n":"ALL PASS\n", fails);
    return fails?1:0;
}
