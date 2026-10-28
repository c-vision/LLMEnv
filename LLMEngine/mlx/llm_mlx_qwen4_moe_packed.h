/* llm_mlx_qwen4_moe_packed.h — MoE qwen4 su pesi PACKED zero-copy (routed + shared).
 * v1: router/top-k e somma in CPU; GEMV per esperto col kernel Q4 del bridge.
 * 0 = gestito, 1 = non gestito (il chiamante usa il percorso MLX). */
#ifndef LLM_MLX_QWEN4_MOE_PACKED_H
#define LLM_MLX_QWEN4_MOE_PACKED_H

#include "llm_mlx_internal.h"
#include "llm_mlx_weights_backend.h"
#include "llm_mlx_wpacked.h"
#include "llm_mlx_linear_packed.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int q4mp_shard_of(const LlmMlxModel* m, const char* base, char* path, size_t cap) {
    char ip[1200]; snprintf(ip, sizeof ip, "%s/model.safetensors.index.json", m->model_dir);
    snprintf(path, cap, "%s/model.safetensors", m->model_dir);
    FILE* f = fopen(ip, "rb"); if (!f) return 0;
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    if (n<=0) { fclose(f); return 0; }
    char* j=(char*)malloc((size_t)n+1); if(!j){fclose(f); return 0;}
    if (fread(j,1,(size_t)n,f)!=(size_t)n){free(j);fclose(f);return 0;} fclose(f); j[n]=0;
    char key[700]; snprintf(key,sizeof key,"%s.weight",base);
    char* p = strstr(j, key);
    if (p) { char* q = strstr(p, ".safetensors"); if (q) { char* b=q; while (b>j && b[-1]!='"') --b; size_t len=(size_t)(q+12-b); if (len<cap){memcpy(path,b,len);path[len]=0;} } }
    free(j); return 0;
}

static void q4mp_eval(mlx_array a) { mlx_vector_array ev=mlx_vector_array_new(); mlx_vector_array_append_value(ev,a); mlx_eval(ev); mlx_vector_array_free(ev); }

/* GEMV di UN esperto: apre la vista 3D dello shard, poi il kernel Q4 sul sotto-tensore. */
static int q4mp_gemv(const char* path, const char* base, int expert, int bits, int group,
                     const float* x, int n_in, float* y, int n_out) {
    LlmMlxPackedW3 w3;
    if (llm_mlx_packed_open3d(path, base, bits, group, &w3) != 0) return -1;
    if (expert < 0 || (uint32_t)expert >= w3.experts || (int)w3.cols != n_in || (int)w3.rows != n_out) { llm_mlx_packed_close3d(&w3); return -1; }
    const uint8_t* wp = (const uint8_t*)w3.w + (size_t)expert*w3.w_stride;
    const uint8_t* sp = (const uint8_t*)w3.scales + (size_t)expert*w3.s_stride;
    const uint8_t* bp = w3.biases ? (const uint8_t*)w3.biases + (size_t)expert*w3.s_stride : NULL;
    int rc = -1;
    LlmGemvQ4Ctx* g = llm_gemvq4_create();
    if (g && llm_gemvq4_load_weights_wonly_nocopy(g, wp, (const uint16_t*)sp, (const uint16_t*)bp, w3.cols, w3.rows, (uint32_t)group) == 0) {
        uint16_t* xh = (uint16_t*)malloc((size_t)n_in*2), *yh = (uint16_t*)malloc((size_t)n_out*2);
        if (xh && yh) {
            for (int i=0;i<n_in;i++){ _Float16 h=(_Float16)x[i]; memcpy(&xh[i],&h,2); }
            if (llm_gemvq4_run(g, xh, yh) == 0) { for (int o=0;o<n_out;o++){ _Float16 h; memcpy(&h,&yh[o],2); y[o]=(float)h; } rc=0; }
        }
        free(xh); free(yh);
    }
    if (g) llm_gemvq4_destroy(g);
    llm_mlx_packed_close3d(&w3);
    return rc;
}

/* Rilascia dal map gli array dei tensori MoE del layer: il forward packed non li usa.
 * Chiamata SOLO dopo un calcolo packed riuscito (mai prima), così il fallback resta sicuro. */
static void q4mp_release_layer(LlmMlxModel* m, int layer_idx) {
    static const char* suf[] = {
        ".mlp.switch_mlp.gate_proj", ".mlp.switch_mlp.up_proj", ".mlp.switch_mlp.down_proj",
        ".mlp.shared_expert.gate_proj", ".mlp.shared_expert.up_proj", ".mlp.shared_expert.down_proj",
        ".mlp.shared_expert_gate"
    };
    static const char* aux[] = {".weight",".scales",".biases"};
    for (size_t b=0;b<sizeof suf/sizeof suf[0];b++)
        for (int a=0;a<3;a++) {
            char nm[300]; snprintf(nm,sizeof nm,"%s.%d%s%s", m->cfg.layer_prefix, layer_idx, suf[b], aux[a]);
            mlx_array e = mlx_array_new();
            for (int si=0; si<m->weights.count; si++) (void)mlx_map_string_to_array_insert(m->weights.shards[si], nm, e);
            mlx_array_free(e);
        }
}

static float q4mp_silu(float v){ return v>=0? v/(1.f+expf(-v)) : v*expf(v)/(1.f+expf(v)); }

/* 0 = calcolato in packed; 1 = non gestito. */
static int qwen4_exp_moe_packed_try(mlx_array* out, LlmMlxModel* m, int layer_idx, mlx_array x) {
    if (!out || !m) return 1;
    if (!llm_mlx_weights_backend_packed() || m->cfg.quant_bits != 4) return 1;
    if (m->cfg.model_type[0] && !strstr(m->cfg.model_type, "qwen4")) return 1;
    const char* pre = m->cfg.layer_prefix; const int g = m->cfg.quant_group_size;
    char pg[256], pu[256], pd[256], pathg[1600];
    snprintf(pg,sizeof pg,"%s.%d.mlp.switch_mlp.gate_proj",pre,layer_idx);
    snprintf(pu,sizeof pu,"%s.%d.mlp.switch_mlp.up_proj",pre,layer_idx);
    snprintf(pd,sizeof pd,"%s.%d.mlp.switch_mlp.down_proj",pre,layer_idx);
    q4mp_shard_of(m, pg, pathg, sizeof pathg);
    { LlmMlxPackedW3 G; if (llm_mlx_packed_open3d(pathg, pg, 4, g, &G) != 0) return 1;
      int F=(int)G.rows, E=(int)G.cols, NE=(int)G.experts; llm_mlx_packed_close3d(&G);
      int K=m->cfg.top_k; if (F<=0||E<=0||NE<=0||K<=0||K>16) return 1;

      /* router (2D packed) */
      char rg[256], pathr[1600]; snprintf(rg,sizeof rg,"%s.%d.mlp.gate",pre,layer_idx);
      q4mp_shard_of(m, rg, pathr, sizeof pathr);
      float* gates=(float*)malloc((size_t)NE*4); float* xh=(float*)malloc((size_t)E*4);
      if (!gates || !xh) { free(gates); free(xh); return 1; }
      { mlx_vector_array ev=mlx_vector_array_new(); mlx_vector_array_append_value(ev,x); mlx_eval(ev); mlx_vector_array_free(ev); }
      memcpy(xh, mlx_array_data_float32(x), (size_t)E*4);
      LlmMlxPackedLinear L;
      if (llm_mlx_linear_packed_open(pathr, rg, 4, g, &L) != 0) { free(gates); free(xh); return 1; }
      mlx_array go = mlx_array_new();
      if (llm_mlx_linear_packed_run(&L, x, &go, m->stream) != 0) { llm_mlx_linear_packed_close(&L); mlx_array_free(go); free(gates); free(xh); return 1; }
      q4mp_eval(go); memcpy(gates, mlx_array_data_float32(go), (size_t)NE*4);
      llm_mlx_linear_packed_close(&L); mlx_array_free(go);

      double mx=-1e300; for (int i=0;i<NE;i++) if (gates[i]>mx) mx=gates[i];
      double sum=0; for (int i=0;i<NE;i++) sum+=exp((double)gates[i]-mx);
      double inv=sum>0?1.0/sum:0.0;
      char* used=(char*)calloc((size_t)NE,1); int sel[16]; double wts[16]; double wsum=0;
      for (int k=0;k<K;k++){ int bi=-1; double bp=-1;
        for (int e=0;e<NE;e++){ if(used[e]) continue; double p=exp((double)gates[e]-mx)*inv; if(p>bp){bp=p;bi=e;} }
        if (bi<0) { free(used); free(gates); free(xh); return 1; }
        used[bi]=1; sel[k]=bi; wts[k]=bp; wsum+=bp; }
      if (wsum>0) for (int k=0;k<K;k++) wts[k]/=wsum;
      free(used); free(gates);

      float *gk=(float*)malloc((size_t)F*4), *uk=(float*)malloc((size_t)F*4), *yk=(float*)malloc((size_t)E*4);
      double* acc=(double*)calloc((size_t)E,sizeof(double));
      int rc=0;
      if (!gk||!uk||!yk||!acc) rc=1;
      for (int k=0;k<K && !rc;k++) {
        if (q4mp_gemv(pathg, pg, sel[k], 4, g, xh, E, gk, F) != 0) { rc=1; break; }
        if (q4mp_gemv(pathg, pu, sel[k], 4, g, xh, E, uk, F) != 0) { rc=1; break; }
        for (int f=0;f<F;f++) gk[f]=q4mp_silu(gk[f])*uk[f];
        if (q4mp_gemv(pathg, pd, sel[k], 4, g, gk, F, yk, E) != 0) { rc=1; break; }
        for (int d=0;d<E;d++) acc[d]+=wts[k]*(double)yk[d];
      }
      /* shared expert */
      if (!rc) {
        char sg[256],su[256],sd[256],sgt[256],ps[1600];
        snprintf(sg,sizeof sg,"%s.%d.mlp.shared_expert.gate_proj",pre,layer_idx);
        snprintf(su,sizeof su,"%s.%d.mlp.shared_expert.up_proj",pre,layer_idx);
        snprintf(sd,sizeof sd,"%s.%d.mlp.shared_expert.down_proj",pre,layer_idx);
        snprintf(sgt,sizeof sgt,"%s.%d.mlp.shared_expert_gate",pre,layer_idx);
        float* hg=(float*)malloc((size_t)F*4); float* hu=(float*)malloc((size_t)F*4); float* hy=(float*)malloc((size_t)E*4);
        float sw=0; int okp = hg&&hu&&hy;
        if (okp) {
          LlmMlxPackedLinear Lg,Lu,Ld,Ls;
          for (int t=0;t<4 && okp;t++) {
            const char* b = t==0?sg:t==1?su:t==2?sd:sgt;
            LlmMlxPackedLinear* Lp = t==0?&Lg:t==1?&Lu:t==2?&Ld:&Ls;
            q4mp_shard_of(m,b,ps,sizeof ps);
            if (llm_mlx_linear_packed_open(ps,b,4,g,Lp)!=0) { okp=0; break; }
            mlx_array o=mlx_array_new();
            if (llm_mlx_linear_packed_run(Lp,x,&o,m->stream)!=0) { mlx_array_free(o); llm_mlx_linear_packed_close(Lp); okp=0; break; }
            q4mp_eval(o);
            if (t==0) memcpy(hg, mlx_array_data_float32(o), (size_t)F*4);
            else if (t==1) memcpy(hu, mlx_array_data_float32(o), (size_t)F*4);
            else if (t==3) sw = mlx_array_data_float32(o)[0];
            else { /* down: non usato qui */ }
            mlx_array_free(o);
            if (t != 2) llm_mlx_linear_packed_close(Lp);   /* Ld resta aperto per il down */
          }
          if (okp) {
            for (int f=0;f<F;f++) hg[f]=q4mp_silu(hg[f])*hu[f];
            int sh[2]={1,F}; mlx_array gsa=mlx_array_new_data(hg, sh, 2, MLX_FLOAT32);
            mlx_array o=mlx_array_new();
            if (llm_mlx_linear_packed_run(&Ld, gsa, &o, m->stream)==0) {
              q4mp_eval(o); const float* yd=mlx_array_data_float32(o);
              float sv = sw>=0? 1.f/(1.f+expf(-sw)) : expf(sw)/(1.f+expf(sw));
              for (int d=0;d<E;d++) acc[d]+=(double)sv*(double)yd[d];
            } else okp=0;
            mlx_array_free(gsa); mlx_array_free(o); llm_mlx_linear_packed_close(&Ld);
          }
        }
        free(hg); free(hu); free(hy); if (!okp) rc=1;
      }
      if (!rc) {
        float* o=(float*)malloc((size_t)E*4);
        if (o) { for (int d=0;d<E;d++) o[d]=(float)acc[d]; int sh[2]={1,E}; *out=mlx_array_new_data(o, sh, 2, MLX_FLOAT32); free(o); }
        else rc=1;
      }
      free(acc); free(gk); free(uk); free(yk); free(xh);
      if (rc == 0) q4mp_release_layer(m, layer_idx);
      return rc;
    }
}

#endif
