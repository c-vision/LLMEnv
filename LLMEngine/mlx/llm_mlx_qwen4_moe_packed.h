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
    const size_t rootlen = strlen(m->model_dir);
    FILE* f = fopen(ip, "rb"); if (!f) return 0;
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    if (n<=0) { fclose(f); return 0; }
    char* j=(char*)malloc((size_t)n+1); if(!j){fclose(f); return 0;}
    if (fread(j,1,(size_t)n,f)!=(size_t)n){free(j);fclose(f);return 0;} fclose(f); j[n]=0;
    char key[700]; snprintf(key,sizeof key,"%s.weight",base);
    char* p = strstr(j, key);
    if (p) { char* q = strstr(p, ".safetensors"); if (q) { char* b=q; while (b>j && b[-1]!='"') --b; size_t len=(size_t)(q+12-b); if (rootlen+1+len<cap){memcpy(path,m->model_dir,rootlen);path[rootlen]='/';memcpy(path+rootlen+1,b,len);path[rootlen+1+len]=0;} } }
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
        llm_gemvq4_set_scale_bf16(g, w3.scale_bf16);
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
#undef Q4J
static void q4mp_release_layer(LlmMlxModel* m, int layer_idx) {
    static const char* suf[] = {
        ".mlp.switch_mlp.gate_proj", ".mlp.switch_mlp.up_proj", ".mlp.switch_mlp.down_proj",
        ".mlp.shared_expert.gate_proj", ".mlp.shared_expert.up_proj", ".mlp.shared_expert.down_proj"
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

/* y[r*n_out+j] = x_row r @ W^T (tensore quantizzato qualsiasi bits/group, letto dal map MLX). 1 = ok. */
static int q4mp_mlx_small(LlmMlxModel* m, const char* base, mlx_array xflat, float* y, int n_out, int nrow) {
    char nm[300]; mlx_array w = mlx_array_new(), sc = mlx_array_new(), b = mlx_array_new();
    mlx_array o = mlx_array_new(), o32 = mlx_array_new(); int ok = 0;
    snprintf(nm, sizeof nm, "%s", base);
    if (multishard_get_triple(&m->weights, nm, &w, &sc, &b)) {
        int g = m->cfg.quant_group_size, bt = m->cfg.quant_bits; char mode[16];
        llm_mlx_quant_derive(w, sc, g, bt, "affine", &g, &bt, mode, sizeof mode);
        if (quantized_linear_cfg(&o, xflat, w, sc, b, g, bt, mode, m->stream) && mlx_astype(&o32, o, MLX_FLOAT32, m->stream) == 0) {
            q4mp_eval(o32);
            const float* d = mlx_array_data_float32(o32);
            if (d && (int)mlx_array_size(o32) == n_out * nrow) { memcpy(y, d, sizeof(float) * (size_t)n_out * (size_t)nrow); ok = 1; }
        }
    }
    mlx_array_free(w); mlx_array_free(sc); mlx_array_free(b); mlx_array_free(o); mlx_array_free(o32);
    return ok;
}

static float q4mp_silu(float v){ return v>=0? v/(1.f+expf(-v)) : v*expf(v)/(1.f+expf(v)); }

/* 0 = calcolato in packed; 1 = non gestito. */
static int qwen4_exp_moe_packed_try(mlx_array* out, LlmMlxModel* m, int layer_idx, mlx_array x) {
    if (!out || !m) return 1;
    if (!llm_mlx_weights_backend_packed() || m->cfg.quant_bits != 4) return 1;
    if (m->cfg.model_type[0] && !strstr(m->cfg.model_type, "qwen4")) return 1;
    const int nd0 = mlx_array_ndim(x);
    if (nd0 != 2 && nd0 != 3) return 1;
    const int* xs = mlx_array_shape(x);
    int nrow0 = 1; for (int d = 0; d < nd0-1; d++) nrow0 *= xs[d];
    mlx_array xflat = mlx_array_new();
    if (nd0 == 3) { int sh2[2] = {nrow0, xs[nd0-1]}; if (mlx_reshape(&xflat, x, sh2, 2, m->stream) != 0) { mlx_array_free(xflat); return 1; } }
    else if (mlx_array_set(&xflat, x) != 0) { mlx_array_free(xflat); return 1; }
    const char* pre = m->cfg.layer_prefix; const int g = m->cfg.quant_group_size;
    char pg[256], pu[256], pd[256], pathg[1600], pathu[1600], pathd[1600];
    char rgt[256];
    #define Q4J(name,tail) snprintf(name,sizeof name,"%s%d.%s", pre, layer_idx, tail)
    Q4J(rgt,"mlp.shared_expert_gate"); Q4J(pg,"mlp.switch_mlp.gate_proj"); Q4J(pu,"mlp.switch_mlp.up_proj"); Q4J(pd,"mlp.switch_mlp.down_proj");
    q4mp_shard_of(m, pg, pathg, sizeof pathg);
    q4mp_shard_of(m, pu, pathu, sizeof pathu); q4mp_shard_of(m, pd, pathd, sizeof pathd);   /* gli shard possono differire per tensore */
    { LlmMlxPackedW3 G; if (llm_mlx_packed_open3d(pathg, pg, 4, g, &G) != 0) return 1;
      int F=(int)G.rows, E=(int)G.cols, NE=(int)G.experts; llm_mlx_packed_close3d(&G);
      int K=m->cfg.top_k; if (F<=0||E<=0||NE<=0||K<=0||K>16) return 1;

      /* router (2D packed) su tutte le righe */
      char rg[256], pathr[1600]; Q4J(rg,"mlp.gate");
      q4mp_shard_of(m, rg, pathr, sizeof pathr);
      float* gates=(float*)malloc((size_t)NE*4*(size_t)nrow0);
      float* xh=(float*)malloc((size_t)E*4*(size_t)nrow0);
      if (!gates || !xh) { free(gates); free(xh); return 1; }
      { mlx_array x32 = mlx_array_new(); if (mlx_astype(&x32, xflat, MLX_FLOAT32, m->stream) != 0) { mlx_array_free(x32); free(gates); free(xh); return 1; } mlx_array_free(xflat); xflat = x32; }
      { mlx_vector_array ev=mlx_vector_array_new(); mlx_vector_array_append_value(ev,xflat); mlx_eval(ev); mlx_vector_array_free(ev); }
      memcpy(xh, mlx_array_data_float32(xflat), (size_t)E*4*(size_t)nrow0);
      /* router e shared_expert_gate sono Q8/gs64 nel checkpoint (config.json, override per-modulo): il kernel packed e' solo Q4,
       * quindi restano sul percorso MLX (tensori minuscoli, NON rilasciati da packed_release_wired). */
      float* sgate = (float*)malloc(sizeof(float) * (size_t)nrow0);
      if (!sgate) { free(gates); free(xh); return 1; }
      if (!q4mp_mlx_small(m, rg, xflat, gates, NE, nrow0) || !q4mp_mlx_small(m, rgt, xflat, sgate, 1, nrow0)) { free(sgate); free(gates); free(xh); return 1; }

      float *gk=(float*)malloc((size_t)F*4), *uk=(float*)malloc((size_t)F*4), *yk=(float*)malloc((size_t)E*4);
      double* acc=(double*)calloc((size_t)E,sizeof(double));
      float* obuf=(float*)malloc((size_t)E*4*(size_t)nrow0);
      int rc = (gk&&uk&&yk&&acc&&obuf) ? 0 : 1;
      for (int r=0;r<nrow0 && !rc;r++) {
          const float* gr = gates + (size_t)r*NE;
          double mx=-1e300; for (int i=0;i<NE;i++) if (gr[i]>mx) mx=gr[i];
          double sum=0; for (int i=0;i<NE;i++) sum+=exp((double)gr[i]-mx);
          double inv=sum>0?1.0/sum:0.0;
          char* used=(char*)calloc((size_t)NE,1); int sel[16]; double wts[16]; double wsum=0;
          for (int k=0;k<K;k++){ int bi=-1; double bp=-1;
            for (int e=0;e<NE;e++){ if(used[e]) continue; double pr=exp((double)gr[e]-mx)*inv; if(pr>bp){bp=pr;bi=e;} }
            if (bi<0){ fprintf(stderr, "[moe-packed] riga %d: router senza esperto selezionabile (NaN nei gates?)\n", r); rc=1; break; }
            used[bi]=1; sel[k]=bi; wts[k]=bp; wsum+=bp; }
          if (!rc && wsum>0) for (int k=0;k<K;k++) wts[k]/=wsum;
          free(used);
          if (rc) break;
          memset(acc, 0, (size_t)E*sizeof(double));
          for (int k=0;k<K && !rc;k++) {
              if (q4mp_gemv(pathg, pg, sel[k], 4, g, xh + (size_t)r*E, E, gk, F) != 0) { rc=1; break; }
              if (q4mp_gemv(pathu, pu, sel[k], 4, g, xh + (size_t)r*E, E, uk, F) != 0) { rc=1; break; }
              for (int f=0;f<F;f++) gk[f]=q4mp_silu(gk[f])*uk[f];
              if (q4mp_gemv(pathd, pd, sel[k], 4, g, gk, F, yk, E) != 0) { rc=1; break; }
              for (int d=0;d<E;d++) acc[d]+=wts[k]*(double)yk[d];
          }
          /* shared expert per riga */
          if (!rc) {
              char sg[256],su[256],sd[256],sgt[256],ps[1600];
              Q4J(sg,"mlp.shared_expert.gate_proj"); Q4J(su,"mlp.shared_expert.up_proj");
              Q4J(sd,"mlp.shared_expert.down_proj");
              float* hg=(float*)malloc((size_t)F*4); float* hu=(float*)malloc((size_t)F*4);
              float sw = sgate[r]; int okp = hg&&hu;
              if (okp) {
                  LlmMlxPackedLinear Lg,Lu,Ld;
                  for (int t=0;t<3 && okp;t++) {
                      const char* bb = t==0?sg:t==1?su:sd;
                      LlmMlxPackedLinear* Lp = t==0?&Lg:t==1?&Lu:&Ld;
                      q4mp_shard_of(m,bb,ps,sizeof ps);
                      if (llm_mlx_linear_packed_open(ps,bb,4,g,Lp)!=0) { okp=0; break; }
                      if (t==2) continue;   /* down_proj: ingresso = attivazione F, eseguito dopo; qui solo aperto */
                      int sh1[2]={1,E}; mlx_array xrow = mlx_array_new_data(xh + (size_t)r*E, sh1, 2, MLX_FLOAT32);
                      mlx_array o=mlx_array_new();
                      if (llm_mlx_linear_packed_run(Lp,xrow,&o,m->stream)!=0) { mlx_array_free(xrow); mlx_array_free(o); llm_mlx_linear_packed_close(Lp); okp=0; break; }
                      q4mp_eval(o);
                      if (t==0) memcpy(hg, mlx_array_data_float32(o), (size_t)F*4);
                      else if (t==1) memcpy(hu, mlx_array_data_float32(o), (size_t)F*4);
                      mlx_array_free(xrow); mlx_array_free(o);
                      if (t != 2) llm_mlx_linear_packed_close(Lp);
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
              free(hg); free(hu); if (!okp) rc=1;
          }
          if (!rc) for (int d=0;d<E;d++) obuf[(size_t)r*E+d]=(float)acc[d];
      }
      if (!rc) {
          int sh[2]={nrow0,E};
          mlx_array flatout = mlx_array_new_data(obuf, sh, 2, MLX_FLOAT32);
          if (nd0 == 3) { int osh[3] = {xs[0], xs[1], E}; rc = mlx_reshape(out, flatout, osh, 3, m->stream); mlx_array_free(flatout); }
          else { *out = flatout; }
      }
      free(obuf); free(acc); free(gk); free(uk); free(yk); free(xh); free(gates); free(sgate); mlx_array_free(xflat);
      return rc;
    }
}

#endif
