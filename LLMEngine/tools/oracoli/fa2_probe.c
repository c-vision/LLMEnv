/* fa2_probe.c — ORACOLO: FA2 tiled (f16) con ONLINE softmax su piu' key-tile + causale.
 * NQ=8 query (posizioni QPOS..QPOS+7), NKFULL=128 chiavi, DK=DV=64, key-tile KT=64.
 * O = softmax(Q K^T/sqrt(DK) con maschera causale) V. MMA per QK^T e PV. */
#include <mlx/c/mlx.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static const char* HDR = "using namespace metal;\n";
static const char* SRC =
  "    constexpr uint NQ=8, DK=64, DV=64, KT=64, NT=2, QPOS=120;\n"
  "    uint tid = thread_position_in_threadgroup.x;\n"
  "    threadgroup half kt[DK*KT];\n"
  "    threadgroup half vt[KT*DV];\n"
  "    threadgroup half zt[NQ*KT];\n"
  "    threadgroup half pt[NQ*KT];\n"
  "    threadgroup half ot[NQ*DV];\n"
  "    threadgroup float oacc[NQ*DV];\n"
  "    threadgroup float mrow[NQ], lrow[NQ], frow[NQ];\n"
  "    threadgroup half zbuf[64];\n"
  "    for (uint i = tid; i < NQ*DV; i += 32u) oacc[i] = 0.0f;\n"
  "    for (uint i = tid; i < 64u; i += 32u) zbuf[i] = (half)0;\n"
  "    for (uint m = tid; m < NQ; m += 32u) { mrow[m] = -INFINITY; lrow[m] = 0.0f; frow[m] = 0.0f; }\n"
  "    /* A = Q fragments [NQ,DK] (indipendenti dalla tile) */\n"
  "    simdgroup_half8x8 A[DK/8];\n"
  "    for (uint k = 0; k < DK/8; ++k) simdgroup_load(A[k], q + k*8u, DK);\n"
  "    for (uint tl = 0; tl < NT; ++tl) {\n"
  "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  "        /* carica e traspone K tile -> kt[DK,KT]; carica V tile */\n"
  "        for (uint i = tid; i < DK*KT; i += 32u) kt[(i%DK)*KT + (i/DK)] = kk[tl*KT*DK + i];\n"
  "        for (uint i = tid; i < KT*DV; i += 32u) vt[i] = vv[tl*KT*DV + i];\n"
  "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  "        /* QK^T: zt[NQ,KT] = Q @ Kt */\n"
  "        for (uint nb = 0; nb < KT/8; ++nb) {\n"
  "            simdgroup_half8x8 acc; simdgroup_load(acc, zbuf, 8u);\n"
  "            for (uint k = 0; k < DK/8; ++k) {\n"
  "                simdgroup_half8x8 Bk; simdgroup_load(Bk, kt + k*8u*KT + nb*8u, KT);\n"
  "                simdgroup_multiply_accumulate(acc, A[k], Bk, acc);\n"
  "            }\n"
  "            simdgroup_store(acc, zt + nb*8u, KT);\n"
  "        }\n"
  "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  "        /* online softmax per-riga con maschera causale (pos. query = QPOS+m) */\n"
  "        if (tid < NQ) {\n"
  "            uint m = tid; int qpos = (int)QPOS + (int)m;\n"
  "            float mx = -INFINITY;\n"
  "            for (uint j = 0; j < KT; ++j) { int kn = (int)(tl*KT + j);\n"
  "                if (kn > qpos) continue;\n"
  "                mx = max(mx, (float)zt[m*KT+j] * rsqrt((float)DK)); }\n"
  "            float mnew = max(mrow[m], mx);\n"
  "            float f = exp(mrow[m] - mnew);\n"
  "            float s = 0.0f;\n"
  "            for (uint j = 0; j < KT; ++j) { int kn = (int)(tl*KT + j); float e = 0.0f;\n"
  "                if (kn <= qpos) e = exp((float)zt[m*KT+j] * rsqrt((float)DK) - mnew);\n"
  "                pt[m*KT+j] = (half)e; s += e; }\n"
  "            frow[m] = f; lrow[m] = lrow[m]*f + s; mrow[m] = mnew;\n"
  "        }\n"
  "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  "        /* PV: ot[NQ,DV] = P @ V */\n"
  "        simdgroup_half8x8 Pa[KT/8];\n"
  "        for (uint k = 0; k < KT/8; ++k) simdgroup_load(Pa[k], pt + k*8u, KT);\n"
  "        for (uint nb = 0; nb < DV/8; ++nb) {\n"
  "            simdgroup_half8x8 acc; simdgroup_load(acc, zbuf, 8u);\n"
  "            for (uint k = 0; k < KT/8; ++k) {\n"
  "                simdgroup_half8x8 Bk; simdgroup_load(Bk, vt + (k*8u)*DV + nb*8u, DV);\n"
  "                simdgroup_multiply_accumulate(acc, Pa[k], Bk, acc);\n"
  "            }\n"
  "            simdgroup_store(acc, ot + nb*8u, DV);\n"
  "        }\n"
  "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  "        /* rescale-add: oacc = oacc*frow + ot */\n"
  "        for (uint i = tid; i < NQ*DV; i += 32u) oacc[i] = oacc[i] * frow[i/DV] + (float)ot[i];\n"
  "    }\n"
  "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  "    for (uint i = tid; i < NQ*DV; i += 32u) out[i] = (half)(oacc[i] / lrow[i/DV]);\n";

static float h2f(uint16_t h){ unsigned e=(h>>10)&0x1f,m=h&0x3ff; int sg=h>>15; if(e==0)return 0; if(e==0x1f)return sg?-INFINITY:INFINITY; float v=1.0f+m/1024.0f; return sg?-ldexpf(v,e-15):ldexpf(v,e-15); }

int main(void){
  mlx_device dev=mlx_device_new_type(MLX_GPU,0); mlx_set_default_device(dev);
  mlx_stream s=mlx_default_gpu_stream_new();
  const int NQ=8,NKF=128,DK=64,DV=64,QPOS=120;
  float* Q=(float*)malloc(NQ*DK*4); float* K=(float*)malloc(NKF*DK*4); float* V=(float*)malloc(NKF*DV*4);
  for(int i=0;i<NQ*DK;i++) Q[i]=sinf(i*0.017f)*0.5f;
  for(int i=0;i<NKF*DK;i++) K[i]=cosf(i*0.011f)*0.5f;
  for(int i=0;i<NKF*DV;i++) V[i]=sinf(i*0.023f)*0.5f;
  mlx_array q32=mlx_array_new_data(Q,(int[]){NQ,DK},2,MLX_FLOAT32);
  mlx_array k32=mlx_array_new_data(K,(int[]){NKF,DK},2,MLX_FLOAT32);
  mlx_array v32=mlx_array_new_data(V,(int[]){NKF,DV},2,MLX_FLOAT32);
  mlx_array q=mlx_array_new(),k=mlx_array_new(),v=mlx_array_new();
  mlx_astype(&q,q32,MLX_FLOAT16,s); mlx_astype(&k,k32,MLX_FLOAT16,s); mlx_astype(&v,v32,MLX_FLOAT16,s);
  mlx_array_eval(q); mlx_array_eval(k); mlx_array_eval(v);

  mlx_vector_string in=mlx_vector_string_new();
  mlx_vector_string_append_value(in,"q"); mlx_vector_string_append_value(in,"kk"); mlx_vector_string_append_value(in,"vv");
  mlx_vector_string on=mlx_vector_string_new(); mlx_vector_string_append_value(on,"out");
  mlx_fast_metal_kernel kern=mlx_fast_metal_kernel_new("fa2", in, on, SRC, HDR, true, false);
  if(kern.ctx==NULL){ printf("kernel build FAIL\n"); return 1; }
  mlx_fast_metal_kernel_config cfg=mlx_fast_metal_kernel_config_new();
  mlx_fast_metal_kernel_config_add_output_arg(cfg,(int[]){NQ,DV},2,MLX_FLOAT16);
  mlx_fast_metal_kernel_config_set_grid(cfg,32,1,1);
  mlx_fast_metal_kernel_config_set_thread_group(cfg,32,1,1);
  mlx_vector_array ins=mlx_vector_array_new();
  mlx_vector_array_append_value(ins,q); mlx_vector_array_append_value(ins,k); mlx_vector_array_append_value(ins,v);
  mlx_vector_array outs=mlx_vector_array_new();
  int rc=mlx_fast_metal_kernel_apply(&outs,kern,ins,cfg,s);
  if(rc!=0){ printf("apply FAIL rc=%d\n",rc); return 1; }
  mlx_array O=mlx_array_new(); mlx_vector_array_get(&O,outs,0); mlx_array_eval(O);
  const uint16_t* Op=mlx_array_data_uint16(O);

  double maxd=0; int bad=0;
  for(int m=0;m<NQ;m++){
    int qpos=QPOS+m; float sc[128]; float mx=-1e30f;
    for(int n=0;n<NKF;n++){ float a=0; for(int d=0;d<DK;d++) a+=Q[m*DK+d]*K[n*DK+d];
      sc[n]=(n<=qpos)? a/sqrtf((float)DK) : -1e30f; if(sc[n]>mx)mx=sc[n]; }
    float sum=0; for(int n=0;n<NKF;n++){sc[n]=expf(sc[n]-mx); sum+=sc[n];}
    for(int dv=0;dv<DV;dv++){ float o=0; for(int n=0;n<NKF;n++) o+=sc[n]*V[n*DV+dv]; o/=sum;
      float got=h2f(Op[m*DV+dv]); double df=fabs((double)got-(double)o); if(df>maxd)maxd=df; if(df>0.01)bad++; }
  }
  printf("FA2 (f16, online+causale): bad=%d/%d maxdiff=%.4g -> %s\n", bad, NQ*DV, maxd, bad?"FAIL":"OK");
  return 0;
}
