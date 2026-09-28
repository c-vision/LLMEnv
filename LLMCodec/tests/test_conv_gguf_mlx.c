/* test_conv_gguf_mlx.c — selfcheck del round-trip MLX affine -> GGUF -> MLX affine:
 * quantizza un tensore f32, lo scrive safetensors, converte a GGUF e ritorno, e
 * confronta il risultato col f32 originale. Esercita llmcodec_mlx_to_gguf/llmcodec_gguf_to_mlx. */
#include "llmcodec_safetensors.h"
#include "llmcodec_moe_quant.h"
#include "llmcodec_gguf_mlx.h"
#include "llmcodec_ggml_quants.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
int main(void){
  system("rm -rf /tmp/m1 /tmp/m2 /tmp/t.gguf; mkdir -p /tmp/m1");
  const uint32_t rows=4, cols=256, group=32, bits=4;
  float x[4*256]; for(int i=0;i<4*256;i++) x[i]=sinf(i*0.3f);
  size_t wb=llmcodec_moe_affine_w_bytes(rows,cols,bits), sb=llmcodec_moe_affine_sc_bytes(rows,cols,group);
  uint32_t* w=(uint32_t*)malloc(wb); uint16_t* sc=(uint16_t*)malloc(sb),*bs=(uint16_t*)malloc(sb);
  if (llmcodec_moe_affine_quantize(x,rows,cols,group,bits,w,sc,bs)!=0){printf("quant fail\n");return 1;}
  { int shp[2]={(int)rows,(int)(cols*bits/32)}; LlmCodecStTensor t[1];
    t[0].name="w.weight"; t[0].dtype="U32"; t[0].shape=shp; t[0].ndim=2; t[0].data=w;
    int shp2[2]={(int)rows,(int)(cols/group)};
    LlmCodecStTensor t3[3]; t3[0]=t[0];
    t3[1].name="w.scales"; t3[1].dtype="F16"; t3[1].shape=shp2; t3[1].ndim=2; t3[1].data=sc;
    t3[2].name="w.biases"; t3[2].dtype="F16"; t3[2].shape=shp2; t3[2].ndim=2; t3[2].data=bs;
    if(llmcodec_safetensors_write("/tmp/m1/model.safetensors",t3,3)!=0){printf("st write fail\n");return 1;} }
  { FILE* f=fopen("/tmp/m1/config.json","wb"); if(f){fprintf(f,"{}");fclose(f);} }
  if (llmcodec_mlx_to_gguf("/tmp/m1","/tmp/t.gguf",12 /*Q4_K*/,group,bits)!=0){printf("mlx->gguf fail\n");return 1;}
  if (llmcodec_gguf_to_mlx("/tmp/t.gguf","/tmp/m2",group,bits)!=0){printf("gguf->mlx fail\n");return 1;}
  /* leggi m2.weight + scales/biases e confronta col f32 originale */
  LlmCodecStTensorInfo* ti=NULL; int n=0; uint64_t base=0;
  if (llmcodec_safetensors_read("/tmp/m2/model.safetensors",&ti,&n,&base)!=0){printf("read m2 fail\n");return 1;}
  int iw=-1,is=-1,ib=-1; for(int i=0;i<n;i++){ if(!strcmp(ti[i].name,"w.weight"))iw=i; if(!strcmp(ti[i].name,"w.scales"))is=i; if(!strcmp(ti[i].name,"w.biases"))ib=i; }
  printf("m2 tensori=%d (w=%d s=%d b=%d)\n",n,iw,is,ib);
  if(iw<0||is<0||ib<0){printf("FAIL componenti\n");return 1;}
  FILE* f=fopen("/tmp/m2/model.safetensors","rb"); uint8_t *wr,*sr,*br;
  wr=(uint8_t*)malloc(ti[iw].end-ti[iw].begin); fseeko(f,base+ti[iw].begin,SEEK_SET); fread(wr,1,ti[iw].end-ti[iw].begin,f);
  sr=(uint8_t*)malloc(ti[is].end-ti[is].begin); fseeko(f,base+ti[is].begin,SEEK_SET); fread(sr,1,ti[is].end-ti[is].begin,f);
  br=(uint8_t*)malloc(ti[ib].end-ti[ib].begin); fseeko(f,base+ti[ib].begin,SEEK_SET); fread(br,1,ti[ib].end-ti[ib].begin,f); fclose(f);
  float y[4*256]; llmcodec_moe_affine_dequantize((const uint32_t*)wr,(const uint16_t*)sr,(const uint16_t*)br,rows,cols,group,bits,1,y);
  double md=0; for(int i=0;i<4*256;i++){ double e=fabs(y[i]-x[i]); if(e>md)md=e; }
  printf("[conv-e2e] MLX->GGUF(Q4_K)->MLX maxerr=%.4e\n", md);
  printf(md<0.5?"PASS\n":"FAIL\n");
  return md<0.5?0:1;
}
