/* A1 (GGUF->MLX) con uscita fp8: un ruolo il cui `quant` e' "fp8_block" produce
 * F8_E4M3 (weight) + F8_E8M0 (scale) nel safetensors di uscita. */
#include "llmcodec_safetensors.h"
#include "llmcodec_gguf_mlx.h"
#include "llmcodec_gguf_writer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
int main(void){
  system("rm -rf /tmp/a1i /tmp/a1.gguf /tmp/a1o; mkdir -p /tmp/a1i");
  const int ROWS=64, COLS=64;
  float src[64*64]; for(int i=0;i<ROWS*COLS;i++) src[i]=sinf(i*0.13f)*2.0f;
  int sw[2]={ROWS,COLS};
  LlmCodecStTensor t[1]; t[0].name="layers.0.ffn.weight"; t[0].dtype="F32"; t[0].shape=sw; t[0].ndim=2; t[0].data=src;
  if(llmcodec_safetensors_write("/tmp/a1i/model.safetensors",t,1)!=0){printf("st write fail\n");return 1;}
  /* MLX -> GGUF (F32 copiato) */
  if(llmcodec_mlx_to_gguf("/tmp/a1i","/tmp/a1.gguf",12,32,4)!=0){printf("FAIL mlx->gguf\n");return 1;}
  /* .llmarch: FFN -> fp8_block */
  FILE* g=fopen("/tmp/a1.llmarch","wb");
  fprintf(g,"{\"arch\":\"a1\",\"scale_suffix\":\".scale\",\"roles\":[{\"match\":\"ffn\",\"role\":\"FFN\"}],"
            "\"quant\":{\\FFN\":\"fp8_block\"},\"quant_params\":{\\group\":32,\"bits\":4},"
            "\"convert_target\":{\"default\":{\"codec\":\"mlx_affine\",\"p0\":32,\"p1\":4},\"roles\":{\"FFN\":{\"codec\":\"fp8\",\"fmt\":0}}},"
            "\"codecs\":{\"fp8_block\":\"fp8_block\"}}");
  fclose(g);
  /* GGUF -> MLX con uscita fp8 (group=0,bits=0 -> dal file) */
  if(llmcodec_conv_gguf_to_mlx_arch("/tmp/a1.gguf","/tmp/a1o","/tmp/a1.llmarch",0,0)!=0){printf("FAIL gguf->mlx\n");return 1;}
  LlmCodecStTensorInfo* ti=NULL; int n=0; uint64_t base=0;
  if(llmcodec_safetensors_read("/tmp/a1o/model.safetensors",&ti,&n,&base)!=0){printf("FAIL read m2\n");return 1;}
  int iw=-1,is=-1; for(int i=0;i<n;i++){ if(!strcmp(ti[i].name,"layers.0.ffn.weight"))iw=i; if(!strcmp(ti[i].name,"layers.0.ffn.scale"))is=i; }
  printf("[a1fp8] m2 tensori=%d w=%d(%s) s=%d(%s)\n",n,iw,iw>=0?ti[iw].dtype:"-",is,is>=0?ti[is].dtype:"-");
  int ok=(iw>=0 && is>=0 && strcmp(ti[iw].dtype,"F8_E4M3")==0 && strcmp(ti[is].dtype,"F8_E8M0")==0);
  llmcodec_safetensors_info_free(ti);
  printf(ok?"PASS\n":"FAIL\n"); return ok?0:1;
}
