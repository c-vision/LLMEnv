/* Tabella ENGRAM: fp8 e4m3 [rows,cols] + scala ue8m0 [rows, cols/32] (row-block) -> GGUF. */
#include "llmcodec_safetensors.h"
#include "llmcodec_gguf_mlx.h"
#include "llmcodec_gguf_writer.h"
#include "llmcodec_ggml_quants.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
int main(void){
  system("rm -rf /tmp/eng /tmp/eng.gguf; mkdir -p /tmp/eng");
  const int ROWS=32, COLS=256;
  uint8_t* w=(uint8_t*)malloc(ROWS*COLS); memset(w,0x38,ROWS*COLS);   /* e4m3 0x38=1.0 */
  uint8_t* sc=(uint8_t*)malloc(ROWS*(COLS/32)); memset(sc,127,ROWS*(COLS/32)); /* ue8m0 127=1.0 */
  int sw[2]={ROWS,COLS}, ss[2]={ROWS,COLS/32};
  LlmCodecStTensor t[2];
  t[0].name="layers.1.engram.embed.weight"; t[0].dtype="F8_E4M3"; t[0].shape=sw; t[0].ndim=2; t[0].data=w;
  t[1].name="layers.1.engram.embed.scale";  t[1].dtype="U8";      t[1].shape=ss; t[1].ndim=2; t[1].data=sc;
  if(llmcodec_safetensors_write("/tmp/eng/model.safetensors",t,2)!=0){printf("st write fail\n");return 1;}
  FILE* g=fopen("/tmp/eng.llmarch","wb");
  fprintf(g,"{\"arch\":\"eng\",\"scale_suffix\":\".scale\",\"roles\":[{\"match\":\"engram\",\"role\":\"ENGRAM\"}],"
            "\"quant\":{\"ENGRAM\":\"q4_k\"},\"codecs\":{\"fp8_row\":\"fp8_row\",\"fp8_block\":\"fp8_block\"},"
            "\"weight_source\":{\"engram_table\":{\"dtype\":\"F8_E4M3\",\"codec\":\"fp8\",\"fmt\":2,\"role\":\"ENGRAM\",\"scale_suffix\":\".scale\"}}}");
  fclose(g);
  if(llmcodec_conv_mlx_to_gguf_arch("/tmp/eng","/tmp/eng.gguf","/tmp/eng.llmarch",12,0,0)!=0){printf("FAIL conv\n");return 1;}
  LlmCodecGgufFileInfo fi; memset(&fi,0,sizeof fi);
  if(llmcodec_gguf_read_info("/tmp/eng.gguf",&fi)!=0){printf("FAIL read\n");return 1;}
  printf("[engram] tensori=%llu tipo=%d dims=[%llu,%llu]\n",(unsigned long long)fi.n_tensors,
         fi.n_tensors?fi.tensors[0].ggml_type:-1,
         fi.n_tensors?(unsigned long long)fi.tensors[0].dims[0]:0,
         fi.n_tensors?(unsigned long long)fi.tensors[0].dims[1]:0);
  int ok=(fi.n_tensors==1 && fi.tensors[0].ggml_type==LLMCODEC_GGML_T_Q4_K && fi.tensors[0].dims[0]==COLS && fi.tensors[0].dims[1]==ROWS);
  if(ok){
    uint64_t nb=0; llmcodec_ggml_type_bytes(LLMCODEC_GGML_T_Q4_K,(uint64_t)ROWS*COLS,&nb);
    uint8_t* raw=(uint8_t*)malloc(nb); FILE* f=fopen("/tmp/eng.gguf","rb");
    fseeko(f,fi.data_start+fi.tensors[0].offset,SEEK_SET); fread(raw,1,nb,f); fclose(f);
    float* y=(float*)malloc((size_t)ROWS*COLS*4); llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_Q4_K,raw,y,(int64_t)ROWS*COLS);
    double md=0; for(int i=0;i<ROWS*COLS;i++){double dd=fabs(y[i]-1.0); if(dd>md)md=dd;}
    printf("[engram] dequant vs 1.0 maxerr=%.3e\n",md); ok = ok && md<0.05; free(raw); free(y);
  }
  llmcodec_gguf_free_info(&fi); free(w); free(sc);
  printf(ok?"PASS\n":"FAIL\n"); return ok?0:1;
}
