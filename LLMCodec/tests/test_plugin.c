/* Verifica il registry PLUGIN: un codec non built-in e' risolto da una dylib e usato in A2. */
#include "llmcodec_safetensors.h"
#include "llmcodec_gguf_mlx.h"
#include "llmcodec_gguf_writer.h"
#include "llmcodec.h"
#include "llmcodec_ggml_quants.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
int main(void){
  system("rm -rf /tmp/plg /tmp/plg.gguf; mkdir -p /tmp/plg");
  const int ROWS=4, COLS=256;
  float src[4*256]; for(int i=0;i<4*256;i++) src[i]=sinf(i*0.3f);
  int sw[2]={ROWS,COLS};
  LlmCodecStTensor t[1]; t[0].name="layers.0.ffn.weight"; t[0].dtype="F32"; t[0].shape=sw; t[0].ndim=2; t[0].data=src;
  if(llmcodec_safetensors_write("/tmp/plg/model.safetensors",t,1)!=0){printf("st write fail\n");return 1;}
  FILE* g=fopen("/tmp/plg.llmarch","wb");
  fprintf(g,"{\"arch\":\"plg\",\"scale_suffix\":\".scales\",\"roles\":[{\"match\":\"ffn\",\"role\":\"FFN\"}],"
            "\"quant\":{\"FFN\":\"q4_k\"},\"codecs\":{\"test_id\":\"test_id\"},"
            "\"weight_source\":{\"raw\":{\"dtype\":\"F32\",\"codec\":\"test_id\"}}}");
  fclose(g);
  const LlmCodec* pc=llmcodec_resolve("test_id");
  printf("[plugin] resolve test_id = %s\n", pc?"ok":"NULL");
  if(!pc){ printf("FAIL resolve (metti libllmcodec_test_id.dylib in CWD/LLMCODEC_PLUGIN_DIR)\n"); return 1; }
  if(llmcodec_conv_mlx_to_gguf_arch("/tmp/plg","/tmp/plg.gguf","/tmp/plg.llmarch",12,32,4)!=0){printf("FAIL conv\n");return 1;}
  LlmCodecGgufFileInfo fi; memset(&fi,0,sizeof fi);
  if(llmcodec_gguf_read_info("/tmp/plg.gguf",&fi)!=0){printf("FAIL read\n");return 1;}
  printf("[plugin] GGUF tensori=%llu tipo=%d dims=[%llu,%llu]\n",(unsigned long long)fi.n_tensors,
         fi.n_tensors?fi.tensors[0].ggml_type:-1,(unsigned long long)fi.tensors[0].dims[0],(unsigned long long)fi.tensors[0].dims[1]);
  int ok=(fi.n_tensors==1 && fi.tensors[0].ggml_type==LLMCODEC_GGML_T_Q4_K && fi.tensors[0].dims[0]==COLS && fi.tensors[0].dims[1]==ROWS);
  if(ok){
    uint64_t nb=0; llmcodec_ggml_type_bytes(LLMCODEC_GGML_T_Q4_K,(uint64_t)ROWS*COLS,&nb);
    uint8_t* raw=(uint8_t*)malloc(nb); FILE* f=fopen("/tmp/plg.gguf","rb");
    fseeko(f,fi.data_start+fi.tensors[0].offset,SEEK_SET); fread(raw,1,nb,f); fclose(f);
    float* y=(float*)malloc((size_t)ROWS*COLS*4); llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_Q4_K,raw,y,(int64_t)ROWS*COLS);
    double md=0; for(int i=0;i<ROWS*COLS;i++){double dd=fabs(y[i]-src[i]); if(dd>md)md=dd;}
    printf("[plugin] dequant vs sorgente maxerr=%.4e\n",md); ok = ok && md<0.5; free(raw); free(y);
  }
  llmcodec_gguf_free_info(&fi);
  printf(ok?"PASS\n":"FAIL\n"); return ok?0:1;
}
