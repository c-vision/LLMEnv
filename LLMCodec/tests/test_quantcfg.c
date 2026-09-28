/* Per-tensor mode dal config.json: default mxfp4 + override affine -> codec diversi per tensore. */
#include "llmcodec_safetensors.h"
#include "llmcodec_gguf_mlx.h"
#include "llmcodec_gguf_writer.h"
#include "llmcodec_ggml_quants.h"
#include "llmcodec_moe_quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
int main(void){
    system("rm -rf /tmp/qc /tmp/qc.gguf; mkdir -p /tmp/qc");
    const int R=4, C=256;
    /* tensore "a" affine (group 64, bits 4) */
    float sa[R*C]; for(int i=0;i<R*C;i++) sa[i]=sinf(i*0.3f);
    size_t wb=llmcodec_moe_affine_w_bytes(R,C,4), sb=llmcodec_moe_affine_sc_bytes(R,C,64);
    uint32_t* wa=(uint32_t*)malloc(wb); uint16_t* sca=(uint16_t*)malloc(sb),*bsa=(uint16_t*)malloc(sb);
    llmcodec_moe_affine_quantize(sa,R,C,64,4,wa,sca,bsa);
    /* tensore "b" mxfp4 (e2m1 + e8m0 u8, group 32): nibble 2 (=1.0), scala 127 (=1.0) */
    uint32_t wb4[R*(C/8)]; for(int i=0;i<R*(C/8);i++) wb4[i]=0x22222222u;
    uint8_t scb[R*(C/32)]; for(int i=0;i<R*(C/32);i++) scb[i]=127;
    int sh2[2]={R,C/8}, sh2s[2]={R,C/64}, shb[2]={R,C/32};
    LlmCodecStTensor t[4];
    t[0].name="layers.0.mlp.a.weight";  t[0].dtype="U32"; t[0].shape=sh2;  t[0].ndim=2; t[0].data=wa;
    t[1].name="layers.0.mlp.a.scales";  t[1].dtype="F16"; t[1].shape=sh2s; t[1].ndim=2; t[1].data=sca;
    t[2].name="layers.0.mlp.a.biases";  t[2].dtype="F16"; t[2].shape=sh2s; t[2].ndim=2; t[2].data=bsa;
    LlmCodecStTensor t2[2];
    t2[0]=t[0]; t2[0].name="layers.0.mlp.b.weight"; t2[0].data=wb4;
    t2[1]=t[1]; t2[1].name="layers.0.mlp.b.scales"; t2[1].dtype="U8"; t2[1].shape=shb; t2[1].data=scb;
    llmcodec_safetensors_write("/tmp/qc/a.safetensors", t, 3);
    llmcodec_safetensors_write("/tmp/qc/b.safetensors", t2, 2);
    /* index + config (default mxfp4, override affine per "a") */
    FILE* ix=fopen("/tmp/qc/model.safetensors.index.json","wb");
    fprintf(ix,"{\"weight_map\":{\"layers.0.mlp.a.weight\":\"a.safetensors\",\"layers.0.mlp.a.scales\":\"a.safetensors\",\"layers.0.mlp.a.biases\":\"a.safetensors\",\"layers.0.mlp.b.weight\":\"b.safetensors\",\"layers.0.mlp.b.scales\":\"b.safetensors\"}}"); fclose(ix);
    FILE* cf=fopen("/tmp/qc/config.json","wb");
    fprintf(cf,"{\"quantization\":{\"group_size\":32,\"bits\":4,\"mode\":\"mxfp4\",\"layers.0.mlp.a\":{\"group_size\":64,\"bits\":4,\"mode\":\"affine\"}}}"); fclose(cf);
    FILE* g=fopen("/tmp/qc.llmarch","wb");
    fprintf(g,"{\"arch\":\"qc\",\"scale_suffix\":\".scales\",\"bias_suffix\":\".biases\",\"roles\":[{\"match\":\"mlp\",\"role\":\"FFN\"}],"
              "\"quant\":{\"FFN\":\"q4_k\"},\"codecs\":{\"mlx_affine\":\"mlx_affine\",\"fp8\":\"fp8\"},"
              "\"weight_source\":{\"affine\":{\"dtype\":\"U32\",\"codec\":\"mlx_affine\",\"mode\":\"affine\"},"
              "\"mxfp4\":{\"dtype\":\"U32\",\"codec\":\"fp4\",\"mode\":\"mxfp4\",\"fmt\":1,\"p0\":1,\"p1\":32}}}"); fclose(g);
    if(llmcodec_conv_mlx_to_gguf_arch("/tmp/qc","/tmp/qc.gguf","/tmp/qc.llmarch",12,32,4)!=0){printf("FAIL conv\n");return 1;}
    LlmCodecGgufFileInfo fi; memset(&fi,0,sizeof fi);
    if(llmcodec_gguf_read_info("/tmp/qc.gguf",&fi)!=0){printf("FAIL read\n");return 1;}
    printf("[quantcfg] GGUF tensori=%llu\n",(unsigned long long)fi.n_tensors);
    int ok=(fi.n_tensors==2);
    if(ok){
        for(uint64_t i=0;i<fi.n_tensors && ok;i++){
            uint64_t nb=0; llmcodec_ggml_type_bytes(fi.tensors[i].ggml_type, R*C, &nb);
            uint8_t* raw=(uint8_t*)malloc(nb); FILE* f=fopen("/tmp/qc.gguf","rb");
            fseeko(f,fi.data_start+fi.tensors[i].offset,SEEK_SET); fread(raw,1,nb,f); fclose(f);
            float* y=(float*)malloc((size_t)R*C*4); llmcodec_ggml_dequant_row(fi.tensors[i].ggml_type,raw,y,R*C);
            double md=0;
            if(strcmp(fi.tensors[i].name,"layers.0.mlp.a")==0) for(int k=0;k<R*C;k++){double d=fabs(y[k]-sa[k]); if(d>md)md=d;}
            else for(int k=0;k<R*C;k++){double d=fabs(y[k]-1.0); if(d>md)md=d;}
            printf("[quantcfg] %-20s tipo=%d maxerr=%.3e\n", fi.tensors[i].name, fi.tensors[i].ggml_type, md);
            ok = ok && md < 0.5; free(raw); free(y);
        }
    }
    llmcodec_gguf_free_info(&fi); free(wa); free(sca); free(bsa);
    printf(ok?"PASS\n":"FAIL\n"); return ok?0:1;
}
