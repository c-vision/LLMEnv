/* llm_ggml_test.c — verifica tabella tipi ggml + dequant semplici. */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "llm_ggml.h"
static float f32_to_f16_roundtrip(float x){ /* scrittura fp16 manuale per il test */
    unsigned u; memcpy(&u,&x,4); unsigned sign=(u>>16)&0x8000, exp=(u>>23)&0xFF, man=u&0x7FFFFF;
    if (exp==0) return 0; exp = exp-127+15; unsigned h = sign | (exp<<10) | (man>>13);
    unsigned f = ((h&0x8000u)<<16) | (((h>>10)&0x1F)+112u<<23) | ((h&0x3FF)<<13);
    float o; memcpy(&o,&f,4); return o;
}
int main(void){
    int fail=0, ok=1;
    ok = llm_ggml_type_size(LLM_GGML_TYPE_Q8_0)==34 && llm_ggml_blck_size(LLM_GGML_TYPE_Q8_0)==32 &&
         llm_ggml_type_size(LLM_GGML_TYPE_Q2_K)==84 && llm_ggml_blck_size(LLM_GGML_TYPE_Q2_K)==256 &&
         llm_ggml_type_size(LLM_GGML_TYPE_IQ2_XXS)==66 && llm_ggml_nbytes(LLM_GGML_TYPE_Q8_0, 64)==68;
    printf("%s %-40s\n", ok?"PASS":"FAIL", "type table (size/blck/nbytes)"); if(!ok) fail++;
    /* Q8_0: blocco 32 con d=0.5, qs = -16..15 */
    { unsigned char blk[34]; uint16_t dh; float d=0.5f;
      /* fp16(0.5)=0x3800 */ dh=0x3800; memcpy(blk,&dh,2);
      for(int j=0;j<32;j++) blk[2+j]=(unsigned char)(int8_t)(j-16);
      float out[32]; ok = llmcodec_ggml_dequant_row(LLM_GGML_TYPE_Q8_0, blk, out, 32) == 0;
      for(int j=0;j<32 && ok;j++) if(fabsf(out[j] - 0.5f*(float)(j-16))>1e-6f) ok=0;
    }
    printf("%s %-40s\n", ok?"PASS":"FAIL", "dequant Q8_0"); if(!ok) fail++;
    /* F32 passthrough */
    { float in[4]={1,-2,3,-4}, out[4]; ok=llmcodec_ggml_dequant_row(LLM_GGML_TYPE_F32,(uint8_t*)in,out,4)==0;
      for(int i=0;i<4&&ok;i++) if(out[i]!=in[i]) ok=0; }
    printf("%s %-40s\n", ok?"PASS":"FAIL", "dequant F32"); if(!ok) fail++;
    /* F16 (0.5, -2.0) */
    { uint16_t h[2]={0x3800,0xC000}; float out[2]; ok=llmcodec_ggml_dequant_row(LLM_GGML_TYPE_F16,(uint8_t*)h,out,2)==0;
      ok = ok && out[0]==0.5f && out[1]==-2.0f; }
    printf("%s %-40s\n", ok?"PASS":"FAIL", "dequant F16"); if(!ok) fail++;
    /* Q2_K: blocco costruito a mano; d=1.0, dmin=0.5, scales[0]=0x12, qs[0]=0b11111111 */
    { static unsigned char b[84]; memset(b,0,sizeof b);
      b[0]=0x12; b[16]=0xFF; /* q per group0 = 3 */
      uint16_t dh=0x3C00u, mh=0x3800u; memcpy(b+80,&dh,2); memcpy(b+82,&mh,2);
      static float out[256]; ok = llmcodec_ggml_dequant_row(LLM_GGML_TYPE_Q2_K, b, out, 256) == 0;
      /* dst[0] = 1.0*2*3 - 0.5*1 = 5.5 ; group1 (k=16) scales[1]=0 -> min 0 -> 0 */
      ok = ok && fabsf(out[0] - 5.5f) < 1e-4f && fabsf(out[16]) < 1e-6f; }
    printf("%s %-40s\n", ok?"PASS":"FAIL", "dequant Q2_K (layout)"); if(!ok) fail++;
    /* IQ2_XXS: blocco d=1.0, qs tutti 0 -> aux32=0, grid[0]=0x08.. (8), signs 0 -> w=1.0 */
    { static unsigned char b[66]; memset(b,0,sizeof b);
      uint16_t dh=0x3C00u; memcpy(b,&dh,2);
      static float out[256]; ok = llmcodec_ggml_dequant_row(LLM_GGML_TYPE_IQ2_XXS, b, out, 256) == 0;
      for (int i=0;i<256 && ok;i++) if (fabsf(out[i]-1.0f) > 1e-4f) ok = 0; }
    printf("%s %-40s\n", ok?"PASS":"FAIL", "dequant IQ2_XXS (blocco zero)"); if(!ok) fail++;
    /* IQ2_XXS segno: aux32[1]=127 -> li=0 sign 127 (tutti negativi), li>=1 positivi */
    { static unsigned char b[66]; memset(b,0,sizeof b);
      uint16_t dh=0x3C00u; memcpy(b,&dh,2);
      uint16_t q2[4] = { 0u, 0u, 127u, 0u }; memcpy(b+2, q2, 8);
      static float out[256]; ok = llmcodec_ggml_dequant_row(LLM_GGML_TYPE_IQ2_XXS, b, out, 256) == 0;
      for (int j=0;j<8 && ok;j++) if (fabsf(out[j] + 1.0f) > 1e-4f) ok = 0;      /* li0 negativo */
      for (int j=8;j<32 && ok;j++) if (fabsf(out[j] - 1.0f) > 1e-4f) ok = 0;      /* li1..3 positivi */
    }
    printf("%s %-40s\n", ok?"PASS":"FAIL", "dequant IQ2_XXS (segno)"); if(!ok) fail++;
    printf(fail?"FAILURES: %d\n":"ALL PASS\n", fail);
    return fail;
}
