/* Plugin CODEC "fp8": block [32,32] (fmt 0), fp4 e2m1 expert (fmt 1), fp8 row-block (fmt 2). */
#include "llmcodec.h"
#include "llmcodec_fp8_fp4.h"
#include <string.h>
#include <stdlib.h>
static int f_sizes(int fmt, int p0, int p1, size_t rows, size_t cols, size_t* ob, size_t* ab) {
    (void)p0; (void)p1;
    if (fmt == 1) { int br=p0?p0:1, bc=p1?p1:32; if (cols % 2) return -1; *ob = rows*cols/2; *ab = ((rows+br-1)/br)*((cols+bc-1)/bc); return 0; }
    if (fmt == 2) { if (cols % 32) return -1; *ob = rows*cols; *ab = rows*(cols/32); return 0; }
    if (fmt == 3) { *ob = rows*cols; *ab = 0; return 0; }   /* e5m2 raw */
    { int br=p0?p0:32, bc=p1?p1:32; *ob = rows*cols; *ab = ((rows+br-1)/br)*((cols+bc-1)/bc); } return 0;
}
static int f_decode(const void* in, size_t in_len, int fmt, int p0, int p1, size_t rows, size_t cols,
                    const void* aux, size_t aux_len, float* out) {
    (void)in_len; (void)p0; (void)p1; (void)aux_len;
    if (fmt == 3) { const uint8_t* b=(const uint8_t*)in; for(size_t i=0;i<rows*cols;i++) out[i]=llmcodec_fp8_e5m2_to_f32(b[i]); return 0; }
    if (aux == NULL) return -1;
    if (fmt == 2) llmcodec_dsv41_dequant_fp8_row((const uint8_t*)in, (const uint8_t*)aux, (int)rows, (int)cols, out);
    else               llmcodec_dsv41_dequant_fp8_block_sz((const uint8_t*)in, (const uint8_t*)aux, (int)rows, (int)cols, p0?p0:32, p1?p1:32, out);
    return 0;
}
static int f_encode(const float* in, size_t rows, size_t cols, int fmt, int p0, int p1,
                    void* out, size_t out_cap, size_t* out_len, void* aux, size_t aux_cap, size_t* aux_len) {
    (void)out_cap; (void)aux_cap; (void)p0; (void)p1;
    if (aux == NULL) return -1;
    if (fmt == 2) { if (cols % 32) return -1; llmcodec_dsv41_quant_fp8_row(in, (uint8_t*)out, (uint8_t*)aux, (int)rows, (int)cols);
                    *out_len = rows*cols; *aux_len = rows*(cols/32); return 0; }
    if (fmt == 0) { llmcodec_dsv41_quant_fp8_block(in, (uint8_t*)out, (uint8_t*)aux, (int)rows, (int)cols);
                    *out_len = rows*cols; *aux_len = ((rows+31)/32)*((cols+31)/32); return 0; }
    return -1;   /* fp4 encode non ancora implementato */
}
static const char* f_dt(int which,int fmt){ if(fmt==3) return which==0 ? "F8_E5M2" : NULL; return which==0 ? "F8_E4M3" : "F8_E8M0"; }
static int f_shp(int fmt,int p0,int p1,size_t d0,size_t d1,size_t* r,size_t* c){(void)p0;(void)p1;*r=d0;*c=d1;return 0;}
static int f_ssh(int w,int fmt,int p0,int p1,size_t r,size_t c,size_t* d0,size_t* d1){
    if(fmt==3){ *d0=r; *d1=c; return 0; }
    int br=(p0?p0:32), bc=(p1?p1:32);
    if(w==0){ *d0=r; *d1=c; }
    else { *d0=(r+br-1)/br; *d1=(c+bc-1)/bc; }
    return 0;}
static const LlmCodec C = { "fp8", LLMCODEC_CODEC_ABI_VERSION, LLMCODEC_CODEC_AUX_SCALE, f_dt, f_shp, f_ssh, f_sizes, f_decode, f_encode };
const LlmCodec* llmcodec_describe(void) { return &C; }
