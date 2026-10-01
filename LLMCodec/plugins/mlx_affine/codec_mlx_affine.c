/* Plugin CODEC "mlx_affine": affine MLX (U32 packed + scales/biases F16). fmt: 0.
 * aux = scales ++ biases (ciascuno sc_bytes). */
#include "llmcodec.h"
#include "llmcodec_moe_quant.h"
#include <stdint.h>
#define P0 1  /* p0 = group */
static int a_sizes(int fmt, int p0, int p1, size_t rows, size_t cols, size_t* ob, size_t* ab) {
    (void)fmt; int bits = p1 ? p1 : 4, group = p0 ? p0 : 32;
    *ob = llmcodec_moe_affine_w_bytes((uint32_t)rows, (uint32_t)cols, (uint32_t)bits);
    *ab = 2 * llmcodec_moe_affine_sc_bytes((uint32_t)rows, (uint32_t)cols, (uint32_t)group);
    return 0;
}
static int a_decode(const void* in, size_t in_len, int fmt, int p0, int p1, size_t rows, size_t cols,
                    const void* aux, size_t aux_len, float* out) {
    (void)fmt; (void)in_len;
    uint32_t group = p0 ? (uint32_t)p0 : 32; uint32_t bits = (uint32_t)(p1 ? p1 : 4);
    size_t sb = llmcodec_moe_affine_sc_bytes((uint32_t)rows, (uint32_t)cols, group);
    if (aux == NULL || aux_len < 2*sb) return -1;
    const uint16_t* sc = (const uint16_t*)aux; const uint16_t* bs = (const uint16_t*)((const uint8_t*)aux + sb);
    llmcodec_moe_affine_dequantize((const uint32_t*)in, sc, bs, (uint32_t)rows, (uint32_t)cols, group, bits, 1, out);
    return 0;
}
static int a_encode(const float* in, size_t rows, size_t cols, int fmt, int p0, int p1, void* out, size_t oc, size_t* ol, void* aux, size_t ac, size_t* al){ (void)in;(void)rows;(void)cols;(void)fmt;(void)p0;(void)p1;(void)out;(void)oc;(void)ol;(void)aux;(void)ac;(void)al; return -1; }
static const char* a_dt(int which,int fmt){ (void)fmt; return which==0 ? "U32" : "F16"; }
static int a_shp(int fmt,int p0,int p1,size_t d0,size_t d1,size_t* r,size_t* c){(void)fmt;int bits=p1?p1:4;*r=d0;*c=(bits?d1*32/(size_t)bits:d1);return 0;}
static int a_ssh(int w,int fmt,int p0,int p1,size_t r,size_t c,size_t* d0,size_t* d1){(void)fmt;int bits=p1?p1:4,grp=p0?p0:32; if(w==0){*d0=r;*d1=c*(size_t)bits/32;}else{*d0=r;*d1=c/(size_t)grp;} return 0;}
static const LlmCodec C = { "mlx_affine", LLMCODEC_CODEC_ABI_VERSION, LLMCODEC_CODEC_AUX_SCALES_BIASES, a_dt, a_shp, a_ssh, a_sizes, a_decode, a_encode };
const LlmCodec* llmcodec_describe(void) { return &C; }
