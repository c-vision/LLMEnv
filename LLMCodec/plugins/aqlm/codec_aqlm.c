/* Plugin CODEC "aqlm": AQLM (Additive Quantization). Port del kernel CPU di riferimento
 * (Vahe1994/AQLM inference_kernels/numba_kernel.py, Apache-2.0).
 *   codes     int8  [out_groups, in_groups, num_codebooks]
 *   codebooks f16/f32 [num_codebooks, codebook_size=256, out_group_size, in_group_size]
 *   scales    f16/f32 [out_groups]
 * W[out, in] = scales[out_group] * sum_c codebooks[c][codes[in_group, out, c]][out_in_group, in_in_group]
 * Parametri: p0 = num_codebooks, p1 = in_group_size (out_group_size assunto 1).
 * aux = codebooks ++ scales. Main = codes. */
#include "llmcodec.h"
#include "llmcodec_fp16.h"
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
static float a_f16(const uint8_t* p){ uint16_t h=*(const uint16_t*)p; uint32_t s=(h>>15)&1,e=(h>>10)&31,m=h&1023,u;
    if(e==0){ if(m==0)u=s<<31; else{e=127-15+1; while(!(m&1024)){m<<=1;e--;} m&=1023; u=(s<<31)|((uint32_t)e<<23)|((uint32_t)m<<13);} }
    else if(e==31){ u=(s<<31)|0x7f800000|((uint32_t)m<<13);} else u=(s<<31)|((uint32_t)(e-15+127)<<23)|((uint32_t)m<<13);
    float f; memcpy(&f,&u,4); return f; }
static int a_sizes(int fmt,int p0,int p1,size_t rows,size_t cols,size_t* ob,size_t* ab){
    (void)fmt; int nc=p0?p0:1; int ig=p1?p1:8; size_t og=1;
    size_t og_n=(rows+og-1)/og, ig_n=(cols+ig-1)/ig;
    (void)og_n; *ob=(size_t)og_n*ig_n*nc;               /* codes int8 */
    *ab=(size_t)nc*256*og*(size_t)ig*((fmt==1)?4:2) + og_n*2;  /* codebooks f16/f32 + scale f16 */
    return 0; }
static int a_shape(int fmt,int p0,int p1,size_t d0,size_t d1,size_t* r,size_t* c){
    (void)fmt; int ig=p1?p1:8; (void)ig; *r=d0; *c=d1; return 0; }   /* gia' logico (out,in) */
static int a_ssh(int w,int fmt,int p0,int p1,size_t r,size_t c,size_t* d0,size_t* d1){ (void)w;(void)fmt;(void)r;(void)c;(void)p0;(void)p1;*d0=0;*d1=0;return -1; }
static const char* a_dt(int which,int fmt){ (void)fmt; return which==0?"I8":"F16"; }
static int a_decode(const void* in,size_t il,int fmt,int p0,int p1,size_t rows,size_t cols,const void* aux,size_t al,float* out){
    (void)il;(void)fmt;
    int nc=p0?p0:1, ig=(int)(p1?p1:8); size_t og=1;
    size_t out_groups=(rows+og-1)/og, in_groups=(cols+ig-1)/ig;
    int esz=(fmt==1)?4:2; size_t cb_sz=(size_t)nc*256*og*(size_t)ig*esz;
    if(aux==NULL || al < cb_sz + out_groups*2) return -1;
    const int8_t* codes=(const int8_t*)in;
    const uint8_t* cb=(const uint8_t*)aux; const uint8_t* sc=(const uint8_t*)aux + cb_sz;
    /* azzera */
    for(size_t i=0;i<rows*cols;i++) out[i]=0.0f;
    for(size_t ogi=0; ogi<out_groups; ogi++){
        float scale=a_f16(sc+ogi*2);
        for(size_t ii=0; ii<og; ii++){
            size_t o=ogi*og+ii; if(o>=rows) continue;
            for(size_t j=0; j<in_groups; j++){
                for(size_t t=0; t<(size_t)ig; t++){
                    size_t cidx=j*ig+t; if(cidx>=cols) continue;
                    float acc=0.0f;
                    for(int c=0;c<nc;c++){
                        int code=(uint8_t)codes[(ogi*in_groups + j)*nc + c];
                        size_t coff=((size_t)c*256 + (size_t)code)*og*(size_t)ig;
                        size_t e=coff + (ii*(size_t)ig + t);
                        if(esz==4){ float v; memcpy(&v,cb+e*4,4); acc+=v; } else acc += a_f16(cb + e*2);
                    }
                    out[o*cols+cidx] = acc*scale;
                }
            }
        }
    }
    return 0; }
/* k-means (Lloyd) deterministico: K centroidi di dim `d` da `n` vettori. */
static void a_kmeans(const float* x, size_t n, int d, int K, float* cent, int iters) {
    for (int k = 0; k < K; k++) {
        size_t idx = (n > 0) ? (size_t)k * n / (size_t)K : 0; if (idx >= n) idx = n - 1;
        memcpy(cent + (size_t)k * d, x + idx * (size_t)d, (size_t)d * 4);
    }
    if (n == 0) return;
    int* asg = (int*)malloc(n * 4);
    for (int it = 0; it < iters; it++) {
        for (size_t i = 0; i < n; i++) {
            int best = 0; float bd = 1e30f;
            for (int k = 0; k < K; k++) {
                float e = 0.0f;
                for (int j = 0; j < d; j++) { float df = x[i * (size_t)d + j] - cent[(size_t)k * d + j]; e += df * df; }
                if (e < bd) { bd = e; best = k; }
            }
            asg[i] = best;
        }
        double* sum = (double*)calloc((size_t)K * d, 8); int* cnt = (int*)calloc((size_t)K, 4);
        for (size_t i = 0; i < n; i++) { int k = asg[i]; cnt[k]++; for (int j = 0; j < d; j++) sum[(size_t)k * d + j] += x[i * (size_t)d + j]; }
        for (int k = 0; k < K; k++) if (cnt[k] > 0) for (int j = 0; j < d; j++) cent[(size_t)k * d + j] = (float)(sum[(size_t)k * d + j] / cnt[k]);
        free(sum); free(cnt);
    }
    free(asg);
}

/* Encoder AQLM (contropartita del decoder): quantizzazione ADDITIVA a `nc` codebook appresi per
 * residui (k-means sul residuo di ogni livello). Scala per out-group = 1 (tutto nei codebook).
 * Layout di `out` (codes int8 [out_groups, in_groups, nc]) e `aux` (codebooks [nc,256,og,ig] +
 * scale f16) IDENTICO al decoder, quindi il round-trip e' verificabile. */
static int a_encode(const float* in, size_t rows, size_t cols, int fmt, int p0, int p1,
                    void* out, size_t oc, size_t* ol, void* aux, size_t ac, size_t* al) {
    if (in == NULL || out == NULL || aux == NULL) return -1;
    int nc = p0 ? p0 : 1; int ig = p1 ? p1 : 8; size_t og = 1;
    size_t out_groups = (rows + og - 1) / og, in_groups = (cols + (size_t)ig - 1) / (size_t)ig;
    int esz = (fmt == 1) ? 4 : 2;
    size_t cb_sz = (size_t)nc * 256 * og * (size_t)ig * (size_t)esz;
    size_t sc_sz = out_groups * 2;
    if (oc < out_groups * in_groups * (size_t)nc) return -1;
    if (ac < cb_sz + sc_sz) return -1;
    int8_t* codes = (int8_t*)out;
    uint8_t* cbp = (uint8_t*)aux; uint8_t* scp = (uint8_t*)aux + cb_sz;
    size_t nblk = rows * in_groups;
    float* blk = (float*)malloc(nblk * (size_t)ig * 4);
    if (blk == NULL) return -1;
    for (size_t o = 0; o < rows; o++)
        for (size_t j = 0; j < in_groups; j++) {
            float* dst = blk + (o * in_groups + j) * (size_t)ig;
            for (int t = 0; t < ig; t++) { size_t cidx = j * (size_t)ig + t; dst[t] = (cidx < cols) ? in[o * cols + cidx] : 0.0f; }
        }
    for (size_t g = 0; g < out_groups; g++) { uint16_t h = llmcodec_f16_from_f32(1.0f); memcpy(scp + g * 2, &h, 2); }
    float* res = (float*)malloc(nblk * (size_t)ig * 4);
    if (res == NULL) { free(blk); return -1; }
    memcpy(res, blk, nblk * (size_t)ig * 4);
    for (int c = 0; c < nc; c++) {
        float* cent = (float*)malloc((size_t)256 * (size_t)ig * 4);
        if (cent == NULL) { free(blk); free(res); return -1; }
        a_kmeans(res, nblk, ig, 256, cent, 12);
        for (size_t b = 0; b < nblk; b++) {
            int best = 0; float bd = 1e30f;
            for (int k = 0; k < 256; k++) {
                float e = 0.0f;
                for (int j = 0; j < ig; j++) { float df = res[b * (size_t)ig + j] - cent[(size_t)k * ig + j]; e += df * df; }
                if (e < bd) { bd = e; best = k; }
            }
            codes[b * (size_t)nc + c] = (int8_t)best;
            for (int j = 0; j < ig; j++) res[b * (size_t)ig + j] -= cent[(size_t)best * ig + j];
        }
        for (int k = 0; k < 256; k++) for (int j = 0; j < ig; j++) {
            size_t e = ((size_t)c * 256 + k) * og * (size_t)ig + j;
            if (esz == 4) { float v = cent[(size_t)k * ig + j]; memcpy(cbp + e * 4, &v, 4); }
            else { uint16_t h = llmcodec_f16_from_f32(cent[(size_t)k * ig + j]); memcpy(cbp + e * 2, &h, 2); }
        }
        free(cent);
    }
    free(blk); free(res);
    if (ol) *ol = out_groups * in_groups * (size_t)nc;
    if (al) *al = cb_sz + sc_sz;
    return 0;
}
static const LlmCodec C = { "aqlm", LLMCODEC_CODEC_ABI_VERSION, LLMCODEC_CODEC_AUX_SCALE, a_dt, a_shape, a_ssh, a_sizes, a_decode, a_encode };
const LlmCodec* llmcodec_describe(void){ return &C; }
