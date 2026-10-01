/* Plugin CODEC "ggml": TUTTI i tipi GGML (fmt = ggml_type). Decoder+encoder.
 * Il core NON contiene questa implementazione: e' una dylib. */
#include "llmcodec.h"
#include "llmcodec_ggml_quants.h"
#include "llmcodec_gguf_writer.h"
#include <stdlib.h>
#include <string.h>

static int g_sizes(int fmt, int p0, int p1, size_t rows, size_t cols, size_t* ob, size_t* ab) {
    (void)p0; (void)p1;
    uint64_t n = 0; if (llmcodec_ggml_type_bytes(fmt, (uint64_t)rows*cols, &n) != 0) return -1;
    *ob = (size_t)n; *ab = 0; return 0;
}
static int g_decode(const void* in, size_t in_len, int fmt, int p0, int p1, size_t rows, size_t cols,
                    const void* aux, size_t aux_len, float* out) {
    (void)in_len; (void)aux; (void)aux_len; (void)p0; (void)p1;
    uint64_t rb = 0; if (llmcodec_ggml_type_bytes(fmt, cols, &rb) != 0) return -1;
    for (size_t r = 0; r < rows; ++r)
        if (llmcodec_ggml_dequant_row(fmt, (const uint8_t*)in + r*rb, out + r*cols, (int64_t)cols) != 0) return -1;
    return 0;
}
static int g_encode(const float* in, size_t rows, size_t cols, int fmt, int p0, int p1, void* out, size_t out_cap, size_t* out_len, void* aux, size_t aux_cap, size_t* aux_len){ (void)in;(void)rows;(void)cols;(void)fmt;(void)p0;(void)p1;(void)out;(void)out_cap;(void)out_len;(void)aux;(void)aux_cap;(void)aux_len; return -1; }
static const char* g_dt(int which,int fmt){ (void)fmt; return which==0 ? "GGML" : NULL; }
static int g_shp(int fmt,int p0,int p1,size_t d0,size_t d1,size_t* r,size_t* c){(void)fmt;(void)p0;(void)p1;*r=d0;*c=d1;return 0;}
static int g_ssh(int w,int fmt,int p0,int p1,size_t r,size_t c,size_t* a,size_t* b){(void)w;(void)fmt;(void)p0;(void)p1;*a=r;*b=c;return 0;}
static const LlmCodec C = { "ggml", LLMCODEC_CODEC_ABI_VERSION, LLMCODEC_CODEC_AUX_NONE, g_dt, g_shp, g_ssh, g_sizes, g_decode, g_encode };
const LlmCodec* llmcodec_describe(void) { return &C; }
