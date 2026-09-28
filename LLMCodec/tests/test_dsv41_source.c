/* Sorgente DS4.1 block-scaled (fp8 e4m3 + .scale ue8m0) -> GGUF (ricetta q4_k). */
#include "llmcodec_safetensors.h"
#include "llmcodec_gguf_mlx.h"
#include "llmcodec_gguf_writer.h"
#include "llmcodec_ggml_quants.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    system("rm -rf /tmp/d41 /tmp/d41.gguf; mkdir -p /tmp/d41");
    const int OUT = 64, IN = 256;
    uint8_t* w = (uint8_t*)malloc(OUT*IN); memset(w, 0x38, OUT*IN);   /* e4m3 0x38 = 1.0 */
    uint8_t* sc = (uint8_t*)malloc(2*8); memset(sc, 127, 16);          /* ue8m0 127 = 1.0 */
    int sw[2] = { OUT, IN }, ss[2] = { 2, 8 };
    LlmCodecStTensor t[2];
    t[0].name="layers.0.attn.wq_a.weight"; t[0].dtype="F8_E4M3"; t[0].shape=sw; t[0].ndim=2; t[0].data=w;
    t[1].name="layers.0.attn.wq_a.scale";  t[1].dtype="U8";      t[1].shape=ss; t[1].ndim=2; t[1].data=sc;
    if (llmcodec_safetensors_write("/tmp/d41/model.safetensors", t, 2) != 0) { printf("st write fail\n"); return 1; }
    FILE* g = fopen("/tmp/d41.llmarch", "wb");
    fprintf(g, "{\"arch\":\"dsv41\",\"scale_suffix\":\".scale\","
               "\"roles\":[{\"match\":\"attn.wq_a\",\"role\":\"ATTN_Q_A\"}],"
               "\"quant\":{\"ATTN_Q_A\":\"q4_k\"},"
               "\"weight_source\":{\"fp8_e4m3\":{\"dtype\":\"F8_E4M3\",\"codec\":\"fp8\",\"fmt\":0,\"p0\":32,\"p1\":32,\"scale_suffix\":\".scale\"}}}");
    fclose(g);

    if (llmcodec_conv_mlx_to_gguf_arch("/tmp/d41", "/tmp/d41.gguf", "/tmp/d41.llmarch", 12 /*Q4_K*/, 32, 4) != 0) {
        printf("FAIL conv\n"); return 1;
    }
    LlmCodecGgufFileInfo fi; memset(&fi, 0, sizeof fi);
    if (llmcodec_gguf_read_info("/tmp/d41.gguf", &fi) != 0) { printf("FAIL read\n"); return 1; }
    printf("[dsv41src] GGUF tensori=%llu tipo=%d dims=[%llu,%llu]\n", (unsigned long long)fi.n_tensors,
           fi.n_tensors?fi.tensors[0].ggml_type:-1,
           fi.n_tensors?(unsigned long long)fi.tensors[0].dims[0]:0,
           fi.n_tensors?(unsigned long long)fi.tensors[0].dims[1]:0);
    int ok = (fi.n_tensors == 1 && fi.tensors[0].ggml_type == LLMCODEC_GGML_T_Q4_K
              && fi.tensors[0].dims[0] == IN && fi.tensors[0].dims[1] == OUT);
    if (ok) {
        uint64_t nb = 0; llmcodec_ggml_type_bytes(LLMCODEC_GGML_T_Q4_K, (uint64_t)OUT*IN, &nb);
        uint8_t* raw = (uint8_t*)malloc(nb);
        FILE* f = fopen("/tmp/d41.gguf", "rb");
        fseeko(f, fi.data_start + fi.tensors[0].offset, SEEK_SET); fread(raw, 1, nb, f); fclose(f);
        float* y = (float*)malloc((size_t)OUT*IN*4);
        llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_Q4_K, raw, y, (int64_t)OUT*IN);
        double md = 0; for (int i = 0; i < OUT*IN; ++i) { double d = fabs(y[i]-1.0); if (d>md) md=d; }
        printf("[dsv41src] dequant vs 1.0 maxerr=%.3e\n", md);
        ok = ok && md < 0.05;
        free(raw); free(y);
    }
    llmcodec_gguf_free_info(&fi);
    free(w); free(sc);
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
