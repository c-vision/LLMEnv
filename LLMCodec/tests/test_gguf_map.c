/* test_gguf_map.c — verifica la MAPPA verso nomi canonici GGUF (`gguf` nella .llmarch) e lo
 * STACKING 3D degli esperti per-esperto. Sorgente sintetica DS4.1-like (fp8 e4m3 + .scale ue8m0).
 * Atteso: `layers.0.attn.wq_a.weight` -> `blk.0.attn_q_a.weight`;
 *         `layers.0.ffn.experts.{0,1}.w1.weight` -> `blk.0.ffn_gate_exps.weight` 3D [IN,OUT,2]. */
#include "llmcodec_safetensors.h"
#include "llmcodec_gguf_mlx.h"
#include "llmcodec_gguf_read.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    system("rm -rf /tmp/gmap /tmp/gmap.gguf; mkdir -p /tmp/gmap");
    const int OUT = 64, IN = 256;
    uint8_t* w = (uint8_t*)malloc(OUT*IN); memset(w, 0x38, OUT*IN);      /* e4m3 1.0 */
    uint8_t* sc = (uint8_t*)malloc(16); memset(sc, 127, 16);            /* ue8m0 1.0 */
    int sw[2] = { OUT, IN }, ss[2] = { 2, 8 };
    LlmCodecStTensor t[6];
    int k = 0;
    t[k].name="layers.0.attn.wq_a.weight"; t[k].dtype="F8_E4M3"; t[k].shape=sw; t[k].ndim=2; t[k].data=w; k++;
    t[k].name="layers.0.attn.wq_a.scale";  t[k].dtype="U8";       t[k].shape=ss; t[k].ndim=2; t[k].data=sc; k++;
    t[k].name="layers.0.ffn.experts.0.w1.weight"; t[k].dtype="F8_E4M3"; t[k].shape=sw; t[k].ndim=2; t[k].data=w; k++;
    t[k].name="layers.0.ffn.experts.0.w1.scale";  t[k].dtype="U8";       t[k].shape=ss; t[k].ndim=2; t[k].data=sc; k++;
    t[k].name="layers.0.ffn.experts.1.w1.weight"; t[k].dtype="F8_E4M3"; t[k].shape=sw; t[k].ndim=2; t[k].data=w; k++;
    t[k].name="layers.0.ffn.experts.1.w1.scale";  t[k].dtype="U8";       t[k].shape=ss; t[k].ndim=2; t[k].data=sc; k++;
    if (llmcodec_safetensors_write("/tmp/gmap/model.safetensors", t, k) != 0) { printf("st write fail\n"); return 1; }

    FILE* g = fopen("/tmp/gmap.llmarch", "wb");
    fprintf(g, "{\"arch\":\"dsv41\",\"scale_suffix\":\".scale\",\"roles\":["
               "{\"match\":\"attn.wq_a\",\"role\":\"ATTN_Q_A\",\"gguf\":\"blk.%%d.attn_q_a.weight\"},"
               "{\"match\":\"ffn.experts.\",\"role\":\"ROUTED_EXPERT\",\"gguf\":\"blk.%%d.ffn_%%s_exps.weight\",\"stack\":true}"
               "],\"quant\":{\"ATTN_Q_A\":\"q4_k\",\"ROUTED_EXPERT\":\"q2_k\"},"
               "\"weight_source\":{\"fp8_e4m3\":{\"dtype\":\"F8_E4M3\",\"codec\":\"fp8\",\"fmt\":0,\"p0\":32,\"p1\":32,\"scale_suffix\":\".scale\"}}}");
    fclose(g);

    if (llmcodec_conv_mlx_to_gguf_arch("/tmp/gmap", "/tmp/gmap.gguf", "/tmp/gmap.llmarch", 12, 32, 4) != 0) {
        printf("FAIL conv\n"); return 1;
    }
    LlmCodecGgufFileInfo fi; memset(&fi, 0, sizeof fi);
    if (llmcodec_gguf_read_info("/tmp/gmap.gguf", &fi) != 0) { printf("FAIL read\n"); return 1; }

    int ok_a = 0, ok_e = 0;
    for (uint64_t i = 0; i < fi.n_tensors; ++i) {
        LlmCodecGgufTensorInfo* ti = &fi.tensors[i];
        printf("[gmap] %s ndim=%d dims=[%llu,%llu,%llu]\n", ti->name, ti->ndim,
               (unsigned long long)ti->dims[0], (unsigned long long)ti->dims[1], (unsigned long long)ti->dims[2]);
        if (strcmp(ti->name, "blk.0.attn_q_a.weight") == 0 && ti->ndim == 2 && ti->dims[0] == IN && ti->dims[1] == OUT) ok_a = 1;
        if (strcmp(ti->name, "blk.0.ffn_gate_exps.weight") == 0 && ti->ndim == 3 && ti->dims[0] == IN && ti->dims[1] == OUT && ti->dims[2] == 2) ok_e = 1;
    }
    printf("[gmap] rename=%d stacking3d=%d (tensori=%llu)\n", ok_a, ok_e, (unsigned long long)fi.n_tensors);
    llmcodec_gguf_free_info(&fi);
    free(w); free(sc);
    int ok = ok_a && ok_e && fi.n_tensors == 2;
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
