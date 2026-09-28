/* Verifica la RICETTA per-ruolo: ogni tensore va nel tipo GGML dichiarato dalla .llmarch
 * (role->fmt), non in un ggml_type unico. */
#include "llmcodec_rolemap.h"
#include "llmcodec_gguf_mlx.h"
#include "llmcodec_gguf_writer.h"
#include "llmcodec_ggml_quants.h"
#include "llmcodec_safetensors.h"
#include "llmcodec_moe_quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void mk_affine(const char* path, const char* base, uint32_t rows, uint32_t cols,
                      uint32_t group, uint32_t bits) {
    float x[1024]; for (uint32_t i = 0; i < rows*cols; ++i) x[i] = sinf(i*0.17f);
    size_t wb = llmcodec_moe_affine_w_bytes(rows, cols, bits), sb = llmcodec_moe_affine_sc_bytes(rows, cols, group);
    uint32_t* w = (uint32_t*)malloc(wb); uint16_t* sc = (uint16_t*)malloc(sb), *bs = (uint16_t*)malloc(sb);
    llmcodec_moe_affine_quantize(x, rows, cols, group, bits, w, sc, bs);
    char n0[128], n1[128], n2[128];
    snprintf(n0,sizeof n0,"%s.weight",base); snprintf(n1,sizeof n1,"%s.scales",base); snprintf(n2,sizeof n2,"%s.biases",base);
    int shp[2] = { (int)rows, (int)(cols*bits/32) }, shp2[2] = { (int)rows, (int)(cols/group) };
    LlmCodecStTensor t[3];
    t[0].name=n0; t[0].dtype="U32"; t[0].shape=shp;  t[0].ndim=2; t[0].data=w;
    t[1].name=n1; t[1].dtype="F16"; t[1].shape=shp2; t[1].ndim=2; t[1].data=sc;
    t[2].name=n2; t[2].dtype="F16"; t[2].shape=shp2; t[2].ndim=2; t[2].data=bs;
    if (llmcodec_safetensors_write(path, t, 3) != 0) { printf("st write fail\n"); exit(1); }
    free(w); free(sc); free(bs);
}

int main(void) {
    system("rm -rf /tmp/rit /tmp/rit.gguf; mkdir -p /tmp/rit");
    /* fixture: 2 tensori affine in 2 shard, ruoli diversi */
    mk_affine("/tmp/rit/model-00001-of-00002.safetensors", "layers.0.attn.wq_a", 4, 256, 32, 4);
    mk_affine("/tmp/rit/model-00002-of-00002.safetensors", "head", 4, 256, 32, 4);
    FILE* ix = fopen("/tmp/rit/model.safetensors.index.json", "wb");
    fprintf(ix, "{\"metadata\":{},\"weight_map\":{\"layers.0.attn.wq_a.weight\":\"model-00001-of-00002.safetensors\","
                "\"head.weight\":\"model-00002-of-00002.safetensors\"}}");
    fclose(ix);
    /* ricetta .llmarch inline: ATTN_Q_A -> q4_k, OUTPUT -> q6_k */
    FILE* g = fopen("/tmp/recipe.llmarch", "wb");
    fprintf(g, "{\"arch\":\"test\",\"scale_suffix\":\".scales\",\"roles\":["
               "{\"match\":\"attn.wq_a\",\"role\":\"ATTN_Q_A\"},{\"match\":\"head.weight\",\"role\":\"OUTPUT\"}],"
               "\"quant\":{\"ATTN_Q_A\":\"q4_k\",\"OUTPUT\":\"q6_k\",\"recipe\":\"_M\"}}");
    fclose(g);

    /* ggml_type di default = Q8_0: se la ricetta e' applicata NON deve comparire. */
    if (llmcodec_conv_mlx_to_gguf_arch("/tmp/rit", "/tmp/rit.gguf", "/tmp/recipe.llmarch", LLMCODEC_GGML_T_Q8_0, 32, 4) != 0) {
        printf("FAIL conv\n"); return 1;
    }
    LlmCodecGgufFileInfo fi; memset(&fi, 0, sizeof fi);
    if (llmcodec_gguf_read_info("/tmp/rit.gguf", &fi) != 0) { printf("FAIL read\n"); return 1; }
    int wa = -1, wo = -1;
    for (uint64_t i = 0; i < fi.n_tensors; ++i) {
        if (strcmp(fi.tensors[i].name, "layers.0.attn.wq_a") == 0) wa = fi.tensors[i].ggml_type;
        if (strcmp(fi.tensors[i].name, "head") == 0) wo = fi.tensors[i].ggml_type;
    }
    printf("[recipe] wq_a=%d (atteso Q4_K=%d)  head=%d (atteso Q6_K=%d)\n",
           wa, LLMCODEC_GGML_T_Q4_K, wo, LLMCODEC_GGML_T_Q6_K);
    int ok = (wa == LLMCODEC_GGML_T_Q4_K && wo == LLMCODEC_GGML_T_Q6_K);
    llmcodec_gguf_free_info(&fi);
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
