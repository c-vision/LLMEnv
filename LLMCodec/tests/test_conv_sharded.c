/* Verifica l'I/O sharded del converter MLX->GGUF: 2 shard + index.json. */
#include "llmcodec_safetensors.h"
#include "llmcodec_moe_quant.h"
#include "llmcodec_gguf_mlx.h"
#include "llmcodec_gguf_writer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void mk_affine(const char* path, const char* base, uint32_t rows, uint32_t cols,
                      uint32_t group, uint32_t bits, float phase) {
    float x[1024];
    for (uint32_t i = 0; i < rows*cols; ++i) x[i] = sinf(i*0.3f + phase);
    size_t wb = llmcodec_moe_affine_w_bytes(rows, cols, bits), sb = llmcodec_moe_affine_sc_bytes(rows, cols, group);
    uint32_t* w = (uint32_t*)malloc(wb); uint16_t* sc = (uint16_t*)malloc(sb), *bs = (uint16_t*)malloc(sb);
    if (llmcodec_moe_affine_quantize(x, rows, cols, group, bits, w, sc, bs) != 0) { printf("quant fail\n"); exit(1); }
    char n0[128], n1[128], n2[128];
    snprintf(n0, sizeof n0, "%s.weight", base);
    snprintf(n1, sizeof n1, "%s.scales", base);
    snprintf(n2, sizeof n2, "%s.biases", base);
    int shp[2]  = { (int)rows, (int)(cols*bits/32) };
    int shp2[2] = { (int)rows, (int)(cols/group) };
    LlmCodecStTensor t[3];
    t[0].name=n0; t[0].dtype="U32"; t[0].shape=shp;  t[0].ndim=2; t[0].data=w;
    t[1].name=n1; t[1].dtype="F16"; t[1].shape=shp2; t[1].ndim=2; t[1].data=sc;
    t[2].name=n2; t[2].dtype="F16"; t[2].shape=shp2; t[2].ndim=2; t[2].data=bs;
    if (llmcodec_safetensors_write(path, t, 3) != 0) { printf("st write fail %s\n", path); exit(1); }
    free(w); free(sc); free(bs);
}

int main(void) {
    system("rm -rf /tmp/m1s /tmp/m1s.gguf; mkdir -p /tmp/m1s");
    mk_affine("/tmp/m1s/model-00001-of-00002.safetensors", "w", 4, 256, 32, 4, 0.0f);
    mk_affine("/tmp/m1s/model-00002-of-00002.safetensors", "z", 4, 256, 32, 4, 1.0f);
    FILE* f = fopen("/tmp/m1s/model.safetensors.index.json", "wb");
    if (f == NULL) { printf("index write fail\n"); return 1; }
    fprintf(f, "{\"metadata\":{},\"weight_map\":{"
               "\"w.weight\":\"model-00001-of-00002.safetensors\","
               "\"w.scales\":\"model-00001-of-00002.safetensors\","
               "\"w.biases\":\"model-00001-of-00002.safetensors\","
               "\"z.weight\":\"model-00002-of-00002.safetensors\","
               "\"z.scales\":\"model-00002-of-00002.safetensors\","
               "\"z.biases\":\"model-00002-of-00002.safetensors\"}}");
    fclose(f);

    if (llmcodec_mlx_to_gguf("/tmp/m1s", "/tmp/m1s.gguf", 12 /*Q4_K*/, 32, 4) != 0) { printf("FAIL mlx->gguf\n"); return 1; }
    LlmCodecGgufFileInfo fi; memset(&fi, 0, sizeof fi);
    if (llmcodec_gguf_read_info("/tmp/m1s.gguf", &fi) != 0) { printf("FAIL read info\n"); return 1; }
    int hw = 0, hz = 0;
    for (uint64_t i = 0; i < fi.n_tensors; ++i) {
        if (strcmp(fi.tensors[i].name, "w") == 0) hw = 1;
        if (strcmp(fi.tensors[i].name, "z") == 0) hz = 1;
    }
    printf("[sharded] GGUF tensori=%llu (w=%d z=%d)\n", (unsigned long long)fi.n_tensors, hw, hz);
    int ok = (fi.n_tensors == 2 && hw && hz);
    printf(ok ? "PASS\n" : "FAIL\n");
    llmcodec_gguf_free_info(&fi);
    return ok ? 0 : 1;
}
