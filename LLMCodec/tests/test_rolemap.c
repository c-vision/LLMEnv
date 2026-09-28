/* Verifica la mappa ruoli (.llmarch) + il converter role-aware. Self-contained. */
#include "llmcodec_rolemap.h"
#include "llmcodec_gguf_mlx.h"
#include "llmcodec_gguf_writer.h"
#include "llmcodec_safetensors.h"
#include "llmcodec_moe_quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef LLMCODEC_ABI_DIR
#define LLMCODEC_ABI_DIR "../../ABI"
#endif

static void mk_fixture(void) {
    system("rm -rf /tmp/mr /tmp/mr.gguf; mkdir -p /tmp/mr");
    const uint32_t rows = 4, cols = 256, group = 32, bits = 4;
    float x[1024]; for (uint32_t i = 0; i < rows*cols; ++i) x[i] = sinf(i*0.3f);
    size_t wb = llmcodec_moe_affine_w_bytes(rows, cols, bits), sb = llmcodec_moe_affine_sc_bytes(rows, cols, group);
    uint32_t* w = (uint32_t*)malloc(wb); uint16_t* sc = (uint16_t*)malloc(sb), *bs = (uint16_t*)malloc(sb);
    llmcodec_moe_affine_quantize(x, rows, cols, group, bits, w, sc, bs);
    int shp[2] = { (int)rows, (int)(cols*bits/32) }, shp2[2] = { (int)rows, (int)(cols/group) };
    LlmCodecStTensor t[3];
    t[0].name="model.layers.0.self_attn.q_proj.weight"; t[0].dtype="U32"; t[0].shape=shp;  t[0].ndim=2; t[0].data=w;
    t[1].name="model.layers.0.self_attn.q_proj.scales"; t[1].dtype="F16"; t[1].shape=shp2; t[1].ndim=2; t[1].data=sc;
    t[2].name="model.layers.0.self_attn.q_proj.biases"; t[2].dtype="F16"; t[2].shape=shp2; t[2].ndim=2; t[2].data=bs;
    llmcodec_safetensors_write("/tmp/mr/model.safetensors", t, 3);
    free(w); free(sc); free(bs);
}

int main(void) {
    char path[512]; snprintf(path, sizeof path, "%s/olmoe.llmarch", LLMCODEC_ABI_DIR);
    LlmCodecRoleMap rm;
    if (llmcodec_rolemap_load(path, &rm) != 0) { printf("FAIL load %s\n", path); return 1; }
    printf("[rolemap] arch=%s ruoli=%d scale_suffix=%s\n", rm.arch, rm.n, rm.scale_suffix);
    const char* r_emb = llmcodec_rolemap_of(&rm, "model.embed_tokens.weight");
    const char* r_exp = llmcodec_rolemap_of(&rm, "model.layers.0.mlp.experts.3.gate_proj.weight");
    const char* r_attn = llmcodec_rolemap_of(&rm, "model.layers.0.self_attn.q_proj.weight");
    const char* r_other = llmcodec_rolemap_of(&rm, "tensor_inesistente");
    printf("[rolemap] embed=%s expert=%s q=%s altro=%s is_expert=%d\n",
           r_emb, r_exp, r_attn, r_other, llmcodec_rolemap_is_expert(r_exp));
    int ok = strcmp(rm.arch, "olmoe") == 0 && rm.n > 0
             && strcmp(r_emb, "EMBEDDING") == 0
             && llmcodec_rolemap_is_expert(r_exp) == 1
             && strcmp(r_other, "OTHER") == 0;

    mk_fixture();
    if (llmcodec_conv_mlx_to_gguf_arch("/tmp/mr", "/tmp/mr.gguf", path, 12 /*Q4_K*/, 32, 4) != 0) {
        printf("FAIL conv arch\n"); return 1;
    }
    LlmCodecGgufFileInfo fi; memset(&fi, 0, sizeof fi);
    if (llmcodec_gguf_read_info("/tmp/mr.gguf", &fi) != 0) { printf("FAIL read\n"); return 1; }
    printf("[rolemap] GGUF tensori=%llu\n", (unsigned long long)fi.n_tensors);
    ok = ok && fi.n_tensors == 1;
    llmcodec_gguf_free_info(&fi);

    /* A1 role-aware: il config.json deve contenere l'architettura "olmoe". */
    system("rm -rf /tmp/mr_mlx");
    if (llmcodec_conv_gguf_to_mlx_arch("/tmp/mr.gguf", "/tmp/mr_mlx", path, 32, 4) != 0) {
        printf("FAIL conv gguf->mlx arch\n"); return 1;
    }
    char cj[4096] = {0};
    { FILE* f = fopen("/tmp/mr_mlx/config.json", "rb");
      if (f) { size_t r = fread(cj, 1, sizeof cj - 1, f); cj[r] = 0; fclose(f); } }
    printf("[rolemap] config.json: %s", cj);
    ok = ok && (strstr(cj, "\"olmoe\"") != NULL);
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
