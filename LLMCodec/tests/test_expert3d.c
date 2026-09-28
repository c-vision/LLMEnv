/* Round-trip 3D: esperti STACKED [E,out,in] MLX affine -> GGUF [in,out,E] -> MLX affine. */
#include "llmcodec_gguf_mlx.h"
#include "llmcodec_gguf_writer.h"
#include "llmcodec_safetensors.h"
#include "llmcodec_moe_quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define E 3
#define ROWS 4
#define COLS 256
#define GROUP 32
#define BITS 4

int main(void) {
    system("rm -rf /tmp/e3 /tmp/e3.gguf /tmp/e3out; mkdir -p /tmp/e3");
    float x[E*ROWS*COLS]; for (int i = 0; i < E*ROWS*COLS; ++i) x[i] = sinf(i*0.05f);
    size_t wbs = llmcodec_moe_affine_w_bytes(ROWS, COLS, BITS), sbs = llmcodec_moe_affine_sc_bytes(ROWS, COLS, GROUP);
    uint32_t* w = (uint32_t*)malloc(wbs*E); uint16_t* sc = (uint16_t*)malloc(sbs*E), *bs = (uint16_t*)malloc(sbs*E);
    for (int e = 0; e < E; ++e)
        llmcodec_moe_affine_quantize(x + (size_t)e*ROWS*COLS, ROWS, COLS, GROUP, BITS,
                                w + (size_t)e*(wbs/4), sc + (size_t)e*(sbs/2), bs + (size_t)e*(sbs/2));

    int sw[3] = { E, ROWS, (int)(COLS*BITS/32) }, ss[3] = { E, ROWS, (int)(COLS/GROUP) };
    LlmCodecStTensor t[3];
    t[0].name="layers.0.mlp.experts.gate_proj.weight"; t[0].dtype="U32"; t[0].shape=sw; t[0].ndim=3; t[0].data=w;
    t[1].name="layers.0.mlp.experts.gate_proj.scales"; t[1].dtype="F16"; t[1].shape=ss; t[1].ndim=3; t[1].data=sc;
    t[2].name="layers.0.mlp.experts.gate_proj.biases"; t[2].dtype="F16"; t[2].shape=ss; t[2].ndim=3; t[2].data=bs;
    if (llmcodec_safetensors_write("/tmp/e3/model.safetensors", t, 3) != 0) { printf("st write fail\n"); return 1; }

    if (llmcodec_mlx_to_gguf("/tmp/e3", "/tmp/e3.gguf", 12 /*Q4_K*/, GROUP, BITS) != 0) { printf("FAIL mlx->gguf\n"); return 1; }
    LlmCodecGgufFileInfo fi; memset(&fi, 0, sizeof fi);
    if (llmcodec_gguf_read_info("/tmp/e3.gguf", &fi) != 0) { printf("FAIL read\n"); return 1; }
    printf("[expert3d] GGUF tensori=%llu dims=[%llu,%llu,%llu] ndim=%d\n", (unsigned long long)fi.n_tensors,
           (unsigned long long)fi.tensors[0].dims[0], (unsigned long long)fi.tensors[0].dims[1],
           (unsigned long long)fi.tensors[0].dims[2], fi.tensors[0].ndim);
    int ok = (fi.n_tensors == 1 && fi.tensors[0].ndim == 3
              && fi.tensors[0].dims[0] == COLS && fi.tensors[0].dims[1] == ROWS && fi.tensors[0].dims[2] == E);
    llmcodec_gguf_free_info(&fi);

    if (llmcodec_gguf_to_mlx("/tmp/e3.gguf", "/tmp/e3out", GROUP, BITS) != 0) { printf("FAIL gguf->mlx\n"); return 1; }
    LlmCodecStTensorInfo* ti = NULL; int n = 0; uint64_t base = 0;
    if (llmcodec_safetensors_read("/tmp/e3out/model.safetensors", &ti, &n, &base) != 0) { printf("FAIL read m2\n"); return 1; }
    int iw=-1,is=-1,ib=-1; for (int i=0;i<n;i++){ if(!strcmp(ti[i].name,"layers.0.mlp.experts.gate_proj.weight"))iw=i;
        if(!strcmp(ti[i].name,"layers.0.mlp.experts.gate_proj.scales"))is=i;
        if(!strcmp(ti[i].name,"layers.0.mlp.experts.gate_proj.biases"))ib=i; }
    printf("[expert3d] m2 w=%d s=%d b=%d ndim_w=%d\n", iw, is, ib, iw>=0?ti[iw].ndim:-1);
    ok = ok && iw>=0 && is>=0 && ib>=0 && ti[iw].ndim==3 && ti[iw].shape[0]==E;
    if (ok) {
        FILE* f = fopen("/tmp/e3out/model.safetensors", "rb");
        size_t wl = (size_t)(ti[iw].end-ti[iw].begin), sl = (size_t)(ti[is].end-ti[is].begin);
        uint8_t* wr=(uint8_t*)malloc(wl), *sr=(uint8_t*)malloc(sl), *br=(uint8_t*)malloc(sl);
        fseeko(f, base+ti[iw].begin, SEEK_SET); fread(wr,1,wl,f);
        fseeko(f, base+ti[is].begin, SEEK_SET); fread(sr,1,sl,f);
        fseeko(f, base+ti[ib].begin, SEEK_SET); fread(br,1,sl,f); fclose(f);
        double md = 0;
        for (int e = 0; e < E; ++e) {
            float y[ROWS*COLS];
            llmcodec_moe_affine_dequantize((const uint32_t*)(wr + (size_t)e*wl/E), (const uint16_t*)(sr + (size_t)e*sl/E),
                                      (const uint16_t*)(br + (size_t)e*sl/E), ROWS, COLS, GROUP, BITS, 1, y);
            for (int i = 0; i < ROWS*COLS; ++i) { double d = fabs(y[i]-x[(size_t)e*ROWS*COLS+i]); if (d>md) md=d; }
        }
        printf("[expert3d] round-trip maxerr=%.4e\n", md);
        ok = ok && md < 0.5;
        free(wr); free(sr); free(br);
    }
    free(w); free(sc); free(bs);
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
