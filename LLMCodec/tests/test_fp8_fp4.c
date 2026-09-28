/* Verifica i dequant/quant fp8/fp4 DS4.1 (blocchi + round-trip encoder->decoder). */
#include "llmcodec_fp8_fp4.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

int main(void) {
    /* 1) blocco fp8 [64,64]: w=0x38 (1.0), scala ue8m0 byte 127 (=1.0) su griglia [2,2] */
    uint8_t w8[64*64]; uint8_t sc8[2*2];
    memset(w8, 0x38, sizeof w8); memset(sc8, 127, sizeof sc8);
    float d8[64*64];
    llmcodec_dsv41_dequant_fp8_block(w8, sc8, 64, 64, d8);
    int bad8 = 0; for (int i = 0; i < 64*64; ++i) if (fabsf(d8[i] - 1.0f) > 1e-6f) bad8++;

    /* 2) esperto fp4 [2,64]: byte 0x21 -> low code 1 (0.5), high code 2 (1.0); scala 127 (=1.0) */
    uint8_t w4[2*32]; uint8_t sc4[2*2];
    memset(w4, 0x21, sizeof w4); memset(sc4, 127, sizeof sc4);
    float d4[2*64];
    llmcodec_dsv41_dequant_fp4_expert(w4, sc4, 2, 64, d4);
    int bad4 = 0;
    for (int r = 0; r < 2; ++r) for (int c = 0; c < 64; ++c) {
        float exp = (c & 1) ? 1.0f : 0.5f;
        if (fabsf(d4[r*64+c] - exp) > 1e-6f) bad4++;
    }
    printf("[fp8fp4] fp8_block bad=%d  fp4_expert bad=%d\n", bad8, bad4);

    /* 3) round-trip ENCODER -> DECODER (fp8 block [32,32] e fp8 row) */
    float src[64*64]; for (int i = 0; i < 64*64; ++i) src[i] = sinf(i*0.11f)*3.0f;
    uint8_t ew[64*64], es[2*2]; float rt[64*64];
    llmcodec_dsv41_quant_fp8_block(src, ew, es, 64, 64);
    llmcodec_dsv41_dequant_fp8_block(ew, es, 64, 64, rt);
    double blk_rel = 0; for (int i = 0; i < 64*64; ++i) { double d = fabs(rt[i]-src[i])/(fabs(src[i])+1e-3); if (d>blk_rel) blk_rel=d; }
    float src2[32*64]; for (int i = 0; i < 32*64; ++i) src2[i] = cosf(i*0.07f)*2.0f;
    uint8_t ew2[32*64], es2[32*2]; float rt2[32*64];
    llmcodec_dsv41_quant_fp8_row(src2, ew2, es2, 32, 64);
    llmcodec_dsv41_dequant_fp8_row(ew2, es2, 32, 64, rt2);
    double row_rel = 0; for (int i = 0; i < 32*64; ++i) { double d = fabs(rt2[i]-src2[i])/(fabs(src2[i])+1e-3); if (d>row_rel) row_rel=d; }
    printf("[fp8fp4] round-trip block rel=%.3e  row rel=%.3e\n", blk_rel, row_rel);

    /* 4) pack/unpack attivazioni: fp8 (grp32 ue8m0) e fp4 (grp16 e4m3), round-trip */
    int W = 256; float ax[256]; for (int i = 0; i < W; ++i) ax[i] = 0.7f*sinf(i*0.3f) - 0.2f;
    uint8_t pk[256 + 8]; float un[256];
    llmcodec_pack_fp8(ax, W, pk); llmcodec_unpack_fp8(pk, W, un);
    double p8 = 0; for (int i = 0; i < W; ++i) { double d = fabs(un[i]-ax[i])/(fabs(ax[i])+1e-2); if (d>p8) p8=d; }
    uint8_t pk4[256/2 + 16]; float un4[256];
    llmcodec_pack_activation(ax, W, 4, 16, 1, pk4); llmcodec_unpack_activation(pk4, W, 4, 16, 1, un4);
    double p4 = 0; for (int i = 0; i < W; ++i) { double d = fabs(un4[i]-ax[i]); if (d>p4) p4=d; }   /* fp4: errore ASSOLUTO */
    printf("[fp8fp4] pack/unpack fp8 rel=%.3e  fp4(gs16,e4m3) abs=%.3e\n", p8, p4);

    int ok = (bad8 == 0 && bad4 == 0 && blk_rel < 0.06 && row_rel < 0.06 && p8 < 0.06 && p4 < 0.18);
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
