/* MLX mxfp4 (fmt 4), nvfp4 (fmt 5, global opzionale), e5m2 (fmt 3) — via registry (plugin fp8). */
#include "llmcodec.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
int main(void) {
    const LlmCodec* c = llmcodec_resolve("fp8");
    if (!c) { printf("FAIL resolve fp8\n"); return 1; }
    const int R = 4, C = 64;

    /* mxfp4 (fmt 4): weight U32 [R,C/8] + scala u8 e8m0 [R,C/32]; nibble 2 = 1.0, 127 = 1.0 */
    uint32_t w[R*8]; for (int i = 0; i < R*8; i++) w[i] = 0x22222222u;
    uint8_t sc[R*2]; for (int i = 0; i < R*2; i++) sc[i] = 127;
    float out[R*C];
    if (llmcodec_resolve("fp4")->decode(w, sizeof w, 1, 1, 32, R, C, sc, sizeof sc, out) != 0) { printf("FAIL mxfp4 decode\n"); return 1; }
    double md = 0; for (int i = 0; i < R*C; i++) { double d = fabs(out[i]-1.0); if (d > md) md = d; }

    /* e5m2 (fmt 3) */
    uint8_t e5[4] = { 0x3C, 0xBC, 0x00, 0x7C }; float o5[4];
    if (c->decode(e5, 4, 3, 0, 0, 1, 4, NULL, 0, o5) != 0) { printf("FAIL e5m2 decode\n"); return 1; }

    /* nvfp4 (fmt 5): weight U32 [R,C/8] + scala E4M3 u8 [R,C/16]; nibble 2 = 1.0, 0x38 = 1.0 */
    uint8_t scn[R*4]; for (int i = 0; i < R*4; i++) scn[i] = 0x38;
    float on[R*C];
    if (llmcodec_resolve("nvfp")->decode(w, sizeof w, 0, 0, 16, R, C, scn, sizeof scn, on) != 0) { printf("FAIL nvfp4 decode\n"); return 1; }
    double mdn = 0; for (int i = 0; i < R*C; i++) { double d = fabs(on[i]-1.0); if (d > mdn) mdn = d; }

    /* nvfp4 + GLOBAL (OCP/NVIDIA): scales u8 [R,C/16] + global f32 = 2.0 in coda */
    uint8_t aux2[R*4 + 4]; memcpy(aux2, scn, R*4); float g = 2.0f; memcpy(aux2 + R*4, &g, 4);
    float og[R*C];
    if (llmcodec_resolve("nvfp")->decode(w, sizeof w, 0, 0, 16, R, C, aux2, sizeof aux2, og) != 0) { printf("FAIL nvfp4+global\n"); return 1; }
    double mdg = 0; for (int i = 0; i < R*C; i++) { double d = fabs(og[i]-2.0); if (d > mdg) mdg = d; }

    printf("[mxfp4/nvfp4/e5m2] mxfp4=%.2e nvfp4=%.2e nvfp4_g=%.2e e5m2=[%g %g %g %g]\n",
           md, mdn, mdg, o5[0], o5[1], o5[2], o5[3]);
    int ok = (md < 1e-5) && (mdn < 1e-5) && (mdg < 1e-5)
             && fabsf(o5[0]-1.0f) < 1e-6 && fabsf(o5[1]+1.0f) < 1e-6 && o5[2] == 0.0f && isinf(o5[3]);
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
