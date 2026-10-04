/* llm_dsv41_fp8_test.c — packing FP8 e4m3 vs riferimento MLX.
 *   cc -O2 -o /tmp/llm_dsv41_fp8_test native/llmengine_mlx/llm_dsv41_fp8_test.c -lm */
#include <stdio.h>
#include "llm_dsv41_fp8.h"
#include "llm_dsv41_fp8_ref.h"
static int g_fail = 0;
static int nclose(const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) { float d = a[i]-b[i]; if (d<0) d=-d; float m = b[i]<0?-b[i]:b[i]; if (m<1.0f) m=1.0f;
        if (d > 1e-3f*m) { printf("  [%d] got=%.7g exp=%.7g\n", i, a[i], b[i]); return 0; } }
    return 1;
}
static int beq(const unsigned char* a, const unsigned char* b, int n) {
    for (int i = 0; i < n; i++) if (a[i] != b[i]) { printf("  byte[%d] got=%d exp=%d\n", i, a[i], b[i]); return 0; }
    return 1;
}
#define CHECK(cond, name, ...) do { if (cond) printf("PASS %-42s\n", name); else { printf("FAIL %-42s | ", name); printf(__VA_ARGS__); printf("\n"); g_fail++; } } while (0)
int main(void) {
    const int L = REF_FP8_L, W = REF_FP8_W, B = REF_FP8_BYTES;
    printf("llm_dsv41_fp8 — packing e4m3\n");
    unsigned char packed[128]; float un[128];
    for (int l = 0; l < L; l++) llmcodec_pack_fp8(REF_FP8_PRE + l * W, W, packed + l * B);
    CHECK(beq(packed, REF_FP8_PACKED, L * B), "T1 pack byte-identico", "");
    for (int l = 0; l < L; l++) llmcodec_unpack_fp8(REF_FP8_PACKED + l * B, W, un + l * W);
    CHECK(nclose(un, REF_FP8_UNPACKED, L * W), "T2 unpack del packed di riferimento", "");
    for (int l = 0; l < L; l++) llmcodec_unpack_fp8(packed + l * B, W, un + l * W);
    CHECK(nclose(un, REF_FP8_UNPACKED, L * W), "T3 unpack del nostro packed", "");
    if (g_fail == 0) printf("ALL PASS\n"); else printf("FAILURES: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
