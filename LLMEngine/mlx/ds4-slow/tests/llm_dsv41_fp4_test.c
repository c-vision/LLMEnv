/* llm_dsv41_fp4_test.c — packing FP4 E2M1 (e FP8 scala e4m3) vs riferimento MLX.
 *   cc -O2 -o /tmp/llm_dsv41_fp4_test native/llmengine_mlx/llm_dsv41_fp4_test.c -lm */
#include <stdio.h>
#include "llm_dsv41_fp4.h"
#include "llm_dsv41_fp4_ref.h"
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

#define CASE(N) do { \
    const int L = REF_##N##_L, W = REF_##N##_W, S = REF_##N##_BYTES / L; \
    unsigned char packed[512]; float un[512]; \
    for (int l = 0; l < L; l++) \
        llmcodec_pack_activation(REF_##N##_PRE + (size_t)l * W, W, REF_##N##_BITS, \
                                  REF_##N##_GS, REF_##N##_E4, packed + (size_t)l * S); \
    CHECK(beq(packed, REF_##N##_PACKED, L * S), "pack " #N " byte-identico", ""); \
    for (int l = 0; l < L; l++) \
        llmcodec_unpack_activation(REF_##N##_PACKED + (size_t)l * S, W, REF_##N##_BITS, \
                                    REF_##N##_GS, REF_##N##_E4, un + (size_t)l * W); \
    CHECK(nclose(un, REF_##N##_UNPACKED, L * W), "unpack " #N " (rif)", ""); \
    for (int l = 0; l < L; l++) \
        llmcodec_unpack_activation(packed + (size_t)l * S, W, REF_##N##_BITS, \
                                    REF_##N##_GS, REF_##N##_E4, un + (size_t)l * W); \
    CHECK(nclose(un, REF_##N##_UNPACKED, L * W), "unpack " #N " (nostro)", ""); \
} while (0)

int main(void) {
    printf("llm_dsv41_fp4 — packing E2M1 / e4m3\n");
    CASE(A);   /* bits=4 gs=32 e4m3=0 : index q/key */
    CASE(B);   /* bits=4 gs=16 e4m3=1 : KV compresso */
    CASE(C);   /* bits=8 gs=16 e4m3=1 : variante */
    if (g_fail == 0) printf("ALL PASS\n"); else printf("FAILURES: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
