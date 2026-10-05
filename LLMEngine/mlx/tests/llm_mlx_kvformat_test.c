/* llm_mlx_kvformat_test.c — §9.A M10c: test MODEL-FREE del contratto GMB_KV_FORMAT. */
#include "llm_mlx_kvformat.h"

#include <stdio.h>
#include <string.h>

static int fail(const char* m) { printf("FAIL: %s\n", m); return 1; }

int main(void) {
    char cls[32];

    llm_mlx_kvformat_classify("f16", "f16", cls, sizeof cls);  if (strcmp(cls, "NATIVE") != 0) return fail("f16/f16 != NATIVE");
    llm_mlx_kvformat_classify("f16", "bf16", cls, sizeof cls); if (strcmp(cls, "SUPPORTED") != 0) return fail("f16/bf16 != SUPPORTED");
    llm_mlx_kvformat_classify("f16", "q8_0", cls, sizeof cls); if (strcmp(cls, "QUANTIZABLE") != 0) return fail("f16/q8_0 != QUANTIZABLE");
    llm_mlx_kvformat_classify("f16", "fp4", cls, sizeof cls);  if (strcmp(cls, "RECONSTRUCTABLE") != 0) return fail("f16/fp4 != RECONSTRUCTABLE");
    llm_mlx_kvformat_classify("f16", "weird", cls, sizeof cls);if (strcmp(cls, "UNSUPPORTED") != 0) return fail("f16/weird != UNSUPPORTED");

    if (llm_mlx_kvformat_validate("f16", "f16", NULL, 0) != 0) return fail("validate f16/f16");
    if (llm_mlx_kvformat_validate("f16", "bf16", NULL, 0) != 0) return fail("validate f16/bf16");
    if (llm_mlx_kvformat_validate("f16", "q8_0", NULL, 0) != 0) return fail("validate f16/q8_0");

    /* fp4 NON è una conversione runtime universale: rifiutata se non nativa. */
    if (llm_mlx_kvformat_validate("f16", "fp4", NULL, 0) == 0) return fail("validate f16/fp4 doveva rifiutare");
    if (llm_mlx_kvformat_validate("f16", "mxfp4", NULL, 0) == 0) return fail("validate f16/mxfp4 doveva rifiutare");
    if (llm_mlx_kvformat_validate("f16", "weird", NULL, 0) == 0) return fail("validate f16/weird doveva rifiutare");

    /* Se il modello È nativo fp4, fp4 è ammesso. */
    if (llm_mlx_kvformat_validate("fp4", "fp4", NULL, 0) != 0) return fail("validate fp4/fp4");

    printf("OK llm_mlx_kvformat_test\n");
    return 0;
}
