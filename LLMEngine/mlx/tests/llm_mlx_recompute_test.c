/* llm_mlx_recompute_test.c — §9.A M10a: test MODEL-FREE della validazione dell'azione RECOMPUTE. */
#include "llmengine_mlx.h"

#include <stdio.h>
#include <stdint.h>

static int fail(const char* m) { printf("FAIL: %s\n", m); return 1; }

int main(void) {
    int32_t t[4] = { 1, 2, 3, 4 };
    /* Handle NULL -> rifiutato (nessun deref). */
    if (llm_ai_mlx_session_recompute_range(NULL, NULL, NULL, 0, 1) != -1) return fail("null");
    /* Range invalido -> rifiutato PRIMA di toccare la sessione. */
    if (llm_ai_mlx_session_recompute_range((void*)1, (void*)1, t, -1, 2) != -1) return fail("from<0");
    if (llm_ai_mlx_session_recompute_range((void*)1, (void*)1, t, 2, 2) != -1) return fail("from==to");
    if (llm_ai_mlx_session_recompute_range((void*)1, (void*)1, t, 3, 2) != -1) return fail("to<from");
    printf("OK llm_mlx_recompute_test\n");
    return 0;
}
