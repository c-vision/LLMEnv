/* llm_oq_a8_selftest_test.c — driver CLI del self-test A8/OQ GEMV dell'engine MLX:
 * chiama llm_ai_mlx_oq_a8_selftest(16,64,64) e stampa l'esito. */
#include "llmengine_mlx.h"
#include <stdio.h>
int main(void) {
    char buf[192];
    int rc = llm_ai_mlx_oq_a8_selftest(16, 64, 64, buf, (int)sizeof buf);
    printf("rc=%d  %s\n", rc, buf);
    return rc == 0 ? 0 : 1;
}
