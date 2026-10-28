/* qwen4_native_helpers_test.c — P0 gate: parità degli helper host vs riferimento math.h.
 * Model-free, nessun MLX/modello. Exit 0 = PASS. */
#include "qwen4_native.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static int check(const char* name, double got, double ref, double tol) {
    double d = fabs(got - ref);
    int ok = d <= tol;
    if (!ok) printf("FAIL %s: got=%.9g ref=%.9g diff=%.3g\n", name, got, ref, d);
    return ok;
}

int main(void) {
    int ok = 1;
    for (double x = -12.0; x <= 12.0; x += 0.25) {
        ok &= check("sigmoid", qwen4_native_sigmoid((float)x), 1.0 / (1.0 + exp(-x)), 1e-6);
        double sp = (x > 20 ? x : (x < -20 ? exp(x) : log(1.0 + exp(x))));
        ok &= check("softplus", qwen4_native_softplus((float)x), sp, 1e-6);
        ok &= check("silu", qwen4_native_silu((float)x), x / (1.0 + exp(-x)), 1e-5);
    }
    /* sanity costanti di forma (config reale) */
    ok &= (QWEN4_HIDDEN==2560 && QWEN4_HC_COUNT==4 && QWEN4_HC_LOWRANK==320);
    ok &= (QWEN4_EXPERTS==512 && QWEN4_TOP_K==10 && QWEN4_MOE_INTER==640);
    printf("qwen4_native P0 helpers: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
