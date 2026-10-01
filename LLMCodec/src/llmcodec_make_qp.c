/* llmcodec_make_qp.c — helper APACHE usato dall'encode K-quant (ggml_quants.c):
 * `llmcodec_make_qp_quants` (scala del blocco, variante imatrix) + il suo `nearest_int`.
 * L'IQ-ENCODE (iq2/iq1/iq3) e' PUL (gmbllm); qui resta solo cio' che serve al runtime. */
#include "llmcodec_iq2_xxs.h"
#include <math.h>
#include <string.h>
#define IQ2_GROUP_MAX_EPS 1e-15f

static int llmcodec_iq2_nearest_int(float fval) {
    float v = fval + 12582912.0f; int i; memcpy(&i, &v, 4);
    return (i & 0x007fffff) - 0x00400000;
}


float llmcodec_make_qp_quants(int n, int nmax, const float* x, uint8_t* L, const float* quant_weights) {
    float mx = 0;
    for (int i = 0; i < n; ++i) mx = mx > x[i] ? mx : x[i];
    if (mx < IQ2_GROUP_MAX_EPS) { for (int i = 0; i < n; ++i) L[i] = 0; return 0.f; }
    float iscale = nmax / mx;
    for (int i = 0; i < n; ++i) L[i] = (uint8_t)llmcodec_iq2_nearest_int(iscale * x[i]);
    float scale = 1/iscale, best_mse = 0;
    for (int i = 0; i < n; ++i) { float diff = x[i] - scale*L[i]; float w = quant_weights[i]; best_mse += w*diff*diff; }
    for (int is = -4; is <= 4; ++is) {
        if (is == 0) continue;
        float isc = (0.1f*is + nmax)/mx, sc = 1/isc, mse = 0;
        for (int i = 0; i < n; ++i) {
            int l = llmcodec_iq2_nearest_int(isc*x[i]); l = l > nmax ? nmax : l;
            float diff = x[i] - sc*l; float w = quant_weights[i]; mse += w*diff*diff;
        }
        if (mse < best_mse) { best_mse = mse; iscale = isc; }
    }
    float sumlx = 0, suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = llmcodec_iq2_nearest_int(iscale * x[i]); l = l > nmax ? nmax : l; L[i] = (uint8_t)l;
        float w = quant_weights[i]; sumlx += w*x[i]*l; suml2 += w*l*l;
    }
    for (int itry = 0; itry < 5; ++itry) {
        int nch = 0;
        for (int i = 0; i < n; ++i) {
            float w = quant_weights[i];
            float slx = sumlx - w*x[i]*L[i], sl2 = suml2 - w*L[i]*L[i];
            if (slx > 0 && sl2 > 0) {
                int nl = llmcodec_iq2_nearest_int(x[i] * sl2 / slx); nl = nl > nmax ? nmax : nl;
                if (nl != L[i]) {
                    slx += w*x[i]*nl; sl2 += w*nl*nl;
                    if (slx*slx*suml2 > sumlx*sumlx*sl2) { L[i] = (uint8_t)nl; sumlx = slx; suml2 = sl2; ++nch; }
                }
            }
        }
        if (!nch) break;
    }
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

