/* llm_mlx_kvformat.c — §9.A M10c: contratto GMB_KV_FORMAT. Vedi llm_mlx_kvformat.h. */
#include "llm_mlx_kvformat.h"

#include <stdio.h>
#include <string.h>
#include <strings.h>

static const char* kvclass_name(int c) {
    switch (c) {
        case LLM_KV_DTYPE_NATIVE:         return "NATIVE";
        case LLM_KV_DTYPE_SUPPORTED:      return "SUPPORTED";
        case LLM_KV_DTYPE_QUANTIZABLE:    return "QUANTIZABLE";
        case LLM_KV_DTYPE_RECONSTRUCTABLE:return "RECONSTRUCTABLE";
        default:                          return "UNSUPPORTED";
    }
}

static int kvclass_of(const char* d) {
    if (d == NULL || !*d) return LLM_KV_DTYPE_UNSUPPORTED;
    if (!strcasecmp(d, "f16") || !strcasecmp(d, "float16")) return LLM_KV_DTYPE_SUPPORTED;
    if (!strcasecmp(d, "bf16") || !strcasecmp(d, "bfloat16")) return LLM_KV_DTYPE_SUPPORTED;
    if (!strcasecmp(d, "f32") || !strcasecmp(d, "float32")) return LLM_KV_DTYPE_SUPPORTED;
    if (!strcasecmp(d, "q8_0") || !strcasecmp(d, "int8") || !strcasecmp(d, "i8")) return LLM_KV_DTYPE_QUANTIZABLE;
    if (!strcasecmp(d, "fp8") || !strcasecmp(d, "fp8_e4m3") || !strcasecmp(d, "e4m3")) return LLM_KV_DTYPE_QUANTIZABLE;
    if (!strcasecmp(d, "fp4") || !strcasecmp(d, "mxfp4") || !strcasecmp(d, "nvfp4")) return LLM_KV_DTYPE_RECONSTRUCTABLE;
    return LLM_KV_DTYPE_UNSUPPORTED;
}

int llm_mlx_kvformat_classify(const char* native_dtype, const char* requested, char* cls_out, size_t cls_cap) {
    int c;
    if (native_dtype && requested && strcasecmp(native_dtype, requested) == 0) c = LLM_KV_DTYPE_NATIVE;
    else c = kvclass_of(requested);
    if (cls_out && cls_cap) snprintf(cls_out, cls_cap, "%s", kvclass_name(c));
    return c;
}

int llm_mlx_kvformat_validate(const char* native_dtype, const char* requested, char* reason, size_t reason_cap) {
    if (reason && reason_cap) reason[0] = 0;
    if (requested == NULL || !*requested) {
        if (reason) snprintf(reason, reason_cap, "storage dtype mancante");
        return -1;
    }
    int c = llm_mlx_kvformat_classify(native_dtype, requested, NULL, 0);
    /* NATIVE / SUPPORTED / QUANTIZABLE ammesse; RECONSTRUCTABLE ammessa SOLO se nativa (già NATIVE
     * sopra); UNSUPPORTED rifiutata. */
    if (c == LLM_KV_DTYPE_UNSUPPORTED) {
        if (reason) snprintf(reason, reason_cap, "dtype di storage '%s' NON supportato", requested);
        return -1;
    }
    if (c == LLM_KV_DTYPE_RECONSTRUCTABLE) {
        if (reason) snprintf(reason, reason_cap,
            "dtype '%s' (fp4-family) ammesso SOLO se dtype nativo del modello (addestrato): conversione runtime non consentita",
            requested);
        return -1;
    }
    return 0;
}
