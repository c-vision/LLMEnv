/* llm_mlx_kernel_fallback.h — rende VISIBILE il fallback dei kernel fusi.
 *
 * PERCHE' ESISTE. I kernel fusi (MoE, GDN, GeGLU, swiglu, MPP) hanno guardie di dtype/layout e
 * su rifiuto tornano false/-1: il chiamante usa allora il path eager, spesso **senza dire nulla**.
 * Un rifiuto silenzioso ci ha gia' ingannati: con `LLM_F32_SCALES` le scale vengono promosse a f32
 * e la guardia bf16 di un kernel non aggancia -> un kernel creduto attivo e non attivo (misurato
 * in sessione sull'MPP). E' la stessa lezione di due fonti esterne:
 *   - fix oMLX `736c90f`: "error + False instead of silent success" (i kernel DSA gating fp16/bf16
 *     rifiutavano in silenzio i latenti promossi a fp32, con fallback a materializzazione densa);
 *   - articolo mlx-optiq: "assert on how many expert tensors your matcher found rather than on
 *     whether it decided to stream" (un matcher che non matcha non solleva nulla).
 *
 * USO (nei kernel, subito prima del rifiuto):
 *   if (mlx_array_dtype(scales) != dt) { LLM_KFALLBACK("moe_fused", "scale dtype != attivazione"); return false; }
 *
 * COSTO. Nullo se `LLM_KERNEL_FALLBACK` non e' impostata (un `getenv` cachato). Quando e' attiva
 * stampa le prime occorrenze per (kernel, motivo), con un contatore: serve a rispondere alla domanda
 * "QUALE ramo e' stato preso" (branch-first) senza inondare e senza alterare il comportamento.
 */
#ifndef LLM_MLX_KERNEL_FALLBACK_H
#define LLM_MLX_KERNEL_FALLBACK_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static bool llm_mlx_kfallback_enabled(void) {
    static int st = -1;
    if (st < 0) st = (getenv("LLM_KERNEL_FALLBACK") != NULL) ? 1 : 0;
    return st == 1;
}

/* Una riga per (kernel, motivo), con contatore, cap a 8 occorrenze per coppia. */
static void llm_kernel_fallback_note(const char* kernel, const char* why) {
    if (!llm_mlx_kfallback_enabled()) return;
    static struct { char k[48]; char w[96]; int n; } seen[48];
    static int ns = 0;
    if (kernel == NULL) kernel = "?";
    if (why == NULL) why = "?";
    for (int i = 0; i < ns; i++)
        if (strcmp(seen[i].k, kernel) == 0 && strcmp(seen[i].w, why) == 0) {
            seen[i].n++;
            if (seen[i].n <= 8)
                fprintf(stderr, "[kfallback] %s: %s (#%d)\n", kernel, why, seen[i].n);
            return;
        }
    if (ns < 48) {
        snprintf(seen[ns].k, sizeof(seen[ns].k), "%s", kernel);
        snprintf(seen[ns].w, sizeof(seen[ns].w), "%s", why);
        seen[ns].n = 1; ns++;
        fprintf(stderr, "[kfallback] %s: %s (primo)\n", kernel, why);
    }
}

#define LLM_KFALLBACK(k, w) llm_kernel_fallback_note((k), (w))

#ifdef __cplusplus
}
#endif

#endif /* LLM_MLX_KERNEL_FALLBACK_H */
