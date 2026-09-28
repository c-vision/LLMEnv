/* llm_mlx_kernel.h — RUNTIME KERNEL REPLACEMENT (roadmap 8.11 fase 3 Safe Surgery).
 *
 * Registro di implementazioni ("kernel") selezionabili per NOME a runtime, senza
 * ricompilare. E' l'Operator Registry del piano (paragrafo 8.2): l'architettura puo'
 * scegliere l'implementazione ottimale di un'operazione (SDPA full/sliding, quant
 * affine/mxfp4/nvfp4, kernel alternativi) in base a config/hardware/workload.
 *
 * NOTA: le firme dei kernel reali differiscono per operazione. Questo modulo fornisce
 * il meccanismo generico (register/get/dispatch per nome + conteggio) come fondamento;
 * l'associazione a operazioni specifiche (es. una LlmKernelFn per la SDPA) si fa nel
 * motore. La smoketest usa kernel di esempio con firma void(int).
 *
 * Convenzione (come llm_mlx_ops.h): puro C, nessuno stato globale.
 */
#ifndef LLM_MLX_KERNEL_H
#define LLM_MLX_KERNEL_H

#include <stdbool.h>
#include <string.h>

#define LLM_MAX_KERNELS 32

/* Firma di esempio per un kernel (adattare per operazione nel motore). */
typedef void (*LlmKernelFn)(int arg);

typedef struct {
    const char* name;
    LlmKernelFn fn;
} LlmKernelEntry;

typedef struct {
    LlmKernelEntry entries[LLM_MAX_KERNELS];
    int count;
} LlmKernelRegistry;

static inline void llm_kernel_registry_init(LlmKernelRegistry* r) {
    if (!r) return;
    r->count = 0;
}

/* Registra un kernel per nome. Ritorna true se registrato (o sostituito). */
static inline bool llm_kernel_register(LlmKernelRegistry* r, const char* name, LlmKernelFn fn) {
    if (!r || !name || !fn) return false;
    for (int i = 0; i < r->count; i++) {
        if (strcmp(r->entries[i].name, name) == 0) { /* sostituzione a runtime */
            r->entries[i].fn = fn;
            return true;
        }
    }
    if (r->count >= LLM_MAX_KERNELS) return false;
    r->entries[r->count].name = name;
    r->entries[r->count].fn = fn;
    r->count++;
    return true;
}

/* Dispatcha il kernel per nome. Ritorna true se trovato e chiamato, false se assente. */
static inline bool llm_kernel_dispatch(LlmKernelRegistry* r, const char* name, int arg) {
    if (!r || !name) return false;
    for (int i = 0; i < r->count; i++) {
        if (strcmp(r->entries[i].name, name) == 0) {
            r->entries[i].fn(arg);
            return true;
        }
    }
    return false;
}

/* Numero di kernel registrati. */
static inline int llm_kernel_count(const LlmKernelRegistry* r) {
    return r ? r->count : 0;
}

#endif /* LLM_MLX_KERNEL_H */
