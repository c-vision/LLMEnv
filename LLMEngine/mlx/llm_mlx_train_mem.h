/* llm_mlx_train_mem.h — §D R8: MEMORY ESTIMATOR per il training (LoRA/full).
 *
 * Stima il picco di memoria di un run di training dato il modello e la config, con un
 * breakdown esplicito (pesi + trainable + gradi + stati AdamW + attivazioni). Serve a
 * decidere se una config entra in RAM PRIMA di lanciarla (gate prodotto R5, macchine piccole).
 *
 * Modello (approssimazioni DOCUMENTATE, non un oracolo):
 *  - pesi: `params * bytes_per_weight` (quant: 4-bit ~0.55 B/param con scales, 8-bit ~1.1, f16 2, f32 4);
 *  - trainable: LoRA = sum(in*rank + rank*out) sui target; full = params;
 *  - gradi: trainable * 4 (f32);
 *  - AdamW: trainable * 8 (m + v f32);
 *  - attivazioni: `batch * seq * hidden * dtype_bytes * act_factor` (act_factor ~ costo
 *    fwd+bwd per layer relativo a hidden; default 8, misurabile). Contributo dominante a seq
 *    grandi: e' il termine da calibrare (LLM_PHASE/memprof).
 * PURO. */
#ifndef LLM_MLX_TRAIN_MEM_H
#define LLM_MLX_TRAIN_MEM_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef enum { LLM_Q4 = 0, LLM_Q8, LLM_F16, LLM_F32 } LlmWeightDtype;

typedef struct {
    uint64_t params;        /* parametri totali del modello */
    LlmWeightDtype wdt;     /* dtype/quant dei pesi a riposo */
    int full_finetune;      /* 1 = full (trainable = params), 0 = LoRA */
    int lora_rank;          /* rank LoRA (se !full) */
    uint64_t lora_in_out;   /* somma (in+out) sui target LoRA (se !full): trainable = rank * lora_in_out */
    int batch;
    int seq;
    int hidden;
    float act_factor;       /* default 8 */
    float overhead;         /* moltiplicatore per temporanei/allocator (default 1.15) */
} LlmTrainMemCfg;

typedef struct {
    double weights_mb, trainable_mb, grads_mb, opt_mb, activ_mb, total_mb, peak_mb;
    uint64_t trainable_params;
} LlmTrainMemEst;

static inline double llm_bytes_per_weight(LlmWeightDtype d) {
    switch (d) {
        case LLM_Q4:  return 0.55;   /* 4-bit + scales/biases (misurato ~0.55 B/param su 4-bit g64) */
        case LLM_Q8:  return 1.1;    /* 8-bit + scales (~1.06) */
        case LLM_F16: return 2.0;
        case LLM_F32: return 4.0;
        default:      return 2.0;
    }
}

static inline LlmTrainMemEst llm_train_mem_estimate(const LlmTrainMemCfg* c) {
    LlmTrainMemEst e; memset(&e, 0, sizeof(e));
    const double MB = 1024.0 * 1024.0;
    double af = c->act_factor > 0 ? c->act_factor : 8.0f;
    double ov = c->overhead > 0 ? c->overhead : 1.15f;

    uint64_t trainable = c->full_finetune ? c->params : (uint64_t)c->lora_rank * c->lora_in_out;
    e.trainable_params = trainable;
    e.weights_mb  = (double)c->params * llm_bytes_per_weight(c->wdt) / MB;
    /* full: i trainable SONO i pesi (gia' contati in weights) -> nessun doppio conteggio. */
    e.trainable_mb = c->full_finetune ? 0.0 : (double)trainable * 4.0 / MB;
    e.grads_mb    = (double)trainable * 4.0 / MB;
    e.opt_mb      = (double)trainable * 8.0 / MB;          /* AdamW: m + v */
    e.activ_mb    = (double)c->batch * c->seq * c->hidden * 4.0 * af / MB;   /* f32 attivazioni */
    e.total_mb    = e.weights_mb + e.trainable_mb + e.grads_mb + e.opt_mb + e.activ_mb;
    e.peak_mb     = e.total_mb * ov;
    return e;
}

/* Verdetto vs budget (MB). Ritorna 1 se entra, 0 no. */
static inline int llm_train_mem_fits(const LlmTrainMemEst* e, double budget_mb) {
    return e->peak_mb <= budget_mb;
}

#endif /* LLM_MLX_TRAIN_MEM_H */
