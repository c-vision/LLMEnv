/* llm_mlx_which_path.h — item W1 (2026-09-17): DUMP DEL RAMO ATTIVO per i percorsi critici.
 *
 * PERCHE' (regola AGENTS §0sexies, branch-first): in un motore con decine di flag/fallback, una
 * regressione di 5-10x e' quasi sempre il RAMO SBAGLIATO (flag non impostato, fallback silenzioso,
 * condizione di eleggibilita'), non un kernel lento. Prima di misurare si deve far dire all'engine
 * QUALE percorso sta girando e PER QUALE flag. Caso reale: `LLM_KVQUANT=1` da solo non attivava il
 * kernel dequant-on-read (serviva un secondo flag separato) -> percorso legacy O(nc)/step per giorni
 * invisibile (vedi Hard-Lessons/Ramo-Sbagliato-Prima-Del-Kernel.md).
 *
 * COSTO: una volta per load (non per token). Attivato da LLM_WHICH_PATH=1; se non attivo, no-op.
 * USO: riempire la struct al call-site (dove gli accessor/le config sono in scope) e chiamare
 * llm_mlx_which_path_dump(&w). I campi sono 0/1 per i flag booleani, -1 = non applicabile.
 */
#ifndef LLM_MLX_WHICH_PATH_H
#define LLM_MLX_WHICH_PATH_H

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    const char* model_type;   /* es. qwen3_5_moe */
    const char* dt_name;      /* "bf16" | "f16" | "fp32" (flavor di attivazione attivo)  */
    const char* dtype_src;    /* chi l'ha deciso: "cfg" | "LLM_BF16_DENSE" | "LLM_ACT16" | ... */
    const char* quant_name;   /* "affine" | "mxfp4" | "none" */
    int quant_bits;
    /* KV */
    int kv_read;              /* 1 = dequant-on-read (kernel Metal), 0 = percorso fp16 */
    int kv_read_src;          /* 0=auto(kv_quantize) 1=LLM_KV_DEQUANT_READ 2=n/a */
    int kv_inplace;           /* LLM_KV_NOSLICE/append in-place attivo */
    int kv_slice;             /* LLM_KVSLICE (buffer a capacita' statica) */
    int kv_seg;               /* LLM_KVSEG */
    int kv_f16;               /* KV forzata f16 */
    int kv_metal;             /* LLMMETAL_KVMETAL (pool Metal per layer) */
    int kv_max_tokens;        /* rolling window (>0 = attiva) */
    int kv_quant;             /* cfg.kv_quantize */
    int kv_quant_bits;        /* cfg.kv_quantize_bits */
    int kv_quant_thresh;      /* cfg.kv_quantize_threshold */
    /* MoE */
    int moe_fuse;             /* LLMENGINE_MOE_FUSE_MOE_DECODE */
    int moe_mxfp4;            /* LLMENGINE_MOE_FUSE_MOE_MXFP4 */
    int moe_gateup;           /* LLMENGINE_MOE_GATEUP_FUSE */
    int moe_n;                /* n. esperti / top_k */
    int moe_topk;
    /* attenzione */
    int at_flash_decode;      /* LLM_GEMMA_FLASH_DECODE */
    int at_flash_prefill;     /* LLM_GEMMA_FLASH_PREFILL */
    int at_fuse_rope;         /* LLM_FUSE_ROPE */
    int at_sliding;           /* 1 = il modello ha layer sliding */
    int at_sinks;             /* 1 = il modello usa attention sinks */
    /* pesi / quantizzato */
    int w_packed_qkv;         /* LLM_PACKED_QKV */
    int w_wcache;             /* cache dei lookup pesi (LLM_WCACHE) */
    int w_reshape_canon;      /* LLM_RESHAPE_CANONICAL */
    int w_scales_f32;         /* hoisting scale fp32 */
    int w_scales_f16;         /* hoisting scale f16 */
    int w_scales_bf16;        /* hoisting scale bf16 (opt-in) */
    int w_scales_promoted;    /* quanti tensori ha promosso il loader (0 = nessuno) */
    int w_gemvq4;             /* LLM_GEMVQ4 */
    int w_oq_a8;              /* LLMMETAL_OQ_A8 */
    /* esecuzione */
    int x_compile;            /* LLM_COMPILE */
    int x_fixkv;              /* LLM_FIXKV */
    int x_compile_dense;      /* LLM_COMPILE_DENSE */
    int x_async;              /* LLM_ASYNC_DECODE */
    int x_batch_decode;       /* LLM_BATCH_DECODE */
    int x_sampler_legacy;     /* LLM_SAMPLER_LEGACY */
} LlmWhichPath;

static void llm_mlx_which_path_dump(const LlmWhichPath* w) {
    if (w == NULL) return;
    fprintf(stderr,
        "[which-path] model=%s quant=%s(bits=%d) attivazione=%s (da %s)\n",
        w->model_type ? w->model_type : "?", w->quant_name ? w->quant_name : "?", w->quant_bits,
        w->dt_name ? w->dt_name : "?", w->dtype_src ? w->dtype_src : "?");
    fprintf(stderr,
        "  KV      : read=%s (da %s) | inplace=%d kvslice=%d kvseg=%d kv_f16=%d kvmetal=%d | "
        "max_tokens=%d | quant=%d bits=%d thresh=%d\n",
        w->kv_read ? "dequant-on-read" : "fp16", w->kv_read_src == 1 ? "LLM_KV_DEQUANT_READ" : (w->kv_read_src == 0 ? "auto(cfg.kv_quantize)" : "n/a"),
        w->kv_inplace, w->kv_slice, w->kv_seg, w->kv_f16, w->kv_metal,
        w->kv_max_tokens, w->kv_quant, w->kv_quant_bits, w->kv_quant_thresh);
    fprintf(stderr,
        "  MoE     : fuse=%d mxfp4=%d gateup_fuse=%d | esperti=%d top_k=%d\n",
        w->moe_fuse, w->moe_mxfp4, w->moe_gateup, w->moe_n, w->moe_topk);
    fprintf(stderr,
        "  ATTENZIONE: flash_decode=%d flash_prefill=%d fuse_rope=%d sliding=%d sinks=%d\n",
        w->at_flash_decode, w->at_flash_prefill, w->at_fuse_rope, w->at_sliding, w->at_sinks);
    fprintf(stderr,
        "  PESI    : packed_qkv=%d wcache=%d reshape_canon=%d | scales f32=%d f16=%d bf16=%d "
        "(promossi=%d) | gemvq4=%d oq_a8=%d\n",
        w->w_packed_qkv, w->w_wcache, w->w_reshape_canon,
        w->w_scales_f32, w->w_scales_f16, w->w_scales_bf16, w->w_scales_promoted,
        w->w_gemvq4, w->w_oq_a8);
    fprintf(stderr,
        "  ESECUZIONE: compile=%d fixkv=%d compile_dense=%d async=%d batch_decode=%d sampler_legacy=%d\n",
        w->x_compile, w->x_fixkv, w->x_compile_dense, w->x_async, w->x_batch_decode, w->x_sampler_legacy);
}

#endif /* LLM_MLX_WHICH_PATH_H */
