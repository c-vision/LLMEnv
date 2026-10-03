/* ============================================================================
 * llm_mlx_diff_forward.c — ABI Apache dei PRIMITIVI del forward DIFFERENZIABILE
 * (ex llm_mlx_train.c, lato INFERENZA). Esegue il forward del modello (backbone + final_norm +
 * lm_head) dentro il grafo MLX: NON produce/alterà pesi. La LOSS (masked CE), grad, DPO,
 * optimizer e training vivono nel PUL (GMBasic/gmbllm) che chiama questa ABI.
 * Perché Apache: il forward differenziabile è l'unico modo di avere PARITÀ train/inferenza
 * ("una sola sorgente"); il confine è "esegue vs produce" (ADR §4). Il PUL costruisce la loss
 * SOPRA il logits restituito qui.
 * ============================================================================ */
#include "llmengine_mlx_diff_forward.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_lora.h"

#include <stdio.h>
#include <string.h>

/* 1 = il forward deve essere COMPOSABILE (nessun kernel Metal custom, che non ha vjp);
 * letto dalle ENABLE dei kernel fusi (es. swiglu Metal) per forzare le op MLX differenziabili. */
int g_llm_mlx_train_composable = 0;

int llm_ai_mlx_diff_setup(llm_ai_mlx_model_handle mh) {
    LlmMlxModel* m = (LlmMlxModel*)mh;
    if (!m) return -1;
    m->lora_enabled = true;
    g_llm_mlx_train_composable = 1;
    return 0;
}

static int mlx_diff_lora_slot(LlmMlxModel* m, int layer, const char* proj, int rank, float scale) {
    (void)rank; (void)scale;
    if (!m) return -1;
    for (int i = 0; i < m->lora_n; i++)
        if (m->lora_layer[i] == layer && strcmp(m->lora_proj[i], proj) == 0) return i;
    if (m->lora_n >= m->lora_cap) {
        int ncap = m->lora_cap ? m->lora_cap * 2 : 64;
        mlx_array* nA = (mlx_array*)realloc(m->lora_A, (size_t)ncap * sizeof(mlx_array));
        mlx_array* nB = (mlx_array*)realloc(m->lora_B, (size_t)ncap * sizeof(mlx_array));
        int* nL = (int*)realloc(m->lora_layer, (size_t)ncap * sizeof(int));
        char (*nP)[32] = (char (*)[32])realloc(m->lora_proj, (size_t)ncap * 32);
        float* nS = (float*)realloc(m->lora_scale, (size_t)ncap * sizeof(float));
        int* nE = (int*)realloc(m->lora_expert, (size_t)ncap * sizeof(int));
        if (!nA || !nB || !nL || !nP || !nS || !nE) return -1;
        m->lora_A = nA; m->lora_B = nB; m->lora_layer = nL; m->lora_proj = nP;
        m->lora_scale = nS; m->lora_expert = nE; m->lora_cap = ncap;
    }
    int idx = m->lora_n;
    m->lora_layer[idx] = layer;
    llm_mlx_mode_set(m->lora_proj[idx], sizeof(m->lora_proj[idx]), proj);
    m->lora_scale[idx] = scale;
    m->lora_expert[idx] = -1;
    m->lora_A[idx] = mlx_array_new();
    m->lora_B[idx] = mlx_array_new();
    m->lora_n++;
    return idx;
}

int llm_ai_mlx_diff_set_lora_mlx(llm_ai_mlx_model_handle mh, int layer, const char* proj,
                                 mlx_array A, mlx_array B, void* extra) {
    (void)extra;
    LlmMlxModel* m = (LlmMlxModel*)mh;
    if (!m || !proj) return -1;
    m->lora_enabled = true;
    int li = mlx_diff_lora_slot(m, layer, proj, 0, 1.0f);
    if (li < 0) return -1;
    if (mlx_array_set(&m->lora_A[li], A) != 0) return -1;
    if (mlx_array_set(&m->lora_B[li], B) != 0) return -1;
    return 0;
}

void llm_ai_mlx_diff_set_eager_sdpa(int on) { g_llm_mlx_train_composable = on ? 1 : 0; }

mlx_stream llm_ai_mlx_diff_stream(llm_ai_mlx_model_handle mh) {
    LlmMlxModel* m = (LlmMlxModel*)mh;
    if (m) return m->stream;
    mlx_stream none;
    memset(&none, 0, sizeof none);
    return none;
}

/* Forward DIFFERENZIABILE → logits [1,T,V]. Il PUL costruisce la CE sopra questi. If `sess_out`
 * != NULL riceve la sessione creata (il chiamante la distrugge dopo aver usato i logits). */
mlx_array llm_ai_mlx_diff_forward_logits(llm_ai_mlx_model_handle mh, const int32_t* toks, int T,
                                         llm_ai_mlx_session_handle* sess_out, bool* ok) {
    if (ok) *ok = false;
    if (sess_out) *sess_out = NULL;
    LlmMlxModel* m = (LlmMlxModel*)mh;
    mlx_array logits = mlx_array_new();
    if (!m || !toks || T <= 0) return logits;
    llm_ai_mlx_session_handle sh = llm_ai_mlx_create_session(mh);
    if (!sh) return logits;
    LlmMlxSession* s = (LlmMlxSession*)sh;
    OutputTensors t;
    if (!load_output_tensors(m, &t)) { llm_ai_mlx_destroy_session(sh); return logits; }
    mlx_array h = mlx_array_new();
    bool good = forward_tokens_batch(m, s, t.embed_w, t.embed_s, t.embed_b,
                                     toks, T, 0, s->k_cache, s->v_cache, s->has_cache, &h);
    if (good) {
        mlx_array normed = mlx_array_new();
        good = llm_mlx_final_norm(m, h, t.final_norm_w, &normed);
        if (good) good = quantized_linear_cfg(&logits, normed, t.out_w, t.out_s, t.out_b,
                                              m->cfg.quant_group_size, m->cfg.quant_bits,
                                              m->cfg.quant_mode, m->stream);
        /* final_logit_softcap (Gemma/gpt-oss): stessa trasformazione del sampling. */
        if (good && m->cfg.final_logit_softcap > 0.0f) {
            mlx_array capped = mlx_array_new();
            if (llm_mlx_logit_softcap(&capped, logits, m->cfg.final_logit_softcap, m->stream) == 0)
                mlx_array_set(&logits, capped);
            mlx_array_free(capped);
        }
        mlx_array_free(normed);
        if (good && ok) *ok = true;
    }
    mlx_array_free(h);
    free_output_tensors(&t);
    if (good) { if (sess_out) *sess_out = sh; else llm_ai_mlx_destroy_session(sh); }
    else llm_ai_mlx_destroy_session(sh);
    return logits;
}
