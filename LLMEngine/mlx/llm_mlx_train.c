/* ============================================================================
 * llm_mlx_train.c — ABI Apache dei PRIMITIVI del forward differenziabile (llmengine_mlx_train.h).
 *
 * È INFERENZA (esegue il forward + la cross-entropy dentro il grafo MLX): non produce/alterà
 * pesi. La composizione di loss/grad/optimizer/training vive nel PUL (gmbllm, GMBasic).
 * Riusa: forward_tokens_batch (backbone), llm_mlx_final_norm, quantized_linear_cfg (lm_head),
 * e le ops MLX (softmax/log/take_along_axis/sum) per la CE. La LoRA è già nel grafo
 * (m->lora_A/B sono mlx_array applicati da llm_mlx_lora_apply_mlx).
 * ============================================================================ */
#include "llmengine_mlx_train.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_lora.h"

#include <stdio.h>
#include <string.h>

/* 1 = modalità TRAINING: il forward deve essere COMPOSABILE (nessun kernel Metal custom, che
 * non ha vjp). Letto dalle ENABLE dei kernel fusi (es. swiglu Metal) per forzare il fallback
 * alle op MLX native differenziabili. Impostato da setup()/set_eager_sdpa(). */
int g_llm_mlx_train_composable = 0;

int llm_ai_mlx_train_setup(llm_ai_mlx_model_handle mh) {
    LlmMlxModel* m = (LlmMlxModel*)mh;
    if (!m) return -1;
    /* Nulla da preparare: il motore è già grafo-MLX; abilitiamo la LoRA (le A/B sono settate
     * da set_lora_mlx). Attiviamo la modalità composabile (training differenziabile). */
    m->lora_enabled = true;
    g_llm_mlx_train_composable = 1;
    return 0;
}

/* Ritorna (creandolo se serve) lo slot LoRA per (layer, proj) SENZA copiare dati: A/B restano
 * array del chiamante (mlx_array_set = retain per refcount). */
static int mlx_train_lora_slot(LlmMlxModel* m, int layer, const char* proj, int rank, float scale) {
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

int llm_ai_mlx_train_set_lora_mlx(llm_ai_mlx_model_handle mh, int layer, const char* proj,
                                  mlx_array A, mlx_array B, void* extra) {
    (void)extra;
    LlmMlxModel* m = (LlmMlxModel*)mh;
    if (!m || !proj) return -1;
    m->lora_enabled = true;
    int li = mlx_train_lora_slot(m, layer, proj, /*rank*/0, 1.0f);
    if (li < 0) return -1;
    if (mlx_array_set(&m->lora_A[li], A) != 0) return -1;
    if (mlx_array_set(&m->lora_B[li], B) != 0) return -1;
    return 0;
}

void llm_ai_mlx_train_set_eager_sdpa(int on) { g_llm_mlx_train_composable = on ? 1 : 0; }

mlx_stream llm_ai_mlx_train_stream(llm_ai_mlx_model_handle mh) {
    LlmMlxModel* m = (LlmMlxModel*)mh;
    if (m) return m->stream;
    mlx_stream none;
    memset(&none, 0, sizeof none);
    return none;
}

mlx_array llm_ai_mlx_train_forward_ce(llm_ai_mlx_model_handle mh,
                                      const int32_t* toks, const int32_t* tgt,
                                      const uint8_t* mask, int T,
                                      bool sum_mode, bool* ok) {
    if (ok) *ok = false;
    LlmMlxModel* m = (LlmMlxModel*)mh;
    mlx_array loss = mlx_array_new();
    if (!m || !toks || !tgt || T <= 0) return loss;

    llm_ai_mlx_session_handle sh = llm_ai_mlx_create_session(mh);
    if (!sh) return loss;
    LlmMlxSession* s = (LlmMlxSession*)sh;
    OutputTensors t;
    if (!load_output_tensors(m, &t)) { llm_ai_mlx_destroy_session(sh); return loss; }

    mlx_array h = mlx_array_new();
    bool good = forward_tokens_batch(m, s, t.embed_w, t.embed_s, t.embed_b,
                                     toks, T, 0, s->k_cache, s->v_cache, s->has_cache, &h);
    if (good) {
        mlx_array normed = mlx_array_new();
        good = llm_mlx_final_norm(m, h, t.final_norm_w, &normed);
        mlx_array logits = mlx_array_new();
        if (good) good = quantized_linear_cfg(&logits, normed, t.out_w, t.out_s, t.out_b,
                                              m->cfg.quant_group_size, m->cfg.quant_bits,
                                              m->cfg.quant_mode, m->stream);
        /* final_logit_softcap (Gemma/gpt-oss): stessa trasformazione del sampling. */
        mlx_array capped = mlx_array_new();
        if (good && m->cfg.final_logit_softcap > 0.0f)
            good = llm_mlx_logit_softcap(&capped, logits, m->cfg.final_logit_softcap, m->stream);
        else if (good) good = mlx_array_set(&capped, logits) == 0;
        /* log_softmax STABILE = x - logsumexp(x) sull'ultimo asse (poi gather dei target).
         * NB: NON `log(softmax(x))` — la vjp di 1/p con p~0 su vocab grande (128k) produce
         * inf/NaN nei gradienti (misurato: A/B tutti NaN). logsumexp ha vjp stabile. */
        mlx_array lse = mlx_array_new(), logp = mlx_array_new();
        if (good) good = mlx_logsumexp_axis(&lse, capped, -1, /*keepdims=*/true, m->stream) == 0;
        if (good) good = mlx_subtract(&logp, capped, lse, m->stream) == 0;
        /* target [1,T] -> [1,T,1] */
        mlx_array tv = mlx_array_new(), te = mlx_array_new();
        if (good) {
            int tshp[2] = { 1, T };
            tv = mlx_array_new_data(tgt, tshp, 2, MLX_INT32);
            good = mlx_expand_dims(&te, tv, -1, m->stream) == 0;
        }
        mlx_array g = mlx_array_new(), nll = mlx_array_new();
        if (good) good = mlx_take_along_axis(&g, logp, te, -1, m->stream) == 0;
        if (good) good = mlx_negative(&nll, g, m->stream) == 0;   /* NLL [1,T,1] */
        /* maschera opzionale: [1,T] -> [1,T,1] (0 = ignora). */
        mlx_array pen = nll;
        if (good && mask) {
            float* mf = (float*)malloc((size_t)T * sizeof(float));
            if (mf) {
                for (int i = 0; i < T; i++) mf[i] = mask[i] ? 1.0f : 0.0f;
                int mshp2[2] = { 1, T };
                mlx_array mv = mlx_array_new_data(mf, mshp2, 2, MLX_FLOAT32);
                mlx_array me = mlx_array_new();
                if (mlx_expand_dims(&me, mv, -1, m->stream) == 0) {
                    mlx_array sm = mlx_array_new();
                    if (mlx_multiply(&sm, nll, me, m->stream) == 0) { mlx_array_set(&pen, sm); }
                    mlx_array_free(sm);
                }
                mlx_array_free(me); mlx_array_free(mv); free(mf);
            }
        }
        mlx_array summed = mlx_array_new();
        if (good) good = mlx_sum(&summed, pen, /*keepdims=*/false, m->stream) == 0;
        if (good) {
            if (sum_mode) {
                mlx_array_set(&loss, summed);
            } else {
                float denom = (float)T;
                if (mask) { int c = 0; for (int i = 0; i < T; i++) if (mask[i]) c++; denom = c > 0 ? (float)c : 1.0f; }
                mlx_array d = mlx_array_new_float(denom);
                mlx_array dq = mlx_array_new();
                if (mlx_divide(&dq, summed, d, m->stream) == 0) mlx_array_set(&loss, dq);
                mlx_array_free(dq); mlx_array_free(d);
            }
        }
        if (good && ok) *ok = true;
        mlx_array_free(summed); mlx_array_free(h); mlx_array_free(normed); mlx_array_free(logits);
        mlx_array_free(capped); mlx_array_free(lse); mlx_array_free(logp);
        mlx_array_free(tv); mlx_array_free(te); mlx_array_free(g); mlx_array_free(nll);
    } else {
        mlx_array_free(h);
    }
    free_output_tensors(&t);
    llm_ai_mlx_destroy_session(sh);
    return loss;
}
