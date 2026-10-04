/* ============================================================================
 * llm_mlx_session_logprob.c — dominio LOGPROBS (ABI v2): log p(token) alla posizione "pending".
 *
 * Riusa la stessa distribuzione di final_step (final_norm + lm_head + softcap) e la normalizza
 * in log-softmax in host. Non modifica MAI lo stato della sessione: legge pending_h (come
 * greedy_prob) e restituisce. Per questo e' sicuro chiamarla PRIMA di un passo di generate per
 * catturare la logprob del token che lo step emettera'.
 *
 * generate_probs cuce il tutto in un helper: snapshot della distribuzione -> generate(1) ->
 * log p(id emesso). Esatta anche con sampling, perche' l'id emesso e' cercato nella distribuzione
 * PRE-step del modello.
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Logits (f32, host) alla posizione pending + logsumexp. Scrive *out_logits (malloc, vocab float;
 * il chiamante libera) e *out_logsumexp. Ritorna vocab (>0) o -1 su errore. Non tocca lo stato. */
static int position_logits_host(LlmMlxSession* s, float** out_logits, double* out_logsumexp) {
    if (s == NULL || out_logits == NULL || out_logsumexp == NULL) return -1;
    if (!s->has_pending) return -1;
    LlmMlxModel* m = s->model;

    /* Modello DELEGATO (DS4/DS4.1): logprob non esposto dall'ABI a 9 metodi -> non supportato. */
    if (m->deleg) return -1;

    OutputTensors t;
    if (!load_output_tensors(m, &t)) return -1;
    mlx_array normed = mlx_array_new();
    bool ok = llm_mlx_final_norm(m, s->pending_h, t.final_norm_w, &normed);
    mlx_array logits = mlx_array_new();
    /* Ternary-Bonsai: lm_head e' elencato -> si ruota il suo ingresso (come final_step). */
    if (ok) ok = llm_bonsai_rotate(&normed, m, -1, "lm_head.weight", false);
    if (ok) ok = quantized_linear_cfg(&logits, normed, t.out_w, t.out_s, t.out_b,
                                      m->cfg.quant_group_size, m->cfg.quant_bits, m->cfg.quant_mode, m->stream);
    mlx_array capped = mlx_array_new();
    if (ok && m->cfg.final_logit_softcap > 0.0f) {
        ok = llm_mlx_logit_softcap(&capped, logits, m->cfg.final_logit_softcap, m->stream);
    } else if (ok) {
        ok = mlx_array_set(&capped, logits) == 0;
    }
    mlx_array capped32 = mlx_array_new();
    if (ok) ok = llm_mlx_logits_f32(&capped32, capped, m->stream);
    if (ok) ok = mlx_array_eval(capped32) == 0;

    float* lg = NULL;
    int V = 0;
    if (ok) {
        const float* d = mlx_array_data_float32(capped32);
        size_t n = mlx_array_size(capped32);
        ok = (d != NULL && n > 0);
        if (ok) {
            V = (int)n;
            lg = (float*)malloc((size_t)V * sizeof(float));
            if (!lg) ok = false;
            else memcpy(lg, d, (size_t)V * sizeof(float));
        }
    }
    mlx_array_free(normed);
    mlx_array_free(logits);
    mlx_array_free(capped);
    mlx_array_free(capped32);
    free_output_tensors(&t);
    if (!ok || lg == NULL) { free(lg); return -1; }

    double mx = lg[0];
    for (int i = 1; i < V; i++) if (lg[i] > mx) mx = lg[i];
    double sum = 0.0;
    for (int i = 0; i < V; i++) sum += exp((double)lg[i] - mx);
    *out_logits = lg;
    *out_logsumexp = mx + log(sum);
    return V;
}

int llm_ai_mlx_session_logprob_of(llm_ai_mlx_session_handle handle, int32_t token_id,
                                  float* out_logprob) {
    if (handle == NULL || out_logprob == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    float* lg = NULL;
    double lse = 0.0;
    int V = position_logits_host(s, &lg, &lse);
    if (V <= 0) { free(lg); return -1; }
    if (token_id < 0 || token_id >= V) { free(lg); return -1; }
    *out_logprob = (float)((double)lg[token_id] - lse);
    free(lg);
    return 0;
}

int llm_ai_mlx_session_last_logprob(llm_ai_mlx_session_handle handle, float* out_logprob) {
    if (handle == NULL || out_logprob == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    if (!s->has_pending) return -1;
    float* lg = NULL;
    double lse = 0.0;
    int V = position_logits_host(s, &lg, &lse);
    if (V <= 0) { free(lg); return -1; }
    int32_t id = s->pending_next_id;
    if (id < 0 || id >= V) { free(lg); return -1; }
    *out_logprob = (float)((double)lg[id] - lse);
    free(lg);
    return 0;
}

int llm_ai_mlx_session_topk(llm_ai_mlx_session_handle handle, int k, int32_t* out_ids,
                            float* out_logprobs) {
    if (handle == NULL || k <= 0 || out_ids == NULL || out_logprobs == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    float* lg = NULL;
    double lse = 0.0;
    int V = position_logits_host(s, &lg, &lse);
    if (V <= 0) { free(lg); return -1; }
    if (k > V) k = V;
    /* Selezione dei k logits maggiori mantenendo l'ordine decrescente per inserimento: i k
     * migliori finora stanno in [0,k) e l'ultimo e' il minimo del gruppo. O(V*k), con k
     * richiesto piccolo (top_logprobs tipico <= 20). */
    for (int i = 0; i < k; i++) { out_ids[i] = -1; out_logprobs[i] = -INFINITY; }
    for (int v = 0; v < V; v++) {
        float x = lg[v];
        if (x <= out_logprobs[k - 1]) continue;
        int j = k - 1;
        while (j > 0 && out_logprobs[j - 1] < x) {
            out_logprobs[j] = out_logprobs[j - 1];
            out_ids[j] = out_ids[j - 1];
            j--;
        }
        out_logprobs[j] = x;
        out_ids[j] = v;
    }
    /* Da logit a log-softmax: log p = logit - logsumexp (stesso denominatore del campione). */
    for (int i = 0; i < k; i++)
        if (out_ids[i] >= 0) out_logprobs[i] = (float)((double)out_logprobs[i] - lse);
    free(lg);
    return k;
}

int llm_ai_mlx_session_generate_probs(llm_ai_mlx_session_handle handle, int max_new, int32_t eos,
                                      int32_t* out_ids, float* out_logprobs) {
    if (handle == NULL || max_new <= 0 || out_ids == NULL || out_logprobs == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    int n = 0;
    for (int i = 0; i < max_new; i++) {
        /* 1. distribuzione del modello alla posizione di partenza di QUESTO step. */
        float* lg = NULL;
        double lse = 0.0;
        int V = position_logits_host(s, &lg, &lse);
        if (V <= 0) { free(lg); return (n > 0) ? n : -1; }
        /* 2. passo di generate standard (rispetta sampling/async/eos come generate_ids). */
        int32_t id = 0;
        int got = llm_ai_mlx_session_generate(handle, 1, eos, &id);
        if (got < 0) { free(lg); return (n > 0) ? n : -1; }
        if (got == 0) { free(lg); break; }   /* EOS non emesso: come generate */
        out_ids[i] = id;
        out_logprobs[i] = (id >= 0 && id < V) ? (float)((double)lg[id] - lse) : 0.0f;
        free(lg);
        n++;
    }
    return n;
}

/* Entropia (bit) della distribuzione completa alla posizione pending. Usata per il logits report.
 * Ritorna 0 e scrive *out_entropy (-1.0 se non calcolabile). */
int llm_ai_mlx_session_entropy(llm_ai_mlx_session_handle handle, float* out_entropy) {
    if (handle == NULL || out_entropy == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    float* lg = NULL; double lse = 0.0;
    int V = position_logits_host(s, &lg, &lse);
    if (V <= 0) { free(lg); *out_entropy = -1.0f; return -1; }
    double H = 0.0;
    for (int i = 0; i < V; i++) { double p = exp((double)lg[i] - lse); if (p > 1e-12) H -= p * log(p); }
    *out_entropy = (float)(H / log(2.0));
    free(lg);
    return 0;
}
