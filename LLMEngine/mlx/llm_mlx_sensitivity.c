/* ============================================================================
 * llm_mlx_sensitivity.c — Model Analyze Sensitivity / Quantize Auto / Dequantize (Safe Surgery).
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ─────────────────────────────────────────────────────────────────────────────

 * MODEL ANALYZE SENSITIVITY (roadmap 8.11 fase 3 — Safe Surgery): misura per ogni

 * layer l'errore relativo dei pesi quando la proiezione principale viene down-quantizzata

 * dal suo bits nativo a Q4 (dequant Q8->float, ri-quant Q4, dequant, ||w-q4(w)||/||w||).

 * Layer con errore alto = high sensitivity (conservare precisione); errore basso =

 * candidato a down-quantizzazione. Base per la quantizzazione selettiva per-sensitivity

 * (`MODEL QUANTIZE AUTO`). Usa la proiezione rappresentativa: q_proj (attention/full),

 * in_proj_qkv (GDN), mlp.gate (MoE).

 *

 * Ritorna 0 ok, -1 handle nullo. Stampa la tabella su stderr.

 */

/* Misura la sensitivity di un layer (errore relativo dei pesi bits_nativo -> Q4).
 * Ritorna: 1 ok (*err valorizzato), 0 non applicabile (gia' Q4 o lookup fallito), -1 errore. */
static int llm_mlx_sensitivity_one(LlmMlxModel* m, int layer, const char* proj, float* err) {
    const ModelConfig* c = &m->cfg;
    char namebuf[256];
    tname(c->layer_prefix, layer, proj, namebuf, sizeof(namebuf));
    mlx_array w=mlx_array_new(), s=mlx_array_new(), b=mlx_array_new();
    if (!multishard_get_triple(&m->weights, namebuf, &w, &s, &b) || s.ctx == NULL) {
        mlx_array_free(w); mlx_array_free(s); mlx_array_free(b); return 0;
    }
    int b_orig = llm_mlx_derive_bits(w, s, c->quant_group_size, c->quant_bits);
    if (b_orig <= 4) { mlx_array_free(w); mlx_array_free(s); mlx_array_free(b); return 0; }
    mlx_optional_int gsz = {c->quant_group_size, true};
    mlx_optional_dtype dt = {MLX_FLOAT32, true};
    mlx_array wf = mlx_array_new();
    bool ok = mlx_dequantize(&wf, w, s, b, gsz, (mlx_optional_int){b_orig,true}, "affine", g_mlx_empty_arr, dt, m->stream) == 0;
    mlx_vector_array q4 = mlx_vector_array_new();
    if (ok) ok = mlx_quantize(&q4, wf, gsz, (mlx_optional_int){4,true}, "affine", g_mlx_empty_arr, m->stream) == 0;
    mlx_array qw=mlx_array_new(), qs=mlx_array_new(), qb=mlx_array_new();
    if (ok) ok = mlx_vector_array_get(&qw, q4, 0) == 0 && mlx_vector_array_get(&qs, q4, 1) == 0
                && mlx_vector_array_get(&qb, q4, 2) == 0;
    mlx_array w4 = mlx_array_new();
    if (ok) ok = mlx_dequantize(&w4, qw, qs, qb, gsz, (mlx_optional_int){4,true}, "affine", g_mlx_empty_arr, dt, m->stream) == 0;
    mlx_array diff=mlx_array_new();
    if (ok) ok = mlx_subtract(&diff, wf, w4, m->stream) == 0;
    mlx_array d2=mlx_array_new(), w2=mlx_array_new();
    if (ok) ok = mlx_multiply(&d2, diff, diff, m->stream) == 0;
    if (ok) ok = mlx_multiply(&w2, wf, wf, m->stream) == 0;
    mlx_array sd=mlx_array_new(), sw=mlx_array_new();
    if (ok) ok = mlx_sum(&sd, d2, false, m->stream) == 0 && mlx_sum(&sw, w2, false, m->stream) == 0;
    mlx_array rd=mlx_array_new(), rw=mlx_array_new();
    if (ok) ok = mlx_sqrt(&rd, sd, m->stream) == 0 && mlx_sqrt(&rw, sw, m->stream) == 0;
    mlx_array er=mlx_array_new();
    if (ok) ok = mlx_divide(&er, rd, rw, m->stream) == 0;
    float ev = 0.0f;
    if (ok) { ok = mlx_array_eval(er) == 0; if (ok) { const float* d = mlx_array_data_float32(er); if (d) ev = d[0]; } }
    *err = ev;
    mlx_array_free(w); mlx_array_free(s); mlx_array_free(b);
    mlx_array_free(wf); mlx_array_free(qw); mlx_array_free(qs); mlx_array_free(qb); mlx_array_free(w4);
    mlx_array_free(diff); mlx_array_free(d2); mlx_array_free(w2); mlx_array_free(sd); mlx_array_free(sw);
    mlx_array_free(rd); mlx_array_free(rw); mlx_array_free(er);
    mlx_vector_array_free(q4);
    return ok ? 1 : -1;
}

const char* llm_mlx_layer_proj(const ModelConfig* c, int layer) {
    if (c->is_linear_layer && c->is_linear_layer[layer]) return "linear_attn.in_proj_qkv";
    if (c->n_experts > 0) return "mlp.gate";
    return "self_attn.q_proj";
}

int llm_ai_mlx_model_sensitivity(llm_ai_mlx_model_handle handle) {

    if (handle == NULL) return -1;

    LlmMlxModel* m = (LlmMlxModel*)handle;

    const ModelConfig* c = &m->cfg;

    char namebuf[256];

    fprintf(stderr, "MODEL sensitivity (errore relativo dei pesi: bits_nativo -> Q4):\n");

    int analyzed = 0;

    for (int layer = 0; layer < c->n_layers; layer++) {

        const char* proj = "self_attn.q_proj";

        if (c->is_linear_layer && c->is_linear_layer[layer]) proj = "linear_attn.in_proj_qkv";

        else if (c->n_experts > 0) proj = "mlp.gate";

        tname(c->layer_prefix, layer, proj, namebuf, sizeof(namebuf));

        mlx_array w=mlx_array_new(), s=mlx_array_new(), b=mlx_array_new();

        bool ok = multishard_get_triple(&m->weights, namebuf, &w, &s, &b);

        if (!ok || s.ctx == NULL) { fprintf(stderr, "  layer %2d: lookup %s fallito\n", layer, namebuf);

            mlx_array_free(w); mlx_array_free(s); mlx_array_free(b); continue; }

        int b_orig = llm_mlx_derive_bits(w, s, c->quant_group_size, c->quant_bits);

        if (b_orig <= 4) { /* gia' Q4 o meno: niente da down-quantizzare */

            fprintf(stderr, "  layer %2d: %s bits=%d (gia' basso)\n", layer, proj, b_orig);

            mlx_array_free(w); mlx_array_free(s); mlx_array_free(b); analyzed++; continue; }

        mlx_optional_int gsz = {c->quant_group_size, true};

        mlx_optional_dtype dt = {MLX_FLOAT32, true};

        /* w_float = dequant(w, s, b, orig) */

        mlx_array wf = mlx_array_new();

        ok = mlx_dequantize(&wf, w, s, b, gsz, (mlx_optional_int){b_orig,true}, "affine", g_mlx_empty_arr, dt, m->stream) == 0;

        /* q4 = quant(wf, 4); w4 = dequant(q4, 4) */

        mlx_vector_array q4 = mlx_vector_array_new();

        if (ok) ok = mlx_quantize(&q4, wf, gsz, (mlx_optional_int){4,true}, "affine", g_mlx_empty_arr, m->stream) == 0;

        mlx_array qw=mlx_array_new(), qs=mlx_array_new(), qb=mlx_array_new();

        if (ok) ok = mlx_vector_array_get(&qw, q4, 0) == 0 && mlx_vector_array_get(&qs, q4, 1) == 0
                    && mlx_vector_array_get(&qb, q4, 2) == 0; /* biases del quant Q4 */


        mlx_array w4 = mlx_array_new();

        if (ok) ok = mlx_dequantize(&w4, qw, qs, qb, gsz, (mlx_optional_int){4,true}, "affine", g_mlx_empty_arr, dt, m->stream) == 0;

        /* err = sqrt(sum((wf-w4)^2)) / sqrt(sum(wf^2)) */

        mlx_array diff=mlx_array_new();

        if (ok) ok = mlx_subtract(&diff, wf, w4, m->stream) == 0;

        mlx_array d2=mlx_array_new(), w2=mlx_array_new();

        if (ok) ok = mlx_multiply(&d2, diff, diff, m->stream) == 0;

        if (ok) ok = mlx_multiply(&w2, wf, wf, m->stream) == 0;

        mlx_array sd=mlx_array_new(), sw=mlx_array_new();

        if (ok) ok = mlx_sum(&sd, d2, false, m->stream) == 0 && mlx_sum(&sw, w2, false, m->stream) == 0;

        mlx_array rd=mlx_array_new(), rw=mlx_array_new();

        if (ok) ok = mlx_sqrt(&rd, sd, m->stream) == 0 && mlx_sqrt(&rw, sw, m->stream) == 0;

        mlx_array err=mlx_array_new();

        if (ok) ok = mlx_divide(&err, rd, rw, m->stream) == 0;

        float ev = 0.0f;

        if (ok) { ok = mlx_array_eval(err) == 0; if (ok) { const float* d = mlx_array_data_float32(err); if (d) ev = d[0]; } }

        fprintf(stderr, "  layer %2d: %s bits=%d err(->Q4)=%.4f\n", layer, proj, b_orig, ev);

        analyzed++;

        mlx_array_free(w); mlx_array_free(s); mlx_array_free(b);

        mlx_array_free(wf); mlx_array_free(qw); mlx_array_free(qs); mlx_array_free(qb); mlx_array_free(w4);

        mlx_array_free(diff); mlx_array_free(d2); mlx_array_free(w2); mlx_array_free(sd); mlx_array_free(sw);

        mlx_array_free(rd); mlx_array_free(rw); mlx_array_free(err);

        mlx_vector_array_free(q4);

    }

    fprintf(stderr, "  analyzed %d layer\n", analyzed);

    return 0;

}



/* MODEL QUANTIZE AUTO (roadmap fase 3 Safe Surgery): usa la sensitivity per-layer per

 * produrre una policy di quantizzazione selettiva. `keep_fraction` (0..1) = frazione dei

 * layer piu' sensibili che restano al bits nativo; gli altri (basso errore) sono candidati

 * a down-quantizzazione a Q4. Stampa la policy per-layer (raccomandazione: NON modifica i

 * pesi caricati, che richiederebbe ricaricare). Ritorna 0 ok, -1 errore.

 */

int llm_ai_mlx_model_quantize_auto(llm_ai_mlx_model_handle handle, float keep_fraction) {

    if (handle == NULL) return -1;

    LlmMlxModel* m = (LlmMlxModel*)handle;

    const ModelConfig* c = &m->cfg;

    if (keep_fraction < 0.05f) keep_fraction = 0.05f;

    if (keep_fraction > 1.0f) keep_fraction = 1.0f;

    /* sensibili: quelli che NON possono scendere a Q4 (b_orig<=4 o errore alto) */

    int keep_count = (int)(c->n_layers * keep_fraction);

    if (keep_count < 1) keep_count = 1;

    float* err = (float*)calloc((size_t)c->n_layers, sizeof(float));

    int* measured = (int*)calloc((size_t)c->n_layers, sizeof(int));

    if (!err || !measured) { free(err); free(measured); return -1; }

    int n_measured = 0;

    for (int i = 0; i < c->n_layers; i++) {

        float e = 0; int r = llm_mlx_sensitivity_one(m, i, llm_mlx_layer_proj(c, i), &e);

        if (r == 1) { err[i] = e; measured[i] = 1; n_measured++; }

    }

    fprintf(stderr, "MODEL QUANTIZE AUTO (keep_fraction=%.2f, %d/%d layer misurabili):\n", keep_fraction, n_measured, c->n_layers);

    fprintf(stderr, "  layer | sens | policy\n");

    /* policy: i keep_count layer piu' sensibili restano al bits nativo, gli altri -> Q4 */

    /* ordina gli indici per errore decrescente */

    int* order = (int*)malloc((size_t)c->n_layers * sizeof(int));

    for (int i = 0; i < c->n_layers; i++) order[i] = i;

    for (int i = 0; i < c->n_layers; i++) for (int j = i+1; j < c->n_layers; j++)

        if (measured[order[j]] && (!measured[order[i]] || err[order[j]] > err[order[i]])) { int t=order[i]; order[i]=order[j]; order[j]=t; }

    int* keep = (int*)calloc((size_t)c->n_layers, sizeof(int));

    for (int k = 0; k < keep_count && k < c->n_layers; k++) keep[order[k]] = 1;

    for (int i = 0; i < c->n_layers; i++) {

        const char* proj = llm_mlx_layer_proj(c, i);

        if (!measured[i]) { fprintf(stderr, "  %4d |  n/a  | (gia Q4 o non misurabile)\n", i); continue; }

        const char* pol = keep[i] ? "KEEP native" : "DOWN to Q4";
        fprintf(stderr, "  %4d | %.4f | %s | %s\n", i, err[i], pol, proj);

    }

    free(err); free(measured); free(order); free(keep);

    return 0;

}



/* MODEL DEQUANTIZE / WEIGHT CONVERSION (roadmap fase 3 Safe Surgery): dequantizza la

 * proiezione principale di un layer a float32 e mostra le statistiche dei pesi

 * (min/max/mean/std, quantili 1/99%). Utile per capire la distribuzione e decidere

 * la quantizzazione; base per esportare i pesi in un formato non-quantizzato.

 * Ritorna 0 ok, -1 errore/out of range.

 */

int llm_ai_mlx_model_dequantize_layer(llm_ai_mlx_model_handle handle, int layer) {

    if (handle == NULL) return -1;

    LlmMlxModel* m = (LlmMlxModel*)handle;

    const ModelConfig* c = &m->cfg;

    if (layer < 0 || layer >= c->n_layers) return -1;

    const char* proj = llm_mlx_layer_proj(c, layer);

    char namebuf[256];

    tname(c->layer_prefix, layer, proj, namebuf, sizeof(namebuf));

    mlx_array w=mlx_array_new(), s=mlx_array_new(), b=mlx_array_new();

    if (!multishard_get_triple(&m->weights, namebuf, &w, &s, &b) || s.ctx == NULL) {

        fprintf(stderr, "dequantize_layer: %s non trovato/quantizzato\n", namebuf);

        mlx_array_free(w); mlx_array_free(s); mlx_array_free(b); return -1;

    }

    int b_orig = llm_mlx_derive_bits(w, s, c->quant_group_size, c->quant_bits);

    mlx_optional_dtype dt = {MLX_FLOAT32, true};

    mlx_array wf = mlx_array_new();

    bool ok = mlx_dequantize(&wf, w, s, b, (mlx_optional_int){c->quant_group_size,true}, (mlx_optional_int){b_orig,true}, "affine", g_mlx_empty_arr, dt, m->stream) == 0;

    if (!ok) { mlx_array_free(w); mlx_array_free(s); mlx_array_free(b); mlx_array_free(wf); return -1; }

    /* statistiche in host (una sola eval + lettura) */

    mlx_array_eval(wf);

    const float* data = (const float*)mlx_array_data_float32(wf);

    int n = (int)mlx_array_size(wf);

    if (!data || n <= 0) { mlx_array_free(w); mlx_array_free(s); mlx_array_free(b); mlx_array_free(wf); return -1; }

    double sum=0, sumsq=0; float mn=data[0], mx=data[0];

    for (int i = 0; i < n; i++) { float v=data[i]; sum+=v; sumsq+=(double)v*v; if(v<mn)mn=v; if(v>mx)mx=v; }

    double mean = sum/n, var = (sumsq/n) - mean*mean; double std = var>0?sqrt(var):0.0;

    /* quantili approssimati (campionamento a passi) */

    float q1=data[0], q99=data[0];

    if (n > 100) {

        int i1=(int)(n*0.01), i99=(int)(n*0.99);

        if (i1<0)i1=0; if(i1>=n)i1=n-1; if(i99<0)i99=0; if(i99>=n)i99=n-1;

        q1=data[i1]; q99=data[i99];

    }

    fprintf(stderr, "MODEL dequantize layer %d (%s, bits=%d):\n", layer, proj, b_orig);

    fprintf(stderr, "  n=%d min=%.6f max=%.6f mean=%.6f std=%.6f q1%%=%.6f q99%%=%.6f\n", n, mn, mx, mean, std, q1, q99);

    mlx_array_free(w); mlx_array_free(s); mlx_array_free(b); mlx_array_free(wf);

    return 0;

}
