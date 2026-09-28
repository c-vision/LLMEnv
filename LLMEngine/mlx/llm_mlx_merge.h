/* llm_mlx_merge.h — MODEL MERGING (roadmap 8.11 fase 6).
 *
 * Fusione di pesi/contributi di modelli o adapter compatibili (stesse dimensioni).
 * - weighted sum: out = sum_k w_k * m_k  (media ponderata dei modelli).
 * - TIES: media dei SEGNI dominanti (risolve i conflitti di segno tra modelli).
 * - DARE: sparsifica (droppa i valori piccoli, scala i restanti) prima della somma.
 *
 * Questo e' il fondamento puro su array float (testabile senza MLX); l'applicazione ai
 * pesi reali del motore (shard) e la composizione di modelli/compound/router sono il
 * passo successivo.
 *
 * Convenzione: puro C, nessuno stato globale.
 */
#ifndef LLM_MLX_MERGE_H
#define LLM_MLX_MERGE_H

#include <math.h>
#include <stdlib.h>

/* Weighted sum di `n` vettori di lunghezza `len`, pesi `w`. out = sum_k w[k]*m_k[i]. */
static inline void llm_merge_weighted_sum(const float** mats, const float* w, int n, int len, float* out) {
    for (int i = 0; i < len; i++) {
        double acc = 0.0;
        for (int k = 0; k < n; k++) acc += (double)w[k] * mats[k][i];
        out[i] = (float)acc;
    }
}

/* TIES: per ogni elemento, segno dominante (somma dei segni dei modelli pesati);
 * il valore risultante usa la magnitudo media solo dove il segno e' concorde.
 * Se la somma dei segni e' 0 (conflitto), il valore e' 0. */
static inline void llm_merge_ties(const float** mats, const float* w, int n, int len, float* out) {
    for (int i = 0; i < len; i++) {
        double sign_sum = 0.0, mag_sum = 0.0;
        for (int k = 0; k < n; k++) {
            double v = (double)w[k] * mats[k][i];
            sign_sum += (v > 0) ? 1.0 : (v < 0 ? -1.0 : 0.0);
            mag_sum += fabs(v);
        }
        if (sign_sum == 0.0) out[i] = 0.0f;
        else out[i] = (float)((sign_sum > 0 ? 1.0 : -1.0) * (mag_sum / n));
    }
}

/* DARE: sparsifica (droppa |v|<drop_ratio*max_abs), poi scala i restanti di 1/(1-drop).
 * Opera su ogni modello prima della somma ponderata. */
static inline void llm_merge_dare(const float** mats, const float* w, int n, int len,
                                  float drop_ratio, float* out) {
    if (drop_ratio < 0) drop_ratio = 0;
    if (drop_ratio >= 1) drop_ratio = 0.9f;
    for (int i = 0; i < len; i++) {
        double acc = 0.0;
        for (int k = 0; k < n; k++) {
            double v = (double)w[k] * mats[k][i];
            /* soglia di drop relativa al massimo del modello k */
            double maxabs = 0.0;
            for (int j = 0; j < len; j++) { double a = fabs((double)mats[k][j]); if (a > maxabs) maxabs = a; }
            double thr = maxabs * drop_ratio;
            if (fabs((double)mats[k][i]) >= thr) v /= (1.0 - drop_ratio);
            else v = 0.0;
            acc += v;
        }
        out[i] = (float)acc;
    }
}

#endif /* LLM_MLX_MERGE_H */
