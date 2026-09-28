/* llm_mlx_moe_stream.h — helper HOST per lo streaming esperti (passo b.2, additivo).
 *
 * Perché. Il path MoE dell'engine (`llm_mlx_moe.c`) passa a `mlx_gather_qmm` il tensore
 * STACKED completo [n_experts, out, in] + gli indici GLOBALI del router. Per streammare (solo
 * i k bundle attivi dallo store, `llm_mlx_expert_store.h`) bisogna prima:
 *   (1) calcolare gli esperti UNICI attivi fra tutte le righe (n_tokens*top_k);
 *   (2) rimappare ogni riga -> posizione locale [0, n_unique) nel piccolo stacked.
 * Questo file fa SOLO (1)+(2) in puro C: nessuna dipendenza MLX, non tocca il path esistente.
 * Il passo successivo (MLX) impila i bundle degli esperti di `uniq[]` e chiama gather_qmm con
 * `pos[]` come indici. Vedi Knowledge/MLX-Modello-Qwen4-Exp.md. */
#ifndef LLM_MLX_MOE_STREAM_H
#define LLM_MLX_MOE_STREAM_H

/* Dato `ids` [M] (id esperto per riga), riempie:
 *   pos[i]  = posizione locale (0..n_unique-1) dell'esperto della riga i
 *   uniq[j] = id esperto della posizione j  (j in [0, n_unique))
 * `seen` e' uno scratch del chiamante di dimensione >= n_experts, inizializzato a -1.
 * Ritorna n_unique (>=0), oppure -1 se un id e' fuori [0, n_experts).
 * Capacita' richieste: pos[M], uniq[min(M,n_experts)]. */
static inline int llm_mlx_stream_unique(const int* ids, int M, int n_experts,
                                        int* seen, int* pos, int* uniq) {
    int nu = 0;
    for (int i = 0; i < M; i++) {
        int e = ids[i];
        if (e < 0 || e >= n_experts) return -1;
        if (seen[e] < 0) { seen[e] = nu; uniq[nu++] = e; }
        pos[i] = seen[e];
    }
    return nu;
}

#endif /* LLM_MLX_MOE_STREAM_H */
