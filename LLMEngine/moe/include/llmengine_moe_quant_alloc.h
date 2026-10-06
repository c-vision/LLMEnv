/* llmengine_moe_quant_alloc.h — §E3: allocazione bit per-layer/per-tensore guidata da SENSIBILITÀ.
 *
 * Idea (riferimento esterno `mlx-optiq`, metodo `static`): misurare offline, per ogni tensore,
 * l'errore di ricostruzione della quantizzazione a un dato numero di bit (pesato dall'importanza
 * delle colonne quando disponibile, es. imatrix §E2) e allocare più bit ai tensori più sensibili
 * sotto un budget di bit medi (bpw). Un solo misuratore, due consumatori (bit di quantizzazione e,
 * in futuro, rank LoRA). Nessuna calibrazione: la misura è sui PESI (non serve un forward).
 *
 * Vive nel runtime condiviso (§0bis): VM/C, C99 e i tool la condividono. */
#ifndef LLMENGINE_MOE_QUANT_ALLOC_H
#define LLMENGINE_MOE_QUANT_ALLOC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Errore relativo di quantizzare x[rows*cols] (row-major) a `bits`/`group` col quantizzatore
 * affine RTN di llmcodec_moe_quant, misurato come
 *     rel = sqrt( Σ w_i (x_i - x̂_i)² / Σ w_i x_i² )
 * dove w_i = imp[colonna(i)] se `imp`!=NULL (importanze per-colonna, `cols` valori, come l'imatrix),
 * altrimenti 1 (RMS uniforme). `imp` NON altera il quantizzatore (RTN), pesa solo l'ERRORE: la misura
 * resta O(1) per tensore/candidato (nessuna grid-search) ed è comparabile tra tensori.
 * Ritorna 0 e scrive *out_rel; <0 su parametri invalidi o memoria. */
int llmengine_moe_quant_rel_error(const float* x, const float* imp, uint32_t rows, uint32_t cols,
                        uint32_t group, uint32_t bits, double* out_rel);

/* Allocatore greedy (variante knapsack a budget medio): N tensori, `nparams[i]` pesi quantizzabili,
 * `rel[i*ncand + c]` = errore relativo del tensore i al candidato `cand[c]` (bit CRESCENTI).
 * Parte da tutti al candidato 0 e, finché la media pesata dei bit (per nparams) resta
 * ≤ `target_avg_bits`, promuove di un candidato il tensore con il miglior rapporto
 * (riduzione d'errore)/(bit spesi). Solo promozioni con guadagno positivo (non spreca budget).
 * Scrive `out_cand[i]` (indice del candidato scelto) e ritorna 0; -1 su input invalido. */
int llmengine_moe_quant_alloc_bits(const uint64_t* nparams, const double* rel, int n, int ncand,
                         const int* cand, double target_avg_bits, int* out_cand);

/* §6.1 S3 — variante ANCORATA: come sopra, ma con un FLOOR per-tensore `min_cand[i]` (>=0, indice
 * di candidato). `out_cand` parte dal floor (la ricetta baseline per ruolo) e il greedy promuove
 * SOLO sopra il floor entro il budget medio. Ritorna 0 se rispetta il budget, -2 se il floor da
 * solo lo supera (infeasible; `out_cand` resta il floor), -1 su input invalido. */
int llmengine_moe_quant_alloc_bits_min(const uint64_t* nparams, const double* rel, int n, int ncand,
                         const int* cand, const int* min_cand, double target_avg_bits, int* out_cand);

#ifdef __cplusplus
}
#endif

#endif /* LLMENGINE_MOE_QUANT_ALLOC_H */
