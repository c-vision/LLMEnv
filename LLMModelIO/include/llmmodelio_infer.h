#ifndef LLMMODELIO_INFER_H
#define LLMMODELIO_INFER_H
#include <stdint.h>
/* Forward pass minimale (MoE transformer) per CALIBRARE il routing: conta, per ogni layer, quante
 * volte ogni esperto e' selezionato sui token dati. Riferimento: llama.cpp/src/models/olmoe.cpp.
 * Ritorna un array [n_layer*n_expert] di double (caller: free) o NULL. */

/* Campioni di attivazioni reali (per la stima Monte Carlo di ΔWᵀΣΔW in llmmodelio_roundtrip_rel_cov,
 * vedi llmmodelio_model.c): numero fisso di vettori catturati per ruolo, scritti nel GGUF imatrix
 * come "<tensor>.samples" ([cols, K] o [cols, K*n_expert] per i tensori con esperti). Slot non
 * riempiti restano a zero (calloc) e contribuiscono 0 alla somma — nessun bias, nessun conteggio
 * da propagare al lettore. Le due costanti DEVONO restare identiche fra scrittore (llmmodelio_infer.c)
 * e lettore (llmmodelio_model.c): stesso runtime, ricompilati insieme. */
#define LLMMODELIO_SAMP_K_SHARED 64
#define LLMMODELIO_SAMP_K_EXPERT 48
int llmmodelio_imatrix_compute(const char* model_path, const int32_t* tokens, int32_t n_tokens, const char* out_path, int chunk);
double* llmmodelio_routing_counts(const char* model_path, const int32_t* tokens, int32_t n_tokens,
                               int* out_n_layer, int* out_n_expert);
#endif
