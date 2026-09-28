#ifndef LLMMODELIO_IMATRIX_H
#define LLMMODELIO_IMATRIX_H
#include <stdint.h>
/* Legge un imatrix GGUF (llama-imatrix) per il tensore `model_tensor_name` (cerca
 * "<name>.in_sum2" + "<name>.counts"), ritorna tutti i pesi normalizzati F32. Per i tensori
 * MoE il layout e' [esperto][input], come llama.cpp: NON vengono mediati fra esperti.
 * Scrive il numero totale di pesi in *out_len. NULL se assente/incoerente. Caller: free(). */
float* llmmodelio_imatrix_expert_usage(const char* imatrix_path, const char* model_tensor_name, int64_t* out_len);
float* llmmodelio_imatrix_weights(const char* imatrix_path, const char* model_tensor_name, int64_t* out_len);

/* Legge "<model_tensor_name>.samples" (campioni Monte Carlo di attivazioni reali, vedi
 * llmmodelio_infer.h): tensore GGUF [cols, k_total] (dims[0]=cols piu' veloce). Ritorna il buffer
 * grezzo [k_total*cols] (layout: campione s all'offset s*cols), *out_cols=dims[0], *out_k_total=
 * dims[1] (= LLMMODELIO_SAMP_K_SHARED per i tensori condivisi, LLMMODELIO_SAMP_K_EXPERT*n_expert per
 * quelli per-esperto — il chiamante sa quale caso applica dal nome/ndim del tensore modello).
 * NULL se assente (imatrix vecchio senza campioni: il chiamante deve ricadere sul diagonale). */
float* llmmodelio_imatrix_samples(const char* imatrix_path, const char* model_tensor_name, int64_t* out_k_total, int64_t* out_cols);
#endif
