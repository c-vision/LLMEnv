/* llmmetal_oq_a8_bridge.h — ABI C verso i kernel oQ A8 (int8-activation × Q4/Q5, tensor unit M5).
 * Implementazione ObjC++ (llmmetal_oq_a8_bridge.mm) che usa l'API C++ di MLX per accedere ai
 * BUFFER METAL degli array (zero-copy: nessuna copia host<->device, nessun puntatore host).
 * Layout (permutazione v8, trasposizioni) fatto con op MLX (GPU). Vedi
 * Knowledge/oQ-A8-INT8-Activation-GEMM-M5-NAX.md. */
#ifndef LLMMETAL_OQ_A8_BRIDGE_H
#define LLMMETAL_OQ_A8_BRIDGE_H

#include <stdint.h>
#include <mlx/c/mlx.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LlmMetalOqA8Ctx LlmMetalOqA8Ctx;   /* opaco lato C */

/* Carica le due metallib (qmm NAX + quantize). NULL se non disponibili. */
LlmMetalOqA8Ctx* llmmetal_oq_a8_create(const char* qmm_lib, const char* quant_lib);
void llmmetal_oq_a8_destroy(LlmMetalOqA8Ctx* ctx);

/* Esegue Y = X · W^T con attivazioni int8 e pesi Q4/Q5 (bits 4/5), ZERO-COPY sui buffer Metal
 * degli array MLX. Vincoli: x f16 [.., K] con K%64==0; w U32 [N, K*bits/32] con N%64==0;
 * scales/biases f16 [N, G] (G=K/64, layout MLX); out f16 con l'ultima dim = N (già allocato dal
 * chiamante, tipicamente mlx_zeros). Ritorna 0 su successo, <0 altrimenti. */
int llmmetal_oq_a8_forward(mlx_array x, mlx_array w, mlx_array scales, mlx_array biases,
                      mlx_array out, mlx_stream stream, int bits);

/* Variante **primitive MLX** (in-graph, una sola submit, zero sync per matmul): ritorna l'array
 * di output (f16, ultima dim = N) da usare nel grafo; se la costruzione fallisce ritorna un
 * mlx_array con ctx NULL. bits==4. */
mlx_array llmmetal_oq_a8_prim(mlx_array x, mlx_array w, mlx_array scales, mlx_array biases, mlx_stream stream, int bits);

/* FEDELTA' oMLX: Stage A separata e CONDIVISA (qwen35_oq_a8.py: stage_a + apply_plan).
 * Quantizza UNA volta l'attivazione e riusa (qa,sa,ra) per ogni proiezione che la consuma.
 * Convenzione bridge: mlx_array NON inizializzati (ctx NULL), liberare con mlx_array_free. */
int llmmetal_oq_a8_stage_a(mlx_array x, mlx_array* qa, mlx_array* sa, mlx_array* ra, mlx_stream stream);
mlx_array llmmetal_oq_a8_qmm(mlx_array qa, mlx_array sa, mlx_array ra, mlx_array w, mlx_array scales,
                        mlx_array biases, mlx_stream stream, int bits, int want_bf16);

/* N-FUSIONE: due proiezioni (gate+up) in UNA QMM con N concatenato. A M piccolo la griglia e'
 * N/64 x 1: raddoppiare N raddoppia i blocchi e la banda sale (~2x misurato: 67 -> 145 GB/s a
 * N 12288 -> 24576). Il concat dei pesi e' CACHATO per coppia (buffer ptr) ed e' una copia
 * una-tantum: +100% di memoria per quei tensori (env-gated). out = [.., N1+N2]: il chiamante
 * taglia le due meta' con mlx_slice. Ritorna 0 su successo. */
int llmmetal_oq_a8_qmm_fused2(mlx_array* out, mlx_array qa, mlx_array sa, mlx_array ra,
                         mlx_array w1, mlx_array s1, mlx_array b1,
                         mlx_array w2, mlx_array s2, mlx_array b2,
                         mlx_stream stream, int bits, int want_bf16);

#ifdef __cplusplus
}
#endif

#endif
