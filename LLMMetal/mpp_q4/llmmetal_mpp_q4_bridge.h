/* llmmetal_mpp_q4_bridge.h — ABI C verso il kernel q4 affine su MPP (tensor unit Apple).
 * Implementazione ObjC++ che usa l'API C++ di MLX per i buffer Metal (zero-copy). La metallib e'
 * build-time (-std=metal4.0 -mmacosx-version-min=26.2): le tensor-op MPP NON funzionano via
 * mlx_fast_metal_kernel (output ~0 silenzioso). */
#ifndef LLMMETAL_MPP_Q4_BRIDGE_H
#define LLMMETAL_MPP_Q4_BRIDGE_H
#include <stdint.h>
#include <mlx/c/mlx.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct LlmMetalMppQ4Ctx LlmMetalMppQ4Ctx;
LlmMetalMppQ4Ctx* llmmetal_mpp_q4_create(const char* metallib_path);
void llmmetal_mpp_q4_destroy(LlmMetalMppQ4Ctx* ctx);
/* Y = X @ dequant(W)^T. x bf16 [M,K] (M in {8,16,32,64}, K%GS==0), w U32 [N,K/8], sc/bi bf16
 * [N,K/GS] (GS in {32,64}), out bf16 [M,N] preallocato (N%128==0). 0 su successo. */
int llmmetal_mpp_q4_forward(mlx_array x, mlx_array w, mlx_array sc, mlx_array bi, mlx_array out,
                       int M, int GS);
/* Variante in-graph (via veloce): costruisce il nodo del grafo MLX e ritorna l'array di output
 * (schema [.., N]). Nessuna synchronize: l'esecuzione avviene nell'eval di MLX. ctx.ctx==NULL se
 * la forma non e' supportata (il chiamante ripiega su qmm). */
mlx_array llmmetal_mpp_q4_prim(mlx_array x, mlx_array w, mlx_array sc, mlx_array bi, mlx_stream s,
                          int M, int GS);
/* Variante "tile" (schema Splash, q4_mpp_tiles.h): M in {16,24,32}, somme d'ingresso condivise
 * per il termine bias, stride diretto dal layout MLX [N, K/8] (nessun repack). x bf16 [M,K]
 * (K%GS==0), sc/bi bf16 [N,K/GS] (GS in {32,64}), out bf16 [M,N] (N%128==0). ctx.ctx==NULL se
 * la forma non e' supportata. E' la variante che BATTE qmm MLX (probe: 1.2-2.7x). */
mlx_array llmmetal_mpp_q4_tile_prim(mlx_array x, mlx_array w, mlx_array sc, mlx_array bi, mlx_stream s,
                               int M, int GS);
#ifdef __cplusplus
}
#endif
#endif
