/* llm_mlx_fused_residual_rms.h — kernel Metal fuso (decode) "residual + RMSNorm":
 *   y = x + w * rmsnorm(r)     (rmsnorm(r) = r * rsqrt(mean(r^2)+eps))
 * Sostituisce la coppia {RMSNorm -> Add} o {Add -> RMSNorm} con UN dispatch.
 * Gate: LLM_FUSE_RESIDUAL_RMS (=0 off, =1 on), LLM_FUSE_RESIDUAL_RMS_VERIFY (calcola entrambi). */
#ifndef LLM_MLX_FUSED_RESIDUAL_RMS_H
#define LLM_MLX_FUSED_RESIDUAL_RMS_H

#include "mlx/c/mlx.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

bool llm_mlx_fused_residual_rms_enabled(void);
bool llm_mlx_fused_residual_rms_verify_enabled(void);

/* y = x + w * rmsnorm(r). x/r: [..., H] stesso dtype; w: [H]. Ritorna false (senza scrivere y_out)
 * su geometria non supportata o launch fallito -> il chiamante ripiega sull'originale. */
bool llm_mlx_fused_residual_rms(
    mlx_array* y_out, mlx_array x, mlx_array r, mlx_array w, float eps, mlx_stream stream);

#ifdef __cplusplus
}
#endif

#endif
