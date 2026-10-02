/* llm_mlx_array_count.h — §M/L25 + K4: contatore di ownership degli handle `mlx_array`.
 *
 * Wrappa `mlx_array_new*` / `mlx_array_free` con contatori (`net = new - free` deve restare
 * COSTANTE a regime: post-prefill e tra i token). In mlx-c `mlx_array_set(dst, src)` CONDIVIDE il
 * riferimento (non trasferisce): ogni `mlx_array_new()` che resta sorgente di un `set` va liberato,
 * altrimenti il `Data` resta vivo dopo il `detach()` (classe del leak ~2 GB di `rms_norm_cfg`).
 *
 * IMPORTANTE (K4): va incluso **PRIMA** degli header di progetto (`llm_mlx_ops.h`, ...) così anche
 * le chiamate di `mlx_array_new*`/`free` dentro gli **header inline** vengono contate. Prima i macro
 * erano definiti DOPO tutti gli include → contavano solo il `.c` e il net derivava **negativo**
 * (limite di scope, non un leak). L'header include `<mlx/c/mlx.h>` per primo (le dichiarazioni delle
 * funzioni devono precedere i macro).
 *
 * Default `LLM_COUNT_ARRAYS=0` (zero overhead). Attivare con `-DLLM_COUNT_ARRAYS=1`.
 * Vedi Knowledge/LLMEnv-Core-Difetti-vs-mlxlm.md. */
#ifndef LLM_MLX_ARRAY_COUNT_H
#define LLM_MLX_ARRAY_COUNT_H

#include <mlx/c/mlx.h>
#include <stdio.h>   /* fprintf (llm_mlx_array_balance) */

#ifndef LLM_COUNT_ARRAYS
#define LLM_COUNT_ARRAYS 0
#endif

#if LLM_COUNT_ARRAYS
/* Contatori GLOBALI (definiti in llm_mlx_stats.c): tutte le TU incrementano lo STESSO contatore,
 * altrimenti il probe in stats.c leggerebbe i propri (sempre 0). */
extern long g_arr_new_ct, g_arr_free_ct, g_arr_ne_free;
/* Tutte le funzioni di CREAZIONE usate vanno contate, altrimenti il net e' falsato (i free
 * contati senza le new corrispondenti lo mandano negativo). */
static mlx_array llm_counted_new(void) { g_arr_new_ct++; return mlx_array_new(); }
static mlx_array llm_counted_new_data(const void* d, const int* sh, size_t n, mlx_dtype t) { g_arr_new_ct++; return mlx_array_new_data(d, sh, n, t); }
static mlx_array llm_counted_new_f32(float v) { g_arr_new_ct++; return mlx_array_new_float32(v); }
static mlx_array llm_counted_new_float(float v) { g_arr_new_ct++; return mlx_array_new_float(v); }
static mlx_array llm_counted_new_int(int v) { g_arr_new_ct++; return mlx_array_new_int(v); }
static void llm_counted_free(mlx_array a) { g_arr_free_ct++; if (a.ctx != NULL) g_arr_ne_free++; mlx_array_free(a); }
/* Variadiche: i call-site C usano compound literal `(const int[]){a,b}` la cui virgola interna
 * spezzerebbe un macro a N argomenti fissi ("too many arguments", 2 siti dsv41). */
#define mlx_array_new(...) llm_counted_new(__VA_ARGS__)
#define mlx_array_new_data(...) llm_counted_new_data(__VA_ARGS__)
#define mlx_array_new_float32(...) llm_counted_new_f32(__VA_ARGS__)
#define mlx_array_new_float(...) llm_counted_new_float(__VA_ARGS__)
#define mlx_array_new_int(...) llm_counted_new_int(__VA_ARGS__)
#define mlx_array_free(...) llm_counted_free(__VA_ARGS__)
void llm_mlx_array_balance(const char* tag);   /* definita in llm_mlx_stats.c */
#else
#define llm_mlx_array_balance(tag) ((void)0)
#endif

#endif /* LLM_MLX_ARRAY_COUNT_H */
