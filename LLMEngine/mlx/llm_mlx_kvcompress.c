// llm_mlx_kvcompress.c — F0 spike CateKV/Minima-KV
#include "llm_mlx_kvcompress.h"
#include <stdlib.h>
#include <string.h>

// mlx dtypes: check via mlx API if available (F0 spike)
#ifdef __cplusplus
extern "C" {
#endif
bool llm_mlx_kvcompress_has_fp8(void) {
    // F0 spike: verifica se mlx supporta float8 (MLX v0.21+ ha MLX_FLOAT8_E4M3)
    // Per ora, check env LLM_KVFP8_FORCE=1 per forzare (test)
    if (getenv("LLM_KVFP8_FORCE")) return true;
    // Prova a cercare il simbolo mlx dtype (se linkato, altrimenti false → documentato come non disponibile)
    return false; // F0: documenta come non disponibile su mlx locale attuale (0.21.0 non ha fp8)
}
bool llm_mlx_kvcompress_has_tq3(void) { return false; } // TQ3 richiede kernel custom 3-bit, non in mlx

bool llm_mlx_kvcompress_fp8_enabled(void) {
    static int s = -1;
    if (s < 0) s = getenv("LLM_KVFP8") ? 1 : 0;
    return s;
}
bool llm_mlx_kvcompress_tq3_enabled(void) {
    static int s = -1;
    if (s < 0) s = getenv("LLM_KVTQ3") ? 1 : 0;
    return s;
}
bool llm_mlx_kvcompress_verify_enabled(void) {
    static int s = -1;
    if (s < 0) s = getenv("LLM_KVFP8_VERIFY") || getenv("LLM_KVTQ3_VERIFY") ? 1 : 0;
    return s;
}

// F1: slice per età — i token vecchi (>P) vanno in fp8 (dequant manuale su K/V fuso)
int llm_mlx_kvcompress_fp8_slice(int n_tokens, int prefix_len) {
    if (!llm_mlx_kvcompress_fp8_enabled()) return 0;
    return prefix_len;
}
// F2: TQ3 low-importance — quantizzazione 3-bit dei token sotto soglia (heads stabili)
int llm_mlx_kvcompress_tq3_quantize(int n_tokens, float *importance) {
    if (!llm_mlx_kvcompress_tq3_enabled()) return 0;
    int cnt = 0;
    for (int i = 0; i < n_tokens; i++) if (importance[i] < 0.3f) cnt++;
    return cnt;
}
#ifdef __cplusplus
}
#endif
