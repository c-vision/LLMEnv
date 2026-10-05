/* llm_mlx_recompute.c — §9.A M10a (azione RUNTIME): ricostruzione di un range KV per ri-prefill.
 *
 * Quando il KV di un range [from,to) è stato scartato/evaso, l'azione RECOMPUTE lo RICOSTRUISCE
 * ri-eseguendo il forward di quel range (ri-prefill) invece di ricaricarlo dal disco: rewind a
 * `from` (mantiene `from` token) + append dei token [from,to). Compone le primitive di sessione
 * già verificate (nessuna logica duplicata). La DECISIONE (recompute vs persist) è del modello di
 * costo `llm_mlx_kvcost`.
 */
#include "llmengine_mlx.h"

int llm_ai_mlx_session_recompute_range(llm_ai_mlx_session_handle sh, llm_ai_mlx_model_handle mh,
                                       const int32_t* tokens, int from, int to) {
    if (sh == NULL || mh == NULL || tokens == NULL) return -1;
    if (from < 0 || to <= from) return -1;
    /* Ricostruisci il KV del range: riporta la sessione a `from` token, poi ri-prefilla [from,to). */
    int rc = llm_ai_mlx_session_rewind(sh, mh, tokens, to, from);
    if (rc != 0) return -1;
    return llm_ai_mlx_session_append(sh, tokens + from, to - from);
}
