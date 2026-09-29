/* ============================================================================
 * llm_mlx_embed.c — dominio EMBEDDING (CAP_EMBED, ABI v4): embedding IDS-based.
 *
 * PERCHE' IDS. Il runner MLX e' IDS-only (la tokenizzazione sta a monte, nel consumer): non
 * espone `embed_text` (richiederebbe un tokenizer nel motore). L'embedding si calcola quindi
 * dagli ID gia' tokenizzati: `capture_hidden` (data-gen) produce l'hidden dell'ultimo layer per
 * ogni posizione, e qui se ne fa la MEDIA (mean-pool) su tutte le posizioni.
 *
 * SENZA STATO. Si crea una sessione temporanea, si cattura a chunk (la KV cresce causalmente
 * tra i chunk), si distrugge: la sessione del chiamante e la sua KV non vengono toccate. La
 * media e' lineare, quindi il chunking non cambia il risultato.
 *
 * Non normalizza: ritorna il mean-pool grezzo (dim float). La normalizzazione L2 e' politica del
 * consumer (FastLLM la applica, come fa il percorso llama.cpp).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"

#include <stdlib.h>
#include <string.h>

/* Dimensione di un chunk di cattura: limita la memoria del buffer hidden (chunk * hidden_size
 * float) e resta sotto qualunque prefill_chunk_size configurato. */
#define LLM_MLX_EMBED_CHUNK 512

int llm_ai_mlx_embed_dim(llm_ai_mlx_model_handle handle) {
    LlmMlxModel* m = (LlmMlxModel*)handle;
    if (m == NULL) return -1;
    int d = m->cfg.hidden_size;
    return d > 0 ? d : -1;
}

int llm_ai_mlx_embed_ids(llm_ai_mlx_model_handle handle, const int32_t* token_ids, int n_ids,
                         float* out, int cap) {
    if (handle == NULL || token_ids == NULL || n_ids <= 0 || out == NULL) return -1;
    LlmMlxModel* m = (LlmMlxModel*)handle;
    int dim = m->cfg.hidden_size;
    if (dim <= 0 || cap < dim) return -1;

    llm_ai_mlx_session_handle s = llm_ai_mlx_create_session(handle);
    if (s == NULL) return -1;
    (void)llm_ai_mlx_session_set_prefill_chunk(s, LLM_MLX_EMBED_CHUNK);

    for (int d = 0; d < dim; d++) out[d] = 0.0f;
    int total = 0;
    int rc = -1;
    float* h = (float*)malloc((size_t)LLM_MLX_EMBED_CHUNK * (size_t)dim * sizeof(float));
    if (h == NULL) {
        llm_ai_mlx_destroy_session(s);
        return -1;
    }
    for (int off = 0; off < n_ids; off += LLM_MLX_EMBED_CHUNK) {
        int c = n_ids - off;
        if (c > LLM_MLX_EMBED_CHUNK) c = LLM_MLX_EMBED_CHUNK;
        int got = llm_ai_mlx_session_capture_hidden(s, token_ids + off, c, h, c * dim);
        if (got <= 0) break;
        for (int i = 0; i < got * dim; i++) out[i % dim] += h[i];
        total += got;
    }
    if (total > 0) {
        float inv = 1.0f / (float)total;
        for (int d = 0; d < dim; d++) out[d] *= inv;
        rc = dim;
    }
    free(h);
    llm_ai_mlx_destroy_session(s);
    return rc;
}
