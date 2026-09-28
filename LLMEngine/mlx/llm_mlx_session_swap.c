/* ============================================================================
 * llm_mlx_session_swap.c — swap-out/swap-in sessione su disco (KV safetensors + meta binario; path DS4.1 whole-model park/restore).
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llm_dsv41_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* FASE 2b: swap-out su disco -- vedi il commento di MlxSwapMeta sopra per il perche' del
 * formato a due file (.mlxkv safetensors + .mlxmeta binario) e dell'assenza di model_handle
 * nel meta. Precondizione has_pending: una sessione senza alcun append non ha nulla di
 * significativo da persistere, fallisce esplicitamente invece di scrivere un file vuoto.
 * Stesso schema .tmp + rename() di llm_ai_session_swap_out in llmengine_llama.cpp per
 * l'atomicita' (un crash a meta' scrittura non deve mai corrompere uno swap precedente allo
 * stesso prefix). */
int llm_ai_mlx_session_swap_out(llm_ai_mlx_session_handle handle, const char* path_prefix) {
    if (handle == NULL || path_prefix == NULL) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    if (!s->has_pending) return -1;
    LlmMlxModel* m = s->model;

    /* DeepSeek V4.1 whole-model: park via history (il cache si ricostruisce al restore). */
    if (m->dsv41) {
        LlmDsv41Cache* c = (LlmDsv41Cache*)s->dsv41;
        if (!c || !s->history || s->history_len <= 0) return -1;
        char path[1100], tmp[1110];
        snprintf(path, sizeof path, "%s.dsv41meta", path_prefix);
        snprintf(tmp, sizeof tmp, "%s.tmp", path);
        FILE* f = fopen(tmp, "wb");
        if (!f) return -1;
        int32_t hdr[5] = { (int32_t)0x4D313444, c->offset, s->pending_next_id,
                           s->has_pending ? 1 : 0, s->history_len };
        if (fwrite(hdr, 4, 5, f) != 5 || fwrite(s->history, 4, (size_t)s->history_len, f) != (size_t)s->history_len) {
            fclose(f); remove(tmp); return -1;
        }
        if (fclose(f) != 0 || rename(tmp, path) != 0) { remove(tmp); return -1; }
        llm_dsv41_cache_free(c); free(c); s->dsv41 = NULL; s->parked = true;
        return 0;
    }

    char kv_tmp[1024], kv_final[1024], meta_tmp[1024], meta_final[1024];
    snprintf(kv_tmp, sizeof(kv_tmp), "%s.mlxkv.tmp", path_prefix);
    snprintf(kv_final, sizeof(kv_final), "%s.mlxkv", path_prefix);
    snprintf(meta_tmp, sizeof(meta_tmp), "%s.mlxmeta.tmp", path_prefix);
    snprintf(meta_final, sizeof(meta_final), "%s.mlxmeta", path_prefix);

    mlx_map_string_to_array params = mlx_map_string_to_array_new();
    mlx_map_string_to_string meta_map = mlx_map_string_to_string_new();
    bool ok = true;
    for (int layer = 0; ok && layer < m->cfg.n_layers; layer++) {
        char kname[32], vname[32];
        snprintf(kname, sizeof(kname), "k.%d", layer);
        snprintf(vname, sizeof(vname), "v.%d", layer);
        ok = mlx_map_string_to_array_insert(params, kname, s->k_cache[layer]) == 0 &&
             mlx_map_string_to_array_insert(params, vname, s->v_cache[layer]) == 0;
    }
    if (ok) ok = mlx_map_string_to_array_insert(params, "pending_h", s->pending_h) == 0;
    if (ok) ok = mlx_save_safetensors(kv_tmp, params, meta_map) == 0;
    mlx_map_string_to_array_free(params);
    mlx_map_string_to_string_free(meta_map);

    char kv_tmp_st[1040];
    snprintf(kv_tmp_st, sizeof(kv_tmp_st), "%s.safetensors", kv_tmp);
    if (!ok) { remove(kv_tmp_st); return -1; }

    MlxSwapMeta meta = {0};
    meta.cache_offset = s->cache_offset;
    meta.pending_next_id = s->pending_next_id;
    meta.has_pending = s->has_pending;
    meta.n_layers = m->cfg.n_layers;
    meta.hidden_size = m->cfg.hidden_size;
    if (!write_mlx_meta_file(meta_tmp, &meta)) {
        remove(kv_tmp_st);
        remove(meta_tmp);
        return -1;
    }

    /* Commit: rename() e' atomico per-file sullo stesso filesystem, quindi un crash tra
     * queste due rename non lascia mai un file a meta' scritto al path finale -- il caso
     * peggiore e' un .mlxkv fresco accanto a un .mlxmeta stale/assente, che swap-in rileva
     * (magic/dimensioni non combacianti) e rifiuta invece di caricare stato incoerente.
     * NOTA estensione: mlx_save_safetensors APPENDE ".safetensors" al path passato (vedi
     * mlx::core::save_safetensors) -- il file reale scritto e' kv_tmp+".safetensors", quindi
     * la rename finale deve usare quel nome, non kv_tmp nudo. */
    char kv_final_st[1040];
    snprintf(kv_final_st, sizeof(kv_final_st), "%s.safetensors", kv_final);
    if (rename(kv_tmp_st, kv_final_st) != 0) {
        remove(kv_tmp_st);
        remove(meta_tmp);
        return -1;
    }
    if (rename(meta_tmp, meta_final) != 0) {
        remove(meta_tmp);
        return -1;
    }
    return 0;
}

/* Ricrea una sessione da file scritti da llm_ai_mlx_session_swap_out. model_handle e'
 * ricevuto esplicitamente dal chiamante (il completer C# lo ha gia' come campo privato) --
 * vedi il commento su MlxSwapMeta sul perche' non viaggia nel meta-file. n_layers/
 * hidden_size nel meta sono confrontati contro il modello prima di caricare qualunque dato,
 * cosi' un modello incompatibile fallisce esplicitamente invece di produrre garbage. */
llm_ai_mlx_session_handle llm_ai_mlx_session_swap_in(llm_ai_mlx_model_handle model_handle,
                                                      const char* path_prefix) {
    if (model_handle == NULL || path_prefix == NULL) return NULL;
    LlmMlxModel* m = (LlmMlxModel*)model_handle;

    /* DeepSeek V4.1 whole-model: restore da history + rebuild del cache. */
    if (m->dsv41) {
        char dpath[1100];
        snprintf(dpath, sizeof dpath, "%s.dsv41meta", path_prefix);
        FILE* f = fopen(dpath, "rb");
        if (!f) return NULL;
        int32_t hdr[5];
        if (fread(hdr, 4, 5, f) != 5 || (uint32_t)hdr[0] != 0x4D313444u) { fclose(f); return NULL; }
        int hn = hdr[4];
        int32_t* toks = hn > 0 ? (int32_t*)malloc((size_t)hn * 4) : NULL;
        if (hn > 0 && (!toks || fread(toks, 4, (size_t)hn, f) != (size_t)hn)) { fclose(f); free(toks); return NULL; }
        fclose(f);
        LlmMlxSession* s = llm_mlx_session_create(m);
        if (!s) { free(toks); return NULL; }
        if (hn > 0) session_history_append(s, toks, hn);
        llm_mlx_sampling_t sm0 = llm_mlx_session_sampling(s);
        LlmDsv41Sampling ds = { sm0.active, sm0.temperature, sm0.top_p, sm0.top_k, sm0.seed,
                                sm0.repetition_penalty, sm0.frequency_penalty, sm0.presence_penalty,
                                sm0.hist, sm0.hist_n };
        int32_t pending = 0;
        if (!llm_dsv41_engine_prefill((LlmDsv41Engine*)m->dsv41, &s->dsv41, toks, hn,
                                      &s->cache_offset, &pending, &s->has_pending, &ds, m->stream)) {
            free(toks); llm_mlx_session_destroy(s); return NULL;
        }
        s->pending_next_id = pending;
        free(toks);
        return s;
    }

    char kv_path[1024], meta_path[1024];
    snprintf(kv_path, sizeof(kv_path), "%s.mlxkv", path_prefix);
    snprintf(meta_path, sizeof(meta_path), "%s.mlxmeta", path_prefix);

    MlxSwapMeta meta;
    if (!read_mlx_meta_file(meta_path, &meta)) return NULL;
    if (meta.n_layers != m->cfg.n_layers || meta.hidden_size != m->cfg.hidden_size) {
        fprintf(stderr, "llm_ai_mlx_session_swap_in: modello incompatibile (n_layers/hidden_size non combaciano)\n");
        return NULL;
    }

    mlx_map_string_to_array params = mlx_map_string_to_array_new();
    mlx_map_string_to_string meta_map = mlx_map_string_to_string_new();
    /* mlx_load_safetensors legge il path ESATTO (nessun append di estensione, a differenza di
     * save) -- e il file scritto da swap_out e' <prefix>.mlxkv.safetensors (save appende).
     * Stessa disciplina del load dei pesi in llm_ai_mlx_load_model: il caricamento da disco
     * va su uno stream CPU ("Load::eval_gpu Not implemented" altrimenti), memoria unificata
     * Apple Silicon = utilizzabile dalle op GPU senza copie esplicite. */
    mlx_stream cpu_stream = mlx_default_cpu_stream_new();
    char kv_path_st[1040];
    snprintf(kv_path_st, sizeof(kv_path_st), "%s.safetensors", kv_path);
    if (mlx_load_safetensors(&params, &meta_map, kv_path_st, cpu_stream) != 0) {
        mlx_stream_free(cpu_stream);
        mlx_map_string_to_array_free(params);
        mlx_map_string_to_string_free(meta_map);
        return NULL;
    }
    mlx_stream_free(cpu_stream);
    mlx_map_string_to_string_free(meta_map);

    LlmMlxSession* s = (LlmMlxSession*)calloc(1, sizeof(LlmMlxSession));
    s->model = m;
    s->n_layers = m->cfg.n_layers;
    s->k_cache = (mlx_array*)malloc((size_t)m->cfg.n_layers * sizeof(mlx_array));
    s->v_cache = (mlx_array*)malloc((size_t)m->cfg.n_layers * sizeof(mlx_array));
    s->aux_cache = (mlx_array*)malloc((size_t)m->cfg.n_layers * sizeof(mlx_array));
    s->has_cache = (bool*)malloc((size_t)m->cfg.n_layers * sizeof(bool));
    bool ok = true;
    for (int layer = 0; ok && layer < m->cfg.n_layers; layer++) {
        char kname[32], vname[32];
        snprintf(kname, sizeof(kname), "k.%d", layer);
        snprintf(vname, sizeof(vname), "v.%d", layer);
        mlx_array k = mlx_array_new(), v = mlx_array_new();
        ok = mlx_map_string_to_array_get(&k, params, kname) == 0 &&
             mlx_map_string_to_array_get(&v, params, vname) == 0;
        s->k_cache[layer] = k;
        s->v_cache[layer] = v;
        s->aux_cache[layer] = mlx_array_new();
    }
    for (int layer = 0; layer < m->cfg.n_layers; layer++) s->has_cache[layer] = ok;
    s->pending_h = mlx_array_new();
    if (ok) ok = mlx_map_string_to_array_get(&s->pending_h, params, "pending_h") == 0;
    mlx_map_string_to_array_free(params);

    if (!ok) {
        llm_mlx_session_destroy(s);
        return NULL;
    }

    s->cache_offset = meta.cache_offset;
    s->pending_next_id = meta.pending_next_id;
    s->has_pending = meta.has_pending;
    return s;
}
