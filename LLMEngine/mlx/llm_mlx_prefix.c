/* llm_mlx_prefix.c — dominio PREFIX/SESSION (kvdisk prefix save/restore/park/rewind,
 * checkpoint, evict LRU, append prefix, last_hidden). Estratto VERBATIM da llmengine_mlx.c
 * (split fase 2). */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "llm_mlx_kvdisk_io.h"
#include "llm_mlx_kvclasses.h"   /* §9.A M10d: classi KV RUNTIME/PERSISTENT */
#include <sys/stat.h>   /* mkdir (D15: dir cache prefix) */
#include "llm_pflash.h"

/* D15: path EFFETTIVO della prefix-cache persistente (PER-MODELLO, evita collisioni tra modelli):
 *   "off"/"0"/"none" -> "" (nessuna persistenza, in-RAM);  "auto"/""/"1"/"on" -> <base>/prefix_<hash(model_dir)>.cache
 *   con <base> = $HOME/.cache/llm_engine (o $TMPDIR/llm_engine);  qualunque altro valore = path esplicito. */
const char* llm_prefix_cache_path(LlmMlxModel* m, char* buf, int cap) {
    if (!buf || cap <= 0) return "";
    buf[0] = '\0';
    char v[2048] = "";
    int is_set = (m && llm_ai_mlx_opt_get_str("prefix_cache", v, (int)sizeof v) == 0);
    if (is_set && (strcmp(v, "off") == 0 || strcmp(v, "0") == 0 || strcmp(v, "none") == 0)) return buf;
    if (is_set && v[0] != '\0' && strcmp(v, "auto") != 0 && strcmp(v, "1") != 0 && strcmp(v, "on") != 0) {
        snprintf(buf, (size_t)cap, "%s", v); return buf;
    }
    const char* home = getenv("HOME");
    const char* base = (home && home[0]) ? home : (getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
    char dir[2100];
    snprintf(dir, sizeof dir, "%s/.cache/llm_engine", base);
    (void)mkdir(dir, 0700);
    const char* md = (m && m->model_dir[0]) ? m->model_dir : "model";
    uint64_t h = llm_kv_disk_hash(md, strlen(md));
    snprintf(buf, (size_t)cap, "%s/prefix_%016llx.cache", dir, (unsigned long long)h);
    return buf;
}

bool llm_kvdisk_enabled(LlmMlxModel* m) {
    if (!m) return false;
    if (m->kvdisk_state < 0) {
        /* D15: se `prefix_cache` e' impostata il tier e' PERSISTENTE (path `<prefix_cache>.kv`):
         * i blob KV dei prefissi sopravvivono al riavvio. Altrimenti scratch storico (`kvdisk`). */
        char pc[2048];
        if (llm_prefix_cache_path(m, pc, (int)sizeof pc)[0] != '\0') {
            char kpath[2200];
            snprintf(kpath, sizeof kpath, "%s.kv", pc);
            int cap_mb = 0; (void)llm_ai_mlx_opt_get_int("kvdisk_cap_mb", &cap_mb);
            uint64_t cap = (cap_mb > 0) ? (uint64_t)cap_mb * 1024ull * 1024ull : 0;
            m->kvdisk = llm_kv_disk_open_mode(kpath, /*max_entries=*/8192, cap, /*persistent=*/1);
            m->kvdisk_state = (m->kvdisk.f != NULL && m->kvdisk.e != NULL) ? 1 : 0;
            if (m->kvdisk_state == 1)
                fprintf(stderr, "[kvdisk] prefix-cache PERSISTENTE: %s\n", kpath);
            /* §9.A M10d: registra il tier KV PERSISTENTE (disco) nel manager delle classi KV. */
            { char tkv[96]; snprintf(tkv, sizeof tkv, "kvdisk.%p", (void*)m);
              (void)llm_mlx_kvclasses_register(tkv, LLM_KVCLASS_PERSISTENT, cap); }
        } else {
        /* Registro opzioni (niente env): `kvdisk` = off|on|<path>, `kvdisk_cap_mb` = budget. */
        char kv[2048];
        int is_set = (llm_ai_mlx_opt_get_str("kvdisk", kv, (int)sizeof kv) == 0);
        if (!is_set || kv[0] == '\0' || strcmp(kv, "0") == 0 || strcmp(kv, "off") == 0) {
            m->kvdisk_state = 0;
        } else {
            const char* path = (strcmp(kv, "1") == 0 || strcmp(kv, "on") == 0) ? "/tmp/llm_kvdisk.scratch" : kv;
            int cap_mb = 0;
            (void)llm_ai_mlx_opt_get_int("kvdisk_cap_mb", &cap_mb);
            uint64_t cap = (cap_mb > 0) ? (uint64_t)cap_mb * 1024ull * 1024ull : 0;
            m->kvdisk = llm_kv_disk_open(path, /*max_entries=*/8192, cap);
            m->kvdisk_state = (m->kvdisk.f != NULL && m->kvdisk.e != NULL) ? 1 : 0;
            if (m->kvdisk_state == 1)
                fprintf(stderr, "[kvdisk] tier SSD KV attivo: %s (cap=%llu MB)\n",
                        path, (unsigned long long)(cap >> 20));
        }
        }
    }
    return m->kvdisk_state == 1;
}

/* D15/B7: identita' di FORMATO della KV (bf16 | int<bits>:<thr>) come SALt nelle chiavi della
 * prefix-cache/checkpoint/park: blocchi di formati diversi NON devono attraversarsi (riusare una
 * KV fp16 in una sessione int8 sarebbe errato). Vedi llm_ai_mlx_model_kv_format(). */
static uint64_t llm_prefix_fmt_salt(LlmMlxModel* m) {
    char fmt[32] = "bf16";
    if (m != NULL) llm_ai_mlx_model_kv_format((llm_ai_mlx_model_handle)m, fmt, sizeof fmt);
    return llm_kv_disk_hash(fmt, strlen(fmt));
}
static uint64_t llm_key_with_fmt(uint64_t tok_hash, uint64_t fmt_salt) {
    uint64_t z = tok_hash ^ (fmt_salt + 0x9E3779B97F4A7C15ULL + (tok_hash << 6) + (tok_hash >> 2));
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
/* Chiave del prefisso: FNV-1a dei token (dipende da tutto il contesto) + salt di formato. */
static uint64_t llm_prefix_kvdisk_key(LlmMlxModel* m, const int32_t* toks, int n) {
    return llm_key_with_fmt(llm_kv_disk_hash(toks, (size_t)n * sizeof(int32_t)), llm_prefix_fmt_salt(m));
}

/* Spilla su disco la KV dei soli layer GLOBAL (i local sliding-window sono ricomputabili
 * O(window), §G-ter) e libera gli array MLX (tokens/name/history restano). 0 ok / -1. */
static int llm_prefix_spill(LlmMlxModel* m, SavedPrefix* p) {
    if (!m || !p || !p->has_kv || p->token_len <= 0 || m->kvdisk_state != 1) return -1;
    /* Modelli con layer SLIDING (gpt-oss): la KV sliding e' finestrata e NON viene spillata
     * (vedi skip sotto), quindi il blob su disco resterebbe PARZIALE -> alla reuse si
     * imposta cache_offset=N con i layer sliding vuoti => CMLX "non-empty mlx_array".
     * Non spillare affatto: la reuse fallira' il restore e fara' un prefill completo
     * (fail-open, corretto). In RAM il riuso resta valido. */
    if (m->cfg.is_sliding_layer) return -1;
    uint8_t* buf = NULL; size_t len = 0, cap = 0;
    if (llm_kvdisk_buf_reserve(&buf, &cap, len, 20) != 0) return -1;
    uint32_t magic = LLM_KVDISK_BLOB_MAGIC, ver = LLM_KVDISK_BLOB_VER, nt = 0;
    int32_t coff = p->cache_offset, tlen = p->token_len;
    memcpy(buf + 0, &magic, 4); memcpy(buf + 4, &ver, 4);
    memcpy(buf + 8, &coff, 4);  memcpy(buf + 12, &tlen, 4);
    memcpy(buf + 16, &nt, 4);
    len = 20;
    for (int l = 0; l < p->n_layers; l++) {
        if (m->cfg.is_sliding_layer && l < m->cfg.n_layers && m->cfg.is_sliding_layer[l]) continue;
        if (!p->has_cache || !p->has_cache[l]) continue;
        if (llm_kvdisk_blob_append_tensor(&buf, &len, &cap, p->k_cache[l], l, 0) == 0) nt++;
        if (llm_kvdisk_blob_append_tensor(&buf, &len, &cap, p->v_cache[l], l, 1) == 0) nt++;
    }
    if (nt == 0) { free(buf); return -1; }
    memcpy(buf + 16, &nt, 4);
    int rc = llm_kv_disk_put(&m->kvdisk, llm_prefix_kvdisk_key(m, p->tokens, p->token_len), buf, len);
    free(buf);
    if (rc != 0) return -1;
    /* NB: has_cache NON viene toccato qui; il chiamante libera gli array dei soli layer
     * spillati (global). I layer local (sliding) restano in RAM con has_cache=true. */
    m->kvdisk_spills++;
    return 0;
}

/* Ripristina da disco la KV spillata nel prefisso p (layer global). 0 ok / -1. */
static int llm_prefix_restore(LlmMlxModel* m, SavedPrefix* p) {
    if (!m || !p || p->token_len <= 0 || m->kvdisk_state != 1) return -1;
    /* Speculare alla spill: un blob di un modello SLIDING e' intrinsecamente PARZIALE (la KV
     * finestrata non e' salvata) -> rifiutare il restore. La reuse fara' un prefill completo. */
    if (m->cfg.is_sliding_layer) return -1;
    uint64_t key = llm_prefix_kvdisk_key(m, p->tokens, p->token_len);
    size_t need = 0;
    if (!llm_kv_disk_has(&m->kvdisk, key, &need) || need < 20) return -1;
    uint8_t* buf = (uint8_t*)malloc(need);
    if (!buf) return -1;
    size_t got = 0;
    if (llm_kv_disk_get(&m->kvdisk, key, buf, need, &got) != 0 || got < 20) { free(buf); return -1; }
    uint32_t magic = 0, ver = 0, nt = 0; int32_t coff = 0;
    memcpy(&magic, buf + 0, 4); memcpy(&ver, buf + 4, 4);
    memcpy(&coff, buf + 8, 4);  memcpy(&nt, buf + 16, 4);
    if (magic != LLM_KVDISK_BLOB_MAGIC || ver != LLM_KVDISK_BLOB_VER) { free(buf); return -1; }
    if (!p->k_cache) p->k_cache = (mlx_array*)calloc((size_t)m->cfg.n_layers, sizeof(mlx_array));
    if (!p->v_cache) p->v_cache = (mlx_array*)calloc((size_t)m->cfg.n_layers, sizeof(mlx_array));
    if (!p->has_cache) p->has_cache = (bool*)calloc((size_t)m->cfg.n_layers, sizeof(bool));
    if (!p->k_cache || !p->v_cache || !p->has_cache) { free(buf); return -1; }
    if (p->n_layers == 0) p->n_layers = m->cfg.n_layers;
    size_t off = 20; int n_ok = 0;
    for (uint32_t i = 0; i < nt; i++) {
        mlx_array a = mlx_array_new(); int layer = -1, kind = -1;
        if (llm_kvdisk_blob_read_tensor(buf, got, &off, &a, &layer, &kind) != 0) { mlx_array_free(a); break; }
        if (layer < 0 || layer >= m->cfg.n_layers) { mlx_array_free(a); break; }
        mlx_array* dst = (kind == 0) ? &p->k_cache[layer] : &p->v_cache[layer];
        mlx_array_set(dst, a);
        mlx_array_free(a);
        p->has_cache[layer] = true;
        n_ok++;
    }
    free(buf);
    if (n_ok == 0) return -1;
    p->cache_offset = coff;
    p->has_kv = true;
    p->on_disk = false;
    m->kvdisk_restores++;
    return 0;
}

/* §G-ter — CHECKPOINT/REWIND SU DISCO (agentic rewind). Chiave = token-prefix; blob = layer
 * GLOBAL + cache_offset. Scritto ai confini di m->kvckpt_block token durante il prefill.
 * Scopo: dopo un rewind (l'agent loop rimuove il reasoning) ripristinare il prefisso piu' lungo
 * disponibile e ri-prefillare SOLO il suffisso mancante, invece dell'intero prompt. */
static uint64_t llm_session_ckpt_key(LlmMlxModel* m, const int32_t* toks, int n) {
    return llm_key_with_fmt(llm_kv_disk_hash(toks, (size_t)n * sizeof(int32_t)), llm_prefix_fmt_salt(m));
}

/* Serializza i layer della sessione alla posizione pos su llm_kv_disk. 0/-1.
 * key_tok/key_n (P3, 2026-09-19): prefisso dei token per la CHIAVE quando si checkpointa
 * DENTRO il prefill chunked (la history della sessione non contiene ancora token_ids).
 * NULL -> usa s->history (comportamento storico: checkpoint a fine append). */
int llm_session_ckpt_write_k(LlmMlxModel* m, LlmMlxSession* s, int pos,
                                    const int32_t* key_tok, int key_n) {
    if (!m || !s || pos <= 0 || m->kvdisk_state != 1) return -1;
    if (key_tok == NULL && !s->history) return -1;
    uint8_t* buf = NULL; size_t len = 0, cap = 0;
    if (llm_kvdisk_buf_reserve(&buf, &cap, len, 20) != 0) return -1;
    uint32_t magic = LLM_KVDISK_BLOB_MAGIC, ver = LLM_KVDISK_BLOB_VER, nt = 0;
    int32_t coff = pos, tlen = pos;
    memcpy(buf + 0, &magic, 4); memcpy(buf + 4, &ver, 4);
    memcpy(buf + 8, &coff, 4);  memcpy(buf + 12, &tlen, 4);
    memcpy(buf + 16, &nt, 4);
    len = 20;
    for (int l = 0; l < s->n_layers; l++) {
        if (m->cfg.is_sliding_layer && l < m->cfg.n_layers && m->cfg.is_sliding_layer[l]) continue;
        if (!s->has_cache || !s->has_cache[l]) continue;
        if (llm_kvdisk_blob_append_tensor(&buf, &len, &cap, s->k_cache[l], l, 0) == 0) nt++;
        if (llm_kvdisk_blob_append_tensor(&buf, &len, &cap, s->v_cache[l], l, 1) == 0) nt++;
    }
    if (nt == 0) { free(buf); return -1; }
    memcpy(buf + 16, &nt, 4);
    int kn = key_n > 0 ? key_n : s->history_len;
    if (kn > pos) kn = pos;
    const int32_t* kt = key_tok != NULL ? key_tok : s->history;
    int rc = llm_kv_disk_put(&m->kvdisk, llm_session_ckpt_key(m, kt, kn), buf, len);
    free(buf);
    if (rc != 0) return -1;
    m->kvckpt_writes++;
    return 0;
}

/* Ripristina il checkpoint piu' lungo <= keep_tokens (multipli del blocco) nella sessione, a
 * partire dai token del prefisso. Ritorna la posizione ripristinata (>0) o -1. Dopo il ripristino
 * cache_offset/history = pos: il chiamante appende il suffisso mancante (re-prefill minimo).
 * NB: i layer local (sliding) non sono checkointati -> restano vuoti; K6: per architetture ibride
 * (is_sliding_layer != NULL) la ricerca del checkpoint arretra automaticamente di una finestra,
 * cosi' il re-prefill del suffisso ricostruisce >= window token dei layer local (vedi sotto). */
int llm_ai_mlx_session_rewind(llm_ai_mlx_session_handle sh, llm_ai_mlx_model_handle mh,
                              const int32_t* tokens, int n_tokens, int keep_tokens) {
    if (!sh || !mh || !tokens || n_tokens <= 0) return -1;
    LlmMlxSession* s = (LlmMlxSession*)sh;
    LlmMlxModel* m = (LlmMlxModel*)mh;
    if (!llm_kvdisk_enabled(m) || m->kvckpt_block <= 0 || m->kvdisk_state != 1) return -1;
    if (keep_tokens > n_tokens) keep_tokens = n_tokens;
    int p = (keep_tokens / m->kvckpt_block) * m->kvckpt_block;
    /* K6 (§G-ter): i layer SLIDING/local NON sono checkpointati (solo i global) -> il re-prefill
     * del suffisso deve ricostruirli, quindi deve partire >= window token PRIMA del taglio. Arretra
     * il punto di ricerca di una finestra (allineata al blocco): il checkpoint trovato sara' a
     * <= keep_tokens-window, e il suffisso re-prefillato copre l'intera finestra dei layer locali.
     * Se non resta spazio per un checkpoint utile -> full prefill (ritorna -1). */
    if (m->cfg.is_sliding_layer != NULL && m->cfg.sliding_window > 0) {
        p -= m->cfg.sliding_window;
        if (p < m->kvckpt_block) return -1;
        p = (p / m->kvckpt_block) * m->kvckpt_block;
    }
    size_t need = 0;
    for (; p > 0; p -= m->kvckpt_block)
        if (llm_kv_disk_has(&m->kvdisk, llm_session_ckpt_key(m, tokens, p), &need) && need > 20) break;
    if (p <= 0) return -1;
    uint8_t* buf = (uint8_t*)malloc(need);
    if (!buf) return -1;
    size_t got = 0;
    if (llm_kv_disk_get(&m->kvdisk, llm_session_ckpt_key(m, tokens, p), buf, need, &got) != 0 || got < 20) { free(buf); return -1; }
    uint32_t magic = 0, ver = 0, nt = 0; int32_t coff = 0;
    memcpy(&magic, buf + 0, 4); memcpy(&ver, buf + 4, 4);
    memcpy(&coff, buf + 8, 4);  memcpy(&nt, buf + 16, 4);
    if (magic != LLM_KVDISK_BLOB_MAGIC || ver != LLM_KVDISK_BLOB_VER) { free(buf); return -1; }
    /* reset della sessione (KV + history), poi restore dei soli layer global */
    for (int l = 0; l < s->n_layers; l++) {
        mlx_array_free(s->k_cache[l]); s->k_cache[l] = mlx_array_new();
        mlx_array_free(s->v_cache[l]); s->v_cache[l] = mlx_array_new();
        s->has_cache[l] = false;
    }
    mlx_array_free(s->pending_h); s->pending_h = mlx_array_new();
    s->has_pending = false; s->pending_next_id = 0;
    size_t off = 20; int n_ok = 0;
    for (uint32_t i = 0; i < nt; i++) {
        mlx_array a = mlx_array_new(); int layer = -1, kind = -1;
        if (llm_kvdisk_blob_read_tensor(buf, got, &off, &a, &layer, &kind) != 0) { mlx_array_free(a); break; }
        if (layer < 0 || layer >= s->n_layers) { mlx_array_free(a); break; }
        mlx_array* dst = (kind == 0) ? &s->k_cache[layer] : &s->v_cache[layer];
        mlx_array_set(dst, a);
        mlx_array_free(a);
        s->has_cache[layer] = true;
        n_ok++;
    }
    free(buf);
    if (n_ok == 0) return -1;
    (void)coff; /* == p (posizione del checkpoint) */
    s->cache_offset = p;
    free(s->history); s->history = NULL; s->history_len = 0; s->history_cap = 0;
    s->history = (int32_t*)malloc((size_t)p * sizeof(int32_t));
    if (s->history) { memcpy(s->history, tokens, (size_t)p * sizeof(int32_t)); s->history_len = p; s->history_cap = p; }
    m->kvckpt_rewinds++;
    return p;
}

/* §G-ter EVICT — PARK/UNPARK della sessione sul tier SSD.
 * park: scrive TUTTA la KV della sessione (tutti i layer) + pending_h/pending_next_id su
 * llm_kv_disk e LIBERA gli array MLX => RAM ∝ sequenze ATTIVE (non a tutte quelle viste).
 * unpark: ricostruisce la KV da disco (restore on-demand, chiamato automaticamente al prossimo
 * append/generate). history/cache_offset restano in RAM (tiny) per poter ricomputare la chiave. */
#define LLM_KVDISK_LAYER_PENDING (-2)   /* layer fittizio: pending_h */
#define LLM_KVDISK_LAYER_META    (-3)   /* layer fittizio: pending_next_id (int32) */
#define LLM_KV_PARK_SALT         0x9E3779B97F4A7C15ULL  /* namespace dei park (no collisione coi ckpt) */

int llm_ai_mlx_session_park(llm_ai_mlx_session_handle sh, llm_ai_mlx_model_handle mh) {
    if (!sh || !mh) return -1;
    LlmMlxSession* s = (LlmMlxSession*)sh;
    LlmMlxModel* m = (LlmMlxModel*)mh;
    if (!llm_kvdisk_enabled(m) || m->kvdisk_state != 1) return -1;
    if (s->parked) return 0;
    if (s->cache_offset <= 0 || !s->history) return -1;
    uint8_t* buf = NULL; size_t len = 0, cap = 0;
    if (llm_kvdisk_buf_reserve(&buf, &cap, len, 20) != 0) return -1;
    uint32_t magic = LLM_KVDISK_BLOB_MAGIC, ver = LLM_KVDISK_BLOB_VER, nt = 0;
    int32_t coff = s->cache_offset, tlen = s->cache_offset;
    memcpy(buf + 0, &magic, 4); memcpy(buf + 4, &ver, 4);
    memcpy(buf + 8, &coff, 4);  memcpy(buf + 12, &tlen, 4);
    memcpy(buf + 16, &nt, 4);
    len = 20;
    for (int l = 0; l < s->n_layers; l++) {
        if (!s->has_cache || !s->has_cache[l]) continue;
        if (llm_kvdisk_blob_append_tensor(&buf, &len, &cap, s->k_cache[l], l, 0) == 0) nt++;
        if (llm_kvdisk_blob_append_tensor(&buf, &len, &cap, s->v_cache[l], l, 1) == 0) nt++;
    }
    if (s->has_pending && mlx_array_ndim(s->pending_h) > 0)
        if (llm_kvdisk_blob_append_tensor(&buf, &len, &cap, s->pending_h, LLM_KVDISK_LAYER_PENDING, 2) == 0) nt++;
    {   int shp[1] = { 1 };
        mlx_array pn = mlx_array_new_data(&s->pending_next_id, shp, 1, MLX_INT32);
        if (mlx_array_ndim(pn) > 0 && llm_kvdisk_blob_append_tensor(&buf, &len, &cap, pn, LLM_KVDISK_LAYER_META, 3) == 0) nt++;
        mlx_array_free(pn);
    }
    if (nt == 0) { free(buf); return -1; }
    memcpy(buf + 16, &nt, 4);
    s->park_key = llm_session_ckpt_key(m, s->history, s->history_len) ^ LLM_KV_PARK_SALT;
    int rc = llm_kv_disk_put(&m->kvdisk, s->park_key, buf, len);
    free(buf);
    if (rc != 0) return -1;
    /* libera la RAM: array KV + pending_h (history/cache_offset restano: tiny) */
    for (int l = 0; l < s->n_layers; l++) {
        mlx_array_free(s->k_cache[l]); s->k_cache[l] = mlx_array_new();
        mlx_array_free(s->v_cache[l]); s->v_cache[l] = mlx_array_new();
        s->has_cache[l] = false;
    }
    mlx_array_free(s->pending_h); s->pending_h = mlx_array_new();
    s->parked = true;
    m->kvpark_writes++;
    return 0;
}

int llm_session_unpark(LlmMlxModel* m, LlmMlxSession* s) {
    if (!m || !s || !s->parked) return -1;
    /* Modello DELEGATO (DS4/DS4.1): unpark non e' nell'ABI a 9 metodi -> non supportato. */
    if (m->deleg) return -1;
    size_t need = 0;
    if (!llm_kv_disk_has(&m->kvdisk, s->park_key, &need) || need < 20) return -1;
    uint8_t* buf = (uint8_t*)malloc(need);
    if (!buf) return -1;
    size_t got = 0;
    if (llm_kv_disk_get(&m->kvdisk, s->park_key, buf, need, &got) != 0 || got < 20) { free(buf); return -1; }
    uint32_t magic = 0, ver = 0, nt = 0; int32_t coff = 0;
    memcpy(&magic, buf + 0, 4); memcpy(&ver, buf + 4, 4);
    memcpy(&coff, buf + 8, 4);  memcpy(&nt, buf + 16, 4);
    if (magic != LLM_KVDISK_BLOB_MAGIC || ver != LLM_KVDISK_BLOB_VER) { free(buf); return -1; }
    for (int l = 0; l < s->n_layers; l++) {
        mlx_array_free(s->k_cache[l]); s->k_cache[l] = mlx_array_new();
        mlx_array_free(s->v_cache[l]); s->v_cache[l] = mlx_array_new();
        s->has_cache[l] = false;
    }
    mlx_array_free(s->pending_h); s->pending_h = mlx_array_new();
    s->has_pending = false; s->pending_next_id = 0;
    size_t off = 20; int n_ok = 0;
    for (uint32_t i = 0; i < nt; i++) {
        mlx_array a = mlx_array_new(); int layer = -1, kind = -1;
        if (llm_kvdisk_blob_read_tensor(buf, got, &off, &a, &layer, &kind) != 0) { mlx_array_free(a); break; }
        if (layer == LLM_KVDISK_LAYER_PENDING) {
            mlx_array_set(&s->pending_h, a); s->has_pending = true;
        } else if (layer == LLM_KVDISK_LAYER_META) {
            const int32_t* d = mlx_array_data_int32(a);
            if (d) s->pending_next_id = d[0];
        } else if (layer >= 0 && layer < s->n_layers) {
            mlx_array* dst = (kind == 0) ? &s->k_cache[layer] : &s->v_cache[layer];
            mlx_array_set(dst, a);
            s->has_cache[layer] = true;
        } else { mlx_array_free(a); break; }
        mlx_array_free(a);
        n_ok++;
    }
    free(buf);
    if (n_ok == 0) return -1;
    s->cache_offset = coff;
    s->parked = false;
    m->kvpark_reads++;
    return 0;
}

/* Eviction LRU dei prefissi salvati: se il numero di prefissi supera m->prefix_max,
 * rimuove quello usato meno di recente (libera cache/tokens/history e compatta l'array).
 * §G: con l'opzione di registro `kvdisk` attiva, la KV del LRU ancora in RAM viene prima SPILLATA su disco
 * (unified memory liberata, tokens/name conservati: il riuso la ripristina da disco).
 * Ritorna true se ha evittato (o spillato) almeno un prefisso. */
static bool llm_prefix_evict_lru(LlmMlxModel* m) {
    if (!m || m->prefix_max <= 0 || m->n_prefixes < m->prefix_max) return false;
    if (llm_kvdisk_enabled(m)) {
        int cand = -1; uint64_t oldest = UINT64_MAX;
        /* C5: spill SELETTIVO — scrivere su SSD un prefisso mai riusato e' I/O sprecato (SGLang HiCache: dal 2o hit).
         * I prefissi sotto soglia, se vittime, vengono evitti senza spill. 0 = spill incondizionato (comportamento storico). */
        int min_hits = 1; (void)llm_ai_mlx_opt_get_int("prefix_spill_min_hits", &min_hits);
        for (int i = 0; i < m->n_prefixes; i++)
            if (m->prefixes[i].has_kv && !m->prefixes[i].on_disk && m->prefixes[i].hits >= min_hits && m->prefixes[i].last_used < oldest) {
                oldest = m->prefixes[i].last_used; cand = i;
            }
        if (cand >= 0) {
            SavedPrefix* p = &m->prefixes[cand];
            if (llm_prefix_spill(m, p) == 0) {
                /* libera SOLO i layer spillati (global): i local sliding restano in RAM */
                for (int l = 0; l < p->n_layers; l++) {
                    if (m->cfg.is_sliding_layer && l < m->cfg.n_layers && m->cfg.is_sliding_layer[l]) continue;
                    mlx_array_free(p->k_cache[l]); p->k_cache[l] = mlx_array_new();
                    mlx_array_free(p->v_cache[l]); p->v_cache[l] = mlx_array_new();
                    p->has_cache[l] = false;
                }
                p->has_kv = false;
                p->on_disk = true;
                p->last_used = ++m->lru_clock; /* ora e' cold: non ri-spillare subito */
                return true;
            }
        }
    }
    int victim = 0;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < m->n_prefixes; i++) {
        if (m->prefixes[i].last_used < oldest) { oldest = m->prefixes[i].last_used; victim = i; }
    }
    SavedPrefix* p = &m->prefixes[victim];
    if (p->on_disk && p->token_len > 0 && m->kvdisk_state == 1)
        llm_kv_disk_del(&m->kvdisk, llm_prefix_kvdisk_key(m, p->tokens, p->token_len));
    for (int l = 0; l < p->n_layers; l++) { mlx_array_free(p->k_cache[l]); mlx_array_free(p->v_cache[l]); }
    free(p->k_cache); free(p->v_cache); free(p->has_cache); free(p->tokens); free(p->history);
    memset(p, 0, sizeof(*p));
    if (victim != m->n_prefixes - 1) {
        memmove(&m->prefixes[victim], &m->prefixes[victim + 1], (size_t)(m->n_prefixes - victim - 1) * sizeof(SavedPrefix));
        memset(&m->prefixes[m->n_prefixes - 1], 0, sizeof(SavedPrefix));
    }
    m->n_prefixes--;
    return true;
}

/* 1 se il modello ha uno stato ricorrente FUORI dalla cache K/V (DeltaNet/GDN, convoluzione corta, SSM, RWKV, KDA). Il
 * prefisso salvato/ripristinato e il rollback n-gram riportano solo K/V: lo stato ricorrente resterebbe quello della fine
 * della sequenza e il testo si corrompe (Qwen3.5 "![]()", LFM2 "<|pad|>"). Per questi modelli niente riuso di prefissi. */
int llm_mlx_model_has_recurrent_state(const LlmMlxModel* m) {
    static const char* const k_rec[] = { "kimi_linear", "lfm2", "lfm2_moe", "mamba", "mamba2", "falcon_mamba", "falcon_h1",
        "jamba", "granitemoehybrid", "olmo_hybrid", "rwkv7", "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_next" };
    if (m == NULL || m->cfg.model_type == NULL) return 0;
    for (size_t i = 0; i < sizeof k_rec / sizeof k_rec[0]; i++)
        if (strcmp(m->cfg.model_type, k_rec[i]) == 0) return 1;
    return 0;
}

int llm_ai_mlx_model_save_prefix(llm_ai_mlx_model_handle mh, const char* name, llm_ai_mlx_session_handle sh) {

    if (!mh || !name || !sh) return -1;

    LlmMlxModel* m = (LlmMlxModel*)mh;

    LlmMlxSession* s = (LlmMlxSession*)sh;

    if (!s->has_pending) return -1; /* niente da salvare senza prefill */
    if (s->history_len <= 0) return -1; /* D15: non sovrascrivere un prefisso valido con uno vuoto */

    /* Modello DELEGATO (DS4/DS4.1): prefix-cache non esposto dall'ABI a 9 metodi -> non supportato. */
    if (m->deleg) return -1;
    if (llm_mlx_model_has_recurrent_state(m)) return -1;   /* stato ricorrente non salvabile col prefisso (vedi sopra) */

    int idx = -1;

    for (int i = 0; i < m->n_prefixes; i++) if (strcmp(m->prefixes[i].name, name) == 0) { idx = i; break; }

    if (idx < 0) {

        if (m->n_prefixes >= m->prefix_cap) {

            int ncap = m->prefix_cap ? m->prefix_cap * 2 : 4;

            SavedPrefix* np = (SavedPrefix*)realloc(m->prefixes, (size_t)ncap * sizeof(SavedPrefix));

            if (!np) return -1;

            m->prefixes = np; m->prefix_cap = ncap;

        }

        if (m->prefix_max > 0 && m->n_prefixes >= m->prefix_max) llm_prefix_evict_lru(m);
        idx = m->n_prefixes++;
        /* slot nuova: la memoria da realloc NON e' azzerata -> il ramo "libera il vecchio contenuto"
         * sotto leggerebbe n_layers/k_cache spazzatura (SIGSEGV a n_layers>0 && k_cache==NULL). */
        memset(&m->prefixes[idx], 0, sizeof(SavedPrefix));

    }

    SavedPrefix* p = &m->prefixes[idx];
    p->last_used = ++m->lru_clock; /* segna l'uso piu' recente */


    /* libera il vecchio contenuto se gia' presente */

    if (p->n_layers > 0) {

        for (int l = 0; l < p->n_layers; l++) { mlx_array_free(p->k_cache[l]); mlx_array_free(p->v_cache[l]); }

        free(p->k_cache); free(p->v_cache); free(p->has_cache); free(p->tokens); free(p->history);

    }

    memset(p, 0, sizeof(*p));
    p->last_used = ++m->lru_clock; /* segna l'uso piu' recente */

    llm_mlx_mode_set(p->name, sizeof(p->name), name);

    p->n_layers = s->n_layers;

    p->tokens = (int32_t*)malloc((size_t)(s->history_len ? s->history_len : 1) * sizeof(int32_t));

    if (p->tokens) { p->token_len = s->history_len; if (s->history_len) memcpy(p->tokens, s->history, (size_t)s->history_len * sizeof(int32_t)); }

    p->k_cache = (mlx_array*)calloc((size_t)s->n_layers, sizeof(mlx_array));

    p->v_cache = (mlx_array*)calloc((size_t)s->n_layers, sizeof(mlx_array));

    p->has_cache = (bool*)calloc((size_t)s->n_layers, sizeof(bool));

    for (int i = 0; i < s->n_layers; i++) {

        p->k_cache[i] = mlx_array_new(); llm_mlx_cache_checkpoint(&p->k_cache[i], s->k_cache[i], m->stream);

        p->v_cache[i] = mlx_array_new(); llm_mlx_cache_checkpoint(&p->v_cache[i], s->v_cache[i], m->stream);

        p->has_cache[i] = s->has_cache[i];

    }

    p->cache_offset = s->cache_offset;

    p->history = (int32_t*)malloc((size_t)(s->history_len ? s->history_len : 1) * sizeof(int32_t));

    if (p->history) { p->history_len = s->history_len; p->history_cap = s->history_len; if (s->history_len) memcpy(p->history, s->history, (size_t)s->history_len * sizeof(int32_t)); }

    p->pending_next_id = s->pending_next_id;
    p->has_kv = true; /* salvato in-process: la KV e' presente e riusabile */
    /* radix-trie: registra i token del prefisso con prefix_id = indice in prefixes */
    if (p->token_len > 0) llm_radix_insert(&m->radix, p->tokens, p->token_len, idx);
    /* paged cache CONTENT-ADDRESSED: registra i blocchi KV del prefisso nel pool.
     * hash CUMULATIVO (chunk_i dipende dal contesto precedente): h_i = FNV(h_{i-1}, token_i).
     * Pagine pinnate = prefissi stabili del contesto lungo (mai evictate). */
    if (p->token_len > 0 && m->kvpool.block_size > 0) {
        const int bs = m->kvpool.block_size;
        uint64_t acc = 0;
        for (int off = 0; off < p->token_len; off += bs) {
            int n = (p->token_len - off) < bs ? (p->token_len - off) : bs;
            /* h_prev = hash del blocco precedente gia' accumulato: rende il blocco dipendente
             * dal contesto, cosi' blocchi con stessi token ma contesto diverso NON collidono */
            acc = llm_kv_block_hash(p->tokens + off, n);
            acc = acc ^ (acc << 1); /* mescola con il blocco precedente (cumulativo) */
            llm_kv_page_alloc_by_hash(&m->kvpool, idx, acc);
        }
    }

    /* D15: PERSISTENZA (opt-in `prefix_cache`) — spilla la KV su kvdisk (se attivo) e salva il file
     * dei prefissi (nome+token): un riavvio li riconosce e (con il tier SSD) la KV si RESTAURA,
     * evitando il re-prefill. Spillando si libera la RAM (on_disk=true → il riuso ripristina). */
    {
        char pp[2048];
        if (llm_prefix_cache_path(m, pp, (int)sizeof pp)[0] != '\0') {
            if (m->kvdisk_state < 0) (void)llm_kvdisk_enabled(m);
            if (m->kvdisk_state == 1 && p->has_kv && llm_prefix_spill(m, p) == 0) {
                for (int l = 0; l < p->n_layers; l++) {
                    if (m->cfg.is_sliding_layer && l < m->cfg.n_layers && m->cfg.is_sliding_layer[l]) continue;
                    if (!p->has_cache || !p->has_cache[l]) continue;
                    mlx_array_free(p->k_cache[l]); p->k_cache[l] = mlx_array_new();
                    mlx_array_free(p->v_cache[l]); p->v_cache[l] = mlx_array_new();
                    p->has_cache[l] = false;
                }
                p->has_kv = false; p->on_disk = true;
            }
            (void)llm_ai_mlx_model_prefix_save((llm_ai_mlx_model_handle)m, pp);
        }
    }

    return 0;

}

/* Append che RIUUSA un prefisso salvato: se token_ids inizia coi token del prefisso,

 * ripristina cache/offset/history e prefill SOLO la coda (risparmio del prefill ripetuto).

 * Se il prompt non inizia col prefisso, fa l'append normale. Ritorna 1 se ha riusato, 0 se

 * append normale, -1 errore.

 */

/* Configura il limite massimo di prefissi salvati (eviction LRU oltre). 0/disattiva = nessun limite. */
int llm_ai_mlx_model_set_prefix_max(llm_ai_mlx_model_handle mh, int n) {
    if (mh == NULL) return -1;
    LlmMlxModel* m = (LlmMlxModel*)mh;
    m->prefix_max = n;
    if (n > 0) while (llm_prefix_evict_lru(m)) {}
    return 0;
}

int llm_ai_mlx_session_append_reuse_prefix(llm_ai_mlx_session_handle sh, llm_ai_mlx_model_handle mh,

                                          const char* name, const int32_t* token_ids, int n_ids) {

    if (!sh || !mh || !name || !token_ids || n_ids <= 0) return -1;

    /* PFlash: su una sessione FREDDA con PFlash attivo il riuso del prefisso va SALTATO (la
     * sequenza compressa e' una "positional fiction": la KV salvata sul prefisso originale
     * avrebbe posizioni sbagliate per i token tenuti). Il chiamante sotto comprimera' (o
     * rifiutera') da solo, con la stessa decisione del percorso non-prefix. */
    {
        LlmPflashConfig pcfg;
        LlmMlxSession* ps = (LlmMlxSession*)sh;
        if (llm_pflash_config_from_env(&pcfg) && ps->cache_offset == 0 && ps->history_len == 0 &&
            llm_pflash_should_attempt(&pcfg, n_ids))
            return llm_ai_mlx_session_append(sh, token_ids, n_ids);
    }

    LlmMlxModel* m = (LlmMlxModel*)mh;

    LlmMlxSession* s = (LlmMlxSession*)sh;

    SavedPrefix* p = NULL;

    for (int i = 0; i < m->n_prefixes; i++) if (strcmp(m->prefixes[i].name, name) == 0) { p = &m->prefixes[i]; break; }

    if (!p) return llm_ai_mlx_session_append(sh, token_ids, n_ids); /* prefisso sconosciuto */

    if (n_ids < p->token_len) return llm_ai_mlx_session_append(sh, token_ids, n_ids); /* troppo corto */

    int match = 1;

    for (int i = 0; i < p->token_len; i++) if (token_ids[i] != p->tokens[i]) { match = 0; break; }

    if (!match) return llm_ai_mlx_session_append(sh, token_ids, n_ids); /* prefisso non combacia */
    p->hits++;   /* C5: contatore dei riusi (politica di spill selettivo) */
    /* KV assente: prefisso caricato da disco (solo keys) oppure SPILLATO nel tier SSD (§G).
     * Nel secondo caso la ripristiniamo dallo scratch e proseguiamo col riuso normale;
     * altrimenti re-prefill dell'intero prompt (fail-open, corretto). */
    if (!p->has_kv) {
        int _rst = (p->on_disk && llm_kvdisk_enabled(m)) ? llm_prefix_restore(m, p) : -99;
        if (llm_ai_mlx_opt_env("FMLX_PREFIX_DEBUG"))
            fprintf(stderr, "[prefix-reuse] name=%s token_len=%d cache_off=%d on_disk=%d restore=%d\n",
                    name, p->token_len, p->cache_offset, (int)p->on_disk, _rst);
        if (_rst == 0) {
            /* KV ripristinata da disco: continua col riuso normale (sotto) */
        } else {
            return llm_ai_mlx_session_append(sh, token_ids, n_ids);
        }
    }

    /* ripristina lo stato del prefisso (alias MLX refcount) e segna l'uso LRU */

    for (int l = 0; l < s->n_layers; l++) {

        /* nei modelli ibridi (LFM2: livelli a convoluzione) la cache K del prefisso non esiste: mlx_array_set su un array mai
         * inizializzato lancia, quindi si azzera la destinazione invece di copiarla */
        if (p->k_cache[l].ctx) mlx_array_set(&s->k_cache[l], p->k_cache[l]);
        else { mlx_array_free(s->k_cache[l]); s->k_cache[l] = mlx_array_new(); }

        if (p->v_cache[l].ctx) mlx_array_set(&s->v_cache[l], p->v_cache[l]);
        else { mlx_array_free(s->v_cache[l]); s->v_cache[l] = mlx_array_new(); }

        s->has_cache[l] = p->has_cache[l];

    }

    s->cache_offset = p->cache_offset;

    free(s->history); s->history = NULL; s->history_len = 0; s->history_cap = 0;


    if (p->history_len) {

        s->history = (int32_t*)malloc((size_t)p->history_len * sizeof(int32_t));

        memcpy(s->history, p->history, (size_t)p->history_len * sizeof(int32_t));

        s->history_len = p->history_len; s->history_cap = p->history_len;

    }

    s->pending_next_id = p->pending_next_id;

    s->has_pending = true;

    /* prefill della sola coda */

    int tail = n_ids - p->token_len;

    if (tail <= 0) return 1;

    return (llm_ai_mlx_session_append(sh, token_ids + p->token_len, tail) == 0) ? 1 : -1;

}



/* Prefix caching AUTOMATICO: cerca tra TUTTI i prefissi salvati quello con il match di prefisso

 * piu' lungo con l'inizio di token_ids; se il match >= min_match, lo riusa e prefill SOLO la coda.

 * A differenza di append_reuse_prefix (match per nome), qui il miglior prefisso e' scelto

 * automaticamente: utile per i chunk RAG, dove domande diverse sullo stesso vault condividono chunk

 * di contesto (KV Cache per il RAG). Ritorna il numero di token riusati se ha riusato, 0 se append

 * normale (nessun match utile), -1 errore.

 */

int llm_ai_mlx_session_append_auto_prefix(llm_ai_mlx_session_handle sh, llm_ai_mlx_model_handle mh,

                                          const int32_t* token_ids, int n_ids, int min_match) {

    if (!sh || !mh || !token_ids || n_ids <= 0) return -1;

    LlmMlxModel* m = (LlmMlxModel*)mh;

    if (min_match < 1) min_match = 8; /* soglia minima di token riusati perche' valga la pena */

    /* Modello DELEGATO (DS4/DS4.1): auto-prefix non esposto dall'ABI a 9 metodi -> non supportato. */
    if (m->deleg) return -1;

    /* Stato ricorrente fuori da K/V: il prefisso riporta solo K/V -> append lineare (prefill completo, corretto). */
    if (llm_mlx_model_has_recurrent_state(m)) return llm_ai_mlx_session_append(sh, token_ids, n_ids);

    /* DFlash2 (drafter esplicito O auto-rilevato): la radix prefix-cache e' INCOMPATIBILE con la
     * cattura-ctx del drafter. `append_reuse_prefix` NON esegue il prefill (riusa la KV salvata),
     * quindi `llm_mlx_df2_preload`/arm non partono -> la cattura non e' armata -> il drafter
     * degenera (loop di token, es. `<think>` x N). Forza l'append LINEARE, che esegue il prefill e
     * arma la cattura. Diagnosi 2026-10-03 (test DFlash2Native: il provider "greedy", auto-detect
     * del sibling, andava in loop; con append lineare = 19s PASS). */
    if (llm_mlx_df2_will_use(m)) return llm_ai_mlx_session_append(sh, token_ids, n_ids);

    /* radix-trie: trova il prefisso salvato piu' lungo (O(prefix_len)) */
    int best_pid = -1;
    int d = llm_radix_longest_match(&m->radix, token_ids, n_ids, &best_pid);
    if (llm_ai_mlx_opt_env("FMLX_PREFIX_DEBUG"))
        fprintf(stderr, "[prefix-auto] n_ids=%d match=%d best_pid=%d n_prefixes=%d\n",
                n_ids, d, best_pid, m->n_prefixes);
    if (d >= min_match && best_pid >= 0 && best_pid < m->n_prefixes)
        return llm_ai_mlx_session_append_reuse_prefix(sh, mh, m->prefixes[best_pid].name, token_ids, n_ids);
    return llm_ai_mlx_session_append(sh, token_ids, n_ids); /* nessun match utile */

}



/* Debug: stampa i prefissi salvati (nome, token_len, last_used, n_layers). Ritorna il count. */
int llm_ai_mlx_model_prefix_debug(llm_ai_mlx_model_handle mh) {
    if (mh == NULL) return -1;
    LlmMlxModel* m = (LlmMlxModel*)mh;
    fprintf(stderr, "prefixes: n=%d max=%d kvdisk=%s spills=%ld restores=%ld ckpt_block=%d ckpt_writes=%ld rewinds=%ld parks=%ld unparks=%ld\n", m->n_prefixes,
            m->prefix_max, m->kvdisk_state == 1 ? "on" : "off", m->kvdisk_spills, m->kvdisk_restores,
            m->kvckpt_block, m->kvckpt_writes, m->kvckpt_rewinds, m->kvpark_writes, m->kvpark_reads);
    for (int i = 0; i < m->n_prefixes; i++) {
        fprintf(stderr, "  [%d] name=%s tokens=%d last_used=%llu layers=%d has_kv=%d on_disk=%d\n", i,
                m->prefixes[i].name, m->prefixes[i].token_len,
                (unsigned long long)m->prefixes[i].last_used, m->prefixes[i].n_layers,
                (int)m->prefixes[i].has_kv, (int)m->prefixes[i].on_disk);
    }
    return m->n_prefixes;
}

/* Registra un prefisso CARICATO DA DISCO (solo keys, niente KV): alloca la SavedPrefix
 * con k/v_cache vuoti (has_kv=false) e lo inserisce nel radix-trie. Il riuso
 * (append_reuse_prefix) fara' il re-prefill del prefisso. Ritorna l'indice o -1. */
static int llm_prefix_register_persisted(LlmMlxModel* m, const char* name, const int32_t* tokens, int token_len) {
    if (!m || !name || !tokens || token_len <= 0) return -1;
    if (m->n_prefixes >= m->prefix_cap) {
        int ncap = m->prefix_cap ? m->prefix_cap * 2 : 4;
        SavedPrefix* np = (SavedPrefix*)realloc(m->prefixes, (size_t)ncap * sizeof(SavedPrefix));
        if (!np) return -1;
        m->prefixes = np; m->prefix_cap = ncap;
    }
    if (m->prefix_max > 0 && m->n_prefixes >= m->prefix_max) llm_prefix_evict_lru(m);
    int idx = m->n_prefixes++;
    SavedPrefix* p = &m->prefixes[idx];
    memset(p, 0, sizeof(*p));
    p->last_used = ++m->lru_clock;
    llm_mlx_mode_set(p->name, sizeof(p->name), name);
    p->n_layers = m->cfg.n_layers;
    p->tokens = (int32_t*)malloc((size_t)token_len * sizeof(int32_t));
    if (p->tokens) { memcpy(p->tokens, tokens, (size_t)token_len * sizeof(int32_t)); p->token_len = token_len; }
    /* D15: la history del prefisso CARICATO coincide coi suoi token. Serve al riuso: senza di essa
     * la sessione riusata resterebbe a history_len=0 e un successivo save_prefix sovrascriverebbe
     * il prefisso con 0 token (persistenza degradata dopo il primo riavvio). */
    p->history = (int32_t*)malloc((size_t)token_len * sizeof(int32_t));
    if (p->history) { memcpy(p->history, tokens, (size_t)token_len * sizeof(int32_t)); p->history_len = token_len; p->history_cap = token_len; }
    p->k_cache = (mlx_array*)calloc((size_t)p->n_layers, sizeof(mlx_array));
    p->v_cache = (mlx_array*)calloc((size_t)p->n_layers, sizeof(mlx_array));
    p->has_cache = (bool*)calloc((size_t)p->n_layers, sizeof(bool));
    for (int l = 0; l < p->n_layers; l++) { p->k_cache[l] = mlx_array_new(); p->v_cache[l] = mlx_array_new(); }
    /* D15: se il tier SSD e' attivo la KV spillata al save e' su disco -> il riuso la RESTAURA
     * (on_disk=true, niente re-prefill); altrimenti keys-only -> re-prefill (has_kv=false). */
    p->has_kv = false;
    p->on_disk = llm_kvdisk_enabled(m);
    if (p->token_len > 0) llm_radix_insert(&m->radix, p->tokens, p->token_len, idx);
    return idx;
}

/* Formato file prefix cache (binario, atomico via .tmp + rename — pattern rapidMLX):
 *   magic "LLMPFX1" (8) | u32 version=1 | u32 count
 *   per prefisso: u32 name_len | name | u32 token_len | int32 tokens[] */
#define LLM_PREFIX_FILE_MAGIC "LLMPFX1"
#define LLM_PREFIX_FILE_VERSION 1u

/* Salva i KEYS dei prefissi salvati (nome + token) su disco. La KV resta in memoria
 * (non serializzabile da MLX array); al load i prefissi sono riconosciuti ma richiedono
 * re-prefill. Write atomica: path.tmp -> rename. Ritorna 0 ok, -1 errore. */
int llm_ai_mlx_model_prefix_save(llm_ai_mlx_model_handle mh, const char* path) {
    if (mh == NULL || path == NULL) return -1;
    LlmMlxModel* m = (LlmMlxModel*)mh;
    if (m->n_prefixes == 0) return 0;

    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE* f = fopen(tmp, "wb");
    if (!f) return -1;
    bool err = false;
    if (fwrite(LLM_PREFIX_FILE_MAGIC, 1, 8, f) != 8) err = true;
    uint32_t v = LLM_PREFIX_FILE_VERSION, cnt = (uint32_t)m->n_prefixes;
    if (!err && fwrite(&v, sizeof(v), 1, f) != 1) err = true;
    if (!err && fwrite(&cnt, sizeof(cnt), 1, f) != 1) err = true;
    for (int i = 0; !err && i < m->n_prefixes; i++) {
        SavedPrefix* p = &m->prefixes[i];
        uint32_t nl = (uint32_t)strlen(p->name) + 1;
        uint32_t tl = (uint32_t)p->token_len;
        if (fwrite(&nl, sizeof(nl), 1, f) != 1) { err = true; break; }
        if (fwrite(p->name, 1, nl, f) != nl) { err = true; break; }
        if (fwrite(&tl, sizeof(tl), 1, f) != 1) { err = true; break; }
        if (tl && fwrite(p->tokens, sizeof(int32_t), tl, f) != tl) { err = true; break; }
    }
    if (!err) fflush(f);
    fclose(f);
    if (err) { remove(tmp); return -1; }
    if (rename(tmp, path) != 0) { remove(tmp); return -1; }
    return 0;
}

/* Carica i keys dei prefissi da disco e li registra (has_kv=false, trie ricostruito).
 * File mancante/corrotto -> ritorna -1 senza errori fatali (fallback: nessun prefisso
 * persistito, il runtime continua). Ritorna il numero di prefissi caricati. */
int llm_ai_mlx_model_prefix_load(llm_ai_mlx_model_handle mh, const char* path) {
    if (mh == NULL || path == NULL) return -1;
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    char magic[8];
    uint32_t v = 0, cnt = 0;
    bool err = false;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, LLM_PREFIX_FILE_MAGIC, 8) != 0) err = true;
    if (!err && fread(&v, sizeof(v), 1, f) != 1) err = true;
    if (!err && fread(&cnt, sizeof(cnt), 1, f) != 1) err = true;
    if (!err && (v != LLM_PREFIX_FILE_VERSION || cnt > 10000)) err = true;
    int loaded = 0;
    for (uint32_t i = 0; !err && i < cnt; i++) {
        uint32_t nl = 0, tl = 0;
        if (fread(&nl, sizeof(nl), 1, f) != 1 || nl == 0 || nl > 64) { err = true; break; }
        char name[65];
        if (fread(name, 1, nl, f) != nl) { err = true; break; }
        name[nl - 1] = '\0';
        if (fread(&tl, sizeof(tl), 1, f) != 1 || tl > 1000000) { err = true; break; }
        int32_t* toks = tl ? (int32_t*)malloc((size_t)tl * sizeof(int32_t)) : NULL;
        if (tl && fread(toks, sizeof(int32_t), tl, f) != tl) { free(toks); err = true; break; }
        if (llm_prefix_register_persisted((LlmMlxModel*)mh, name, toks, (int)tl) >= 0) loaded++;
        free(toks);
    }
    fclose(f);
    return err ? (loaded > 0 ? loaded : -1) : loaded;
}

int llm_ai_mlx_session_last_hidden(llm_ai_mlx_session_handle handle, float* out, int cap) {
    if (handle == NULL || out == NULL || cap <= 0) return -1;
    LlmMlxSession* s = (LlmMlxSession*)handle;
    if (!s->has_pending) return -1;
    int hidden = s->model->cfg.hidden_size;
    if (cap < hidden) return -hidden;

    mlx_array h32 = mlx_array_new();
    bool ok = mlx_astype(&h32, s->pending_h, MLX_FLOAT32, s->model->stream) == 0;
    if (ok) {
        const float* data = mlx_array_data_float32(h32);
        int n = (int)mlx_array_size(h32);
        if (data == NULL || n < hidden) ok = false;
        else memcpy(out, data, (size_t)hidden * sizeof(float));
    }
    mlx_array_free(h32);
    return ok ? hidden : -1;
}
