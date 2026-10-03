/* llm_mlx_ops.h — primitivi MLX COMPONIBILI e riusabili (refactor modulare, 2026-09-01).
 *
 * Obiettivo (direttiva 🧩): estrarre i mattoni comuni dell'attention/decoder in funzioni
 * riusabili, cosi' un'architettura nuova (Flash-Next/qwen4_exp, qwen3_5_moe, gpt_oss) e' la
 * composizione di questi moduli, non un blocco monolitico nuovo.
 *
 * Convenzione: ogni funzione e' `static inline` o `static`, pura (nessuno stato globale),
 * e prende i parametri espliciti. I blocchi esistenti in llmengine_mlx.c possono migrare
 * uno alla volta; i blocchi NUOVI le usano direttamente.
 *
 * Le funzioni assumono gli stessi contratti del motore: array mlx (lazy), stream esplicito,
 * B=1, layout [B, n_heads, L, head_dim] per q/k/v nell'SDPA.
 */
#ifndef LLM_MLX_OPS_H
#define LLM_MLX_OPS_H

#include <math.h>    /* sqrtf */
#include "llm_engine_opts.h"   /* opt_env: registro */
#include <string.h>  /* strcmp */
#include <stdlib.h>  /* malloc/free */
#include "mlx/c/mlx.h"
#include "llm_mlx_decode_audit.h"  /* P0-ter: contatori strutturali del decode (LLM_DECODE_AUDIT) */

/* ─────────────────────────────────────────────────────────────────────────────
 * SDPA causale GQA (Generalized Multi-Query Attention) con window opzionale.
 *
 * Unifica gli ~8 call-site identici del motore:
 *   scale = 1/sqrt(head_dim); mask_mode="causal"; no_mask; no_sinks
 *
 * window_size > 0 -> sliding window: K/V oltre la finestra viene mascherato via
 * mask_mode="causal" + un mask esplicito (le MLX fast SDPA supportano solo causal
 * o none; la window la costruiamo noi azzerando i logit fuori finestra). Questo
 * serve per layer sliding (DFlash2 draft, gemma4 sliding, future architetture)
 * e per il long-context (attenzione locale = costo O(N·window) non O(N²)).
 *
 * Input (layout già trasposti [B, n_heads, L, head_dim]):
 *   q, k, v : query/key/value
 *   scale   : scala attenzione (di norma 1/sqrt(head_dim); gemma4 usa 1.0 fisso)
 *   window  : 0 = attenzione piena; >0 = sliding window (solo ultimi `window` K)
 *   stream  : mlx stream
 * Output: *out = SDPA(q,k,v) con la finestra applicata, [B,n_heads,L,head_dim].
 * Ritorna true se ok.
 */
/* SDPA causale/sliding con attention sinks OPZIONALI (GPT-OSS): l'array `sinks`
 * (per-head, [n_heads], appreso) NON è un no-op: MLX prepende una posizione "sink"
 * con logit per-head, fa softmax e la slice via (fast.cpp:765-783). sinks/mask
 * vengono castati al dtype di q (bf16 se attivazioni bf16, es. GPT-OSS BF16 nativo:
 * MLX richiede che mask/sinks promuovano al dtype dell'attenzione). */
/* ── LLM_KV_DEQUANT_READ: hook (function-pointer) per l'attenzione decode che legge la KV
 * cold QUANTIZZATA direttamente (kernel Metal dequant-on-read), evitando dequant+concat.
 * llm_mlx_ops.h NON dipende dal kernel: il .c del host installa fn/ctx prima di attn e li
 * azzera dopo. Se fn e' NULL (default) o non applica (prefill Lq>1, window>0, sinks), si
 * prosegue col path normale. */
typedef bool (*llm_kvq_sdpa_hook_fn)(mlx_array* out, mlx_array q, mlx_array k, mlx_array v,
                                     float scale, void* ctx, mlx_stream stream);
/* 2026-09-29: gli hook erano `static` in questo header -> UNA COPIA PER TU; dopo lo split
 * del monolite il .c che li installa (layer_forward) e i .c delle architetture attention
 * (llama/qwen/...) sono TU DIVERSE, quindi l'hook installato non era visibile all'arco ->
 * mai invocato (KV-quant Splash inerte). Ora sono simboli GLOBALI unici (definiti in
 * llm_mlx_layer_forward.c). */
typedef struct { llm_kvq_sdpa_hook_fn fn; void* ctx; } llm_sdpa_hook_slot;
extern llm_sdpa_hook_slot llm_kvq_hook;

/* ── §G-bis leva 1: hook (function-pointer) per l'attenzione decode SPARSA a livello
 * pagina. Stesso meccanismo del kvq hook: il .c installa fn/ctx prima di attn e li azzera
 * dopo; il dispatch calcola gli score query-vs-summary-di-pagina, seleziona le top-k pagine
 * e chiama llm_mlx_pagesel_decode_sdpa (gather + SDPA). Se fn e' NULL (default) o il dispatch
 * ritorna false, si prosegue col path normale (parita'). */
typedef bool (*llm_pagesel_sdpa_hook_fn)(mlx_array* out, mlx_array q, mlx_array k, mlx_array v,
                                         float scale, void* ctx, mlx_stream stream);
extern llm_sdpa_hook_slot llm_pagesel_hook;

/* §D R3.3: hook per SDPA EAGER (matmul+softmax+matmul, differenziabile) — l'autograd MLX non
 * ha il VJP per il CustomKernel di `fast_scaled_dot_product_attention`. Il training forward lo
 * installa al posto della fast_sdpa. Stesso meccanismo degli altri hook. */
typedef bool (*llm_sdpa_hook_fn)(mlx_array* out, mlx_array q, mlx_array k, mlx_array v,
                                 float scale, void* ctx, mlx_stream stream);
extern llm_sdpa_hook_slot llm_train_sdpa_hook;

/* MASCHERA ADDITIVA "pad" per il decode MLA (DeepSeek/Kimi) — 2026-09-16.
 * Nel forward MLA la maschera causale a decode veniva ricostruita OGNI LAYER OGNI TOKEN con
 * arange(0..S) + greater_equal + where(0,-inf) + astype + reshape (~10 op MLX con una O(S)):
 * con 27-64 layer = centinaia di op identiche per token. La maschera dipende SOLO da
 * (S, offset, dtype) -> identica per tutti i layer MLA del forward. Qui si costruisce in C coi
 * BIT PATTERN del dtype (0 ammesso, -inf escluso: f16 0x0000/0xFC00, bf16 0x0000/0xFF80,
 * f32 0x00000000/0xFF800000) e si cacha su (Nk, valid, dtype): 1 costruzione per token invece di
 * n_layers. LLM_MLA_MASKCACHE=0 -> percorso storico (op MLX). */
static bool llm_mla_maskcache_enabled(void) {
    static int st = -1;
    if (st < 0) { const char* e = llm_ai_mlx_opt_env("LLM_MLA_MASKCACHE"); st = (e != NULL && strcmp(e, "0") == 0) ? 0 : 1; }
    return st == 1;
}
static mlx_array llm_mlx_pad_mask_cached(int Nk, int valid, mlx_dtype dt, mlx_stream stream) {
    static int c_nk = -1, c_valid = -1, c_dt = -1;
    static mlx_array c_arr = {0};
    mlx_array out = mlx_array_new();
    if (Nk <= 0 || valid < 0) return out;
    if (dt != MLX_FLOAT16 && dt != MLX_BFLOAT16 && dt != MLX_FLOAT32) return out;
    if (c_nk == Nk && c_valid == valid && c_dt == (int)dt && c_arr.ctx != NULL) {
        mlx_array_set(&out, c_arr);
        return out;
    }
    size_t esz = (dt == MLX_FLOAT32) ? 4u : 2u;
    uint32_t* buf = (uint32_t*)malloc((size_t)Nk * esz);
    if (buf == NULL) return out;
    if (dt == MLX_FLOAT32) {
        for (int i = 0; i < Nk; i++) buf[i] = (i < valid) ? 0x00000000u : 0xFF800000u;
    } else {
        uint16_t* b16 = (uint16_t*)buf;
        uint16_t ninf = (dt == MLX_BFLOAT16) ? 0xFF80u : 0xFC00u;
        for (int i = 0; i < Nk; i++) b16[i] = (i < valid) ? 0x0000u : ninf;
    }
    const int sh[4] = {1, 1, 1, Nk};
    mlx_array arr = mlx_array_new_data(buf, sh, 4, dt);
    free(buf);
    if (arr.ctx == NULL) return out;
    if (c_arr.ctx != NULL) mlx_array_free(c_arr);
    c_arr = mlx_array_new();
    mlx_array_set(&c_arr, arr);
    mlx_array_free(arr);
    c_nk = Nk; c_valid = valid; c_dt = (int)dt;
    mlx_array_set(&out, c_arr);
    return out;
}

static bool llm_mlx_gqa_sdpa_sinks(mlx_array* out, mlx_array q, mlx_array k, mlx_array v,
                                   float scale, int window, mlx_array sinks, mlx_stream stream)
{
    if (llm_kvq_hook.fn != NULL && llm_kvq_hook.ctx != NULL && window <= 0 && sinks.ctx == NULL) {
        if (llm_kvq_hook.fn(out, q, k, v, scale, llm_kvq_hook.ctx, stream)) return true;
    }
    if (llm_pagesel_hook.fn != NULL && llm_pagesel_hook.ctx != NULL && window <= 0 && sinks.ctx == NULL) {
        if (llm_pagesel_hook.fn(out, q, k, v, scale, llm_pagesel_hook.ctx, stream)) return true;
    }
    if (llm_train_sdpa_hook.fn != NULL && window <= 0 && sinks.ctx == NULL) {
        if (llm_train_sdpa_hook.fn(out, q, k, v, scale, llm_train_sdpa_hook.ctx, stream)) return true;
    }
    mlx_array no_mask = mlx_array_new();
    /* sinks e mask devono avere un dtype che promuove a quello dell'attenzione (bf16 se
     * le attivazioni sono bf16, es. GPT-OSS BF16 nativo). Cast al dtype di q.
     * sinks.ctx==NULL = nessun sink (wrapper llm_mlx_gqa_sdpa) -> passare vuoto. */
    mlx_array sinks_c = mlx_array_new();
    if (sinks.ctx != NULL) {
        /* Niente AsType quando il dtype coincide gia' (gpt-oss bf16: era un nodo AsType per
         * layer per token, inutile). Cast solo se serve davvero. */
        bool sok = (mlx_array_dtype(sinks) == mlx_array_dtype(q))
                       ? (mlx_array_set(&sinks_c, sinks) == 0)
                       : (mlx_astype(&sinks_c, sinks, mlx_array_dtype(q), stream) == 0);
        if (!sok) { mlx_array_free(sinks_c); mlx_array_free(no_mask); return false; }
    }
    if (window <= 0) {
        /* attenzione causale piena (+ sinks): MLX calcola offset=kL-qL internamente */
        /* LLM_KVF16 (#61): cache KV in fp16 con attivazione fp32 (Llama/Bonsai) -> fast_sdpa
         * richiede dtype omogenei: q va in fp16 e il risultato torna al dtype originale.
         * Innescato SOLO se k/v sono fp16 (default-OFF: mai nel percorso di produzione). */
        mlx_dtype qdt0 = mlx_array_dtype(q);
        bool q_harmony = (qdt0 != MLX_FLOAT16 && mlx_array_dtype(k) == MLX_FLOAT16
                          && mlx_array_dtype(v) == MLX_FLOAT16);
        mlx_array qf = mlx_array_new();
        if (q_harmony && mlx_astype(&qf, q, MLX_FLOAT16, stream) != 0) {
            mlx_array_free(qf); mlx_array_free(sinks_c); mlx_array_free(no_mask); return false;
        }
        if (q_harmony && sinks.ctx != NULL) {
            /* fast_sdpa richiede che i sinks PROMUOVANO al dtype dell'output (fp16 in
             * harmony): senza questo cast gpt-oss (sinks bf16/fp32 + cache fp16) fallisce
             * con "Type of sinks must promote to output type float16". */
            mlx_array sk16 = mlx_array_new();
            if (mlx_astype(&sk16, sinks_c, MLX_FLOAT16, stream) == 0) {
                mlx_array_free(sinks_c); sinks_c = sk16;
            } else { mlx_array_free(sk16); }
        }
        /* Decode (Lq==1): mlx_lm (`create_attention_mask`, cache.py:114) ritorna None ->
         * mask_mode "" (kernel flash-decode `sdpa_vector`, bandwidth-optimal). La maschera
         * causale a Lq==1 e' comunque inutile (l'unica query vede tutto il passato), ma
         * forzava un kernel piu' lento. Prefill (Lq>1): "causal" come mlx_lm. */
        const int lq = (mlx_array_ndim(q) >= 3) ? (int)mlx_array_shape(q)[2] : 0;
        static int sdpaca_state = -1;
        if (sdpaca_state < 0) sdpaca_state = (llm_ai_mlx_opt_env("LLM_SDPA_DECODE_CAUSAL") != NULL) ? 1 : 0;
        const char* mm = (lq == 1 && !sdpaca_state) ? "" : "causal";
        bool ok = mlx_fast_scaled_dot_product_attention(out, q_harmony ? qf : q, k, v, scale, mm,
                                                         no_mask, sinks_c, stream) == 0;
        if (ok && q_harmony) {
            mlx_array back = mlx_array_new();
            ok = mlx_astype(&back, *out, qdt0, stream) == 0;
            /* mlx_array_set rilascia gia' internamente il vecchio *out (contratto, vedi NOTA
             * nel blocco llama): NON fare mlx_array_free(*out) qui, che sarebbe un doppio
             * rilascio sull'array che il grafo lazy di back ancora referenzia -> SIGSEGV. */
            if (ok) mlx_array_set(out, back);
            mlx_array_free(back);
        }
        mlx_array_free(qf);
        mlx_array_free(sinks_c);
        mlx_array_free(no_mask);
        return ok;
    }
    /* Sliding window: maschera ADDITIVA pre-softmax (causale + finestra) passata via
     * mask_arr con mask_mode="" -- MLX la somma ai logit QK^T PRIMA del softmax
     * (mlx/fast.cpp:762 `scores = add(scores, mask)`). La PRECEDENTE implementazione
     * (SDPA causale piena + aggiunta di -inf all'OUTPUT, post-softmax) era SBAGLIATA:
     * il softmax aveva gia' mescolato tutti i token, quindi i token fuori finestra NON
     * venivano rimossi e l'output conteneva -inf (poi NaN).
     *
     * NOTA shape (bug trovato dal vivo 2026-09-01): la mask deve essere [1,1,Lq,Nk]
     * con Nk = lunghezza della KV-cache (asse -2 di k), NON [1,1,Lq,Lq]. Con la cache
     * (decode Lq=1, Nk>=1; chunked prefill Nk=Lq+cache) una mask [1,1,Lq,Lq] broadcasta
     * i SOLI Lq valori su tutti gli Nk key -> a decode non applicava NE' il window NE'
     * la causal (mask_mode="") -> il modello vedeva TUTTO il passato (needle leak).
     * Posizione assoluta della query i = (Nk - Lq + i) (la cache copre 0..Nk-1).
     * Keep (i,j) iff j <= qabs AND j >= qabs-(window-1) -- identico a mlx_lm
     * create_causal_mask(offset): linds>=rinds AND linds<rinds+window.
     *
     * CACHE DELLA MASK (2026-09-07, commit 9b75886d; GATE DEFAULT OFF dallo stesso giorno):
     * a decode (Lq=1) con cache trimmata alla finestra (fix 392290e9), Nk e' FISSO (=window)
     * per ogni step -> la mask e' IDENTICA a ogni token e si puo' ricostruire SOLO se cambia
     * (Lq, Nk, window, dtype) invece di malloc+fill+new_data per ogni layer sliding per ogni
     * token. Verificata correttezza (parita' greedy IDENTICA su gemma-4-31b) ma NESSUN guadagno
     * di velocita' misurato (il costo della mask e' nel rumore) -> resta DIETRO FLAG
     * LLM_MASK_CACHE=1, default OFF: senza il flag il comportamento e' identico al pre-fix. */
    int ndim = (int)mlx_array_ndim(q);
    int Lq = ndim > 2 ? (int)mlx_array_shape(q)[ndim - 2] : 1;
    int Nk = (int)mlx_array_shape(k)[ndim - 2];
    if (Nk < Lq) Nk = Lq; /* safety */
    /* LLM_MASK_CACHE: non impostato = AUTO (cache con cap di memoria); =0 = kill-switch
     * (sempre ricostruita, comportamento pre-P0-bis); =1 = forza la cache a qualsiasi size.
     * LLM_MASK_CACHE_MB = cap (default 256 MB). Letto UNA volta (mai getenv nel percorso caldo).
     *
     * P0-bis (2026-09-16): la mask sliding e' IDENTICA per tutti i layer sliding della stessa
     * geometria (Lq,Nk,window,dtype) DENTRO LO STESSO FORWARD. A decode e' [1,1,1,Nk] (minuscola,
     * identica a ogni token con la cache trimmata alla finestra: gemma-4 ha 50 layer sliding su
     * 60 = 50 costruzioni + 50 astype per token); a prefill e' [1,1,Lq,Nk] e veniva ricostruita
     * una volta per layer (a Lq=4096: 67 MB x 12 layer ≈ 0,8 GB di scritture -- il +4,5% misurato
     * su gpt-oss). Il CAP tiene fuori i contesti enormi, dove la mask sarebbe GB. */
    static int s_maskcache_state = -2; /* -2 = non letto; 1 = forza ON; 0 = OFF; -1 = auto */
    static long s_maskcache_cap = -1;
    if (s_maskcache_state == -2) {
        const char* e = llm_ai_mlx_opt_env("LLM_MASK_CACHE");
        s_maskcache_state = (e == NULL) ? -1 : (strcmp(e, "0") == 0 ? 0 : 1);
        const char* c = llm_ai_mlx_opt_env("LLM_MASK_CACHE_MB");
        s_maskcache_cap = (c != NULL && atol(c) > 0) ? atol(c) : 256;
    }
    const long mask_bytes = (long)Lq * (long)Nk * 4L;
    const bool use_mask_cache = (s_maskcache_state == 1)
        || (s_maskcache_state == -1 && mask_bytes <= s_maskcache_cap * 1024L * 1024L);
    if (use_mask_cache) {
        static mlx_array s_mask_cache;      /* cache del modulo (motore single-thread
                                              * per sessione, come tutto il resto del file) */
        static int s_mask_Lq = -1, s_mask_Nk = -1, s_mask_win = -1;
        static mlx_dtype s_mask_dt = (mlx_dtype)0;
        static bool s_mask_init = false;
        if (!s_mask_init) { s_mask_cache = mlx_array_new(); s_mask_init = true; }
        bool need_rebuild = (s_mask_cache.ctx == NULL) || Lq != s_mask_Lq || Nk != s_mask_Nk
                            || window != s_mask_win || s_mask_dt != mlx_array_dtype(q);
        if (need_rebuild) {
            int mask_shape[4] = {1, 1, Lq, Nk};
            float* md = (float*)malloc((size_t)Lq * (size_t)Nk * sizeof(float));
            if (md == NULL) { mlx_array_free(sinks_c); mlx_array_free(no_mask); return false; }
            for (int i = 0; i < Lq; i++) {
                int qabs = (Nk - Lq) + i;
                for (int j = 0; j < Nk; j++) {
                    bool causal_ok = j <= qabs;
                    bool in_window = j >= qabs - (window - 1);
                    md[i * Nk + j] = (causal_ok && in_window) ? 0.0f : -INFINITY;
                }
            }
            mlx_array rebuilt = mlx_array_new_data(md, mask_shape, 4, MLX_FLOAT32);
            free(md);
            if (rebuilt.ctx == NULL) { mlx_array_free(sinks_c); mlx_array_free(no_mask); return false; }
            mlx_array_free(s_mask_cache);
            s_mask_cache = rebuilt;
            s_mask_Lq = Lq; s_mask_Nk = Nk; s_mask_win = window; s_mask_dt = mlx_array_dtype(q);
        }
        mlx_array mask = mlx_array_new();
        mlx_array_set(&mask, s_mask_cache);
        if (mlx_array_dtype(q) != MLX_FLOAT32) {
            mlx_array mb = mlx_array_new();
            bool castok = mlx_astype(&mb, mask, mlx_array_dtype(q), stream) == 0;
            mlx_array_free(mask);
            if (!castok) { mlx_array_free(sinks_c); mlx_array_free(no_mask); return false; }
            mask = mb;
        }
        bool ok_cached = mlx_fast_scaled_dot_product_attention(out, q, k, v, scale, "", mask, sinks_c, stream) == 0;
        mlx_array_free(mask);          /* NON liberare s_mask_cache: vive per il prossimo step */
        mlx_array_free(sinks_c);
        mlx_array_free(no_mask);
        return ok_cached;
    }
    int mask_shape[4] = {1, 1, Lq, Nk};
    float* md = (float*)malloc((size_t)Lq * (size_t)Nk * sizeof(float));
    if (md == NULL) { mlx_array_free(sinks_c); mlx_array_free(no_mask); return false; }
    for (int i = 0; i < Lq; i++) {
        int qabs = (Nk - Lq) + i;
        for (int j = 0; j < Nk; j++) {
            bool causal_ok = j <= qabs;
            bool in_window = j >= qabs - (window - 1);
            md[i * Nk + j] = (causal_ok && in_window) ? 0.0f : -INFINITY;
        }
    }
    mlx_array mask = mlx_array_new_data(md, mask_shape, 4, MLX_FLOAT32);
    free(md);
    /* MLX richiede che la mask promuova al dtype dell'attenzione. q può essere
     * bf16/f16 quando il modello è promosso (gemma4 = sliding-window + bf16:
     * è il primo checkpoint a combinare finestra E flavor 16-bit, e senza
     * questo cast SDPA aborta "Mask type must promote to output type bfloat16").
     * Il percorso cached (LLM_MASK_CACHE) lo fa già sopra: qui è lo specchio per
     * il default. Costruiamo la mask in fp32 per il fill -INF, poi astype a q. */
    if (mlx_array_dtype(q) != MLX_FLOAT32) {
        mlx_array mb = mlx_array_new();
        bool castok = mlx_astype(&mb, mask, mlx_array_dtype(q), stream) == 0;
        mlx_array_free(mask);
        if (!castok) { mlx_array_free(sinks_c); mlx_array_free(no_mask); return false; }
        mask = mb;
    }
    bool ok = mlx_fast_scaled_dot_product_attention(out, q, k, v, scale, "", mask, sinks_c, stream) == 0;
    mlx_array_free(mask);
    mlx_array_free(sinks_c);
    mlx_array_free(no_mask);
    return ok;
}

/* Append KV IN-PLACE (2026-09-13): replica `keys[..., a:b, :] = x` di mlx_lm via
 * mlx_slice_update_inplace -> append O(1), niente ricopia O(cap) del concatenate.
 * Riusabile da tutti i blocchi attention densi. La cache deve essere [1,nkv,S,hd];
 * se la capacita' S >= off+n_tokens scrive in-place, altrimenti migra a un buffer
 * arrotondato a `step`. Ritorna in *k_view/*v_view la vista troncata [0..off+n) da
 * passare alla SDPA (il buffer puo' avere padding). false = non applicabile. */
static int g_kv_inplace_state = -1;
static bool llm_mlx_kv_inplace_enabled(void) {
    /* LLM_KV_DEQUANT_READ: cache = soli recenti (cold nel pending) -> no append in-place. */
    if (llm_kvq_hook.fn != NULL) return false;
    if (g_kv_inplace_state < 0) {
        const char* e = llm_ai_mlx_opt_env("LLM_KV_INPLACE");
        g_kv_inplace_state = (e != NULL && strcmp(e, "0") == 0) ? 0 : 1;
    }
    return g_kv_inplace_state == 1;
}
static bool llm_mlx_kv_append_inplace(mlx_array* k_cache, mlx_array* v_cache,
                                      mlx_array k_new, mlx_array v_new,
                                      int off, int n_tokens, int step,
                                      mlx_array* k_view, mlx_array* v_view,
                                      mlx_stream stream) {
    if (k_cache == NULL || v_cache == NULL || k_view == NULL || v_view == NULL) return false;
    if (off < 0 || n_tokens <= 0) return false;
    if ((*k_cache).ctx == NULL || (*v_cache).ctx == NULL) return false;
    if (mlx_array_ndim(*k_cache) != 4 || mlx_array_ndim(*v_cache) != 4) return false;
    const int* ksh = mlx_array_shape(*k_cache);
    if (ksh == NULL) return false;
    const int nkv = ksh[1], cap = ksh[2], kd = ksh[3];
    const int vd = (int)mlx_array_shape(*v_cache)[3];
    const int need = off + n_tokens;
    const int esz = llm_da_esz(mlx_array_dtype(*k_cache));
    /* FIX W5 (2026-10-03): con la KV-quant int8 attiva la cache FP16 e' SOLO la finestra recente,
     * mentre `off` e' la posizione LOGICA (include il cold nel pool) -> la cache NON contiene
     * `off` token. Il grow path copierebbe `off` token da una cache piu' corta -> slice_update con
     * update/slice incoerenti (broadcast error a >16k). Se `off > cap` la cache non rappresenta
     * `off` token -> NON e' il caso in-place (fallback al path normale). */
    if (off > cap) return false;
    if (cap >= need) {
        int s0[4] = {0,0,off,0}, st[4] = {1,1,1,1};
        int sk[4] = {1,nkv,need,kd}, sv[4] = {1,nkv,need,vd};
        if (mlx_slice_update_inplace(k_cache, k_new, s0,4, sk,4, st,4, stream) != 0) {
            if (llm_da_enabled()) g_da.kv_fail++;
            return false;
        }
        if (mlx_slice_update_inplace(v_cache, v_new, s0,4, sv,4, st,4, stream) != 0) {
            if (llm_da_enabled()) g_da.kv_fail++;
            return false;
        }
        llm_da_note_kv_append(nkv, n_tokens, kd, vd, esz, /*grew=*/0, off);
    } else {
        if (step <= 0) step = 256;
        const int newcap = ((need + step - 1) / step) * step;
        int zk[4] = {1,nkv,newcap,kd}, zv[4] = {1,nkv,newcap,vd}, st[4] = {1,1,1,1}, s0[4] = {0,0,0,0};
        mlx_array zK = mlx_array_new(), zV = mlx_array_new();
        if (mlx_zeros(&zK, zk,4, mlx_array_dtype(*k_cache), stream) != 0 ||
            mlx_zeros(&zV, zv,4, mlx_array_dtype(*v_cache), stream) != 0) {
            mlx_array_free(zK); mlx_array_free(zV); return false;
        }
        mlx_array K1 = mlx_array_new(), V1 = mlx_array_new();
        if (off > 0) {
            int ck[4] = {1,nkv,off,kd}, cv[4] = {1,nkv,off,vd};
            if (mlx_slice_update(&K1, zK, *k_cache, s0,4, ck,4, st,4, stream) != 0 ||
                mlx_slice_update(&V1, zV, *v_cache, s0,4, cv,4, st,4, stream) != 0) {
                mlx_array_free(K1); mlx_array_free(V1); mlx_array_free(zK); mlx_array_free(zV); return false;
            }
        } else { mlx_array_set(&K1, zK); mlx_array_set(&V1, zV); }
        int w0[4] = {0,0,off,0}, wk[4] = {1,nkv,need,kd}, wv[4] = {1,nkv,need,vd};
        mlx_array K2 = mlx_array_new(), V2 = mlx_array_new();
        if (mlx_slice_update(&K2, K1, k_new, w0,4, wk,4, st,4, stream) != 0 ||
            mlx_slice_update(&V2, V1, v_new, w0,4, wv,4, st,4, stream) != 0) {
            mlx_array_free(K2); mlx_array_free(V2); mlx_array_free(K1); mlx_array_free(V1);
            mlx_array_free(zK); mlx_array_free(zV); return false;
        }
        mlx_array_set(k_cache, K2); mlx_array_set(v_cache, V2);
        llm_da_note_kv_append(nkv, n_tokens, kd, vd, esz, /*grew=*/1, off);
        mlx_array_free(zK); mlx_array_free(zV); mlx_array_free(K1); mlx_array_free(V1);
        mlx_array_free(K2); mlx_array_free(V2);
    }
    /* NB (2026-09-17): ipotesi "mlx_slice copia -> copiamo la KV per token" TESTATA col self-test
     * `llm_ai_mlx_kv_view_selftest` e **REFUTATA**: `mlx_slice` alloca 0 MB (e' una vista strided in
     * MLX, non una materializzazione). Nessun fix necessario qui: le due mlx_slice restano. */
    int p0[4] = {0,0,0,0}, ps[4] = {1,1,1,1};
    int pk[4] = {1,nkv,need,kd}, pv[4] = {1,nkv,need,vd};
    if (mlx_slice(k_view, *k_cache, p0,4, pk,4, ps,4, stream) != 0) {
        if (llm_da_enabled()) g_da.kv_fail++;
        return false;
    }
    if (mlx_slice(v_view, *v_cache, p0,4, pv,4, ps,4, stream) != 0) {
        if (llm_da_enabled()) g_da.kv_fail++;
        return false;
    }
    return true;
}

/* SDPA causale/sliding senza sinks (default di tutti gli altri modelli). */
static bool llm_mlx_gqa_sdpa(mlx_array* out, mlx_array q, mlx_array k, mlx_array v,
                             float scale, int window, mlx_stream stream)
{
    mlx_array no_sinks = mlx_array_new(); /* .ctx==NULL -> fast_sdpa senza sinks */
    bool ok = llm_mlx_gqa_sdpa_sinks(out, q, k, v, scale, window, no_sinks, stream);
    mlx_array_free(no_sinks);
    return ok;
}

/* (Il dispatch quantizzato/non-quantizzato resta in quantized_linear_cfg nel .c, che gia'
 * delega a plain_linear per mode="none" — commit 0cd0a538. Niente duplicazione qui.) */

/* ─────────────────────────────────────────────────────────────────────────────
 * KV quant per-layer (E3 — policy Recent FP16 / Cold Q8-Q4).
 *
 * Modulo per il long-context: a contesto lungo la KV-cache cresce O(N·L·H·D) e satura la
 * memoria unificata. Quantizzare i blocchi KV piu' vecchi (cold) a Q8/Q4 riduce la memoria
 * mantenendo i recenti in FP16 (qualita' sul contesto attivo).
 *
 * Espone le PRIMITIVE pure (quantize/dequantize di un blocco KV affine). La POLICY (quale
 * layer/blocco rendere cold, soglia) resta nel chiamante — coerenza con la visione modulare.
 */

/* LLM_KVROT (2026-09-12, adattato da VeloxQuant-MLX — pipeline RaBitQ/NSNQuant/TurboQuant):
 * Ruota il blocco KV lungo l'ULTIMA dim (head_dim) con una matrice di Hadamard NORMALIZZATA
 * (ortonormale, H=H^-1) PRIMA della quantizzazione; l'inverso si applica dopo il dequantize.
 * Rende la distribuzione piu' Gaussiana/isotropa -> riduce l'errore di quantizzazione su KV
 * heavy-tailed / a varianza per-canale molto diversa (validato: MSE heavy-tail inf->0,26,
 * per-channel inf->3,63; Gaussiana ~neutra, ratio 1,001). E' il passo rotate->quantize->pack
 * che LLMEnv non aveva. `dim` deve essere potenza di 2 (altrimenti no-op). Default OFF. */
static int llm_mlx_kvrot_enabled(void) {
    static int st = -1;
    if (st < 0) st = (llm_ai_mlx_opt_env("LLM_KVROT") != NULL) ? 1 : 0;
    return st;
}
static mlx_array g_kvrot_H = {0};
static int g_kvrot_dim = -1;
/* Ruota x [..., dim] con la Hadamard normalizzata [dim,dim] (simmetrica/ortonormale: H=H^-1).
 * Cache della matrice per dim. Ritorna false (no-op) se dim non e' potenza di 2 o alloc fallita. */
static bool llm_mlx_hadamard_rotate(mlx_array* out, mlx_array x, int dim, mlx_stream s)
{
    if (dim <= 0 || (dim & (dim - 1)) != 0) return false;
    if (g_kvrot_dim != dim || g_kvrot_H.ctx == NULL) {
        float* h = (float*)malloc((size_t)dim * (size_t)dim * sizeof(float));
        if (h == NULL) return false;
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++) {
                int v = i & j, pc = 0;
                while (v) { pc ^= (v & 1); v >>= 1; }
                h[i * dim + j] = (pc ? -1.0f : 1.0f) / sqrtf((float)dim);
            }
        const int hs[2] = {dim, dim};
        mlx_array H = mlx_array_new_data(h, hs, 2, MLX_FLOAT32);
        free(h);
        if (H.ctx == NULL) return false;
        mlx_array Hf = mlx_array_new();
        bool ok = (mlx_array_dtype(x) == MLX_FLOAT32)
                      ? (mlx_array_set(&Hf, H) == 0)
                      : (mlx_astype(&Hf, H, mlx_array_dtype(x), s) == 0);
        mlx_array_free(H);
        if (!ok) { mlx_array_free(Hf); return false; }
        if (g_kvrot_H.ctx != NULL) mlx_array_free(g_kvrot_H);
        g_kvrot_H = Hf;
        g_kvrot_dim = dim;
    }
    return mlx_matmul(out, x, g_kvrot_H, s) == 0;   /* [...,dim] @ [dim,dim] -> [...,dim] */
}

/* KIVI (2026-09-12, da VeloxQuant-MLX/KIVI): quantizzazione ASIMMETRICA **K per-CANALE /
 * V per-TOKEN**. K ha outlier PERSISTENTI per-canale (alcuni canali grandi su tutti i token):
 * la quantizzazione per-token (group lungo i canali) spreca bit; K per-canale (group lungo i
 * TOKEN, i.e. quantizza K^T sull'ultimo asse) riduce l'errore — validato in MLX: MSE di K
 * 0,069 -> 0,046 (**−34%**) con outlier di canale. V ha outlier per-token -> V per-token
 * (default). Tuning-free, nessuna calibrazione. Gate LLM_KIVI (default OFF). `per_channel=true`
 * per K, false per V. Compose con LLM_KVROT (rotazione prima). */
static int llm_mlx_kivi_enabled(void) {
    static int st = -1;
    if (st < 0) st = (llm_ai_mlx_opt_env("LLM_KIVI") != NULL) ? 1 : 0;
    return st;
}
static bool llm_mlx_kv_quantize_kivi(mlx_vector_array* out, mlx_array kv, int bits, int group_size,
                                     bool per_channel, mlx_stream stream)
{
    mlx_optional_int gsz = {group_size, true};
    mlx_optional_int bts = {bits, true};
    mlx_array null_gs = mlx_array_new();
    mlx_array xt = mlx_array_new();
    mlx_array xin = kv;
    if (per_channel) {   /* K: quantizza sull'asse TOKEN (trasponi la coppia finale) */
        int nd = (int)mlx_array_ndim(kv);
        if (nd >= 2) {
            int perm[8];
            for (int i = 0; i < nd; i++) perm[i] = i;
            int t = perm[nd - 1]; perm[nd - 1] = perm[nd - 2]; perm[nd - 2] = t;
            if (mlx_transpose_axes(&xt, kv, perm, nd, stream) == 0) xin = xt;
        }
    }
    mlx_array rot = mlx_array_new();
    if (llm_mlx_kvrot_enabled()) {   /* LLM_KVROT: rotazione prima (componibile) */
        const int nd = (int)mlx_array_ndim(xin);
        int dim = nd > 0 ? (int)mlx_array_shape(xin)[nd - 1] : 0;
        if (llm_mlx_hadamard_rotate(&rot, xin, dim, stream)) xin = rot;
    }
    bool ok = mlx_quantize(out, xin, gsz, bts, "affine", null_gs, stream) == 0;
    mlx_array_free(null_gs);
    mlx_array_free(rot);
    mlx_array_free(xt);
    return ok;
}
/* Dequant di K per-canale: dequantizza (l'asse è TOKEN) e ri-traspone. `qn`=quantizzato. */
static bool llm_mlx_kv_dequantize_kivi(mlx_array* out, mlx_array w, mlx_array scales, mlx_array biases,
                                       int bits, int group_size, mlx_stream stream)
{
    mlx_optional_int gsz = {group_size, true};
    mlx_optional_int bts = {bits, true};
    mlx_optional_dtype dt = {MLX_FLOAT32, true};
    mlx_array null_gs = mlx_array_new();
    mlx_array deq = mlx_array_new();
    bool ok = mlx_dequantize(&deq, w, scales, biases, gsz, bts, "affine", null_gs, dt, stream) == 0;
    mlx_array_free(null_gs);
    if (ok) {   /* ri-trasponi la coppia finale per tornare [.., S, D] */
        int nd = (int)mlx_array_ndim(deq);
        if (nd >= 2) {
            int perm[8];
            for (int i = 0; i < nd; i++) perm[i] = i;
            int t = perm[nd - 1]; perm[nd - 1] = perm[nd - 2]; perm[nd - 2] = t;
            ok = mlx_transpose_axes(out, deq, perm, nd, stream) == 0;
        } else { ok = mlx_array_set(out, deq) == 0; }
    }
    mlx_array_free(deq);
    if (ok && llm_mlx_kvrot_enabled()) {
        const int nd = (int)mlx_array_ndim(*out);
        int dim = nd > 0 ? (int)mlx_array_shape(*out)[nd - 1] : 0;
        mlx_array rot = mlx_array_new();
        if (llm_mlx_hadamard_rotate(&rot, *out, dim, stream)) mlx_array_set(out, rot);
        mlx_array_free(rot);
    }
    return ok;
}

/* --- Mixed-precision 2-tier: ZipCache (per-token) / Kitty (per-canale) ---------------
 * Da VeloxQuant (ZipCache: saliency = norma L2 della chiave, top-frazione a bit alti;
 * Kitty: rank per-canale per varianza online). Qui la saliency e' la MEDIA|.| per-token
 * (asse canali) o la VARIANZA per-canale (asse token); la soglia e' la media della saliency
 * (frazione adattiva, no top-k). Validato nel lab (tools/kv_quant_lab.py): Kitty -47% K,
 * ZipCache -63% V. Ritorna le DUE quantizzazioni (bits_lo/bits_hi) + un mask booleano
 * broadcastabile; il dequant combina con where(mask, hi, lo). Gate LLM_KVMIXED. */
static int llm_mlx_kvmixed_enabled(void) {
    static int st = -1;
    if (st < 0) st = (llm_ai_mlx_opt_env("LLM_KVMIXED") != NULL) ? 1 : 0;
    return st;
}
static bool llm_mlx_kv_saliency_mask(mlx_array* mask, mlx_array kv, bool per_token, mlx_stream s)
{
    int nd = (int)mlx_array_ndim(kv);
    if (nd < 2) return false;
    int ax = per_token ? (nd - 1) : (nd - 2);
    mlx_array ab = mlx_array_new();
    if (mlx_abs(&ab, kv, s) != 0) { mlx_array_free(ab); return false; }
    mlx_array r = mlx_array_new();
    bool ok = per_token ? (mlx_mean_axes(&r, ab, &ax, 1, false, s) == 0)
                        : (mlx_var_axes(&r, ab, &ax, 1, false, 0, s) == 0);
    mlx_array_free(ab);
    mlx_array m = mlx_array_new();
    if (ok) ok = mlx_mean(&m, r, false, s) == 0;
    mlx_array cond = mlx_array_new();
    if (ok) ok = mlx_greater(&cond, r, m, s) == 0;
    mlx_array_free(m);
    mlx_array_free(r);
    if (!ok) { mlx_array_free(cond); return false; }
    /* per-token: sal e' [.., S] (senza l'ultima dim) -> appendi dim finale 1 per broadcast */
    if (per_token) {
        const int* sh = mlx_array_shape(cond);
        int shape[8];
        for (int i = 0; i < nd; i++) shape[i] = sh[i];
        shape[nd - 1] = 1;
        mlx_array rsh = mlx_array_new();
        if (mlx_reshape(&rsh, cond, shape, nd, s) != 0) { mlx_array_free(rsh); mlx_array_free(cond); return false; }
        mlx_array_free(cond);
        cond = rsh;
    }
    mlx_array_set(mask, cond);
    mlx_array_free(cond);
    return true;
}
static bool llm_mlx_kv_quantize_mixed(mlx_vector_array* out_lo, mlx_vector_array* out_hi, mlx_array* mask,
                                      mlx_array kv, int bits_lo, int bits_hi, int gs, bool per_token,
                                      mlx_stream s)
{
    mlx_optional_int gsz = {gs, true};
    mlx_optional_int blo = {bits_lo, true}, bhi = {bits_hi, true};
    mlx_array null_gs = mlx_array_new();
    mlx_array mk = mlx_array_new();
    bool ok = llm_mlx_kv_saliency_mask(&mk, kv, per_token, s);
    if (ok) ok = mlx_quantize(out_lo, kv, gsz, blo, "affine", null_gs, s) == 0;
    if (ok) ok = mlx_quantize(out_hi, kv, gsz, bhi, "affine", null_gs, s) == 0;
    mlx_array_free(null_gs);
    if (ok) mlx_array_set(mask, mk);
    mlx_array_free(mk);
    return ok;
}
static bool llm_mlx_kv_dequantize_mixed(mlx_array* out, mlx_vector_array lo, mlx_vector_array hi,
                                        mlx_array mask, int bits_lo, int bits_hi, int gs, mlx_stream s)
{
    mlx_optional_int gsz = {gs, true}, blo = {bits_lo, true}, bhi = {bits_hi, true};
    mlx_optional_dtype dt = {MLX_FLOAT32, true};
    mlx_array null_gs = mlx_array_new();
    mlx_array lw = mlx_array_new(), ls = mlx_array_new(), lb = mlx_array_new();
    mlx_array hw = mlx_array_new(), hs = mlx_array_new(), hb = mlx_array_new();
    mlx_array dlo = mlx_array_new(), dhi = mlx_array_new(), res = mlx_array_new();
    bool ok = mlx_vector_array_get(&lw, lo, 0) == 0 && mlx_vector_array_get(&ls, lo, 1) == 0 &&
              mlx_vector_array_get(&lb, lo, 2) == 0 && mlx_vector_array_get(&hw, hi, 0) == 0 &&
              mlx_vector_array_get(&hs, hi, 1) == 0 && mlx_vector_array_get(&hb, hi, 2) == 0;
    if (ok) ok = mlx_dequantize(&dlo, lw, ls, lb, gsz, blo, "affine", null_gs, dt, s) == 0;
    if (ok) ok = mlx_dequantize(&dhi, hw, hs, hb, gsz, bhi, "affine", null_gs, dt, s) == 0;
    if (ok) ok = mlx_where(&res, mask, dhi, dlo, s) == 0;
    mlx_array_free(null_gs);
    mlx_array_free(lw); mlx_array_free(ls); mlx_array_free(lb);
    mlx_array_free(hw); mlx_array_free(hs); mlx_array_free(hb);
    mlx_array_free(dlo);
    mlx_array_free(dhi);
    if (ok) mlx_array_set(out, res);
    mlx_array_free(res);
    return ok;
}

/* --- NSNQuant (VeloxQuant, adattato da paper): Normalize-Shift-Normalize per-token + rotazione
 * Hadamard, poi quantize. Il "shift/normalize" per-token centra e scala i canali, la rotazione
 * Hadamard (ortonormale, auto-inversa) rende il residuo piu' Gaussiano. Validato nel lab
 * (tools/kv_quant_lab.py): MSE ratio K 0,635 (pari a rot/KIVI). Streaming-safe (per-token).
 * Salva mu/sd per-token per l'inverso: out = [w, scales, biases, mu, sd] (5 elementi).
 * Gate LLM_NSN (default OFF). */
static int llm_mlx_kv_nsn_enabled(void) {
    static int st = -1;
    if (st < 0) st = (llm_ai_mlx_opt_env("LLM_NSN") != NULL) ? 1 : 0;
    return st;
}
static bool llm_mlx_kv_quantize_nsn(mlx_vector_array* out, mlx_array kv, int bits, int group_size, mlx_stream stream)
{
    int nd = (int)mlx_array_ndim(kv);
    if (nd < 2) return false;
    const int* sh = mlx_array_shape(kv);
    int dim = sh[nd - 1];
    int ax = nd - 1;
    mlx_array mu = mlx_array_new(), sd = mlx_array_new(), sub = mlx_array_new();
    mlx_array sdf = mlx_array_new(), xn = mlx_array_new();
    bool ok = mlx_mean_axes(&mu, kv, &ax, 1, true, stream) == 0;
    if (ok) ok = mlx_std_axes(&sd, kv, &ax, 1, true, 0, stream) == 0;
    if (ok) ok = mlx_subtract(&sub, kv, mu, stream) == 0;
    mlx_array eps = mlx_array_new_float(1e-6f);
    if (ok) ok = mlx_add(&sdf, sd, eps, stream) == 0;
    if (ok) ok = mlx_divide(&xn, sub, sdf, stream) == 0;
    if (ok) {   /* rotazione Hadamard (auto-inversa), applicata anche se LLM_KVROT e' off */
        mlx_array r = mlx_array_new();
        if (llm_mlx_hadamard_rotate(&r, xn, dim, stream)) { mlx_array_free(xn); xn = r; }
        else mlx_array_free(r);
    }
    mlx_optional_int gsz = {group_size, true}, bts = {bits, true};
    mlx_array null_gs = mlx_array_new();
    if (ok) ok = mlx_quantize(out, xn, gsz, bts, "affine", null_gs, stream) == 0;
    if (ok) ok = (mlx_vector_array_append_value(*out, mu) == 0) &&
                 (mlx_vector_array_append_value(*out, sd) == 0);
    mlx_array_free(null_gs); mlx_array_free(eps);
    mlx_array_free(mu); mlx_array_free(sd); mlx_array_free(sub);
    mlx_array_free(sdf); mlx_array_free(xn);
    return ok;
}
static bool llm_mlx_kv_dequantize_nsn(mlx_array* out, mlx_vector_array qn, int bits, int group_size, mlx_stream stream)
{
    if (mlx_vector_array_size(qn) < 5) return false;
    mlx_array w = mlx_array_new(), sc = mlx_array_new(), bi = mlx_array_new();
    mlx_array mu = mlx_array_new(), sd = mlx_array_new();
    bool ok = mlx_vector_array_get(&w, qn, 0) == 0 && mlx_vector_array_get(&sc, qn, 1) == 0 &&
              mlx_vector_array_get(&bi, qn, 2) == 0 && mlx_vector_array_get(&mu, qn, 3) == 0 &&
              mlx_vector_array_get(&sd, qn, 4) == 0;
    mlx_optional_int gsz = {group_size, true}, bts = {bits, true};
    mlx_optional_dtype dt = {MLX_FLOAT32, true};
    mlx_array null_gs = mlx_array_new();
    mlx_array x = mlx_array_new();
    if (ok) ok = mlx_dequantize(&x, w, sc, bi, gsz, bts, "affine", null_gs, dt, stream) == 0;
    if (ok) {   /* rotazione inversa (Hadamard e' auto-inversa) */
        int nd = (int)mlx_array_ndim(x);
        const int* sh = mlx_array_shape(x);
        int dim = nd > 0 ? sh[nd - 1] : 0;
        mlx_array r = mlx_array_new();
        if (llm_mlx_hadamard_rotate(&r, x, dim, stream)) { mlx_array_free(x); x = r; }
        else mlx_array_free(r);
    }
    mlx_array mul = mlx_array_new(), res = mlx_array_new();
    if (ok) ok = mlx_multiply(&mul, x, sd, stream) == 0;
    if (ok) ok = mlx_add(&res, mul, mu, stream) == 0;
    mlx_array_free(null_gs);
    mlx_array_free(w); mlx_array_free(sc); mlx_array_free(bi); mlx_array_free(mu); mlx_array_free(sd);
    mlx_array_free(x); mlx_array_free(mul);
    if (ok) mlx_array_set(out, res);
    mlx_array_free(res);
    return ok;
}

/* --- GEAR (VeloxQuant): X ≈ quant(X) + low-rank del RESIDUO. Validato nel lab: MSE ratio 0,891
 * (il residuo recupera ~11%). out = [w, scales, biases, U_r, S_r, Vh_r]. Dequant:
 * dequant(base) + (U_r * S_r) @ Vh_r. SVD batched sulle ultime 2 dim. rank r=min(rank, min(S,D)).
 * Adatto a RICOMPRESSIONE PERIODICA della cache (SVD sull'intera cache), non per-step: applicabile
 * al long-context/offline storage dei modelli futuri. Gate LLM_GEAR (default OFF). */
static int llm_mlx_kv_gear_enabled(void) {
    static int st = -1;
    if (st < 0) st = (llm_ai_mlx_opt_env("LLM_GEAR") != NULL) ? 1 : 0;
    return st;
}
static bool llm_mlx_kv_quantize_gear(mlx_vector_array* out, mlx_array kv, int bits, int group_size,
                                     int rank, mlx_stream stream)
{
    int nd = (int)mlx_array_ndim(kv);
    if (nd < 2) return false;
    mlx_optional_int gsz = {group_size, true}, bts = {bits, true};
    mlx_array null_gs = mlx_array_new();
    mlx_vector_array base = mlx_vector_array_new();
    bool ok = mlx_quantize(&base, kv, gsz, bts, "affine", null_gs, stream) == 0;
    mlx_array bw = mlx_array_new(), bs = mlx_array_new(), bb = mlx_array_new();
    if (ok) ok = mlx_vector_array_get(&bw, base, 0) == 0 && mlx_vector_array_get(&bs, base, 1) == 0 &&
                 mlx_vector_array_get(&bb, base, 2) == 0;
    mlx_optional_dtype dt = {MLX_FLOAT32, true};
    mlx_array deq = mlx_array_new();
    if (ok) ok = mlx_dequantize(&deq, bw, bs, bb, gsz, bts, "affine", null_gs, dt, stream) == 0;
    mlx_array_free(bw); mlx_array_free(bs); mlx_array_free(bb);
    mlx_array res = mlx_array_new();
    if (ok) ok = mlx_subtract(&res, kv, deq, stream) == 0;
    mlx_vector_array svd = mlx_vector_array_new();
    if (ok) ok = mlx_linalg_svd(&svd, res, /*compute_uv=*/true, stream) == 0;
    mlx_array U = mlx_array_new(), S = mlx_array_new(), Vh = mlx_array_new();
    if (ok && mlx_vector_array_size(svd) >= 3) {
        ok = mlx_vector_array_get(&U, svd, 0) == 0 && mlx_vector_array_get(&S, svd, 1) == 0 &&
             mlx_vector_array_get(&Vh, svd, 2) == 0;
    } else if (ok) { ok = false; }
    /* r = min(rank, min(S,D)) */
    int Us[8], Up[8], Usv[8], Ss[8], Sp[8], Ssv[8], Vs[8], Vp[8], Vsv[8];
    mlx_array Ur = mlx_array_new(), Sr = mlx_array_new(), Vhr = mlx_array_new();
    if (ok) {
        const int* ush = mlx_array_shape(U);
        const int* ssh = mlx_array_shape(S);
        const int* vsh = mlx_array_shape(Vh);
        int r = rank;
        if (ssh != NULL && ssh[nd - 1] < r) r = ssh[nd - 1];
        if (r < 1) r = 1;
        for (int i = 0; i < nd; i++) { Us[i] = 0; Up[i] = ush[i]; Usv[i] = 1; }
        Up[nd - 2] = r;   /* U[..., :r, :] */
        ok = mlx_slice(&Ur, U, Us, nd, Up, nd, Usv, nd, stream) == 0;
        for (int i = 0; i < nd; i++) { Ss[i] = 0; Sp[i] = ssh[i]; Ssv[i] = 1; }
        Sp[nd - 1] = r;   /* S[..., :r] */
        if (ok) ok = mlx_slice(&Sr, S, Ss, nd, Sp, nd, Ssv, nd, stream) == 0;
        for (int i = 0; i < nd; i++) { Vs[i] = 0; Vp[i] = vsh[i]; Vsv[i] = 1; }
        Vp[nd - 2] = r;   /* Vh[..., :r, :] */
        if (ok) ok = mlx_slice(&Vhr, Vh, Vs, nd, Vp, nd, Vsv, nd, stream) == 0;
    }
    if (ok) {
        for (int i = 0; i < 3; i++) {
            mlx_array e = mlx_array_new();
            if (mlx_vector_array_get(&e, base, i) == 0) mlx_vector_array_append_value(*out, e);
            mlx_array_free(e);
        }
        ok = mlx_vector_array_append_value(*out, Ur) == 0 &&
             mlx_vector_array_append_value(*out, Sr) == 0 &&
             mlx_vector_array_append_value(*out, Vhr) == 0;
    }
    mlx_array_free(null_gs); mlx_vector_array_free(base); mlx_vector_array_free(svd);
    mlx_array_free(deq); mlx_array_free(res);
    mlx_array_free(U); mlx_array_free(S); mlx_array_free(Vh);
    mlx_array_free(Ur); mlx_array_free(Sr); mlx_array_free(Vhr);
    return ok;
}
static bool llm_mlx_kv_dequantize_gear(mlx_array* out, mlx_vector_array qn, int bits, int group_size, mlx_stream stream)
{
    if (mlx_vector_array_size(qn) < 6) return false;
    mlx_array w = mlx_array_new(), sc = mlx_array_new(), bi = mlx_array_new();
    mlx_array Ur = mlx_array_new(), Sr = mlx_array_new(), Vhr = mlx_array_new();
    bool ok = mlx_vector_array_get(&w, qn, 0) == 0 && mlx_vector_array_get(&sc, qn, 1) == 0 &&
              mlx_vector_array_get(&bi, qn, 2) == 0 && mlx_vector_array_get(&Ur, qn, 3) == 0 &&
              mlx_vector_array_get(&Sr, qn, 4) == 0 && mlx_vector_array_get(&Vhr, qn, 5) == 0;
    mlx_optional_int gsz = {group_size, true}, bts = {bits, true};
    mlx_optional_dtype dt = {MLX_FLOAT32, true};
    mlx_array null_gs = mlx_array_new();
    mlx_array deq = mlx_array_new(), us = mlx_array_new(), low = mlx_array_new(), res = mlx_array_new();
    if (ok) ok = mlx_dequantize(&deq, w, sc, bi, gsz, bts, "affine", null_gs, dt, stream) == 0;
    if (ok) ok = mlx_multiply(&us, Ur, Sr, stream) == 0;        /* [.., S, r] * [.., r] -> [.., S, r] */
    if (ok) ok = mlx_matmul(&low, us, Vhr, stream) == 0;        /* [.., S, r] @ [.., r, D] -> [.., S, D] */
    if (ok) ok = mlx_add(&res, deq, low, stream) == 0;
    mlx_array_free(null_gs);
    mlx_array_free(w); mlx_array_free(sc); mlx_array_free(bi);
    mlx_array_free(Ur); mlx_array_free(Sr); mlx_array_free(Vhr);
    mlx_array_free(deq); mlx_array_free(us); mlx_array_free(low);
    if (ok) mlx_array_set(out, res);
    mlx_array_free(res);
    return ok;
}

/* --- Registro/dispatch dichiarativo dei metodi di compressione KV (infrastruttura "LLVM-like":
 * il metodo e' una coppia pack/unpack selezionabile per nome; le primitive sono i mattoni).
 * Insieme alla policy (sotto), permette a un modello/comando futuro di DICHIARARE la propria
 * compressione (K e V separati) senza toccare il codice. --- */
static bool llm_mlx_kv_quantize_rot(mlx_vector_array* out, mlx_array kv, int bits, int group_size, mlx_stream s)
{
    int nd = (int)mlx_array_ndim(kv);
    if (nd < 2) return false;
    const int* sh = mlx_array_shape(kv);
    int dim = sh[nd - 1];
    mlx_array rot = mlx_array_new();
    mlx_array x = kv;
    bool have_rot = llm_mlx_hadamard_rotate(&rot, kv, dim, s);
    if (have_rot) x = rot;
    mlx_optional_int g = {group_size, true}, b = {bits, true};
    mlx_array ng = mlx_array_new();
    bool ok = mlx_quantize(out, x, g, b, "affine", ng, s) == 0;
    mlx_array_free(ng);
    if (have_rot) mlx_array_free(rot);
    return ok;
}
static bool llm_mlx_kv_dequantize_rot(mlx_array* out, mlx_vector_array packed, int bits, int group_size, mlx_stream s)
{
    if (mlx_vector_array_size(packed) < 3) return false;
    mlx_array w = mlx_array_new(), sc = mlx_array_new(), bi = mlx_array_new();
    bool ok = mlx_vector_array_get(&w, packed, 0) == 0 && mlx_vector_array_get(&sc, packed, 1) == 0 &&
              mlx_vector_array_get(&bi, packed, 2) == 0;
    mlx_optional_int g = {group_size, true}, b = {bits, true};
    mlx_optional_dtype dt = {MLX_FLOAT32, true};
    mlx_array ng = mlx_array_new();
    mlx_array x = mlx_array_new();
    if (ok) ok = mlx_dequantize(&x, w, sc, bi, g, b, "affine", ng, dt, s) == 0;
    if (ok) {
        int nd = (int)mlx_array_ndim(x);
        const int* sh = mlx_array_shape(x);
        int dim = nd > 0 ? sh[nd - 1] : 0;
        mlx_array r = mlx_array_new();
        if (llm_mlx_hadamard_rotate(&r, x, dim, s)) { mlx_array_free(x); x = r; } else mlx_array_free(r);
    }
    mlx_array_free(ng); mlx_array_free(w); mlx_array_free(sc); mlx_array_free(bi);
    if (ok) mlx_array_set(out, x);
    mlx_array_free(x);
    return ok;
}

typedef enum {
    LLM_KVM_NONE = 0, LLM_KVM_ROTATE, LLM_KVM_KIVI_K, LLM_KVM_KIVI_V,
    LLM_KVM_KITTY, LLM_KVM_ZIPCACHE, LLM_KVM_NSN, LLM_KVM_GEAR
} LlmMlxKvMethod;

static LlmMlxKvMethod llm_mlx_kv_method_from_name(const char* name) {
    if (name == NULL) return LLM_KVM_NONE;
    if (!strcmp(name, "none")) return LLM_KVM_NONE;
    if (!strcmp(name, "rotate")) return LLM_KVM_ROTATE;
    if (!strcmp(name, "kivi_k")) return LLM_KVM_KIVI_K;
    if (!strcmp(name, "kivi_v")) return LLM_KVM_KIVI_V;
    if (!strcmp(name, "kitty")) return LLM_KVM_KITTY;
    if (!strcmp(name, "zipcache")) return LLM_KVM_ZIPCACHE;
    if (!strcmp(name, "nsn")) return LLM_KVM_NSN;
    if (!strcmp(name, "gear")) return LLM_KVM_GEAR;
    return LLM_KVM_NONE;
}
/* pack: [.., S, D] -> vettore di array (formato dipendente dal metodo).
 * I metodi mixed (kitty/zipcache) ritornano [lo(3), hi(3), mask]; gli altri [w, s, b, extras]. */
static bool llm_mlx_kv_pack(mlx_vector_array* out, mlx_array kv, LlmMlxKvMethod m, int bits, int group_size,
                            int rank, mlx_stream s) {
    switch (m) {
        case LLM_KVM_ROTATE: return llm_mlx_kv_quantize_rot(out, kv, bits, group_size, s);
        case LLM_KVM_KIVI_K: return llm_mlx_kv_quantize_kivi(out, kv, bits, group_size, true, s);
        case LLM_KVM_KIVI_V: return llm_mlx_kv_quantize_kivi(out, kv, bits, group_size, false, s);
        case LLM_KVM_NSN:    return llm_mlx_kv_quantize_nsn(out, kv, bits, group_size, s);
        case LLM_KVM_GEAR:   return llm_mlx_kv_quantize_gear(out, kv, bits, group_size, rank, s);
        case LLM_KVM_KITTY:
        case LLM_KVM_ZIPCACHE: {
            mlx_vector_array lo = mlx_vector_array_new(), hi = mlx_vector_array_new();
            mlx_array mask = mlx_array_new();
            bool per_token = (m == LLM_KVM_ZIPCACHE);
            bool ok = llm_mlx_kv_quantize_mixed(&lo, &hi, &mask, kv, bits, bits + 2, group_size, per_token, s);
            if (ok) {
                for (int i = 0; i < 3; i++) { mlx_array e = mlx_array_new(); mlx_vector_array_get(&e, lo, i); mlx_vector_array_append_value(*out, e); mlx_array_free(e); }
                for (int i = 0; i < 3; i++) { mlx_array e = mlx_array_new(); mlx_vector_array_get(&e, hi, i); mlx_vector_array_append_value(*out, e); mlx_array_free(e); }
                ok = mlx_vector_array_append_value(*out, mask) == 0;
            }
            mlx_vector_array_free(lo); mlx_vector_array_free(hi); mlx_array_free(mask);
            return ok;
        }
        default: return false;   /* NONE: nessuna compressione */
    }
}
static bool llm_mlx_kv_unpack(mlx_array* out, mlx_vector_array packed, LlmMlxKvMethod m, int bits, int group_size,
                              mlx_stream s) {
    switch (m) {
        case LLM_KVM_ROTATE: return llm_mlx_kv_dequantize_rot(out, packed, bits, group_size, s);
        case LLM_KVM_NSN:    return llm_mlx_kv_dequantize_nsn(out, packed, bits, group_size, s);
        case LLM_KVM_GEAR:   return llm_mlx_kv_dequantize_gear(out, packed, bits, group_size, s);
        case LLM_KVM_KIVI_K:
        case LLM_KVM_KIVI_V: {
            if (mlx_vector_array_size(packed) < 3) return false;
            mlx_array w = mlx_array_new(), sc = mlx_array_new(), bi = mlx_array_new();
            mlx_vector_array_get(&w, packed, 0);
            mlx_vector_array_get(&sc, packed, 1);
            mlx_vector_array_get(&bi, packed, 2);
            /* KIVI dequant ri-traspone (K per-canale); per V (per-token) il transpose e' un
             * dettaglio della pack: qui ricomponiamo col dequant KIVI generico. */
            bool ok = llm_mlx_kv_dequantize_kivi(out, w, sc, bi, bits, group_size, s);
            mlx_array_free(w); mlx_array_free(sc); mlx_array_free(bi);
            return ok;
        }
        case LLM_KVM_KITTY:
        case LLM_KVM_ZIPCACHE: {
            if (mlx_vector_array_size(packed) < 7) return false;
            mlx_vector_array lo = mlx_vector_array_new(), hi = mlx_vector_array_new();
            mlx_array mask = mlx_array_new();
            for (int i = 0; i < 3; i++) { mlx_array e = mlx_array_new(); mlx_vector_array_get(&e, packed, i); mlx_vector_array_append_value(lo, e); mlx_array_free(e); }
            for (int i = 0; i < 3; i++) { mlx_array e = mlx_array_new(); mlx_vector_array_get(&e, packed, 3 + i); mlx_vector_array_append_value(hi, e); mlx_array_free(e); }
            mlx_vector_array_get(&mask, packed, 6);
            bool ok = llm_mlx_kv_dequantize_mixed(out, lo, hi, mask, bits, bits + 2, group_size, s);
            mlx_vector_array_free(lo); mlx_vector_array_free(hi); mlx_array_free(mask);
            return ok;
        }
        default: return false;
    }
}

/* Policy KV dichiarativa: un modello/comando la dichiara come stringa (K e V separati + retention).
 *   LLM_KV_POLICY="method=kivi_k,vmethod=none,bits=4,group=64,rank=8,retain=snapkv,budget=4096"
 * Tutti i campi opzionali; default = nessuna compressione/flusso invariato. */
typedef struct {
    LlmMlxKvMethod method;    /* metodo per K */
    LlmMlxKvMethod v_method;  /* metodo per V (di solito NONE o KIVI_V) */
    int bits;
    int group_size;
    int gear_rank;
    int retain;               /* 0=off, 1=snapkv */
    int retain_budget, retain_sink, retain_recent, retain_obs;
} LlmMlxKvPolicy;

static LlmMlxKvPolicy llm_mlx_kv_policy_from_env(void) {
    LlmMlxKvPolicy p;
    memset(&p, 0, sizeof p);
    p.method = LLM_KVM_NONE; p.v_method = LLM_KVM_NONE;
    p.bits = 4; p.group_size = 64; p.gear_rank = 8;
    p.retain_budget = 1024; p.retain_sink = 4; p.retain_recent = 128; p.retain_obs = 32;
    const char* s = llm_ai_mlx_opt_env("LLM_KV_POLICY");
    if (s == NULL || *s == '\0') return p;
    char buf[512];
    snprintf(buf, sizeof buf, "%s", s);
    char* save = NULL;
    for (char* tok = strtok_r(buf, ",", &save); tok != NULL; tok = strtok_r(NULL, ",", &save)) {
        char* eq = strchr(tok, '=');
        if (eq == NULL) continue;
        *eq = '\0';
        const char* k = tok;
        const char* v = eq + 1;
        if (!strcmp(k, "method")) p.method = llm_mlx_kv_method_from_name(v);
        else if (!strcmp(k, "vmethod")) p.v_method = llm_mlx_kv_method_from_name(v);
        else if (!strcmp(k, "bits")) p.bits = atoi(v);
        else if (!strcmp(k, "group")) p.group_size = atoi(v);
        else if (!strcmp(k, "rank")) p.gear_rank = atoi(v);
        else if (!strcmp(k, "retain")) p.retain = (strcmp(v, "snapkv") == 0) ? 1 : 0;
        else if (!strcmp(k, "budget")) p.retain_budget = atoi(v);
        else if (!strcmp(k, "sink")) p.retain_sink = atoi(v);
        else if (!strcmp(k, "recent")) p.retain_recent = atoi(v);
        else if (!strcmp(k, "obs")) p.retain_obs = atoi(v);
    }
    return p;
}

/* --- Token eviction/retention (famiglia VeloxQuant che RIDUCE la RSS) -> leva del CONTESTO
 * LUNGO. Building block componibili (la selezione completa richiede position-tracking + mask
 * nell'attention block -> progetto Dense-Long):
 *  - `llm_mlx_kv_snapkv_scores`: score per token = somma dell'attenzione delle ultime `obs`
 *    righe di K verso tutte le chiavi (key-as-query proxy di SnapKV; niente Q/cumulativo).
 *    Ranking monotono (la normalizzazione softmax non cambia l'ordine) -> exp non serve.
 *  - `llm_mlx_kv_select`: gather di K/V lungo l'asse sequenza per indice (sink+top+recent).
 * Lab: tools/kv_evict_lab.py -> SnapKV 5,66x / H2O 5,67x (budget 12,5%); StreamingLLM/random
 * peggio => la retention selettiva batte la finestra pura. Gate LLM_KVRETAIN (default OFF). */
static int llm_mlx_kvretain_enabled(void) {
    static int st = -1;
    if (st < 0) st = (llm_ai_mlx_opt_env("LLM_KVRETAIN") != NULL) ? 1 : 0;
    return st;
}
static int llm_mlx_kvretain_param(const char* name, int dflt) {
    const char* v = getenv(name);
    if (v == NULL || *v == '\0') return dflt;
    int x = atoi(v);
    return x > 0 ? x : dflt;
}
static bool llm_mlx_kv_snapkv_scores(mlx_array* scores, mlx_array k, int obs, mlx_stream s)
{
    int nd = (int)mlx_array_ndim(k);
    if (nd < 2) return false;
    const int* sh = mlx_array_shape(k);
    int S = sh[nd - 2];
    if (obs > S) obs = S;
    if (obs < 1) obs = 1;
    int start[8], stop[8], stride[8];
    for (int i = 0; i < nd; i++) { start[i] = 0; stop[i] = sh[i]; stride[i] = 1; }
    start[nd - 2] = S - obs; stop[nd - 2] = S;
    mlx_array kq = mlx_array_new();
    if (mlx_slice(&kq, k, start, nd, stop, nd, stride, nd, s) != 0) { mlx_array_free(kq); return false; }
    /* kt = K trasposta [.., D, S] ; prod = kq @ kt -> [.., obs, S] ; softmax sulle CHIAVI (S) ;
     * score = somma sull'asse delle query proxy (obs) -> [.., S]. Stabile (niente exp grezzo). */
    int perm[8];
    for (int i = 0; i < nd; i++) perm[i] = i;
    int t = perm[nd - 1]; perm[nd - 1] = perm[nd - 2]; perm[nd - 2] = t;
    mlx_array kt = mlx_array_new();
    bool ok = mlx_transpose_axes(&kt, k, perm, nd, s) == 0;
    mlx_array prod = mlx_array_new();
    if (ok) ok = mlx_matmul(&prod, kq, kt, s) == 0;
    mlx_array_free(kq);
    mlx_array_free(kt);
    mlx_array sm = mlx_array_new();
    /* ATTENZIONE: mlx_softmax (senza axis) normalizza sull'INTERO array in mlx-c; il softmax
     * va sulle CHIAVI (ultimo asse di prod = kq @ k^T -> S) => mlx_softmax_axis(..., -1, ...). */
    if (ok) ok = mlx_softmax_axis(&sm, prod, /*axis=*/-1, /*precise=*/false, s) == 0;
    mlx_array_free(prod);
    mlx_array sc = mlx_array_new();
    int ax = nd - 2;
    if (ok) ok = mlx_sum_axes(&sc, sm, &ax, 1, false, s) == 0;
    mlx_array_free(sm);
    if (ok) mlx_array_set(scores, sc);
    mlx_array_free(sc);
    return ok;
}
static bool llm_mlx_kv_select(mlx_array* out_k, mlx_array* out_v, mlx_array k, mlx_array v,
                              mlx_array idx, mlx_stream s)
{
    int nd = (int)mlx_array_ndim(k);
    if (nd < 2) return false;
    /* take_along_axis vuole indici broadcastabili su tutti gli assi tranne `axis`: idx e'
     * [.., seq_axis] -> espandi a [.., seq_axis, 1] (broadcast sulla dim canali). */
    mlx_array ie = mlx_array_new();
    if (mlx_expand_dims(&ie, idx, nd - 1, s) != 0) { mlx_array_free(ie); return false; }
    mlx_array ok_ = mlx_array_new(), ov = mlx_array_new();
    bool r = mlx_take_along_axis(&ok_, k, ie, nd - 2, s) == 0;
    if (r) r = mlx_take_along_axis(&ov, v, ie, nd - 2, s) == 0;
    mlx_array_free(ie);
    if (r) { mlx_array_set(out_k, ok_); mlx_array_set(out_v, ov); }
    mlx_array_free(ok_);
    mlx_array_free(ov);
    return r;
}
/* Retention completa (SnapKV): tiene `budget` token = sink iniziali + recent finali (sempre) +
 * i migliori per score; indici ORDINATI in modo crescente (posizioni originali, per la causalita').
 * `k`/`v` layout [.., S, D]. Se budget>=S passthrough. Gate LLM_KVRETAIN (chiamante). */
static bool llm_mlx_kv_retain(mlx_array* out_k, mlx_array* out_v, mlx_array k, mlx_array v,
                              int budget, int sink, int recent, int obs, mlx_stream s)
{
    int nd = (int)mlx_array_ndim(k);
    if (nd < 2) return false;
    const int* sh = mlx_array_shape(k);
    int S = sh[nd - 2];
    if (budget <= 0 || budget >= S || sink + recent >= budget) {
        mlx_array_set(out_k, k);
        mlx_array_set(out_v, v);
        return true;
    }
    mlx_array scores = mlx_array_new();
    if (!llm_mlx_kv_snapkv_scores(&scores, k, obs, s)) { mlx_array_free(scores); return false; }
    mlx_array pos = mlx_array_new();
    mlx_arange(&pos, 0, S, 1, MLX_INT32, s);
    mlx_array sink_a = mlx_array_new_int(sink);
    mlx_array rec_a = mlx_array_new_int(S - recent);
    mlx_array c1 = mlx_array_new(), c2 = mlx_array_new(), cond = mlx_array_new();
    mlx_array inf = mlx_array_new_float(3.4028235e38f);
    bool ok = mlx_less(&c1, pos, sink_a, s) == 0;
    if (ok) ok = mlx_greater_equal(&c2, pos, rec_a, s) == 0;
    if (ok) ok = mlx_logical_or(&cond, c1, c2, s) == 0;
    mlx_array s2 = mlx_array_new();
    if (ok) ok = mlx_where(&s2, cond, inf, scores, s) == 0;
    mlx_array neg = mlx_array_new(), order = mlx_array_new(), top = mlx_array_new();
    if (ok) ok = mlx_negative(&neg, s2, s) == 0;
    if (ok) ok = mlx_argsort_axis(&order, neg, (int)mlx_array_ndim(neg) - 1, s) == 0;
    if (ok) {
        int ond = (int)mlx_array_ndim(order);
        const int* osh = mlx_array_shape(order);
        int st[8], sp[8], sv[8];
        for (int i = 0; i < ond; i++) { st[i] = 0; sp[i] = osh[i]; sv[i] = 1; }
        st[ond - 1] = 0;
        sp[ond - 1] = budget;
        ok = mlx_slice(&top, order, st, ond, sp, ond, sv, ond, s) == 0;
    }
    mlx_array idx = mlx_array_new();
    if (ok) ok = mlx_sort_axis(&idx, top, (int)mlx_array_ndim(top) - 1, s) == 0;   /* crescente */
    if (llm_mlx_kvretain_param("LLM_KVRETAIN_DEBUG", 0)) {
        fprintf(stderr, "[kvretain] S=%d budget=%d nd=%d scores_nd=%d order_nd=%d top_nd=%d idx_nd=%d\n",
                S, budget, nd, mlx_array_ndim(scores), mlx_array_ndim(order), mlx_array_ndim(top), mlx_array_ndim(idx));
    }
    if (ok) ok = llm_mlx_kv_select(out_k, out_v, k, v, idx, s);
    mlx_array_free(scores); mlx_array_free(pos); mlx_array_free(sink_a); mlx_array_free(rec_a);
    mlx_array_free(c1); mlx_array_free(c2); mlx_array_free(cond); mlx_array_free(inf);
    mlx_array_free(s2); mlx_array_free(neg); mlx_array_free(order); mlx_array_free(top);
    mlx_array_free(idx);
    return ok;
}

static bool llm_mlx_kv_quantize(mlx_vector_array* out, mlx_array kv, int bits, int group_size, mlx_stream stream)
{
    mlx_optional_int gsz = {group_size, true};
    mlx_optional_int bts = {bits, true};
    mlx_array null_gs = mlx_array_new();
    mlx_array rot = mlx_array_new();
    mlx_array kvin = kv;
    if (llm_mlx_kvrot_enabled()) {   /* LLM_KVROT: ruota PRIMA della quantizzazione */
        const int nd = (int)mlx_array_ndim(kv);
        int dim = nd > 0 ? (int)mlx_array_shape(kv)[nd - 1] : 0;
        if (llm_mlx_hadamard_rotate(&rot, kv, dim, stream)) kvin = rot;
    }
    bool ok = mlx_quantize(out, kvin, gsz, bts, "affine", null_gs, stream) == 0;
    mlx_array_free(null_gs);
    mlx_array_free(rot);
    return ok;
}

static bool llm_mlx_kv_dequantize(mlx_array* out, mlx_array w, mlx_array scales, mlx_array biases,
                                  int bits, int group_size, mlx_stream stream)
{
    mlx_optional_int gsz = {group_size, true};
    mlx_optional_int bts = {bits, true};
    mlx_optional_dtype dt = {MLX_FLOAT32, true};
    mlx_array null_gs = mlx_array_new();
    bool ok = mlx_dequantize(out, w, scales, biases, gsz, bts, "affine", null_gs, dt, stream) == 0;
    mlx_array_free(null_gs);
    if (ok && llm_mlx_kvrot_enabled()) {   /* LLM_KVROT: inverso (H=H^-1) dopo il dequantize */
        const int nd = (int)mlx_array_ndim(*out);
        int dim = nd > 0 ? (int)mlx_array_shape(*out)[nd - 1] : 0;
        mlx_array rot = mlx_array_new();
        if (llm_mlx_hadamard_rotate(&rot, *out, dim, stream)) mlx_array_set(out, rot);
        mlx_array_free(rot);
    }
    return ok;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Policy KV quant applicativa (E3): quantizza la parte COLD (piu' vecchia) della
 * cache a Q8/Q4, mantiene i recenti FP16. Pura: prende la cache e la soglia,
 * restituisce (quant, scales, biases, n_cold) — il chiamante decide se/come usare.
 *
 * cache: [B, H_kv, N, D] (layout del motore). Se N <= threshold -> nessuna quant.
 * n_cold = N - threshold (i token piu' vecchi da quantizzare).
 * ritorna true se ok; quant_w/sc/b vuoti e n_cold=0 se N<=threshold.
 */
static bool llm_mlx_kv_apply_policy(mlx_vector_array* out, mlx_array cache, int threshold,
                                    int bits, int group_size, int* n_cold, mlx_stream stream)
{
    int ndim = (int)mlx_array_ndim(cache);
    int N = ndim > 2 ? (int)mlx_array_shape(cache)[ndim - 2] : 0;
    *n_cold = (N > threshold && threshold >= 0) ? (N - threshold) : 0;
    if (*n_cold <= 0) return true;   /* nulla da quantizzare */
    /* slice della parte cold: [B, Hkv, n_cold, D] lungo l'asse sequenza (ndim-2) */
    int cold_start[4] = {0, 0, 0, 0};
    int cold_stop[4] = {1, (int)mlx_array_shape(cache)[1], *n_cold, (int)mlx_array_shape(cache)[ndim-1]};
    int ones[4] = {1, 1, 1, 1};
    mlx_array cold = mlx_array_new();
    bool ok = mlx_slice(&cold, cache, cold_start, 4, cold_stop, 4, ones, 4, stream) == 0;
    if (ok) ok = llm_mlx_kv_quantize(out, cold, bits, group_size, stream);
    mlx_array_free(cold);
    return ok;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * NSA attention sparsa — DECODE (L=1), modulo long-context.
 *
 * Caso reale del decode: una singola query (L=1) seleziona i top-k token di K piu'
 * rilevanti tra i CAUSALI (passati), li estrae da K/V e fa l'attention solo su
 * quelli + una sliding window locale. Riduce il costo da O(N) (attenzione piena su
 * tutta la cache) a O(topk + window) per passo di decode.
 * Reference: DeepSeek NSA (2502.11089) — compression + selection + sliding.
 *
 * LOSSY (con topk < N l'output NON e' identico all'attention piena — trade-off
 * velocita'/qualita', da attivare a contesto lungo e verificare per qualita').
 * Pura; il chiamante decide quando usarla.
 *
 * Input (gia' trasposti): q [B,nh,1,D], k/v [B,nh,N,D].
 * topk = token selezionati; window = sliding locale (0 = nessuna); scale = attenzione.
 * Output: attention sparsa [B,nh,1,D]. Ritorna true se ok.
 *
 * (Il prefill sparsa, L>1 con selezione per-query, e' il passo successivo — richiede
 * loop o SDPA a 5D; qui il decode L=1 e' il caso semplice e completo.)
 */
static bool llm_mlx_nsa_decode_sdpa(mlx_array* out, mlx_array q, mlx_array k, mlx_array v,
                                    float scale, int topk, int window, mlx_stream stream)
{
    int qnd = (int)mlx_array_ndim(q);
    int L = qnd > 2 ? (int)mlx_array_shape(q)[qnd - 2] : 1;
    int N = qnd > 2 ? (int)mlx_array_shape(k)[qnd - 2] : 1;
    int D = qnd > 3 ? (int)mlx_array_shape(q)[qnd - 1] : 1;
    int nh = qnd > 1 ? (int)mlx_array_shape(q)[1] : 1;
    int B = qnd > 0 ? (int)mlx_array_shape(q)[0] : 1;

    /* Fallback: se non serve selezione, attention piena (parita'). */
    if (L != 1 || topk <= 0 || topk >= N) {
        return llm_mlx_gqa_sdpa(out, q, k, v, scale, window, stream);
    }

    /* 1. scores = q·k^T [B,nh,1,N] -> media heads [B,1,1,N] -> squeeze a [B,1,N] */
    mlx_array kt = mlx_array_new();
    int perm[4] = {0, 1, 3, 2};
    if (mlx_transpose_axes(&kt, k, perm, 4, stream) != 0) { mlx_array_free(kt); return false; }
    mlx_array qk = mlx_array_new();
    if (mlx_matmul(&qk, q, kt, stream) != 0) { mlx_array_free(kt); mlx_array_free(qk); return false; }
    mlx_array qk_m = mlx_array_new();
    if (mlx_mean_axis(&qk_m, qk, 1, true, stream) != 0) {
        mlx_array_free(kt); mlx_array_free(qk); mlx_array_free(qk_m); return false;
    }
    /* qk_m [B,1,1,N] -> squeeze assi 1,2 -> [B,1,N] per topk per-query */
    mlx_array qk_sq = mlx_array_new();
    int s1[1] = {1};
    if (mlx_squeeze_axes(&qk_sq, qk_m, s1, 1, stream) != 0) {
        mlx_array_free(kt); mlx_array_free(qk); mlx_array_free(qk_m); mlx_array_free(qk_sq); return false;
    }
    int s2[1] = {1};
    mlx_array qk_s = mlx_array_new();
    if (mlx_squeeze_axes(&qk_s, qk_sq, s2, 1, stream) != 0) {
        mlx_array_free(kt); mlx_array_free(qk); mlx_array_free(qk_m); mlx_array_free(qk_sq); mlx_array_free(qk_s); return false;
    }

    /* 2. Mask causale: -inf dove k_idx > query_idx (per decode L=1, query e' l'ultimo
     * token -> tutti i k_idx < N-1 sono passati, l'ultimo e' il corrente). Solo i primi
     * N-1 token sono candidati causali. */
    int mk_shape[2] = { B, N };
    float* md = (float*)malloc((size_t)B * (size_t)N * sizeof(float));
    for (int j = 0; j < N; j++) md[j] = (j <= N - 1) ? 0.0f : -INFINITY; /* tutti passati+corrente */
    mlx_array mask = mlx_array_new_data(md, mk_shape, 2, MLX_FLOAT32);
    free(md);
    mlx_array qk_c = mlx_array_new();
    bool ok = mlx_add(&qk_c, qk_s, mask, stream) == 0;
    mlx_array_free(mask);

    /* 3. topk per-query -> idx [B,1,topk] (una sola query), via argpartition_axis
     * (restituisce gli INDICI, come il MoE; topk_axis restituirebbe i valori float).
     * argpartition(kth=-topk) ordina e gli ultimi topk sono i maggiori -> slice. */
    mlx_array part = mlx_array_new();
    mlx_array idx = mlx_array_new();
    if (ok) ok = mlx_argpartition_axis(&part, qk_c, /*kth=*/-topk, /*axis=*/-1, stream) == 0;
    mlx_array_free(qk_c);
    if (ok) {
        int pnd = (int)mlx_array_ndim(part);
        int last = (int)mlx_array_shape(part)[pnd - 1];
        int pstart[2] = {0, last - topk};
        int pstop[2] = {B, last};
        int pstrides[2] = {1, 1};
        mlx_array view = mlx_array_new();
        ok = mlx_slice(&view, part, pstart, 2, pstop, 2, pstrides, 2, stream) == 0;
        if (ok) ok = mlx_contiguous(&idx, view, /*allow_col_major=*/false, stream) == 0;
        mlx_array_free(view);
    }
    mlx_array_free(part);

    /* 4. gather k/v per-query (pattern 5D): k->[B,nh,1,N,D], idx condiviso tra le head
     * -> reshape [B,1,1,topk,1] poi broadcast a [B,nh,1,topk,1],
     * take_along_axis(axis 3) -> [B,nh,1,topk,D]. */
    int k5shape[5] = { B, nh, 1, N, D };
    int i5s[5] = { B, 1, 1, topk, 1 };
    int i5b[5] = { B, nh, 1, topk, 1 };
    mlx_array k5 = mlx_array_new(), idx5 = mlx_array_new(), v5 = mlx_array_new();
    mlx_array idx5b = mlx_array_new();
    mlx_array k_sel = mlx_array_new(), v_sel = mlx_array_new();
    if (ok) ok = mlx_reshape(&k5, k, k5shape, 5, stream) == 0;
    if (ok) ok = mlx_reshape(&v5, v, k5shape, 5, stream) == 0;
    if (ok) ok = mlx_reshape(&idx5, idx, i5s, 5, stream) == 0;
    if (ok) ok = mlx_broadcast_to(&idx5b, idx5, i5b, 5, stream) == 0;
    if (ok) ok = mlx_take_along_axis(&k_sel, k5, idx5b, 3, stream) == 0;
    if (ok) ok = mlx_take_along_axis(&v_sel, v5, idx5b, 3, stream) == 0;
    mlx_array_free(idx5b);

    /* 5. k_sel/v_sel [B,nh,1,topk,D] -> squeeze asse 2 -> [B,nh,topk,D] per SDPA */
    mlx_array k_sel3 = mlx_array_new(), v_sel3 = mlx_array_new();
    int sq2[1] = {2};
    if (ok) ok = mlx_squeeze_axes(&k_sel3, k_sel, sq2, 1, stream) == 0;
    if (ok) ok = mlx_squeeze_axes(&v_sel3, v_sel, sq2, 1, stream) == 0;

    /* 6. SDPA causale su q [B,nh,1,D] vs k_sel3/v_sel3 [B,nh,topk,D]. */
    if (ok) ok = llm_mlx_gqa_sdpa(out, q, k_sel3, v_sel3, scale, 0, stream);

    mlx_array_free(kt); mlx_array_free(qk); mlx_array_free(qk_m);
    mlx_array_free(qk_sq); mlx_array_free(qk_s);
    mlx_array_free(idx); mlx_array_free(k5); mlx_array_free(idx5); mlx_array_free(v5);
    mlx_array_free(k_sel); mlx_array_free(v_sel);
    mlx_array_free(k_sel3); mlx_array_free(v_sel3);
    return ok;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * §G-bis leva 1 — decode con attenzione SPARSA a LIVELLO PAGINA.
 *
 * A differenza di llm_mlx_nsa_decode_sdpa (che calcola q·k su TUTTO N), qui le pagine da
 * attendere le scegli il chiamante con llm_mlx_kvpagesel.h (score query-vs-SUMMARY di pagina,
 * O(pagine)) e q·k viene calcolato SOLO sui token delle pagine scelte -> i byte KV letti sono
 * O(k*page), non O(N). LOSSY se non tutte le pagine sono selezionate.
 *
 * q [B,nh,1,D]; k/v [B,nkv,N,D]; `pages` = indici CRESCENTI delle pagine selezionate (page =
 * token per pagina). In decode L=1 tutti i token selezionati sono <= N-1 (passato+corrente):
 * nessuna maschera causale necessaria.
 */
static bool llm_mlx_pagesel_decode_sdpa(mlx_array* out, mlx_array q, mlx_array k, mlx_array v,
                                        float scale, int page, const int* pages, int n_sel,
                                        mlx_stream stream)
{
    int qnd = (int)mlx_array_ndim(q);
    int L = qnd > 2 ? (int)mlx_array_shape(q)[qnd - 2] : 1;
    int N = qnd > 2 ? (int)mlx_array_shape(k)[qnd - 2] : 1;
    int D = qnd > 3 ? (int)mlx_array_shape(q)[qnd - 1] : 1;
    int nh = qnd > 1 ? (int)mlx_array_shape(q)[1] : 1;
    int nkv = (int)mlx_array_ndim(k) > 1 ? (int)mlx_array_shape(k)[1] : nh; /* GQA: nkv puo' < nh */
    int B = qnd > 0 ? (int)mlx_array_shape(q)[0] : 1;
    if (L != 1 || page <= 0 || !pages || n_sel <= 0)
        return llm_mlx_gqa_sdpa(out, q, k, v, scale, 0, stream);

    /* indice (host) dei token delle pagine selezionate; M variabile (l'ultima pagina puo'
     * essere parziale). Dedup implicito: `pages` crescenti e disgiunte. */
    int32_t* ih = (int32_t*)malloc((size_t)n_sel * (size_t)page * sizeof(int32_t));
    if (!ih) return false;
    int M = 0;
    for (int i = 0; i < n_sel; i++) {
        int p = pages[i]; if (p < 0) continue;
        int lo = p * page, hi = lo + page; if (hi > N) hi = N;
        for (int t = lo; t < hi; t++) ih[M++] = (int32_t)t;
    }
    if (M == 0) { free(ih); return llm_mlx_gqa_sdpa(out, q, k, v, scale, 0, stream); }

    int ishape[5] = { B, 1, 1, M, 1 };
    int i5b[5] = { B, nkv, 1, M, 1 };          /* idx condiviso tra le kv head */
    int k5shape[5] = { B, nkv, 1, N, D };
    mlx_array idx = mlx_array_new_data(ih, ishape, 5, MLX_INT32);
    free(ih);
    mlx_array k5 = mlx_array_new(), v5 = mlx_array_new(), idx5b = mlx_array_new();
    mlx_array k_sel = mlx_array_new(), v_sel = mlx_array_new();
    bool ok = (idx.ctx != NULL);
    if (ok) ok = mlx_reshape(&k5, k, k5shape, 5, stream) == 0;
    if (ok) ok = mlx_reshape(&v5, v, k5shape, 5, stream) == 0;
    if (ok) ok = mlx_broadcast_to(&idx5b, idx, i5b, 5, stream) == 0;
    if (ok) ok = mlx_take_along_axis(&k_sel, k5, idx5b, 3, stream) == 0;
    if (ok) ok = mlx_take_along_axis(&v_sel, v5, idx5b, 3, stream) == 0;
    mlx_array k3 = mlx_array_new(), v3 = mlx_array_new();
    int sq2[1] = {2};
    if (ok) ok = mlx_squeeze_axes(&k3, k_sel, sq2, 1, stream) == 0;
    if (ok) ok = mlx_squeeze_axes(&v3, v_sel, sq2, 1, stream) == 0;
    if (ok) ok = llm_mlx_gqa_sdpa(out, q, k3, v3, scale, 0, stream); /* GQA: nh query vs nkv kv */
    mlx_array_free(idx); mlx_array_free(k5); mlx_array_free(v5); mlx_array_free(idx5b);
    mlx_array_free(k_sel); mlx_array_free(v_sel); mlx_array_free(k3); mlx_array_free(v3);
    return ok;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * GatedDelta — passo ricorrente (linear-attention O(N), il cuore del long-context).
 *
 * Un solo step del GatedDeltaNet (reference mlx-lm gated_delta.py): dato il token
 * corrente e lo stato, aggiorna lo stato associativo S [B,Hv,Dv,Dk] e produce y.
 * E' la primitiva L2 che rende il prefill O(N) invece di O(N²): non calcola QKᵀ
 * quadratico, mantiene una matrice di stato costante.
 *
 *   decay = g;   state = state * decay                       # [B,Hv,Dv,Dk]
 *   kv_mem = (state * k).sum(-1)                             # [B,Hv,Dv]
 *   delta = (v - kv_mem) * beta
 *   state = state + k * delta                                # aggiornamento associativo
 *   y     = (state * q).sum(-1)                              # [B,Hv,Dv]
 *
 * Input (per UN token, gia' squeezati):
 *   q,k [B,Hv,Dk], v [B,Hv,Dv], g/beta [B,Hv] (già repeat a Hv se Hv>Hk),
 *   state [B,Hv,Dv,Dk] (aggiornato in-place via mlx_array_set).
 * Output: *y [B,Hv,Dv]. Ritorna true se ok.
 */
static bool llm_mlx_gated_delta_step_ext(mlx_array* y, mlx_array* state,
                                         mlx_array q, mlx_array k, mlx_array v,
                                         mlx_array g, mlx_array beta, bool g_vector, mlx_stream stream)
{
    /* g scalare [B,Hv]  -> [B,Hv,1,1] (gate per-testa; qwen/GLM GDN);
     * g vettoriale [B,Hv,Dk] -> [B,Hv,1,Dk] (gate per-canale; Kimi KDA compute_g). */
    int gnd = g_vector ? 1 : 2;
    int gax[2] = {2, 3};
    mlx_array g4 = mlx_array_new();
    if (mlx_expand_dims_axes(&g4, g, gax, gnd, stream) != 0) { mlx_array_free(g4); return false; }
    mlx_array state_d = mlx_array_new();
    if (mlx_multiply(&state_d, *state, g4, stream) != 0) {
        mlx_array_free(g4); mlx_array_free(state_d); return false;
    }
    int kax[1] = {2};
    mlx_array k4 = mlx_array_new();
    if (mlx_expand_dims_axes(&k4, k, kax, 1, stream) != 0) {
        mlx_array_free(g4); mlx_array_free(state_d); mlx_array_free(k4); return false;
    }
    mlx_array sk = mlx_array_new();
    if (mlx_multiply(&sk, state_d, k4, stream) != 0) {
        mlx_array_free(g4); mlx_array_free(state_d); mlx_array_free(k4); mlx_array_free(sk); return false;
    }
    mlx_array kv_mem = mlx_array_new();
    if (mlx_sum_axis(&kv_mem, sk, -1, false, stream) != 0) {
        mlx_array_free(g4); mlx_array_free(state_d); mlx_array_free(k4); mlx_array_free(sk); mlx_array_free(kv_mem); return false;
    }
    mlx_array vm = mlx_array_new();
    if (mlx_subtract(&vm, v, kv_mem, stream) != 0) {
        mlx_array_free(g4); mlx_array_free(state_d); mlx_array_free(k4); mlx_array_free(sk); mlx_array_free(kv_mem); mlx_array_free(vm); return false;
    }
    int bax[1] = {2};
    mlx_array beta4 = mlx_array_new();
    if (mlx_expand_dims_axes(&beta4, beta, bax, 1, stream) != 0) {
        mlx_array_free(g4); mlx_array_free(state_d); mlx_array_free(k4); mlx_array_free(sk); mlx_array_free(kv_mem); mlx_array_free(vm); mlx_array_free(beta4); return false;
    }
    mlx_array delta = mlx_array_new();
    if (mlx_multiply(&delta, vm, beta4, stream) != 0) {
        mlx_array_free(g4); mlx_array_free(state_d); mlx_array_free(k4); mlx_array_free(sk); mlx_array_free(kv_mem); mlx_array_free(vm); mlx_array_free(beta4); mlx_array_free(delta); return false;
    }
    int dax[1] = {3};
    mlx_array delta4 = mlx_array_new();
    if (mlx_expand_dims_axes(&delta4, delta, dax, 1, stream) != 0) {
        mlx_array_free(g4); mlx_array_free(state_d); mlx_array_free(k4); mlx_array_free(sk); mlx_array_free(kv_mem); mlx_array_free(vm); mlx_array_free(beta4); mlx_array_free(delta); mlx_array_free(delta4); return false;
    }
    mlx_array kd = mlx_array_new();
    if (mlx_multiply(&kd, k4, delta4, stream) != 0) {
        mlx_array_free(g4); mlx_array_free(state_d); mlx_array_free(k4); mlx_array_free(sk); mlx_array_free(kv_mem); mlx_array_free(vm); mlx_array_free(beta4); mlx_array_free(delta); mlx_array_free(delta4); mlx_array_free(kd); return false;
    }
    mlx_array state_new = mlx_array_new();
    if (mlx_add(&state_new, state_d, kd, stream) != 0) {
        mlx_array_free(g4); mlx_array_free(state_d); mlx_array_free(k4); mlx_array_free(sk); mlx_array_free(kv_mem); mlx_array_free(vm); mlx_array_free(beta4); mlx_array_free(delta); mlx_array_free(delta4); mlx_array_free(kd); mlx_array_free(state_new); return false;
    }
    /* aggiorna state in-place (copia in *state) */
    if (mlx_array_set(state, state_new) != 0) {
        mlx_array_free(g4); mlx_array_free(state_d); mlx_array_free(k4); mlx_array_free(sk); mlx_array_free(kv_mem); mlx_array_free(vm); mlx_array_free(beta4); mlx_array_free(delta); mlx_array_free(delta4); mlx_array_free(kd); mlx_array_free(state_new); return false;
    }
    mlx_array q4 = mlx_array_new();
    if (mlx_expand_dims_axes(&q4, q, kax, 1, stream) != 0) {
        mlx_array_free(g4); mlx_array_free(state_d); mlx_array_free(k4); mlx_array_free(sk); mlx_array_free(kv_mem); mlx_array_free(vm); mlx_array_free(beta4); mlx_array_free(delta); mlx_array_free(delta4); mlx_array_free(kd); mlx_array_free(state_new); mlx_array_free(q4); return false;
    }
    mlx_array sq = mlx_array_new();
    if (mlx_multiply(&sq, *state, q4, stream) != 0) {
        mlx_array_free(g4); mlx_array_free(state_d); mlx_array_free(k4); mlx_array_free(sk); mlx_array_free(kv_mem); mlx_array_free(vm); mlx_array_free(beta4); mlx_array_free(delta); mlx_array_free(delta4); mlx_array_free(kd); mlx_array_free(state_new); mlx_array_free(q4); mlx_array_free(sq); return false;
    }
    bool ok = mlx_sum_axis(y, sq, -1, false, stream) == 0;
    mlx_array_free(g4); mlx_array_free(state_d); mlx_array_free(k4); mlx_array_free(sk); mlx_array_free(kv_mem);
    mlx_array_free(vm); mlx_array_free(beta4); mlx_array_free(delta); mlx_array_free(delta4);
    mlx_array_free(kd); mlx_array_free(state_new); mlx_array_free(q4); mlx_array_free(sq);
    return ok;
}
/* Wrapper retro-compatibile: g SCALARE per-testa [B,Hv] (qwen3_5/qwen3_next/GLM GDN). */
static bool llm_mlx_gated_delta_step(mlx_array* y, mlx_array* state,
                                     mlx_array q, mlx_array k, mlx_array v,
                                     mlx_array g, mlx_array beta, mlx_stream stream)
{
    return llm_mlx_gated_delta_step_ext(y, state, q, k, v, g, beta, /*g_vector=*/false, stream);
}


/* ─────────────────────────────────────────────────────────────────────────────
 * CACHE ABI (§8.4 del piano long-context): CHECKPOINT / ROLLBACK / COMMIT / TRIM.
 *
 * Base per N-Gram / speculative decoding su modelli hybrid (GDN/SSM + attention):
 * un reject deve ripristinare NON solo la KV-cache ma anche lo stato ricorrente.
 * In MLX-C gli array sono IMMUTABILI (nodi del grafo lazy): ogni `append` crea un
 * NUOVO array, non muta il precedente. Quindi un checkpoint e' semplicemente un
 * ALIAS dell'array corrente: resta valido finche' il riferimento e' vivo, e il
 * rollback ripristina il puntatore al vecchio array (che non e' stato toccato).
 * Non serve copia profonda -> costo O(1), nessun eval.
 *
 * Uso tipico (orchestrato a livello motore per l'INTERO set per-layer):
 *   CHECKPOINT(ckpt[l], k_cache[l]); CHECKPOINT(ckpt[l], v_cache[l]);
 *   ... generate draft ...
 *   se reject: ROLLBACK(k_cache[l], ckpt[l]); ROLLBACK(v_cache[l], ckpt[l]);
 *   se accept: COMMIT(ckpt[l]) per ogni l.
 */

/* Salva un checkpoint della cache (alias immutabile O(1), nessun eval). */
static bool llm_mlx_cache_checkpoint(mlx_array* ckpt, mlx_array cache, mlx_stream stream)
{
    (void)stream;
    return mlx_array_set(ckpt, cache) == 0;
}

/* Ripristina la cache al checkpoint salvato (ri-punta al vecchio array immutabile). */
static bool llm_mlx_cache_rollback(mlx_array* cache, mlx_array ckpt, mlx_stream stream)
{
    (void)stream;
    return mlx_array_set(cache, ckpt) == 0;
}

/* Conferma: dimentica il checkpoint (libera il riferimento salvato). */
static bool llm_mlx_cache_commit(mlx_array* ckpt, mlx_stream stream)
{
    (void)stream;
    mlx_array_free(*ckpt);
    *ckpt = mlx_array_new();
    return true;
}

/* Tronca la cache a `keep` posizioni lungo l'ASSE SEQUENZA (2), per KV-cache
 * [B,nh,L,D] (e stato recurrent con asse 2 = dimensione: NON trimmare quello).
 * keep<=0 -> array vuoto. Ritorna il slice in *out (nuovo array, non muta cache). */
static bool llm_mlx_cache_trim(mlx_array* out, mlx_array cache, int keep, mlx_stream stream)
{
    if (keep < 0) keep = 0;
    /* Cache VUOTA (layer GDN/linear-attention di Qwen3.5: nessun `k_full`): niente da trimmare.
     * Senza questo guard, `mlx_array_ndim` su un array vuoto lancia "expected a non-empty
     * mlx_array" (attivato quando kv_max_tokens>0, es. default `context` del consumer). */
    if (cache.ctx == NULL) return true;   /* cache vuota: `out` resta vuoto (come la cache) */
    int ndim = mlx_array_ndim(cache);
    if (ndim < 3) {
        if (keep == 0) { mlx_array_set_int(out, 0); return true; }
        return mlx_array_set(out, cache) == 0;
    }
    const int* sh = mlx_array_shape(cache);
    if (keep >= sh[2]) return mlx_array_set(out, cache) == 0;
    int* starts = (int*)malloc(sizeof(int) * (size_t)ndim);
    int* stops  = (int*)malloc(sizeof(int) * (size_t)ndim);
    int* strides = (int*)malloc(sizeof(int) * (size_t)ndim);
    if (!starts || !stops || !strides) { free(starts); free(stops); free(strides); return false; }
    for (int d = 0; d < ndim; d++) { starts[d] = 0; strides[d] = 1; stops[d] = (d == 2) ? keep : sh[d]; }
    bool ok = mlx_slice(out, cache, starts, ndim, stops, ndim, strides, ndim, stream) == 0;
    free(starts); free(stops); free(strides);
    return ok;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * LoRA contributo su MLX (fase 4): out += (x @ A @ B) * scale.
 * A [in,rank], B [rank,out], x [..,in] (ultimo asse = in). xA = x@A, xAB=xA@B,
 * contrib = xAB*scale, out = out + contrib (out e x stessa batch dim). Pura.
 */
static bool llm_mlx_lora_apply_mlx(mlx_array* out, mlx_array x, mlx_array A, mlx_array B,
                                   float scale, mlx_stream stream)
{
    mlx_array xa = mlx_array_new();
    if (mlx_matmul(&xa, x, A, stream) != 0) { mlx_array_free(xa); return false; }
    mlx_array xab = mlx_array_new();
    if (mlx_matmul(&xab, xa, B, stream) != 0) { mlx_array_free(xa); mlx_array_free(xab); return false; }
    mlx_array sc = mlx_array_new_float32(scale);
    mlx_array scaled = mlx_array_new();
    bool ok = mlx_multiply(&scaled, xab, sc, stream) == 0;
    if (ok) {
        /* Accumula sull'output esistente, oppure (output vuoto, es. contrib per-esperto
         * MoE) imposta il risultato fresco -- mlx_add su un array vuoto fallirebbe
         * ('expected a non-empty mlx_array', bug reale del per-esperto mai colpito
         * finche' non ha girato su un MoE). */
        if (out->ctx == NULL) mlx_array_set(out, scaled);
        else ok = mlx_add(out, *out, scaled, stream) == 0;
    }
    mlx_array_free(xa); mlx_array_free(xab); mlx_array_free(sc); mlx_array_free(scaled);
    return ok;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * RMLX10 Fase A (2026-09-03, vedi vault Todo/(open) - RMLX10-handoff): KV-cache
 * a BUFFER FISSO shape-statica per il decode-step.
 *
 * Prerequisito per la closure mlx_compile (Fase B): ogni passo del decode deve
 * lavorare su shape IDENTICHE a ogni step. Oggi la cache cresce via concat
 * (shape che cambia a ogni token) -> la compilazione e' impossibile. Con questi
 * due moduli il layer mantiene un buffer [1, n_kv, cap, head_dim] persistente
 * (cap = lunghezza asse 2, mai piu' piccola di cache_offset+n_tokens):
 *   - llm_mlx_kv_write_pos: aggiornamento posizionale del buffer (where con
 *     maschera di riga, copia O(cap), shape INVARIATA tra gli step);
 *   - llm_mlx_gqa_sdpa_prefix: SDPA con maschera ADDITIVA esplicita (mode "")
 *     che espone solo il prefisso valido [0, prefix_len) della cache padded.
 *     MLX mode "causal" deriverebbe kL dalla shape del buffer e farebbe vedere
 *     i token futuri (zeri) -> serve la maschera come DATO, non come shape
 *     (schema del branch window di llm_mlx_gqa_sdpa_sinks, senza la finestra).
 * Layout [B, n_heads, L, head_dim]; B=1. Pura.
 */

/* SDPA causale su prefisso di una cache PADDED: k/v hanno shape [1,nh,Nk,hd]
 * con Nk = capacita' (>= prefix_len); solo le prime `prefix_len` chiavi sono
 * valide. La maschera additiva e' 0 per j < prefix_len (query a posizione
 * assoluta prefix_len-Lq+i) e -inf oltre. Ritorna true se ok. */
static bool llm_mlx_gqa_sdpa_prefix(mlx_array* out, mlx_array q, mlx_array k, mlx_array v,
                                    float scale, int prefix_len, mlx_stream stream)
{
    int qnd = (int)mlx_array_ndim(q);
    int Lq = qnd > 2 ? (int)mlx_array_shape(q)[qnd - 2] : 1;
    int Nk = qnd > 2 ? (int)mlx_array_shape(k)[qnd - 2] : 1;
    if (prefix_len < Lq) prefix_len = Lq;
    if (prefix_len > Nk) prefix_len = Nk;
    int mask_shape[4] = {1, 1, Lq, Nk};
    float* md = (float*)malloc((size_t)Lq * (size_t)Nk * sizeof(float));
    if (md == NULL) return false;
    for (int i = 0; i < Lq; i++) {
        int qabs = prefix_len - Lq + i; /* posizione assoluta della query i */
        for (int j = 0; j < Nk; j++)
            md[i * Nk + j] = (j <= qabs) ? 0.0f : -INFINITY;
    }
    mlx_array mask = mlx_array_new_data(md, mask_shape, 4, MLX_FLOAT32);
    free(md);
    if (mlx_array_dtype(q) != MLX_FLOAT32) {
        mlx_array mb = mlx_array_new();
        bool castok = mlx_astype(&mb, mask, mlx_array_dtype(q), stream) == 0;
        mlx_array_free(mask);
        if (!castok) { mlx_array_free(mb); return false; }
        mask = mb;
    }
    mlx_array no_sinks = mlx_array_new();
    bool ok = mlx_fast_scaled_dot_product_attention(out, q, k, v, scale, "", mask, no_sinks, stream) == 0;
    mlx_array_free(mask);
    mlx_array_free(no_sinks);
    return ok;
}

/* Scrittura POSIZIONALE nel buffer KV statico:
 *   cache: [1,nkv,cap,hd] (capacita' `cap`) oppure VUOTO (ctx==NULL) = zeri.
 *   newk : [1,nkv,L,hd]  (blocco da inserire; L = n_tokens, tipicamente 1).
 *   pos  : offset assoluto del blocco (cache_offset). Richiede pos+L <= cap.
 * Ritorna in *out il buffer [1,nkv,cap,hd] con newk alle posizioni [pos,pos+L)
 * e tutto il resto invariato (o zero se cache era vuota). Shape INVARIATA tra
 * step -> compilabile. Il candidato e' concat(zeri(pos), newk, zeri(tail)),
 * la selezione e' derivata su device da arange+confronti (non host), cosi' la
 * Fase B puo' rimpiazzare `pos` costante con un input mlx_array. */
static bool llm_mlx_kv_write_pos(mlx_array* out, mlx_array cache, mlx_array newk,
                                 int pos, int cap, mlx_stream stream)
{
    bool have = (cache.ctx != NULL && mlx_array_ndim(cache) == 4);
    int nkv = (int)mlx_array_shape(newk)[1];
    int L   = (int)mlx_array_shape(newk)[2];
    int hd  = (int)mlx_array_shape(newk)[3];
    mlx_dtype dt = mlx_array_dtype(newk);
    if (pos < 0) pos = 0;
    if (pos + L > cap) L = cap - pos; /* safety clamp (mai atteso nei call-site) */
    if (L <= 0) return false;

    /* candidato: [1,nkv,cap,hd] con il blocco nel punto giusto, zeri altrove. */
    mlx_vector_array parts = mlx_vector_array_new();
    bool ok = true;
    if (pos > 0) {
        int zsh[4] = {1, nkv, pos, hd};
        mlx_array z = mlx_array_new();
        ok = ok && mlx_zeros(&z, zsh, 4, dt, stream) == 0;
        if (ok) mlx_vector_array_append_value(parts, z);
        mlx_array_free(z);
    }
    mlx_vector_array_append_value(parts, newk);
    int tail = cap - pos - L;
    if (tail > 0) {
        int zsh[4] = {1, nkv, tail, hd};
        mlx_array z = mlx_array_new();
        ok = ok && mlx_zeros(&z, zsh, 4, dt, stream) == 0;
        if (ok) mlx_vector_array_append_value(parts, z);
        mlx_array_free(z);
    }
    mlx_array cand = mlx_array_new();
    if (ok) ok = mlx_concatenate_axis(&cand, parts, 2, stream) == 0;
    mlx_vector_array_free(parts);
    if (!ok) { mlx_array_free(cand); return false; }

    if (!have) { /* cache vuota: il candidato E' gia' il risultato */
        mlx_array_set(out, cand);
        mlx_array_free(cand);
        return true;
    }

    /* selezione [1,1,cap,1]: 1 dove va il blocco (posizioni [pos,pos+L)). */
    mlx_array positions = mlx_array_new();
    int posl = pos + L;
    ok = ok && mlx_arange(&positions, 0.0, (double)cap, 1.0, MLX_INT32, stream) == 0;
    mlx_array posc = mlx_array_new_int(pos), poslc = mlx_array_new_int(posl);
    mlx_array ge = mlx_array_new(), lt = mlx_array_new(), sel = mlx_array_new();
    if (ok) ok = mlx_greater_equal(&ge, positions, posc, stream) == 0;
    if (ok) ok = mlx_less(&lt, positions, poslc, stream) == 0;
    if (ok) ok = mlx_logical_and(&sel, ge, lt, stream) == 0;
    int ax[3] = {0, 1, 3};
    mlx_array sel4 = mlx_array_new();
    if (ok) ok = mlx_expand_dims_axes(&sel4, sel, ax, 3, stream) == 0;
    bool okw = ok && mlx_where(out, sel4, cand, cache, stream) == 0;
    mlx_array_free(positions); mlx_array_free(posc); mlx_array_free(poslc);
    mlx_array_free(ge); mlx_array_free(lt); mlx_array_free(sel); mlx_array_free(sel4);
    mlx_array_free(cand);
    return okw;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * RMLX10 Fase B (2026-09-03): varianti DYNAMIC (pos come mlx_array) dei due
 * moduli sopra. Servono alla closure mlx_compile del decode-step: nella trace
 * (mlx_compile) gli input sono tracer senza valore e la cache di compile e'
 * keyed solo su shape/dtype -- un `pos` host (int C) verrebbe BAKED come
 * costante alla prima trace e ogni token successivo userebbe l'offset sbagliato.
 * Con pos ARRAY input, il valore cambia a ogni step ma la chiave (shape) resta
 * la stessa -> UN solo trace riusato. Tutto derivato su device, mai host.
 * Layout [B,n_heads,L,head_dim]; B=1; Lq=1 (decode).
 */

/* SDPA causale su prefisso di cache PADDED, prefix_len dinamico: maschera
 * additiva costruita su device da pos (array int [1]) = posizione assoluta
 * dell'ULTIMA query (decode Lq=1: query a pos, keep j <= pos). Ritorna *out
 * mask [1,1,1,Nk] gia' castata al dtype di q. Pura. */
static bool llm_mlx_gqa_sdpa_prefix_dyn_mask(mlx_array* out, mlx_array q, mlx_array pos,
                                             int Nk, mlx_stream stream)
{
    /* keep: arange(Nk) <= pos  (bool [Nk]) */
    mlx_array positions = mlx_array_new();
    bool ok = mlx_arange(&positions, 0.0, (double)Nk, 1.0, MLX_INT32, stream) == 0;
    mlx_array le = mlx_array_new();
    if (ok) ok = mlx_less_equal(&le, positions, pos, stream) == 0;
    int ax[3] = {0, 1, 3};
    mlx_array le4 = mlx_array_new();
    if (ok) ok = mlx_expand_dims_axes(&le4, le, ax, 3, stream) == 0; /* [1,1,Nk,1] */
    /* mask = where(keep, 0, -inf) [1,1,Nk,1] */
    mlx_array zero_s = mlx_array_new_float32(0.0f), ninf_s = mlx_array_new_float32(-INFINITY);
    mlx_array mraw = mlx_array_new();
    if (ok) ok = mlx_where(&mraw, le4, zero_s, ninf_s, stream) == 0;
    /* forma finale attesa da fast_sdpa: [1,1,Lq,Nk]; Lq=1 -> trasponi */
    mlx_array m = mlx_array_new();
    const int perm[4] = {0, 1, 3, 2};
    if (ok) ok = mlx_transpose_axes(&m, mraw, perm, 4, stream) == 0; /* [1,1,1,Nk] */
    if (ok && mlx_array_dtype(q) != MLX_FLOAT32) {
        mlx_array mb = mlx_array_new();
        ok = mlx_astype(&mb, m, mlx_array_dtype(q), stream) == 0;
        mlx_array_free(m);
        if (ok) m = mb;
    }
    if (ok) mlx_array_set(out, m);
    mlx_array_free(m);
    mlx_array_free(positions); mlx_array_free(le); mlx_array_free(le4);
    mlx_array_free(zero_s); mlx_array_free(ninf_s); mlx_array_free(mraw);
    return ok;
}

/* SDPA causale su prefisso con pos dinamica. k/v: [1,nh,Nk,hd] PADDED (cap);
 * prefix_len = pos+1 derivato su device. Ritorna true se ok. */
static bool llm_mlx_gqa_sdpa_prefix_dyn(mlx_array* out, mlx_array q, mlx_array k, mlx_array v,
                                        mlx_array pos, float scale, mlx_stream stream)
{
    int knd = (int)mlx_array_ndim(k);
    int Nk = knd > 2 ? (int)mlx_array_shape(k)[knd - 2] : 1;
    mlx_array mask = mlx_array_new();
    bool ok = llm_mlx_gqa_sdpa_prefix_dyn_mask(&mask, q, pos, Nk, stream);
    mlx_array no_sinks = mlx_array_new();
    if (ok) ok = mlx_fast_scaled_dot_product_attention(out, q, k, v, scale, "", mask, no_sinks, stream) == 0;
    mlx_array_free(mask);
    mlx_array_free(no_sinks);
    return ok;
}

/* Scrittura posizionale DYNAMIC: pos = array int [1] (input tracer-compilabile).
 * cache: [1,nkv,cap,hd] (gia' alla capacita'); newk [1,nkv,L,hd] (L=1 decode).
 * Risultato [1,nkv,cap,hd] = cache con newk alle posizioni [pos,pos+L) (dove,
 * shape invariata). Nessun host-int: selezione derivata da arange+confronti. */
static bool llm_mlx_kv_write_pos_dyn(mlx_array* out, mlx_array cache, mlx_array newk,
                                     mlx_array pos, int cap, mlx_stream stream)
{
    int nkv = (int)mlx_array_shape(newk)[1];
    int L   = (int)mlx_array_shape(newk)[2];
    int hd  = (int)mlx_array_shape(newk)[3];
    if (L != 1) return false; /* decode step L=1 (niente block write) */
    /* selezione [1,1,cap,1]: positions in [pos, pos+1) */
    mlx_array positions = mlx_array_new();
    bool ok = mlx_arange(&positions, 0.0, (double)cap, 1.0, MLX_INT32, stream) == 0;
    mlx_array pos1 = mlx_array_new();
    mlx_array one_a = mlx_array_new_int(1);
    if (ok) ok = mlx_add(&pos1, pos, one_a, stream) == 0; /* pos+1 */
    mlx_array_free(one_a);
    mlx_array ge = mlx_array_new(), lt = mlx_array_new(), sel = mlx_array_new();
    if (ok) ok = mlx_greater_equal(&ge, positions, pos, stream) == 0;
    if (ok) ok = mlx_less(&lt, positions, pos1, stream) == 0;
    if (ok) ok = mlx_logical_and(&sel, ge, lt, stream) == 0;
    int ax[3] = {0, 1, 3};
    mlx_array sel4 = mlx_array_new();
    if (ok) ok = mlx_expand_dims_axes(&sel4, sel, ax, 3, stream) == 0; /* [1,1,cap,1] */
    /* newk_bcast: [1,nkv,1,hd] -> [1,nkv,cap,hd] (asse 2 broadcastabile da 1) */
    int bsh[4] = {1, nkv, cap, hd};
    mlx_array nb = mlx_array_new();
    if (ok) ok = mlx_broadcast_to(&nb, newk, bsh, 4, stream) == 0;
    bool okw = ok && mlx_where(out, sel4, nb, cache, stream) == 0;
    mlx_array_free(positions); mlx_array_free(pos1); mlx_array_free(ge);
    mlx_array_free(lt); mlx_array_free(sel); mlx_array_free(sel4); mlx_array_free(nb);
    return okw;
}

#endif /* LLM_MLX_OPS_H */
