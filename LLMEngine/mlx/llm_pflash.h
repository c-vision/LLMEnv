/* llm_pflash.h — PFlash: compressione del prompt a contesto lungo (TTFT prefill).
 *
 * Port in C puro del metodo di Rapid-MLX #287 (`vllm_mlx/pflash.py`,
 * Apache-2.0; design originale + fork di riferimento di @michaelasper sul
 * branch `pflash-qwen36-ttft`; discussione issue #287). Nessun codice MLX
 * copiato: qui si reimplementa la SOLA logica di scoring/selezione dei
 * blocchi, che e' deterministica e non tocca il device.
 *
 * Cosa fa (e cosa NON e'):
 *  - NON e' un'ottimizzazione dell'engine: decide QUANTI token entrano nel
 *    prefill. Tiene testa (sink), coda (tail) e una frazione `keep_ratio`
 *    dei blocchi centrali scelti per punteggio; il resto viene SCARTATO.
 *    Il prefill elabora meno token => il TTFT crolla (riferimento Rapid-MLX:
 *    126,6 s -> 11,2 s, ~11x, su un prompt da 135k token Qwen3.6-35B-A3B-4bit).
 *  - E' LOSSY per disegno (recall sul mezzo del prompt): **default AUTO** (promosso
 *    2026-09-18 dopo il gate di qualita': needle recall 4/4 posizioni 5/25/50/75% a ~20k,
 *    control senza fabbricazione), opt-out via `LLM_PFLASH=off`, kill-switch, fail-closed
 *    su tool/integrita'. Sotto `threshold` (32768 token) non comprime: prompt corti invariati.
 *
 * Semantica delle posizioni (decisione di design, port fedele del riferimento):
 *  - il chiamante riceve una sequenza COMPATTATA (i token tenuti, in ordine
 *    originale, senza buchi): il prefill la processa normalmente, quindi le
 *    posizioni RoPE/KV sono quelle della sequenza compressa.
 *  - La sequenza compressa e' una "positional fiction": la posizione i NON
 *    corrisponde alla posizione i del prompt originale. Conseguenza
 *    obbligatoria: una richiesta compressa NON deve usare/salvare la
 *    prefix-cache (riusare una KV calcolata sul prefisso non compresso
 *    inietterebbe stato con posizioni sbagliate in una richiesta successiva
 *    che condivide i token di testa). Il hook in llmengine_mlx.c salta il
 *    percorso di prefix reuse quando PFlash e' attivo su una sessione fredda.
 *
 * Scoring (identico al riferimento, `_score_middle_blocks`):
 *  - per ogni blocco del tratto centrale [sink_end, tail_start):
 *      overlap = sum(query_counts[t] / counts[t])          (token che
 *                riappaiono nella finestra di coda = "query window")
 *      rarity  = sum(1 / counts[t]) / len(block)           (token rari)
 *      recency = (block_end - start) / span                (leggero bias)
 *      stride_bonus = 0.25 se block_index % stride_blocks == 0
 *      score = 4.0*overlap + rarity + 0.05*recency + stride_bonus
 *  - ordinamento deterministico per (-score, start); si riempie il budget
 *    residuo prendendo i token dall'inizio dei blocchi migliori.
 *
 * Header-only di proposito: `native/llmengine_mlx/CMakeLists.txt` elenca i
 * sorgenti uno per uno e aggiungerne uno e' una modifica all'infrastruttura
 * di build (AGENTS: serve permesso esplicito). Cosi' il modulo e' incluso da
 * `llmengine_mlx.c` (hook di prefill) e dal driver `llm_pflash_test.c`
 * (self-test load-free, nessun modello) senza toccare il build system.
 */

#ifndef LLM_PFLASH_H
#define LLM_PFLASH_H

#include <stdint.h>
#include "llm_engine_opts.h"   /* opt_env: registro */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* Modalita' (stessa tassonomia del riferimento: off | auto | always). */
typedef enum {
    LLM_PFLASH_MODE_OFF = 0,
    LLM_PFLASH_MODE_AUTO = 1,   /* comprime solo se n_tokens >= threshold */
    LLM_PFLASH_MODE_ALWAYS = 2  /* comprime sempre (soggetto ai gate) */
} LlmPflashMode;

/* Default = profilo VALIDATO del riferimento (PR #649: keep_ratio 0.20,
 * TTFT 3,87x-8,5x con needle recall 5/5; non i 0.10 del fork). */
typedef struct {
    int mode;              /* LlmPflashMode, default AUTO (default ON; LLM_PFLASH=off per disattivare) */
    int threshold;         /* default 32768 */
    double keep_ratio;     /* default 0.20 (0 < r <= 1) */
    int min_keep_tokens;   /* default 2048 (floor del budget) */
    int sink_tokens;       /* default 256 (testa preservata) */
    int tail_tokens;       /* default 2048 (coda preservata: c'e' la query) */
    int block_size;        /* default 128 */
    int query_window;      /* default 512 (finestra di coda per l'overlap) */
    int stride_blocks;     /* default 8 (0 = disabilitato) */
    int skip_when_tools;   /* default 1: fail-closed sui prompt con tool */
} LlmPflashConfig;

/* Esito: `reason` e' una stringa costante (telemetria/log), i conteggi sono
 * del prompt logico (original_tokens) e della sequenza compressa. */
typedef struct {
    int compressed;            /* 1 = out contiene la sequenza compressa */
    const char* reason;        /* vedi LLM_PFLASH_RSN_* sotto */
    int original_tokens;
    int kept_tokens;
    int middle_tokens_kept;    /* token del tratto centrale sopravvissuti */
} LlmPflashResult;

/* Ragioni (identiche al riferimento, per confrontabilita' della telemetria). */
#define LLM_PFLASH_RSN_COMPRESSED "compressed"
#define LLM_PFLASH_RSN_PROTECTED "protected_prompt"
#define LLM_PFLASH_RSN_TOOLS "tools"
#define LLM_PFLASH_RSN_EMPTY "empty"
#define LLM_PFLASH_RSN_OFF "off"
#define LLM_PFLASH_RSN_THRESHOLD "threshold"
#define LLM_PFLASH_RSN_BUDGET "budget"
#define LLM_PFLASH_RSN_MIDDLE "insufficient_middle_budget"
#define LLM_PFLASH_RSN_ERROR "error"

/* ---------------------------------------------------------------- config */

static inline void llm_pflash_config_defaults(LlmPflashConfig* c) {
    /* PROMOSSO default ON (AUTO) il 2026-09-18: gate di qualita' superato. Opt-out LLM_PFLASH=off. */
    c->mode = LLM_PFLASH_MODE_AUTO;
    c->threshold = 32768;
    c->keep_ratio = 0.20;
    c->min_keep_tokens = 2048;
    c->sink_tokens = 256;
    c->tail_tokens = 2048;
    c->block_size = 128;
    c->query_window = 512;
    c->stride_blocks = 8;
    c->skip_when_tools = 1;
}

static inline int llm_pflash_env_int(const char* name, int dflt) {
    const char* v = getenv(name);
    if (v == NULL || *v == '\0') return dflt;
    return atoi(v);
}

static inline double llm_pflash_env_double(const char* name, double dflt) {
    const char* v = getenv(name);
    if (v == NULL || *v == '\0') return dflt;
    return atof(v);
}

/* Legge la configurazione dall'ambiente. Ritorna 1 se PFlash e' ATTIVO
 * (mode != OFF), 0 altrimenti. Variabili:
 *   LLM_PFLASH=off|0 (kill-switch) | auto (DEFAULT se assente) | always|1|on
 *   LLM_PFLASH_THRESHOLD, LLM_PFLASH_KEEP_RATIO, LLM_PFLASH_MIN_KEEP_TOKENS,
 *   LLM_PFLASH_SINK_TOKENS, LLM_PFLASH_TAIL_TOKENS, LLM_PFLASH_BLOCK_SIZE,
 *   LLM_PFLASH_QUERY_WINDOW, LLM_PFLASH_STRIDE_BLOCKS,
 *   LLM_PFLASH_INCLUDE_TOOLS=1 (disattiva il fail-closed sui tool)
 * I valori fuori range vengono clampati ai default del profilo (mai un
 * comportamento silenzioso diverso da quello dichiarato: si stampa un avviso). */
static inline int llm_pflash_config_from_env(LlmPflashConfig* c) {
    llm_pflash_config_defaults(c);
    const char* m = llm_ai_mlx_opt_env("LLM_PFLASH");
    /* Default PROMOSSO a AUTO il 2026-09-18: env ASSENTE => AUTO col profilo validato.
     * Il kill-switch resta LLM_PFLASH=off|0 (unica via per disattivare del tutto). */
    if (m == NULL || *m == '\0') { /* tiene il default AUTO */ }
    else if (strcmp(m, "0") == 0 || strcmp(m, "off") == 0) return 0;
    else if (strcmp(m, "auto") == 0) c->mode = LLM_PFLASH_MODE_AUTO;
    else if (strcmp(m, "1") == 0 || strcmp(m, "on") == 0 || strcmp(m, "always") == 0)
        c->mode = LLM_PFLASH_MODE_ALWAYS;
    else {
        fprintf(stderr, "[pflash] LLM_PFLASH='%s' non riconosciuto (off|auto|always) -> AUTO\n", m);
    }
    c->threshold = llm_pflash_env_int("LLM_PFLASH_THRESHOLD", c->threshold);
    c->keep_ratio = llm_pflash_env_double("LLM_PFLASH_KEEP_RATIO", c->keep_ratio);
    c->min_keep_tokens = llm_pflash_env_int("LLM_PFLASH_MIN_KEEP_TOKENS", c->min_keep_tokens);
    c->sink_tokens = llm_pflash_env_int("LLM_PFLASH_SINK_TOKENS", c->sink_tokens);
    c->tail_tokens = llm_pflash_env_int("LLM_PFLASH_TAIL_TOKENS", c->tail_tokens);
    c->block_size = llm_pflash_env_int("LLM_PFLASH_BLOCK_SIZE", c->block_size);
    c->query_window = llm_pflash_env_int("LLM_PFLASH_QUERY_WINDOW", c->query_window);
    c->stride_blocks = llm_pflash_env_int("LLM_PFLASH_STRIDE_BLOCKS", c->stride_blocks);
    if (llm_ai_mlx_opt_env("LLM_PFLASH_INCLUDE_TOOLS") != NULL) c->skip_when_tools = 0;
    if (c->threshold < 0) c->threshold = 0;
    if (!(c->keep_ratio > 0.0 && c->keep_ratio <= 1.0)) c->keep_ratio = 0.20;
    if (c->min_keep_tokens < 0) c->min_keep_tokens = 0;
    if (c->sink_tokens < 0) c->sink_tokens = 0;
    if (c->tail_tokens < 0) c->tail_tokens = 0;
    if (c->block_size <= 0) c->block_size = 128;
    if (c->query_window <= 0) c->query_window = 512;
    if (c->stride_blocks < 0) c->stride_blocks = 0;
    return 1;
}

/* Gate operativo: la compressione va TENTATA per questo prompt? (mode off ->
 * 0; auto -> solo sopra soglia; always -> sempre). Il gate NON garantisce che
 * la compressione avvenga: i gate strutturali (tool/integrita'/budget) e il
 * caso "insufficient_middle_budget" possono comunque rifiutarla. */
static inline int llm_pflash_should_attempt(const LlmPflashConfig* c, int n_tokens) {
    if (c->mode == LLM_PFLASH_MODE_OFF) return 0;
    if (c->mode == LLM_PFLASH_MODE_AUTO && n_tokens < c->threshold) return 0;
    return 1;
}

/* ---------------------------------------------------- contatore token O(1) */

/* Mappa aperta int32 -> int32 (solo lookup/update: nessuna iterazione, quindi
 * deterministica per costruzione). Cap = potenza di 2 >= 2*n. */
typedef struct {
    int32_t* keys;
    int32_t* vals;
    int cap;   /* potenza di 2 */
    int mask;
} LlmPflashCounts;

static inline uint32_t llm_pflash_hash32(int32_t k) {
    return (uint32_t)k * 2654435761u;
}

static inline int llm_pflash_counts_init(LlmPflashCounts* m, int n) {
    int cap = 16;
    while (cap < 2 * n && cap < (1 << 30)) cap <<= 1;
    m->keys = (int32_t*)malloc((size_t)cap * sizeof(int32_t));
    m->vals = (int32_t*)calloc((size_t)cap, sizeof(int32_t));
    if (m->keys == NULL || m->vals == NULL) {
        free(m->keys); free(m->vals);
        m->keys = NULL; m->vals = NULL; m->cap = 0; m->mask = 0;
        return 0;
    }
    for (int i = 0; i < cap; i++) m->keys[i] = -1;
    m->cap = cap;
    m->mask = cap - 1;
    return 1;
}

static inline void llm_pflash_counts_free(LlmPflashCounts* m) {
    free(m->keys); free(m->vals);
    m->keys = NULL; m->vals = NULL; m->cap = 0; m->mask = 0;
}

static inline void llm_pflash_counts_add(LlmPflashCounts* m, int32_t k) {
    uint32_t i = llm_pflash_hash32(k) & (uint32_t)m->mask;
    while (m->keys[i] != -1) {
        if (m->keys[i] == k) { m->vals[i]++; return; }
        i = (i + 1) & (uint32_t)m->mask;
    }
    m->keys[i] = k;
    m->vals[i] = 1;
}

/* Ritorna 0 se il token non e' mai stato contato (mai negativo: i conteggi
 * partono da 1). */
static inline int32_t llm_pflash_counts_get(const LlmPflashCounts* m, int32_t k) {
    uint32_t i = llm_pflash_hash32(k) & (uint32_t)m->mask;
    while (m->keys[i] != -1) {
        if (m->keys[i] == k) return m->vals[i];
        i = (i + 1) & (uint32_t)m->mask;
    }
    return 0;
}

/* ------------------------------------------------------- scoring blocchi */

typedef struct {
    int start;
    int end;
    double score;
} LlmPflashBlock;

static inline int llm_pflash_block_cmp(const void* pa, const void* pb) {
    const LlmPflashBlock* a = (const LlmPflashBlock*)pa;
    const LlmPflashBlock* b = (const LlmPflashBlock*)pb;
    if (a->score > b->score) return -1;
    if (a->score < b->score) return 1;
    if (a->start < b->start) return -1;
    if (a->start > b->start) return 1;
    return 0;
}

static inline double llm_pflash_clamp(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline int llm_pflash_keep_budget(int n_tokens, const LlmPflashConfig* c) {
    int ratio_budget = (int)ceil((double)n_tokens * llm_pflash_clamp(c->keep_ratio, 0.0, 1.0));
    int b = ratio_budget > c->min_keep_tokens ? ratio_budget : c->min_keep_tokens;
    if (b < 1) b = 1;
    if (b > n_tokens) b = n_tokens;
    return b;
}

/* ------------------------------------------------------------ compressore */

/* Comprime `tokens` (n) secondo cfg e scrive la sequenza risultante in `out`
 * (capacita' `out_cap`, richiesta >= n). Ritorna 1 se `res` e' valido, 0 su
 * errore di allocazione/capacita' (il chiamante deve usare il prompt
 * ORIGINALE: fail-open sul percorso caldo, mai un prompt troncato a meta').
 *
 * Gate strutturali (in quest'ordine, come il riferimento):
 *   requires_prompt_integrity -> "protected_prompt" (mai comprimere)
 *   has_tools && skip_when_tools -> "tools"        (fail-closed tool-call)
 *   n == 0 -> "empty" · mode off -> "off" · auto sotto soglia -> "threshold"
 *   budget >= n -> "budget" · endpoints >= budget -> "insufficient_middle_budget"
 * In tutti i casi "non compresso" l'output e' la COPIA ESATTA dell'input. */
static inline int llm_pflash_compress(const int32_t* tokens, int n,
                                      const LlmPflashConfig* cfg,
                                      int has_tools, int requires_prompt_integrity,
                                      int32_t* out, int out_cap,
                                      LlmPflashResult* res) {
    if (tokens == NULL || cfg == NULL || out == NULL || res == NULL) return 0;
    if (out_cap < n || n < 0) return 0;
    res->compressed = 0;
    res->reason = LLM_PFLASH_RSN_ERROR;
    res->original_tokens = n > 0 ? n : 0;
    res->kept_tokens = n > 0 ? n : 0;
    res->middle_tokens_kept = 0;
    if (n > 0) memcpy(out, tokens, (size_t)n * sizeof(int32_t));

    /* Gate strutturali: proprieta' della richiesta, non dell'engine. */
    if (requires_prompt_integrity) { res->reason = LLM_PFLASH_RSN_PROTECTED; return 1; }
    if (has_tools && cfg->skip_when_tools) { res->reason = LLM_PFLASH_RSN_TOOLS; return 1; }
    if (n == 0) { res->reason = LLM_PFLASH_RSN_EMPTY; return 1; }
    if (cfg->mode == LLM_PFLASH_MODE_OFF) { res->reason = LLM_PFLASH_RSN_OFF; return 1; }
    if (cfg->mode == LLM_PFLASH_MODE_AUTO && n < cfg->threshold) {
        res->reason = LLM_PFLASH_RSN_THRESHOLD;
        return 1;
    }

    int block_size = cfg->block_size > 0 ? cfg->block_size : 1;
    int keep_budget = llm_pflash_keep_budget(n, cfg);
    if (keep_budget >= n) { res->reason = LLM_PFLASH_RSN_BUDGET; return 1; }

    int sink_end = cfg->sink_tokens > 0 ? cfg->sink_tokens : 0;
    if (sink_end > n) sink_end = n;
    int tail = cfg->tail_tokens > 0 ? cfg->tail_tokens : 0;
    int tail_start = n - tail;
    if (tail_start < sink_end) tail_start = sink_end;

    /* Posizioni preservate = testa + coda. */
    unsigned char* keep = (unsigned char*)calloc((size_t)n, 1);
    if (keep == NULL) return 0;
    int kept_count = 0;
    for (int i = 0; i < sink_end; i++) { keep[i] = 1; kept_count++; }
    for (int i = tail_start; i < n; i++) { keep[i] = 1; kept_count++; }

    int middle_span = tail_start - sink_end;
    int remaining_budget = keep_budget - kept_count;
    if (middle_span > 0 && remaining_budget <= 0) {
        /* Gli endpoint consumano gia' tutto il budget: comprimere qui
         * cancellerebbe TUTTO il corpo (e' il regime in cui cadono le
         * conversazioni agentiche di 2-11k token coi default). Rifiuta. */
        free(keep);
        res->reason = LLM_PFLASH_RSN_MIDDLE;
        return 1;
    }

    int middle_selected = 0;
    if (remaining_budget > 0) {
        /* Conteggi globali + conteggi della finestra di coda (query). */
        LlmPflashCounts counts, qcounts;
        if (!llm_pflash_counts_init(&counts, n)) { free(keep); return 0; }
        for (int i = 0; i < n; i++) llm_pflash_counts_add(&counts, tokens[i]);
        int qwin = cfg->query_window > 0 ? cfg->query_window : 1;
        int qstart = n - qwin; if (qstart < 0) qstart = 0;
        int qn = n - qstart;
        if (!llm_pflash_counts_init(&qcounts, qn > 0 ? qn : 1)) {
            llm_pflash_counts_free(&counts); free(keep); return 0;
        }
        for (int i = qstart; i < n; i++) llm_pflash_counts_add(&qcounts, tokens[i]);

        int max_blocks = (middle_span + block_size - 1) / block_size;
        LlmPflashBlock* blocks = (LlmPflashBlock*)malloc((size_t)max_blocks * sizeof(LlmPflashBlock));
        if (blocks == NULL) {
            llm_pflash_counts_free(&counts); llm_pflash_counts_free(&qcounts);
            free(keep); return 0;
        }
        double span = middle_span > 0 ? (double)middle_span : 1.0;
        int nb = 0;
        int stride = cfg->stride_blocks > 0 ? cfg->stride_blocks : 0;
        for (int bs = sink_end; bs < tail_start; bs += block_size) {
            int be = bs + block_size; if (be > tail_start) be = tail_start;
            double overlap = 0.0, rarity = 0.0;
            for (int t = bs; t < be; t++) {
                int32_t tok = tokens[t];
                int32_t g = llm_pflash_counts_get(&counts, tok);
                if (g <= 0) g = 1;
                int32_t q = llm_pflash_counts_get(&qcounts, tok);
                overlap += (double)q / (double)g;
                rarity += 1.0 / (double)g;
            }
            rarity /= (double)(be - bs);
            double recency = ((double)be - (double)sink_end) / span;
            double stride_bonus = (stride > 0 && (nb % stride) == 0) ? 0.25 : 0.0;
            blocks[nb].start = bs;
            blocks[nb].end = be;
            blocks[nb].score = 4.0 * overlap + rarity + 0.05 * recency + stride_bonus;
            nb++;
        }
        qsort(blocks, (size_t)nb, sizeof(LlmPflashBlock), llm_pflash_block_cmp);

        for (int i = 0; i < nb; i++) {
            int block_len = blocks[i].end - blocks[i].start;
            int slots = remaining_budget - middle_selected;
            if (slots <= 0) break;
            int take = block_len < slots ? block_len : slots;
            for (int t = blocks[i].start; t < blocks[i].start + take; t++) {
                if (!keep[t]) { keep[t] = 1; kept_count++; }
            }
            middle_selected += take;
            if (middle_selected >= remaining_budget) break;
        }
        free(blocks);
        llm_pflash_counts_free(&counts);
        llm_pflash_counts_free(&qcounts);
    }

    if (kept_count >= n) {
        free(keep);
        res->reason = LLM_PFLASH_RSN_BUDGET;
        return 1;
    }
    int w = 0;
    for (int i = 0; i < n; i++) if (keep[i]) out[w++] = tokens[i];
    free(keep);
    res->compressed = 1;
    res->reason = LLM_PFLASH_RSN_COMPRESSED;
    res->kept_tokens = w;
    res->middle_tokens_kept = middle_selected;
    return 1;
}

#endif /* LLM_PFLASH_H */
