/* llm_mlx_decode_audit.h — contatori STRUTTURALI del decode (P0-ter step 1, 2026-09-16).
 *
 * PERCHE' ESISTE. Il prefill e' gia' competitivo/superiore; il gap residuo vs mlx_lm e' nel
 * DECODE (M=1). La dottrina accumulata e' "audit-first": silica-mlx (engine MLX-native, ~2616
 * test) riporta 17 kernel Metal custom -> NESSUN guadagno E2E load-bearing, e a B=1 lo step e'
 * bandwidth-capped (dispatch ~0.3%) -> scrivere kernel prima di sapere DOVE si perde e' la
 * strada sbagliata. Il gate da misurare e' `GPU_ACTIVE/WALL` per token + command buffer /
 * sincronizzazioni / KV rebuild per token.
 *
 * PERCHE' QUESTI CONTATORI SONO UTILI ANCHE A MACCHINA CARICA. I TEMPI sono inquinati dal
 * carico (lezione gpt-oss), ma i CONTEGGI strutturali (quante eval bloccanti, quante
 * submission async, quale path KV, quanti byte di KV) sono DETERMINISTICI: valgono a
 * qualsiasi load. Sono il primo passo dell'audit e non richiedono una finestra pulita.
 *
 * LLM_DECODE_AUDIT=1 abilita. Costo zero quando spento (ogni hook esce subito).
 *
 * Semantica dei path KV (classe di bug "KV rebuild per step", gia' costata 4,6x una volta:
 * append-in-place 7c8618f70, V2-Lite ~21 -> ~98 tok/s):
 *   kv_ok    = append in-place riuscito (O(1) per token)             -> atteso ~ovunque
 *   kv_grow  = migrazione a un buffer piu' grande (O(off) di copia)  -> raro, ammortizzato
 *   kv_fail  = append NON applicabile -> il chiamante ricade sul concatenate O(cap)
 */
#ifndef LLM_MLX_DECODE_AUDIT_H
#define LLM_MLX_DECODE_AUDIT_H

#include <stdio.h>
#include "llm_engine_opts.h"   /* opt_env: registro */
#include <stdlib.h>
#include <string.h>
#include "mlx/c/mlx.h"

typedef struct {
    long tokens;        /* passi di decode (token generati) */
    long fwd;           /* forward_one_token / _dev */
    long final_steps;   /* final_step / _greedy_dev */
    long readbacks;     /* eval BLOCCANTI nel loop (lettura del token) */
    long async_submits; /* mlx_async_eval accodati nel loop */
    long clear_cache;   /* mlx_clear_cache() nel loop (ogni 256, come mlx_lm) */
    long kv_ok;         /* append KV in-place riuscito */
    long kv_grow;       /* append con migrazione di capacita' */
    long kv_fail;       /* append non applicabile (fallback del chiamante) */
    long kv_metal;      /* passi serviti dal bridge KV Metal (LLMMETAL_KVMETAL) */
    long kv_seg;        /* passi serviti dal path a segmenti */
    long kv_new_bytes;  /* byte scritti nella KV per i soli token nuovi */
    long weight_bytes;        /* byte di PESI ATTIVI letti per token (MoE: solo top_k esperti) */
    long weight_bytes_total;  /* byte di TUTTI i pesi (MoE: tutti gli esperti) - informativo */
    long kv_copy_bytes; /* byte RICOPIATI dalle migrazioni (costo O(off)) */
} LlmDecodeAudit;

static LlmDecodeAudit g_da;
static long g_da_state = -1;

static inline int llm_da_enabled(void) {
    if (g_da_state < 0) g_da_state = (llm_ai_mlx_opt_env("LLM_DECODE_AUDIT") != NULL) ? 1 : 0;
    return g_da_state == 1;
}
static inline void llm_da_reset(void) { memset(&g_da, 0, sizeof(g_da)); }
static inline void llm_da_set_weight_bytes(long wb) { g_da.weight_bytes = wb; }
static inline void llm_da_set_weight_bytes_total(long wb) { g_da.weight_bytes_total = wb; }

/* Byte per elemento del dtype della cache KV (f16/bf16 = 2, altro = 4). */
static inline int llm_da_esz(mlx_dtype dt) {
    return (dt == MLX_FLOAT16 || dt == MLX_BFLOAT16) ? 2 : 4;
}

/* Chiamata dall'append in-place quando va a buon fine (grow=1 se ha migrato). */
static inline void llm_da_note_kv_append(int nkv, int n_tokens, int kd, int vd, int esz, int grew, int old_off) {
    if (!llm_da_enabled()) return;
    if (grew) {
        g_da.kv_grow++;
        g_da.kv_copy_bytes += (long)nkv * (long)old_off * (long)(kd + vd) * esz;
    } else {
        g_da.kv_ok++;
    }
    g_da.kv_new_bytes += (long)nkv * (long)n_tokens * (long)(kd + vd) * esz;
}

/* Report di fine run. wall_s = secondi del loop di decode (TIMING: inquinato dal carico). */
static inline void llm_da_report(const char* tag, double wall_s) {
    if (!llm_da_enabled() || g_da.tokens <= 0) return;
    const double t = (double)g_da.tokens;
    fprintf(stderr,
        "[LLM_DECODE_AUDIT %s] token=%ld wall=%.2fs\n"
        "  per-token: fwd=%.2f final=%.2f readback(blocc.)=%.2f async_submit=%.2f clear_cache=%.3f\n"
        "  KV path: inplace=%ld grow=%ld FAIL(fallback)=%ld | metal=%ld seg=%ld\n"
        "  KV byte: nuovi/token=%.1f KB  ricopiati(migrazioni)=%.2f MB\n",
        tag, g_da.tokens, wall_s,
        (double)g_da.fwd / t, (double)g_da.final_steps / t, (double)g_da.readbacks / t,
        (double)g_da.async_submits / t, (double)g_da.clear_cache / t,
        g_da.kv_ok, g_da.kv_grow, g_da.kv_fail, g_da.kv_metal, g_da.kv_seg,
        (double)g_da.kv_new_bytes / t / 1024.0, (double)g_da.kv_copy_bytes / 1048576.0);
    /* M1 (macuda): BYTES/token e GB/s raggiunti -> classifica memory- vs dispatch/latency-bound.
     * Deterministico tranne il wall: GB/s ~ tetto della banda -> memory-bound; GB/s << tetto -> gap. */
    if (g_da.weight_bytes > 0 && wall_s > 0.0) {
        const double bpt = (double)g_da.weight_bytes + (double)g_da.kv_new_bytes / t;
        const double gbs = bpt / (wall_s * 1e9 / t);
        fprintf(stderr, "  BYTE/token: pesi_attivi=%.1f MB (totali %.1f MB) + kv=%.0f KB = %.1f MB  ->  %.0f GB/s  (%.0f%% di 550 GB/s)\n",
                (double)g_da.weight_bytes / 1048576.0,
                (double)(g_da.weight_bytes_total > 0 ? g_da.weight_bytes_total : g_da.weight_bytes) / 1048576.0,
                (double)g_da.kv_new_bytes / t / 1024.0,
                bpt / 1048576.0, gbs, 100.0 * gbs / 550.0);
    }
}

#endif /* LLM_MLX_DECODE_AUDIT_H */
