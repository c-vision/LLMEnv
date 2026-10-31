/* llm_mlx_qos.h — politica del profilo QoS adattivo (I12) + backoff TERMICO (F1, 2026-10-31).
 *
 * Perche' un header a parte: la politica e' una funzione PURA (stato termico -> divisore del chunk
 * di prefill) cosi' si puo' testare senza caricare un modello. Il profilo si attiva solo con
 * l'opzione di registro `qos=adaptive|utility`; il backoff termico e' governato da:
 *   qos_thermal      = 1 (default) | 0   (abilita/disabilita il backoff)
 *   qos_thermal_sim  = -1 (default, legge lo stato reale) | 0..3 (forza, per test deterministici)
 *
 * Nessuna variabile d'ambiente: sono voci del registro opzioni.
 *
 * Stati termici (stessa scala di `com.apple.system.thermalpressurelevel`):
 *   0 nominal · 1 fair · 2 serious · 3 critical · -1 sconosciuto
 */
#ifndef LLM_MLX_QOS_H
#define LLM_MLX_QOS_H

/* Divisore del chunk di prefill per stato termico. Il chunk si riduce quando la macchina scalda:
 * burst GPU piu' piccoli, piu' spazio alle altre app. Valori: 4 (nominale/fair, comportamento I12),
 * 8 (serious), 16 (critical). Con stato sconosciuto (-1) si resta a 4: nessun peggioramento. */
static inline int llm_mlx_qos_chunk_divisor(int thermal_state) {
    if (thermal_state >= 3) return 16;
    if (thermal_state == 2) return 8;
    return 4;
}

/* Il QoS del thread: UTILITY gia' dal profilo adattivo; sotto stress termico CRITICAL scende a
 * BACKGROUND (cede il turno alle altre app). 1 = critical, 0 = altrimenti. */
static inline int llm_mlx_qos_wants_background(int thermal_state) {
    return thermal_state >= 3;
}

/* Soglia minima sotto la quale non si scende: un chunk di 128 token resta efficiente e sotto
 * quel valore il costo per chunk (sync + overhead) domina. */
#define LLM_MLX_QOS_CHUNK_FLOOR 128

static inline int llm_mlx_qos_apply_thermal(int chunk, int thermal_state) {
    int d = llm_mlx_qos_chunk_divisor(thermal_state);
    int c = chunk / d;
    return (c < LLM_MLX_QOS_CHUNK_FLOOR) ? LLM_MLX_QOS_CHUNK_FLOOR : c;
}

#endif /* LLM_MLX_QOS_H */
