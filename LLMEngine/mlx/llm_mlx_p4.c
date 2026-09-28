/* llm_mlx_p4.c — dominio P4: MEMORY PLAN + ADMISSION. Estratto da llmengine_mlx.c (split fase 2), VERBATIM, funzioni da static a extern. */
#include "llm_mlx_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <sys/types.h>

uint64_t llm_p4_sysctl_u64(const char* name) {
    uint64_t v = 0; size_t n = sizeof(v);
    if (sysctlbyname(name, &v, &n, NULL, 0) != 0) return 0;
    return v;
}
uint64_t llm_p4_kv_per_token(const ModelConfig* c) {
    if (!c) return 0;
    const uint64_t per = (uint64_t)c->n_kv_heads * (uint64_t)c->head_dim * 2ull * 2ull; /* K+V f16 */
    uint64_t tot = 0;
    for (int l = 0; l < c->n_layers; l++) {
        if (c->is_linear_layer && c->is_linear_layer[l]) continue;
        tot += per;
    }
    return tot;
}
uint64_t llm_p4_kv_state(const ModelConfig* c) {
    if (!c || !c->is_linear_layer) return 0;
    const uint64_t hv = (uint64_t)(c->linear_num_value_heads > 0 ? c->linear_num_value_heads : c->n_kv_heads);
    const uint64_t dv = (uint64_t)(c->linear_value_head_dim > 0 ? c->linear_value_head_dim : c->head_dim);
    const uint64_t dk = (uint64_t)(c->linear_key_head_dim > 0 ? c->linear_key_head_dim : c->head_dim);
    const uint64_t ck = (uint64_t)(c->linear_conv_kernel_dim > 0 ? c->linear_conv_kernel_dim : 4);
    uint64_t tot = 0;
    for (int l = 0; l < c->n_layers; l++) {
        if (!c->is_linear_layer[l]) continue;
        tot += hv * dv * dk * 2ull;                       /* stato S */
        tot += ck * (2ull * dk + dv) * 2ull;              /* conv_state (q+k+v) */
    }
    return tot;
}
/* Calcola (una volta) e stampa il MEMORY PLAN. */
void llm_p4_plan_init(LlmMlxModel* m) {
    if (!m || m->p4_plan_done) return;
    m->p4_plan_done = 1;
    const char* re = getenv("LLM_P4_RESERVE_MB");
    m->p4_reserve_mb = re ? atoi(re) : 2048;
    uint64_t active = 0;
    if (mlx_get_active_memory((size_t*)&active) != 0) active = 0;
    m->p4_weights_bytes = active;   /* al primo append: pesi+modello, KV ancora ~0 */
    { uint64_t exb = llm_mlx_moe_swap_expert_bytes(); /* streaming: gli esperti NON sono materializzati */
      if (exb > 0 && m->p4_weights_bytes > exb) m->p4_weights_bytes -= exb; }
    const uint64_t wired = llm_p4_sysctl_u64("iogpu.wired_limit_mb") * 1024ull * 1024ull;
    const uint64_t ram = llm_p4_sysctl_u64("hw.memsize");
    uint64_t limit = ram ? (ram / 4ull) * 3ull : 0;
    if (wired && (limit == 0 || wired < limit)) limit = wired;
    const uint64_t reserve = (uint64_t)m->p4_reserve_mb * 1024ull * 1024ull;
    const uint64_t used = m->p4_weights_bytes + reserve;
    m->p4_budget_bytes = (limit > used) ? (limit - used) : 0;
    m->p4_kv_per_token = llm_p4_kv_per_token(&m->cfg);
    m->p4_state_bytes = llm_p4_kv_state(&m->cfg);
    m->p4_enforce = (limit > 0);   /* senza un limite noto non si rifiuta nulla */
    /* Kill-switch diagnostico (F3 qwen4_exp 2026-09-25): su modelli il cui footprint
     * legittimamente si avvicina al wired limit (104GB su 96GB wired) il budget KV calcola
     * 0 e l'admission RIFIUTA anche un singolo token, rendendo impossibile il test per-layer.
     * LLM_P4_ENFORCE=0 disattiva SOLO il rifiuto (il plan resta stampato). Default invariato. */
    { const char* pe = getenv("LLM_P4_ENFORCE"); if (pe && atoi(pe) == 0) m->p4_enforce = 0; }
    fprintf(stderr, "[p4-plan] limit=%.1fGB pesi=%.1fGB riserva=%dMB -> budget_kv=%.2fGB | "
                    "kv/token=%.1fKB stato/sessione=%.1fMB (token max ~%llu)\n",
            limit / 1073741824.0, m->p4_weights_bytes / 1073741824.0, m->p4_reserve_mb,
            m->p4_budget_bytes / 1073741824.0, m->p4_kv_per_token / 1024.0,
            m->p4_state_bytes / 1048576.0,
            (unsigned long long)(m->p4_kv_per_token ? m->p4_budget_bytes / m->p4_kv_per_token : 0));
}
uint64_t llm_p4_kv_needed(LlmMlxModel* m, int offset, int n) {
    if (m == NULL || m->p4_kv_per_token == 0) return 0;
    return (uint64_t)(offset + n) * m->p4_kv_per_token + m->p4_state_bytes;
}
/* P4: KV+stato di TUTTE le sessioni attive registrate (il conto multi-sessione). */
uint64_t llm_p4_used_sessions(LlmMlxModel* m, const struct LlmMlxSession* exclude) {
    if (m == NULL || m->p4_kv_per_token == 0) return 0;
    uint64_t tot = 0;
    for (int i = 0; i < m->p4_nsessions; i++) {
        const struct LlmMlxSession* ss = m->p4_sessions[i];
        if (ss == NULL || ss == exclude) continue;
        tot += llm_p4_kv_needed(m, (int)ss->cache_offset, 0);
    }
    return tot;
}
