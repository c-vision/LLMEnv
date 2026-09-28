/* test_moe_mem.c — M10 gate: contabilita' per categoria, soglie NORMAL/WARNING/CRITICAL,
 * policy prefetch e ordine di eviction (PREFETCH prima degli attivi). */
#include "llmengine_moe_mem.h"

#include <assert.h>
#include <stdio.h>

int main(void) {
    /* budget 1000 byte, warn 0.80, crit 0.95 */
    LlmEngineMoeMemMgr* m = NULL;
    assert(llmengine_moe_mem_create(&m, 1000, 0.80, 0.95) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_mem_pressure(m) == LLMENGINE_MOE_PRESS_NORMAL);
    assert(llmengine_moe_mem_allow_prefetch(m) == 1);

    /* 1. contabilita' per categoria */
    assert(llmengine_moe_mem_reserve(m, LLMENGINE_MOE_MEM_DENSE, 100) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_mem_reserve(m, LLMENGINE_MOE_MEM_EXPERT_RAM, 200) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_mem_reserve(m, LLMENGINE_MOE_MEM_KV, 100) == LLMENGINE_MOE_OK);
    LlmEngineMoeMemStats s;
    llmengine_moe_mem_stats(m, &s);
    assert(s.used_bytes == 400);
    assert(s.per_cat[LLMENGINE_MOE_MEM_DENSE] == 100);
    assert(s.per_cat[LLMENGINE_MOE_MEM_EXPERT_RAM] == 200);
    assert(s.per_cat[LLMENGINE_MOE_MEM_KV] == 100);
    assert(s.per_cat[LLMENGINE_MOE_MEM_PREFETCH] == 0);

    /* 2. soglia WARNING a 800/1000 = 0.80: prefetch RIDOTTO (fattore 0.5) */
    assert(llmengine_moe_mem_reserve(m, LLMENGINE_MOE_MEM_TEMP, 400) == LLMENGINE_MOE_OK); /* 800 -> 0.80 */
    assert(llmengine_moe_mem_pressure(m) == LLMENGINE_MOE_PRESS_WARNING);
    assert(llmengine_moe_mem_prefetch_factor(m) == 0.5);
    assert(llmengine_moe_mem_allow_prefetch(m) == 1); /* ridotto, non sospeso */

    /* 3. soglia CRITICAL a 950/1000 = 0.95: prefetch SOSPESO (fattore 0) */
    assert(llmengine_moe_mem_reserve(m, LLMENGINE_MOE_MEM_EXPERT_DEVICE, 150) == LLMENGINE_MOE_OK); /* 950 */
    assert(llmengine_moe_mem_pressure(m) == LLMENGINE_MOE_PRESS_CRITICAL);
    assert(llmengine_moe_mem_prefetch_factor(m) == 0.0);
    assert(llmengine_moe_mem_allow_prefetch(m) == 0);

    /* 4. sforare il budget fallisce (950 + 100 > 1000) */
    assert(llmengine_moe_mem_reserve(m, LLMENGINE_MOE_MEM_TEMP, 100) == LLMENGINE_MOE_ERR_NO_MEMORY);
    llmengine_moe_mem_stats(m, &s);
    assert(s.used_bytes == 950); /* rifiutato: nessuna variazione */

    /* 5. ordine di eviction a CRITICAL: priorita' PREFETCH < expert attivi < KV */
    assert(llmengine_moe_mem_evict_priority(LLMENGINE_MOE_MEM_PREFETCH) == 0);
    assert(llmengine_moe_mem_evict_priority(LLMENGINE_MOE_MEM_EXPERT_RAM) == 1);
    assert(llmengine_moe_mem_evict_priority(LLMENGINE_MOE_MEM_TEMP) == 1);
    assert(llmengine_moe_mem_evict_priority(LLMENGINE_MOE_MEM_KV) == 2);
    assert(llmengine_moe_mem_evict_priority(LLMENGINE_MOE_MEM_PREFETCH) <
           llmengine_moe_mem_evict_priority(LLMENGINE_MOE_MEM_EXPERT_RAM));

    /* 6. release riporta la pressione a NORMAL */
    llmengine_moe_mem_release(m, LLMENGINE_MOE_MEM_TEMP, 400);
    llmengine_moe_mem_release(m, LLMENGINE_MOE_MEM_EXPERT_DEVICE, 150);
    assert(llmengine_moe_mem_pressure(m) == LLMENGINE_MOE_PRESS_NORMAL);
    assert(llmengine_moe_mem_allow_prefetch(m) == 1);
    assert(llmengine_moe_mem_cat_used(m, LLMENGINE_MOE_MEM_TEMP) == 0);

    /* 7. release eccedente clampata a zero */
    llmengine_moe_mem_release(m, LLMENGINE_MOE_MEM_DENSE, 9999);
    assert(llmengine_moe_mem_cat_used(m, LLMENGINE_MOE_MEM_DENSE) == 0);
    llmengine_moe_mem_stats(m, &s);
    assert(s.used_bytes == 300); /* 200 EXPERT_RAM + 100 KV */

    /* 8. argomenti invalidi */
    LlmEngineMoeMemMgr* bad = NULL;
    assert(llmengine_moe_mem_create(&bad, 0, 0.8, 0.95) == LLMENGINE_MOE_ERR_INVALID);
    assert(llmengine_moe_mem_create(&bad, 1000, 0.95, 0.80) == LLMENGINE_MOE_ERR_INVALID);

    llmengine_moe_mem_destroy(m);
    printf("test_moe_mem: PASS\n");
    return 0;
}
