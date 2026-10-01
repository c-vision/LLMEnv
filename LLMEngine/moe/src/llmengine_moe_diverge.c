/* llmengine_moe_diverge.c — MoE divergence-aware routing (§4.7 SIMT): SORT/GROUP -> COMPACT.
 *
 * I token di un batch vengono assegnati a esperti diversi (routing divergente); per eseguire GEMM
 * per-esperto in modo efficiente si COMPATTANO le assegnazioni per esperto (sort by expert id,
 * group) producendo, per ogni esperto, l'intervallo contiguo di assegnazioni (expert_start[e]..
 * expert_start[e+1]) nell'ordine ordinato. E' il primo stadio (sort/group->compact) della pipeline
 * divergence-aware `sort/group -> compact -> GEMM -> scatter`. Counting-sort STABILE: O(n + E). */
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

int llmengine_moe_diverge_plan(const int* assign_expert, int n_assign, int n_experts,
                               int* expert_start /*[n_experts+1]*/, int* sorted /*[n_assign] or NULL*/) {
    if (assign_expert == NULL || expert_start == NULL || n_assign < 0 || n_experts < 0) return -1;
    for (int e = 0; e <= n_experts; ++e) expert_start[e] = 0;
    for (int i = 0; i < n_assign; ++i) {
        int e = assign_expert[i];
        if (e < 0 || e >= n_experts) return -1;      /* routing fuori range -> errore esplicito */
        expert_start[e + 1]++;
    }
    for (int e = 0; e < n_experts; ++e) expert_start[e + 1] += expert_start[e];   /* prefix sum */
    if (sorted != NULL) {
        int* cursor = (int*)malloc((size_t)(n_experts > 0 ? n_experts : 1) * sizeof(int));
        if (cursor == NULL) return -1;
        for (int e = 0; e < n_experts; ++e) cursor[e] = expert_start[e];
        for (int i = 0; i < n_assign; ++i) { int e = assign_expert[i]; sorted[cursor[e]++] = i; }
        free(cursor);
    }
    return 0;
}
