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

/* Pipeline COMPLETA divergence-aware (CPU reference per il port GPU/MLX):
 * compact (sort/group) -> GEMM per-esperto -> scatter (accumulo pesato).
 *   x:   [T, H] input attivazioni
 *   W:   [E, H, O]  pesi per-esperto (row-major: W[e*H*O + h*O + o])
 *   aex: [T*K] esperto assegnato (topk), aw: [T*K] peso gate
 *   out: [T, O] output (accumulato)
 * Per ogni esperto e: prende i token assegnati (compact), y_g = x_g @ W[e], somma w * y_g in out. */
int llmengine_moe_diverge_forward(const float* x, int T, int H, int O,
                                  const float* W, int E,
                                  const int* aex, const float* aw, int K,
                                  float* out) {
    if (!x || !W || !aex || !aw || !out || T < 0 || H <= 0 || O <= 0 || E <= 0 || K <= 0) return -1;
    int n = T * K;
    int* start = (int*)malloc((size_t)(E + 1) * sizeof(int));
    int* sorted = (int*)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));
    if (!start || !sorted) { free(start); free(sorted); return -1; }
    if (llmengine_moe_diverge_plan(aex, n, E, start, sorted) != 0) { free(start); free(sorted); return -1; }
    for (int t = 0; t < T; ++t) for (int o = 0; o < O; ++o) out[t*O + o] = 0.0f;
    for (int e = 0; e < E; ++e) {
        for (int k = start[e]; k < start[e+1]; ++k) {
            int a = sorted[k]; int ti = a / K; float w = aw[a];
            for (int o = 0; o < O; ++o) {
                float acc = 0.0f;
                for (int h = 0; h < H; ++h) acc += x[ti*H + h] * W[((size_t)e*H + h)*O + o];
                out[ti*O + o] += w * acc;
            }
        }
    }
    free(start); free(sorted);
    return 0;
}
