/* test_moe_backend_metal.c — M8 gate: compute Metal == CPU (tolleranza) + remap slot. */
#include "llmengine_moe_backend_metal.h"
#include "llmengine_moe_slot.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define M 8u
#define K 16u

static float frand(uint64_t* s) { *s = *s * 6364136223846793005ull + 1442695040888963407ull; return (float)((*s >> 40) & 0xFFFF) / 65536.0f - 0.5f; }

static void cpu_ref(const float* blob, const float* in, uint32_t m, uint32_t k, float* out) {
    for (uint32_t j = 0; j < m; ++j) { float s = 0; for (uint32_t i = 0; i < k; ++i) s += blob[j * k + i] * in[i]; out[j] = s; }
}

int main(void) {
    if (!llmengine_moe_metal_available()) { printf("test_moe_backend_metal: SKIP (nessun device Metal)\n"); return 0; }

    LlmEngineMoeMetalBackend* b = NULL;
    assert(llmengine_moe_metal_create(&b, NULL) == LLMENGINE_MOE_OK);

    uint64_t seed = 7;
    float in[K]; for (uint32_t i = 0; i < K; ++i) in[i] = frand(&seed);
    float blob[M * K]; for (uint32_t i = 0; i < M * K; ++i) blob[i] = frand(&seed);
    float ref[M]; cpu_ref(blob, in, M, K, ref);

    void* addr = NULL;
    assert(llmengine_moe_metal_upload(b, blob, sizeof(blob), &addr) == LLMENGINE_MOE_OK);
    float out[M]; memset(out, 0, sizeof(out));
    assert(llmengine_moe_metal_expert_compute(b, addr, sizeof(blob), in, K, out, M) == LLMENGINE_MOE_OK);
    for (uint32_t j = 0; j < M; ++j) assert(fabsf(out[j] - ref[j]) < 1e-4f);

    /* integrazione slot pool: due esperti in 2 slot; dopo l'eviction il compute usa lo slot giusto */
    LlmEngineMoeSlotPool* pool = NULL;
    assert(llmengine_moe_slot_pool_create(&pool, 2, sizeof(blob)) == LLMENGINE_MOE_OK);
    float blob2[M * K]; for (uint32_t i = 0; i < M * K; ++i) blob2[i] = frand(&seed);
    float ref2[M]; cpu_ref(blob2, in, M, K, ref2);

    LlmEngineMoeExpertKey ev = {0, 0}; uint32_t evs = 0xFFFFFFFF, s0 = 0;
    assert(llmengine_moe_slot_pool_acquire(pool, (LlmEngineMoeExpertKey){0, 0}, &s0, &ev, &evs) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_slot_pool_set_ready(pool, (LlmEngineMoeExpertKey){0, 0}, addr, sizeof(blob)) == LLMENGINE_MOE_OK);
    uint32_t s1 = 0;
    assert(llmengine_moe_slot_pool_acquire(pool, (LlmEngineMoeExpertKey){0, 1}, &s1, &ev, &evs) == LLMENGINE_MOE_OK);
    void* addr2 = NULL;
    assert(llmengine_moe_metal_upload(b, blob2, sizeof(blob2), &addr2) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_slot_pool_set_ready(pool, (LlmEngineMoeExpertKey){0, 1}, addr2, sizeof(blob2)) == LLMENGINE_MOE_OK);
    /* entrambi nello slot giusto */
    LlmEngineMoeSlot d0, d1;
    assert(llmengine_moe_slot_pool_get(pool, llmengine_moe_slot_pool_lookup(pool, (LlmEngineMoeExpertKey){0, 0}), &d0) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_slot_pool_get(pool, llmengine_moe_slot_pool_lookup(pool, (LlmEngineMoeExpertKey){0, 1}), &d1) == LLMENGINE_MOE_OK);
    float o0[M], o1[M];
    assert(llmengine_moe_metal_expert_compute(b, d0.address, sizeof(blob), in, K, o0, M) == LLMENGINE_MOE_OK);
    assert(llmengine_moe_metal_expert_compute(b, d1.address, sizeof(blob2), in, K, o1, M) == LLMENGINE_MOE_OK);
    for (uint32_t j = 0; j < M; ++j) { assert(fabsf(o0[j] - ref[j]) < 1e-4f); assert(fabsf(o1[j] - ref2[j]) < 1e-4f); }

    assert(llmengine_moe_metal_sync(b) == LLMENGINE_MOE_OK);
    llmengine_moe_metal_release(b, addr); llmengine_moe_metal_release(b, addr2);
    llmengine_moe_slot_pool_destroy(pool);
    llmengine_moe_metal_destroy(b);
    printf("test_moe_backend_metal: PASS (GPU == CPU entro 1e-4)\n");
    return 0;
}
