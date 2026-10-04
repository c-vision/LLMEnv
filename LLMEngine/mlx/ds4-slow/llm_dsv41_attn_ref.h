/* GENERATO da tools/gen_dsv41_attn_reference.py — NON modificare a mano. */
#ifndef LLM_DSV41_ATTN_REF_H
#define LLM_DSV41_ATTN_REF_H
#define NEG (-INFINITY)
#define REF_SWA_NCASE 3
#define REF_SWA_W0 3
static const int REF_SWA_IDX0[15] = {
    0, -1, -1, 0, 1, -1, 0, 1, 2, 1, 2, 3, 2, 3, 4,
};
#define REF_SWA_W1 3
static const int REF_SWA_IDX1[6] = {
    0, 1, 2, 1, 2, 3,
};
#define REF_SWA_W2 4
static const int REF_SWA_IDX2[12] = {
    1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6,
};
static const float REF_CP_KV_IN0[12] = {
    -0.931, -0.529, 0.738, -0.931, -0.529, 0.738, -0.931, -0.529, 0.738, -0.931, -0.529, 0.738,
};
static const float REF_CP_GATE_IN0[12] = {
    0.8, 0.451, -0.615, 0.8, 0.451, -0.615, 0.8, 0.451, -0.615, 0.8, 0.451, -0.615,
};
#define REF_CP_CACHE_LEN0 0
#define REF_CP_CNEW0 0
#define REF_CP_M0 2
static const float REF_CP_CACHE_KV0[1] = {
    0.0,
};
static const float REF_CP_CACHE_GATE0[1] = {
    0.0,
};
static const float REF_CP_POOLED0[6] = {
    -0.931, -0.529, 0.738, -0.931, -0.529, 0.738,
};
static const float REF_CP_NEW_KV0[1] = {
    0.0,
};
static const float REF_CP_NEW_GATE0[1] = {
    0.0,
};
static const float REF_CP_KV_IN1[12] = {
    0.833, 0.949, 0.929, 0.833, 0.949, 0.929, 0.833, 0.949, 0.929, 0.833, 0.949, 0.929,
};
static const float REF_CP_GATE_IN1[12] = {
    0.12, -0.072, -0.424, 0.12, -0.072, -0.424, 0.12, -0.072, -0.424, 0.12, -0.072, -0.424,
};
#define REF_CP_CACHE_LEN1 1
#define REF_CP_CNEW1 1
#define REF_CP_M1 2
static const float REF_CP_CACHE_KV1[3] = {
    -0.15, 0.908, 0.667,
};
static const float REF_CP_CACHE_GATE1[3] = {
    -0.863, -0.113, -0.243,
};
static const float REF_CP_POOLED1[6] = {
    0.5653321, 0.9289202, 0.7861768, 0.833, 0.949, 0.929,
};
static const float REF_CP_NEW_KV1[3] = {
    0.833, 0.949, 0.929,
};
static const float REF_CP_NEW_GATE1[3] = {
    0.12, -0.072, -0.424,
};
static const float REF_IT_SCORES[6] = {
    0.5, NEG, 0.9, 0.1, NEG, 0.7,
};
#define REF_IT_N 6
#define REF_IT_K 3
static const int REF_IT_IDX[3] = {
    0, 2, 5,
};
static const float REF_CB_SCORES[10] = {
    0.565, 0.906, 0.395, -0.452, 0.015, 0.667, -0.906, -0.03, -0.914, -0.918,
};
#define REF_CB_W 10
#define REF_CB_LEN 7
#define REF_CB_CNT 2
#define REF_CB_BLK 4
static const int REF_CB_OUT[2] = {
    0, 1,
};
static const float REF_SD_Q[6] = {
    -0.453, -0.611, 0.214, -0.453, -0.611, 0.214,
};
static const float REF_SD_KEYS[24] = {
    0.835, -0.074, -0.696, 0.835, -0.074, -0.696, 0.598, -0.154, -0.505, 0.598, -0.154, -0.505,
    -0.082, -0.677, -0.314, -0.082, -0.677, -0.314, -0.319, -0.757, -0.123, -0.319, -0.757, -0.123,
};
static const int REF_SD_IDX[4] = {
    0, 2, -1, 3,
};
static const float REF_SD_SINK[2] = {
    0.2, -0.1,
};
static const float REF_SD_SCALE[1] = {
    0.5773503,
};
static const float REF_SD_OUT[6] = {
    0.1641154, -0.2947109, -0.2668163, 0.1774074, -0.3185802, -0.2884263,
};
#endif /* LLM_DSV41_ATTN_REF_H */
