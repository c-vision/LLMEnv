/* GENERATO da tools/gen_dsv41_moe_reference.py — NON modificare a mano. */
#ifndef LLM_DSV41_MOE_REF_H
#define LLM_DSV41_MOE_REF_H
static const float REF_GATE_X[3] = {
    0.757, -0.522, -0.444,
};
static const float REF_GATE_W[18] = {
    0.52, 0.956, -0.253, 0.197, 0.353, -0.607, 0.102, -0.423, 0.82, 0.312,
    0.343, -0.301, 0.031, -0.939, 0.363, 0.687, -0.768, 0.679,
};
static const float REF_GATE_B[6] = {
    -0.16, 0.876, -0.062, -0.797, -0.604, -0.804,
};
#define REF_GATE_D 3
#define REF_GATE_E 6
#define REF_GATE_ACT 2
#define REF_GATE_SF0 2
#define REF_GATE_NRM0 1
#define REF_GATE_RS0 1
static const int REF_GATE_IDX0[2] = {
    1, 2,
};
static const float REF_GATE_WGT0[2] = {
    0.5265525, 0.4734475,
};
#define REF_GATE_SF1 0
#define REF_GATE_NRM1 1
#define REF_GATE_RS1 1
static const int REF_GATE_IDX1[2] = {
    1, 2,
};
static const float REF_GATE_WGT1[2] = {
    0.5745479, 0.4254521,
};
#define REF_GATE_SF2 1
#define REF_GATE_NRM2 0
#define REF_GATE_RS2 1.3
static const int REF_GATE_IDX2[2] = {
    1, 2,
};
static const float REF_GATE_WGT2[2] = {
    0.7258238, 0.6285383,
};
static const float REF_SW_GATE[4] = {
    0.174, 0.022, -0.535, 0.156,
};
static const float REF_SW_UP[4] = {
    -0.063, -0.058, -0.344, 0.72,
};
#define REF_SW_LIMIT 1.5
#define REF_SW_WEIGHT 0.7
#define REF_SW_D 4
static const float REF_SW_OUT[4] = {
    -0.004169653, -0.0004515124, 0.04758281, 0.04237213,
};
static const float REF_MOE_ROUTED[6] = {
    -0.409, -0.992, -0.626, 0.236, -0.493, 0.309,
};
static const float REF_MOE_SHARED[3] = {
    -0.646, 0.929, -0.435,
};
#define REF_MOE_ACT 2
#define REF_MOE_D 3
static const float REF_MOE_OUT[3] = {
    -0.819, -0.556, -0.752,
};
#endif /* LLM_DSV41_MOE_REF_H */
