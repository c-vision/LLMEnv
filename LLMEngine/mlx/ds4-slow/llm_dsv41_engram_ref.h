/* GENERATO da tools/gen_dsv41_engram_reference.py — NON modificare a mano. */
#ifndef LLM_DSV41_ENGRAM_REF_H
#define LLM_DSV41_ENGRAM_REF_H
#include <stdint.h>
#define REF_EN_DEPTH 2
#define REF_EN_NHEADS 2
#define REF_EN_PAD 2
#define REF_EN_NUMEMB 4062
static const int64_t REF_EN_TOKEN_MAP[40] = {
    0LL, 1LL, 2LL, 3LL, 4LL, 5LL,
    6LL, 7LL, 8LL, 9LL, 10LL, 11LL,
    12LL, 13LL, 14LL, 15LL, 16LL, 17LL,
    18LL, 19LL, 20LL, 21LL, 22LL, 23LL,
    24LL, 25LL, 26LL, 27LL, 28LL, 29LL,
    30LL, 31LL, 32LL, 33LL, 34LL, 35LL,
    36LL, 37LL, 38LL, 39LL,
};
#define REF_EN_V 40
#define REF_EN_L 5
static const int64_t REF_EN_IDS[5] = {
    3LL, 7LL, 7LL, 11LL, 3LL,
};
static const int64_t REF_EN_HIST[2] = {
    -1LL, 9LL,
};
static const int64_t REF_EN_MULT[3] = {
    3607112625930809LL, 3066986572837701LL, 2018164768241499LL,
};
static const int64_t REF_EN_PRIMES[4] = {
    1009LL, 1013LL, 1019LL, 1021LL,
};
static const int64_t REF_EN_OFFSETS[4] = {
    0LL, 1009LL, 2022LL, 3041LL,
};
static const int64_t REF_EN_HASH[20] = {
    882LL, 1206LL, 2973LL, 3796LL, 666LL, 1794LL,
    2296LL, 3707LL, 103LL, 1511LL, 2738LL, 3999LL,
    415LL, 1748LL, 2360LL, 4017LL, 448LL, 1563LL,
    2269LL, 3209LL,
};
static const int64_t REF_EN_HISTOUT[2] = {
    11LL, 3LL,
};
static const int64_t REF_EN_MASK[5] = {
    0LL, 1LL, 0LL, 0LL, 1LL,
};
static const int64_t REF_EN_HASH2[20] = {
    882LL, 1206LL, 2973LL, 3796LL, 133LL, 1450LL,
    2755LL, 3404LL, 846LL, 1589LL, 2995LL, 3908LL,
    415LL, 1748LL, 2507LL, 3842LL, 133LL, 1450LL,
    2755LL, 3404LL,
};
static const int64_t REF_EN_HISTOUT2[2] = {
    11LL, -1LL,
};
#define REF_EN_HC 2
#define REF_EN_DIM 3
static const float REF_EN_H[6] = {
    0.426, 0.982, -0.808, 0.839, -0.295, -0.53,
};
static const float REF_EN_KEY[6] = {
    -0.254, 0.459, -0.617, -0.598, -0.809, -0.284,
};
static const float REF_EN_VALUE[3] = {
    -0.491, 0.379, -0.426,
};
static const float REF_EN_QW[6] = {
    1.83, 0.856, 0.765, 1.53, 0.278, 0.322,
};
static const float REF_EN_KW[6] = {
    1.593, 0.776, 0.956, 0.093, 1.322, 0.568,
};
static const float REF_EN_GATE_OUT[6] = {
    0.09299906, 1.239041, -1.096917, 0.5611712, -0.08054556, -0.771049,
};
#endif /* LLM_DSV41_ENGRAM_REF_H */
