/* llmcodec_ggml_iq_grids.h — dichiarazioni delle codebook grid IQ (dati di formato, generati). */
#ifndef LLMCODEC_GGML_IQ_GRIDS_H
#define LLMCODEC_GGML_IQ_GRIDS_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
extern const uint64_t iq1s_grid[2048];
extern const uint64_t iq2xxs_grid[256];
extern const uint64_t iq2xs_grid[512];
extern const uint64_t iq2s_grid[1024];
extern const uint32_t iq3xxs_grid[256];
extern const uint32_t iq3s_grid[512];
extern const uint8_t kmask_iq2xs[8];
extern const uint8_t ksigns_iq2xs[128];
extern const uint16_t kgrid_2bit_256[256];
extern const uint16_t kgrid_2bit_512[512];
extern const uint16_t kgrid_1bit_2048[2048];
extern const uint16_t kgrid_2bit_1024[1024];
extern const uint16_t kgrid_256[256];
extern const uint16_t kgrid_512[512];
#ifdef __cplusplus
}
#endif
#endif
