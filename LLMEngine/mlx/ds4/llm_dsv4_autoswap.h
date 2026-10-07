/* llm_dsv4_autoswap.h — auto-sizing della cache esperti DS4.
 * Vive con DS4 (non nel motore MLX): l'engine ds4-slow la usa da solo, come fa il ds4 originale.
 * Tutta la RAM utilizzabile DOPO i pesi non-esperto, con i tetti di sicurezza. */
#ifndef LLM_DSV4_AUTOSWAP_H
#define LLM_DSV4_AUTOSWAP_H

#include "llm_dsv4_engine.h"     /* LlmDsv4Engine, LlmDsv4PackedExperts, llm_ggml_nbytes */
#include "llm_mlx_memgov.h"      /* llm_memgov_from_registry (utility condivisa) */

#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <stdio.h>

/* Ritorna il numero di bundle della cache esperti (1 bundle gate+up+down ~ 7 MB). */
static inline int llm_dsv4_autoswap_bundles(const LlmDsv4Engine* e) {
    const LlmDsv4Config* cc = &e->w.cfg;
    uint64_t eb = 0;
    int il;
    for (il = 0; il < cc->n_layer && e->w.exps; il++) {
        const LlmDsv4PackedExperts* pe = &e->w.exps[il];
        uint64_t rg = (uint64_t)llm_ggml_nbytes(pe->gate_type, pe->in_dim);
        uint64_t rd = (uint64_t)llm_ggml_nbytes(pe->down_type, pe->ff_dim);
        eb += (uint64_t)cc->n_expert * ((uint64_t)cc->n_ff_exp * rg * 2ull + (uint64_t)cc->n_embd * rd);
    }
    uint64_t maplen = (uint64_t)e->maplen;
    uint64_t nonexp = (eb > 0 && maplen > eb) ? (maplen - eb) : maplen;
    uint64_t ram = 0; size_t sz = sizeof ram;
    if (sysctlbyname("hw.memsize", &ram, &sz, NULL, 0) != 0) ram = 0;
    uint64_t limit = ram ? (ram / 4ull) * 3ull : 0;
    int32_t wired = 0; sz = sizeof wired;
    if (sysctlbyname("iogpu.wired_limit_mb", &wired, &sz, NULL, 0) == 0 && wired > 0)
        limit = (uint64_t)wired * 1024ull * 1024ull;
    { LlmMemGov mg = llm_memgov_from_registry();
      uint64_t avail = (limit > nonexp + mg.reserve_bytes) ? (limit - nonexp - mg.reserve_bytes) : 0;
      { vm_statistics64_data_t vs; mach_msg_type_number_t vc = HOST_VM_INFO64_COUNT;
        if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vs, &vc) == KERN_SUCCESS) {
            uint64_t pg = (uint64_t)vm_page_size;
            uint64_t freeish = ((uint64_t)vs.free_count + (uint64_t)vs.inactive_count + (uint64_t)vs.speculative_count) * pg;
            uint64_t margin = 12ull << 30;
            uint64_t cap_now = freeish > margin ? freeish - margin : 0;
            if (avail > cap_now) avail = cap_now;
        }
      }
      { int b = (int)(avail / (7ull * 1024 * 1024));
        if (b < 256) b = 256;
        fprintf(stderr, "DSV4 cache: esperti=%.1fGB non-esperto=%.1fGB limit=%.1fGB -> %d bundle (~%.1fGB)\n",
                eb / 1073741824.0, nonexp / 1073741824.0, limit / 1073741824.0, b, (double)b * 7.0 / 1024.0);
        return b; }
    }
}

#endif
