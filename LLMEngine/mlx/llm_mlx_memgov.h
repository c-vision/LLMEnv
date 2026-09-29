/* llm_mlx_memgov.h — MEMORY GOVERNOR (port dell'idea `MemoryGovernor` di Splash).
 *
 * Tiene ~2 GiB liberi per macOS e fa crescere le cache (pool KV paged, cache esperti) solo
 * quando resta >= 1 GiB libero dopo la crescita. Se la memoria non basta, ATTENDE fino a un
 * timeout e poi segnala `resource_timeout` (-2), invece di spingere il sistema in swap.
 *
 * Il budget e' PURI (decisioni testabili); la misura della memoria libera e' OS-specifica
 * (macOS: host_statistics64, free+inactive+purgeable). Convenzione (come kvpage/ops):
 * `static inline`.
 */
#ifndef LLM_MLX_MEMGOV_H
#define LLM_MLX_MEMGOV_H

#include <stdint.h>
#include <stdbool.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <unistd.h>
#else
#include <unistd.h>
#endif

#define LLM_MEMGOV_RESOURCE_TIMEOUT (-2)

typedef struct {
    int      on;                     /* governor attivo */
    uint64_t reserve_bytes;          /* da LASCIARE liberi per il sistema (~2 GiB) */
    uint64_t grow_min_free_bytes;    /* libero MINIMO richiesto per crescere (~1 GiB) */
    int      timeout_ms;             /* attesa massima per la memoria */
} LlmMemGov;

static inline LlmMemGov llm_memgov_default(void) {
    LlmMemGov g;
    g.on = 1;
    g.reserve_bytes = 2ull * 1024ull * 1024ull * 1024ull;   /* 2 GiB */
    g.grow_min_free_bytes = 1ull * 1024ull * 1024ull * 1024ull; /* 1 GiB */
    g.timeout_ms = 30000;
    return g;
}

/* Memoria libera (reclamabile) del sistema. 0 = non misurabile. */
static inline uint64_t llm_memgov_system_free_bytes(void) {
#if defined(__APPLE__)
    mach_port_t host = mach_host_self();
    vm_size_t page_size = 0;
    if (host_page_size(host, &page_size) != KERN_SUCCESS || page_size == 0) page_size = 16384;
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm, &count) != KERN_SUCCESS) return 0;
    uint64_t pages = (uint64_t)vm.free_count + (uint64_t)vm.inactive_count
                   + (uint64_t)vm.purgeable_count + (uint64_t)vm.speculative_count;
    return pages * (uint64_t)page_size;
#else
    long ps = sysconf(_SC_PAGESIZE); if (ps <= 0) ps = 4096;
    long avail = sysconf(_SC_AVPHYS_PAGES); if (avail <= 0) return 0;
    return (uint64_t)avail * (uint64_t)ps;
#endif
}

/* Libero minimo richiesto per allocare `need` byte senza violare il budget. */
static inline uint64_t llm_memgov_need_free(const LlmMemGov* g, uint64_t need) {
    return g->reserve_bytes + g->grow_min_free_bytes + need;
}

/* C'e' spazio per `need` byte dati `free_b` byte liberi? (puro, testabile) */
static inline bool llm_memgov_has_room(const LlmMemGov* g, uint64_t free_b, uint64_t need) {
    if (!g->on) return true;
    return free_b >= llm_memgov_need_free(g, need);
}

/* Attende che ci sia spazio per `need` byte, fino a `timeout_ms`. 0 = ok, -2 = resource_timeout.
 * Se il governor e' OFF o la memoria non e' misurabile, non blocca (0). */
static inline int llm_memgov_wait(const LlmMemGov* g, uint64_t need) {
    if (!g->on) return 0;
    int waited = 0;
    for (;;) {
        uint64_t f = llm_memgov_system_free_bytes();
        if (f == 0) return 0;                 /* non misurabile: non bloccare */
        if (llm_memgov_has_room(g, f, need)) return 0;
        if (waited >= g->timeout_ms) return LLM_MEMGOV_RESOURCE_TIMEOUT;
        usleep(10000);                        /* 10 ms */
        waited += 10;
    }
}

#endif /* LLM_MLX_MEMGOV_H */
