/* llm_mlx_kv_donate_probe.cpp — PROBE SENZA MODELLO: perche' un append in-place sulla KV
 * cache non e' donatable, e quale oggetto lo impedisce. C++ diretto su libmlx (link statico
 * dei .a gia' presenti in build/_deps): NESSUN tocco alla dylib condivisa, NESSUN modello.
 *
 * Context (bug (open) MLX-Decode-Costo-Contesto-KV-Attention, §8.4/§8.8): sull'engine reale
 * il 43,8% degli append non e' donatable e la popolazione e' BIMODALE (step con 0 append
 * copianti e step con TUTTI e 72 copianti). Il costo e' O(cap) per append -> pendenza
 * ctx-proporzionale. `is_donatable()` e' PUBLIC su mlx::core::array: qui lo si legge
 * direttamente, punto per punto, sulla sequenza reale del decode.
 *
 * Ipotesi testata: il grafo NON EVALUATO del passo precedente (l'engine fa
 * `mlx_async_eval(next_id)` e la frontier delle cache non e' materializzata) tiene in vita i
 * nodi SDPA che referenziano le VISTE (`mlx_slice`) della cache -> `array_desc_.use_count()>1`
 * o `data.use_count()>1` sul buffer della cache -> append non donabile -> copia intera.
 * Bracci:
 *   A = sync per passo (eval(attn) subito)        -> previsione: sempre donabile
 *   B = un grafo in volo (attn del passo prima)   -> previsione: MAI donabile
 *   C = due grafi in volo (lag 2)                 -> idem, ma mostra la dipendenza dal lag
 *   D = niente vista: SDPA sulla cache intera     -> previsione: donabile anche con B
 *
 * Build (dal repo, niente CMake, niente ri-build della dylib):
 *   cd native/llmengine_mlx && /usr/bin/c++ -std=c++20 -O2 \
 *     -I../../libs/mlx-deps/mlx-src -I../../libs/mlx-deps/mlx-c-src \
 *     -Ibuild/_deps/metal_cpp-src/include \
 *     llm_mlx_kv_donate_probe.cpp build/_deps/mlx-build/libmlx.a \
 *     -framework Metal -framework Foundation -framework QuartzCore -framework Accelerate \
 *     -o /tmp/llm_kv_donate_probe
 */
#include <mlx/mlx.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace mlx::core;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

struct Arm {
    const char* name;
    int lag;         /* 0 = eval subito; N = N grafi tenuti in volo */
    bool whole_buf;  /* true = SDPA sulla cache intera (nessuna vista) */
    bool frontier;   /* true = eval esplicito delle cache a fine passo */
};

static void run_arm(const Arm& arm, int nkv, int nh, int hd, int n_ctx, int steps, Device dev, Stream st) {
    const int cap = ((n_ctx + 256 + 255) / 256) * 256; /* come llm_mlx_kv_cap_for */
    array cache = zeros({1, nkv, cap, hd}, bfloat16, st);
    array vc = zeros({1, nkv, cap, hd}, bfloat16, st);
    eval(std::vector<array>{cache, vc}); /* = prefill materializzato: buffer pulito e contiguo */
    array q = zeros({1, nh, 1, hd}, bfloat16, st);
    const float scale = 1.0f / sqrtf((float)hd);

    std::vector<array> inflight;
    long denied = 0, total = 0;
    long denied_first = -1;
    int off = n_ctx;
    const double t_wall = now_ms();

    for (int s = 0; s < steps; s++, off++) {
        const bool don_k = cache.is_donatable();
        const bool don_v = vc.is_donatable();
        if (denied_first < 0 && !don_k) denied_first = s;
        total += 2;
        if (!don_k) denied++;
        if (!don_v) denied++;

        array kn = zeros({1, nkv, 1, hd}, bfloat16, st);
        array vn = zeros({1, nkv, 1, hd}, bfloat16, st);
        /* L'append di produzione: slice_update con donation (in C e'
         * mlx_slice_update_inplace; qui la stessa primitiva). */
        cache = slice_update(cache, kn, Shape{0, 0, off, 0}, Shape{1, nkv, off + 1, hd}, st);
        vc = slice_update(vc, vn, Shape{0, 0, off, 0}, Shape{1, nkv, off + 1, hd}, st);

        /* attn + un nodo a valle (l'o_proj+residual del layer): e' CIO' che il passo tiene
         * in volo quando l'eval e' asincrono. */
        array o = add(
            arm.whole_buf
                ? fast::scaled_dot_product_attention(q, cache, vc, scale, "", {}, {}, st)
                : fast::scaled_dot_product_attention(
                      q,
                      slice(cache, Shape{0, 0, 0, 0}, Shape{1, nkv, off + 1, hd}, Shape{1, 1, 1, 1}, st),
                      slice(vc, Shape{0, 0, 0, 0}, Shape{1, nkv, off + 1, hd}, Shape{1, 1, 1, 1}, st),
                      scale, "", {}, {}, st),
            array(0.f, bfloat16),
            st);
        if (arm.lag == 0) {
            eval(o);
        } else {
            inflight.push_back(o);
            if ((int)inflight.size() > arm.lag) {
                eval(inflight.front());
                inflight.erase(inflight.begin());
            }
        }
        if (arm.frontier) eval(std::vector<array>{cache, vc});
        if (s < 3)
            printf("  [%s] step %2d off=%5d  donabile PRIMA append: cache=%d v=%d   |   "
                   "DOPO eval del passo: cache=%d v=%d\n",
                   arm.name, s, off, (int)don_k, (int)don_v,
                   (int)cache.is_donatable(), (int)vc.is_donatable());
        else if (s == 0 || s == steps - 1)
            printf("  [%s] step %2d  off=%5d  donabile(cache)=%d donabile(v)=%d\n",
                   arm.name, s, off, (int)don_k, (int)don_v);
    }
    eval(inflight);
    const double twall = (now_ms() - t_wall) / (double)steps;
    printf("  [%s] cap=%d  append non DONABILI %ld/%ld (%.1f%%)  primo al step %ld  "
           "wall=%.3f ms/passo\n",
           arm.name, cap, denied, total, 100.0 * denied / total, denied_first, twall);
    fflush(stdout);
}

int main(int argc, char** argv) {
    const int nkv = (argc > 1) ? atoi(argv[1]) : 8;
    const int nh = (argc > 2) ? atoi(argv[2]) : 32;
    const int hd = (argc > 3) ? atoi(argv[3]) : 128;
    const int n_ctx = (argc > 4) ? atoi(argv[4]) : 16000;
    const int steps = (argc > 5) ? atoi(argv[5]) : 24;
    Device dev = default_device();
    Stream st = default_stream(Device::gpu);
    printf("# KV donate probe (C++ diretto, nessun modello): nkv=%d nh=%d hd=%d ctx=%d steps=%d\n",
           nkv, nh, hd, n_ctx, steps);
    const Arm arms[] = {
        {"A sync-per-passo      ", 0, false, false},
        {"B 1 grafo in volo     ", 1, false, false},
        {"C 2 grafi in volo     ", 2, false, false},
        {"D in-volo senza vista ", 1, true,  false},
        {"E in-volo + frontier  ", 1, false, true},
    };
    for (const Arm& a : arms) run_arm(a, nkv, nh, hd, n_ctx, steps, dev, st);
    return 0;
}
