/* llm_mlx_fused_residual_rms.cpp — kernel Metal fuso "residual + RMSNorm".
 *
 * Calcola  y = x + w * rmsnorm(r)  in UN solo dispatch (rmsnorm(r) = r * rsqrt(mean(r^2)+eps)),
 * dove `r` e' l'uscita del sublayer (attn/ffn), `x` il residuo in ingresso, `w` il peso della
 * norma. Sostituisce la coppia  { RMSNorm -> Add }  (stile "sandwich" gemma) o { Add -> RMSNorm }
 * (stile llama, con x e r scambiati). Toglie UN nodo per layer + il round-trip dell'intermedio
 * della norma (scrivi+rileggi [.., H]).
 *
 * Modellato su llm_mlx_fused_rope_append.cpp: kernel via mlx_fast_metal_kernel, gate env
 * `LLM_FUSE_RESIDUAL_RMS` (default OFF, opt-in) e `LLM_FUSE_RESIDUAL_RMS_VERIFY` (calcola entrambi,
 * stampa il max_abs_diff e USA l'originale). Fallback SILENZIOSO su ogni geometria non supportata.
 */
#include "mlx/c/mlx.h"
#include "mlx/c/fast.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

namespace {

constexpr const char* kHeader = R"(
#include <metal_stdlib>
using namespace metal;
)";

constexpr const char* kSource = R"(
    const uint H = (uint)Hidden;
    const uint TG = (uint)TG_N;
    uint row = thread_position_in_grid.y;
    uint tid = thread_position_in_grid.x;
    if (row >= (uint)NRows || tid >= TG) return;

    const device T* xrow = x + (unsigned long)row * H;
    const device T* rrow = r + (unsigned long)row * H;
    const device T* wrow = w;
    device T* yrow = y + (unsigned long)row * H;

    float ss = 0.0f;
    for (uint i = tid; i < H; i += TG) { float v = (float)rrow[i]; ss += v * v; }

    threadgroup float sh[TG];
    sh[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = TG / 2u; st > 0u; st >>= 1u) {
        if (tid < st) sh[tid] += sh[tid + st];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = metal::rsqrt(sh[0] / (float)H + eps_arr[0]);

    for (uint i = tid; i < H; i += TG) {
        float v = (float)rrow[i];
        float wv = (float)wrow[i];
        yrow[i] = (T)((float)xrow[i] + v * inv * wv);
    }
)";

std::mutex g_kernel_mutex;
mlx_fast_metal_kernel g_kernel = {nullptr};

bool ensure_kernel() {
    std::lock_guard<std::mutex> guard(g_kernel_mutex);
    if (g_kernel.ctx != nullptr) return true;
    const char* inputs[] = {"x", "r", "w", "eps_arr"};
    const char* outputs[] = {"y"};
    mlx_vector_string iv = mlx_vector_string_new();
    mlx_vector_string ov = mlx_vector_string_new();
    for (const char* name : inputs) mlx_vector_string_append_value(iv, name);
    for (const char* name : outputs) mlx_vector_string_append_value(ov, name);
    g_kernel = mlx_fast_metal_kernel_new(
        "llm_fused_residual_rms", iv, ov, kSource, kHeader,
        /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
    mlx_vector_string_free(iv);
    mlx_vector_string_free(ov);
    return g_kernel.ctx != nullptr;
}

std::mutex g_reason_mutex;
const char* g_reject_reason = nullptr;
void reject_once(const char* why) {
    std::lock_guard<std::mutex> guard(g_reason_mutex);
    if (g_reject_reason != nullptr) return;
    g_reject_reason = why;
    fprintf(stderr, "[llm_fused_residual_rms] rejected: %s\n", why);
}

int g_enable_state = -1;
int g_verify_state = -1;

}  // namespace

extern "C" bool llm_mlx_fused_residual_rms_enabled(void) {
    extern int g_llm_mlx_train_composable;
    if (g_llm_mlx_train_composable) return false;   /* training: no kernel Metal (no vjp) */
    if (g_enable_state < 0) {
        const char* e = getenv("LLM_FUSE_RESIDUAL_RMS");
        g_enable_state = (e != nullptr && e[0] == '0') ? 0 : (e != nullptr ? 1 : 0);
    }
    return g_enable_state == 1;
}

extern "C" bool llm_mlx_fused_residual_rms_verify_enabled(void) {
    if (g_verify_state < 0) g_verify_state = (getenv("LLM_FUSE_RESIDUAL_RMS_VERIFY") != nullptr) ? 1 : 0;
    return g_verify_state == 1;
}

#define LLM_REJ(why) do { reject_once(why); return false; } while (0)

extern "C" bool llm_mlx_fused_residual_rms(
    mlx_array* y_out, mlx_array x, mlx_array r, mlx_array w, float eps, mlx_stream stream) {
    if (y_out == nullptr || x.ctx == nullptr || r.ctx == nullptr || w.ctx == nullptr)
        LLM_REJ("null handle");
    if (mlx_array_ndim(x) < 1 || mlx_array_ndim(r) != mlx_array_ndim(x))
        LLM_REJ("x/r rank mismatch");
    const int* xs = mlx_array_shape(x);
    const int* rs = mlx_array_shape(r);
    const int* ws = mlx_array_shape(w);
    if (xs == nullptr || rs == nullptr || ws == nullptr) LLM_REJ("null shape");
    const int nd = mlx_array_ndim(x);
    const int H = xs[nd - 1];
    if (H <= 0 || rs[nd - 1] != H) LLM_REJ("last dim mismatch");
    if (mlx_array_ndim(w) != 1 || ws[0] != H) LLM_REJ("weight shape != [H]");
    long long rows = 1;
    for (int i = 0; i < nd - 1; i++) {
        if (xs[i] != rs[i]) LLM_REJ("leading dim mismatch");
        rows *= (long long)xs[i];
    }
    if (rows <= 0) LLM_REJ("empty rows");
    if (!ensure_kernel()) LLM_REJ("metal kernel registration failed");

    const int TG_N = (H >= 128) ? 128 : ((H >= 64) ? 64 : ((H >= 32) ? 32 : 1));
    const float epsv = eps;
    const int eps_shape[1] = {1};
    mlx_array eps_arr = mlx_array_new_data(&epsv, eps_shape, 1, MLX_FLOAT32);

    /* Un solo template dtype T (mlx non supporta dtype multipli nel custom kernel): porto
     * x/r/w al dtype COMUNE = il piu' largo fra x e r (cast solo WIDENING = esatto), output =
     * quel dtype. Replica `x + w*rmsnorm(r)` promosso. */
    const mlx_dtype dx = mlx_array_dtype(x), dr = mlx_array_dtype(r), dw = mlx_array_dtype(w);
    const mlx_dtype dc = (mlx_array_itemsize(r) >= mlx_array_itemsize(x)) ? dr : dx;
    mlx_array xc = mlx_array_new(), rcv = mlx_array_new(), wc = mlx_array_new();
    bool cc = true;
    if (dx == dc) cc = cc && mlx_array_set(&xc, x) == 0; else cc = cc && mlx_astype(&xc, x, dc, stream) == 0;
    if (dr == dc) cc = cc && mlx_array_set(&rcv, r) == 0; else cc = cc && mlx_astype(&rcv, r, dc, stream) == 0;
    if (dw == dc) cc = cc && mlx_array_set(&wc, w) == 0; else cc = cc && mlx_astype(&wc, w, dc, stream) == 0;
    if (!cc) { mlx_array_free(xc); mlx_array_free(rcv); mlx_array_free(wc); mlx_array_free(eps_arr); LLM_REJ("common-dtype cast failed"); }

    mlx_array ins_data[4] = {xc, rcv, wc, eps_arr};
    mlx_vector_array inputs = mlx_vector_array_new_data(ins_data, 4);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dc);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Hidden", H);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NRows", (int)rows);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TG_N", TG_N);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, xs, nd, dc);
    mlx_fast_metal_kernel_config_set_grid(cfg, TG_N, (int)rows, 1);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, TG_N, 1, 1);

    mlx_vector_array outputs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outputs, g_kernel, inputs, cfg, stream);
    bool ok = rc == 0 && mlx_vector_array_size(outputs) == 1;
    mlx_array yo = mlx_array_new();
    if (ok) ok = mlx_vector_array_get(&yo, outputs, 0) == 0;
    if (ok) ok = mlx_array_set(y_out, yo) == 0;

    mlx_array_free(yo);
    mlx_vector_array_free(outputs);
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(inputs);
    mlx_array_free(eps_arr);
    mlx_array_free(xc); mlx_array_free(rcv); mlx_array_free(wc);
    return ok;
}
