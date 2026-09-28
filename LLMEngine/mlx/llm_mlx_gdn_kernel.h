/* llm_mlx_gdn_kernel.h — KERNEL METAL FUSO per il GDN prefill (collo del prefill).
 *
 * §2026-09-01: il collo del prefill è il loop per-token del GDN (~32 chiamate MLX per step).
 * Questo modulo integra il kernel Metal blocked-sequential fuso (adattato da rapidMLX
 * `vllm_mlx/gdn_prefill.py`, che a sua volta deriva dal progetto oMLX `qwen35_prefill`,
 * entrambi Apache-2.0) via `mlx_fast_metal_kernel` di MLX C — la stessa ricorrenza esatta,
 * ma fusa in un solo kernel Metal (staging in threadgroup, riduzione simd_shuffle, niente
 * barrier nel loop hot). ~2× sul prefill GDN.
 *
 * Contratto layout (validato dalla funzione):
 *   q,k: [B,T,Hk,Dk] (NON ripetuti; il kernel ripete via hk = hv/(Hv/Hk)); Dk deve = 128
 *   v:   [B,T,Hv,Dv]  (Dv % 32 == 0)
 *   g,beta: [B,T,Hv]  (cast fp32)
 *   state_in/out: [B,Hv,Dv,Dk] fp32
 *   T: input scalare int32 (numero di token) — usato per gli stride
 *   y:   [B,T,Hv,Dv] (dtype di q)
 * Template: InT (dtype), Dk, Dv, Hk, Hv.
 */
#ifndef LLM_MLX_GDN_KERNEL_H
#define LLM_MLX_GDN_KERNEL_H

#include "mlx/c/mlx.h"
#include <stdlib.h>
#include <string.h>

static const char llm_gdn_kernel_header[] =
    "\n"
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
;

static const char llm_gdn_kernel_src[] =
    "\n"
    "    constexpr int TB = 32;                             // time block\n"
    "    constexpr int DB = 32;                             // dv rows per threadgroup\n"
    "    const int tid = thread_position_in_threadgroup.x;  // 0..255\n"
    "    const int blk = threadgroup_position_in_grid.x;    // Dv/DB block\n"
    "    const int hv  = threadgroup_position_in_grid.y;\n"
    "    const int b   = threadgroup_position_in_grid.z;\n"
    "    const int hk  = hv / (Hv / Hk);\n"
    "    const int dv0 = blk * DB;\n"
    "\n"
    "    // thread -> (dv row, 16-wide d segment); 8 threads per dv row, all in\n"
    "    // the same simdgroup (lane = (dv%4)*8 + seg).\n"
    "    const int dv  = tid / 8;            // 0..31\n"
    "    const int seg = tid % 8;            // 0..7\n"
    "    const int d0  = seg * 16;\n"
    "\n"
    "    threadgroup InT k_s[TB][Dk + 8];\n"
    "    threadgroup InT q_s[TB][Dk + 8];\n"
    "    threadgroup InT v_s[TB][DB + 8];\n"
    "    threadgroup float g_s[TB];\n"
    "    threadgroup InT gv_s[TB][Dk + 8];\n"
    "    threadgroup float b_s[TB];\n"
    "\n"
    "    const device InT* k_base = k + ((size_t)b * T * Hk + hk) * Dk;\n"
    "    const device InT* q_base = q + ((size_t)b * T * Hk + hk) * Dk;\n"
    "    const device InT* v_base = v + ((size_t)b * T * Hv + hv) * Dv + dv0;\n"
    "    const size_t krow = (size_t)Hk * Dk;\n"
    "\n"
    "    // state fragment in registers: [dv0+dv][d0..d0+16]\n"
    "    float4 st[4];\n"
    "    {\n"
    "        const device float4* S_in = (const device float4*)(\n"
    "            state_in + (((size_t)b * Hv + hv) * Dv + dv0 + dv) * Dk + d0);\n"
    "        for (int i = 0; i < 4; ++i) st[i] = S_in[i];\n"
    "    }\n"
    "\n"
    "    device InT* y_base = y + ((size_t)b * T * Hv + hv) * Dv + dv0;\n"
    "\n"
    "    for (int t0 = 0; t0 < T; t0 += TB) {\n"
    "        const int tt = min(TB, T - t0);\n"
    "        // cooperative staging (coalesced): k/q rows, v slice, g/beta\n"
    "        for (int p = tid; p < tt * Dk; p += 256) {\n"
    "            const int r = p / Dk, d = p % Dk;\n"
    "            k_s[r][d] = k_base[(size_t)(t0 + r) * krow + d];\n"
    "            q_s[r][d] = q_base[(size_t)(t0 + r) * krow + d];\n"
    "        }\n"
    "        for (int p = tid; p < tt * DB; p += 256) {\n"
    "            const int r = p / DB, d = p % DB;\n"
    "            v_s[r][d] = v_base[(size_t)(t0 + r) * Hv * Dv + d];\n"
    "        }\n"
    "        if (GVEC) {\n"
    "            for (int p = tid; p < tt * Dk; p += 256) {\n"
    "                const int r = p / Dk, d = p % Dk;\n"
    "                gv_s[r][d] = (InT)g[(((size_t)b * T + t0 + r) * Hv + hv) * Dk + d];\n"
    "            }\n"
    "        } else {\n"
    "            for (int p = tid; p < tt; p += 256) {\n"
    "                g_s[p] = g[((size_t)b * T + t0 + p) * Hv + hv];\n"
    "            }\n"
    "        }\n"
    "        for (int p = tid; p < tt; p += 256) {\n"
    "            b_s[p] = beta[((size_t)b * T + t0 + p) * Hv + hv];\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "\n"
    "        for (int t = 0; t < tt; ++t) {\n"
    "            const float gt = GVEC ? 0.0f : g_s[t];\n"
    "            const float bt = b_s[t];\n"
    "            const threadgroup vec<InT,4>* k4 =\n"
    "                (const threadgroup vec<InT,4>*)&k_s[t][d0];\n"
    "            const threadgroup vec<InT,4>* q4 =\n"
    "                (const threadgroup vec<InT,4>*)&q_s[t][d0];\n"
    "            float4 kf[4];\n"
    "            for (int i = 0; i < 4; ++i) kf[i] = float4(k4[i]);\n"
    "            // kv_mem = (g*state) . k ; decay applied to state first\n"
    "            float4 p4 = 0.0f;\n"
    "            const threadgroup vec<InT,4>* gv4 = (const threadgroup vec<InT,4>*)&gv_s[t][d0];\n"
    "            for (int i = 0; i < 4; ++i) {\n"
    "                if (GVEC) st[i] *= float4(gv4[i]); else st[i] *= gt;\n"
    "                p4 += st[i] * kf[i];\n"
    "            }\n"
    "            float part = p4.x + p4.y + p4.z + p4.w;\n"
    "            // reduce across the 8 segment-threads of this dv row\n"
    "            part += simd_shuffle_down(part, 4);\n"
    "            part += simd_shuffle_down(part, 2);\n"
    "            part += simd_shuffle_down(part, 1);\n"
    "            const float kv_mem = simd_shuffle(part, (tid % 32) / 8 * 8);\n"
    "            const float delta = ((float)v_s[t][dv] - kv_mem) * bt;\n"
    "\n"
    "            float4 o4 = 0.0f;\n"
    "            for (int i = 0; i < 4; ++i) {\n"
    "                st[i] += kf[i] * delta;\n"
    "                o4 += st[i] * float4(q4[i]);\n"
    "            }\n"
    "            float out = o4.x + o4.y + o4.z + o4.w;\n"
    "            out += simd_shuffle_down(out, 4);\n"
    "            out += simd_shuffle_down(out, 2);\n"
    "            out += simd_shuffle_down(out, 1);\n"
    "            if (seg == 0) {\n"
    "                y_base[(size_t)(t0 + t) * Hv * Dv + dv] = (InT)out;\n"
    "            }\n"
    "            if (SSEQ) {\n"
    "                device float4* ss = (device float4*)(state_seq + ((((size_t)b * T + (t0 + t)) * Hv + hv) * Dv + (dv0 + dv)) * Dk + d0);\n"
    "                for (int i = 0; i < 4; ++i) ss[i] = st[i];\n"
    "            }\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "\n"
    "    {\n"
    "        device float4* S_out = (device float4*)(\n"
    "            state_out + (((size_t)b * Hv + hv) * Dv + dv0 + dv) * Dk + d0);\n"
    "        for (int i = 0; i < 4; ++i) S_out[i] = st[i];\n"
    "    }\n"
;

/* cache del kernel compilato (static per modulo, una sola compilazione) */
typedef struct {
    mlx_fast_metal_kernel k;
    int block_t;
    bool built;
} llm_gdn_kernel_cache;

static llm_gdn_kernel_cache llm_gdn_kernel_singleton[2][2] = { {{0},0,false},{{0},0,false},{{0},0,false},{{0},0,false} };

static int llm_mlx_gdn_kernel_block_t(mlx_dtype in_dtype) {
    return (in_dtype == MLX_FLOAT32) ? 16 : 32;
}

/* Compila (una volta) il kernel per il dtype dato. sseq=1 -> variante che esporta
 * `state_seq` [B,T,Hv,Dv,Dk] (stato dopo OGNI token), usata dalla cattura speculativa. */
static int llm_mlx_gdn_kernel_build(int block_t, int gvec, int sseq) {
    llm_gdn_kernel_cache* C = &llm_gdn_kernel_singleton[gvec & 1][sseq & 1];
    if (C->built && C->block_t == block_t)
        return 0;
    if (C->built) { mlx_fast_metal_kernel_free(C->k); C->built = false; }
    /* TB hardcoded a block_t + SSEQ */
    char* src = (char*)malloc(strlen(llm_gdn_kernel_src) + 96);
    strcpy(src, llm_gdn_kernel_src);
    char *tb = strstr(src, "constexpr int TB = 32;");
    if (tb) {
        char repl[64]; snprintf(repl, sizeof(repl), "constexpr int TB = %d;", block_t);
        memcpy(tb, repl, strlen(repl));
    }
    /* prependi constexpr int SSEQ */
    { char pre[48]; snprintf(pre, sizeof(pre), "constexpr int SSEQ = %d;\n", sseq ? 1 : 0);
      size_t pl = strlen(pre); memmove(src + pl, src, strlen(src) + 1); memcpy(src, pre, pl); }
    const char* in_names[] = {"q","k","v","g","beta","state_in","T"};
    const char* out_names[] = {"y","state_out","state_seq"};
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i=0;i<7;i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    for (int i=0;i<3;i++) mlx_vector_string_append_value(ov, out_names[i]);
    char name[64]; snprintf(name, sizeof(name), "llm_gdn_blocked_seq_tb%d_gv%d_s%d", block_t, gvec, sseq ? 1 : 0);
    C->k = mlx_fast_metal_kernel_new(name, iv, ov, src, llm_gdn_kernel_header,
                                     /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false);
    mlx_vector_string_free(iv); mlx_vector_string_free(ov);
    free(src);
    C->block_t = block_t;
    C->built = true;
    return 0;
}

/* Prefill GDN fuso: calcola y e lo stato finale in un solo kernel Metal.
 * state_seq_out (opzionale): se NON-NULL, il kernel esporta anche lo stato dopo OGNI
 * token in [B,T,Hv,Dv,Dk] fp32 (cattura speculativa -> rollback O(1) senza re-forward).
 * Ritorna 0 su successo, -1 se ineleggibile/errore (chiamante deve fare fallback). */
static int llm_mlx_gdn_prefill_sseq(mlx_array* y_out, mlx_array* state_out, mlx_array* state_seq_out,
                               mlx_array q, mlx_array k, mlx_array v,
                               mlx_array g, mlx_array beta, mlx_array state,
                               int T, int g_vector, mlx_stream stream) {
    if (!y_out || !state_out) return -1;
    const int sseq = (state_seq_out != NULL) ? 1 : 0;
    mlx_dtype dq = mlx_array_dtype(q);
    const int* qs = mlx_array_shape(q);
    const int* vs = mlx_array_shape(v);
    const int* ss = mlx_array_shape(state);
    if (mlx_array_ndim(q) != 4) return -1;
    int B = qs[0], TT = qs[1], Hk = qs[2], Dk = qs[3];
    int Hv = vs[2], Dv = vs[3];
    if (Dk != 128) return -1;
    if (Dv % 32 != 0) return -1;
    if (Hk <= 0 || Hv < Hk || Hv % Hk != 0) return -1;
    if (T != TT) T = TT;
    mlx_array kc = mlx_array_new(), vc = mlx_array_new();
    if (mlx_array_dtype(k) != dq) {
        if (mlx_astype(&kc, k, dq, stream) != 0) { mlx_array_free(kc); mlx_array_free(vc); return -1; }
        k = kc;
    }
    if (mlx_array_dtype(v) != dq) {
        if (mlx_astype(&vc, v, dq, stream) != 0) { mlx_array_free(kc); mlx_array_free(vc); return -1; }
        v = vc;
    }
    int block_t = llm_mlx_gdn_kernel_block_t(dq);
    if (llm_mlx_gdn_kernel_build(block_t, g_vector, sseq) != 0) return -1;
    mlx_array Tarr = mlx_array_new_data(&T, NULL, 0, MLX_INT32);
    mlx_vector_array inputs = mlx_vector_array_new();
    mlx_vector_array_append_value(inputs, q);
    mlx_vector_array_append_value(inputs, k);
    mlx_vector_array_append_value(inputs, v);
    mlx_vector_array_append_value(inputs, g);
    mlx_vector_array_append_value(inputs, beta);
    mlx_vector_array_append_value(inputs, state);
    mlx_vector_array_append_value(inputs, Tarr);
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", dq);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Dk", Dk);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Dv", Dv);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Hk", Hk);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "Hv", Hv);
    mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GVEC", g_vector ? 1 : 0);
    int yshape[4] = {B, TT, Hv, Dv};
    int sshape[4] = {B, Hv, Dv, Dk};
    mlx_fast_metal_kernel_config_add_output_arg(cfg, yshape, 4, dq);
    mlx_fast_metal_kernel_config_add_output_arg(cfg, sshape, 4, MLX_FLOAT32);
    if (sseq) {
        int q5[5] = {B, TT, Hv, Dv, Dk};
        mlx_fast_metal_kernel_config_add_output_arg(cfg, q5, 5, MLX_FLOAT32);
    } else {
        int d1[1] = {1};
        mlx_fast_metal_kernel_config_add_output_arg(cfg, d1, 1, MLX_FLOAT32);
    }
    mlx_fast_metal_kernel_config_set_grid(cfg, 256 * (Dv/32), Hv, B);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1);
    mlx_vector_array outputs = mlx_vector_array_new();
    int rc = mlx_fast_metal_kernel_apply(&outputs, llm_gdn_kernel_singleton[g_vector & 1][sseq & 1].k, inputs, cfg, stream);
    size_t on = mlx_vector_array_size(outputs);
    if (rc != 0) { fprintf(stderr, "gdn_kernel: apply rc=%d out=%zu (fallback loop)\n", rc, on); }
    if (rc == 0 && on == 3) {
        mlx_array y0 = mlx_array_new(), s0 = mlx_array_new(), q0 = mlx_array_new();
        int g0 = mlx_vector_array_get(&y0, outputs, 0);
        int g1 = mlx_vector_array_get(&s0, outputs, 1);
        int g2 = mlx_vector_array_get(&q0, outputs, 2);
        if (g0 == 0 && g1 == 0) {
            mlx_array_set(y_out, y0);
            mlx_array_set(state_out, s0);
            if (sseq && g2 == 0) mlx_array_set(state_seq_out, q0);
        } else { fprintf(stderr, "gdn_kernel: get rc=%d/%d/%d (fallback)\n", g0, g1, g2); rc = -1; }
        mlx_array_free(y0); mlx_array_free(s0); mlx_array_free(q0);
    } else if (rc == 0) { rc = -1; }
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(inputs);
    mlx_vector_array_free(outputs);
    mlx_array_free(Tarr);
    mlx_array_free(kc);
    mlx_array_free(vc);
    return rc == 0 ? 0 : -1;
}

static int llm_mlx_gdn_prefill(mlx_array* y_out, mlx_array* state_out,
                               mlx_array q, mlx_array k, mlx_array v,
                               mlx_array g, mlx_array beta, mlx_array state,
                               int T, int g_vector, mlx_stream stream) {
    return llm_mlx_gdn_prefill_sseq(y_out, state_out, NULL, q, k, v, g, beta, state, T, g_vector, stream);
}

#endif /* LLM_MLX_GDN_KERNEL_H */
