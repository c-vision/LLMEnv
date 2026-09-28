/* llm_mlx_blockskip.h — Block-skip KV (port di DS4 `kernel_flash_attn_ext_blk`) e padding del
 * blocco finale (`kernel_flash_attn_ext_pad`). Header-only, API MLX C.
 *
 * Obiettivo: a contesto lungo una parte dei blocchi KV e' interamente mascherata (causale/finestra/
 * top-k sparse): il flag per-blocco permette di SALTARLI. Semantica ds4 (mask additiva f32,
 * 0=attend, -inf=masked): per (query-block, key-block) -> 1=attivo, 2=tutto-zero, 0=tutto-mascherato.
 * Verifica: llm_mlx_blockskip_test.c vs calcolo diretto. */
#ifndef LLM_MLX_BLOCKSKIP_H
#define LLM_MLX_BLOCKSKIP_H

#include <stdlib.h>
#include <stdbool.h>

#include <mlx/c/mlx.h>

#define LLM_BLOCKSKIP_FNEG (1e30f)

/* Data una maschera ADDITIVA f32 [., ., S, K] (0=attend, -inf=masked), calcola i flag per blocco
 * (bq x bk): int8 [nbq, nbk] con 1=attivo, 2=tutto-zero, 0=tutto-mascherato. */
static inline bool llm_mlx_block_skip_flags(mlx_array* out, mlx_array mask, int bq, int bk,
                                            mlx_stream s) {
    int nd = (int)mlx_array_ndim(mask);
    if (nd < 2 || bq <= 0 || bk <= 0) return false;
    const int* sh = mlx_array_shape(mask);
    int S = sh[nd - 2], K = sh[nd - 1];
    int nbq = (S + bq - 1) / bq, nbk = (K + bk - 1) / bk;
    if (nbq <= 0 || nbk <= 0) return false;
    /* pad S,K ai multipli di blocco con -inf */
    mlx_array padded = mlx_array_new();
    {
        int axes[2] = { nd - 2, nd - 1 };
        int lo[2] = { 0, 0 }, hi[2] = { nbq * bq - S, nbk * bk - K };
        mlx_array pv = mlx_array_new_float(-INFINITY);
        if (mlx_pad(&padded, mask, axes, 2, lo, 2, hi, 2, pv, "constant", s) != 0) { mlx_array_free(pv); return false; }
        mlx_array_free(pv);
    }
    /* reshape [.., nbq, bq, nbk, bk] */
    int rsh[8], rn = 0;
    for (int i = 0; i < nd - 2; i++) rsh[rn++] = sh[i];
    rsh[rn++] = nbq; rsh[rn++] = bq; rsh[rn++] = nbk; rsh[rn++] = bk;
    mlx_array blocks = mlx_array_new();
    if (mlx_reshape(&blocks, padded, rsh, (size_t)rn, s) != 0) return false;
    /* max over bk (axis rn-1) keepdims, then max over bq (axis rn-3) keepdims -> mmax [..,nbq,1,nbk,1]
     * min analogo per mmin */
    mlx_array mmax = mlx_array_new(), mmin = mlx_array_new(), t = mlx_array_new();
    if (mlx_max_axis(&t, blocks, rn - 1, true, s) != 0) return false;
    if (mlx_max_axis(&mmax, t, rn - 3, true, s) != 0) return false;
    if (mlx_min_axis(&t, blocks, rn - 1, true, s) != 0) return false;
    if (mlx_min_axis(&mmin, t, rn - 3, true, s) != 0) return false;
    /* squeeze a [.., nbq, nbk] */
    mlx_array smax = mlx_array_new(), smin = mlx_array_new();
    if (mlx_squeeze_axis(&smax, mmax, rn - 1, s) != 0) return false;
    if (mlx_squeeze_axis(&smax, smax, rn - 3, s) != 0) return false;
    if (mlx_squeeze_axis(&smin, mmin, rn - 1, s) != 0) return false;
    if (mlx_squeeze_axis(&smin, smin, rn - 3, s) != 0) return false;
    /* flags: all_masked = smax <= -inf  ; all_zero = (smin==0 && smax==0) */
    mlx_array neginf = mlx_array_new_float(-INFINITY), zero = mlx_array_new_float(0.0f);
    mlx_array all_masked = mlx_array_new(), all_zero = mlx_array_new();
    mlx_array zmin = mlx_array_new(), zmax = mlx_array_new(), az = mlx_array_new();
    if (mlx_less_equal(&all_masked, smax, neginf, s) != 0) return false;
    if (mlx_equal(&zmin, smin, zero, s) != 0) return false;
    if (mlx_equal(&zmax, smax, zero, s) != 0) return false;
    if (mlx_logical_and(&az, zmin, zmax, s) != 0) return false;
    all_zero = az;
    mlx_array one = mlx_array_new_float(1.0f), two = mlx_array_new_float(2.0f), zero_f = mlx_array_new_float(0.0f);
    mlx_array f1 = mlx_array_new(), f2 = mlx_array_new();
    if (mlx_where(&f1, all_zero, two, one, s) != 0) return false;      /* zero? 2 : 1 */
    if (mlx_where(&f2, all_masked, zero_f, f1, s) != 0) return false;  /* masked? 0 : ... */
    if (mlx_astype(out, f2, MLX_INT8, s) != 0) return false;
    return true;
}

/* Active (uint8/bool) da keep-mask booleana [., ., S, K] (true=attend): OR su ogni blocco bq x bk. */
static inline bool llm_mlx_block_skip_active(mlx_array* out, mlx_array keep, int bq, int bk,
                                             mlx_stream s) {
    int nd = (int)mlx_array_ndim(keep);
    if (nd < 2 || bq <= 0 || bk <= 0) return false;
    const int* sh = mlx_array_shape(keep);
    int S = sh[nd - 2], K = sh[nd - 1];
    int nbq = (S + bq - 1) / bq, nbk = (K + bk - 1) / bk;
    mlx_array b = mlx_array_new(), k8 = mlx_array_new();
    if (mlx_astype(&k8, keep, MLX_FLOAT32, s) != 0) return false;
    mlx_array padded = mlx_array_new();
    {
        int axes[2] = { nd - 2, nd - 1 };
        int lo[2] = { 0, 0 }, hi[2] = { nbq * bq - S, nbk * bk - K };
        mlx_array pv = mlx_array_new_float(0.0f);
        if (mlx_pad(&padded, k8, axes, 2, lo, 2, hi, 2, pv, "constant", s) != 0) return false;
    }
    int rsh[8], rn = 0;
    for (int i = 0; i < nd - 2; i++) rsh[rn++] = sh[i];
    rsh[rn++] = nbq; rsh[rn++] = bq; rsh[rn++] = nbk; rsh[rn++] = bk;
    if (mlx_reshape(&b, padded, rsh, (size_t)rn, s) != 0) return false;
    mlx_array t = mlx_array_new(), m = mlx_array_new(), sm = mlx_array_new(), sm2 = mlx_array_new();
    if (mlx_max_axis(&t, b, rn - 1, true, s) != 0) return false;
    if (mlx_max_axis(&m, t, rn - 3, true, s) != 0) return false;
    if (mlx_squeeze_axis(&sm, m, rn - 1, s) != 0) return false;
    if (mlx_squeeze_axis(&sm2, sm, rn - 3, s) != 0) return false;
    mlx_array half = mlx_array_new_float(0.5f);
    if (mlx_greater(&sm2, sm2, half, s) != 0) return false;
    if (mlx_astype(out, sm2, MLX_UINT8, s) != 0) return false;
    return true;
}

/* Padding dell'ultima dimensione a multiplo di `block` con `value` (kernel_flash_attn_ext_pad). */
static inline bool llm_mlx_pad_final_block(mlx_array* out, mlx_array x, int block, float value,
                                           mlx_stream s) {
    int nd = (int)mlx_array_ndim(x);
    if (nd < 1 || block <= 0) return false;
    const int* sh = mlx_array_shape(x);
    int L = sh[nd - 1];
    int pad = (block - (L % block)) % block;
    if (pad == 0) { if (mlx_array_set(out, x) != 0) return false; return true; }
    int axes[1] = { nd - 1 }, lo[1] = { 0 }, hi[1] = { pad };
    mlx_array pv = mlx_array_new_float(value);
    int rc = mlx_pad(out, x, axes, 1, lo, 1, hi, 1, pv, "constant", s);
    mlx_array_free(pv);
    return rc == 0;
}

/* Attenzione a blocchi che SALTA i blocchi KV interamente mascherati (consumer dei flag).
 * q [B,H,S,D] f32 · k,v [B,H,K,D] f32 · addmask [B,1,S,K] f32 (0=attend, -inf=masked) · scale.
 * out [B,H,S,D]. Parità con SDPA densa: i blocchi saltati sono tutti -inf (contributo 0). */
#define GBS_RET(op, res, ...) do { if ((op)(&(res), __VA_ARGS__, s) != 0) { return false; } } while (0)

static inline bool gbs_slice_axis(mlx_array* res, mlx_array a, int axis, int lo, int hi, mlx_stream s) {
    int nd = (int)mlx_array_ndim(a);
    const int* sh = mlx_array_shape(a);
    int st[8], sp[8], sd[8];
    for (int i = 0; i < nd; i++) { st[i] = (i == axis) ? lo : 0; sp[i] = (i == axis) ? hi : sh[i]; sd[i] = 1; }
    return mlx_slice(res, a, st, (size_t)nd, sp, (size_t)nd, sd, (size_t)nd, s) == 0;
}

static inline bool llm_mlx_attention_blockskip(mlx_array* out, mlx_array q, mlx_array k, mlx_array v,
                                               float scale, mlx_array addmask, int bq, int bk,
                                               mlx_stream s) {
    int qnd = (int)mlx_array_ndim(q);
    if (qnd != 4 || bq <= 0 || bk <= 0) return false;
    const int* qsh = mlx_array_shape(q);
    const int* ksh = mlx_array_shape(k);
    int B = qsh[0], H = qsh[1], S = qsh[2], D = qsh[3], K = ksh[2];
    if (ksh[0] != B) return false;
    /* GQA: k/v possono avere n_kv_heads < H; replica a H (H % nkv == 0). */
    mlx_array k2 = mlx_array_new(), v2 = mlx_array_new();
    if (ksh[1] != H) {
        if (ksh[1] <= 0 || H % ksh[1]) return false;
        if (mlx_broadcast_to(&k2, k, (const int[]){ B, H, K, D }, 4, s) != 0) return false;
        if (mlx_broadcast_to(&v2, v, (const int[]){ B, H, K, D }, 4, s) != 0) return false;
        k = k2; v = v2;
    }
    int nbq = (S + bq - 1) / bq, nbk = (K + bk - 1) / bk;
    int Sp = nbq * bq, Kp = nbk * bk;

    /* pad q a Sp (axis 2) e addmask a [B,1,Sp,Kp] (-inf); k,v a Kp */
    mlx_array qp = mlx_array_new(), kp = mlx_array_new(), vp = mlx_array_new(), mp = mlx_array_new();
    {
        mlx_array z = mlx_array_new_float(0.0f);
        int aq[1] = { 2 }, lo1[1] = { 0 }, hi1[1] = { Sp - S };
        if (mlx_pad(&qp, q, aq, 1, lo1, 1, hi1, 1, z, "constant", s) != 0) return false;
        int ak[1] = { 2 }, hik[1] = { Kp - K };
        if (mlx_pad(&kp, k, ak, 1, lo1, 1, hik, 1, z, "constant", s) != 0) return false;
        if (mlx_pad(&vp, v, ak, 1, lo1, 1, hik, 1, z, "constant", s) != 0) return false;
        mlx_array neg = mlx_array_new_float(-INFINITY);
        int am[2] = { 2, 3 }, lo2[2] = { 0, 0 }, hi2[2] = { Sp - S, Kp - K };
        if (mlx_pad(&mp, addmask, am, 2, lo2, 2, hi2, 2, neg, "constant", s) != 0) return false;
        mlx_array_free(z); mlx_array_free(neg);
    }
    /* active flags [B,1,nbq,nbk] */
    mlx_array keep = mlx_array_new(), active = mlx_array_new();
    {
        mlx_array z = mlx_array_new_float(0.0f), eq = mlx_array_new();
        GBS_RET(mlx_equal, eq, mp, z);
        GBS_RET(mlx_astype, keep, eq, MLX_BOOL);
        mlx_array_free(z);
        if (!llm_mlx_block_skip_active(&active, keep, bq, bk, s)) return false;
    }
    mlx_array_eval(active);
    const uint8_t* af = mlx_array_data_uint8(active);
    if (!af) return false;

    /* accumula per query-block */
    mlx_vector_array parts = mlx_vector_array_new();
    mlx_array mneg = mlx_array_new_float(-INFINITY), zero = mlx_array_new_float(0.0f);
    for (int qi = 0; qi < nbq; qi++) {
        mlx_array qblk = mlx_array_new(), mblkq = mlx_array_new();
        if (!gbs_slice_axis(&qblk, qp, 2, qi * bq, (qi + 1) * bq, s)) return false;
        if (!gbs_slice_axis(&mblkq, mp, 2, qi * bq, (qi + 1) * bq, s)) return false;
        mlx_array m = mlx_array_new(), l = mlx_array_new(), acc = mlx_array_new();
        GBS_RET(mlx_broadcast_to, m, mneg, (const int[]){ B, H, bq, 1 }, 4);
        GBS_RET(mlx_broadcast_to, l, zero, (const int[]){ B, H, bq, 1 }, 4);
        GBS_RET(mlx_zeros, acc, (const int[]){ B, H, bq, D }, 4, MLX_FLOAT32);
        for (int kj = 0; kj < nbk; kj++) {
            if (af[qi * nbk + kj] == 0) continue;   /* SKIP blocco tutto-mascherato */
            mlx_array kblk = mlx_array_new(), vblk = mlx_array_new(), mblk = mlx_array_new();
            if (!gbs_slice_axis(&kblk, kp, 2, kj * bk, (kj + 1) * bk, s)) return false;
            if (!gbs_slice_axis(&vblk, vp, 2, kj * bk, (kj + 1) * bk, s)) return false;
            if (!gbs_slice_axis(&mblk, mblkq, 3, kj * bk, (kj + 1) * bk, s)) return false;
            /* sc = qblk·kblk^T * scale + mblk */
            mlx_array sc = mlx_array_new();
            {
                mlx_vector_array ops = mlx_vector_array_new();
                mlx_vector_array_append_value(ops, qblk);
                mlx_vector_array_append_value(ops, kblk);
                if (mlx_einsum(&sc, "bhsd,bhkd->bhsk", ops, s) != 0) { mlx_vector_array_free(ops); return false; }
                mlx_vector_array_free(ops);
            }
            mlx_array scl = mlx_array_new_float(scale), sc2 = mlx_array_new();
            GBS_RET(mlx_multiply, sc2, sc, scl);
            GBS_RET(mlx_add, sc2, sc2, mblk);
            /* mnew = max(m, max(sc2,-1,keep)) */
            mlx_array rowmax = mlx_array_new(), mnew = mlx_array_new();
            GBS_RET(mlx_max_axis, rowmax, sc2, 3, true);
            GBS_RET(mlx_maximum, mnew, m, rowmax);
            mlx_array p = mlx_array_new(), corr = mlx_array_new(), psum = mlx_array_new();
            GBS_RET(mlx_subtract, p, sc2, mnew);
            GBS_RET(mlx_exp, p, p);
            GBS_RET(mlx_subtract, corr, m, mnew);
            GBS_RET(mlx_exp, corr, corr);
            GBS_RET(mlx_sum_axis, psum, p, 3, true);
            mlx_array l1 = mlx_array_new(), l2 = mlx_array_new();
            GBS_RET(mlx_multiply, l1, l, corr);
            GBS_RET(mlx_add, l2, l1, psum); l = l2;
            mlx_array pv = mlx_array_new(), a1 = mlx_array_new(), a2 = mlx_array_new();
            {
                mlx_vector_array ops = mlx_vector_array_new();
                mlx_vector_array_append_value(ops, p);
                mlx_vector_array_append_value(ops, vblk);
                if (mlx_einsum(&pv, "bhsk,bhkd->bhsd", ops, s) != 0) { mlx_vector_array_free(ops); return false; }
                mlx_vector_array_free(ops);
            }
            GBS_RET(mlx_multiply, a1, acc, corr);
            GBS_RET(mlx_add, a2, a1, pv); acc = a2;
            m = mnew;
        }
        mlx_array oblk = mlx_array_new();
        GBS_RET(mlx_divide, oblk, acc, l);
        mlx_vector_array_append_value(parts, oblk);
    }
    mlx_array catp = mlx_array_new();
    if (mlx_concatenate_axis(&catp, parts, 2, s) != 0) return false;
    mlx_vector_array_free(parts);
    /* slice a S */
    if (S == Sp) { int rc = mlx_array_set(out, catp) == 0; if (rc) mlx_array_eval(*out); return rc; }
    if (!gbs_slice_axis(out, catp, 2, 0, S, s)) return false;
    mlx_array_eval(*out);
    return true;
}

#endif /* LLM_MLX_BLOCKSKIP_H */
