/* llm_dsv41_metal_test.c — parità del kernel Metal Sinkhorn vs la versione semantica MLX.
 * Build: tools/dsv41_metal_test.sh */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mlx/c/mlx.h>
#include "llm_dsv41_mlx.h"
#include "llm_dsv41_fp4.h"

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("llm_dsv41_metal — sinkhorn Metal vs semantico\n");
    if (llm_dsv41_metal_enabled()) { printf("(nota: LLMMETAL_DSV41_METAL attivo; uso il semantico diretto)\n"); }
    mlx_stream s = mlx_default_gpu_stream_new();
    const int N = 2, S = 3;
    const int sh[4] = { N, S, 4, 4 };
    float buf[N * S * 16];
    for (int i = 0; i < N * S * 16; i++) buf[i] = 0.05f + 0.1f * (float)((i * 37) % 23);
    mlx_array comb = mlx_array_new_data(buf, sh, 4, MLX_FLOAT32);
    const float eps = 1e-6f;
    const int iters = 20;

    bool ok = true;
    mlx_array sem = llm_dsv41_mlx_sinkhorn(comb, eps, iters, s, &ok);
    mlx_array met = mlx_array_new();
    bool mok = llm_dsv41_metal_sinkhorn(&met, comb, eps, iters, s);
    int fail = 0;
    if (!ok || !mok) { printf("FAIL %-46s (sem=%d metal=%d)\n", "sinkhorn metal available", ok, mok); return 1; }
    mlx_array_eval(sem); mlx_array_eval(met);
    const float* a = mlx_array_data_float32(sem);
    const float* b = mlx_array_data_float32(met);
    int bad = 0;
    for (int i = 0; i < N * S * 16; i++) {
        float d = fabsf(a[i] - b[i]); float m = fabsf(a[i]) > 1e-3f ? fabsf(a[i]) : 1e-3f;
        if (d > 1e-5f * m) { if (bad++ < 4) printf("  [%d] sem=%.7g metal=%.7g\n", i, a[i], b[i]); }
    }
    printf("%s %-46s\n", bad ? "FAIL" : "PASS", "sinkhorn metal == semantico");
    if (bad) fail++;

    /* hc_post: semantico vs metal */
    {
        const int S = 3, DIM = 8, HC = 4;
        int xs[3] = { 1, S, DIM }, rs2[4] = { 1, S, HC, DIM }, ps[3] = { 1, S, HC }, cs[4] = { 1, S, HC, HC };
        float xb[1*3*DIM], rb[1*3*HC*DIM], pb[1*3*HC], cb[1*3*HC*HC];
        for (int i = 0; i < 1*3*DIM; i++) xb[i] = 0.1f * (float)((i * 13) % 11) - 0.5f;
        for (int i = 0; i < 1*3*HC*DIM; i++) rb[i] = 0.1f * (float)((i * 7) % 17) - 0.8f;
        for (int i = 0; i < 1*3*HC; i++) pb[i] = 0.5f + 0.1f * (float)(i % 5);
        for (int i = 0; i < 1*3*HC*HC; i++) cb[i] = 0.05f + 0.02f * (float)((i * 3) % 13);
        mlx_array x = mlx_array_new_data(xb, xs, 3, MLX_FLOAT32);
        mlx_array res = mlx_array_new_data(rb, rs2, 4, MLX_FLOAT32);
        mlx_array po = mlx_array_new_data(pb, ps, 3, MLX_FLOAT32);
        mlx_array co = mlx_array_new_data(cb, cs, 4, MLX_FLOAT32);
        bool okp = true;
        mlx_array hs = llm_dsv41_mlx_hc_post(x, res, po, co, s, &okp);
        mlx_array hm = mlx_array_new();
        bool mokp = llm_dsv41_metal_hc_post(&hm, x, res, po, co, s);
        if (!okp || !mokp) { printf("FAIL %-46s\n", "hc_post metal available"); fail++; }
        else {
            mlx_array_eval(hs); mlx_array_eval(hm);
            const float* a2 = mlx_array_data_float32(hs); const float* b2 = mlx_array_data_float32(hm);
            int badp = 0; for (int i = 0; i < S*HC*DIM; i++) { float d = fabsf(a2[i]-b2[i]); float mm = fabsf(a2[i])>1e-3f?fabsf(a2[i]):1e-3f; if (d>1e-5f*mm) badp++; }
            printf("%s %-46s\n", badp ? "FAIL" : "PASS", "hc_post metal == semantico");
            if (badp) fail++;
        }
    }
    /* hc_projection: mixes = flat @ fn^T * rsqrt(mean+eps) */
    {
        const int S = 3, D = 32;
        float xb[1*3*D], fb[24*D];
        for (int i = 0; i < 1*3*D; i++) xb[i] = 0.1f * (float)((i * 5) % 9) - 0.4f;
        for (int i = 0; i < 24*D; i++) fb[i] = 0.05f * (float)((i * 11) % 7) - 0.3f;
        int xs2[3] = { 1, S, D }, fs2[2] = { 24, D };
        mlx_array x = mlx_array_new_data(xb, xs2, 3, MLX_FLOAT32);
        mlx_array fn = mlx_array_new_data(fb, fs2, 2, MLX_FLOAT32);
        float eps = 1e-20f;
        /* semantico: matmul + rsqrt */
        mlx_array fnt = mlx_array_new(), mix0 = mlx_array_new(), sq = mlx_array_new(), mn = mlx_array_new(), rs = mlx_array_new(), esca = mlx_array_new_float(eps);
        mlx_transpose(&fnt, fn, s); mlx_matmul(&mix0, x, fnt, s); mlx_square(&sq, x, s);
        mlx_mean_axis(&mn, sq, 2, true, s); mlx_add(&mn, mn, esca, s); mlx_rsqrt(&rs, mn, s); mlx_multiply(&mix0, mix0, rs, s);
        mlx_array mixm = mlx_array_new();
        bool mokp = llm_dsv41_metal_hc_projection(&mixm, x, fn, eps, s);
        if (!mokp) { printf("FAIL %-46s\n", "hc_projection metal available"); fail++; }
        else {
            mlx_array_eval(mix0); mlx_array_eval(mixm);
            const float* a2 = mlx_array_data_float32(mix0); const float* b2 = mlx_array_data_float32(mixm);
            int badp = 0; for (int i = 0; i < S*24; i++) if (fabsf(a2[i]-b2[i]) > 5e-3f) badp++;  /* ordine fma vs matmul MLX */
            printf("%s %-46s\n", badp ? "FAIL" : "PASS", "hc_projection metal == semantico");
            if (badp) fail++;
        }
    }
    /* hc_pre_norm: hc_pre + rmsnorm fusi */
    {
        const int S = 3, D = 32;
        int xs2[4] = { 1, S, 4, D }, ps2[3] = { 1, S, 4 }, ws2[1] = { D };
        float xb[1*3*4*D], pb[1*3*4], wb[D];
        for (int i = 0; i < 1*3*4*D; i++) xb[i] = 0.1f * (float)((i * 5) % 9) - 0.4f;
        for (int i = 0; i < 1*3*4; i++) pb[i] = 0.5f + 0.1f * (float)(i % 5);
        for (int i = 0; i < D; i++) wb[i] = 0.5f + 0.05f * (float)(i % 7);
        float eps = 1e-20f;
        mlx_array x = mlx_array_new_data(xb, xs2, 4, MLX_FLOAT32);
        mlx_array po = mlx_array_new_data(pb, ps2, 3, MLX_FLOAT32);
        mlx_array ww = mlx_array_new_data(wb, ws2, 1, MLX_FLOAT32);
        bool okp = true;
        mlx_array hp = llm_dsv41_mlx_hc_pre(x, po, s, &okp);
        mlx_array hsem = llm_dsv41_mlx_rmsnorm(hp, ww, eps, s, &okp);
        mlx_array hmet = mlx_array_new();
        bool mokp = llm_dsv41_metal_hc_pre_norm(&hmet, x, po, ww, eps, s);
        if (!okp || !mokp) { printf("FAIL %-46s\n", "hc_pre_norm metal available"); fail++; }
        else {
            mlx_array_eval(hsem); mlx_array_eval(hmet);
            const float* a2 = mlx_array_data_float32(hsem); const float* b2 = mlx_array_data_float32(hmet);
            int badp = 0; for (int i = 0; i < S*D; i++) if (fabsf(a2[i]-b2[i]) > 1e-3f) badp++;
            printf("%s %-46s\n", badp ? "FAIL" : "PASS", "hc_pre_norm metal == semantico");
            if (badp) fail++;
        }
    }
    /* fp8 activation round-trip: semantico (pack/unpack) vs kernel */
    {
        const int S = 2, D = 64;
        int xs2[2] = { S, D };
        float xb[S*D];
        for (int i = 0; i < S*D; i++) xb[i] = 0.3f * (float)((i * 7) % 19) - 2.5f;
        mlx_array x = mlx_array_new_data(xb, xs2, 2, MLX_FLOAT32);
        bool okp = true;
        mlx_array fs = llm_dsv41_mlx_fp8_roundtrip(x, D, s, &okp);
        mlx_array fm = mlx_array_new();
        bool mokp = llm_dsv41_metal_fp8_roundtrip(&fm, x, s);
        if (!okp || !mokp) { printf("FAIL %-46s\n", "fp8 roundtrip metal available"); fail++; }
        else {
            mlx_array_eval(fs); mlx_array_eval(fm);
            const float* a2 = mlx_array_data_float32(fs); const float* b2 = mlx_array_data_float32(fm);
            int badp = 0; for (int i = 0; i < S*D; i++) if (fabsf(a2[i]-b2[i]) > 1e-6f) { if(badp<4) fprintf(stderr,"  fp8[%d] sem=%.7g met=%.7g\n",i,a2[i],b2[i]); badp++; }
            printf("%s %-46s\n", badp ? "FAIL" : "PASS", "fp8 roundtrip metal == semantico");
            if (badp) fail++;
        }
    }
    /* packed_index_scores: kernel su chiavi fp4-packed vs semantico */
    {
        const int L = 2, H = 2, D = 32, N = 4;
        const int ksz = D / 2 + D / 32;
        int qs[4] = { 1, L, H, D }, ks[3] = { 1, N, ksz }, ws[3] = { 1, L, H };
        float qb[1*2*2*32], kb[4*32], wb[1*2*2];
        for (int i = 0; i < 1*2*2*32; i++) qb[i] = 0.2f * (float)((i * 3) % 7) - 0.6f;
        for (int i = 0; i < N*D; i++) kb[i] = 0.4f * (float)((i * 5) % 13) - 2.4f;
        for (int i = 0; i < 1*2*2; i++) wb[i] = 0.5f + 0.1f * (float)i;
        unsigned char packed[N*ksz]; float kdec[N*D];
        for (int r = 0; r < N; r++) {
            llmcodec_pack_activation(kb + r*D, D, 4, 32, 0, packed + r*ksz);
            llmcodec_unpack_activation(packed + r*ksz, D, 4, 32, 0, kdec + r*D);
        }
        mlx_array q = mlx_array_new_data(qb, qs, 4, MLX_FLOAT32);
        mlx_array keys = mlx_array_new_data(packed, ks, 3, MLX_UINT8);
        mlx_array w = mlx_array_new_data(wb, ws, 3, MLX_FLOAT32);
        mlx_array scores = mlx_array_new();
        bool mokp = llm_dsv41_metal_index_scores(&scores, q, keys, w, 100, 1, 0,
                                                 mlx_array_new(), 0, s);
        if (!mokp) { printf("FAIL %-46s\n", "index_scores metal available"); fail++; }
        else {
            mlx_array_eval(scores);
            const float* a2 = mlx_array_data_float32(scores);
            int badp = 0;
            for (int l = 0; l < L; l++) for (int r = 0; r < N; r++) {
                float exp = 0.0f;
                for (int h = 0; h < H; h++) {
                    float dot = 0.0f;
                    for (int d = 0; d < D; d++) dot += qb[(l*H+h)*D + d] * kdec[r*D + d];
                    if (dot > 0.0f) exp += dot * wb[l*H + h];
                }
                float got = a2[l*N + r];
                if (fabsf(got - exp) > 1e-3f) { if (badp<4) fprintf(stderr, "  idx[%d,%d] got=%.6g exp=%.6g\n", l, r, got, exp); badp++; }
            }
            printf("%s %-46s\n", badp ? "FAIL" : "PASS", "packed_index_scores == semantico");
            if (badp) fail++;
        }
    }
    /* packed_sparse_attention: window fp8 + pooled fp4 vs semantico */
    {
        const int L = 2, H = 2, D = 32, W = 4, C = 2, NW = 4, NC = 2;
        const int wsz = D + D / 32, psz = D / 2 + D / 16;
        float qb[1*2*2*32], wf[4*32], pf[2*32], sk[2];
        for (int i = 0; i < 1*2*2*32; i++) qb[i] = 0.2f*(float)((i*3)%7)-0.6f;
        for (int i = 0; i < 4*32; i++) wf[i] = 0.5f*(float)((i*5)%11)-2.5f;
        for (int i = 0; i < 2*32; i++) pf[i] = 0.3f*(float)((i*7)%9)-1.2f;
        for (int i = 0; i < 2; i++) sk[i] = 0.1f*(float)i;
        int wis[3]={1,L,W}, cis[3]={1,L,C}, qs[4]={1,L,H,D}, ws2[3]={1,NW,wsz}, ps2[3]={1,NC,psz}, ss[1]={H};
        unsigned char wpack[NW*wsz], ppack[NC*psz]; float wdec[NW*32], pdec[NC*32];
        for (int r = 0; r < NW; r++) { llmcodec_pack_fp8(wf + r*D, D, wpack + r*wsz); llmcodec_unpack_fp8(wpack + r*wsz, D, wdec + r*D); }
        for (int r = 0; r < NC; r++) { llmcodec_pack_activation(pf + r*D, D, 4, 16, 1, ppack + r*psz); llmcodec_unpack_activation(ppack + r*psz, D, 4, 16, 1, pdec + r*D); }
        int32_t wi_b[1*2*4], ci_b[1*2*2];
        for (int l = 0; l < L; l++) for (int j = 0; j < W; j++) wi_b[l*W+j] = j % NW;
        for (int l = 0; l < L; l++) for (int j = 0; j < C; j++) ci_b[l*C+j] = j % NC;
        mlx_array q = mlx_array_new_data(qb, qs, 4, MLX_FLOAT32);
        mlx_array wnd = mlx_array_new_data(wpack, ws2, 3, MLX_UINT8);
        mlx_array poo = mlx_array_new_data(ppack, ps2, 3, MLX_UINT8);
        mlx_array wi = mlx_array_new_data(wi_b, wis, 3, MLX_INT32);
        mlx_array ci = mlx_array_new_data(ci_b, cis, 3, MLX_INT32);
        mlx_array sink = mlx_array_new_data(sk, ss, 1, MLX_FLOAT32);
        float scale = 1.0f / sqrtf((float)D);
        mlx_array out = mlx_array_new();
        bool mokp = llm_dsv41_metal_packed_attention(&out, q, wnd, poo, wi, ci, sink, scale, s);
        if (!mokp) { printf("FAIL %-46s\n", "packed_attention metal available"); fail++; }
        else {
            mlx_array_eval(out);
            const float* o = mlx_array_data_float32(out);
            int badp = 0;
            for (int l = 0; l < L; l++) for (int h = 0; h < H; h++) {
                float keys[6][32]; int nk = 0;
                for (int j = 0; j < W; j++) { int r = wi_b[l*W+j]; for (int d=0; d<D; d++) keys[nk][d] = wdec[r*D+d]; nk++; }
                for (int j = 0; j < C; j++) { int r = ci_b[l*C+j]; for (int d=0; d<D; d++) keys[nk][d] = pdec[r*D+d]; nk++; }
                float logits[8], m = sk[h];
                for (int i = 0; i < nk; i++) { float dot=0; for (int d=0; d<D; d++) dot += qb[(l*H+h)*D+d]*keys[i][d]; logits[i]=dot*scale; if (logits[i]>m) m=logits[i]; }
                float den = expf(sk[h]-m); for (int i=0;i<nk;i++) den += expf(logits[i]-m);
                for (int d = 0; d < D; d++) {
                    float acc=0; for (int i=0;i<nk;i++) acc += (expf(logits[i]-m)/den)*keys[i][d];
                    float got = o[((l*H+h))*D + d];
                    if (fabsf(got-acc) > 1e-3f) { if (badp<4) fprintf(stderr,"  att[%d,%d,%d] got=%.6g exp=%.6g\n",l,h,d,got,acc); badp++; }
                }
            }
            printf("%s %-46s\n", badp ? "FAIL" : "PASS", "packed_sparse_attention == semantico");
            if (badp) fail++;
        }
    }
    /* packed_index_scores MMA (H>=8, D<=128): parità vs semantico */
    {
        const int L = 2, H = 8, D = 32, N = 8;
        const int ksz = D / 2 + D / 32;
        int qs[4] = { 1, L, H, D }, ks[3] = { 1, N, ksz }, ws[3] = { 1, L, H };
        float qb[1*2*8*32], kb[8*32], wb[1*2*8];
        for (int i = 0; i < 1*2*8*32; i++) qb[i] = 0.2f*(float)((i*3)%7)-0.6f;
        for (int i = 0; i < N*D; i++) kb[i] = 0.4f*(float)((i*5)%13)-2.4f;
        for (int i = 0; i < 1*2*8; i++) wb[i] = 0.5f + 0.1f*(float)(i%3);
        unsigned char packed[N*ksz]; float kdec[N*D];
        for (int r = 0; r < N; r++) { llmcodec_pack_activation(kb+r*D, D, 4, 32, 0, packed+r*ksz); llmcodec_unpack_activation(packed+r*ksz, D, 4, 32, 0, kdec+r*D); }
        mlx_array q = mlx_array_new_data(qb, qs, 4, MLX_FLOAT32);
        mlx_array keys = mlx_array_new_data(packed, ks, 3, MLX_UINT8);
        mlx_array w = mlx_array_new_data(wb, ws, 3, MLX_FLOAT32);
        mlx_array scores = mlx_array_new();
        bool mokp = llm_dsv41_metal_index_scores(&scores, q, keys, w, 100, 1, 0, mlx_array_new(), 0, s);
        if (!mokp) { printf("FAIL %-46s\n", "index_scores MMA available"); fail++; }
        else {
            mlx_array_eval(scores);
            const float* a2 = mlx_array_data_float32(scores);
            int badp = 0;
            for (int l = 0; l < L; l++) for (int r = 0; r < N; r++) {
                float exp = 0.0f;
                for (int h = 0; h < H; h++) { float dot=0; for (int d=0; d<D; d++) dot += qb[(l*H+h)*D+d]*kdec[r*D+d]; if (dot>0) exp += dot*wb[l*H+h]; }
                if (fabsf(a2[l*N+r]-exp) > 1e-3f) { if (badp<4) fprintf(stderr,"  idxm[%d,%d] got=%.6g exp=%.6g\n",l,r,a2[l*N+r],exp); badp++; }
            }
            printf("%s %-46s\n", badp ? "FAIL" : "PASS", "packed_index_scores MMA == semantico");
            if (badp) fail++;
        }
    }
    /* packed_sparse_attention MMA (L>8): parità vs semantico */
    {
        const int L = 9, H = 2, D = 32, W = 4, C = 1, NW = 4, NC = 1;
        const int wsz = D + D / 32, psz = D / 2 + D / 16;
        float qb[9*2*32], wf[4*32], pf[1*32], sk[2];
        for (int i = 0; i < 9*2*32; i++) qb[i] = 0.2f*(float)((i*3)%7)-0.6f;
        for (int i = 0; i < 4*32; i++) wf[i] = 0.5f*(float)((i*5)%11)-2.5f;
        for (int i = 0; i < 1*32; i++) pf[i] = 0.3f*(float)((i*7)%9)-1.2f;
        for (int i = 0; i < 2; i++) sk[i] = 0.1f*(float)i;
        int wis[3]={1,L,W}, cis[3]={1,L,C}, qs[4]={1,L,H,D}, ws2[3]={1,NW,wsz}, ps2[3]={1,NC,psz}, ss[1]={H};
        unsigned char wpack[NW*wsz], ppack[NC*psz]; float wdec[NW*32], pdec[NC*32];
        for (int r = 0; r < NW; r++) { llmcodec_pack_fp8(wf+r*D, D, wpack+r*wsz); llmcodec_unpack_fp8(wpack+r*wsz, D, wdec+r*D); }
        for (int r = 0; r < NC; r++) { llmcodec_pack_activation(pf+r*D, D, 4, 16, 1, ppack+r*psz); llmcodec_unpack_activation(ppack+r*psz, D, 4, 16, 1, pdec+r*D); }
        int32_t wi_b[9*4], ci_b[9*1]; for (int l=0;l<L;l++) for (int j=0;j<W;j++) wi_b[l*W+j]=j%NW;
        for (int l=0;l<L;l++) ci_b[l]=0;
        mlx_array q = mlx_array_new_data(qb, qs, 4, MLX_FLOAT32);
        mlx_array wnd = mlx_array_new_data(wpack, ws2, 3, MLX_UINT8);
        mlx_array wi = mlx_array_new_data(wi_b, wis, 3, MLX_INT32);
        mlx_array poo = mlx_array_new_data(ppack, ps2, 3, MLX_UINT8);
        mlx_array ci = mlx_array_new_data(ci_b, cis, 3, MLX_INT32);
        mlx_array sink = mlx_array_new_data(sk, ss, 1, MLX_FLOAT32);
        float scale = 1.0f/sqrtf((float)D);
        mlx_array out = mlx_array_new();
        bool mokp = llm_dsv41_metal_packed_attention(&out, q, wnd, poo, wi, ci, sink, scale, s);
        if (!mokp) { printf("FAIL %-46s\n", "packed_attention MMA available"); fail++; }
        else {
            mlx_array_eval(out);
            const float* o = mlx_array_data_float32(out);
            int badp = 0;
            for (int l=0;l<L;l++) for (int h=0;h<H;h++) {
                float keys[8][32]; int nk=0;
                for (int j=0;j<W;j++){ int r=wi_b[l*W+j]; for(int d=0;d<D;d++) keys[nk][d]=wdec[r*D+d]; nk++; }
                for (int j=0;j<C;j++){ int r=ci_b[l*C+j]; for(int d=0;d<D;d++) keys[nk][d]=pdec[r*D+d]; nk++; }
                float logits[8], m=sk[h];
                for (int i=0;i<nk;i++){ float dot=0; for(int d=0;d<D;d++) dot += qb[(l*H+h)*D+d]*keys[i][d]; logits[i]=dot*scale; if(logits[i]>m)m=logits[i]; }
                float den=expf(sk[h]-m); for(int i=0;i<nk;i++) den+=expf(logits[i]-m);
                for (int d=0;d<D;d++){ float acc=0; for(int i=0;i<nk;i++) acc += (expf(logits[i]-m)/den)*keys[i][d];
                    if (fabsf(o[((l*H+h))*D+d]-acc)>1e-3f){ if(badp<4) fprintf(stderr,"  attm[%d,%d,%d] got=%.6g exp=%.6g\n",l,h,d,o[((l*H+h))*D+d],acc); badp++; } }
            }
            printf("%s %-46s\n", badp ? "FAIL" : "PASS", "packed_sparse_attention MMA == semantico");
            if (badp) fail++;
        }
    }
    /* tile_topk: kernel vs top-k C diretto */
    {
        const int L = 3, M = 8, K = 3;
        int ss[3] = { 1, L, M };
        float sb[3*8];
        for (int i = 0; i < L*M; i++) sb[i] = sinf((float)i*1.7f) + 0.013f*(float)i;
        mlx_array sc = mlx_array_new_data(sb, ss, 3, MLX_FLOAT32);
        mlx_array ks = mlx_array_new(), ki = mlx_array_new();
        bool mokp = llm_dsv41_metal_tile_topk(&ks, &ki, sc, K, s);
        if (!mokp) { printf("FAIL %-46s\n", "tile_topk metal available"); fail++; }
        else {
            mlx_array_eval(ki);
            const int* b2 = mlx_array_data_int32(ki);
            int badp = 0;
            for (int l = 0; l < L; l++) {
                int want[8]; float ws[8];
                for (int i = 0; i < M; i++) { want[i]=i; ws[i]=sb[l*M+i]; }
                for (int x=0;x<M;x++) for (int y=x+1;y<M;y++) if (ws[y]>ws[x] || (ws[y]==ws[x] && want[y]<want[x])) { float t=ws[x];ws[x]=ws[y];ws[y]=t; int ti=want[x];want[x]=want[y];want[y]=ti; }
                int have[8]; for (int j=0;j<K;j++) have[j]=b2[l*K+j];
                for (int x=0;x<K;x++) for (int y=x+1;y<K;y++) if (have[y]<have[x]){int t=have[x];have[x]=have[y];have[y]=t;}
                int w2[8]; for (int j=0;j<K;j++) w2[j]=want[j];
                for (int x=0;x<K;x++) for (int y=x+1;y<K;y++) if (w2[y]<w2[x]){int t=w2[x];w2[x]=w2[y];w2[y]=t;}
                for (int j=0;j<K;j++) if (have[j]!=w2[j]) { if (badp<4) fprintf(stderr,"  topk l=%d have=%d want=%d\n",l,have[j],w2[j]); badp++; }
            }
            printf("%s %-46s\n", badp ? "FAIL" : "PASS", "tile_topk == top-k C");
            if (badp) fail++;
        }
    }
    printf(fail ? "FAILURES: %d\n" : "ALL PASS\n", fail);
    return fail;
}
