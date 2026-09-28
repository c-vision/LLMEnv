/* llm_mlx_blockskip_test.c — verifica block-skip (DS4 kernel_flash_attn_ext_blk) e padding blocco
 * vs calcolo diretto. Build: tools/llm_mlx_blockskip_test.sh */
#include <stdio.h>
#include <math.h>
#include <mlx/c/mlx.h>
#include "llm_mlx_blockskip.h"

#define S 8
#define K 8
#define W 8   /* finestra = tutta: triangolare inferiore */

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("llm_mlx_blockskip — block-skip vs diretto\n");
    mlx_stream st = mlx_default_gpu_stream_new();
    /* keep-mask causale+finestra */
    bool keep[S][K];
    float addm[S][K];
    for (int i = 0; i < S; i++) for (int j = 0; j < K; j++) {
        bool k = (j <= i) && (j >= i - W + 1);
        keep[i][j] = k;
        addm[i][j] = k ? 0.0f : -INFINITY;
    }
    mlx_array keepa = mlx_array_new_data(keep, (const int[]){ 1, 1, S, K }, 4, MLX_BOOL);
    mlx_array mada  = mlx_array_new_data(addm, (const int[]){ 1, 1, S, K }, 4, MLX_FLOAT32);
    const int bq = 2, bk = 2, nbq = (S + bq - 1) / bq, nbk = (K + bk - 1) / bk;
    int fail = 0;

    /* 1) active */
    mlx_array act = mlx_array_new();
    if (!llm_mlx_block_skip_active(&act, keepa, bq, bk, st)) { printf("FAIL active call\n"); return 1; }
    mlx_array_eval(act);
    const uint8_t* a = mlx_array_data_uint8(act);
    int bad = 0;
    for (int qi = 0; qi < nbq; qi++) for (int kj = 0; kj < nbk; kj++) {
        int any = 0;
        for (int i = 0; i < bq; i++) for (int j = 0; j < bk; j++) {
            int r = qi * bq + i, c = kj * bk + j;
            if (r < S && c < K && keep[r][c]) any = 1;
        }
        if (a[qi * nbk + kj] != (any ? 1 : 0)) { if (bad < 4) printf("  act[%d,%d]=%d exp %d\n", qi, kj, a[qi*nbk+kj], any); bad++; }
    }
    printf("%s %-46s\n", bad ? "FAIL" : "PASS", "block_skip_active == diretto"); if (bad) fail++;

    /* 2) flags {1 attivo, 2 tutto-zero, 0 tutto-mascherato} */
    mlx_array fl = mlx_array_new();
    if (!llm_mlx_block_skip_flags(&fl, mada, bq, bk, st)) { printf("FAIL flags call\n"); return 1; }
    mlx_array_eval(fl);
    const int8_t* f = mlx_array_data_int8(fl);
    bad = 0;
    for (int qi = 0; qi < nbq; qi++) for (int kj = 0; kj < nbk; kj++) {
        float mn = INFINITY, mx = -INFINITY;
        for (int i = 0; i < bq; i++) for (int j = 0; j < bk; j++) {
            int r = qi * bq + i, c = kj * bk + j;
            float v = (r < S && c < K) ? addm[r][c] : -INFINITY;
            if (v < mn) mn = v; if (v > mx) mx = v;
        }
        int exp = (mx <= -INFINITY) ? 0 : ((mn == 0.0f && mx == 0.0f) ? 2 : 1);
        if (f[qi * nbk + kj] != exp) { if (bad < 4) printf("  fl[%d,%d]=%d exp %d\n", qi, kj, f[qi*nbk+kj], exp); bad++; }
    }
    printf("%s %-46s\n", bad ? "FAIL" : "PASS", "block_skip_flags == diretto"); if (bad) fail++;

    /* 3) padding blocco finale */
    {
        float xb[5] = { 1, 2, 3, 4, 5 };
        mlx_array x = mlx_array_new_data(xb, (const int[]){ 5 }, 1, MLX_FLOAT32);
        mlx_array p = mlx_array_new();
        if (!llm_mlx_pad_final_block(&p, x, 4, -1.0f, st)) { printf("FAIL pad call\n"); fail++; }
        else {
            mlx_array_eval(p);
            const float* d = mlx_array_data_float32(p);
            int ok = ((int)mlx_array_shape(p)[0] == 8) && d[0]==1 && d[4]==5 && d[5]==-1.0f && d[7]==-1.0f;
            printf("%s %-46s\n", ok ? "PASS" : "FAIL", "pad_final_block"); if (!ok) fail++;
        }
    }
    /* 4) consumer: attenzione a blocchi (skip) vs softmax densa diretta */
    {
        const int B=1,H=1,Sq=3,Kq=3,Dq=2, bq=2, bk=2;
        float qb[1*1*3*2], kb[1*1*3*2], vb[1*1*3*2], mb[1*1*3*3];
        for (int i=0;i<1*1*3*2;i++){ qb[i]=0.1f*(float)((i*3)%7)-0.3f; kb[i]=0.1f*(float)((i*5)%9)-0.4f; vb[i]=0.1f*(float)((i*7)%11)-0.5f; }
        for (int i=0;i<Sq;i++) for (int j=0;j<Kq;j++) mb[i*Kq+j] = (j<=i)?0.0f:-INFINITY;
        mlx_array q = mlx_array_new_data(qb,(const int[]){B,H,Sq,Dq},4,MLX_FLOAT32);
        mlx_array k = mlx_array_new_data(kb,(const int[]){B,H,Kq,Dq},4,MLX_FLOAT32);
        mlx_array v = mlx_array_new_data(vb,(const int[]){B,H,Kq,Dq},4,MLX_FLOAT32);
        mlx_array ma= mlx_array_new_data(mb,(const int[]){B,1,Sq,Kq},4,MLX_FLOAT32);
        float scale = 1.0f/sqrtf((float)Dq);
        mlx_array out = mlx_array_new();
        if (!llm_mlx_attention_blockskip(&out, q,k,v, scale, ma, bq, bk, st)) { printf("FAIL blockskip consumer call\n"); fail++; }
        else {
            mlx_array_eval(out);
            const float* o = mlx_array_data_float32(out);
            int badp=0;
            for (int i=0;i<Sq;i++){
                float sc[3], mx=-1e30f;
                for (int j=0;j<Kq;j++){ float d=0; for(int e=0;e<Dq;e++) d += qb[i*Dq+e]*kb[j*Dq+e]; sc[j]=d*scale + mb[i*Kq+j]; if(sc[j]>mx)mx=sc[j]; }
                float den=0; for(int j=0;j<Kq;j++){ sc[j]=expf(sc[j]-mx); den+=sc[j]; }
                for (int e=0;e<Dq;e++){ float acc=0; for(int j=0;j<Kq;j++) acc += (sc[j]/den)*vb[j*Dq+e];
                    float got=o[i*Dq+e]; if (fabsf(got-acc)>1e-3f){ if(badp<4) printf("  bs[%d,%d] got=%.6g exp=%.6g\n",i,e,got,acc); badp++; } }
            }
            printf("%s %-46s\n", badp?"FAIL":"PASS", "attention_blockskip == softmax densa"); if(badp) fail++;
        }
    }
    /* 5) consumer GQA (nh=2, nkv=1) */
    {
        const int B=1,Hh=2,nkv=1,Sq=3,Kq=3,Dq=2, bq=3, bk=2;
        float qb[1*2*3*2], kb[1*1*3*2], vb[1*1*3*2], mb[1*1*3*3];
        for (int i=0;i<6;i++){ qb[i]=0.1f*(float)((i*3)%7)-0.3f; }
        for (int i=0;i<6;i++){ qb[6+i]=qb[i]; }
        for (int i=0;i<1*1*3*2;i++){ kb[i]=0.1f*(float)((i*5)%9)-0.4f; vb[i]=0.1f*(float)((i*7)%11)-0.5f; }
        for (int i=0;i<Sq;i++) for (int j=0;j<Kq;j++) mb[i*Kq+j]=(j<=i)?0.0f:-INFINITY;
        mlx_array q = mlx_array_new_data(qb,(const int[]){B,Hh,Sq,Dq},4,MLX_FLOAT32);
        float kb2[1*2*3*2], vb2[1*2*3*2];
        for (int h=0;h<2;h++) for (int i=0;i<6;i++){ kb2[h*6+i]=kb[i]; vb2[h*6+i]=vb[i]; }
        mlx_array k = mlx_array_new_data(kb2,(const int[]){B,Hh,Kq,Dq},4,MLX_FLOAT32);
        mlx_array v = mlx_array_new_data(vb2,(const int[]){B,Hh,Kq,Dq},4,MLX_FLOAT32);
        mlx_array ma= mlx_array_new_data(mb,(const int[]){B,1,Sq,Kq},4,MLX_FLOAT32);
        float sc1 = 1.0f/sqrtf((float)Dq);
        mlx_array out = mlx_array_new();
        if (!llm_mlx_attention_blockskip(&out, q,k,v, sc1, ma, bq, bk, st)) { printf("FAIL blockskip GQA call\n"); fail++; }
        else {
            mlx_array_eval(out); const float* o = mlx_array_data_float32(out);
            int badp=0;
            for (int h=0;h<Hh;h++) for (int i=0;i<Sq;i++){
                float sc[3], mx=-1e30f;
                for (int j=0;j<Kq;j++){ float d=0; for(int e=0;e<Dq;e++) d += qb[((h)*Sq+i)*Dq+e]*kb[j*Dq+e]; sc[j]=d*sc1+mb[i*Kq+j]; if(sc[j]>mx)mx=sc[j]; }
                float den=0; for(int j=0;j<Kq;j++){ sc[j]=expf(sc[j]-mx); den+=sc[j]; }
                for (int e=0;e<Dq;e++){ float acc=0; for(int j=0;j<Kq;j++) acc += (sc[j]/den)*vb[j*Dq+e];
                    float got=o[((h)*Sq+i)*Dq+e]; if (fabsf(got-acc)>1e-3f){ if(badp<4) printf("  gqa[%d,%d,%d] got=%.6g exp=%.6g\n",h,i,e,got,acc); badp++; } }
            }
            printf("%s %-46s\n", badp?"FAIL":"PASS", "attention_blockskip GQA == softmax densa"); if(badp) fail++;
        }
    }
    printf(fail ? "FAILURES: %d\n" : "ALL PASS\n", fail);
    return fail;
}
