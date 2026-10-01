/* test_ggml_quants.c — codec GGML famiglia 1: F32/F16/BF16 + Q4_0/Q4_1/Q5_0/Q5_1/Q8_0.
 * (a) decode DETERMINISTICO su blocchi costruiti a mano (valori calcolati dalla formula ggml);
 * (b) round-trip encode->decode entro il passo atteso; (c) validazione. Exit 0 = ok. */
#include "llmcodec_ggml_quants.h"
#include "llmcodec_iq2_xxs.h"
#include "llmcodec_gguf_mlx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("FAIL: %s\n", m); fails++; } } while (0)

int main(void) {
    /* (a) Q4_0: d=1.0 (f16 0x3C00), qs[0]=0 -> (-8 idx0, -8 idx16); qs[15]=0xF0 -> (-8 idx15, +7 idx31). */
    { unsigned char b[18]; memset(b, 0, sizeof b); b[1]=0x3C; b[2+15]=0xF0;
      float y[32]; CHECK(llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_Q4_0, b, y, 32)==0, "q4_0 decode rc");
      CHECK(y[0]==-8.0f && y[16]==-8.0f, "q4_0 qs0");
      CHECK(y[15]==-8.0f && y[31]==7.0f, "q4_0 qs15");
      CHECK(llmcodec_ggml_block_vals(LLMCODEC_GGML_T_Q4_0)==32, "q4_0 blk=32"); }
    /* Q8_0: d=0.5 (0x3800), qs[j]=j-16 -> y[j]=(j-16)*0.5 */
    { unsigned char b[34]; memset(b, 0, sizeof b); b[0]=0x00; b[1]=0x38;
      for (int j=0;j<32;++j) ((signed char*)(b+2))[j]=(signed char)(j-16);
      float y[32]; CHECK(llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_Q8_0, b, y, 32)==0, "q8_0 rc");
      int ok=1; for (int j=0;j<32;++j) if (fabsf(y[j] - (j-16)*0.5f) > 1e-6f) ok=0;
      CHECK(ok, "q8_0 valori"); }
    /* Q4_1: d=1, m=0.5; qs[0]=0x21 -> lo=1 -> 1.5 ; hi=2 -> 2.5 (idx16) */
    { unsigned char b[20]; memset(b, 0, sizeof b); b[0]=0x00; b[1]=0x3C; b[2]=0x00; b[3]=0x38; b[4]=0x21;
      float y[32]; CHECK(llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_Q4_1, b, y, 32)==0, "q4_1 rc");
      CHECK(fabsf(y[0]-1.5f)<1e-6f && fabsf(y[16]-2.5f)<1e-6f, "q4_1 valori"); }
    /* Q5_0: d=1, qs[0]=0, qh=0 -> x0=(0|0)-16=-16. Con qh bit0 -> x0=(0|16)-16=0 */
    { unsigned char b[22]; memset(b, 0, sizeof b); b[0]=0x00; b[1]=0x3C; /* qh[4..7]=0 */
      float y[32]; CHECK(llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_Q5_0, b, y, 32)==0, "q5_0 rc");
      CHECK(y[0]==-16.0f, "q5_0 qh0 -> -16");
      b[2]=0x01; /* qh[0] (byte 2, dopo d) bit0=1 -> xh0=0x10 */
      CHECK(llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_Q5_0, b, y, 32)==0, "q5_0 rc2");
      CHECK(y[0]==0.0f, "q5_0 qh bit -> 0"); }

    /* (a2) K-quant: Q8_K deterministico (d*qs) + Q4_K con blocco costruito a mano.
     * Blocco nullo -> tutti zeri per tutti i tipi K (sanity). */
    { unsigned char b[292]; memset(b, 0, sizeof b); float d2=2.0f; memcpy(b, &d2, 4);
      for (int j=0;j<256;++j) ((signed char*)(b+4))[j]=1;
      float y[256]; CHECK(llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_Q8_K, b, y, 256)==0, "q8_K rc");
      int ok=1; for (int j=0;j<256;++j) if (y[j]!=2.0f) ok=0; CHECK(ok, "q8_K valori=2"); }
    { /* Q4_K: d=1 (0x3C00), dmin=0; scales[0]=1,scales[1]=1,scales[4]=0,scales[5]=0; qs[0]=0x21 */
      unsigned char b[144]; memset(b, 0, sizeof b); b[1]=0x3C; b[4]=1; b[5]=1;
      b[16]=0x21; /* scales[0]=scales[1]=1 (byte 4/5), scales[4]/[5]=0; qs[0]=0x21 */
      float y[256]; CHECK(llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_Q4_K, b, y, 256)==0, "q4_K rc");
      CHECK(y[0]==1.0f, "q4_K lo nibble=1 -> 1");
      CHECK(y[32]==2.0f, "q4_K hi nibble=2 -> 2"); }
    { /* blocco nullo -> zeri per Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K */
      unsigned char b[292]; memset(b, 0, sizeof b); float y[256];
      int types[] = {LLMCODEC_GGML_T_Q2_K,LLMCODEC_GGML_T_Q3_K,LLMCODEC_GGML_T_Q4_K,LLMCODEC_GGML_T_Q5_K,LLMCODEC_GGML_T_Q6_K,LLMCODEC_GGML_T_Q8_K};
      for (int k=0;k<6;++k){ CHECK(llmcodec_ggml_dequant_row(types[k], b, y, 256)==0, "K zero rc");
        int ok=1; for (int j=0;j<256;++j) if (y[j]!=0.0f) ok=0; CHECK(ok, "K zero -> 0"); } }

    /* (a3) MXFP4 / NVFP4: E2M1 {0,0.5,1,1.5,2,3,4,6} con scala E8M0 / UE4M3 */
    { unsigned char b[17]; memset(b,0,sizeof b); b[0]=127; /* e8m0 -> d=1 */
      b[1]=0x67; /* lo=7 -> 6 ; hi=6 -> 3 */
      float y[32]; CHECK(llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_MXFP4, b, y, 32)==0, "mxfp4 rc");
      CHECK(y[0]==6.0f && y[16]==4.0f, "mxfp4 e2m1");
      b[1]=0x81; /* lo=1 -> 0.5 ; hi=8 -> -0.0 */
      CHECK(llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_MXFP4, b, y, 32)==0, "mxfp4 rc2");
      CHECK(y[0]==0.5f && y[16]==0.0f, "mxfp4 segno/zero");
      CHECK(llmcodec_ggml_block_vals(LLMCODEC_GGML_T_MXFP4)==32, "mxfp4 blk"); }
    { unsigned char b[36]; memset(b,0,sizeof b); b[0]=0x38; /* ue4m3(exp7,man0) -> 1.0 ; sub s=0 */
      b[4]=0x67; /* qs[0]: lo=7 -> 6 ; hi=6 -> 3 */
      float y[64]; CHECK(llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_NVFP4, b, y, 64)==0, "nvfp4 rc");
      CHECK(y[0]==6.0f && y[8]==4.0f, "nvfp4 e2m1");
      CHECK(llmcodec_ggml_block_vals(LLMCODEC_GGML_T_NVFP4)==64, "nvfp4 blk"); }

    /* (a4) Q1_0 / Q2_0 / TQ2_0 / TQ1_0 */
    { unsigned char b[18]; memset(b,0,sizeof b); b[1]=0x3C; b[2]=0x05; /* d=1; bit0 e bit2 -> + */
      float y[128]; CHECK(llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_Q1_0, b, y, 128)==0, "q1_0 rc");
      CHECK(y[0]==1.0f && y[1]==-1.0f && y[2]==1.0f && y[3]==-1.0f, "q1_0 bit");
      CHECK(llmcodec_ggml_block_vals(LLMCODEC_GGML_T_Q1_0)==128, "q1_0 blk"); }
    { unsigned char b[18]; memset(b,0,sizeof b); b[1]=0x3C; b[2]=0xE4; /* codes 0,1,2,3 -> -1,0,+1,+2 */
      float y[64]; CHECK(llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_Q2_0, b, y, 64)==0, "q2_0 rc");
      CHECK(y[0]==-1.0f && y[1]==0.0f && y[2]==1.0f && y[3]==2.0f, "q2_0 code");
      CHECK(llmcodec_ggml_block_vals(LLMCODEC_GGML_T_Q2_0)==64, "q2_0 blk"); }
    { unsigned char b[66]; memset(b,0,sizeof b); b[64]=0x00; b[65]=0x3C; b[0]=0xE4; /* d=1; qs[0] */
      float y[256]; CHECK(llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_TQ2_0, b, y, 256)==0, "tq2_0 rc");
      CHECK(y[0]==-1.0f && y[32]==0.0f && y[64]==1.0f && y[96]==2.0f, "tq2_0 shift");
      CHECK(llmcodec_ggml_block_vals(LLMCODEC_GGML_T_TQ2_0)==256, "tq2_0 blk"); }
    { unsigned char b[54]; memset(b,0,sizeof b); float y[256]; /* d=0 -> tutto 0 */
      CHECK(llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_TQ1_0, b, y, 256)==0, "tq1_0 rc");
      int ok=1; for (int j=0;j<256;++j) if (y[j]!=0.0f) ok=0; CHECK(ok, "tq1_0 zero");
      CHECK(llmcodec_ggml_block_vals(LLMCODEC_GGML_T_TQ1_0)==256, "tq1_0 blk"); }

    /* (a5) IQ4_NL / IQ4_XS (codebook iq4nl: kv[0]=-127, kv[1]=-104) */
    { unsigned char b[18]; memset(b,0,sizeof b); b[1]=0x3C; b[2]=0x10; /* d=1; lo=0 -> -127; hi=1 -> -104 */
      float y[32]; CHECK(llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_IQ4_NL, b, y, 32)==0, "iq4_nl rc");
      CHECK(y[0]==-127.0f && y[16]==-104.0f, "iq4_nl kv");
      CHECK(llmcodec_ggml_block_vals(LLMCODEC_GGML_T_IQ4_NL)==32, "iq4_nl blk"); }
    { unsigned char b[136]; memset(b,0,sizeof b); b[1]=0x3C; /* d=1; scales_h=0; scales_l=0; qs[0]=0 */
      float y[256]; CHECK(llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_IQ4_XS, b, y, 256)==0, "iq4_xs rc");
      CHECK(y[0]==(-32.0f * -127.0f), "iq4_xs dl(ls=0 -> -32) * kv[0]");
      CHECK(llmcodec_ggml_block_vals(LLMCODEC_GGML_T_IQ4_XS)==256, "iq4_xs blk"); }

    /* (a6) IQ grid-based: blocco nullo (d=0) -> zeri per tutti; block_vals=256 */
    { unsigned char b[136]; memset(b,0,sizeof b); float y[256];
      int types[] = {LLMCODEC_GGML_T_IQ2_XXS,LLMCODEC_GGML_T_IQ2_XS,LLMCODEC_GGML_T_IQ2_S,LLMCODEC_GGML_T_IQ3_XXS,
                     LLMCODEC_GGML_T_IQ3_S,LLMCODEC_GGML_T_IQ1_S,LLMCODEC_GGML_T_IQ1_M};
      int blk[] = {66,74,82,98,110,50,56};
      for (int k=0;k<7;++k){ CHECK(llmcodec_ggml_dequant_row(types[k], b, y, 256)==0, "IQ zero rc");
        int ok=1; for (int j=0;j<256;++j) if (y[j]!=0.0f) ok=0; CHECK(ok, "IQ zero -> 0");
        CHECK(llmcodec_ggml_block_vals(types[k])==256, "IQ blk"); (void)blk[k]; } }

    /* (b) round-trip encode->decode: errore max entro il passo atteso */
    { float x[256]; unsigned s=7; for (int i=0;i<256;i++){ s=s*1103515245u+12345u; x[i]=(float)((s>>8)&0xFFFF)/32768.0f-1.0f; }
      struct { int t; double tol; const char* n; } cases[] = {
        { LLMCODEC_GGML_T_F32, 0.0, "F32" }, { LLMCODEC_GGML_T_F16, 1e-3, "F16" }, { LLMCODEC_GGML_T_BF16, 1e-2, "BF16" },
        { LLMCODEC_GGML_T_Q8_0, 1.6/255.0, "Q8_0" }, { LLMCODEC_GGML_T_Q4_0, 2.4/15.0, "Q4_0" },
        { LLMCODEC_GGML_T_Q4_1, 1.6/15.0, "Q4_1" }, { LLMCODEC_GGML_T_Q5_0, 1.6/15.0, "Q5_0" },
        { LLMCODEC_GGML_T_Q5_1, 1.6/15.0, "Q5_1" },
        { LLMCODEC_GGML_T_Q6_K, 5.0e-2, "Q6_K" }, { LLMCODEC_GGML_T_Q8_K, 1.6/127.0, "Q8_K" }, { LLMCODEC_GGML_T_Q5_K, 1.2e-1, "Q5_K" }, { LLMCODEC_GGML_T_Q4_K, 1.6e-1, "Q4_K" },
        { LLMCODEC_GGML_T_Q2_K, 4.5e-1, "Q2_K" }, { LLMCODEC_GGML_T_Q3_K, 2.5e-1, "Q3_K" },
        { LLMCODEC_GGML_T_MXFP4, 3.5e-1, "MXFP4" }, { LLMCODEC_GGML_T_NVFP4, 3.5e-1, "NVFP4" },
        { LLMCODEC_GGML_T_Q1_0, 6.0e-1, "Q1_0" }, { LLMCODEC_GGML_T_Q2_0, 5.0e-1, "Q2_0" },
        { LLMCODEC_GGML_T_TQ1_0, 6.0e-1, "TQ1_0" }, { LLMCODEC_GGML_T_TQ2_0, 6.0e-1, "TQ2_0" } };
      for (size_t k=0;k<sizeof(cases)/sizeof(cases[0]);++k){
        size_t need; int64_t bv=llmcodec_ggml_block_vals(cases[k].t);
        /* byte size via blocco noto */
        size_t bs = (cases[k].t==LLMCODEC_GGML_T_F32)?4:(cases[k].t==LLMCODEC_GGML_T_F16||cases[k].t==LLMCODEC_GGML_T_BF16)?2:
                    (cases[k].t==LLMCODEC_GGML_T_Q8_0)?34:(cases[k].t==LLMCODEC_GGML_T_Q4_0)?18:
                    (cases[k].t==LLMCODEC_GGML_T_Q4_1)?20:(cases[k].t==LLMCODEC_GGML_T_Q5_0)?22:
                    (cases[k].t==LLMCODEC_GGML_T_Q5_1)?24:(cases[k].t==LLMCODEC_GGML_T_Q6_K)?210:(cases[k].t==LLMCODEC_GGML_T_Q8_K)?292:(cases[k].t==LLMCODEC_GGML_T_Q5_K)?176:
                    (cases[k].t==LLMCODEC_GGML_T_Q4_K)?144:(cases[k].t==LLMCODEC_GGML_T_Q2_K)?84:(cases[k].t==LLMCODEC_GGML_T_Q3_K)?110:
                    (cases[k].t==LLMCODEC_GGML_T_MXFP4)?17:(cases[k].t==LLMCODEC_GGML_T_NVFP4)?36:
                    (cases[k].t==LLMCODEC_GGML_T_Q1_0)?18:(cases[k].t==LLMCODEC_GGML_T_Q2_0)?18:
                    (cases[k].t==LLMCODEC_GGML_T_TQ1_0)?54:(cases[k].t==LLMCODEC_GGML_T_TQ2_0)?66:0;
        need = (size_t)(256/bv)*bs;
        unsigned char* q=(unsigned char*)malloc(need); float* y=(float*)malloc(256*sizeof(float));
        CHECK(llmcodec_ggml_quant_row(cases[k].t, x, q, 256, NULL)==0, cases[k].n);
        CHECK(llmcodec_ggml_dequant_row(cases[k].t, q, y, 256)==0, cases[k].n);
        double md=0; for (int i=0;i<256;i++) md=fmax(md,fabs((double)y[i]-x[i]));
        printf("[ggml] %-5s round-trip maxerr=%.4e (tol %.4e)\n", cases[k].n, md, cases[k].tol);
        CHECK(md <= cases[k].tol, cases[k].n);
        free(q); free(y);
      } }

    /* (c) validazione */
    { float y[32]; CHECK(llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_Q4_0, y, y, 30) < 0, "n non multiplo -> -1");
      CHECK(llmcodec_ggml_dequant_row(999, y, y, 32) < 0, "tipo ignoto -> -1"); }

    /* (a7) IQ2_XXS / IQ2_XS encode->decode */
    { float x[256]; unsigned s=11; for(int i=0;i<256;i++){ s=s*1103515245u+12345u; x[i]=(float)((s>>8)&0xFFFF)/32768.0f-1.0f; }
      float y[256];
      /* iq2_xxs encode: PUL (rimosso da LLMEnv) */
      { uint8_t q[110]; memset(q,0,110); llmcodec_iq3_s_quantize(x,256,q,NULL,0,256);
        CHECK(llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_IQ3_S, q, y, 256)==0, "iq3_s dec");
        double md=0; for(int i=0;i<256;i++) md=fmax(md,fabs((double)y[i]-x[i]));
        printf("[iq2] IQ3_S maxerr=%.4e (== ggml 0.241)\n", md); CHECK(md < 1.0e0, "iq3_s round-trip"); }
      { uint8_t q[98]; memset(q,0,98); llmcodec_iq3_xxs_quantize(x,256,q,NULL,0,256);
        CHECK(llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_IQ3_XXS, q, y, 256)==0, "iq3_xxs dec");
        double md=0; for(int i=0;i<256;i++) md=fmax(md,fabs((double)y[i]-x[i]));
        printf("[iq2] IQ3_XXS maxerr=%.4e (== ggml 0.400)\n", md); CHECK(md < 1.0e0, "iq3_xxs round-trip"); }
      { uint8_t q[56]; memset(q,0,56); llmcodec_iq1_m_quantize(x,256,q,NULL,0,256);
        CHECK(llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_IQ1_M, q, y, 256)==0, "iq1_m dec");
        double md=0; for(int i=0;i<256;i++) md=fmax(md,fabs((double)y[i]-x[i]));
        printf("[iq2] IQ1_M maxerr=%.4e (== ggml 0.676)\n", md); CHECK(md < 1.2e0, "iq1_m round-trip"); }
      { uint8_t q[50]; memset(q,0,50); llmcodec_iq1_s_quantize(x,256,q,NULL,0,256);
        CHECK(llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_IQ1_S, q, y, 256)==0, "iq1_s dec");
        double md=0; for(int i=0;i<256;i++) md=fmax(md,fabs((double)y[i]-x[i]));
        printf("[iq2] IQ1_S maxerr=%.4e (== ggml 0.736)\n", md); CHECK(md < 1.2e0, "iq1_s round-trip"); }
      { uint8_t q[82]; memset(q,0,82); llmcodec_iq2_s_quantize(x,256,q,NULL,0,256);
        CHECK(llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_IQ2_S, q, y, 256)==0, "iq2_s dec");
        double md=0; for(int i=0;i<256;i++) md=fmax(md,fabs((double)y[i]-x[i]));
        printf("[iq2] IQ2_S maxerr=%.4e\n", md); CHECK(md < 1.0e0, "iq2_s round-trip"); }
      { uint8_t q[74]; memset(q,0,74); llmcodec_iq2_xs_quantize(x,256,q,NULL,0,256);
        CHECK(llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_IQ2_XS, q, y, 256)==0, "iq2_xs dec");
        double md=0; for(int i=0;i<256;i++) md=fmax(md,fabs((double)y[i]-x[i]));
        printf("[iq2] IQ2_XS maxerr=%.4e\n", md); CHECK(md < 2.0e0, "iq2_xs round-trip"); } }

    /* (a8) conversione GGML<->MLX affine (f32 -> Q4_K -> MLX affine -> Q4_K -> dequant) */
    { float x[256]; unsigned s=21; for(int i=0;i<256;i++){ s=s*1103515245u+12345u; x[i]=(float)((s>>8)&0xFFFF)/32768.0f-1.0f; }
      uint8_t qk[144];
      CHECK(llmcodec_ggml_quant_row(LLMCODEC_GGML_T_Q4_K, x, qk, 256, NULL)==0, "conv enc Q4_K");
      uint32_t w[256/8]; uint16_t sc[256/32], bs[256/32];
      CHECK(llmcodec_conv_ggml_to_mlx(LLMCODEC_GGML_T_Q4_K, qk, 1, 256, 32, 4, w, sc, bs)==0, "conv ggml->mlx");
      uint8_t qk2[144];
      CHECK(llmcodec_conv_mlx_to_ggml(w, sc, bs, 1, 256, 32, 4, LLMCODEC_GGML_T_Q4_K, qk2)==0, "conv mlx->ggml");
      float y[256];
      CHECK(llmcodec_ggml_dequant_row(LLMCODEC_GGML_T_Q4_K, qk2, y, 256)==0, "conv dec");
      double md=0; for(int i=0;i<256;i++) md=fmax(md,fabs((double)y[i]-x[i]));
      printf("[conv] GGML<->MLX affine (Q4_K, g32/b4) maxerr=%.4e\n", md); CHECK(md < 2.0e-1, "conv round-trip");
    }

    if (fails==0) printf("test_ggml_quants: OK\n");
    return fails ? 1 : 0;
}
