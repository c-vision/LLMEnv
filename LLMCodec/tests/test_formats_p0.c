/* Verifica i codec P0 nuovi: mxfp8, bitnet, gptq, awq (via registry). */
#include "llmcodec.h"
#include "llmcodec_fp16.h"
#include "llmcodec_rolemap.h"
#include "llmcodec_higgs_grid.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
int main(void){
    /* --- mxfp8 (fmt 0): weight U32 [R,C/4] (4 e4m3/u32) + scala e8m0 u8 [R,C/32] --- */
    const int R=4, C=64;
    uint32_t wm[R*(C/4)]; memset(wm, 0x38383838u, sizeof wm);   /* 0x38 = e4m3 1.0 */
    uint8_t sm[R*(C/32)]; memset(sm, 127, sizeof sm);           /* e8m0 127 = 1.0 */
    const LlmCodec* mx=llmcodec_resolve("mxfp");
    float om[R*C]; double mdmx=1e9;
    if(mx && mx->decode(wm,sizeof wm,0,0,32,R,C,sm,sizeof sm,om)==0){ mdmx=0; for(int i=0;i<R*C;i++){double d=fabs(om[i]-1.0); if(d>mdmx)mdmx=d;} }
    printf("[p0] mxfp8 = %s (maxerr %.2e)\n", mx?"ok":"NULL", mdmx);

    /* --- bitnet: ternario 2-bit (4/byte) + scala f16 per riga. code 01=+1 --- */
    const int R2=4, C2=64;
    uint8_t wb[R2*(C2/4)]; for(size_t i=0;i<sizeof wb;i++) wb[i]=0x55;  /* 01,01,01,01 -> +1 */
    uint16_t h1=0x3C00; uint8_t sb[R2*2]; for(int i=0;i<R2;i++) memcpy(sb+i*2,&h1,2);   /* f16 1.0 */
    const LlmCodec* bn=llmcodec_resolve("bitnet");
    float ob[R2*C2]; double mdbn=1e9;
    if(bn && bn->decode(wb,sizeof wb,0,0,0,R2,C2,sb,sizeof sb,ob)==0){ mdbn=0; for(int i=0;i<R2*C2;i++){double d=fabs(ob[i]-1.0); if(d>mdbn)mdbn=d;} }
    printf("[p0] bitnet = %s (maxerr %.2e)\n", bn?"ok":"NULL", mdbn);

    /* --- gptq: in=32,out=8,group=32. qweight [in/8,out], scales [g,out], qzeros [g,out/8] --- */
    const int IN=32, OUT=8, GRP=32; int G=IN/GRP;
    uint32_t qw[IN/8*OUT]; for(size_t i=0;i<sizeof qw/4;i++) qw[i]=0x11111111u;   /* q=1 */
    uint8_t auc[ (size_t)G*OUT*2 + (size_t)G*(OUT/8)*4 ];
    uint16_t one=0x3C00; for(int o=0;o<OUT;o++) memcpy(auc+o*2,&one,2);            /* scales = 1.0 */
    memset(auc + (size_t)G*OUT*2, 0, (size_t)G*(OUT/8)*4);                         /* qzeros = 0 */
    const LlmCodec* gq=llmcodec_resolve("gptq");
    float og[OUT*IN]; double mdg=1e9;
    if(gq && gq->decode(qw,sizeof qw,0,GRP,0,OUT,IN,auc,sizeof auc,og)==0){ mdg=0; for(int i=0;i<OUT*IN;i++){double d=fabs(og[i]-1.0); if(d>mdg)mdg=d;} }
    printf("[p0] gptq = %s (maxerr %.2e)\n", gq?"ok":"NULL", mdg);

    /* --- awq: in=32,out=8,group=32. qweight [in,out/8], scales [g,out], qzeros [g,out/8] --- */
    uint32_t qwa[IN*(OUT/8)]; for(size_t i=0;i<sizeof qwa/4;i++) qwa[i]=0x11111111u;
    const LlmCodec* aw=llmcodec_resolve("awq");
    float oa[OUT*IN]; double mda=1e9;
    if(aw && aw->decode(qwa,sizeof qwa,0,GRP,0,OUT,IN,auc,sizeof auc,oa)==0){ mda=0; for(int i=0;i<OUT*IN;i++){double d=fabs(oa[i]-1.0); if(d>mda)mda=d;} }
    printf("[p0] awq = %s (maxerr %.2e)\n", aw?"ok":"NULL", mda);

    /* bnb NF4: weight U8 (2 nibble/byte) + absmax f32 per blocco 64; 0xFF=idx15=1.0 */
    const int R3=2, C3=4;
    uint8_t wnb[R3*(C3/2)]; memset(wnb,0xFF,sizeof wnb);
    float abn[R3*1]; for(int i=0;i<R3;i++) abn[i]=1.0f;
    const LlmCodec* bn2=llmcodec_resolve("bnb");
    float obn[R3*C3]; double mdbn2=1e9;
    if(bn2 && bn2->decode(wnb,sizeof wnb,0,0,64,R3,C3,abn,sizeof abn,obn)==0){ mdbn2=0; for(int i=0;i<R3*C3;i++){double d=fabs(obn[i]-1.0); if(d>mdbn2)mdbn2=d;} }
    printf("[p0] bnb NF4 = %s (maxerr %.2e)\n", bn2?"ok":"NULL", mdbn2);

    /* hqq (affine-con-zero): int4 packed + scales f16 ++ zeros f16 per gruppo.
       q=1, scale=1.0, zero=0 -> 1.0. */
    const int RH=2, CH=8, GH=8;   /* 1 gruppo per riga */
    uint8_t wh[RH*(CH/2)]; memset(wh,0x11,sizeof wh);
    uint8_t auh[(size_t)RH*2 + (size_t)RH*2]; uint16_t h1b=0x3C00, h0=0x0000;
    for(int i=0;i<RH;i++){ memcpy(auh+i*2,&h1b,2); memcpy(auh+RH*2+i*2,&h0,2); }
    const LlmCodec* hq=llmcodec_resolve("hqq");
    float oh[RH*CH]; double mdh=1e9;
    if(hq && hq->decode(wh,sizeof wh,0,GH,4,RH,CH,auh,sizeof auh,oh)==0){ mdh=0; for(int i=0;i<RH*CH;i++){double d=fabs(oh[i]-1.0); if(d>mdh)mdh=d;} }
    printf("[p0] hqq = %s (maxerr %.2e)\n", hq?"ok":"NULL", mdh);

    /* turboquant (famiglia): codebook 16 (4-bit) + scala; idx 0 -> codebook[0]=5.0; seed 0 (no rot) */
    const int RT=2, CT=8;
    uint32_t wt[RT*1]; memset(wt,0x00,sizeof wt);   /* tutti idx 0 */
    uint8_t aut[32 + RT*2]; uint16_t cb0=0x4500 /*5.0*/, one_t=0x3C00;
    memcpy(aut,&cb0,2); for(int k=1;k<16;k++) memcpy(aut+k*2,&(uint16_t){0},2);
    for(int i=0;i<RT;i++) memcpy(aut+32+i*2,&one_t,2);
    const LlmCodec* tq=llmcodec_resolve("turboquant");
    float ot[RT*CT]; double mdt=1e9;
    if(tq && tq->decode(wt,sizeof wt,0,0,64,RT,CT,aut,sizeof aut,ot)==0){ mdt=0; for(int i=0;i<RT*CT;i++){double d=fabs(ot[i]-5.0); if(d>mdt)mdt=d;} }
    printf("[p0] turboquant = %s (maxerr %.2e)\n", tq?"ok":"NULL", mdt);

    /* aqlm: nc=1, ig=4, og=1; rows=2 out, cols=8 in (2 in_group). codes=0; codebook[0][0][0][t]=t+1; scale=1 */
    const int RA=2, CA=8, NC=1, IG=4;
    int8_t ca[RA*CA/IG*NC]; memset(ca,0,sizeof ca);
    size_t cbsz=(size_t)NC*256*1*IG*2 + RA*2;
    uint8_t aua[cbsz]; memset(aua,0,sizeof aua);
    for(int t=0;t<IG;t++){ uint16_t h=llmcodec_f16_from_f32((float)(t+1)); memcpy(aua+t*2,&h,2); }
    for(int i=0;i<RA;i++){ uint16_t one=0x3C00; memcpy(aua+(size_t)NC*256*IG*2+i*2,&one,2); }
    const LlmCodec* aq=llmcodec_resolve("aqlm");
    float oaq[RA*CA]; double mdaq=1e9;
    if(aq && aq->decode(ca,sizeof ca,0,NC,IG,RA,CA,aua,sizeof aua,oaq)==0){ mdaq=0; for(int o=0;o<RA;o++) for(int c=0;c<CA;c++){ double d=fabs(oaq[o*CA+c]-((c%IG)+1)); if(d>mdaq)mdaq=d; } }
    printf("[p0] aqlm = %s (maxerr %.2e)\n", aq?"ok":"NULL", mdaq);

    /* vptq: nc=1, ncent=4, vl=4, gs=2, numidx=1 -> rows=4, cols=2; idx 0; centroids[0][0][t]=t+1 */
    const int RV=4, CV=2, NCV=1, NCENT=4, VL=4;
    int16_t iv[NCV*1*2]; memset(iv,0,sizeof iv);
    uint8_t auv[NCV*NCENT*VL*2]; memset(auv,0,sizeof auv);
    for(int t=0;t<VL;t++){ uint16_t h=llmcodec_f16_from_f32((float)(t+1)); memcpy(auv+t*2,&h,2); }
    const LlmCodec* vq=llmcodec_resolve("vptq");
    float ovq[RV*CV]; double mdv=1e9;
    if(vq && vq->decode(iv,sizeof iv,0,NCV,NCENT,RV,CV,auv,sizeof auv,ovq)==0){ mdv=0; for(int o=0;o<RV;o++) for(int c=0;c<CV;c++){ double d=fabs(ovq[o*CV+c]-((o%VL)+1)); if(d>mdv)mdv=d; } }
    printf("[p0] vptq = %s (maxerr %.2e)\n", vq?"ok":"NULL", mdv);

    /* spqr: 1 tile 2x4; word = s=1|z=0|q=1 x4 ; 2o ordine ss={1,0},zz={0,0} -> w=1.0 */
    const int RQ=2, CQ=4;
    uint64_t dense[1]; dense[0]= (uint64_t)1 | ((uint64_t)1<<6) | ((uint64_t)1<<9) | ((uint64_t)1<<12) | ((uint64_t)1<<15);
    uint8_t auq[8]; { uint16_t one=0x3C00, zero=0; memcpy(auq,&one,2); memcpy(auq+2,&zero,2); memcpy(auq+4,&zero,2); memcpy(auq+6,&zero,2); }
    const LlmCodec* sq=llmcodec_resolve("spqr");
    float oq[RQ*CQ]; double mdq=1e9;
    if(sq && sq->decode(dense,sizeof dense,0,4,2,RQ,CQ,auq,sizeof auq,oq)==0){ mdq=0; for(int i=0;i<RQ*CQ;i++){double d=fabs(oq[i]-1.0); if(d>mdq)mdq=d;} }
    printf("[p0] spqr = %s (maxerr %.2e)\n", sq?"ok":"NULL", mdq);

    /* higgs: 2 coppie (4 elem), idx 0, norm f16=1.0 -> out = grid[0] */
    const int NPH=2;
    uint8_t ih[NPH]; memset(ih,0,sizeof ih);
    uint16_t h1_=0x3C00; uint8_t augh[NPH]; memcpy(augh,&h1_,2);
    const LlmCodec* hg=llmcodec_resolve("higgs");
    float ohg[NPH*2]; double mdhg=1e9;
    if(hg && hg->decode(ih,sizeof ih,0,4,256,1,NPH*2,augh,sizeof augh,ohg)==0){ mdhg=0;
        for(int q=0;q<NPH;q++){ double d0=fabs(ohg[2*q]-LLMCODEC_HIGGS_GRID[0][0]); if(d0>mdhg)mdhg=d0; double d1=fabs(ohg[2*q+1]-LLMCODEC_HIGGS_GRID[0][1]); if(d1>mdhg)mdhg=d1; } }
    printf("[p0] higgs = %s (maxerr %.2e)\n", hg?"ok":"NULL", mdhg);

    /* compressed-tensors: metadata -> mode/bits/group */
    FILE* cf=fopen("/tmp/ct_config.json","wb");
    fprintf(cf,"{\"quantization_config\":{\"quant_method\":\"compressed-tensors\",\"format\":\"pack-quantized\",\"config_groups\":{\"g0\":{\"weights\":{\"num_bits\":4,\"group_size\":128,\"symmetric\":true}}}}}"); fclose(cf);
    LlmCodecQuantCfg qc; int ctr=llmcodec_ct_load("/tmp/ct_config.json",&qc);
    printf("[p0] compressed-tensors = %s (mode=%s bits=%d group=%d)\n", ctr==0?"ok":"FAIL", qc.mode, qc.bits, qc.group);

    int ok = (mdmx<1e-5 && mdbn<1e-5 && mdg<1e-5 && mda<1e-5 && mdbn2<1e-5 && mdh<1e-5 && mdt<1e-5 && mdaq<1e-5 && mdv<1e-5 && mdq<1e-5 && mdhg<1e-5) && ctr==0 && strcmp(qc.mode,"affine")==0 && qc.bits==4 && qc.group==128;
    printf(ok?"PASS\n":"FAIL\n"); return ok?0:1;
}
