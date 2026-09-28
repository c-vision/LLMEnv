/* llm_gguf_test.c — verifica lettore GGUF + config V4 dal GGUF. */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "llm_dsv4.h"

static unsigned char B[4096]; static size_t BP;
static void w_u32(unsigned v){ for(int i=0;i<4;i++) B[BP++]=(unsigned char)(v>>(8*i)); }
static void w_u64(uint64_t v){ for(int i=0;i<8;i++) B[BP++]=(unsigned char)(v>>(8*i)); }
static void w_f32(float f){ unsigned u; memcpy(&u,&f,4); w_u32(u); }
static void w_str(const char* s){ size_t n=strlen(s); w_u64(n); memcpy(B+BP,s,n); BP+=n; }
static void w_kv_u32(const char* k, unsigned v){ w_str(k); w_u32(4); w_u32(v); }
static void w_kv_f32(const char* k, float f){ w_str(k); w_u32(6); w_f32(f); }
static void w_kv_str(const char* k, const char* s){ w_str(k); w_u32(8); w_str(s); }
static void w_ti(const char* nm, int nd, const int64_t* d, int ty, uint64_t off){
    w_str(nm); w_u32((unsigned)nd); for(int i=0;i<nd;i++) w_u64((uint64_t)d[i]); w_u32((unsigned)ty); w_u64(off);
}

int main(void) {
    BP = 0;
    memcpy(B, "GGUF", 4); BP = 4;
    w_u32(3); w_u64(2); w_u64(5);
    w_kv_u32("deepseek4.block_count", 43);
    w_kv_u32("deepseek4.embedding_length", 4096);
    w_kv_u32("deepseek4.attention.head_count", 64);
    w_kv_u32("deepseek4.attention.head_count_kv", 1);
    w_kv_f32("deepseek4.hyper_connection.epsilon", 1e-6f);
    /* n_kv dichiarato 5 sopra; 2 tensori: token_embd (F32) e q8 (Q8_0) */
    int64_t dims[2] = { 4096, 129280 };
    w_ti("token_embd.weight", 2, dims, 0, 0);
    w_ti("test.q8", 1, (int64_t[]){ 64 }, 8, 0);   /* Q8_0, 64 elem, offset 0 */
    while (BP % 32) B[BP++] = 0;

    LlmEngineMoeGguf g;
    /* dati Q8_0: 2 blocchi da 34 byte a offset 0 del blob (dopo l'allineamento) */
    { size_t blob = BP; /* gia' allineato a 32 sopra */
      for (int b = 0; b < 2; b++) { uint16_t dh=0x3800u; memcpy(B+blob+b*34,&dh,2);
        for(int j=0;j<32;j++) B[blob+b*34+2+j]=(unsigned char)(int8_t)j; }
      BP = blob + 68; }
    int fail = 0;
    if (!llm_gguf_parse(&g, B, BP)) { char why[160]; printf("FAIL parse: %s\n", llm_gguf_errmsg(&g, why, sizeof why)); return 1; }
    int ok = (g.n_tensors == 2 && g.n_kv == 5 && g.nti == 2 && g.nkv == 5);
    printf("%s %-40s\n", ok ? "PASS" : "FAIL", "gguf header/kv/tensor count"); if (!ok) fail++;
    ok = (llm_gguf_u(&g, "deepseek4.block_count", 0, NULL) == 43 &&
          llm_gguf_u(&g, "deepseek4.embedding_length", 0, NULL) == 4096);
    printf("%s %-40s\n", ok ? "PASS" : "FAIL", "gguf u32 lookup"); if (!ok) fail++;
    { double e = llm_gguf_f(&g, "deepseek4.hyper_connection.epsilon", 0, NULL);
      ok = (e > 9e-7 && e < 1.1e-6); }
    printf("%s %-40s\n", ok ? "PASS" : "FAIL", "gguf f32 lookup"); if (!ok) fail++;
    ok = (strcmp(g.ti[0].name, "token_embd.weight") == 0 && g.ti[0].ndim == 2 &&
          g.ti[0].dims[1] == 129280);
    printf("%s %-40s\n", ok ? "PASS" : "FAIL", "gguf tensor info"); if (!ok) fail++;
    /* dequant del tensore Q8_0 (idx 1): d=0.5, q=j -> w=0.5*j */
    { static float out[64]; ok = llm_gguf_dequant_tensor(&g, 1, out) && llm_gguf_tensor_nbytes(&g,1)==68;
      for (int b=0;b<2 && ok;b++) for(int j=0;j<32;j++) if (fabsf(out[b*32+j]-0.5f*(float)j)>1e-6f) ok=0; }
    printf("%s %-40s\n", ok ? "PASS" : "FAIL", "dequant tensore Q8_0 (blob)"); if (!ok) fail++;

    /* lookup per nome + load f32 */
    ok = (llm_gguf_find_tensor(&g, "test.q8") == 1 && llm_gguf_find_tensor(&g, "nope") == -1);
    printf("%s %-40s\n", ok ? "PASS" : "FAIL", "find_tensor per nome"); if (!ok) fail++;
    { int64_t n = 0; float* w = llm_gguf_load_f32(&g, "test.q8", &n);
      ok = (w && n == 64); for (int j = 0; j < 64 && ok; j++) if (fabsf(w[j] - 0.5f*(float)(j%32)) > 1e-6f) ok = 0;
      free(w); }
    printf("%s %-40s\n", ok ? "PASS" : "FAIL", "load_f32(test.q8)"); if (!ok) fail++;

    LlmDsv4Config c;
    ok = llm_dsv4_config_from_gguf(&g, &c);
    ok = ok && c.n_layer == 43 && c.n_embd == 4096 && c.n_head == 64 && c.n_head_kv == 1 &&
         c.n_expert == 256 && c.n_swa == 128 && c.n_hc == 4;
    printf("%s %-40s\n", ok ? "PASS" : "FAIL", "config da GGUF (override + default)"); if (!ok) fail++;

    /* open_file: scrive il buffer su file temporaneo e lo riapre via mmap */
    { const char* path = "/tmp/llmcodec_gguf_open_test.gguf";
      FILE* f = fopen(path, "wb"); ok = (f != NULL);
      if (f) { ok = (fwrite(B, 1, BP, f) == BP); fclose(f); }
      LlmEngineMoeGguf g2; void* map = NULL; size_t ml = 0;
      ok = ok && llm_gguf_open_file(path, &g2, &map, &ml);
      if (ok) {
          int64_t n = 0; float* w = llm_gguf_load_f32(&g2, "test.q8", &n);
          ok = (w && n == 64); for (int j = 0; j < 64 && ok; j++) if (fabsf(w[j] - 0.5f*(float)(j%32)) > 1e-6f) ok = 0;
          free(w);
          llm_gguf_close_file(&g2, map, ml);
      }
      unlink(path);
      printf("%s %-40s\n", ok ? "PASS" : "FAIL", "open_file (mmap) + load_f32"); if (!ok) fail++;
    }
    printf(fail ? "FAILURES: %d\n" : "ALL PASS\n", fail);
    return fail;
}
