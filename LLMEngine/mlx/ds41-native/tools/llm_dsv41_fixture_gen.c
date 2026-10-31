/* llm_dsv41_fixture_gen.c — fixture safetensors del modello tiny CSA2 DeepSeek V4.1, in C puro.
 *
 * Sostituisce `gen_dsv41_st_ref.py` (Python vietato). I pesi NON si rigenerano: sono gli stessi
 * array gia' presenti in `llm_dsv41_fwd_csa_ref.h` (prodotti col modello di riferimento), quindi i
 * logits attesi restano validi. I probe quantizzati si ricostruiscono per INVERSIONE dai valori
 * dequantizzati attesi di `llm_dsv41_st_ref.h` (byte fp8/fp4 + scala ue8m0 che li riproducono
 * esattamente), invece di ripetere l'RNG del generatore originale.
 *
 * Scrive in <out> (default /tmp/dsv41_st):
 *   tiny.safetensors, sharded/{shard0,shard1}.safetensors + model.safetensors.index.json + config.json
 * Uso: llm_dsv41_fixture_gen [--out <dir>] [--no-probes]. Build/run: tools/dsv41_fixtures.sh */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "../llm_dsv41_fwd_csa_ref.h"
#include "../llm_dsv41_st_ref.h"
#include "../llm_dsv41_dspark_spec_ref.h"

#define MAXT 160
typedef struct {
    char name[96];
    const char* dtype;
    int nd, shape[3];
    const void* data;
    size_t bytes;
} Tens;
static Tens T[MAXT];
static int NT = 0;

static void add(const char* name, const char* dtype, size_t esz, const void* data, int nd, int a, int b, int c) {
    Tens* t = &T[NT++];
    if (NT > MAXT) { fprintf(stderr, "troppi tensori\n"); exit(2); }
    snprintf(t->name, sizeof t->name, "%s", name);
    t->dtype = dtype; t->nd = nd;
    t->shape[0] = a; t->shape[1] = b; t->shape[2] = c;
    t->data = data;
    t->bytes = esz * (size_t)a * (nd > 1 ? (size_t)b : 1) * (nd > 2 ? (size_t)c : 1);
}
static void f1(const char* n, const float* d, int a)               { add(n, "F32", 4, d, 1, a, 0, 0); }
static void f2(const char* n, const float* d, int a, int b)        { add(n, "F32", 4, d, 2, a, b, 0); }

static void write_st(const char* path, int from, int to) {
    FILE* f = fopen(path, "wb");
    char* hdr = (char*)malloc(1 << 20);
    size_t hl = 0, off = 0;
    if (!f || !hdr) { perror(path); exit(1); }
    hl += (size_t)sprintf(hdr + hl, "{");
    for (int i = from; i < to; i++) {
        Tens* t = &T[i];
        hl += (size_t)sprintf(hdr + hl, "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[", i > from ? "," : "", t->name, t->dtype);
        for (int k = 0; k < t->nd; k++) hl += (size_t)sprintf(hdr + hl, "%s%d", k ? "," : "", t->shape[k]);
        hl += (size_t)sprintf(hdr + hl, "],\"data_offsets\":[%zu,%zu]}", off, off + t->bytes);
        off += t->bytes;
    }
    hl += (size_t)sprintf(hdr + hl, "}");
    while (hl % 8) hdr[hl++] = ' ';
    uint64_t n = hl;   /* little-endian host (arm64/x86) */
    fwrite(&n, 8, 1, f);
    fwrite(hdr, 1, hl, f);
    for (int i = from; i < to; i++) fwrite(T[i].data, 1, T[i].bytes, f);
    fclose(f);
    free(hdr);
}

/* ── probe: inversione dei valori dequantizzati ─────────────────────────────── */
static float e4m3(int b) {   /* e4m3fn, exp bias 7 */
    int e = (b >> 3) & 15, m = b & 7;
    float v = e == 0 ? ldexpf((float)m / 8.0f, -6) : ldexpf(1.0f + (float)m / 8.0f, e - 7);
    return (b & 0x80) ? -v : v;
}
static const float E2M1[16] = { 0, .5f, 1, 1.5f, 2, 3, 4, 6, 0, -.5f, -1, -1.5f, -2, -3, -4, -6 };

static int find_e4m3(float v) {
    for (int b = 0; b < 256; b++) if ((b & 0x7F) != 0x7F && e4m3(b) == v) return b;
    return -1;
}
static int find_e2m1(float v) {
    for (int b = 0; b < 16; b++) if (E2M1[b] == v && b != 8) return b;
    return -1;
}

static uint8_t fp8_w[64 * 64], fp8_s[4], fp4_w[64 * 32], fp4_s[64 * 2];
static uint16_t bf16_w[64];

static int build_probes(void) {
    for (int bi = 0; bi < 2; bi++) for (int bj = 0; bj < 2; bj++) {   /* fp8: blocchi 32x32 */
        int ok = 0;
        for (int s = 120; s < 136 && !ok; s++) {
            float sc = ldexpf(1.0f, s - 127);
            int good = 1;
            for (int i = 0; i < 32 && good; i++) for (int j = 0; j < 32 && good; j++)
                good = find_e4m3(REF_ST_FP8[(bi * 32 + i) * 64 + bj * 32 + j] / sc) >= 0;
            if (!good) continue;
            for (int i = 0; i < 32; i++) for (int j = 0; j < 32; j++)
                fp8_w[(bi * 32 + i) * 64 + bj * 32 + j] =
                    (uint8_t)find_e4m3(REF_ST_FP8[(bi * 32 + i) * 64 + bj * 32 + j] / sc);
            fp8_s[bi * 2 + bj] = (uint8_t)s; ok = 1;
        }
        if (!ok) { fprintf(stderr, "probe fp8: blocco (%d,%d) non invertibile\n", bi, bj); return -1; }
    }
    for (int r = 0; r < 64; r++) for (int h = 0; h < 2; h++) {   /* fp4: righe x 32 elementi */
        int ok = 0;
        for (int s = 120; s < 136 && !ok; s++) {
            float sc = ldexpf(1.0f, s - 127);
            int good = 1;
            for (int k = 0; k < 32 && good; k++) good = find_e2m1(REF_ST_FP4[r * 64 + h * 32 + k] / sc) >= 0;
            if (!good) continue;
            for (int k = 0; k < 32; k += 2) {
                int lo = find_e2m1(REF_ST_FP4[r * 64 + h * 32 + k] / sc);
                int hi = find_e2m1(REF_ST_FP4[r * 64 + h * 32 + k + 1] / sc);
                fp4_w[r * 32 + h * 16 + k / 2] = (uint8_t)(lo | (hi << 4));
            }
            fp4_s[r * 2 + h] = (uint8_t)s; ok = 1;
        }
        if (!ok) { fprintf(stderr, "probe fp4: riga %d/%d non invertibile\n", r, h); return -1; }
    }
    for (int i = 0; i < 64; i++) {   /* bf16: i valori attesi sono gia' bf16 esatti */
        uint32_t u; memcpy(&u, &REF_ST_BF16[i], 4);
        if (u & 0xFFFFu) { fprintf(stderr, "probe bf16[%d] non esatto\n", i); return -1; }
        bf16_w[i] = (uint16_t)(u >> 16);
    }
    return 0;
}

static void tensors(int probes) {
    const int DIM = REF_CSA_DIM, E = REF_CSA_E, DI = REF_CSA_DI, HD = REF_CSA_H * REF_CSA_D;
    const int OGL = REF_CSA_OG * REF_CSA_OLORA, GRP = HD / REF_CSA_OG, SZ = REF_CSA_SIZE, HCD = REF_CSA_HC * DIM;
    char n[96];
    f2("embed.weight", REF_CSA_EMBED, REF_CSA_VOCAB, DIM);
    f1("norm.weight", REF_CSA_FNORM, DIM);
    f2("head.weight", REF_CSA_HEAD, REF_CSA_VOCAB, DIM);
    static const float zeros[REF_CSA_E] = {0};
#define L(P, nm) (snprintf(n, sizeof n, "layers." #P ".%s", nm), n)
#define LAYER(P) do { \
    f2(L(P,"hc_attn_fn"), REF_CSA_HCAFN##P, SZ, HCD); f1(L(P,"hc_attn_scale"), REF_CSA_HCAS##P, 3); \
    f1(L(P,"hc_attn_base"), REF_CSA_HCAB##P, SZ); \
    f2(L(P,"hc_ffn_fn"), REF_CSA_HCFFN##P, SZ, HCD); f1(L(P,"hc_ffn_scale"), REF_CSA_HCFFS##P, 3); \
    f1(L(P,"hc_ffn_base"), REF_CSA_HCFFB##P, SZ); \
    f1(L(P,"attn_norm.weight"), REF_CSA_ANORM##P, DIM); f1(L(P,"ffn_norm.weight"), REF_CSA_FNORM##P, DIM); \
    f2(L(P,"attn.wq_a.weight"), REF_CSA_WQA##P, REF_CSA_QD, DIM); f1(L(P,"attn.q_norm.weight"), REF_CSA_QN##P, REF_CSA_QD); \
    f2(L(P,"attn.wq_b.weight"), REF_CSA_WQB##P, HD, REF_CSA_QD); f2(L(P,"attn.wkv.weight"), REF_CSA_WKV##P, REF_CSA_D, DIM); \
    f1(L(P,"attn.kv_norm.weight"), REF_CSA_KVN##P, REF_CSA_D); f2(L(P,"attn.wo_a.weight"), REF_CSA_WOA##P, OGL, GRP); \
    f2(L(P,"attn.wo_b.weight"), REF_CSA_WOB##P, DIM, OGL); f1(L(P,"attn.attn_sink"), REF_CSA_SINK##P, REF_CSA_H); \
    f2(L(P,"ffn.gate.weight"), REF_CSA_GW##P, E, DIM); f1(L(P,"ffn.gate.bias"), REF_CSA_GB##P, E); \
    f1(L(P,"ffn.gate.bias_vl"), zeros, E); \
    f2(L(P,"ffn.shared_experts.w1.weight"), REF_CSA_SW1##P, DI, DIM); \
    f2(L(P,"ffn.shared_experts.w2.weight"), REF_CSA_SW2##P, DIM, DI); \
    f2(L(P,"ffn.shared_experts.w3.weight"), REF_CSA_SW3##P, DI, DIM); \
    for (int e = 0; e < E; e++) { \
        snprintf(n, sizeof n, "layers." #P ".ffn.experts.%d.w1.weight", e); f2(n, REF_CSA_EW1##P + (size_t)e*DI*DIM, DI, DIM); \
        snprintf(n, sizeof n, "layers." #P ".ffn.experts.%d.w2.weight", e); f2(n, REF_CSA_EW2##P + (size_t)e*DIM*DI, DIM, DI); \
        snprintf(n, sizeof n, "layers." #P ".ffn.experts.%d.w3.weight", e); f2(n, REF_CSA_EW3##P + (size_t)e*DI*DIM, DI, DIM); \
    } } while (0)
    LAYER(0);
    LAYER(1);
    f2(L(1,"attn.compressor.wkv.weight"), REF_CSA_CMPWKV1, REF_CSA_D, DIM);
    f1(L(1,"attn.compressor.norm.weight"), REF_CSA_CMPNORM1, REF_CSA_D);
    f2(L(1,"attn.compressor.wgate.weight"), REF_CSA_CMPWGATE1, REF_CSA_D, DIM);
    f2(L(1,"attn.indexer.wq_b.weight"), REF_CSA_IDXWQB1, 2 * 32, REF_CSA_QD);
    f2(L(1,"attn.indexer.weights_proj.weight"), REF_CSA_IDXWP1, 2, DIM);
    f2(L(1,"attn.indexer.wk.weight"), REF_CSA_IDXWK1, 32, REF_CSA_D);
    f1(L(1,"attn.indexer.k_norm.weight"), REF_CSA_IDXKN1, 32);
    if (probes) {
        add("probe.fp8.weight", "U8", 1, fp8_w, 2, 64, 64, 0);
        add("probe.fp8.scale", "U8", 1, fp8_s, 2, 2, 2, 0);
        add("probe.fp4.weight", "U8", 1, fp4_w, 2, 64, 32, 0);
        add("probe.fp4.scale", "U8", 1, fp4_s, 2, 64, 2, 0);
        add("probe.bf16.weight", "BF16", 2, bf16_w, 2, 8, 8, 0);
    }
}

static const char CONFIG_JSON[] =
    "{\"model_type\": \"deepseek_v41\", \"text_config\": {"
    "\"hidden_size\": 8, \"moe_intermediate_size\": 8, \"num_hidden_layers\": 2, "
    "\"num_attention_heads\": 2, \"head_dim\": 32, \"qk_rope_head_dim\": 4, "
    "\"q_lora_rank\": 8, \"o_lora_rank\": 8, \"o_groups\": 2, \"vocab_size\": 64, "
    "\"n_routed_experts\": 4, \"n_shared_experts\": 1, \"num_experts_per_tok\": 2, "
    "\"scoring_func\": \"sqrtsoftplus\", \"routed_scaling_factor\": 1.0, "
    "\"rms_norm_eps\": 1e-20, \"sliding_window\": 4, \"hc_mult\": 4, "
    "\"hc_sinkhorn_iters\": 20, \"hc_eps\": 1e-6, \"compress_ratios\": [0, 2], "
    "\"kv_source_layer_ids\": [1], \"index_source_layer_ids\": [1], "
    "\"index_n_heads\": 2, \"index_head_dim\": 32, \"index_topk\": 64, "
    "\"engram_layer_ids\": [], \"engram_num_embeddings\": [], "
    "\"rope_theta\": 10000, \"compress_rope_theta\": 40000}}";


/* ── mtp.safetensors: backbone (1 layer) + stage MTP 0, dal blocco di `llm_dsv41_dspark_spec_ref.h` ──
 * Il blocco REF_DS_* e' lo stesso modello dei riferimenti DSR_* (stesse proposte/bias/confidenza).
 * Il backbone `layers.0.*` riusa gli stessi pesi: i test MTP lo usano solo per parita' MTP-vs-greedy. */
static void mtp_block(const char* P) {
    const int DIM = 8, E = 4, DI = 8, HD = 64, OGL = 16, GRP = 32, SZ = 24, HCD = 32;
    char n[96];
#define N(s) (snprintf(n, sizeof n, "%s%s", P, s), n)
    f2(N("hc_attn_fn"), REF_DS_HCAFN, SZ, HCD); f1(N("hc_attn_scale"), REF_DS_HCAS, 3); f1(N("hc_attn_base"), REF_DS_HCAB, SZ);
    f2(N("hc_ffn_fn"), REF_DS_HCFFN, SZ, HCD);  f1(N("hc_ffn_scale"), REF_DS_HCFFS, 3); f1(N("hc_ffn_base"), REF_DS_HCFFB, SZ);
    f1(N("attn_norm.weight"), REF_DS_ANORM, DIM); f1(N("ffn_norm.weight"), REF_DS_FNORM, DIM);
    f2(N("attn.wq_a.weight"), REF_DS_WQA, 8, DIM); f1(N("attn.q_norm.weight"), REF_DS_QN, 8);
    f2(N("attn.wq_b.weight"), REF_DS_WQB, HD, 8);  f2(N("attn.wkv.weight"), REF_DS_WKV, 32, DIM);
    f1(N("attn.kv_norm.weight"), REF_DS_KVN, 32);  f2(N("attn.wo_a.weight"), REF_DS_WOA, OGL, GRP);
    f2(N("attn.wo_b.weight"), REF_DS_WOB, DIM, OGL); f1(N("attn.attn_sink"), REF_DS_SINK, 2);
    f2(N("ffn.gate.weight"), REF_DS_GW, E, DIM); f1(N("ffn.gate.bias"), REF_DS_GB, E);
    f2(N("ffn.shared_experts.w1.weight"), REF_DS_SW1, DI, DIM);
    f2(N("ffn.shared_experts.w2.weight"), REF_DS_SW2, DIM, DI);
    f2(N("ffn.shared_experts.w3.weight"), REF_DS_SW3, DI, DIM);
    for (int e = 0; e < E; e++) {
        snprintf(n, sizeof n, "%sffn.experts.%d.w1.weight", P, e); f2(n, REF_DS_EW1 + (size_t)e*DI*DIM, DI, DIM);
        snprintf(n, sizeof n, "%sffn.experts.%d.w2.weight", P, e); f2(n, REF_DS_EW2 + (size_t)e*DIM*DI, DIM, DI);
        snprintf(n, sizeof n, "%sffn.experts.%d.w3.weight", P, e); f2(n, REF_DS_EW3 + (size_t)e*DI*DIM, DI, DIM);
    }
#undef N
}

static void write_mtp(const char* out) {
    char p[1024];
    NT = 0;
    f2("embed.weight", REF_DS_EMBED, 64, 8);
    f2("head.weight", REF_DS_HEAD, 64, 8);
    f1("norm.weight", REF_DS_OUTNORM, 8);
    mtp_block("layers.0.");
    mtp_block("mtp.0.");
    f2("mtp.0.main_proj.weight", REF_DS_MAINPROJ, 8, 8);
    f1("mtp.0.main_norm.weight", REF_DS_MAINNORM, 8);
    f1("mtp.0.norm.weight", REF_DS_OUTNORM, 8);
    f2("mtp.0.markov_head.embed.weight", REF_DS_MKEMB, 64, 4);
    f2("mtp.0.markov_head.head.weight", REF_DS_MKHEAD, 64, 4);
    f2("mtp.0.confidence_head.proj.weight", REF_DS_CONFPROJ, 1, 12);
    snprintf(p, sizeof p, "%s/mtp.safetensors", out);
    write_st(p, 0, NT);
    printf("scritto: %s | %d tensori\n", p, NT);
}

int main(int argc, char** argv) {
    const char* out = "/tmp/dsv41_st";
    int probes = 1;
    char p[1024];
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--no-probes")) probes = 0;
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else { fprintf(stderr, "uso: %s [--out <dir>] [--no-probes]\n", argv[0]); return 2; }
    }
    if (probes && build_probes() != 0) return 1;
    tensors(probes);
    snprintf(p, sizeof p, "%s/sharded", out);
    mkdir(out, 0755); mkdir(p, 0755);
    snprintf(p, sizeof p, "%s/tiny.safetensors", out);
    write_st(p, 0, NT);
    printf("scritto: %s | %d tensori\n", p, NT);

    int half = NT / 2;
    snprintf(p, sizeof p, "%s/sharded/shard0.safetensors", out); write_st(p, 0, half);
    snprintf(p, sizeof p, "%s/sharded/shard1.safetensors", out); write_st(p, half, NT);
    snprintf(p, sizeof p, "%s/sharded/model.safetensors.index.json", out);
    FILE* f = fopen(p, "w");
    if (!f) { perror(p); return 1; }
    fprintf(f, "{\"metadata\": {\"format\": \"pt\"}, \"weight_map\": {");
    for (int i = 0; i < NT; i++)
        fprintf(f, "%s\"%s\": \"shard%d.safetensors\"", i ? ", " : "", T[i].name, i < half ? 0 : 1);
    fprintf(f, "}}");
    fclose(f);
    snprintf(p, sizeof p, "%s/sharded/config.json", out);
    f = fopen(p, "w");
    if (!f) { perror(p); return 1; }
    fputs(CONFIG_JSON, f);
    fclose(f);
    printf("scritto: %s/sharded | 2 shard + index + config.json\n", out);
    write_mtp(out);
    return 0;
}
