/* gmb_asr_model.c — loader dei pesi Qwen3-ASR (HF/safetensors multi-shard).
 * Reader safetensors minimale self-contained (header JSON + pread per tensore); nessuna dipendenza.
 * Encoder -> F32 (i bf16 sono convertiti all'load); decoder -> BF16 nativo. Config rilevata dalla
 * presenza di `thinker.audio_tower.layers.18` (0.6B: 18 layer; 1.7B: 24). Riferimento: antirez/qwen-asr. */
#include "gmb_asr.h"

#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* ===== reader safetensors minimale ===== */
typedef struct { char name[160]; char dtype[8]; int ndim; uint64_t shape[8]; uint64_t begin, end; } StT;
typedef struct { int fd; uint64_t data_base; StT* t; int nt; } StShard;
typedef struct { StShard* sh; int ns; } StStore;

static const StT* st_find(const StStore* s, const char* name) {
    for (int i = 0; i < s->ns; i++) for (int j = 0; j < s->sh[i].nt; j++) if (strcmp(s->sh[i].t[j].name, name) == 0) return &s->sh[i].t[j];
    return NULL;
}
static const StShard* st_shard_of(const StStore* s, const StT* t) {
    for (int i = 0; i < s->ns; i++) if (t >= s->sh[i].t && t < s->sh[i].t + s->sh[i].nt) return &s->sh[i];
    return NULL;
}

/* scanner JSON minimale: estrae i tensori dall'header safetensors. */
static int parse_header(const char* js, size_t n, StT** out, int* out_n) {
    size_t i = 0; int cap = 64, nt = 0; StT* t = (StT*)calloc((size_t)cap, sizeof(StT));
    while (i < n && js[i] != '{') i++;
    if (i >= n) return -1; i++;
    for (;;) {
        while (i < n && (js[i] == ' ' || js[i] == ',' || js[i] == '\n' || js[i] == '\r' || js[i] == '\t')) i++;
        if (i >= n || js[i] == '}') break;
        if (js[i] != '"') return -1; i++;
        char key[160]; int kl = 0;
        while (i < n && js[i] != '"') { if (kl < 159) key[kl++] = js[i]; i++; }
        key[kl] = 0; if (i < n) i++;
        while (i < n && js[i] != ':') i++;
        if (i < n) i++;
        if (strcmp(key, "__metadata__") == 0) { /* salta il valore */ int d = 0; while (i < n) { if (js[i] == '{') d++; else if (js[i] == '}') { d--; if (d == 0) { i++; break; } } i++; } continue; }
        while (i < n && js[i] != '{') i++;
        if (i >= n) break; i++;
        StT e; memset(&e, 0, sizeof e); snprintf(e.name, sizeof e.name, "%s", key);
        for (;;) {
            while (i < n && (js[i] == ' ' || js[i] == ',' || js[i] == '\n' || js[i] == '\r' || js[i] == '\t')) i++;
            if (i >= n || js[i] == '}') { i++; break; }
            if (js[i] != '"') break; i++;
            char f[32]; int fl = 0; while (i < n && js[i] != '"') { if (fl < 31) f[fl++] = js[i]; i++; } f[fl] = 0; if (i < n) i++;
            while (i < n && js[i] != ':') i++; if (i < n) i++;
            while (i < n && (js[i] == ' ')) i++;
            if (strcmp(f, "dtype") == 0) { while (i < n && js[i] != '"') i++; if (i < n) i++; int dl = 0; while (i < n && js[i] != '"') { if (dl < 7) e.dtype[dl++] = js[i]; i++; } e.dtype[dl] = 0; if (i < n) i++; }
            else if (strcmp(f, "shape") == 0) { while (i < n && js[i] != '[') i++; if (i < n) i++; int nd = 0; while (i < n && js[i] != ']') { if (js[i] >= '0' && js[i] <= '9') { uint64_t v = 0; while (js[i] >= '0' && js[i] <= '9') { v = v * 10 + (uint64_t)(js[i] - '0'); i++; } if (nd < 8) e.shape[nd++] = v; } else i++; } e.ndim = nd; if (i < n) i++; }
            else if (strcmp(f, "data_offsets") == 0) { while (i < n && js[i] != '[') i++; if (i < n) i++; uint64_t v[2] = {0, 0}; int k = 0; while (i < n && js[i] != ']') { if (js[i] >= '0' && js[i] <= '9') { uint64_t x = 0; while (js[i] >= '0' && js[i] <= '9') { x = x * 10 + (uint64_t)(js[i] - '0'); i++; } if (k < 2) v[k++] = x; } else i++; } e.begin = v[0]; e.end = v[1]; if (i < n) i++; }
            else { while (i < n && js[i] != ',' && js[i] != '}') i++; }
        }
        if (nt == cap) { cap *= 2; t = (StT*)realloc(t, (size_t)cap * sizeof(StT)); }
        t[nt++] = e;
    }
    *out = t; *out_n = nt;
    return 0;
}

static int load_shard(const char* path, StShard* sh) {
    int fd = open(path, O_RDONLY); if (fd < 0) return -1;
    struct stat sb; if (fstat(fd, &sb) != 0 || sb.st_size < 8) { close(fd); return -1; }
    unsigned char p8[8]; if (pread(fd, p8, 8, 0) != 8) { close(fd); return -1; }
    uint64_t hl = 0; for (int i = 0; i < 8; i++) hl |= (uint64_t)p8[i] << (8 * i);
    if (hl > (uint64_t)sb.st_size - 8) { close(fd); return -1; }
    char* js = (char*)malloc((size_t)hl + 1); if (!js || pread(fd, js, (size_t)hl, 8) != (ssize_t)hl) { free(js); close(fd); return -1; }
    js[hl] = 0;
    int rc = parse_header(js, (size_t)hl, &sh->t, &sh->nt); free(js);
    if (rc != 0) { close(fd); return -1; }
    sh->fd = fd; sh->data_base = 8 + hl;
    return 0;
}

static int suffix(const char* s, const char* suf) { size_t a = strlen(s), b = strlen(suf); return a >= b && strcmp(s + a - b, suf) == 0; }

static StStore* store_open(const char* dir) {
    DIR* d = opendir(dir); if (!d) return NULL;
    StStore* s = (StStore*)calloc(1, sizeof(StStore)); s->sh = (StShard*)calloc(256, sizeof(StShard));
    struct dirent* e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.' || !suffix(e->d_name, ".safetensors")) continue;
        char p[1400]; snprintf(p, sizeof p, "%s/%s", dir, e->d_name);
        if (s->ns < 256 && load_shard(p, &s->sh[s->ns]) == 0) s->ns++;
    }
    closedir(d);
    if (s->ns == 0) { free(s->sh); free(s); return NULL; }
    return s;
}
static void store_free(StStore* s) { if (!s) return; for (int i = 0; i < s->ns; i++) { free(s->sh[i].t); close(s->sh[i].fd); } free(s->sh); free(s); }

/* ===== modello ===== */
struct GmbAsrModel {
    GmbAsrEncConfig ecfg; GmbAsrDecConfig dcfg;
    GmbAsrEncoder enc; GmbAsrDecoder dec;
    void** allocs; int n_alloc, cap_alloc;
};

static void* oreg(GmbAsrModel* m, void* p) {
    if (!p) return NULL;
    if (m->n_alloc == m->cap_alloc) { m->cap_alloc = m->cap_alloc ? m->cap_alloc * 2 : 128; m->allocs = (void**)realloc(m->allocs, (size_t)m->cap_alloc * sizeof(void*)); }
    m->allocs[m->n_alloc++] = p; return p;
}

static size_t numel(const StT* t) { size_t n = 1; for (int i = 0; i < t->ndim; i++) n *= (size_t)t->shape[i]; return n; }

/* bf16 -> f32 (encoder e norme decoder). Tutti i tensori del checkpoint sono BF16. */
static float* load_bf16_f32(GmbAsrModel* m, const StStore* s, const char* name) {
    const StT* t = st_find(s, name); if (!t) { fprintf(stderr, "asr: manca %s\n", name); return NULL; }
    const StShard* sh = st_shard_of(s, t); size_t n = numel(t);
    uint16_t* b = (uint16_t*)malloc((n ? n : 1) * sizeof(uint16_t));
    float* a = (float*)malloc((n ? n : 1) * sizeof(float));
    if (!b || !a || pread(sh->fd, b, n * sizeof(uint16_t), (off_t)(sh->data_base + t->begin)) != (ssize_t)(n * sizeof(uint16_t))) { free(b); free(a); return NULL; }
    uint32_t* d = (uint32_t*)(void*)a; for (size_t i = 0; i < n; i++) d[i] = (uint32_t)b[i] << 16;
    free(b); return (float*)oreg(m, a);
}
/* bf16 nativo (decoder). */
static uint16_t* load_bf16(GmbAsrModel* m, const StStore* s, const char* name) {
    const StT* t = st_find(s, name); if (!t) { fprintf(stderr, "asr: manca %s\n", name); return NULL; }
    const StShard* sh = st_shard_of(s, t); size_t n = numel(t);
    uint16_t* b = (uint16_t*)malloc((n ? n : 1) * sizeof(uint16_t));
    if (!b || pread(sh->fd, b, n * sizeof(uint16_t), (off_t)(sh->data_base + t->begin)) != (ssize_t)(n * sizeof(uint16_t))) { free(b); return NULL; }
    return (uint16_t*)oreg(m, b);
}

#define ENC "thinker.audio_tower."

static int load_encoder(GmbAsrModel* m, const StStore* s) {
    GmbAsrEncoder* e = &m->enc; char nm[512];
    e->conv1_w = load_bf16_f32(m, s, ENC "conv2d1.weight"); e->conv1_b = load_bf16_f32(m, s, ENC "conv2d1.bias");
    e->conv2_w = load_bf16_f32(m, s, ENC "conv2d2.weight"); e->conv2_b = load_bf16_f32(m, s, ENC "conv2d2.bias");
    e->conv3_w = load_bf16_f32(m, s, ENC "conv2d3.weight"); e->conv3_b = load_bf16_f32(m, s, ENC "conv2d3.bias");
    e->conv_out_w = load_bf16_f32(m, s, ENC "conv_out.weight");
    if (!e->conv1_w || !e->conv2_w || !e->conv3_w || !e->conv_out_w) return -1;
    int L = m->ecfg.layers;
    for (int i = 0; i < L; i++) {
        GmbAsrEncLayer* l = &e->layers[i];
        snprintf(nm, sizeof nm, ENC "layers.%d.self_attn.q_proj.weight", i); l->wq = load_bf16_f32(m, s, nm);
        snprintf(nm, sizeof nm, ENC "layers.%d.self_attn.q_proj.bias", i);   l->bq = load_bf16_f32(m, s, nm);
        snprintf(nm, sizeof nm, ENC "layers.%d.self_attn.k_proj.weight", i); l->wk = load_bf16_f32(m, s, nm);
        snprintf(nm, sizeof nm, ENC "layers.%d.self_attn.k_proj.bias", i);   l->bk = load_bf16_f32(m, s, nm);
        snprintf(nm, sizeof nm, ENC "layers.%d.self_attn.v_proj.weight", i); l->wv = load_bf16_f32(m, s, nm);
        snprintf(nm, sizeof nm, ENC "layers.%d.self_attn.v_proj.bias", i);   l->bv = load_bf16_f32(m, s, nm);
        snprintf(nm, sizeof nm, ENC "layers.%d.self_attn.out_proj.weight", i); l->wo = load_bf16_f32(m, s, nm);
        snprintf(nm, sizeof nm, ENC "layers.%d.self_attn.out_proj.bias", i);   l->bo = load_bf16_f32(m, s, nm);
        snprintf(nm, sizeof nm, ENC "layers.%d.self_attn_layer_norm.weight", i); l->attn_norm_w = load_bf16_f32(m, s, nm);
        snprintf(nm, sizeof nm, ENC "layers.%d.self_attn_layer_norm.bias", i);   l->attn_norm_b = load_bf16_f32(m, s, nm);
        snprintf(nm, sizeof nm, ENC "layers.%d.fc1.weight", i); l->fc1 = load_bf16_f32(m, s, nm);
        snprintf(nm, sizeof nm, ENC "layers.%d.fc1.bias", i);   l->bfc1 = load_bf16_f32(m, s, nm);
        snprintf(nm, sizeof nm, ENC "layers.%d.fc2.weight", i); l->fc2 = load_bf16_f32(m, s, nm);
        snprintf(nm, sizeof nm, ENC "layers.%d.fc2.bias", i);   l->bfc2 = load_bf16_f32(m, s, nm);
        snprintf(nm, sizeof nm, ENC "layers.%d.final_layer_norm.weight", i); l->ffn_norm_w = load_bf16_f32(m, s, nm);
        snprintf(nm, sizeof nm, ENC "layers.%d.final_layer_norm.bias", i);   l->ffn_norm_b = load_bf16_f32(m, s, nm);
        if (!l->wq || !l->wk || !l->wv || !l->wo) return -1;
    }
    e->ln_post_w = load_bf16_f32(m, s, ENC "ln_post.weight"); e->ln_post_b = load_bf16_f32(m, s, ENC "ln_post.bias");
    e->proj1_w = load_bf16_f32(m, s, ENC "proj1.weight"); e->bproj1 = load_bf16_f32(m, s, ENC "proj1.bias");
    e->proj2_w = load_bf16_f32(m, s, ENC "proj2.weight"); e->bproj2 = load_bf16_f32(m, s, ENC "proj2.bias");
    return (e->ln_post_w && e->proj1_w && e->proj2_w) ? 0 : -1;
}

static int load_decoder(GmbAsrModel* m, const StStore* s) {
    GmbAsrDecoder* d = &m->dec; char nm[512];
    d->tok_embeddings = load_bf16(m, s, "thinker.model.embed_tokens.weight");
    d->norm = load_bf16_f32(m, s, "thinker.model.norm.weight");
    if (!d->tok_embeddings || !d->norm) return -1;
    for (int i = 0; i < m->dcfg.layers; i++) {
        GmbAsrDecLayer* l = &d->layers[i];
        snprintf(nm, sizeof nm, "thinker.model.layers.%d.self_attn.q_proj.weight", i); l->wq = load_bf16(m, s, nm);
        snprintf(nm, sizeof nm, "thinker.model.layers.%d.self_attn.k_proj.weight", i); l->wk = load_bf16(m, s, nm);
        snprintf(nm, sizeof nm, "thinker.model.layers.%d.self_attn.v_proj.weight", i); l->wv = load_bf16(m, s, nm);
        snprintf(nm, sizeof nm, "thinker.model.layers.%d.self_attn.o_proj.weight", i); l->wo = load_bf16(m, s, nm);
        snprintf(nm, sizeof nm, "thinker.model.layers.%d.self_attn.q_norm.weight", i); l->q_norm = load_bf16_f32(m, s, nm);
        snprintf(nm, sizeof nm, "thinker.model.layers.%d.self_attn.k_norm.weight", i); l->k_norm = load_bf16_f32(m, s, nm);
        snprintf(nm, sizeof nm, "thinker.model.layers.%d.input_layernorm.weight", i); l->input_norm = load_bf16_f32(m, s, nm);
        snprintf(nm, sizeof nm, "thinker.model.layers.%d.post_attention_layernorm.weight", i); l->post_attn_norm = load_bf16_f32(m, s, nm);
        snprintf(nm, sizeof nm, "thinker.model.layers.%d.mlp.gate_proj.weight", i); l->gate = load_bf16(m, s, nm);
        snprintf(nm, sizeof nm, "thinker.model.layers.%d.mlp.up_proj.weight", i); l->up = load_bf16(m, s, nm);
        snprintf(nm, sizeof nm, "thinker.model.layers.%d.mlp.down_proj.weight", i); l->down = load_bf16(m, s, nm);
        if (!l->wq || !l->wk || !l->wv || !l->wo || !l->gate || !l->up || !l->down) return -1;
    }
    return 0;
}

GmbAsrModel* gmb_asr_load_model(const char* dir) {
    if (dir == NULL) return NULL;
    StStore* s = store_open(dir); if (!s) { fprintf(stderr, "asr: nessun .safetensors in %s\n", dir); return NULL; }
    GmbAsrModel* m = (GmbAsrModel*)calloc(1, sizeof(GmbAsrModel));
    /* detect 0.6B / 1.7B dalla presenza del layer 18 dell'encoder */
    int is17 = (st_find(s, ENC "layers.18.self_attn.q_proj.weight") != NULL);
    m->ecfg.d_model = is17 ? 1024 : 896; m->ecfg.layers = is17 ? 24 : 18; m->ecfg.heads = is17 ? 16 : 14;
    m->ecfg.head_dim = 64; m->ecfg.ffn_dim = is17 ? 4096 : 3584; m->ecfg.output_dim = is17 ? 2048 : 1024;
    m->ecfg.n_window = 50; m->ecfg.n_window_infer = 800; m->ecfg.chunk_size = 100;
    m->dcfg.hidden = is17 ? 2048 : 1024; m->dcfg.layers = 28; m->dcfg.heads = 16; m->dcfg.kv_heads = 8;
    m->dcfg.head_dim = 128; m->dcfg.intermediate = is17 ? 6144 : 3072; m->dcfg.vocab = 151936;
    m->dcfg.rms_eps = 1e-6f; m->dcfg.rope_theta = 1e6f;
    int rc = load_encoder(m, s); if (rc == 0) rc = load_decoder(m, s);
    if (rc != 0) { store_free(s); gmb_asr_model_free(m); return NULL; }
    store_free(s);
    return m;
}

void gmb_asr_model_free(GmbAsrModel* m) {
    if (!m) return;
    for (int i = 0; i < m->n_alloc; i++) free(m->allocs[i]);
    free(m->allocs); free(m);
}
const GmbAsrEncConfig* gmb_asr_model_ecfg(const GmbAsrModel* m) { return &m->ecfg; }
const GmbAsrDecConfig* gmb_asr_model_dcfg(const GmbAsrModel* m) { return &m->dcfg; }
const GmbAsrEncoder*   gmb_asr_model_encoder(const GmbAsrModel* m) { return &m->enc; }
const GmbAsrDecoder*   gmb_asr_model_decoder(const GmbAsrModel* m) { return &m->dec; }
