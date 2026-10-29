/* ============================================================================
 * llm_mlx_gguf_io.c — supporto GGUF generico: detection, tokenizer JSON BPE, caricamento pesi F16, role map/arch resolve, footprint safetensors.
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"
#include "llmdesc.h"
#include "llmcodec_rolemap.h"
#include "llmcodec_fp16.h"
#include "llm_gguf.h"
#include "llm_ggml.h"
#include "llm_dsv4_tokenizer_json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <execinfo.h>   /* LLM_MLX_ERRTRACE: backtrace sull'error handler MLX */

/* FIX (2026-09-17) — footprint del modello su disco (somma dei .safetensors nella dir).
 * Serve al wired AUTO: un limite wired INFERIORE al footprint garantisce thrash (paging continuo)
 * e allocazioni fallite. Evidenza misurata: Mixtral-8x22B = 73 GB con cap auto di 64 GB ->
 * prefill 3,8 tok/s e decode `n<0` (`decode fallito`), mentre mlx_lm sulla stessa macchina fa
 * 272/22,7 tok/s; Qwen3.8-Flash-Next 97 GB idem. I modelli SOTTO il cap (gpt-oss-120b 61 GB,
 * Qwen3.5-122B 61 GB) non erano toccati -> il pattern dei fallimenti coincide con i modelli
 * piu' grandi del cap. I .safetensors sono memory-mapped, quindi la loro somma e' il footprint. */
size_t llm_mlx_dir_safetensors_bytes(const char* dir) {
    if (dir == NULL) return 0;
    DIR* d = opendir(dir);
    if (d == NULL) return 0;
    size_t total = 0;
    struct dirent* e;
    while ((e = readdir(d)) != NULL) {
        const size_t n = strlen(e->d_name);
        if (n < 12 || strcmp(e->d_name + n - 12, ".safetensors") != 0) continue;
        char path[4096];
        if ((size_t)snprintf(path, sizeof(path), "%s/%s", dir, e->d_name) >= sizeof(path)) continue;
        struct stat st;
        if (stat(path, &st) == 0 && S_ISREG(st.st_mode)) total += (size_t)st.st_size;
    }
    closedir(d);
    return total;
}

/* ── Rilevamento GGUF single-file supportato dal motore MLX ──────────────────
 * Auto-route (BASIC/C#): un .gguf la cui architettura e' gestita direttamente dal motore
 * MLX (oggi solo DS4 = arch GGUF "deepseek4") viene caricato da llm_ai_mlx_load_model;
 * ogni altro .gguf resta a llama.cpp. Header-only llm_gguf => nessun link aggiuntivo. */
int llm_ai_mlx_gguf_supported(const char* path) {
    if (path == NULL || *path == '\0') return 0;
    LlmEngineMoeGguf g; void* map = NULL; size_t maplen = 0;
    if (!llm_gguf_open_file(path, &g, &map, &maplen)) return 0;
    const LlmGgufVal* v = llm_gguf_find_val(&g, "general.architecture");
    int ok = 0;
    if (v != NULL && v->type == GGSTR && v->s != NULL && v->slen > 0) {
        char at[64]; int n = (int)v->slen; if (n > 63) n = 63;
        memcpy(at, v->s, (size_t)n); at[n] = 0;
        if (strcmp(at, "deepseek4") == 0) ok = 1;                       /* forward DS4 dedicato */
        else if (strcmp(at, "deepseek_v41") != 0 && find_arch_ops(at) != NULL)
            ok = 1;                                                     /* GGUF generico via ARCH_TABLE */
    }
    llm_gguf_close_file(&g, map, maplen);
    return ok;
}

/* `tokenizer.ggml.model` == "gpt2": BPE byte-level, costruibile dai KV del GGUF a prescindere dall architettura. */
static int gguf_tok_is_gpt2(const LlmEngineMoeGguf* g) {
    const LlmGgufVal* v = llm_gguf_find_val(g, "tokenizer.ggml.model");
    return v != NULL && v->type == GGSTR && v->s != NULL && v->slen == 4 && memcmp(v->s, "gpt2", 4) == 0;
}

/* JSON del tokenizer BPE byte-level (GPT-2/Llama-3/Qwen) dai KV `tokenizer.ggml.*` del GGUF.
 * Stessa struttura del builder DS4 (buffer + escape), ma pattern di split GPT-2/Llama-3. */
static char* llm_mlx_gguf_tokenizer_json_bpe(const LlmEngineMoeGguf* g, size_t* out_len) {
    const char** toks = NULL; size_t* tlens = NULL;
    long nt = llm_gguf_kv_array_str(g, "tokenizer.ggml.tokens", &toks, &tlens);
    if (nt <= 0) return NULL;
    const char** mrg = NULL; size_t* mlens = NULL;
    long nm = llm_gguf_kv_array_str(g, "tokenizer.ggml.merges", &mrg, &mlens);
    if (nm < 0) nm = 0;
    LlmDsv4TokBuf b; memset(&b, 0, sizeof b);
    LLM_D4PUT(&b, "{\"model\":{\"type\":\"BPE\",\"vocab\":{");
    for (long i = 0; i < nt; i++) {
        if (i) llm_dsv4_tok_c(&b, ',');
        llm_dsv4_tok_jsonstr(&b, toks[i], tlens[i]);
        char num[24]; int k = snprintf(num, sizeof num, ":%ld", i); llm_dsv4_tok_put(&b, num, (size_t)k);
    }
    LLM_D4PUT(&b, "},\"merges\":[");
    for (long i = 0; i < nm; i++) { if (i) llm_dsv4_tok_c(&b, ','); llm_dsv4_tok_jsonstr(&b, mrg[i], mlens[i]); }
    LLM_D4PUT(&b, "]},");
    {   /* speciali (token_type 2/3/4) come added_tokens: match letterale prima del BPE */
        int32_t* tt = NULL; long ntt = llm_gguf_kv_array_i32(g, "tokenizer.ggml.token_type", &tt);
        LLM_D4PUT(&b, "\"added_tokens\":["); int first = 1;
        if (ntt > 0 && tt != NULL) {
            for (long i = 0; i < nt && i < ntt; i++) {
                int ty = (int)tt[i];
                if ((ty != 2 && ty != 3 && ty != 4) || tlens[i] == 0) continue;
                if (!first) llm_dsv4_tok_c(&b, ','); first = 0;
                LLM_D4PUT(&b, "{\"id\":");
                char num[24]; int k = snprintf(num, sizeof num, "%ld", i); llm_dsv4_tok_put(&b, num, (size_t)k);
                LLM_D4PUT(&b, ",\"content\":"); llm_dsv4_tok_jsonstr(&b, toks[i], tlens[i]);
                LLM_D4PUT(&b, ",\"special\":true}");
            }
        }
        free(tt);
        LLM_D4PUT(&b, "],");
    }
    LLM_D4PUT(&b, "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"Split\",\"behavior\":\"Isolated\",\"invert\":false,\"pattern\":{\"Regex\":");
    static const char* pat = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    llm_dsv4_tok_jsonstr(&b, pat, strlen(pat));
    LLM_D4PUT(&b, "}},{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"use_regex\":false}]}}");
    free(toks); free(tlens); free(mrg); free(mlens);
    if (out_len) *out_len = b.len;
    return b.p;
}

/* JSON del tokenizer DS4 (BPE + pre-tokenizer JoyAI) costruita dal GGUF. Ritorna una stringa
 * malloc'ata (libera con llm_ai_mlx_free_str) e ne scrive la lunghezza in *out_len; NULL su
 * errore. Il C# la passa a llmtoken_load_json (byte + len, non stringa: la vocab
 * byte-level puo' non essere UTF-8 valida). */
char* llm_ai_mlx_gguf_tokenizer_json(const char* path, size_t* out_len) {
    if (out_len) *out_len = 0;
    if (path == NULL || *path == '\0') return NULL;
    LlmEngineMoeGguf g; void* map = NULL; size_t maplen = 0;
    if (!llm_gguf_open_file(path, &g, &map, &maplen)) return NULL;
    char* json = NULL;
    /* Preferisci il `tokenizer.gmb.json` VERBATIM scritto dal converter: round-trip esatto,
     * nessun rebuild BPE (che puo' fallire su BPE non-llama / campi mancanti). */
    { const LlmGgufVal* vs = llm_gguf_find_val(&g, "tokenizer.gmb.json");
      if (vs != NULL && vs->type == GGSTR && vs->s != NULL && vs->slen > 0) {
          char* cp = (char*)malloc((size_t)vs->slen);
          if (cp) { memcpy(cp, vs->s, (size_t)vs->slen); if (out_len) *out_len = (size_t)vs->slen; llm_gguf_close_file(&g, map, maplen); return cp; }
      } }
    const LlmGgufVal* v = llm_gguf_find_val(&g, "general.architecture");
    if (v != NULL && v->type == GGSTR && v->s != NULL && v->slen > 0) {
        char at[64]; int n = (int)v->slen; if (n > 63) n = 63;
        memcpy(at, v->s, (size_t)n); at[n] = 0;
        if (strcmp(at, "deepseek4") == 0) json = llm_dsv4_tokenizer_json_from_gguf(&g, out_len);
        else if (strcmp(at, "deepseek_v41") != 0 && (find_arch_ops(at) != NULL || gguf_tok_is_gpt2(&g)))
            json = llm_mlx_gguf_tokenizer_json_bpe(&g, out_len);   /* BPE byte-level generico (anche archi senza entry in ARCH_TABLE, es. qwen4exp) */
    }
    llm_gguf_close_file(&g, map, maplen);
    return json;
}

void llm_ai_mlx_free_str(char* p) { free(p); }

/* Meta chat del GGUF per il templating del consumer (C#): architettura + chat_template REALE +
 * eos/bos token STRING (risolti dall'id nel vocabolario). JSON malloc-ato; null per i campi
 * assenti. Serve a usare il template del MODELLO invece di assumere deepseek_v4 per ogni .gguf
 * (che su un OLMoE generico produce un eos inesistente -> errore "did not encode to one token"). */
char* llm_ai_mlx_gguf_chat_meta(const char* path, size_t* out_len) {
    if (out_len) *out_len = 0;
    if (path == NULL || *path == '\0') return NULL;
    LlmEngineMoeGguf g; void* map = NULL; size_t maplen = 0;
    if (!llm_gguf_open_file(path, &g, &map, &maplen)) return NULL;
    const char* arch = NULL; const char* tpl = NULL;
    const char* eos = NULL; const char* bos = NULL;
    const LlmGgufVal* va = llm_gguf_find_val(&g, "general.architecture");
    if (va && va->type == GGSTR && va->s) arch = va->s;
    const LlmGgufVal* vt = llm_gguf_find_val(&g, "tokenizer.chat_template");
    if (vt && vt->type == GGSTR && vt->s) tpl = vt->s;
    const char** toks = NULL; size_t* tlens = NULL;
    long nt = llm_gguf_kv_array_str(&g, "tokenizer.ggml.tokens", &toks, &tlens);
    const LlmGgufVal* ve = llm_gguf_find_val(&g, "tokenizer.ggml.eos_token_id");
    const LlmGgufVal* vb = llm_gguf_find_val(&g, "tokenizer.ggml.bos_token_id");
    if (toks && nt > 0) {
        if (ve && ve->type != GGARR && ve->u < (uint64_t)nt) eos = toks[ve->u];
        if (vb && vb->type != GGARR && vb->u < (uint64_t)nt) bos = toks[vb->u];
    }
    LlmDsv4TokBuf b; memset(&b, 0, sizeof b);
    LLM_D4PUT(&b, "{\"architecture\":");
    if (arch) llm_dsv4_tok_jsonstr(&b, arch, strlen(arch)); else LLM_D4PUT(&b, "null");
    LLM_D4PUT(&b, ",\"chat_template\":");
    if (tpl) llm_dsv4_tok_jsonstr(&b, tpl, strlen(tpl)); else LLM_D4PUT(&b, "null");
    LLM_D4PUT(&b, ",\"eos_token\":");
    if (eos) llm_dsv4_tok_jsonstr(&b, eos, strlen(eos)); else LLM_D4PUT(&b, "null");
    LLM_D4PUT(&b, ",\"bos_token\":");
    if (bos) llm_dsv4_tok_jsonstr(&b, bos, strlen(bos)); else LLM_D4PUT(&b, "null");
    LLM_D4PUT(&b, "}");
    free(toks); free(tlens);
    llm_gguf_close_file(&g, map, maplen);
    if (out_len) *out_len = b.len;
    return b.p;
}

/* LLM_MLX_ERRTRACE=1: error handler MLX con backtrace (diagnostica temporanea). */
void llm_mlx_err_trace_handler(const char* msg, void* data) {
    (void)data;
    fprintf(stderr, "MLX error: %s\n", msg ? msg : "(null)");
    void* bt[48];
    int n = backtrace(bt, 48);
    backtrace_symbols_fd(bt, n, 2);
}

/* ── GGUF GENERICO (llama/qwen2/qwen3/…): modello in un SINGOLO GGUF la cui architettura è nella
 * ARCH_TABLE standard (NON DS4/DS4.1, che hanno il forward whole-model). Costruisce cfg dai KV
 * `<arch>.*`, carica i tensori densi in F16 sotto il NOME CANONICO GGUF, e risolve la role map
 * `.llmarch` per la traduzione HF->canonica nel lookup (multishard_get). ── */
bool llm_mlx_load_generic_gguf(const char* gguf_path, LlmMlxModel* m, const char* model_type) {
    LlmEngineMoeGguf g; void* map = NULL; size_t maplen = 0;
    if (!llm_gguf_open_file(gguf_path, &g, &map, &maplen)) { fprintf(stderr, "[gguf] non apribile: %s\n", gguf_path); return false; }
    /* Device/stream MLX (come il path normale/DS4.1): GPU se disponibile. Da impostare PRIMA di
     * creare gli array (mlx_array_new_data usa il device di default). */
    { mlx_device gpu = mlx_device_new_type(MLX_GPU, 0);
      bool gpu_avail = false; mlx_device_is_available(&gpu_avail, gpu);
      mlx_device chosen = gpu_avail ? gpu : mlx_device_new_type(MLX_CPU, 0);
      if (!gpu_avail) mlx_device_free(gpu);
      if (mlx_set_default_device(chosen) != 0) { mlx_device_free(chosen); llm_gguf_close_file(&g, map, maplen); return false; }
      mlx_device_free(m->device); m->device = chosen;
      if (mlx_get_default_stream(&m->stream, m->device) != 0) { llm_gguf_close_file(&g, map, maplen); return false; }
      (void)mlx_set_default_stream(m->stream);   /* registra il default del thread (2026-10-03) */
    }
    ModelConfig* c = &m->cfg;
    memset(c, 0, sizeof *c);
    snprintf(c->model_type, sizeof c->model_type, "%s", model_type);
    snprintf(c->layer_prefix, sizeof c->layer_prefix, "%s", "model.layers.");
    char k[128];
    #define GKV_U(suf, def) (snprintf(k, sizeof k, "%s.%s", model_type, (suf)), (int)llm_gguf_u(&g, k, (uint64_t)(def), NULL))
    c->n_layers = GKV_U("block_count", 0);
    c->hidden_size = GKV_U("embedding_length", 0);
    c->n_heads = GKV_U("attention.head_count", 0);
    c->n_kv_heads = GKV_U("attention.head_count_kv", c->n_heads);
    c->intermediate_size = GKV_U("feed_forward_length", 0);
    c->vocab_size = GKV_U("vocab_size", 0);
    c->head_dim = GKV_U("attention.key_length", c->n_heads ? c->hidden_size / c->n_heads : 0);
    /* MoE: nomi canonici GGUF `expert_count`/`expert_used_count`/`expert_feed_forward_length`.
     * Senza questi top_k=0 -> il top-k del router produce una slice vuota (empty mlx_array). */
    c->n_experts = GKV_U("expert_count", 0);
    c->top_k = GKV_U("expert_used_count", 0);
    c->n_shared_experts = GKV_U("expert_shared_count", 0);
    {
        int eff = GKV_U("expert_feed_forward_length", 0);
        c->moe_intermediate_size = eff > 0 ? eff : (c->n_experts > 1 ? c->intermediate_size : 0);
    }
    #undef GKV_U
    snprintf(k, sizeof k, "%s.attention.layer_norm_rms_epsilon", model_type);
    c->rms_norm_eps = (float)llm_gguf_f(&g, k, 1e-5, NULL);
    snprintf(k, sizeof k, "%s.rope.freq_base", model_type);
    c->rope_theta = (float)llm_gguf_f(&g, k, 10000.0, NULL);
    /* Pesi GGUF già dequantizzati in F16 (path "plain", niente quantized_matmul). */
    c->quant_group_size = 0; c->quant_bits = 0;
    snprintf(c->quant_mode, sizeof c->quant_mode, "%s", "");
    c->tie_word_embeddings = (llm_gguf_find_tensor(&g, "output.weight") < 0);
    if (c->vocab_size <= 0) {   /* fallback: molti GGUF non hanno <arch>.vocab_size -> usa il tokenizer */
        const char** tk = NULL; size_t* tl = NULL;
        long nt = llm_gguf_kv_array_str(&g, "tokenizer.ggml.tokens", &tk, &tl);
        if (nt > 0) c->vocab_size = (int)nt;
        free(tk); free(tl);
    }
    if (c->n_layers <= 0 || c->hidden_size <= 0 || c->n_heads <= 0) {
        fprintf(stderr, "[gguf] %s: metadata <arch>.* mancanti (block_count/embedding_length/head_count)\n", gguf_path);
        llm_gguf_close_file(&g, map, maplen); return false;
    }
    /* Pesi: ogni tensore denso -> F16 -> mappa, chiavato col NOME CANONICO GGUF. */
    mlx_map_string_to_array wm = mlx_map_string_to_array_new();
    int nw = 0;
    for (int i = 0; i < g.nti; i++) {
        const char* tn = g.ti[i].name;
        if (strncmp(tn, "mtp.", 4) == 0 || strncmp(tn, "vision.", 7) == 0 ||
            strncmp(tn, "aligner.", 8) == 0 || strncmp(tn, "image_", 6) == 0) continue;
        int64_t n = 0;
        const uint8_t* p = llm_gguf_tensor_data(&g, i, &n);
        if (p == NULL || n <= 0) continue;
        float* f = (float*)malloc((size_t)n * sizeof(float));
        if (f == NULL || llmcodec_ggml_dequant_row(g.ti[i].type, p, f, n) != 0) { free(f); continue; }
        uint16_t* h = (uint16_t*)malloc((size_t)n * sizeof(uint16_t));
        if (h == NULL) { free(f); continue; }
        for (int64_t j = 0; j < n; j++) h[j] = llmcodec_f16_from_f32(f[j]);
        free(f);
        int dims[4]; int nd = g.ti[i].ndim; if (nd > 4) nd = 4;
        for (int d = 0; d < nd; d++) dims[d] = (int)g.ti[i].dims[nd-1-d];   /* GGUF [in,out] -> HF/MLX [out,in] */
        mlx_array arr = mlx_array_new_data(h, dims, nd, MLX_FLOAT16);
        free(h);
        mlx_map_string_to_array_insert(wm, tn, arr);
        mlx_array_free(arr);
        nw++;
    }
    if (nw == 0) { fprintf(stderr, "[gguf] %s: nessun tensore caricato\n", gguf_path); llm_gguf_close_file(&g, map, maplen); return false; }
    m->weights.shards = (mlx_map_string_to_array*)malloc(sizeof(mlx_map_string_to_array));
    if (m->weights.shards == NULL) { llm_gguf_close_file(&g, map, maplen); return false; }
    m->weights.shards[0] = wm; m->weights.count = 1; m->weights.cap = 1;
    /* Role map `.llmarch` (ABI/<arch>.llmarch): UNICA mappa HF<->GGUF, usata dal lookup. */
    { char dir[1024], gp[1200];
      if (llmdesc_abi_dir(dir, sizeof dir)) {
          const char* arch_file = model_type;
          if (strcmp(model_type, "qwen2vl") == 0) arch_file = "qwen2_omni";   /* Qwen2.5-Omni */
          snprintf(gp, sizeof gp, "%s/%s.llmarch", dir, arch_file);
          LlmCodecRoleMap* rm = (LlmCodecRoleMap*)calloc(1, sizeof(LlmCodecRoleMap));
          if (rm != NULL) { if (llmcodec_rolemap_load(gp, rm) == 0) m->weights.gguf_rm = rm;
                            else { fprintf(stderr, "[gguf] role map assente: %s\n", gp); free(rm); } }
      } }
    llm_gguf_close_file(&g, map, maplen);
    fprintf(stderr, "llm_ai_mlx_load_model: %s (GGUF generico) caricato: %d layer, vocab %d, %d tensori\n",
            model_type, c->n_layers, c->vocab_size, nw);
    return true;
}

/* Risolve la `.llmarch` per un modello: da `config.json` (dir MLX) o **dal GGUF**
 * (`general.architecture`, con alias: es. "deepseek4" -> file `ds4.llmarch`, perche' il modello
 * di antirez non e' il DeepSeek V4 ufficiale). 0 ok / -1. */
int llm_mlx_resolve_arch(const char* path, LlmDesc* out) {
    size_t n = strlen(path);
    if (n > 5 && strcmp(path + n - 5, ".gguf") == 0) {
        LlmEngineMoeGguf g; void* map = NULL; size_t maplen = 0;
        if (!llm_gguf_open_file(path, &g, &map, &maplen)) return -1;
        const LlmGgufVal* v = llm_gguf_find_val(&g, "general.architecture");
        char at[64] = "";
        if (v != NULL && v->type == GGSTR && v->s != NULL && v->slen > 0) {
            int k = (int)v->slen; if (k > 63) k = 63; memcpy(at, v->s, (size_t)k); at[k] = 0;
        }
        llm_gguf_close_file(&g, map, maplen);
        if (at[0] == 0) return -1;
        const char* file = at;
        if (strcmp(at, "deepseek4") == 0) file = "ds4";   /* antirez DS4 != DeepSeek V4 ufficiale */
        else if (strcmp(at, "qwen2vl") == 0) file = "qwen2_omni";   /* Qwen2.5-Omni: arch GGUF "qwen2vl", descrizione "qwen2_omni" */
        char dir[1024], gp[1200];
        if (!llmdesc_abi_dir(dir, sizeof dir)) return -1;
        snprintf(gp, sizeof gp, "%s/%s.llmarch", dir, file);
        return llmdesc_load(gp, out);
    }
    return llmdesc_resolve(path, out, NULL, 0);
}
