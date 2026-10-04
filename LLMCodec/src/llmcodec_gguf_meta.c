/* llmcodec_gguf_meta.c — METADATA GGUF per il converter modello->GGUF.
 *
 * Contiene: (1) un mini-lettore JSON per `config.json` (jfind/jnum/jflt/mslurp) usato
 * per gli iperparametri (<arch>.*: block_count, context_length, head_count, rope...);
 * (2) il builder dei KV `general.*`, `<arch>.*`, `tokenizer.ggml.*` (LlmCodecGgufMetaKV) e il
 * registro di ownership (Reg) che li libera.
 *
 * NON legge tokenizer.json: i token/merges arrivano gia' digeriti (LlmCodecTokenizerMeta) dal
 * progetto tokenizer (llmtoken) — unica sorgente. Vedi llmcodec_gguf_meta.h. */
#include "llmcodec_gguf_meta.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ── JSON minimale ── */
static char* mslurp(const char* p) {
    FILE* f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n <= 0) { fclose(f); return NULL; }
    char* b = (char*)malloc((size_t)n + 1); if (!b) { fclose(f); return NULL; }
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return NULL; }
    fclose(f); b[n] = 0; return b;
}
static const char* jfind(const char* s, const char* key) {
    if (!s || !key) return NULL;
    char pat[128]; snprintf(pat, sizeof pat, "\"%s\"", key);
    const char* p = strstr(s, pat); if (!p) return NULL; p += strlen(pat);
    while (*p==' '||*p=='\t'||*p=='\n'||*p=='\r') ++p;
    return (*p==':') ? p+1 : NULL;
}
static long long jnum(const char* s, const char* key, long long def) {
    const char* p = jfind(s, key); if (!p) return def;
    char* e = NULL; long long v = strtoll(p, &e, 10); return (e==p) ? def : v;
}
static double jflt(const char* s, const char* key, double def) {
    const char* p = jfind(s, key); if (!p) return def;
    char* e = NULL; double v = strtod(p, &e); return (e==p) ? def : v;
}
/* stringa JSON tra virgolette per `key` -> out (cap); ritorna 1 se trovata. */
static int jstr(const char* s, const char* key, char* out, size_t cap) {
    const char* p = jfind(s, key); if (!p) return 0;
    while (*p==' '||*p=='\t'||*p=='\n'||*p=='\r') ++p;
    if (*p != '"') return 0;
    ++p; size_t i = 0; while (*p && *p != '"' && i + 1 < cap) out[i++] = *p++; out[i] = 0; return 1;
}
static char* jstr_alloc(const char* s, const char* key) {
    const char* p = jfind(s, key); if (!p) return NULL;
    while (*p==' '||*p=='\t'||*p=='\n'||*p=='\r') ++p;
    if (*p != '"') return NULL; ++p;
    size_t cap = 256, n = 0; char* o = (char*)malloc(cap); if (!o) return NULL;
    for (; *p && *p != '"'; ++p) {
        unsigned char c = (unsigned char)*p;
        if (c == '\\') { ++p; char e = *p;
            if (e=='n') c='\n'; else if (e=='t') c='\t'; else if (e=='r') c='\r';
            else if (e=='b') c='\b'; else if (e=='f') c='\f';
            else if (e=='u') { unsigned v=0; for (int k=0;k<4&&p[1];k++){ ++p; char h=*p; v=(v<<4)|(unsigned)((h>='0'&&h<='9')?h-'0':(h>='a'&&h<='f')?h-'a'+10:(h>='A'&&h<='F')?h-'A'+10:0); }
                if (n+4>=cap){cap*=2;o=(char*)realloc(o,cap);}
                if (v<0x80) o[n++]=(char)v; else if (v<0x800){o[n++]=(char)(0xC0|(v>>6));o[n++]=(char)(0x80|(v&0x3F));}
                else {o[n++]=(char)(0xE0|(v>>12));o[n++]=(char)(0x80|((v>>6)&0x3F));o[n++]=(char)(0x80|(v&0x3F));} continue; }
            else c=(unsigned char)e; }
        if (n+1>=cap){cap*=2;o=(char*)realloc(o,cap);} o[n++]=(char)c;
    }
    o[n]=0; return o;
}

/* ── registrazione allocazioni (per il free finale) ── */
typedef struct { void** p; int n, cap; } Reg;
static void reg(Reg* r, void* x) { if (!x) return; if (r->n==r->cap){r->cap=r->cap?r->cap*2:64; r->p=realloc(r->p,(size_t)r->cap*sizeof(void*));} r->p[r->n++]=x; }
static void reg_free(Reg* r) { for (int i=0;i<r->n;i++) free(r->p[i]); free(r->p); }

/* ── builder KV ── */
typedef struct { LlmCodecGgufMetaKV* kv; int n, cap; Reg reg; } ML;
static LlmCodecGgufMetaKV* ml_add(ML* m, const char* key) {
    if (m->n==m->cap){m->cap=m->cap?m->cap*2:64; m->kv=realloc(m->kv,(size_t)m->cap*sizeof(LlmCodecGgufMetaKV));}
    LlmCodecGgufMetaKV* k=&m->kv[m->n++]; memset(k,0,sizeof *k);
    char* kd=strdup(key); reg(&m->reg,kd); k->key=kd; return k;
}
static void add_u32(ML* m, const char* key, uint32_t v){ LlmCodecGgufMetaKV* k=ml_add(m,key); k->type=LLMCODEC_GGUF_MT_U32; k->u32=v; }
static void add_i32(ML* m, const char* key, int32_t v){ LlmCodecGgufMetaKV* k=ml_add(m,key); k->type=LLMCODEC_GGUF_MT_I32; k->i32=v; }
static void add_f32(ML* m, const char* key, float v){ LlmCodecGgufMetaKV* k=ml_add(m,key); k->type=LLMCODEC_GGUF_MT_F32; k->f32=v; }
static void add_bool(ML* m, const char* key, int v){ LlmCodecGgufMetaKV* k=ml_add(m,key); k->type=LLMCODEC_GGUF_MT_BOOL; k->b=v; }
static void add_str(ML* m, const char* key, const char* v){ if(!v)return; LlmCodecGgufMetaKV* k=ml_add(m,key); k->type=LLMCODEC_GGUF_MT_STRING; char* d=strdup(v); reg(&m->reg,d); k->str=d; }
static void add_arr_str(ML* m, const char* key, char** arr, int n){ LlmCodecGgufMetaKV* k=ml_add(m,key); k->type=LLMCODEC_GGUF_MT_ARRAY; k->arr=(const char* const*)arr; k->arr_n=n; k->arr_count=(uint64_t)n; }

/* chiave "<arch>.<suffix>" */
static void add_u32a(ML* m, const char* arch, const char* suf, uint32_t v){ char b[96]; snprintf(b,sizeof b,"%s.%s",arch,suf); add_u32(m,b,v); }
static void add_f32a(ML* m, const char* arch, const char* suf, float v){ char b[96]; snprintf(b,sizeof b,"%s.%s",arch,suf); add_f32(m,b,v); }
static void add_boola(ML* m, const char* arch, const char* suf, int v){ char b[96]; snprintf(b,sizeof b,"%s.%s",arch,suf); add_bool(m,b,v); }
static void add_arra(ML* m, const char* arch, const char* suf, char** a, int n){ char b[96]; snprintf(b,sizeof b,"%s.%s",arch,suf); add_arr_str(m,b,a,n); }


int llmcodec_gguf_meta_build(const char* model_dir, const char* arch, const LlmCodecTokenizerMeta* tm, LlmCodecGgufMetaKV** out_kv, int* out_n) {
    if (!model_dir || !arch || !out_kv || !out_n) return -1;
    *out_kv=NULL; *out_n=0;
    char p[1300]; snprintf(p,sizeof p,"%s/config.json",model_dir); char* cj=mslurp(p);
    if (!cj) return -1;
    ML m; memset(&m,0,sizeof m);
    /* general */
    add_str(&m,"general.architecture",arch);
    add_str(&m,"general.name",arch);
    /* hyperparametri (nomi HF comuni) */
    long long nl=jnum(cj,"num_hidden_layers",-1);
    if (nl>=0) add_u32a(&m,arch,"block_count",(uint32_t)nl);
    long long cl=jnum(cj,"max_position_embeddings",-1); if (cl<0) cl=jnum(cj,"model_max_length",-1);
    if (cl>=0) add_u32a(&m,arch,"context_length",(uint32_t)cl);
    long long hs=jnum(cj,"hidden_size",-1); if (hs<0) hs=jnum(cj,"n_embd",-1);
    if (hs>=0) add_u32a(&m,arch,"embedding_length",(uint32_t)hs);
    long long ff=jnum(cj,"intermediate_size",-1); if (ff<0) ff=jnum(cj,"n_inner",-1);
    if (ff>=0) add_u32a(&m,arch,"feed_forward_length",(uint32_t)ff);
    long long nh=jnum(cj,"num_attention_heads",-1); if (nh>=0) add_u32a(&m,arch,"attention.head_count",(uint32_t)nh);
    long long nkv=jnum(cj,"num_key_value_heads",-1); if (nkv<0) nkv=nh;
    if (nkv>=0) add_u32a(&m,arch,"attention.head_count_kv",(uint32_t)nkv);
    double eps=jflt(cj,"rms_norm_eps",-1.0); if (eps<0) eps=jflt(cj,"layer_norm_eps",-1.0);
    /* Molti checkpoint omettono eps dal config (default RMSNorm 1e-5, come llama.cpp/OLMoE):
     * senza la chiave il loader llama.cpp rifiuta il modello ("key not found in model"). */
    if (eps<0) eps=1e-5;
    add_f32a(&m,arch,"attention.layer_norm_rms_epsilon",(float)eps);
    double rt=jflt(cj,"rope_theta",-1.0); if (rt>=0) add_f32a(&m,arch,"rope.freq_base",(float)rt);
    long long vs=jnum(cj,"vocab_size",-1); if (vs>=0) add_u32a(&m,arch,"vocab_size",(uint32_t)vs);
    long long hd=jnum(cj,"head_dim",-1); if (hd>=0) add_u32a(&m,arch,"attention.key_length",(uint32_t)hd);
    long long ne=jnum(cj,"num_experts",-1); if (ne<0) ne=jnum(cj,"num_local_experts",-1);
    if (ne>=0) add_u32a(&m,arch,"expert_count",(uint32_t)ne);
    long long nt=jnum(cj,"num_experts_per_tok",-1); if (nt>=0) add_u32a(&m,arch,"expert_used_count",(uint32_t)nt);
    long long nsh=jnum(cj,"num_shared_experts",-1); if (nsh<0) nsh=jnum(cj,"n_shared_experts",-1);
    if (nsh>=0) add_u32a(&m,arch,"expert_shared_count",(uint32_t)nsh);
    /* MoE: il FFN degli esperti ha la SUA dimensione (moe_intermediate_size). */
    long long mi=jnum(cj,"moe_intermediate_size",-1); if (mi<0) mi=jnum(cj,"expert_intermediate_size",-1);
    if (mi>=0) add_u32a(&m,arch,"expert_feed_forward_length",(uint32_t)mi);
    long long fk=jnum(cj,"first_k_dense_replace",-1); if (fk>=0) add_u32a(&m,arch,"attention.first_k_dense_replace",(uint32_t)fk);
    long long ng=jnum(cj,"n_group",-1); if (ng>=0) add_u32a(&m,arch,"expert_group_count",(uint32_t)ng);
    long long tg=jnum(cj,"topk_group",-1); if (tg>=0) add_u32a(&m,arch,"expert_group_used_count",(uint32_t)tg);
    { char sf[32]; if (jstr(cj,"scoring_func",sf,sizeof sf)) { char b[96]; snprintf(b,sizeof b,"%s.expert_scoring_func",arch); add_str(&m,b,sf); } }
    /* MLA (DeepSeek) */
    { long long v; v=jnum(cj,"kv_lora_rank",-1);     if(v>=0) add_u32a(&m,arch,"attention.kv_lora_rank",(uint32_t)v);
                     v=jnum(cj,"qk_nope_head_dim",-1); if(v>=0) add_u32a(&m,arch,"attention.qk_nope_head_dim",(uint32_t)v);
                     v=jnum(cj,"qk_rope_head_dim",-1); if(v>=0) add_u32a(&m,arch,"attention.qk_rope_head_dim",(uint32_t)v);
                     v=jnum(cj,"v_head_dim",-1);       if(v>=0) add_u32a(&m,arch,"attention.v_head_dim",(uint32_t)v);
                     v=jnum(cj,"q_lora_rank",-1);      if(v>=0) add_u32a(&m,arch,"attention.q_lora_rank",(uint32_t)v); }
    /* special token ids */
    long long bos=jnum(cj,"bos_token_id",-1), eos=jnum(cj,"eos_token_id",-1), pad=jnum(cj,"pad_token_id",-1);
    if (bos<0) bos=eos;   /* tokenizer senza bos dedicato (OLMoE): llama.cpp usa eos come bos */
    /* tokenizer: dati pre-digeriti dal progetto tokenizer (tm). NON sono nostri:
     * le stringhe appartengono al chiamante -> NON registrarle nel Reg. */
    if (tm && tm->tokens && tm->n_tokens > 0) {
        add_str(&m,"tokenizer.ggml.model", tm->model ? tm->model : "gpt2");
        const char* const* toks = tm->tokens; int ntok = tm->n_tokens;
        /* llama.cpp usa n_vocab = len(tokenizer.ggml.tokens): DEVE combaciare col vocab_size del
         * modello (l'embedding e' [n_embd, vocab_size]). Se il tokenizer ha meno id del vocab_size
         * (padding del modello, es. Qwen2.5: 151665 token reali vs 151936 di embedding), si estende
         * con placeholder. L'array paddato e' nostro (registrato); le stringhe restano del tokenizer. */
        if (vs > 0 && (long long)ntok < vs) {
            char** pad = (char**)malloc((size_t)vs * sizeof(char*));
            for (int i = 0; i < ntok; i++) pad[i] = (char*)toks[i];
            for (long long i = ntok; i < vs; i++) pad[i] = (char*)"";
            reg(&m.reg, pad);
            toks = (const char* const*)pad; ntok = (int)vs;
        }
        add_arra(&m,"tokenizer.ggml","tokens",(char**)toks,ntok);
        if (tm->merges && tm->n_merges > 0)
            add_arra(&m,"tokenizer.ggml","merges",(char**)tm->merges,tm->n_merges);
    }
    if (bos>=0) add_u32(&m,"tokenizer.ggml.bos_token_id",(uint32_t)bos);
    if (eos>=0) add_u32(&m,"tokenizer.ggml.eos_token_id",(uint32_t)eos);
    if (pad>=0) add_u32(&m,"tokenizer.ggml.padding_token_id",(uint32_t)pad);
    add_bool(&m,"tokenizer.ggml.add_bos_token", jnum(cj,"add_bos_token",1)!=0);
    add_bool(&m,"tokenizer.ggml.add_eos_token", jnum(cj,"add_eos_token",0)!=0);
    /* chat_template REALE (da tokenizer_config.json) -> tokenizer.chat_template. */
    { char tcp[1300]; snprintf(tcp, sizeof tcp, "%s/tokenizer_config.json", model_dir);
      char* tcj = mslurp(tcp);
      if (tcj) { char* tpl = jstr_alloc(tcj, "chat_template"); if (tpl) { add_str(&m, "tokenizer.chat_template", tpl); free(tpl); } free(tcj); } }
    /* tokenizer.json + tokenizer_config.json + config.json VERBATIM (round-trip fedele). */
    { char tjp[1300]; snprintf(tjp, sizeof tjp, "%s/tokenizer.json", model_dir);
      char* tj = mslurp(tjp); if (tj) { add_str(&m, "tokenizer.gmb.json", tj); free(tj); } }
    { char tcp2[1300]; snprintf(tcp2, sizeof tcp2, "%s/tokenizer_config.json", model_dir);
      char* tcj2 = mslurp(tcp2); if (tcj2) { add_str(&m, "tokenizer.gmb.config", tcj2); free(tcj2); } }
    { char cfp[1300]; snprintf(cfp, sizeof cfp, "%s/config.json", model_dir);
      char* cf = mslurp(cfp); if (cf) { add_str(&m, "model.gmb.config", cf); free(cf); } }
    free(cj);
    /* trasferisci ownership: la free libera il Reg (chiavi, str, array) + kv */
    *out_kv=m.kv; *out_n=m.n;
    /* conserva il Reg per il free: lo agganciamo via un campo? Semplice: usiamo una free che
     * ricostruisce i puntatori. Per non complicare, teniamo il Reg in una tabella statica. */
    { extern void llmcodec_gguf_meta_track(LlmCodecGgufMetaKV*, Reg*); llmcodec_gguf_meta_track(m.kv,&m.reg); }
    return 0;
}
void llmcodec_gguf_meta_free_built(LlmCodecGgufMetaKV* kv, int n){ (void)n; extern void llmcodec_gguf_meta_release(LlmCodecGgufMetaKV*); llmcodec_gguf_meta_release(kv); free(kv); }

/* ── ownership KV -> Reg (single-threaded; pochi container per processo) ── */
#define LLMCODEC_META_MAX 64
static struct { LlmCodecGgufMetaKV* kv; Reg reg; int used; } g_meta[LLMCODEC_META_MAX];
void llmcodec_gguf_meta_track(LlmCodecGgufMetaKV* kv, Reg* r) {
    for (int i = 0; i < LLMCODEC_META_MAX; i++) if (!g_meta[i].used) { g_meta[i].kv = kv; g_meta[i].reg = *r; g_meta[i].used = 1; return; }
}
void llmcodec_gguf_meta_release(LlmCodecGgufMetaKV* kv) {
    for (int i = 0; i < LLMCODEC_META_MAX; i++) if (g_meta[i].used && g_meta[i].kv == kv) { reg_free(&g_meta[i].reg); g_meta[i].used = 0; return; }
}
