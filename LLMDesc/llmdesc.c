/* llmdesc.c — parser .llmarch (descrizione d'architettura). JSON minimale (riusa lo stile jv). */
#include "llmdesc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <dlfcn.h>     /* dladdr: per risolvere ABI/ dalla directory della dylib (CWD-indipendente) */
#include <sys/stat.h>  /* mkdir: per l'auto-build del `.gmbarch` in <model>/arch/ */

static char* slurp(const char* p) {
    FILE* f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char* b=(char*)malloc((size_t)n+1); if(!b){fclose(f);return NULL;}
    if (fread(b,1,(size_t)n,f)!=(size_t)n){free(b);fclose(f);return NULL;} b[n]=0; fclose(f); return b;
}
/* copia la stringa JSON che segue `p` (dopo ':' e gli spazi) in out */
static int jcopy_str(const char* p, char* out, size_t cap) {
    if (!p) return 0;
    while (*p==' '||*p=='\t') p++;
    if (*p!='"') return 0; p++;
    size_t i=0; while (*p && *p!='"' && i<cap-1) out[i++]=*p++;
    out[i]=0; return i>0;
}
static int jbool(const char* j, const char* key) {
    char pat[96]; snprintf(pat,sizeof pat,"\"%s\"",key);
    const char* p=j;
    while ((p=strstr(p,pat))) { const char* c=p+strlen(pat); while(*c==' '||*c=='\t')c++; if(*c==':'){c++;while(*c==' '||*c=='\t')c++; return strncmp(c,"true",4)==0;} p++; }
    return 0;
}
static const char* jv(const char* j, const char* key) {
    char pat[96]; snprintf(pat,sizeof pat,"\"%s\"",key);
    const char* p=j;
    while ((p=strstr(p,pat))) { const char* c=p+strlen(pat); while(*c==' '||*c=='\t')c++; if(*c==':'){c++;while(*c==' '||*c=='\t')c++;return c;} p++; }
    return NULL;
}
static long jnum(const char* j, const char* key, long def) {
    const char* v = jv(j, key); if (!v) return def; char* e; long r = strtol(v, &e, 10); return (e==v) ? def : r;
}
static double jflt2(const char* j, const char* key, double def) {
    const char* v = jv(j, key); if (!v) return def; char* e; double r = strtod(v, &e); return (e==v) ? def : r;
}
/* Come jv, ma solo tra i FIGLI DIRETTI dell'oggetto che inizia a `obj` (puntatore a '{'): non guarda negli oggetti annidati. */
static const char* jdirect(const char* obj, const char* key) {
    int depth = 0; size_t kl = strlen(key);
    for (const char* p = obj; *p; p++) {
        if (*p == '"') {
            const char* s = ++p;
            while (*p && *p != '"') { if (*p == '\\' && p[1]) p++; p++; }
            if (!*p) return NULL;
            const char* q = p + 1;
            while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
            if (depth == 1 && *q == ':' && (size_t)(p - s) == kl && memcmp(s, key, kl) == 0) {
                q++;
                while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
                return q;
            }
            continue;
        }
        if (*p == '{' || *p == '[') depth++;
        else if (*p == '}' || *p == ']') { if (--depth == 0) return NULL; }
    }
    return NULL;
}
/* Valore di config.json: figli diretti di `text_config` (modelli annidati) -> figli diretti della radice -> prima occorrenza ovunque.
 * Senza la ricerca per livello, in Qwen4 `text_config.mtp.num_hidden_layers` (1) nascondeva `text_config.num_hidden_layers` (48). */
static const char* cfg_v(const char* cj, const char* key) {
    const char* tc = strstr(cj, "\"text_config\"");
    if (tc) { const char* b = strchr(tc, '{'); if (b) { const char* v = jdirect(b, key); if (v) return v; } }
    const char* r = strchr(cj, '{');
    if (r) { const char* v = jdirect(r, key); if (v) return v; }
    return jv(cj, key);
}
static long cnum(const char* cj, const char* key, long def) {
    const char* v = cfg_v(cj, key); if (!v) return def; char* e; long r = strtol(v, &e, 10); return (e == v) ? def : r;
}
static double cflt(const char* cj, const char* key, double def) {
    const char* v = cfg_v(cj, key); if (!v) return def; char* e; double r = strtod(v, &e); return (e == v) ? def : r;
}

int llmdesc_load(const char* path, LlmDesc* out) {
    if (!path||!out) return -1;
    memset(out,0,sizeof *out);
    char* j=slurp(path); if(!j) return -1;
    jcopy_str(jv(j,"arch"), out->arch, sizeof out->arch);
    jcopy_str(jv(j,"scale_suffix"), out->scale_suffix, sizeof out->scale_suffix);
    jcopy_str(jv(j,"recipe"), out->recipe, sizeof out->recipe);
    jcopy_str(jv(j,"recipe_boost_fmt"), out->boost_fmt, sizeof out->boost_fmt);
    /* geometry: geometria numerica (auto-costruita nel `.gmbarch`; sostituisce il GMAD). */
    { const char* g=jv(j,"geometry"); if (g && *g=='{') {
        out->geometry.present = 1;
        out->geometry.hidden      = (unsigned)jnum(g,"hidden",0);
        out->geometry.n_layers    = (unsigned)jnum(g,"n_layers",0);
        out->geometry.n_heads     = (unsigned)jnum(g,"n_heads",0);
        out->geometry.n_kv_heads  = (unsigned)jnum(g,"n_kv_heads",out->geometry.n_heads);
        out->geometry.head_dim    = (unsigned)jnum(g,"head_dim",0);
        out->geometry.n_experts   = (unsigned)jnum(g,"n_experts",0);
        out->geometry.topk        = (unsigned)jnum(g,"topk",0);
        out->geometry.inter       = (unsigned)jnum(g,"inter",0);
        out->geometry.vocab       = (unsigned)jnum(g,"vocab",0);
        out->geometry.ctx         = (unsigned)jnum(g,"ctx",0);
        out->geometry.rope_variant= (unsigned)jnum(g,"rope_variant",0);
        out->geometry.router_kind = (unsigned)jnum(g,"router_kind",0);
        out->geometry.norm_kind   = (unsigned)jnum(g,"norm_kind",0);
        out->geometry.flags       = (unsigned)jnum(g,"flags",0);
        out->geometry.eps         = (float)jflt2(g,"eps",1e-5);
        out->geometry.rope_theta  = (float)jflt2(g,"rope_theta",10000.0);
    } }
    /* roles: [{ "match":"..","role":"..","submodel":".." }, ...] */
    { const char* r=jv(j,"roles"); if (r && *r=='[') { const char* p=r+1;
        while (*p && *p!=']' && out->n_roles<128) {
            const char* ob=strchr(p,'{'); if(!ob){break;} const char* cb=strchr(ob,'}'); if(!cb){break;}
            char obj[512]; size_t L=(size_t)(cb-ob+1); if(L>=sizeof obj)L=sizeof obj-1; memcpy(obj,ob,L); obj[L]=0;
            LlmDescRole* R=&out->roles[out->n_roles];
            if (jcopy_str(jv(obj,"match"),R->match,sizeof R->match)) { jcopy_str(jv(obj,"role"),R->role,sizeof R->role); jcopy_str(jv(obj,"submodel"),R->submodel,sizeof R->submodel); out->n_roles++; }
            p=cb+1; } } }
    /* quant: { "ROLE":"fmt" | "ROLE":{weights,activation,accum,output}, "recipe":"_M", ... }.
     * QuantIR I11: 4 assi per RUOLO/OP — pesi (storage) + activation/accum/output (precisione del
     * CALCOLO). Forma stringa = solo pesi (retro-compatibile). */
    { const char* q=jv(j,"quant"); if (q && *q=='{') { const char* p=q+1;
        while (*p && *p!='}' && out->n_quant<128) {
            while(*p==' '||*p==',')p++; if(*p=='}'||!*p)break;
            if (*p!='"'){p++;continue;}
            char key[32]; p++; size_t k=0; while(*p&&*p!='"'&&k<31)key[k++]=*p++; key[k]=0; if(*p=='"')p++;
            while(*p==' '||*p=='\t')p++; if(*p==':')p++;
            while(*p==' '||*p=='\t')p++;
            if (*p=='{') {
                const char* eo=p; int dd=0; do { if(*eo=='{')dd++; else if(*eo=='}'){dd--; if(!dd)break;} eo++; } while(*eo);
                if (strcmp(key,"recipe")!=0 && strcmp(key,"recipe_boost_fmt")!=0) {
                    LlmDescQuant* Q=&out->quant[out->n_quant]; snprintf(Q->role,sizeof Q->role,"%s",key);
                    { const char* v=jv(p,"weights");    if (v && v<eo) jcopy_str(v,Q->fmt,sizeof Q->fmt); }
                    { const char* v=jv(p,"activation"); if (v && v<eo) jcopy_str(v,Q->activation,sizeof Q->activation); }
                    { const char* v=jv(p,"accum");      if (v && v<eo) jcopy_str(v,Q->accum,sizeof Q->accum); }
                    { const char* v=jv(p,"output");     if (v && v<eo) jcopy_str(v,Q->output,sizeof Q->output); }
                    if (!Q->fmt[0]) snprintf(Q->fmt,sizeof Q->fmt,"%s","f16");
                    out->n_quant++;
                }
                p=eo; if(*p=='}')p++;
            } else {
                char val[16]; if (jcopy_str(p,val,sizeof val)) {
                    if (strcmp(key,"recipe")==0 || strcmp(key,"recipe_boost_fmt")==0) { /* gia' letti sopra */ }
                    else { LlmDescQuant* Q=&out->quant[out->n_quant]; snprintf(Q->role,sizeof Q->role,"%s",key); snprintf(Q->fmt,sizeof Q->fmt,"%s",val); out->n_quant++; }
                }
                while(*p && *p!=','){ if(*p=='}')break; p++; }
            } } } }
    /* codecs: { "f16": "llmcodec_f16", ... } -> plugin che implementa il formato */
    { const char* cd=jv(j,"codecs");
      if (cd && *cd=='{') { const char* p=cd+1;
        while (*p && *p!='}' && out->n_codec<16) {
            while (*p && *p!='"' && *p!='}') p++;
            if (*p!='"') break; p++;
            char fmt[32]; size_t k=0; while(*p&&*p!='"'&&k<31)fmt[k++]=*p++; fmt[k]=0; if(*p=='"')p++;
            const char* q=p; while(*q && *q!='"' && *q!='}') q++;
            if (*q!='"') break; q++;
            char pl[32]; size_t m=0; while(*q&&*q!='"'&&m<31)pl[m++]=*q++; pl[m]=0; if(*q=='"')q++;
            snprintf(out->codec_fmt[out->n_codec],32,"%s",fmt);
            snprintf(out->codec_plugin[out->n_codec],32,"%s",pl);
            out->n_codec++;
            p=q;
        } } }

    /* structures: { "attention": { "type":"mla", ... }, "vision": { "type":"vit", ... }, ... } */
    { const char* st=jv(j,"structures");
      if (st && *st=='{') { const char* p=st+1;
        while (*p && *p!='}' && out->n_struct<16) {
            while (*p && *p!='"' && *p!='}') p++;
            if (*p!='"') break; p++;
            char key[32]; size_t k=0; while(*p&&*p!='"'&&k<31)key[k++]=*p++; key[k]=0; if(*p=='"')p++;
            const char* obj=strchr(p,'{'); if(!obj) break;
            const char* eo=obj; { int _d=0; do { if(*eo=='{') _d++; else if(*eo=='}') { _d--; if(_d==0) break; } eo++; } while(*eo); }
            char type[32]=""; const char* tv=jv(obj,"type"); if (eo && tv && tv > eo) tv=NULL; jcopy_str(tv,type,sizeof type);
            char prim[32]=""; const char* pv=jv(obj,"prim"); const char* ss0=jv(obj,"steps");
            if (pv && (pv > eo || (ss0 && pv > ss0))) pv=NULL; jcopy_str(pv,prim,sizeof prim);
            { snprintf(out->structures[out->n_struct].key,32,"%s",key);
              snprintf(out->structures[out->n_struct].type,32,"%s",type);
              snprintf(out->structures[out->n_struct].prim,32,"%s",prim); out->n_struct++; }
            /* steps: [ { "op":"..","prim":".." }, ... ] -> pipeline ordinata (livello 3) */
            { const char* stp = jv(obj, "steps"); if (stp != NULL && stp > eo) stp = NULL;
              if (stp != NULL) { const char* q = strchr(stp, '['); if (q) { q++;
                while (*q && *q != ']' && out->n_struct > 0 && out->structures[out->n_struct-1].n_steps < 12) {
                    const char* o = strchr(q, '{'); const char* rb = strchr(q, ']');
                    if (!o || (rb != NULL && rb < o)) break;
                    const char* oc = strchr(o, '}'); if (!oc) break;
                    char op[24] = "", pr[32] = "";
                    const char* ov = jv(o, "op");   if (oc && ov && ov < oc) jcopy_str(ov, op, sizeof op);
                    const char* pv2 = jv(o, "prim"); if (oc && pv2 && pv2 < oc) jcopy_str(pv2, pr, sizeof pr);
                    if (op[0] || pr[0]) { LlmDescStep* S = &out->structures[out->n_struct-1].steps[out->structures[out->n_struct-1].n_steps++];
                        snprintf(S->op, sizeof S->op, "%s", op); snprintf(S->prim, sizeof S->prim, "%s", pr); }
                    q = oc + 1;
                } } } }
            const char* e=eo; if(!e) break; p=e+1;
        } } }

    /* runtime: { "state":"csa2"|{...}, "kv":{ "dtype":"fp4" }, "activation":{ "dtype":"fp8_e4m3" } }.
     * Metadati ENGINE, non codec: li espone `LLM KV` a scopo informativo. Si isola l'oggetto
     * `runtime` per NON confondere `jv("dtype")` con activation.dtype del wrapper esterno. */
    { const char* rt=jv(j,"runtime");
      if (rt && *rt=='{') {
        const char* reo=rt; int rd=0; do { if(*reo=='{')rd++; else if(*reo=='}'){rd--; if(rd==0)break;} reo++; } while(*reo);
        char rtobj[16384]; size_t RL=(size_t)(reo-rt+1); if(RL>=sizeof rtobj)RL=sizeof rtobj-1;
        memcpy(rtobj,rt,RL); rtobj[RL]=0;
        const char* sv=jv(rtobj,"state");
        if (sv && *sv=='"') jcopy_str(sv,out->runtime_state,sizeof out->runtime_state);
        else if (sv && *sv=='{') snprintf(out->runtime_state,sizeof out->runtime_state,"(object)");
        { const char* kv=jv(rtobj,"kv");
          if (kv && *kv=='{') {
            const char* keo=kv; int kd=0; do { if(*keo=='{')kd++; else if(*keo=='}'){kd--; if(kd==0)break;} keo++; } while(*keo);
            char kvobj[4096]; size_t KL=(size_t)(keo-kv+1); if(KL>=sizeof kvobj)KL=sizeof kvobj-1;
            memcpy(kvobj,kv,KL); kvobj[KL]=0;
            jcopy_str(jv(kvobj,"dtype"),out->runtime_kv_dtype,sizeof out->runtime_kv_dtype);
          } }
        { const char* ac=jv(rtobj,"activation");
          if (ac && *ac=='{') {
            const char* aeo=ac; int ad=0; do { if(*aeo=='{')ad++; else if(*aeo=='}'){ad--; if(ad==0)break;} aeo++; } while(*aeo);
            char acobj[2048]; size_t AL=(size_t)(aeo-ac+1); if(AL>=sizeof acobj)AL=sizeof acobj-1;
            memcpy(acobj,ac,AL); acobj[AL]=0;
            jcopy_str(jv(acobj,"dtype"),out->runtime_activation_dtype,sizeof out->runtime_activation_dtype);
          } }
      } }
    /* source: { "format":"gguf", "mmproj":"..." } — dichiarazione di composizione/modality. */
    { const char* sc=jv(j,"source");
      if (sc && *sc=='{') {
        const char* seo=sc; int d=0; do { if(*seo=='{')d++; else if(*seo=='}'){d--; if(d==0)break;} seo++; } while(*seo);
        char sobj[8192]; size_t SL=(size_t)(seo-sc+1); if(SL>=sizeof sobj)SL=sizeof sobj-1;
        memcpy(sobj,sc,SL); sobj[SL]=0;
        jcopy_str(jv(sobj,"format"),out->source_format,sizeof out->source_format);
        jcopy_str(jv(sobj,"mmproj"),out->source_mmproj,sizeof out->source_mmproj);
      } }
    /* runtime.multimodal / runtime.decode (gia' isolati in `rtobj` sopra). */
    { const char* rt=jv(j,"runtime");
      if (rt && *rt=='{') {
        const char* reo=rt; int rd=0; do { if(*reo=='{')rd++; else if(*reo=='}'){rd--; if(rd==0)break;} reo++; } while(*reo);
        char rtobj[16384]; size_t RL=(size_t)(reo-rt+1); if(RL>=sizeof rtobj)RL=sizeof rtobj-1;
        memcpy(rtobj,rt,RL); rtobj[RL]=0;
        const char* mm=jv(rtobj,"multimodal");
        if (mm && *mm=='{') {
          const char* meo=mm; int md=0; do { if(*meo=='{')md++; else if(*meo=='}'){md--; if(md==0)break;} meo++; } while(*meo);
          char mmobj[2048]; size_t ML=(size_t)(meo-mm+1); if(ML>=sizeof mmobj)ML=sizeof mmobj-1;
          memcpy(mmobj,mm,ML); mmobj[ML]=0;
          jcopy_str(jv(mmobj,"vision"),out->mm_vision,sizeof out->mm_vision);
          out->mm_aligner = jbool(mmobj,"aligner");
          out->mm_present = jbool(mmobj,"present");
        }
        const char* dc=jv(rtobj,"decode");
        if (dc && *dc=='{') {
          const char* deo=dc; int dd=0; do { if(*deo=='{')dd++; else if(*deo=='}'){dd--; if(dd==0)break;} deo++; } while(*deo);
          char dobj[2048]; size_t DL=(size_t)(deo-dc+1); if(DL>=sizeof dobj)DL=sizeof dobj-1;
          memcpy(dobj,dc,DL); dobj[DL]=0;
          out->mm_mtp = jbool(dobj,"mtp");
          out->mm_dspark = jbool(dobj,"dspark");
        }
      } }
    free(j);

    /* Valido se ha una name-map (role map `.llmarch`) OPPURE una geometria (`.gmbarch` auto-costruito)
     * OPPURE almeno un arch: un `.gmbarch` geometry-only (senza ruoli) è un descrittore valido. */
    return (out->n_roles > 0 || out->geometry.present || out->arch[0]) ? 0 : -1;
}
/* dall'arch (es. "DeepseekV41ForCausalLM" o "deepseek_v41_text") al nome file .llmarch */
static void arch_slug(const char* arch, char* out, size_t cap) {
    size_t o = 0;
    for (const char* p = arch; *p && o < cap-1; p++) {
        if (*p >= 'A' && *p <= 'Z') { if (o) out[o++] = '_'; out[o++] = (char)(*p - 'A' + 'a'); }
        else if (*p == '_' || (*p >= 'a' && *p <= 'z') || (*p >= '0' && *p <= '9')) out[o++] = *p;
        else { if (o && out[o-1] != '_') out[o++] = '_'; }
    }
    while (o && out[o-1] == '_') o--;
    out[o] = 0;
}
/* Deriva la `geometry` numerica da config.json, IN MEMORIA (nessun file scritto), quando il descrittore non ne ha una. */
static void derive_geom(const char* hf_dir, LlmDesc* out) {
    char cfgp[2048]; snprintf(cfgp, sizeof cfgp, "%s/config.json", hf_dir);
    char* cj = slurp(cfgp); if (!cj) return;
    /* figli diretti di text_config (modelli annidati) -> radice: vedi cnum/cfg_v */
    long hidden = cnum(cj,"hidden_size",0), layers = cnum(cj,"num_hidden_layers",0);
    long heads  = cnum(cj,"num_attention_heads",0), kvh = cnum(cj,"num_key_value_heads",heads);
    long hd = cnum(cj,"head_dim", (heads>0 && hidden>0) ? hidden/heads : 0);
    long nexp = cnum(cj,"num_experts", cnum(cj,"num_local_experts",0));
    long topk = cnum(cj,"num_experts_per_tok", nexp>0 ? 1 : 0);
    long inter = cnum(cj,"moe_intermediate_size", cnum(cj,"intermediate_size",0));
    long vocab = cnum(cj,"vocab_size",0), ctx = cnum(cj,"max_position_embeddings",0);
    out->geometry.hidden=hidden; out->geometry.n_layers=layers; out->geometry.n_heads=heads;
    out->geometry.n_kv_heads=kvh; out->geometry.head_dim=hd; out->geometry.n_experts=nexp;
    out->geometry.topk=topk; out->geometry.inter=inter; out->geometry.vocab=vocab; out->geometry.ctx=ctx;
    out->geometry.eps=(float)cflt(cj,"rms_norm_eps", cflt(cj,"layer_norm_eps",1e-5));
    out->geometry.rope_theta=(float)cflt(cj,"rope_theta",10000.0);
    out->geometry.flags=(nexp>0)?8u:0u;   /* MOE; gli altri flag li deriva il runtime */
    out->geometry.present=1;
    free(cj);
}
/* Carica `path` in `out`; con `need_roles` accetta SOLO una name-map (n_roles>0). Serve a
 * preferire un descrittore COMPLETO (ABI, con ruoli) su uno geometry-only (stub `.gmbarch`
 * autogenerato in <model>/arch/), che altrimenti lo shadowerebbe. */
static int llmdesc_load_pref(const char* path, LlmDesc* out, int need_roles) {
    LlmDesc t;
    if (llmdesc_load(path, &t) != 0) return 0;
    if (need_roles && t.n_roles == 0) return 0;
    *out = t;
    return 1;
}
static int llmdesc_resolve_core(const char* hf_dir, LlmDesc* out, char* out_path, size_t out_path_cap);
/* Wrapper: risolve + garantisce la `geometry` (dal `.gmbarch` o derivata da config.json). GMAD eliminato. */
int llmdesc_resolve(const char* hf_dir, LlmDesc* out, char* out_path, size_t out_path_cap) {
    int rc = llmdesc_resolve_core(hf_dir, out, out_path, out_path_cap);
    if (rc == 0 && out && !out->geometry.present) derive_geom(hf_dir, out);
    return rc;
}
static int llmdesc_resolve_core(const char* hf_dir, LlmDesc* out, char* out_path, size_t out_path_cap) {
    if (!hf_dir || !out) return -1;
    char cfg[2048]; snprintf(cfg, sizeof cfg, "%s/config.json", hf_dir);
    char* j = slurp(cfg); if (!j) return -1;
    /* PREFERISCI model_type (nome canonico pulito, es. "deepseek_v41"); poi architectures[0]. */
    char arch[96] = "";
    jcopy_str(jv(j, "model_type"), arch, sizeof arch);
    if (!arch[0]) { const char* v = jv(j, "architectures");
        if (v && *v == '[') { const char* q = strchr(v, '"'); if (q) { q++; size_t k=0; while (*q && *q!='"' && k<95) arch[k++]=*q++; arch[k]=0; } } }
    free(j);
    char slug[96]; arch_slug(arch, slug, sizeof slug);
    if (strstr(slug, "_text")) { char* p = strstr(slug, "_text"); if (p) *p = 0; }
    if (strstr(slug, "_vision")) { char* p = strstr(slug, "_vision"); if (p) *p = 0; }
    /* Ricerca del descrittore: UNICA sede = la cartella ABI/ di LLMEnv (vedi llmdesc_abi_dir), file `<arch>.gmbarch`
     * (forma canonica) o `<arch>.llmarch` (legacy). Niente piu' file per modello (<model>/arch/) ne' cartelle `arch/`
     * cercate nella cwd: due copie dello stesso descrittore divergono, e uno stub senza ruoli ombreggiava quello completo. */
    char abi[2048];
    if (llmdesc_abi_dir(abi, sizeof abi)) {
        static const char* exts[] = { "gmbarch", "llmarch" };
        for (int x = 0; x < 2; ++x) {
            char path[4096]; snprintf(path, sizeof path, "%s/%s.%s", abi, slug, exts[x]);
            if (llmdesc_load_pref(path, out, 0)) { if (out_path) snprintf(out_path, out_path_cap, "%s", path); return 0; }
        }
    }
    /* FALLBACK GENERICO (2026-10-03): nessun `.gmbarch` per questo modello -> desc MINIMALE
     * (arch = model_type) e role map GENERICO dedotto dalla convenzione HF/GGUF (llmcodec_
     * rolemap_generic). Cosi' un modello a decoder standard si carica/convertе SENZA file arch;
     * un `.gmbarch` esplicito resta la via per sovrascrivere/estendere (MLA, ibridi, multimodale). */
    if (arch[0]) {
        memset(out, 0, sizeof *out);
        snprintf(out->arch, sizeof out->arch, "%s", arch);
        /* DEDUZIONE delle STRUTTURE dal config.json (HF standard): tipo di attenzione (mla/gqa/mha),
         * MoE, norma. Cosi' il percorso COMPOSABILE del motore ha le primitive anche senza
         * `.gmbarch`. Un arch esplicito resta la via per sovrascrivere. */
        char* cj = slurp(cfg);
        if (cj) {
            const char* s_nkv = jv(cj, "num_key_value_heads");
            const char* s_nh  = jv(cj, "num_attention_heads");
            const char* s_kl  = jv(cj, "kv_lora_rank");
            const char* s_qn  = jv(cj, "qk_nope_head_dim");
            int nkv = s_nkv ? atoi(s_nkv) : 0, nh = s_nh ? atoi(s_nh) : 0;
            int kl = s_kl ? atoi(s_kl) : 0, qn = s_qn ? atoi(s_qn) : 0;
            const char* s_ne = jv(cj, "num_experts"); if (!s_ne) s_ne = jv(cj, "n_routed_experts");
            if (!s_ne) s_ne = jv(cj, "num_local_experts"); if (!s_ne) s_ne = jv(cj, "moe_num_experts");
            int ne = s_ne ? atoi(s_ne) : 0;
            const char* attn = (kl > 0 || qn > 0) ? "mla" : (nkv > 0 && nh > 0 && nkv < nh) ? "gqa" : "mha";
            snprintf(out->structures[out->n_struct].key, 32, "attention");
            snprintf(out->structures[out->n_struct].type, 32, "%s", attn); out->n_struct++;
            if (ne > 0 && out->n_struct < 16) { snprintf(out->structures[out->n_struct].key,32,"moe"); out->structures[out->n_struct].type[0]=0; out->n_struct++; }
            if (out->n_struct < 16) { snprintf(out->structures[out->n_struct].key,32,"norm"); snprintf(out->structures[out->n_struct].type,32,"rmsnorm"); out->n_struct++; }
            snprintf(out->source_format, sizeof out->source_format, "safetensors");
            free(cj);
        }
        if (out_path && out_path_cap) out_path[0] = 0;   /* "" -> role map generico */
        return 0;
    }
    if (out_path) out_path[0] = 0;
    return -1;
}
/* Cartella ABI/ canonica: risale dal cwd cercando una dir `ABI` (dove vivono le descrizioni).
 * Ritorna 1 e scrive il path in `out` se trovata. Usata da LLM ARCH LIST e dal resolve. */
/* Risale da `start` (dir o file) cercando <dir>/ABI | <dir>/LLMEnv/ABI | <dir>/../LLMEnv/ABI.
 * (b)/(c): l'engine vive DENTRO LLMEnv ma i consumatori (FastLLM e altri consumer) sono progetti fratelli.
 * Senza la role map .llmarch il loader GGUF generico cerca i nomi HF (lm_head/embed_tokens/norm)
 * che il GGUF non ha -> "lookup fallito". 1 se trovata (path in `out`). */
static int abi_search_up(const char* start, char* out, size_t cap) {
    char dir[2048]; snprintf(dir, sizeof dir, "%s", start ? start : "");
    for (int up = 0; up < 16 && dir[0]; ++up) {
        char path[4096];
        static const char* tails[] = { "/ABI", "/LLMEnv/ABI", "/../LLMEnv/ABI" };
        for (int t = 0; t < 3; ++t) {
            snprintf(path, sizeof path, "%s%s", dir, tails[t]);
            DIR* dp = opendir(path);
            if (dp != NULL) { closedir(dp); if (out && cap) snprintf(out, cap, "%s", path); return 1; }
        }
        char* sl = strrchr(dir, '/');
        if (sl == NULL) break;
        if (sl == dir) { dir[1] = 0; } else *sl = 0;
    }
    return 0;
}
int llmdesc_abi_dir(char* out, size_t cap) {
    /* (0) Override esplicito (installazioni fuori dall'albero sorgente). */
    const char* env = getenv("LLMDESC_ABI_DIR");
    if (env != NULL && *env != '\0') { if (out && cap) snprintf(out, cap, "%s", env); return 1; }
    /* (1) dalla CWD risalendo. */
    char cwd[2048];
    if (getcwd(cwd, sizeof cwd) != NULL && abi_search_up(cwd, out, cap)) return 1;
    /* (2) CWD-INDIPENDENTE: risali dalla directory di QUESTA dylib (es. Resources/libs/<rid>/ ->
     *     .../ABI). Cosi' il loader funziona anche se il processo gira con una CWD fuori dal repo
     *     (es. test/REPL lanciati da /tmp). */
    { Dl_info info; if (dladdr((void*)(uintptr_t)&llmdesc_abi_dir, &info) != 0 && info.dli_fname != NULL
                       && abi_search_up(info.dli_fname, out, cap)) return 1; }
    return 0;
}

const LlmDescStruct* llmdesc_struct(const LlmDesc* a, const char* key) {
    for (int i = 0; i < a->n_struct; i++) if (strcmp(a->structures[i].key, key) == 0) return &a->structures[i];
    return NULL;
}
int llmdesc_unsupported(const LlmDesc* a, char* out, size_t cap) {
    /* Il converter (via ruoli .llmarch + ricetta) gestisce queste; il compute resta l'engine. */
    static const struct { const char* key; const char* type; } supported[] = {
        { "attention", "mla" }, { "attention", "gqa" }, { "moe", NULL }, { "norm", NULL },
        { "compression", "csa" }, { "compression", "csa_hca" }, { "indexer", "indexer" }, { "indexer", "topk_selector" },
        { "engram", "engram" }, { "engram", "deterministic_router" }, { "vision", "vit" },
    };
    int nbad = 0; size_t o = 0;
    o += (size_t)snprintf(out + o, cap - o, "llmarch '%s':", a->arch);
    for (int i = 0; i < a->n_struct && o < cap; i++) {
        int ok = 0;
        for (size_t k = 0; k < sizeof(supported) / sizeof(supported[0]); k++)
            if (strcmp(supported[k].key, a->structures[i].key) == 0 &&
                (supported[k].type == NULL || strcmp(supported[k].type, a->structures[i].type) == 0)) { ok = 1; break; }
        o += (size_t)snprintf(out + o, cap - o, " %s=%s%s", a->structures[i].key, a->structures[i].type,
                              ok ? "" : "(NON supportata)");
        if (!ok) nbad++;
    }
    if (o < cap) snprintf(out + o, cap - o, " -> %d non supportata/e", nbad);
    return nbad;
}
const char* llmdesc_role(const LlmDesc* a, const char* name) {
    if (!a||!name) return "OTHER";
    for (int i=0;i<a->n_roles;i++) if (strstr(name, a->roles[i].match)) return a->roles[i].role;
    return "OTHER";
}
const char* llmdesc_codec_plugin(const LlmDesc* a, const char* fmt) {
    for (int i = 0; i < a->n_codec; i++) if (strcmp(a->codec_fmt[i], fmt) == 0) return a->codec_plugin[i];
    return NULL;
}
const char* llmdesc_struct_prim(const LlmDesc* a, const char* key) {
    for (int i = 0; i < a->n_struct; i++)
        if (strcmp(a->structures[i].key, key) == 0 && a->structures[i].prim[0]) return a->structures[i].prim;
    return NULL;
}
const char* llmdesc_struct_type(const LlmDesc* a, const char* key) {
    for (int i = 0; i < a->n_struct; i++) if (strcmp(a->structures[i].key, key) == 0) return a->structures[i].type;
    return NULL;
}
const char* llmdesc_fmt(const LlmDesc* a, const char* role) {
    if (!a||!role) return NULL;
    for (int i=0;i<a->n_quant;i++) if (strcmp(a->quant[i].role, role)==0) return a->quant[i].fmt;
    return NULL;
}

/* QuantIR I11 — precisione di CALCOLO per ruolo: activation/accum/output (default f16/f32/fmt). */
const char* llmdesc_activation(const LlmDesc* a, const char* role)
{
    if (a && role) for (int i=0;i<a->n_quant;i++)
        if (strcmp(a->quant[i].role, role)==0 && a->quant[i].activation[0]) return a->quant[i].activation;
    return "f16";
}
const char* llmdesc_accum(const LlmDesc* a, const char* role)
{
    if (a && role) for (int i=0;i<a->n_quant;i++)
        if (strcmp(a->quant[i].role, role)==0 && a->quant[i].accum[0]) return a->quant[i].accum;
    return "f32";
}
const char* llmdesc_output(const LlmDesc* a, const char* role)
{
    if (a && role) for (int i=0;i<a->n_quant;i++)
        if (strcmp(a->quant[i].role, role)==0 && a->quant[i].output[0]) return a->quant[i].output;
    { const char* f = llmdesc_fmt(a, role); return f ? f : "f16"; }
    return NULL;
}

const char* llmdesc_source_mmproj(const LlmDesc* a) { return (a && a->source_mmproj[0]) ? a->source_mmproj : ""; }
const char* llmdesc_modality(const LlmDesc* a) {
    if (!a) return "text";
    int vision = 0, audio = 0;
    for (int i = 0; i < a->n_roles; i++) {
        const char* r = a->roles[i].role;
        if (strncmp(r, "AUDIO", 5) == 0) audio = 1;
        else if (strcmp(r, "VISION") == 0 || strcmp(r, "VISION_ALIGNER") == 0) vision = 1;
    }
    if (llmdesc_struct(a, "vision")) vision = 1;
    if (llmdesc_struct(a, "audio")) audio = 1;
    if (audio && vision) return "omni";
    if (audio) return "audio";
    if (vision) return (strstr(a->arch, "ocr") || strstr(a->arch, "paddle")) ? "ocr" : "vision";
    if (a->mm_mtp) return "mtp";
    if (a->mm_vision[0]) return "vision";
    return "text";
}
