/* llmcodec_rolemap.c — lettura minimale di `roles`/`scale_suffix`/`arch` da una `.llmarch`. */
#include "llmcodec_rolemap.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Trova il valore di "key" (dopo i due punti); ritorna NULL se assente. */
static const char* jval(const char* j, const char* key) {
    if (j == NULL || key == NULL) return NULL;
    size_t kl = strlen(key);
    for (const char* p = j; (p = strchr(p, '"')) != NULL; ++p) {
        const char* q = p + 1;
        if (strncmp(q, key, kl) == 0 && q[kl] == '"') {
            const char* c = q + kl + 1;
            while (*c == ' ' || *c == '\t' || *c == '\n' || *c == '\r') ++c;
            if (*c == ':') { ++c; while (*c == ' ' || *c == '\t' || *c == '\n' || *c == '\r') ++c; return c; }
        }
    }
    return NULL;
}

/* Copia una stringa JSON (tra virgolette) in out. Ritorna 1 se trovata. */
static int jstr(const char* v, char* out, size_t cap) {
    if (v == NULL || *v != '"') return 0;
    const char* p = v + 1; size_t n = 0;
    while (*p && *p != '"' && n + 1 < cap) out[n++] = *p++;
    out[n] = 0;
    return 1;
}

int llmcodec_rolemap_load(const char* llmarch_path, LlmCodecRoleMap* out) {
    if (llmarch_path == NULL || out == NULL) return -1;
    memset(out, 0, sizeof *out);
    snprintf(out->scale_suffix, sizeof out->scale_suffix, ".scales");
    snprintf(out->bias_suffix,  sizeof out->bias_suffix,  ".biases");
    out->expert_axis = 3;
    FILE* f = fopen(llmarch_path, "rb");
    if (f == NULL) return -1;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* j = (char*)malloc((size_t)n + 1);
    if (j == NULL) { fclose(f); return -1; }
    if (fread(j, 1, (size_t)n, f) != (size_t)n) { free(j); fclose(f); return -1; }
    fclose(f); j[n] = 0;

    jstr(jval(j, "arch"), out->arch, sizeof out->arch);
    jstr(jval(j, "scale_suffix"), out->scale_suffix, sizeof out->scale_suffix);
    jstr(jval(j, "bias_suffix"),  out->bias_suffix,  sizeof out->bias_suffix);

    /* quant: { "NORM":"f32", "OUTPUT":"q6_k", ..., "recipe":"_M", "recipe_boost":[...] } */
    { const char* q = jval(j, "quant");
      if (q != NULL && *q == '{') {
          const char* p = q + 1;
          while (*p && *p != '}' && out->n_q < 128) {
              p = strchr(p, '"'); if (p == NULL) break;
              char key[40]; size_t kn = 0; ++p;
              while (*p && *p != '"' && kn + 1 < sizeof key) key[kn++] = *p++;
              key[kn] = 0;
              p = strchr(p, ':'); if (p == NULL) break; ++p;
              while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
              if (*p == '[') { const char* e = strchr(p, ']'); if (e) p = e + 1; continue; }
              char val[24] = {0};
              if (jstr(p, val, sizeof val)) {
                  if (strncmp(key, "recipe", 6) != 0) {
                      snprintf(out->q[out->n_q].role, sizeof out->q[out->n_q].role, "%s", key);
                      snprintf(out->q[out->n_q].fmt, sizeof out->q[out->n_q].fmt, "%s", val);
                      out->n_q++;
                  }
                  const char* e = p + 1; while (*e && *e != '"') ++e; p = (*e == '"') ? e + 1 : e;
              } else { ++p; }
          }
      }
    }

    { const char* ct = jval(j, "convert_target");
      if (ct != NULL && *ct == '{') {
          int d0=0; const char* cend=ct; for(;*cend;++cend){ if(*cend=='{')d0++; else if(*cend=='}'){d0--; if(d0==0)break;} }
          const char* dv = jval(ct, "default");
          if (dv && *dv=='{') { int dd=0; const char* de=dv; for(;*de;++de){ if(*de=='{')dd++; else if(*de=='}'){dd--; if(dd==0)break;} }
              char obj[256]; size_t L=(size_t)(de-dv)+1; if(L<sizeof obj){ memcpy(obj,dv,L); obj[L]=0;
                  jstr(jval(obj,"codec"), out->ct_codec, sizeof out->ct_codec);
                  { const char* x=jval(obj,"fmt"); if(x) out->ct_fmt=atoi(x); x=jval(obj,"p0"); if(x) out->ct_p0=atoi(x); x=jval(obj,"p1"); if(x) out->ct_p1=atoi(x); } } }
          const char* rs = jval(ct, "roles");
          if (rs && *rs=='{') { int dd=0; const char* re=rs; for(;*re;++re){ if(*re=='{')dd++; else if(*re=='}'){dd--; if(dd==0)break;} }
              const char* q=rs+1;
              while (q<re && out->n_ctr<16) { q=strchr(q,'"'); if(!q||q>=re) break; char key[32]; size_t kn=0; ++q;
                  while(*q&&*q!='"'&&kn+1<sizeof key) key[kn++]=*q++; key[kn]=0;
                  q=strchr(q,':'); if(!q) break; ++q; while(*q==' '||*q=='\t'||*q=='\n'||*q=='\r') ++q;
                  if(*q!='{'){ if(*q=='"'){const char* e=strchr(q+1,'"'); q=e?e+1:q+1;} continue; }
                  int d2=0; const char* pe=q; for(;*pe;++pe){ if(*pe=='{')d2++; else if(*pe=='}'){d2--; if(d2==0)break;} }
                  char obj[256]; size_t L=(size_t)(pe-q)+1; if(L<sizeof obj){ memcpy(obj,q,L); obj[L]=0;
                      LlmCodecConvertRole* R=&out->ctr[out->n_ctr]; snprintf(R->role,sizeof R->role,"%s",key);
                      jstr(jval(obj,"codec"), R->codec, sizeof R->codec);
                      { const char* x=jval(obj,"fmt"); if(x) R->fmt=atoi(x); x=jval(obj,"p0"); if(x) R->p0=atoi(x); x=jval(obj,"p1"); if(x) R->p1=atoi(x); }
                      out->n_ctr++; }
                  q=pe+1; } }
      }
    }
    { const char* qp = jval(j, "quant_params");
      if (qp != NULL && *qp == '{') {
          int d0=0; const char* pend=qp;
          for (; *pend; ++pend) { if(*pend=='{')d0++; else if(*pend=='}'){d0--; if(d0==0)break;} }
          char obj[256]; size_t len=(size_t)(pend-qp)+1;
          if (len < sizeof obj) { memcpy(obj,qp,len); obj[len]=0;
              const char* g1=jval(obj,"group"); const char* g2=jval(obj,"mlx_group"); if(g1) out->q_group=atoi(g1); else if(g2) out->q_group=atoi(g2);
              const char* b1=jval(obj,"bits");  const char* b2=jval(obj,"mlx_bits");  if(b1) out->q_bits=atoi(b1);  else if(b2) out->q_bits=atoi(b2);
          }
      }
    }
    { const char* cd = jval(j, "codecs");
      if (cd != NULL && *cd == '{') {
          int d0=0; const char* cend=cd;
          for (; *cend; ++cend) { if(*cend=='{')d0++; else if(*cend=='}'){d0--; if(d0==0)break;} }
          const char* q = cd + 1;
          while (q < cend && out->n_codec < 16) {
              q = strchr(q, '"'); if (q == NULL || q >= cend) break;
              char key[24]; size_t kn=0; ++q;
              while (*q && *q != '"' && kn+1 < sizeof key) key[kn++]=*q++;
              key[kn]=0; q = strchr(q, ':'); if(q==NULL) break; ++q;
              while (*q==' '||*q=='\t'||*q=='\n'||*q=='\r') ++q;
              char val[32]={0};
              if (jstr(q, val, sizeof val)) {
                  snprintf(out->codec[out->n_codec].fmt, sizeof out->codec[out->n_codec].fmt, "%s", key);
                  snprintf(out->codec[out->n_codec].plugin, sizeof out->codec[out->n_codec].plugin, "%s", val);
                  out->n_codec++;
                  const char* e=q+1; while(*e && *e!='"') ++e; q = (*e=='"')?e+1:e;
              } else ++q;
          }
      }
    }
    { const char* wx = jval(j, "weight_source");
      if (wx != NULL && *wx == '{') {
          int d0=0; const char* wend=wx;
          for (; *wend; ++wend) { if(*wend=='{')d0++; else if(*wend=='}'){d0--; if(d0==0)break;} }
          const char* q = wx + 1;
          while (q < wend && out->n_ws < 8) {
              q = strchr(q, '"'); if (q == NULL || q >= wend) break;
              char name[24]; size_t kn=0; ++q;
              while (*q && *q != '"' && kn+1 < sizeof name) name[kn++]=*q++;
              name[kn]=0; q = strchr(q, ':'); if(q==NULL) break; ++q;
              while (*q==' '||*q=='\t'||*q=='\n'||*q=='\r') ++q;
              if (*q != '{') { if (*q=='"'){const char* e=strchr(q+1,'"'); q=e?e+1:q+1;} continue; }
              int dd=0; const char* p2=q;
              for (; *p2; ++p2) { if(*p2=='{')dd++; else if(*p2=='}'){dd--; if(dd==0)break;} }
              size_t len=(size_t)(p2-q)+1;
              if (len < 512) {
                  char obj[512]; memcpy(obj,q,len); obj[len]=0;
                  LlmCodecWeightSource* W=&out->ws[out->n_ws];
                  snprintf(W->name, sizeof W->name, "%s", name);
                  jstr(jval(obj,"dtype"), W->dtype, sizeof W->dtype);
                  jstr(jval(obj,"codec"), W->codec, sizeof W->codec);
                  jstr(jval(obj,"role"), W->role, sizeof W->role);
                  jstr(jval(obj,"mode"), W->mode, sizeof W->mode);
                  jstr(jval(obj,"global_suffix"), W->global_suffix, sizeof W->global_suffix);
                  jstr(jval(obj,"scale_suffix"), W->scale_suffix, sizeof W->scale_suffix);
                  jstr(jval(obj,"bias_suffix"),  W->bias_suffix,  sizeof W->bias_suffix);
                  const char* ea = jval(obj,"expert_axis"); if (ea) { int v=atoi(ea); if(v>0) W->expert_axis=v; }
                  { const char* x=jval(obj,"fmt"); if(x) W->fmt=atoi(x); x=jval(obj,"p0"); if(x) W->p0=atoi(x); x=jval(obj,"p1"); if(x) W->p1=atoi(x); }
                  out->n_ws++;
              }
              q = p2 + 1;
          }
      }
    }
    { const char* st = jval(j, "structures");
      if (st != NULL && *st == '{') {
          int d0 = 0; const char* send = st;
          for (; *send; ++send) { if (*send=='{') d0++; else if (*send=='}') { d0--; if (d0==0) break; } }
          const char* q = st + 1;
          while (q < send && out->n_st < 16) {
              q = strchr(q, '"'); if (q == NULL || q >= send) break;
              char key[32]; size_t kn = 0; ++q;
              while (*q && *q != '"' && kn + 1 < sizeof key) key[kn++] = *q++;
              key[kn] = 0;
              q = strchr(q, ':'); if (q == NULL) break; ++q;
              while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') ++q;
              if (*q != '{') { if (*q == '"') { const char* e=strchr(q+1,'"'); q = e?e+1:q+1; } continue; }
              /* oggetto struttura: trova la '}' di chiusura (contando le graffe) */
              int depth = 0; const char* p = q;
              for (; *p; ++p) { if (*p=='{') depth++; else if (*p=='}') { depth--; if (depth==0) break; } }
              size_t len = (size_t)(p - q) + 1;
              if (len < 2048) {
                  char obj[2048]; memcpy(obj, q, len); obj[len] = 0;
                  LlmCodecRoleStruct* S = &out->st[out->n_st];
                  snprintf(S->key, sizeof S->key, "%s", key);
                  jstr(jval(obj, "type"), S->type, sizeof S->type);
                  const char* ea = jval(obj, "expert_axis");
                  if (ea != NULL) { int v = atoi(ea); if (v > 0) out->expert_axis = v; }
                  if (strcmp(key, "moe") == 0) {
                      const char* ea2 = jval(obj, "expert_axis");
                      if (ea2 != NULL) { int v = atoi(ea2); if (v > 0) out->expert_axis = v; }
                  }
                  out->n_st++;
              }
              q = p + 1;
          }
      }
    }

    const char* r = jval(j, "roles");
    if (r != NULL && *r == '[') {
        const char* p = r + 1;
        while (*p && *p != ']' && out->n < 128) {
            p = strchr(p, '{');
            if (p == NULL) break;
            const char* end = strchr(p, '}');
            if (end == NULL) break;
            size_t len = (size_t)(end - p) + 1;
            char obj[1024];
            if (len < sizeof obj) {
                memcpy(obj, p, len); obj[len] = 0;
                LlmCodecRoleRule* R = &out->r[out->n];
                if (jstr(jval(obj, "match"), R->match, sizeof R->match)) {
                    jstr(jval(obj, "role"), R->role, sizeof R->role);
                    jstr(jval(obj, "submodel"), R->submodel, sizeof R->submodel);
                    R->gguf[0] = 0; { const char* gv = jval(obj, "gguf"); if (gv) jstr(gv, R->gguf, sizeof R->gguf); }
                    R->hf[0] = 0; { const char* hv = jval(obj, "hf"); if (hv) jstr(hv, R->hf, sizeof R->hf); }
                    R->stack = 0; { const char* sv = jval(obj, "stack"); if (sv && (strncmp(sv, "true", 4) == 0 || *sv == '1')) R->stack = 1; }
                    out->n++;
                }
            }
            p = end + 1;
        }
    }
    free(j);
    return out->n > 0 ? 0 : -1;
}

const char* llmcodec_rolemap_of(const LlmCodecRoleMap* m, const char* name) {
    if (m == NULL || name == NULL) return "OTHER";
    for (int i = 0; i < m->n; ++i) if (strstr(name, m->r[i].match) != NULL) return m->r[i].role;
    return "OTHER";
}

const char* llmcodec_rolemap_gguf(const LlmCodecRoleMap* m, const char* name) {
    if (m == NULL || name == NULL) return NULL;
    for (int i = 0; i < m->n; ++i) if (strstr(name, m->r[i].match) != NULL)
        return m->r[i].gguf[0] ? m->r[i].gguf : NULL;
    return NULL;
}

int llmcodec_rolemap_stack(const LlmCodecRoleMap* m, const char* name) {
    if (m == NULL || name == NULL) return 0;
    for (int i = 0; i < m->n; ++i) if (strstr(name, m->r[i].match) != NULL) return m->r[i].stack;
    return 0;
}

const char* llmcodec_rolemap_fmt(const LlmCodecRoleMap* m, const char* role) {
    if (m == NULL || role == NULL) return NULL;
    for (int i = 0; i < m->n_q; ++i) if (strcmp(m->q[i].role, role) == 0) return m->q[i].fmt;
    return NULL;
}

const char* llmcodec_rolemap_struct_type(const LlmCodecRoleMap* m, const char* key) {
    if (m == NULL || key == NULL) return NULL;
    for (int i = 0; i < m->n_st; ++i) if (strcmp(m->st[i].key, key) == 0) return m->st[i].type;
    return NULL;
}

int llmcodec_rolemap_expert_axis(const LlmCodecRoleMap* m) { return m ? m->expert_axis : 3; }

const LlmCodecWeightSource* llmcodec_rolemap_source_for_dtype(const LlmCodecRoleMap* m, const char* dtype) {
    if (m == NULL || dtype == NULL) return NULL;
    for (int i = 0; i < m->n_ws; ++i) if (strcmp(m->ws[i].dtype, dtype) == 0) return &m->ws[i];
    return NULL;
}

const LlmCodecWeightSource* llmcodec_rolemap_source_for(const LlmCodecRoleMap* m, const char* dtype, const char* role) {
    if (m == NULL || dtype == NULL) return NULL;
    const LlmCodecWeightSource* by_dtype = NULL;
    for (int i = 0; i < m->n_ws; ++i) {
        if (strcmp(m->ws[i].dtype, dtype) != 0) continue;
        if (role && m->ws[i].role[0] && strcmp(m->ws[i].role, role) == 0) return &m->ws[i];
        if (m->ws[i].role[0] == 0 && by_dtype == NULL) by_dtype = &m->ws[i];
    }
    return by_dtype;
}

int llmcodec_quantcfg_load(const char* config_path, LlmCodecQuantCfg* out) {
    if (config_path == NULL || out == NULL) return -1;
    memset(out, 0, sizeof *out); snprintf(out->mode, sizeof out->mode, "affine"); out->bits = 4; out->group = 64;
    FILE* f = fopen(config_path, "rb"); if (f == NULL) return -1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* j = (char*)malloc((size_t)sz + 1); if (j == NULL) { fclose(f); return -1; }
    if (fread(j, 1, (size_t)sz, f) != (size_t)sz) { free(j); fclose(f); return -1; }
    fclose(f); j[sz] = 0;
    const char* q = jval(j, "quantization_config");   /* chiave HF recente */
    if (q == NULL || *q != '{') q = jval(j, "quantization"); /* fallback/legacy */
    if (q && *q == '{') {
        int d0 = 0; const char* qe = q;
        for (; *qe; ++qe) { if (*qe=='{') d0++; else if (*qe=='}') { d0--; if (d0==0) break; } }
        { const char* m = jval(q, "mode"); if (m) jstr(m, out->mode, sizeof out->mode);
          const char* b = jval(q, "bits"); if (b) out->bits = atoi(b);
          const char* g = jval(q, "group_size"); if (g) out->group = atoi(g); }
        const char* p = q + 1;
        while (p < qe && out->n < 1024) {
            p = strchr(p, '"'); if (p == NULL || p >= qe) break;
            char key[160]; size_t kn = 0; ++p;
            while (*p && *p != '"' && kn + 1 < sizeof key) key[kn++] = *p++;
            key[kn] = 0;
            p = strchr(p, ':'); if (p == NULL) break; ++p;
            while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
            if (*p != '{') { if (*p == '"') { const char* e = strchr(p + 1, '"'); p = e ? e + 1 : p + 1; } else ++p; continue; }
            int d2 = 0; const char* pe = p;
            for (; *pe; ++pe) { if (*pe == '{') d2++; else if (*pe == '}') { d2--; if (d2==0) break; } }
            size_t L = (size_t)(pe - p) + 1;
            if (L < 512) {
                char obj[512]; memcpy(obj, p, L); obj[L] = 0;
                LlmCodecQuantEntry* E = &out->e[out->n];
                snprintf(E->name, sizeof E->name, "%s", key);
                E->bits = out->bits; E->group = out->group; snprintf(E->mode, sizeof E->mode, "%s", out->mode);
                const char* m = jval(obj, "mode"); if (m) jstr(m, E->mode, sizeof E->mode);
                const char* b = jval(obj, "bits"); if (b) E->bits = atoi(b);
                const char* g = jval(obj, "group_size"); if (g) E->group = atoi(g);
                out->n++;
            }
            p = pe + 1;
        }
    }
    free(j);
    return 0;
}

static const char* const g_supported_methods[] = {
    "compressed-tensors", "fp8", "gptq", "awq", "bitnet", "mxfp4", "mxfp8", "nvfp4", "turboquant", "bitsandbytes", "hqq", "quanto", "torchao", "autoround", "quark", "aqlm", "vptq", "spqr", "higgs", NULL };
static const char* const g_known_unsupported[] = {
    "fbgemm",
    "sinq", "fouroversix", "fp-quant", "eetq", "qqq", NULL };

const char* llmcodec_quant_method(const char* config_path, int* out_supported) {
    static char m[48];
    if (out_supported) *out_supported = 0;
    if (config_path == NULL) return NULL;
    FILE* f = fopen(config_path, "rb"); if (f == NULL) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* j = (char*)malloc((size_t)sz + 1); if (j == NULL) { fclose(f); return NULL; }
    if (fread(j, 1, (size_t)sz, f) != (size_t)sz) { free(j); fclose(f); return NULL; }
    fclose(f); j[sz] = 0;
    const char* qc = jval(j, "quantization_config"); if (qc == NULL) qc = j;
    const char* mm = jval(qc, "quant_method"); if (mm == NULL) mm = jval(qc, "mode");
    m[0] = 0; if (mm) jstr(mm, m, sizeof m);
    if (m[0] == 0) { free(j); return NULL; }
    for (int i = 0; g_supported_methods[i]; ++i) if (strcasecmp(m, g_supported_methods[i]) == 0) { if (out_supported) *out_supported = 1; break; }
    free(j);
    return m;
}

int llmcodec_ct_load(const char* config_path, LlmCodecQuantCfg* out) {
    if (config_path == NULL || out == NULL) return -1;
    memset(out, 0, sizeof *out); snprintf(out->mode, sizeof out->mode, "affine"); out->bits = 4; out->group = 128;
    FILE* f = fopen(config_path, "rb"); if (f == NULL) return -1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* j = (char*)malloc((size_t)sz + 1); if (j == NULL) { fclose(f); return -1; }
    if (fread(j, 1, (size_t)sz, f) != (size_t)sz) { free(j); fclose(f); return -1; }
    fclose(f); j[sz] = 0;
    const char* qc = jval(j, "quantization_config");
    if (qc == NULL) { free(j); return -1; }
    { const char* qm = jval(qc, "quant_method"); char qmb[32] = {0}; if (qm) jstr(qm, qmb, sizeof qmb);
      if (strcmp(qmb, "bitsandbytes") == 0) {   /* HF bitsandbytes */
          char qt[16] = "nf4"; const char* qtv = jval(qc, "bnb_4bit_quant_type"); if (qtv) jstr(qtv, qt, sizeof qt);
          snprintf(out->mode, sizeof out->mode, "%s", strcmp(qt, "fp4") == 0 ? "bnb_fp4" : "bnb_nf4");
          const char* bs = jval(qc, "bnb_4bit_blocksize"); if (bs) out->group = atoi(bs);
          out->bits = 4; free(j); return 0; }
      if (qmb[0] && strcmp(qmb, "compressed-tensors") != 0) { free(j); return -1; } }
    /* format -> mode */
    { char fmt[48] = {0}; const char* fv = jval(qc, "format"); if (fv) jstr(fv, fmt, sizeof fmt);
      if      (strstr(fmt, "nvfp4")) snprintf(out->mode, sizeof out->mode, "nvfp4");
      else if (strstr(fmt, "mxfp8")) snprintf(out->mode, sizeof out->mode, "mxfp8");
      else if (strstr(fmt, "mxfp4")) snprintf(out->mode, sizeof out->mode, "mxfp4");
      else if (strstr(fmt, "float")) snprintf(out->mode, sizeof out->mode, "mxfp8");
      else if (strstr(fmt, "int") || strstr(fmt, "pack")) snprintf(out->mode, sizeof out->mode, "affine"); }
    /* config_groups.<g>.weights {num_bits, group_size, symmetric} */
    { const char* cg = jval(qc, "config_groups");
      if (cg) { const char* w = jval(cg, "weights");
          if (w) { const char* nb = jval(w, "num_bits"); if (nb) out->bits = atoi(nb);
                   const char* gs = jval(w, "group_size"); if (gs) out->group = atoi(gs);
                   const char* sy = jval(w, "symmetric"); if (sy && strncmp(sy, "false", 5) == 0) out->group = out->group; } } }
    free(j);
    return 0;
}

const LlmCodecQuantEntry* llmcodec_quantcfg_find(const LlmCodecQuantCfg* c, const char* name) {
    if (c == NULL || name == NULL) return NULL;
    for (int i = 0; i < c->n; ++i) if (strcmp(c->e[i].name, name) == 0) return &c->e[i];
    return NULL;
}

int llmcodec_rolemap_group(const LlmCodecRoleMap* m) { return m ? m->q_group : 0; }
int llmcodec_rolemap_bits(const LlmCodecRoleMap* m)  { return m ? m->q_bits  : 0; }

const char* llmcodec_rolemap_convert(const LlmCodecRoleMap* m, const char* role, int* fmt, int* p0, int* p1) {
    if (m == NULL) return NULL;
    if (role) for (int i=0;i<m->n_ctr;i++) if (strcmp(m->ctr[i].role, role)==0) {
        if (fmt) *fmt = m->ctr[i].fmt; if (p0) *p0 = m->ctr[i].p0; if (p1) *p1 = m->ctr[i].p1;
        return m->ctr[i].codec[0] ? m->ctr[i].codec : NULL;
    }
    if (m->ct_codec[0]) { if (fmt) *fmt = m->ct_fmt; if (p0) *p0 = m->ct_p0; if (p1) *p1 = m->ct_p1; return m->ct_codec; }
    return NULL;
}

const LlmCodecWeightSource* llmcodec_rolemap_source_mode(const LlmCodecRoleMap* m, const char* dtype, const char* mode) {
    if (m == NULL || dtype == NULL) return NULL;
    const LlmCodecWeightSource* nomode = NULL;
    for (int i = 0; i < m->n_ws; ++i) {
        if (strcmp(m->ws[i].dtype, dtype) != 0) continue;
        if (mode && mode[0] && m->ws[i].mode[0] && strcmp(m->ws[i].mode, mode) == 0) return &m->ws[i];
        if (m->ws[i].mode[0] == 0 && nomode == NULL) nomode = &m->ws[i];
    }
    return nomode;
}

const char* llmcodec_rolemap_codec_plugin(const LlmCodecRoleMap* m, const char* fmt) {
    if (m == NULL || fmt == NULL) return NULL;
    for (int i = 0; i < m->n_codec; ++i) if (strcmp(m->codec[i].fmt, fmt) == 0) return m->codec[i].plugin;
    return NULL;
}

int llmcodec_rolemap_unsupported(const LlmCodecRoleMap* m, char* out, size_t cap) {
    /* Il CONVERTER (formato/ruolo) gestisce queste strutture via i ruoli .llmarch + la ricetta
     * (`weight_source`): engram->fp8 row (fmt 2), compression/indexer->fp8, vision->plain/fp8.
     * Il COMPUTE (csa/indexer/engram/vit) resta dell'engine. Vedi Converter-DeepSeek-V41-Strutture. */
    static const struct { const char* key; const char* type; } supported[] = {
        { "attention", "mla" }, { "attention", "gqa" }, { "moe", NULL }, { "norm", NULL },
        { "compression", "csa" }, { "indexer", "indexer" }, { "engram", "engram" }, { "vision", "vit" },
    };
    int nbad = 0; size_t o = 0;
    if (m == NULL) { if (cap) out[0] = 0; return 0; }
    o += (size_t)snprintf(out + o, cap - o, "llmarch '%s':", m->arch);
    for (int i = 0; i < m->n_st && o < cap; ++i) {
        int ok = 0;
        for (size_t k = 0; k < sizeof(supported)/sizeof(supported[0]); ++k)
            if (strcmp(supported[k].key, m->st[i].key) == 0 &&
                (supported[k].type == NULL || strcmp(supported[k].type, m->st[i].type) == 0)) { ok = 1; break; }
        o += (size_t)snprintf(out + o, cap - o, " %s=%s%s", m->st[i].key, m->st[i].type, ok ? "" : "(NON supportata)");
        if (!ok) nbad++;
    }
    if (o < cap) snprintf(out + o, cap - o, " -> %d non supportata/e", nbad);
    return nbad;
}

int llmcodec_rolemap_is_expert(const char* role) {
    return role != NULL && (strstr(role, "EXPERT") != NULL || strstr(role, "expert") != NULL);
}

/* ── Reverse GGUF(canonico) -> HF via i template `hf` ────────────────────────────────
 * Mini pattern-matcher: `%d` = cifre (layer), `%s` = alfanumerico (gate/up/down). Estrae i
 * valori posizionali dal template `gguf` combaciante e li riapplica al template `hf`. */
static int rp_pat_match(const char* pat, const char* s, char vals[][64], int maxv) {
    int nv = 0;
    while (*pat) {
        if (pat[0] == '%' && (pat[1] == 'd' || pat[1] == 's')) {
            pat += 2;
            const char* lit = pat; size_t ll = 0;
            while (lit[ll] && !(lit[ll] == '%' && (lit[ll+1] == 'd' || lit[ll+1] == 's'))) ll++;
            /* Ricerca BOUNDED di `ll` caratteri: `lit` punta dentro il pattern e NON è
             * terminato a ll (usare strstr cercava il literal INTERO col successivo %s -> mai
             * trovato sui pattern multi-placeholder, es. blk.%d.ffn_%s_exps.weight). */
            const char* found = NULL;
            if (ll == 0) found = s + strlen(s);
            else { size_t sl = strlen(s); for (size_t k = 0; k + ll <= sl; ++k) if (memcmp(s + k, lit, ll) == 0) { found = s + k; break; } }
            if (!found) return -1;
            size_t vl = (size_t)(found - s);
            if (vl > 63) return -1;
            if (nv < maxv) { memcpy(vals[nv], s, vl); vals[nv][vl] = 0; }
            nv++;
            s = found;
        } else {
            if (*pat != *s) return -1;
            pat++; s++;
        }
    }
    return *s ? -1 : nv;
}
int llmcodec_rolemap_canonical_to_hf(const LlmCodecRoleMap* m, const char* canon, char* out, size_t cap) {
    if (m == NULL || canon == NULL || out == NULL || cap == 0) return 0;
    out[0] = 0;
    for (int i = 0; i < m->n; ++i) {
        if (!m->r[i].gguf[0] || !m->r[i].hf[0]) continue;
        char vals[8][64];
        int nv = rp_pat_match(m->r[i].gguf, canon, vals, 8);
        if (nv < 0) continue;
        size_t o = 0; int vi = 0;
        for (const char* p = m->r[i].hf; *p && o < cap - 1; ) {
            if (p[0] == '%' && (p[1] == 'd' || p[1] == 's')) {
                const char* v = (vi < nv) ? vals[vi] : "";
                o += (size_t)snprintf(out + o, cap - o, "%s", v);
                vi++; p += 2;
            } else out[o++] = *p++;
        }
        out[o < cap ? o : cap - 1] = 0;
        return 1;
    }
    return 0;
}
