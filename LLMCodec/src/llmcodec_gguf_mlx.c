/* llmcodec_gguf_mlx.c — conversioni GGML ↔ MLX affine. Vedi include/llmcodec_gguf_mlx.h. */
#include "llmcodec_gguf_mlx.h"
#include "llmcodec_rolemap.h"
#include "llmcodec_gguf_meta.h"
#include "llmcodec_fp8_fp4.h"
#include "llmcodec_fp16.h"   /* llmcodec_f16_to_f32/llmcodec_bf16_to_f32 (upcast norm->F32 nel fallback) */
#include "llmcodec.h"
#include <strings.h>
#include "llmcodec_ggml_quants.h"
#include "llmcodec_moe_quant.h"
#include "llmcodec_gguf_writer.h"
#include "llmcodec_safetensors.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

static uint8_t* llmcodec_slurp_range(const char* path, uint64_t off, uint64_t len) {
    FILE* f = fopen(path, "rb"); if (f == NULL) return NULL;
    if (fseeko(f, (off_t)off, SEEK_SET) != 0) { fclose(f); return NULL; }
    uint8_t* b = (uint8_t*)malloc(len ? (size_t)len : 1);
    if (b == NULL) { fclose(f); return NULL; }
    if (len && fread(b, 1, (size_t)len, f) != len) { free(b); fclose(f); return NULL; }
    fclose(f); return b;
}
static int llmcodec_ggml_type_of(const char* dt) {
    if (strcmp(dt, "F32") == 0) return LLMCODEC_GGML_F32;
    if (strcmp(dt, "F16") == 0) return LLMCODEC_GGML_F16;
    if (strcmp(dt, "BF16") == 0) return 30;
    if (strcmp(dt, "F64") == 0) return 28;
    if (strcmp(dt, "I32") == 0) return 26;
    if (strcmp(dt, "I16") == 0) return 25;
    if (strcmp(dt, "I8") == 0)  return 24;
    if (strcmp(dt, "U8") == 0)  return 24;
    return -1;
}
static int llmcodec_dtype_size(const char* dt) {
    if (strcmp(dt, "F32") == 0 || strcmp(dt, "I32") == 0) return 4;
    if (strcmp(dt, "F64") == 0) return 8;
    if (strcmp(dt, "I16") == 0 || strcmp(dt, "F16") == 0 || strcmp(dt, "BF16") == 0) return 2;
    return 1;
}

int llmcodec_conv_ggml_to_mlx(int ggml_type, const void* src, uint32_t rows, uint32_t cols,
                         uint32_t group, uint32_t bits,
                         uint32_t* w_out, uint16_t* sc_out, uint16_t* bs_out) {
    if (src == NULL || w_out == NULL || sc_out == NULL || bs_out == NULL || rows == 0 || cols == 0) return -1;
    const LlmCodec* gc = llmcodec_resolve("ggml");
    const LlmCodec* af = llmcodec_resolve("mlx_affine");
    if (gc == NULL || af == NULL) return -1;
    size_t ob = 0, ab = 0;
    if (af->sizes(0, (int)group, (int)bits, rows, cols, &ob, &ab) != 0) return -1;
    float* f = (float*)malloc((size_t)rows*cols*4);
    uint8_t* aux = (uint8_t*)malloc(ab);
    if (f == NULL || aux == NULL) { free(f); free(aux); return -1; }
    int r = gc->decode(src, 0, ggml_type, 0, 0, rows, cols, NULL, 0, f);
    if (r == 0) { size_t ol = 0, al = 0;
        r = af->encode(f, rows, cols, 0, (int)group, (int)bits, w_out, ob, &ol, aux, ab, &al); }
    if (r == 0) { size_t scb = ab/2; memcpy(sc_out, aux, scb); memcpy(bs_out, aux + scb, scb); }
    free(f); free(aux);
    return r;
}

int llmcodec_conv_mlx_to_ggml(const uint32_t* w, const uint16_t* sc, const uint16_t* bs,
                         uint32_t rows, uint32_t cols, uint32_t group, uint32_t bits,
                         int ggml_type, void* dst) {
    if (w == NULL || sc == NULL || bs == NULL || dst == NULL || rows == 0 || cols == 0) return -1;
    const LlmCodec* af = llmcodec_resolve("mlx_affine");
    const LlmCodec* gc = llmcodec_resolve("ggml");
    if (af == NULL || gc == NULL) return -1;
    size_t ob = 0, ab = 0;
    if (af->sizes(0, (int)group, (int)bits, rows, cols, &ob, &ab) != 0) return -1;
    size_t scb = ab / 2;
    uint8_t* aux = (uint8_t*)malloc(ab);
    float* f = (float*)malloc((size_t)rows*cols*4);
    if (aux == NULL || f == NULL) { free(aux); free(f); return -1; }
    memcpy(aux, sc, scb); memcpy(aux + scb, bs, scb);
    int r = af->decode(w, ob, 0, (int)group, (int)bits, rows, cols, aux, ab, f);
    if (r == 0) { size_t ol = 0, al = 0;
        r = gc->encode(f, rows, cols, ggml_type, 0, 0, dst, (size_t)1<<60, &ol, NULL, 0, &al); }
    free(aux); free(f);
    return r;
}


/* Elenca i file shard da `model.safetensors.index.json` (o il singolo `model.safetensors`). */

/* Decodifica una sorgente nel suo formato dichiarato (dal registry) -> f32 [E,rows,cols].
 * Nessun elenco di codec cablato: family(aux_kind/shape_of/sizes/decode) guida tutto. */
static int llmcodec_decode_src(const char* sp, uint64_t base, const uint8_t* raw, uint64_t rawlen, LlmCodecStTensorInfo* ti, int n, LlmCodecStTensorInfo* t,
                          const LlmCodecWeightSource* ws, const LlmCodecRoleMap* rm, uint32_t dg, uint32_t db,
                          float** f_out, size_t* rows_out, size_t* cols_out, int* e_out) {
    const char* cname = ws ? ws->codec : (strcmp(t->dtype, "U32") == 0 ? "mlx_affine" : NULL);
    if (cname == NULL || cname[0] == 0) return -1;
    const LlmCodec* sc = llmcodec_resolve(cname);
    if (sc == NULL || sc->decode == NULL || sc->shape_of == NULL || sc->sizes == NULL) return -1;
    int sfmt = ws ? ws->fmt : 0;
    int p0 = (ws && ws->p0) ? ws->p0 : (dg ? (int)dg : 0);
    int p1 = (ws && ws->p1) ? ws->p1 : (db ? (int)db : 0);
    int E = (t->ndim == 3) ? (int)t->shape[0] : 1;
    size_t d0 = (t->ndim == 3) ? (size_t)t->shape[1] : (size_t)t->shape[0];
    size_t d1 = (t->ndim == 3) ? (size_t)t->shape[2] : (size_t)t->shape[1];
    size_t rows = 0, cols = 0;
    if (sc->shape_of(sfmt, p0, p1, d0, d1, &rows, &cols) != 0 || rows == 0 || cols == 0) return -1;
    size_t ob = 0, ab = 0;
    if (sc->sizes(sfmt, p0, p1, rows, cols, &ob, &ab) != 0) return -1;
    uint8_t* aux = NULL; size_t auxlen = 0;
    if (sc->aux_kind != LLMCODEC_CODEC_AUX_NONE) {
        char bname[280]; snprintf(bname,sizeof bname,"%s",t->name); { char* wp=strstr(bname,".weight"); if(wp&&wp[7]==0)*wp=0; }
        if (sc->aux_kind == LLMCODEC_CODEC_AUX_SCALE) {
            const char* su=(ws&&ws->scale_suffix[0])?ws->scale_suffix:(rm&&rm->scale_suffix[0]?rm->scale_suffix:".scale");
            char scn[300]; snprintf(scn,sizeof scn,"%s%s",bname,su);
            for (int k=0;k<n;k++) if(strcmp(ti[k].name,scn)==0){ auxlen=(size_t)(ti[k].end-ti[k].begin); aux=llmcodec_slurp_range(sp,base+ti[k].begin,auxlen); break; }
            /* 2o livello opzionale (NVFP4 OCP/NVIDIA): global f32 in coda */
            if (aux && ws && ws->global_suffix[0]) {
                char gsn[300]; snprintf(gsn,sizeof gsn,"%s%s",bname,ws->global_suffix);
                for (int k=0;k<n;k++) if(strcmp(ti[k].name,gsn)==0){
                    uint8_t* g=(uint8_t*)llmcodec_slurp_range(sp,base+ti[k].begin,ti[k].end-ti[k].begin);
                    if (g) { float gv=0.0f; memcpy(&gv,g,4); free(g);
                             uint8_t* na=(uint8_t*)realloc(aux,auxlen+4); if(na){ memcpy(na+auxlen,&gv,4); aux=na; auxlen+=4; } }
                    break; }
            }
        } else {
            const char* su=(ws&&ws->scale_suffix[0])?ws->scale_suffix:(rm&&rm->scale_suffix[0]?rm->scale_suffix:".scales");
            const char* bu=(ws&&ws->bias_suffix[0])?ws->bias_suffix:(rm&&rm->bias_suffix[0]?rm->bias_suffix:".biases");
            char scn[300],bsn[300]; snprintf(scn,sizeof scn,"%s%s",bname,su); snprintf(bsn,sizeof bsn,"%s%s",bname,bu);
            int si=-1,bi=-1; for(int k=0;k<n;k++){ if(strcmp(ti[k].name,scn)==0)si=k; if(strcmp(ti[k].name,bsn)==0)bi=k; }
            if (si>=0 && bi>=0) {
                size_t s1=(size_t)(ti[si].end-ti[si].begin), s2=(size_t)(ti[bi].end-ti[bi].begin);
                uint8_t* p1r=llmcodec_slurp_range(sp,base+ti[si].begin,s1); uint8_t* p2r=llmcodec_slurp_range(sp,base+ti[bi].begin,s2);
                if (p1r && p2r) {
                    /* interleave PER-SLAB: scales[e] ++ biases[e] (l'aux e' per-esperto).
                     * I byte PER ESPERTO si ricavano dal numero di esperti E, non da `ab/2`:
                     * per i tensori esperti 3D `ab` non e' 2*scb e la vecchia formula produceva
                     * un memcpy gigantesco fuori dal buffer (heap-buffer-overflow, trovato con
                     * ASan su Qwen3.6-35B-A3B affine). */
                    size_t scb = (E > 0) ? s1 / (size_t)E : s1; /* byte di scales per esperto */
                    if (scb == 0) scb = s1;
                    ab = 2 * scb;                               /* aux per esperto = scales+biases */
                    aux=(uint8_t*)malloc((size_t)E * ab); auxlen=(size_t)E * ab;
                    for (int e=0; e<E; e++) {
                        size_t off=(size_t)e*scb;
                        size_t c1 = (off < s1) ? (scb <= s1-off ? scb : s1-off) : 0;
                        size_t c2 = (off < s2) ? (scb <= s2-off ? scb : s2-off) : 0;
                        if (c1) memcpy(aux + (size_t)e*ab,          p1r + off, c1);
                        if (c2) memcpy(aux + (size_t)e*ab + scb,    p2r + off, c2);
                    }
                }
                free(p1r); free(p2r);
            }
        }
        if (aux == NULL) return -1;
    }
    float* f=(float*)malloc((size_t)E*rows*cols*4);
    if (f == NULL) { free(aux); return -1; }
    for (int e=0;e<E;e++) {
        const uint8_t* ip = raw + (uint64_t)e*ob;
        const uint8_t* ap = aux ? aux + (uint64_t)e*ab : NULL;
        if (sc->decode(ip, ob, sfmt, p0, p1, rows, cols, ap, ab, f + (size_t)e*rows*cols) != 0) { free(f); free(aux); return -1; }
    }
    free(aux);
    *f_out=f; *rows_out=rows; *cols_out=cols; *e_out=E; return 0;
}

static int llmcodec_list_shards(const char* dir, char names[][256], int max) {
    char ip[1200]; snprintf(ip, sizeof ip, "%s/model.safetensors.index.json", dir);
    FILE* f = fopen(ip, "rb");
    if (f == NULL) { snprintf(names[0], 256, "model.safetensors"); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char* j = (char*)malloc((size_t)n + 1);
    if (j == NULL) { fclose(f); snprintf(names[0], 256, "model.safetensors"); return 1; }
    if (fread(j, 1, (size_t)n, f) != (size_t)n) { free(j); fclose(f); return -1; }
    fclose(f); j[n] = 0;
    int cnt = 0;
    const char* suffix = ".safetensors";
    for (const char* q = j; *q; ++q) {
        if (strncmp(q, suffix, strlen(suffix)) == 0) {
            const char* b = q;               /* fine nome; torna indietro all'apertura '"' */
            const char* start = q; while (start > j && start[-1] != '"') --start;
            size_t len = (size_t)((b + strlen(suffix)) - start);
            if (len < 256) {
                char nm[256]; memcpy(nm, start, len); nm[len] = 0;
                int dup = 0; for (int i = 0; i < cnt; ++i) if (strcmp(names[i], nm) == 0) { dup = 1; break; }
                if (!dup && cnt < max) { snprintf(names[cnt++], 256, "%s", nm); }
            }
        }
    }
    free(j);
    return cnt > 0 ? cnt : -1;
}

static int src_layer_of(const char* name) {
    const char* p = strstr(name, "layers."); if (p) return atoi(p + 7);
    p = strstr(name, "mtp."); if (p) return atoi(p + 4);
    p = strstr(name, "blk."); if (p) return atoi(p + 4);
    return -1;
}
static int src_expert_of(const char* name) {
    const char* p = strstr(name, ".experts."); if (p == NULL) return -1;
    return atoi(p + 9);
}
/* Nome canonico GGUF del tensore: se la regola ha `gguf`, sostituisce %d con l'indice layer;
 * altrimenti conserva il nome sorgente senza `.weight` (comportamento storico). */
/* Sotto-ruolo FFN dal nome sorgente (w1/GATE, w3/UP, w2/DOWN) per il placeholder `%s`. */
static const char* src_subrole(const char* name) {
    if (strstr(name, ".w1.") || strstr(name, "gate_proj")) return "gate";
    if (strstr(name, ".w3.") || strstr(name, "up_proj"))   return "up";
    if (strstr(name, ".w2.") || strstr(name, "down_proj")) return "down";
    return "";
}
static void gguf_target_name(const LlmCodecRoleMap* rm, const char* src, char* out, size_t cap) {
    const char* g = rm ? llmcodec_rolemap_gguf(rm, src) : NULL;
    if (g != NULL) {
        int hd = (strstr(g, "%d") != NULL), hs = (strstr(g, "%s") != NULL);
        if (hd && hs)      { int L = src_layer_of(src); snprintf(out, cap, g, L, src_subrole(src)); }
        else if (hd)       { int L = src_layer_of(src); snprintf(out, cap, g, L); }
        else if (hs)       { snprintf(out, cap, g, src_subrole(src)); }
        else               { snprintf(out, cap, "%s", g); }
        return;
    }
    snprintf(out, cap, "%s", src);   /* nessun mapping: nome sorgente VERBATIM (trovabile dal loader) */
}
/* Accumulatore per lo STACKING 3D degli esperti per-esperto -> un tensore [cols,rows,E]. */
typedef struct { char target[288]; uint32_t cols, rows; int expert, type; uint8_t* data; size_t n; } PendExp;

int llmcodec_mlx_to_gguf_ex(const char* mlx_dir, const char* out_path, int ggml_type, uint32_t group, uint32_t bits, const LlmCodecRoleMap* rm, const LlmCodecTokenizerMeta* tm) {
    if (group == 0 && rm && llmcodec_rolemap_group(rm) > 0) group = (uint32_t)llmcodec_rolemap_group(rm);
    if (bits  == 0 && rm && llmcodec_rolemap_bits(rm)  > 0) bits  = (uint32_t)llmcodec_rolemap_bits(rm);
    if (group == 0) group = 32; if (bits == 0) bits = 4;
    char shard[64][256]; int nsh = llmcodec_list_shards(mlx_dir, shard, 64);
    if (nsh <= 0) return -1;
    LlmCodecQuantCfg qcfg; { char cjp[1200]; snprintf(cjp, sizeof cjp, "%s/config.json", mlx_dir);
                        { int sup=0; const char* qm=llmcodec_quant_method(cjp,&sup);
                          if (qm && !sup) { int known=0; const char* ku[]={"bitsandbytes","hqq","aqlm","vptq","spqr","higgs","quanto","torchao","fbgemm","sinq","autoround","quark","foursoversix","fp-quant",NULL};
                            for(int i=0;ku[i];i++) if(strcasecmp(qm,ku[i])==0) known=1;
                            fprintf(stderr,"[mlx->gguf] ATTENZIONE: quant_method '%s' %s\n", qm, known?"riconosciuto ma NON supportato (P1)":"non riconosciuto"); } }
                        if (llmcodec_quantcfg_load(cjp, &qcfg) != 0) { if (llmcodec_ct_load(cjp, &qcfg) != 0) { memset(&qcfg,0,sizeof qcfg); snprintf(qcfg.mode,sizeof qcfg.mode,"affine"); qcfg.bits=(int)bits; qcfg.group=(int)group; } } }
    if (rm) { char ub[512]; if (llmcodec_rolemap_unsupported(rm, ub, sizeof ub) > 0)
                  fprintf(stderr, "[mlx->gguf] ATTENZIONE, strutture non mappate dal converter: %s\n", ub); }
    LlmCodecGgufTensor* ts = NULL; uint8_t** bufs = NULL; uint64_t** dmv = NULL; int nt = 0, cap = 0, nskip = 0;
    PendExp* pend = NULL; int np = 0, pcap = 0;
    /* Slice opzionale dei layer (test): LLMCODEC_GGUF_LAYERS=l0:l1 -> salta i tensori dei layer fuori
     * finestra (i tensori globali, senza "layers.", restano). -1 = nessuna slice. */
    int only_l0 = -1, only_l1 = -1;
    { const char* e = getenv("LLMCODEC_GGUF_LAYERS"); if (e && *e) { only_l0 = atoi(e); const char* c = strchr(e, ':'); only_l1 = c ? atoi(c + 1) : only_l0 + 1; } }
    for (int sh = 0; sh < nsh; ++sh) {
        char sp[1500]; snprintf(sp, sizeof sp, "%s/%s", mlx_dir, shard[sh]);
        LlmCodecStTensorInfo* ti = NULL; int n = 0; uint64_t base = 0;
        if (llmcodec_safetensors_read(sp, &ti, &n, &base) != 0) continue;
        for (int i = 0; i < n; ++i) {
            LlmCodecStTensorInfo* t = &ti[i];
            size_t ln = strlen(t->name);
            if ((ln > 7 && strcmp(t->name+ln-7, ".scales") == 0) || (ln > 7 && strcmp(t->name+ln-7, ".biases") == 0)) continue;
            if (rm) { size_t sl = strlen(rm->scale_suffix), bl = strlen(rm->bias_suffix);
                      for (int w2 = 0; w2 < rm->n_ws; ++w2) { const char* ws2 = rm->ws[w2].scale_suffix; size_t wl = strlen(ws2);
                          if (wl && ln > wl && strcmp(t->name+ln-wl, ws2) == 0) goto next_tensor; }
                      if ((sl && ln > sl && strcmp(t->name+ln-sl, rm->scale_suffix) == 0) ||
                          (bl && ln > bl && strcmp(t->name+ln-bl, rm->bias_suffix) == 0)) continue; }
            int stack = rm ? llmcodec_rolemap_stack(rm, t->name) : 0;
            if (stack && getenv("LLMCODEC_GGUF_NO_EXPERTS")) continue;   /* check rapido dei nomi (no encode esperti) */
            /* GGUF del modello MAIN: i sotto-modelli MTP/Vision non appartengono (artefatti separati). */
            if (strncmp(t->name,"mtp.",4)==0 || strncmp(t->name,"vision.",7)==0 ||
                strncmp(t->name,"aligner.",8)==0 || strncmp(t->name,"image_",6)==0) continue;
            if (only_l1 >= 0 && strstr(t->name, "layers.") != NULL) { int L = src_layer_of(t->name); if (L < only_l0 || L >= only_l1) continue; }
            uint8_t* raw = llmcodec_slurp_range(sp, base + t->begin, t->end - t->begin);
            if (raw == NULL) continue;
            uint64_t* dm = (uint64_t*)calloc(8, sizeof(uint64_t));
            char gbuf[288]; gguf_target_name(rm, t->name, gbuf, sizeof gbuf);
            const char* gname = gbuf;
            uint8_t* out = NULL; int gtype = 0;
            char modn[280]; snprintf(modn,sizeof modn,"%s",t->name); { char* w=strstr(modn,".weight"); if(w&&w[7]==0)*w=0; }
            const LlmCodecQuantEntry* qe = llmcodec_quantcfg_find(&qcfg, modn);
            const char* tmode = qe ? qe->mode : qcfg.mode;
            int tbits = qe ? qe->bits : qcfg.bits, tgroup = qe ? qe->group : qcfg.group;
            const LlmCodecWeightSource* ws = rm ? llmcodec_rolemap_source_mode(rm, t->dtype, tmode) : NULL;
            if (ws == NULL && rm) ws = llmcodec_rolemap_source_for(rm, t->dtype, llmcodec_rolemap_of(rm, t->name));
            float* f32 = NULL; size_t rows = 0, cols = 0; int E = 1;
            if (llmcodec_decode_src(sp, base, raw, (uint64_t)(t->end-t->begin), ti, n, t, ws, rm, (uint32_t)tgroup, (uint32_t)tbits, &f32, &rows, &cols, &E) == 0) {
                int tt = ggml_type;
                { const char* fmt=llmcodec_rolemap_fmt(rm, llmcodec_rolemap_of(rm,t->name)); if(fmt){int z=llmcodec_ggml_type_of_fmt(fmt); if(z>=0)tt=z;} }
                const LlmCodec* gc = llmcodec_resolve("ggml");
                uint64_t slab_nb=0, aux_nb=0; uint8_t* obuf=NULL; int cerr=1;
                if (gc && gc->sizes(tt,0,0,rows,cols,&slab_nb,&aux_nb)==0) {
                    obuf = (uint8_t*)malloc((size_t)slab_nb*(size_t)E);
                    if (obuf) { cerr=0; for (int e=0; !cerr && e<E; ++e) { size_t ol=0,al=0;
                        if (gc->encode(f32 + (size_t)e*rows*cols, rows, cols, tt, 0,0, obuf+(size_t)e*slab_nb, (size_t)slab_nb, &ol, NULL,0,&al)!=0) cerr=1; } }
                }
                /* FALLBACK (ripristino migrazione): se il tipo richiesto non e' codificabile per
                 * QUESTA shape (i K-quant esigono cols multiplo di 256; 1D/piccoli no) ripiega su F16
                 * -- il GGUF ammette un tipo per-tensore. Senza, tensori validi vengono scartati e
                 * l'errore appare come "dtype non mappato" (es. switch_mlp.down_proj cols=1408, o il
                 * layer denso down_proj cols=10944). */
                if (cerr) {
                    free(obuf); obuf = NULL; tt = LLMCODEC_GGML_T_F16;
                    if (gc && gc->sizes(tt,0,0,rows,cols,&slab_nb,&aux_nb)==0) {
                        obuf = (uint8_t*)malloc((size_t)slab_nb*(size_t)E);
                        if (obuf) { cerr=0; for (int e=0; !cerr && e<E; ++e) { size_t ol=0,al=0;
                            if (gc->encode(f32 + (size_t)e*rows*cols, rows, cols, tt, 0,0, obuf+(size_t)e*slab_nb, (size_t)slab_nb, &ol, NULL,0,&al)!=0) cerr=1; } }
                    }
                }
                free(f32);
                if (!cerr && stack && E == 1) {
                    int e = src_expert_of(t->name);
                    char tname[288]; gguf_target_name(rm, t->name, tname, sizeof tname);
                    if (e < 0) { free(obuf); free(raw); free(dm); continue; }
                    if (np == pcap) { pcap = pcap ? pcap*2 : 256; pend=(PendExp*)realloc(pend,(size_t)pcap*sizeof(PendExp)); }
                    pend[np].cols=cols; pend[np].rows=rows; pend[np].expert=e; pend[np].type=tt;
                    pend[np].n=(size_t)slab_nb; pend[np].data=obuf;
                    snprintf(pend[np].target,sizeof pend[np].target,"%s",tname); np++;
                    free(raw); free(dm); goto next_tensor;
                }
                if (!cerr) {
                    char bname[288]; gguf_target_name(rm, t->name, bname, sizeof bname);
                    uint64_t* dd=(uint64_t*)calloc(8,sizeof(uint64_t)); dd[0]=cols; dd[1]=rows; if (E>1) dd[2]=(uint64_t)E;
                    if (nt == cap) { cap = cap ? cap*2 : 256; ts=(LlmCodecGgufTensor*)realloc(ts,(size_t)cap*sizeof(LlmCodecGgufTensor)); bufs=(uint8_t**)realloc(bufs,(size_t)cap*sizeof(uint8_t*)); dmv=(uint64_t**)realloc(dmv,(size_t)cap*sizeof(uint64_t*)); }
                    ts[nt].name=strdup(bname); ts[nt].ggml_type=tt; ts[nt].dims=dd; ts[nt].ndim=(E>1?3:2); ts[nt].data=obuf;
                    bufs[nt]=obuf; dmv[nt]=dd; nt++;
                    free(raw); free(dm); goto next_tensor;
                }
                free(obuf);
            }
            {
                int gt = llmcodec_ggml_type_of(t->dtype);
                if (gt < 0) { fprintf(stderr, "[mlx->gguf] ERRORE dtype non mappato: %s (%s) — saltato\n", t->name, t->dtype);
                              nskip++; free(raw); free(dm); continue; }
                /* Applica il FORMATO del ROLE (es. NORM -> f32): llama.cpp vuole le norm in F32
                 * (Metal rmsnorm: src[1] dev'essere F32). Le norm 1D non passano da llmcodec_decode_src
                 * (nessun weight_source per F16), quindi qui si fa l'upcast F16/BF16 -> F32. */
                const char* rfmt = rm ? llmcodec_rolemap_fmt(rm, llmcodec_rolemap_of(rm, t->name)) : NULL;
                int rt = (rfmt != NULL) ? llmcodec_ggml_type_of_fmt(rfmt) : -1;
                /* I tensori 1D (norm, bias) vanno in F32 per llama.cpp; inoltre il role puo' forzarlo. */
                if ((rt == LLMCODEC_GGML_T_F32 || t->ndim == 1) && gt != LLMCODEC_GGML_T_F32 && (gt == LLMCODEC_GGML_T_F16 || gt == LLMCODEC_GGML_T_BF16)) {
                    uint64_t numel = 1; for (int d = 0; d < t->ndim; ++d) numel *= (uint64_t)t->shape[d];
                    float* f = (float*)malloc((size_t)numel * 4);
                    if (f != NULL) {
                        const uint16_t* h = (const uint16_t*)raw;
                        for (uint64_t k = 0; k < numel; ++k)
                            f[k] = (gt == LLMCODEC_GGML_T_F16) ? llmcodec_f16_to_f32(h[k]) : llmcodec_bf16_to_f32(h[k]);
                        out = (uint8_t*)f; gtype = LLMCODEC_GGML_T_F32;
                        for (int d = 0; d < t->ndim; ++d) dm[d] = t->shape[t->ndim-1-d];
                        free(raw); raw = NULL;
                    } else { for (int d=0; d<t->ndim; ++d) dm[d]=t->shape[t->ndim-1-d]; out = raw; gtype = gt; raw = NULL; }
                } else {
                    for (int d=0; d<t->ndim; ++d) dm[d]=t->shape[t->ndim-1-d];
                    out = raw; gtype = gt; raw = NULL;
                }
            }
            if (nt == cap) { cap = cap ? cap*2 : 256; ts=(LlmCodecGgufTensor*)realloc(ts,(size_t)cap*sizeof(LlmCodecGgufTensor)); bufs=(uint8_t**)realloc(bufs,(size_t)cap*sizeof(uint8_t*)); dmv=(uint64_t**)realloc(dmv,(size_t)cap*sizeof(uint64_t*)); }
            ts[nt].name = strdup(gname); ts[nt].ggml_type = gtype; ts[nt].dims = dm; ts[nt].ndim = t->ndim; ts[nt].data = out;
            bufs[nt] = out; dmv[nt] = dm; nt++;
            free(raw);
            next_tensor: ;
        }
        llmcodec_safetensors_info_free(ti);
    }
    /* Assembla i gruppi di esperti impilati: [cols, rows, E] con E in ordine di esperto. */
    for (int i = 0; i < np; ++i) {
        if (pend[i].data == NULL) continue;
        int Emax = -1;
        for (int k = 0; k < np; ++k) if (pend[k].data && strcmp(pend[k].target, pend[i].target) == 0 && pend[k].expert > Emax) Emax = pend[k].expert;
        uint32_t E = (uint32_t)(Emax + 1);
        size_t per = pend[i].n; uint8_t* big = (uint8_t*)malloc(per * E); int okc = (big != NULL);
        for (uint32_t e = 0; e < E && okc; ++e) {
            int f = -1; for (int k = 0; k < np; ++k) if (pend[k].data && strcmp(pend[k].target,pend[i].target)==0 && pend[k].expert==(int)e) { f=k; break; }
            if (f < 0) { okc = 0; break; }
            memcpy(big + (size_t)e*per, pend[f].data, per);
        }
        if (okc) {
            uint64_t* dd=(uint64_t*)calloc(8,sizeof(uint64_t)); dd[0]=pend[i].cols; dd[1]=pend[i].rows; dd[2]=E;
            if (nt == cap) { cap = cap ? cap*2 : 256; ts=(LlmCodecGgufTensor*)realloc(ts,(size_t)cap*sizeof(LlmCodecGgufTensor)); bufs=(uint8_t**)realloc(bufs,(size_t)cap*sizeof(uint8_t*)); dmv=(uint64_t**)realloc(dmv,(size_t)cap*sizeof(uint64_t*)); }
            ts[nt].name=strdup(pend[i].target); ts[nt].ggml_type=pend[i].type; ts[nt].dims=dd; ts[nt].ndim=3; ts[nt].data=big;
            bufs[nt]=big; dmv[nt]=dd; nt++;
        } else { free(big); fprintf(stderr, "[mlx->gguf] stacking incompleto per %s\n", pend[i].target); }
        for (int k = 0; k < np; ++k) if (pend[k].data && strcmp(pend[k].target,pend[i].target)==0) { free(pend[k].data); pend[k].data=NULL; }
    }
    free(pend);
    const char* arch = (rm && rm->arch[0]) ? rm->arch : "llm";
    LlmCodecGgufMetaKV* kv = NULL; int nkv = 0;
    if (llmcodec_gguf_meta_build(mlx_dir, arch, tm, (rm && rm->tokenizer_pre[0]) ? rm->tokenizer_pre : NULL, &kv, &nkv) != 0 || nkv == 0) {
        kv = (LlmCodecGgufMetaKV*)calloc(1, sizeof(LlmCodecGgufMetaKV)); if (kv) { kv[0].key="general.architecture"; kv[0].type=LLMCODEC_GGUF_MT_STRING; kv[0].str=arch; nkv=1; }
    }
    if (nskip > 0) fprintf(stderr, "[mlx->gguf] ATTENZIONE: %d tensori saltati (dtype non mappato)\n", nskip);
    int rc = llmcodec_gguf_write(out_path, kv, nkv, ts, nt, 32);
    if (kv) llmcodec_gguf_meta_free_built(kv, nkv);
    for (int i=0;i<nt;i++){ free((void*)ts[i].name); free(bufs[i]); free(dmv[i]); }
    free(ts); free(bufs); free(dmv);
    return rc;
}

static int llmcodec_mkdir_p(const char* p) { char b[1200]; snprintf(b,sizeof b,"%s",p); for(char* q=b+1;*q;++q) if(*q=='/'){*q=0;mkdir(b,0777);*q='/';} mkdir(b,0777); return 0; }
static void llmcodec_write_config_quant(const char* out_dir, uint32_t group, uint32_t bits, const LlmCodecRoleMap* rm) {
    char cp[1300]; snprintf(cp,sizeof cp,"%s/config.json",out_dir);
    FILE* f=fopen(cp,"wb"); if(!f) return;
    if (rm && rm->arch[0])
        fprintf(f,"{\"architectures\":[\"%s\"],\"model_type\":\"%s\",\"quantization\":{\"group_size\":%u,\"bits\":%u}}\n",
                rm->arch, rm->arch, group, bits);
    else
        fprintf(f,"{\"quantization\":{\"group_size\":%u,\"bits\":%u}}\n", group, bits);
    fclose(f);
}

static void jw_str(FILE* o, const char* s) {
    fputc('"', o);
    for (const unsigned char* p = (const unsigned char*)s; *p; ++p) {
        unsigned char c = *p;
        if (c == '"' || c == '\\') { fputc('\\', o); fputc((int)c, o); }
        else if (c == '\n') fputs("\\n", o);
        else if (c == '\r') fputs("\\r", o);
        else if (c == '\t') fputs("\\t", o);
        else if (c < 0x20) fprintf(o, "\\u%04x", (unsigned)c);
        else fputc((int)c, o);
    }
    fputc('"', o);
}

static int llmcodec_write_tokenizer_json(const char* gguf_path, const char* out_dir) {
    LlmCodecGgufMetaKVOwned* kv = NULL; int n = 0; LlmCodecGgufFileInfo fi;
    if (llmcodec_gguf_read_meta(gguf_path, &kv, &n, &fi) != 0) return -1;
    const LlmCodecGgufMetaKVOwned* toks = NULL; const LlmCodecGgufMetaKVOwned* merges = NULL;
    const char* chat_tpl = NULL;
    const char* verbatim_json = NULL; const char* verbatim_cfg = NULL;
    long bos = -1, eos = -1;
    for (int i = 0; i < n; i++) {
        if (strcmp(kv[i].key, "tokenizer.ggml.tokens") == 0) toks = &kv[i];
        else if (strcmp(kv[i].key, "tokenizer.ggml.merges") == 0) merges = &kv[i];
        else if (strcmp(kv[i].key, "tokenizer.ggml.bos_token_id") == 0) bos = (long)kv[i].u32;
        else if (strcmp(kv[i].key, "tokenizer.ggml.eos_token_id") == 0) eos = (long)kv[i].u32;
        else if (strcmp(kv[i].key, "tokenizer.chat_template") == 0) chat_tpl = kv[i].str;
        else if (strcmp(kv[i].key, "tokenizer.gmb.json") == 0) verbatim_json = kv[i].str;
        else if (strcmp(kv[i].key, "tokenizer.gmb.config") == 0) verbatim_cfg = kv[i].str;
    }
    /* SE il GGUF porta il tokenizer.json VERBATIM (conservato in MLX->GGUF): lo si scrive tal
     * quale -> tokenizzazione ESATTA (la ricostruzione sotto e' lossy: manca il pre_tokenizer). */
    if (verbatim_json != NULL) {
        char op[1400]; snprintf(op, sizeof op, "%s/tokenizer.json", out_dir);
        FILE* o = fopen(op, "wb");
        if (o) { fwrite(verbatim_json, 1, strlen(verbatim_json), o); fclose(o); }
        if (verbatim_cfg != NULL) {
            char oc[1400]; snprintf(oc, sizeof oc, "%s/tokenizer_config.json", out_dir);
            FILE* c = fopen(oc, "wb");
            if (c) { fwrite(verbatim_cfg, 1, strlen(verbatim_cfg), c); fclose(c); }
        }
        llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, n);
        return 0;
    }
    int rc = -1;
    if (toks && toks->arr && toks->arr_n > 0) {
        char op[1400]; snprintf(op, sizeof op, "%s/tokenizer.json", out_dir);
        FILE* o = fopen(op, "wb");
        if (o) {
            fputs("{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[", o);
            int first = 1;
            for (int i = 0; i < toks->arr_n; i++) {
                if ((long)i != bos && (long)i != eos) continue;
                if (!first) fputc(',', o); first = 0;
                fprintf(o, "{\"id\":%d,\"content\":", i); jw_str(o, toks->arr[i]);
                fputs(",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":true}", o);
            }
            fputs("],\"normalizer\":null,"
                  "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":["
                  "{\"type\":\"Split\",\"pattern\":{\"Regex\":\"[\\\\r\\\\n]\"},\"behavior\":\"Isolated\",\"invert\":false},"
                  "{\"type\":\"Split\",\"pattern\":{\"Regex\":\"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?\\\\p{L}+|\\\\p{N}{1,3}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+\"},\"behavior\":\"Isolated\",\"invert\":false},"
                  "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,\"use_regex\":false}]},"
                  "\"post_processor\":null,"
                  "\"decoder\":{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":true,\"use_regex\":true},"
                  "\"model\":{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":null,\"continuing_subword_prefix\":null,"
                  "\"end_of_word_suffix\":null,\"fuse_unk\":false,\"byte_fallback\":false,\"vocab\":{", o);
            for (int i = 0; i < toks->arr_n; i++) { if (i) fputc(',', o); jw_str(o, toks->arr[i]); fprintf(o, ":%d", i); }
            fputs("},\"merges\":[", o);
            if (merges && merges->arr) for (int i = 0; i < merges->arr_n; i++) { if (i) fputc(',', o); jw_str(o, merges->arr[i]); }
            fputs("]}}", o);
            fclose(o); rc = 0;
            /* tokenizer_config.json: l'engine MLX legge eos/bos da QUI (non da config.json). */
            char tc[1400]; snprintf(tc, sizeof tc, "%s/tokenizer_config.json", out_dir);
            FILE* t2 = fopen(tc, "wb");
            if (t2) {
                fputs("{\"add_bos_token\":true,\"add_eos_token\":false,\"tokenizer_class\":\"PreTrainedTokenizerFast\"", t2);
                if (bos >= 0 && bos < toks->arr_n) { fputs(",\"bos_token\":", t2); jw_str(t2, toks->arr[bos]); }
                if (eos >= 0 && eos < toks->arr_n) { fputs(",\"eos_token\":", t2); jw_str(t2, toks->arr[eos]); fputs(",\"pad_token\":", t2); jw_str(t2, toks->arr[eos]); }
                if (chat_tpl != NULL) { fputs(",\"chat_template\":", t2); jw_str(t2, chat_tpl); }
                fputs("}\n", t2);
                fclose(t2);
            }
        }
    }
    llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, n);
    return rc;
}

static long long gkv_num(const LlmCodecGgufMetaKVOwned* kv, int n, const char* key) {
    for (int i = 0; i < n; i++) if (strcmp(kv[i].key, key) == 0) {
        if (kv[i].type == LLMCODEC_GGUF_MT_U32) return (long long)kv[i].u32;
        if (kv[i].type == LLMCODEC_GGUF_MT_I32) return (long long)kv[i].i32;
    }
    return -1;
}
static double gkv_flt(const LlmCodecGgufMetaKVOwned* kv, int n, const char* key) {
    for (int i = 0; i < n; i++) if (strcmp(kv[i].key, key) == 0 && kv[i].type == LLMCODEC_GGUF_MT_F32) return (double)kv[i].f32;
    return -1;
}
static const char* gkv_str(const LlmCodecGgufMetaKVOwned* kv, int n, const char* key) {
    for (int i = 0; i < n; i++) if (strcmp(kv[i].key, key) == 0 && kv[i].type == LLMCODEC_GGUF_MT_STRING) return kv[i].str;
    return NULL;
}
static void llmcodec_write_config_from_gguf(const char* gguf_path, const char* out_dir, const char* arch, uint32_t group, uint32_t bits) {
    LlmCodecGgufMetaKVOwned* kv = NULL; int n = 0; LlmCodecGgufFileInfo fi;
    int have = (llmcodec_gguf_read_meta(gguf_path, &kv, &n, &fi) == 0);
    /* SE il GGUF porta il config.json VERBATIM (conservato in MLX->GGUF): lo si scrive tal quale
     * -> tutti i campi HF esatti (rope_scaling/yarn, routed_scaling_factor, norm_topk_prob, ...).
     * La ricostruzione sotto copre solo i campi che il GGUF porta: ne perde di CRITICI -> garbage. */
    { const char* vcfg = have ? gkv_str(kv, n, "model.gmb.config") : NULL;
      if (vcfg != NULL) {
          char cpv[1300]; snprintf(cpv, sizeof cpv, "%s/config.json", out_dir);
          FILE* fv = fopen(cpv, "wb");
          if (fv) { fwrite(vcfg, 1, strlen(vcfg), fv); fclose(fv); }
          llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, n);
          return;
      } }
    char cp[1300]; snprintf(cp, sizeof cp, "%s/config.json", out_dir);
    FILE* f = fopen(cp, "wb"); if (!f) { if (have) { llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, n); } return; }
    fprintf(f, "{\n  \"architectures\": [\"%s\"],\n  \"model_type\": \"%s\",\n  \"tie_word_embeddings\": false", arch, arch);
    if (have) {
        char k[192];
        /* Le chiavi metadato GGUF usano l'arch di `general.architecture` (es. "qwen3moe"), che
         * può DIVERGERE dal nome ABI (`rm->arch`, es. "qwen3_moe"): usando `arch` (ABI) le chiavi
         * non si trovavano e la config usciva VUOTA (hidden=0 -> modello non caricabile). */
        const char* archk = gkv_str(kv, n, "general.architecture");
        if (archk == NULL || !archk[0]) archk = arch;
#define KVN(SUF, JSON) { snprintf(k, sizeof k, "%s.%s", archk, SUF); long long v = gkv_num(kv, n, k); if (v >= 0) fprintf(f, ",\n  \"%s\": %lld", JSON, v); }
        KVN("embedding_length", "hidden_size")
        KVN("block_count", "num_hidden_layers")
        KVN("attention.head_count", "num_attention_heads")
        KVN("attention.head_count_kv", "num_key_value_heads")
        KVN("attention.key_length", "head_dim")
        KVN("vocab_size", "vocab_size")
        KVN("context_length", "max_position_embeddings")
        KVN("expert_count", "n_routed_experts")
        KVN("expert_used_count", "num_experts_per_tok")
        KVN("expert_shared_count", "n_shared_experts")
        KVN("expert_feed_forward_length", "moe_intermediate_size")
        KVN("feed_forward_length", "intermediate_size")
        KVN("expert_group_count", "n_group")
        KVN("expert_group_used_count", "topk_group")
        KVN("attention.first_k_dense_replace", "first_k_dense_replace")
        KVN("attention.kv_lora_rank", "kv_lora_rank")
        KVN("attention.qk_nope_head_dim", "qk_nope_head_dim")
        KVN("attention.qk_rope_head_dim", "qk_rope_head_dim")
        KVN("attention.v_head_dim", "v_head_dim")
        KVN("attention.q_lora_rank", "q_lora_rank")
        /* Fallback vocab_size: se il GGUF non porta `<arch>.vocab_size`, derivalo dalla shape di
         * `token_embd.weight` ([hidden, vocab] in GGUF: dims[1]=vocab) — senza, load_config
         * fallisce ("chiave attesa mancante: vocab_size") e il modello MLX non si carica. */
        { snprintf(k, sizeof k, "%s.vocab_size", archk);
          if (gkv_num(kv, n, k) < 0)
            for (uint64_t ti = 0; ti < fi.n_tensors; ti++)
              if (fi.tensors[ti].name && strcmp(fi.tensors[ti].name, "token_embd.weight") == 0 && fi.tensors[ti].ndim >= 2) {
                fprintf(f, ",\n  \"vocab_size\": %llu", (unsigned long long)fi.tensors[ti].dims[1]); break; } }
#undef KVN
        { snprintf(k, sizeof k, "%s.rope.freq_base", archk); double v = gkv_flt(kv, n, k); fprintf(f, ",\n  \"rope_theta\": %g", v > 0 ? v : 10000.0); }
        { snprintf(k, sizeof k, "%s.attention.layer_norm_rms_epsilon", archk); double v = gkv_flt(kv, n, k); fprintf(f, ",\n  \"rms_norm_eps\": %g", v > 0 ? v : 1e-06); }
        { long long b = gkv_num(kv, n, "tokenizer.ggml.bos_token_id"); if (b >= 0) fprintf(f, ",\n  \"bos_token_id\": %lld", b); }
        { long long e = gkv_num(kv, n, "tokenizer.ggml.eos_token_id"); if (e >= 0) fprintf(f, ",\n  \"eos_token_id\": %lld", e); }
        /* Router scoring (softmax|sigmoid): SEMANTICAMENTE CRITICO (cambia il routing). Senza,
         * l'engine usa l'euristica per-arch -> output incoerente. */
        { snprintf(k, sizeof k, "%s.expert_scoring_func", archk); const char* sf = gkv_str(kv, n, k); if (sf) fprintf(f, ",\n  \"scoring_func\": \"%s\"", sf); }
        llmcodec_gguf_free_info(&fi); llmcodec_gguf_meta_free(kv, n);
    }
    fprintf(f, ",\n  \"quantization\": {\"group_size\": %u, \"bits\": %u}\n}\n", group, bits);
    fclose(f);
}

int llmcodec_gguf_to_mlx_ex(const char* gguf_path, const char* out_dir, uint32_t group, uint32_t bits, const LlmCodecRoleMap* rm) {
    if (group == 0 && rm && llmcodec_rolemap_group(rm) > 0) group = (uint32_t)llmcodec_rolemap_group(rm);
    if (bits  == 0 && rm && llmcodec_rolemap_bits(rm)  > 0) bits  = (uint32_t)llmcodec_rolemap_bits(rm);
    if (group == 0) group = 32; if (bits == 0) bits = 4;
    LlmCodecGgufFileInfo info;
    if (llmcodec_gguf_read_info(gguf_path, &info) != 0) return -1;
    llmcodec_mkdir_p(out_dir);
    int cap = (int)(info.n_tensors * 3 + 4);
    LlmCodecStTensor* st = (LlmCodecStTensor*)calloc((size_t)cap, sizeof(LlmCodecStTensor));
    uint8_t** bufs = (uint8_t**)calloc((size_t)cap, sizeof(uint8_t*));
    int* shp = (int*)calloc((size_t)cap*8, sizeof(int));
    char** nms = (char**)calloc((size_t)cap, sizeof(char*));
    int ns = 0;
    for (uint64_t i = 0; i < info.n_tensors; ++i) {
        LlmCodecGgufTensorInfo* t = &info.tensors[i];
        /* Nome HF di destinazione (se l'arch ha il template `hf`): un artefatto MLX deve usare i
         * nomi HF (model.layers.N.self_attn.q_proj) per essere caricabile dagli strumenti MLX/HF.
         * Fallback: nome canonico GGUF verbatim (caricabile via gguf_rm). */
        char hfn[288]; int hfok = (rm != NULL) ? llmcodec_rolemap_canonical_to_hf(rm, t->name, hfn, sizeof hfn) : 0;
        const char* outname = hfok ? hfn : t->name;
        uint64_t numel = 1; for (int d=0; d<t->ndim; ++d) numel *= t->dims[d];
        uint64_t nb = 0; if (llmcodec_ggml_type_bytes(t->ggml_type, numel, &nb) != 0) continue;
        uint8_t* raw = llmcodec_slurp_range(gguf_path, info.data_start + t->offset, nb);
        if (raw == NULL) continue;
        {   /* A1 target: codec dichiarato dal file (`convert_target`), via registry */
            int tfmt = 0; int tp0 = (int)group, tp1 = (int)bits;
            const char* rn = rm ? llmcodec_rolemap_of(rm, t->name) : NULL;
            const char* tcname = rm ? llmcodec_rolemap_convert(rm, rn, &tfmt, &tp0, &tp1) : NULL;
            if (tcname == NULL) { tcname = "mlx_affine"; tfmt = 0; tp0 = (int)group; tp1 = (int)bits; }
            const LlmCodec* tc = llmcodec_resolve(tcname);
            const LlmCodec* gc = llmcodec_resolve("ggml");
            int plain_in = (t->ggml_type == 0 || t->ggml_type == 1 || t->ggml_type == 30);
            if (!plain_in && tc && gc && tc->encode && tc->store_shape_of && (t->ndim == 2 || t->ndim == 3) && numel > 0) {
                int E = (t->ndim == 3) ? (int)t->dims[2] : 1;
                size_t rows = (size_t)t->dims[1], cols = (size_t)t->dims[0];
                size_t m0=0,m1=0,ob=0,ab=0,sob=0,sab=0;
                if (tc->store_shape_of(0,tfmt,tp0,tp1,rows,cols,&m0,&m1)==0 &&
                    gc->sizes(t->ggml_type,0,0,rows,cols,&sob,&sab)==0 &&
                    tc->sizes(tfmt,tp0,tp1,rows,cols,&ob,&ab)==0) {
                    uint8_t* obuf=(uint8_t*)malloc(ob*E); uint8_t* abuf=(ab?(uint8_t*)malloc(ab*E):NULL);
                    float* f=(float*)malloc(rows*cols*4);
                    int cerr=(!obuf||!f||(ab&&!abuf));
                    for (int e=0; !cerr && e<E; ++e) {
                        if (gc->decode(raw+(uint64_t)e*sob, sob, t->ggml_type, 0,0, rows, cols, NULL,0, f)!=0){cerr=1;break;}
                        size_t ol=0,al=0;
                        if (tc->encode(f, rows, cols, tfmt, tp0, tp1, obuf+(size_t)e*ob, ob, &ol, abuf?abuf+(size_t)e*ab:NULL, ab, &al)!=0) cerr=1;
                    }
                    free(f);
                    if (!cerr) {
                        int od=(E>1)?3:2;
                        char base[300]; snprintf(base,sizeof base,"%s",outname);
                        { size_t bl=strlen(base); if(bl>7&&strcmp(base+bl-7,".weight")==0) base[bl-7]=0; }
                        char b1[300]; snprintf(b1,sizeof b1,"%s.weight",base);
                        nms[ns]=strdup(b1);
                        if(od==3){shp[ns*8]=E; shp[ns*8+1]=(int)m0; shp[ns*8+2]=(int)m1;}else{shp[ns*8]=(int)m0; shp[ns*8+1]=(int)m1;}
                        st[ns].name=nms[ns]; st[ns].dtype=tc->dtype_of?tc->dtype_of(0,tfmt):"F32"; st[ns].shape=&shp[ns*8]; st[ns].ndim=od; st[ns].data=obuf; bufs[ns]=(uint8_t*)obuf; ns++;
                        if (tc->aux_kind != LLMCODEC_CODEC_AUX_NONE && abuf) {
                            size_t a0=0,a1=0; tc->store_shape_of(1,tfmt,tp0,tp1,rows,cols,&a0,&a1);
                            const char* ad = tc->dtype_of?tc->dtype_of(1,tfmt):"F16";
                            const char* ssuf = (rm&&rm->scale_suffix[0])?rm->scale_suffix:".scales";
                            const char* bsuf = (rm&&rm->bias_suffix[0])?rm->bias_suffix:".biases";
                            size_t scb = (tc->aux_kind==LLMCODEC_CODEC_AUX_SCALES_BIASES)? ab/2 : ab;
                            char b2[300]; snprintf(b2,sizeof b2,"%s%s",base,ssuf);
                            nms[ns]=strdup(b2); if(od==3){shp[ns*8]=E; shp[ns*8+1]=(int)a0; shp[ns*8+2]=(int)a1;}else{shp[ns*8]=(int)a0; shp[ns*8+1]=(int)a1;}
                            uint8_t* sb=(uint8_t*)malloc(scb*E); for(int e=0;e<E;e++) memcpy(sb+(size_t)e*scb, abuf+(size_t)e*ab, scb);
                            st[ns].name=nms[ns]; st[ns].dtype=ad; st[ns].shape=&shp[ns*8]; st[ns].ndim=od; st[ns].data=sb; bufs[ns]=(uint8_t*)sb; ns++;
                            if (tc->aux_kind==LLMCODEC_CODEC_AUX_SCALES_BIASES) {
                                char b3[300]; snprintf(b3,sizeof b3,"%s%s",base,bsuf);
                                nms[ns]=strdup(b3); if(od==3){shp[ns*8]=E; shp[ns*8+1]=(int)a0; shp[ns*8+2]=(int)a1;}else{shp[ns*8]=(int)a0; shp[ns*8+1]=(int)a1;}
                                uint8_t* bb=(uint8_t*)malloc(scb*E); for(int e=0;e<E;e++) memcpy(bb+(size_t)e*scb, abuf+(size_t)e*ab+scb, scb);
                                st[ns].name=nms[ns]; st[ns].dtype=ad; st[ns].shape=&shp[ns*8]; st[ns].ndim=od; st[ns].data=bb; bufs[ns]=(uint8_t*)bb; ns++;
                            }
                        }
                        free(abuf); free(raw); continue;   /* obuf: liberato dal free comune (bufs) */
                    }
                    free(obuf); free(abuf);
                }
            }
        }
        /* copia: dequant a f32 */
        float* f = (float*)malloc((size_t)numel*4);
        if (f && llmcodec_ggml_dequant_row(t->ggml_type, raw, f, (int64_t)numel)==0) {
            const char* odt = "F32";
            void* data = f;
            nms[ns]=strdup(outname); for(int d=0;d<t->ndim;++d) shp[ns*8+d]=(int)t->dims[t->ndim-1-d];
            st[ns].name=nms[ns]; st[ns].dtype=odt; st[ns].shape=&shp[ns*8]; st[ns].ndim=t->ndim; st[ns].data=data; bufs[ns]=(uint8_t*)data; ns++;
            if (data != f) free(f);
        } else free(f);
        free(raw);
    }
    char op[1300]; snprintf(op,sizeof op,"%s/model.safetensors",out_dir);
    int rc = llmcodec_safetensors_write(op, st, ns);
    llmcodec_write_config_from_gguf(gguf_path, out_dir, (rm && rm->arch[0]) ? rm->arch : "llm", group, bits);
    llmcodec_write_tokenizer_json(gguf_path, out_dir);
    for (int i=0;i<ns;i++){ free(nms[i]); free(bufs[i]); }
    free(st); free(bufs); free(shp); free(nms); llmcodec_gguf_free_info(&info);
    return rc;
}

int llmcodec_mlx_to_gguf(const char* mlx_dir, const char* out_path, int ggml_type, uint32_t group, uint32_t bits) {
    return llmcodec_mlx_to_gguf_ex(mlx_dir, out_path, ggml_type, group, bits, NULL, NULL);
}

int llmcodec_conv_mlx_to_gguf_arch(const char* mlx_dir, const char* out_path, const char* llmarch_path,
                              int ggml_type, uint32_t group, uint32_t bits) {
    return llmcodec_conv_mlx_to_gguf_arch_ex(mlx_dir, out_path, llmarch_path, ggml_type, group, bits, NULL);
}

int llmcodec_conv_mlx_to_gguf_arch_ex(const char* mlx_dir, const char* out_path, const char* llmarch_path,
                                 int ggml_type, uint32_t group, uint32_t bits, const LlmCodecTokenizerMeta* tm) {
    LlmCodecRoleMap rm;
    if (llmcodec_rolemap_load(llmarch_path, &rm) != 0) return -1;
    return llmcodec_mlx_to_gguf_ex(mlx_dir, out_path, ggml_type, group, bits, &rm, tm);
}

int llmcodec_gguf_to_mlx(const char* gguf_path, const char* out_dir, uint32_t group, uint32_t bits) {
    return llmcodec_gguf_to_mlx_ex(gguf_path, out_dir, group, bits, NULL);
}

int llmcodec_conv_gguf_to_mlx_arch(const char* gguf_path, const char* out_dir, const char* llmarch_path,
                              uint32_t group, uint32_t bits) {
    LlmCodecRoleMap rm;
    if (llmcodec_rolemap_load(llmarch_path, &rm) != 0) return -1;
    return llmcodec_gguf_to_mlx_ex(gguf_path, out_dir, group, bits, &rm);
}
