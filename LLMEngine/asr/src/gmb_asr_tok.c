/* gmb_asr_tok.c — tokenizer Qwen byte-level BPE (vocab.json + merges.txt), self-contained.
 * Decode + encode (GPT-2-style byte-unicode). Reference: Qwen3-ASR. Nessuna dipendenza esterna. */
#include "gmb_asr_tok.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ---- byte<->unicode (GPT-2) ---- */
static uint16_t s_cp2byte[0x11000];
static int s_byte2cp[256];
static int s_init = 0;
static void b2u_init(void) {
    if (s_init) return;
    int bs[256], cs[256], n = 0;
    for (int b = 33; b <= 126; b++) { bs[n] = b; cs[n] = b; n++; }
    for (int b = 161; b <= 172; b++) { bs[n] = b; cs[n] = b; n++; }
    for (int b = 174; b <= 255; b++) { bs[n] = b; cs[n] = b; n++; }
    int extra = 0;
    for (int b = 0; b < 256; b++) { int f = 0; for (int i = 0; i < n; i++) if (bs[i] == b) { f = 1; break; } if (!f) { bs[n] = b; cs[n] = 256 + extra; extra++; n++; } }
    for (int i = 0; i < n; i++) { s_cp2byte[cs[i]] = (uint16_t)bs[i]; s_byte2cp[bs[i]] = cs[i]; }
    s_init = 1;
}

static int u8_cp(const unsigned char* s, int* cl) {
    unsigned char c = s[0];
    if (c < 0x80) { *cl = 1; return c; }
    if ((c >> 5) == 6) { *cl = 2; return ((c & 0x1f) << 6) | (s[1] & 0x3f); }
    if ((c >> 4) == 14) { *cl = 3; return ((c & 0xf) << 12) | ((s[1] & 0x3f) << 6) | (s[2] & 0x3f); }
    if ((c >> 3) == 30) { *cl = 4; return ((c & 7) << 18) | ((s[1] & 0x3f) << 12) | ((s[2] & 0x3f) << 6) | (s[3] & 0x3f); }
    *cl = 1; return -1;
}
static int cp_u8(char* o, int cp) {   /* scrive il codepoint in UTF-8, ritorna #byte */
    if (cp < 0x80) { o[0] = (char)cp; return 1; }
    if (cp < 0x800) { o[0] = (char)(0xc0 | (cp >> 6)); o[1] = (char)(0x80 | (cp & 0x3f)); return 2; }
    if (cp < 0x10000) { o[0] = (char)(0xe0 | (cp >> 12)); o[1] = (char)(0x80 | ((cp >> 6) & 0x3f)); o[2] = (char)(0x80 | (cp & 0x3f)); return 3; }
    o[0] = (char)(0xf0 | (cp >> 18)); o[1] = (char)(0x80 | ((cp >> 12) & 0x3f)); o[2] = (char)(0x80 | ((cp >> 6) & 0x3f)); o[3] = (char)(0x80 | (cp & 0x3f)); return 4;
}

static char* json_unescape(const char* s, int len) {
    char* o = (char*)malloc((size_t)len * 4 + 4); int n = 0;
    for (int i = 0; i < len;) {
        if (s[i] == '\\' && i + 1 < len) {
            char e = s[i + 1]; i += 2;
            if (e == 'u' && i + 4 <= len) { int cp = 0; for (int k = 0; k < 4; k++) { char h = s[i + k]; int d = (h >= '0' && h <= '9') ? h - '0' : (h >= 'a' && h <= 'f') ? h - 'a' + 10 : (h >= 'A' && h <= 'F') ? h - 'A' + 10 : 0; cp = cp * 16 + d; } i += 4; n += cp_u8(o + n, cp); }
            else if (e == 'n') o[n++] = '\n'; else if (e == 't') o[n++] = '\t'; else if (e == 'r') o[n++] = '\r';
            else if (e == '"') o[n++] = '"'; else if (e == '\\') o[n++] = '\\'; else if (e == '/') o[n++] = '/'; else o[n++] = e;
        } else o[n++] = s[i++];
    }
    o[n] = 0; return o;
}

/* ---- hash maps (string -> int32) ---- */
static uint32_t hsh(const char* s, size_t len) { uint32_t h = 2166136261u; for (size_t i = 0; i < len; i++) { h ^= (uint8_t)s[i]; h *= 16777619u; } return h ? h : 1; }
typedef struct { char** k; int32_t* v; uint32_t* h; size_t cap; } Map;
static void map_init(Map* m, size_t n) { m->cap = 16; while (m->cap < n * 2) m->cap <<= 1; m->k = (char**)calloc(m->cap, sizeof(char*)); m->v = (int32_t*)calloc(m->cap, sizeof(int32_t)); m->h = (uint32_t*)calloc(m->cap, sizeof(uint32_t)); }
static void map_put(Map* m, const char* k, size_t len, int32_t v) { uint32_t h = hsh(k, len); size_t mask = m->cap - 1, i = h & mask; while (m->k[i]) { if (m->h[i] == h && strlen(m->k[i]) == len && !memcmp(m->k[i], k, len)) { m->v[i] = v; return; } i = (i + 1) & mask; } char* c = (char*)malloc(len + 1); memcpy(c, k, len); c[len] = 0; m->k[i] = c; m->h[i] = h; m->v[i] = v; }
static int map_get(const Map* m, const char* k, size_t len, int32_t* out) { uint32_t h = hsh(k, len); size_t mask = m->cap - 1, i = h & mask; while (m->k[i]) { if (m->h[i] == h && strlen(m->k[i]) == len && !memcmp(m->k[i], k, len)) { *out = m->v[i]; return 1; } i = (i + 1) & mask; } return 0; }
static void map_free(Map* m) { for (size_t i = 0; i < m->cap; i++) free(m->k[i]); free(m->k); free(m->v); free(m->h); }

struct GmbAsrTok { int vocab; char** id2tok; Map tok2id; Map merge; };

GmbAsrTok* gmb_asr_tok_load(const char* dir) {
    b2u_init();
    char vp[1400], mp[1400]; snprintf(vp, sizeof vp, "%s/vocab.json", dir); snprintf(mp, sizeof mp, "%s/merges.txt", dir);
    FILE* f = fopen(vp, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* js = (char*)malloc((size_t)sz + 1); if (!js || fread(js, 1, (size_t)sz, f) != (size_t)sz) { free(js); fclose(f); return NULL; } js[sz] = 0; fclose(f);
    GmbAsrTok* t = (GmbAsrTok*)calloc(1, sizeof(GmbAsrTok));
    int cap = 200000; t->vocab = cap; t->id2tok = (char**)calloc((size_t)cap, sizeof(char*));
    map_init(&t->tok2id, 200000);
    char* p = js; while (*p && *p != '{') p++; if (*p) p++;
    for (;;) {
        while (*p && (*p == ',' || *p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
        if (!*p || *p == '}') break;
        if (*p != '"') break; p++;
        const char* ks = p; while (*p && *p != '"') { if (*p == '\\') p++; p++; } int kl = (int)(p - ks); if (*p) p++;
        while (*p && *p != ':') p++; if (*p) p++;
        while (*p == ' ') p++;
        long id = strtol(p, &p, 10);
        if (id >= 0 && id < cap) { char* tok = json_unescape(ks, kl); t->id2tok[id] = tok; map_put(&t->tok2id, tok, strlen(tok), (int32_t)id); }
    }
    free(js);
    /* merges.txt */
    map_init(&t->merge, 200000);
    FILE* mf = fopen(mp, "r");
    if (mf) { char line[1024]; int rank = 0; while (fgets(line, sizeof line, mf)) { if (line[0] == '#' || line[0] == '\n') continue; char* sp = strchr(line, ' '); if (!sp) continue; *sp = 0; char* r = sp + 1; size_t rl = strlen(r); while (rl && (r[rl - 1] == '\n' || r[rl - 1] == '\r')) r[--rl] = 0; char key[1024]; int kn = snprintf(key, sizeof key, "%s %s", line, r); map_put(&t->merge, key, (size_t)kn, rank++); } fclose(mf); }
    return t;
}
void gmb_asr_tok_free(GmbAsrTok* t) { if (!t) return; for (int i = 0; i < t->vocab; i++) free(t->id2tok[i]); free(t->id2tok); map_free(&t->tok2id); map_free(&t->merge); free(t); }
int gmb_asr_tok_n_vocab(const GmbAsrTok* t) { return t->vocab; }

static char g_piece[8192];
const char* gmb_asr_tok_decode(GmbAsrTok* t, int id) {
    if (id < 0 || id >= t->vocab || !t->id2tok[id]) { g_piece[0] = 0; return g_piece; }
    const unsigned char* s = (const unsigned char*)t->id2tok[id]; size_t n = 0;
    while (*s) { int cl; int cp = u8_cp(s, &cl); s += cl; if (cp >= 0 && cp < 0x11000 && s_cp2byte[cp]) g_piece[n++] = (char)(s_cp2byte[cp] & 0xff); else n += cp_u8(g_piece + n, cp); if (n > 8000) break; }
    g_piece[n] = 0; return g_piece;
}

/* BPE di un "word" (array di stringhe-simbolo UTF-8) -> id accodati a `ids`. */
static int bpe_word(GmbAsrTok* t, char** sym, int ns, int* ids, int ni) {
    while (ns >= 2) {
        int best = -1, br = 1 << 30;
        for (int i = 0; i + 1 < ns; i++) { char key[1024]; int kl = snprintf(key, sizeof key, "%s %s", sym[i], sym[i + 1]); int32_t r; if (map_get(&t->merge, key, (size_t)kl, &r) && r < br) { br = r; best = i; } }
        if (best < 0) break;
        size_t l1 = strlen(sym[best]), l2 = strlen(sym[best + 1]);
        char* m = (char*)malloc(l1 + l2 + 1); memcpy(m, sym[best], l1); memcpy(m + l1, sym[best + 1], l2); m[l1 + l2] = 0;
        free(sym[best]); sym[best] = m; free(sym[best + 1]);
        for (int i = best + 1; i + 1 < ns; i++) sym[i] = sym[i + 1];
        ns--;
    }
    for (int i = 0; i < ns; i++) { int32_t id; ids[ni++] = map_get(&t->tok2id, sym[i], strlen(sym[i]), &id) ? id : 0; free(sym[i]); }
    return ni;
}

int* gmb_asr_tok_encode(GmbAsrTok* t, const char* utf8, int* out_n) {
    b2u_init();
    /* testo -> stringa byte-unicode (ogni byte dell'input rappresentato dal suo char byte-unicode) */
    size_t bl = strlen(utf8);
    char* bu = (char*)malloc(bl * 4 + 1); size_t bn = 0;
    for (size_t i = 0; i < bl; i++) { char tmp[8]; int l = cp_u8(tmp, s_byte2cp[(unsigned char)utf8[i]]); memcpy(bu + bn, tmp, (size_t)l); bn += (size_t)l; }
    bu[bn] = 0;
    /* split sui confini di spazio (Ġ = byte 0x20 → s_byte2cp[32]): il Ġ inizia un nuovo word. */
    int space_cp = s_byte2cp[32];
    char space_u8[8]; int spl = cp_u8(space_u8, space_cp);
    int* ids = (int*)malloc((bn + 4) * sizeof(int)); int ni = 0;
    /* itera i "word": un word inizia con il Ġ (o all'inizio) e prosegue fino al prossimo Ġ */
    size_t i = 0;
    while (i < bn) {
        size_t wstart = i;
        if ((size_t)spl <= bn - i && !memcmp(bu + i, space_u8, (size_t)spl)) i += (size_t)spl;   /* consuma il Ġ iniziale */
        while (i < bn && !((size_t)spl <= bn - i && !memcmp(bu + i, space_u8, (size_t)spl))) i++;
        /* word = bu[wstart,i) */
        int ns = 0; size_t q = wstart; char** sym = NULL; int scap = 0;
        while (q < i) { int cl; int cp = u8_cp((const unsigned char*)bu + q, &cl); q += (size_t)cl; char tmp[8]; int l = cp_u8(tmp, cp); if (ns == scap) { scap = scap ? scap * 2 : 16; sym = (char**)realloc(sym, (size_t)scap * sizeof(char*)); } sym[ns] = (char*)malloc((size_t)l + 1); memcpy(sym[ns], tmp, (size_t)l); sym[ns][l] = 0; ns++; }
        if (ns) ni = bpe_word(t, sym, ns, ids, ni);
        free(sym);
    }
    free(bu);
    *out_n = ni;
    return ids;
}
