/* llm_grammar_test.c — test di llm_grammar.h (decodifica vincolata JSON/Schema con bitmask), senza modello.
 *   clang -O2 -I LLMEngine/include LLMEngine/mlx/tests/llm_grammar_test.c -o /tmp/grammar_test && /tmp/grammar_test
 * Fuzz: passeggiate casuali sulle maschere; l'output DEVE essere JSON valido e conforme (oracolo indipendente scritto qui). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "llm_grammar.h"

/* ── oracolo: parser JSON stretto (indipendente dalla grammatica) ── */
typedef struct { const char* p; const char* e; } Q;
static void qws(Q* q) { while (q->p < q->e && (*q->p == ' ' || *q->p == '\n' || *q->p == '\t' || *q->p == '\r')) q->p++; }
static int qval(Q* q, int d, char* kinds, int* nk);
static int qstr(Q* q) {
    if (q->p >= q->e || *q->p != '"') return 0; q->p++;
    while (q->p < q->e && *q->p != '"') {
        unsigned char c = (unsigned char)*q->p++;
        if (c < 0x20) return 0;
        if (c == '\\') { if (q->p >= q->e) return 0; char x = *q->p++; if (x == 'u') { for (int i = 0; i < 4; i++) { if (q->p >= q->e) return 0; char h = *q->p++; if (!((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F'))) return 0; } } else if (!strchr("\"\\/bfnrt", x)) return 0; }
    }
    if (q->p >= q->e) return 0; q->p++; return 1;
}
static int qnum(Q* q) {
    const char* s = q->p; if (q->p < q->e && *q->p == '-') q->p++;
    if (q->p >= q->e) return 0;
    if (*q->p == '0') q->p++; else if (*q->p >= '1' && *q->p <= '9') while (q->p < q->e && *q->p >= '0' && *q->p <= '9') q->p++; else return 0;
    if (q->p < q->e && *q->p == '.') { q->p++; if (q->p >= q->e || *q->p < '0' || *q->p > '9') return 0; while (q->p < q->e && *q->p >= '0' && *q->p <= '9') q->p++; }
    if (q->p < q->e && (*q->p == 'e' || *q->p == 'E')) { q->p++; if (q->p < q->e && (*q->p == '+' || *q->p == '-')) q->p++; if (q->p >= q->e || *q->p < '0' || *q->p > '9') return 0; while (q->p < q->e && *q->p >= '0' && *q->p <= '9') q->p++; }
    return q->p > s;
}
static int qval(Q* q, int d, char* kinds, int* nk) {
    qws(q); if (q->p >= q->e || d > 30) return 0;
    char c = *q->p;
    if (c == '{') { q->p++; qws(q); if (q->p < q->e && *q->p == '}') { q->p++; return 1; }
        for (;;) { qws(q); if (!qstr(q)) return 0; qws(q); if (q->p >= q->e || *q->p++ != ':') return 0; if (!qval(q, d + 1, kinds, nk)) return 0; qws(q); if (q->p < q->e && *q->p == ',') { q->p++; continue; } if (q->p < q->e && *q->p == '}') { q->p++; return 1; } return 0; } }
    if (c == '[') { q->p++; qws(q); if (q->p < q->e && *q->p == ']') { q->p++; return 1; }
        for (;;) { if (!qval(q, d + 1, kinds, nk)) return 0; qws(q); if (q->p < q->e && *q->p == ',') { q->p++; continue; } if (q->p < q->e && *q->p == ']') { q->p++; return 1; } return 0; } }
    if (c == '"') return qstr(q);
    if (c == 't' && q->e - q->p >= 4 && !memcmp(q->p, "true", 4)) { q->p += 4; return 1; }
    if (c == 'f' && q->e - q->p >= 5 && !memcmp(q->p, "false", 5)) { q->p += 5; return 1; }
    if (c == 'n' && q->e - q->p >= 4 && !memcmp(q->p, "null", 4)) { q->p += 4; return 1; }
    return qnum(q);
}
static int valid_json(const char* s, int n) { Q q = { s, s + n }; char k[4]; int nk = 0; if (!qval(&q, 0, k, &nk)) return 0; qws(&q); return q.p == q.e; }

/* ── vocabolario sintetico: 256 byte singoli + token multibyte tipici + riempitivo casuale ── */
#define NV 4000
static uint8_t* g_tok[NV]; static int g_len[NV]; static int g_nv; static int g_eos;
static void add_tok(const char* s, int l) { g_tok[g_nv] = (uint8_t*)malloc((size_t)l + 1); memcpy(g_tok[g_nv], s, (size_t)l); g_len[g_nv] = l; g_nv++; }
static void build_vocab(unsigned seed) {
    g_nv = 0; srand(seed);
    for (int b = 0; b < 256; b++) { char c = (char)b; add_tok(&c, 1); }
    const char* mt[] = { "{\"", "\":", "\": ", ", \"", "\"}", "\"]", "[\"", ",", ", ", "true", "false", "null", "name", "age", "tags", "kind", "ok", "\n", "  ", "\n  ", "12", "123", "-1", "0.5", "1e5", "alpha", "beta", "hello", " world", "{}", "[]", "},", "\",", "\\n", "\\\"", "\\u00e9", " \"", "x\"", "\":{" , "}}", "]]", "\"}," };
    for (size_t i = 0; i < sizeof mt / sizeof *mt; i++) add_tok(mt[i], (int)strlen(mt[i]));
    while (g_nv < NV - 1) { char b[8]; int l = 1 + rand() % 6; for (int i = 0; i < l; i++) b[i] = (char)(32 + rand() % 95); add_tok(b, l); }
    g_eos = g_nv; add_tok("", 0);   /* EOS: testo vuoto (token speciale) */
}

static int walk(LlmGrammar* g, char* out, int cap, int maxsteps, int* ended) {
    uint32_t words[(NV + 31) / 32]; int n = 0; *ended = 0;
    llm_grammar_reset(g);
    for (int step = 0; step < maxsteps; step++) {
        int cnt = llm_grammar_fill_mask(g, words);
        if (cnt <= 0) return -1;
        int pick = -1, seen = 0;
        for (int t = 0; t < g_nv; t++) if (words[t >> 5] & (1u << (t & 31))) { seen++; if (rand() % seen == 0) pick = t; }
        /* se il documento e' completo, termina presto con alta probabilita' */
        if (llm_grammar_is_done(g) && (rand() % 4) != 0) pick = g_eos;
        if (llm_grammar_accept(g, pick) != 0) { out[n] = 0; fprintf(stderr, "accept rifiutato (pick=%d len=%d): out=[%s] tok=[%.*s]\n", pick, g_len[pick], out, g_len[pick], (char*)g_tok[pick]); return -2; }
        if (pick == g_eos) { *ended = 1; break; }
        if (n + g_len[pick] >= cap) return -3;
        memcpy(out + n, g_tok[pick], (size_t)g_len[pick]); n += g_len[pick];
    }
    out[n] = 0; return n;
}

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static int has(const char* s, const char* sub) { return strstr(s, sub) != NULL; }

static void run_schema(const char* name, const char* schema, int iters, int (*check)(const char*, int)) {
    char err[200];
    LlmGrammar* g = llm_grammar_new_schema(schema, strlen(schema), err, sizeof err);
    CHECK(g != NULL, "%s: compilazione: %s", name, err);
    if (!g) return;
    uint8_t* tk[NV]; int tl[NV]; for (int i = 0; i < g_nv; i++) { tk[i] = g_tok[i]; tl[i] = g_len[i]; }
    int eos[1] = { g_eos };
    llm_grammar_set_vocab(g, g_nv, (const uint8_t* const*)tk, tl, eos, 1);
    int ok = 0, bad = 0, unfinished = 0;
    for (int it = 0; it < iters; it++) {
        char out[4096]; int ended = 0;
        int n = walk(g, out, sizeof out, 400, &ended);
        if (n < 0) { CHECK(0, "%s: walk errore %d", name, n); bad++; continue; }
        if (!ended) { unfinished++; continue; }
        if (!valid_json(out, n)) { CHECK(0, "%s: JSON non valido: %s", name, out); bad++; continue; }
        if (check && !check(out, n)) { CHECK(0, "%s: non conforme: %s", name, out); bad++; continue; }
        ok++;
    }
    printf("%-22s validi=%d  non_validi=%d  non_terminati(400 passi)=%d  cache hit=%d miss=%d\n", name, ok, bad, unfinished, g->hits, g->misses);
    CHECK(ok > 0, "%s: nessun documento terminato", name);
    llm_grammar_free(g);
}
static int chk_person(const char* s, int n) {
    (void)n;
    if (!has(s, "\"name\"") || !has(s, "\"age\"")) return 0;           /* required */
    if (s[0] != '{' && s[0] != ' ' && s[0] != '\n' && s[0] != '\t' && s[0] != '\r') return 0;
    const char* k = strstr(s, "\"kind\"");
    if (k && !(has(k, "\"a\"") || has(k, "\"bb\"") || has(k, "3"))) return 0;
    return 1;
}
static int chk_nums(const char* s, int n) { (void)n; return strchr(s, '"') == NULL && strchr(s, '{') == NULL; }
static int chk_obj(const char* s, int n) { while (n && (*s == ' ' || *s == '\n' || *s == '\t' || *s == '\r')) { s++; n--; } return n && *s == '{'; }
static int chk_union(const char* s, int n) { (void)n; while (*s == ' ' || *s == '\n' || *s == '\t' || *s == '\r') s++; return *s == '"' || *s == 'n' || (*s >= '0' && *s <= '9') || *s == '-'; }

int main(void) {
    build_vocab(1);
    printf("vocab sintetico: %d token, EOS=%d\n", g_nv, g_eos);
    run_schema("persona", "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\",\"maxLength\":6},\"age\":{\"type\":\"integer\"},"
        "\"tags\":{\"type\":\"array\",\"items\":{\"type\":\"string\"},\"maxItems\":3},\"kind\":{\"enum\":[\"a\",\"bb\",3]},\"ok\":{\"type\":\"boolean\"}},"
        "\"required\":[\"name\",\"age\"]}", 1500, chk_person);
    run_schema("array numeri", "{\"type\":\"array\",\"items\":{\"type\":\"number\"},\"minItems\":1,\"maxItems\":5}", 1500, chk_nums);
    run_schema("union str|null|int", "{\"type\":[\"string\",\"null\",\"integer\"]}", 800, chk_union);
    run_schema("annidato+$ref", "{\"$defs\":{\"p\":{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"number\"},\"y\":{\"type\":\"number\"}},\"required\":[\"x\",\"y\"]}},"
        "\"type\":\"object\",\"properties\":{\"pts\":{\"type\":\"array\",\"items\":{\"$ref\":\"#/$defs/p\"},\"maxItems\":3}},\"required\":[\"pts\"]}", 1000, chk_obj);
    {   /* TOOL-CALLING: unione di due funzioni, avvolta in <tool_call>\n ... \n</tool_call> */
        const char* sch = "{\"anyOf\":["
            "{\"type\":\"object\",\"properties\":{\"name\":{\"const\":\"get_weather\"},\"arguments\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\",\"maxLength\":12},\"days\":{\"type\":\"integer\"}},\"required\":[\"city\"]}},\"required\":[\"name\",\"arguments\"]},"
            "{\"type\":\"object\",\"properties\":{\"name\":{\"const\":\"add\"},\"arguments\":{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"number\"},\"b\":{\"type\":\"number\"}},\"required\":[\"a\",\"b\"]}},\"required\":[\"name\",\"arguments\"]}]}";
        char err[160];
        LlmGrammar* g = llm_grammar_new_schema_wrapped(sch, strlen(sch), "<tool_call>\n", "\n</tool_call>", err, sizeof err);
        CHECK(g != NULL, "tool grammar: %s", err);
        if (g) {
            uint8_t* tk[NV]; int tl[NV]; for (int i = 0; i < g_nv; i++) { tk[i] = g_tok[i]; tl[i] = g_len[i]; }
            int eos[1] = { g_eos }; llm_grammar_set_vocab(g, g_nv, (const uint8_t* const*)tk, tl, eos, 1);
            int ok = 0, bad = 0, unf = 0;
            for (int it = 0; it < 1200; it++) {
                char out[4096]; int ended = 0; int n = walk(g, out, sizeof out, 400, &ended);
                if (n < 0) { CHECK(0, "tool: walk errore %d", n); bad++; continue; }
                if (!ended) { unf++; continue; }
                const char* P = "<tool_call>\n"; const char* S = "\n</tool_call>"; int pl = (int)strlen(P), sl = (int)strlen(S);
                int good = n > pl + sl && !strncmp(out, P, (size_t)pl) && !strcmp(out + n - sl, S) && valid_json(out + pl, n - pl - sl);
                if (good) {
                    const char* body = out + pl;
                    const char* nm = strstr(body, "\"name\"");
                    good = nm && (strstr(nm, "\"get_weather\"") || strstr(nm, "\"add\"")) && strstr(body, "\"arguments\"");
                    if (good && strstr(nm, "\"get_weather\"") && !strstr(body, "\"city\"")) good = 0;
                    if (good && strstr(nm, "\"add\"") && !(strstr(body, "\"a\"") && strstr(body, "\"b\""))) good = 0;
                }
                if (!good) { CHECK(0, "tool: chiamata non conforme: [%s]", out); bad++; continue; }
                ok++;
            }
            printf("%-22s validi=%d  non_validi=%d  non_terminati=%d\n", "tool_call wrapped", ok, bad, unf);
            CHECK(ok > 0, "tool: nessun documento");
            llm_grammar_free(g);
        }
    }
    {   /* json_object generico */
        LlmGrammar* g = llm_grammar_new_json_object();
        uint8_t* tk[NV]; int tl[NV]; for (int i = 0; i < g_nv; i++) { tk[i] = g_tok[i]; tl[i] = g_len[i]; }
        int eos[1] = { g_eos };
        llm_grammar_set_vocab(g, g_nv, (const uint8_t* const*)tk, tl, eos, 1);
        int ok = 0, unf = 0, bad = 0;
        for (int it = 0; it < 1500; it++) {
            char out[4096]; int ended = 0; int n = walk(g, out, sizeof out, 300, &ended);
            if (n < 0) { CHECK(0, "json_object: walk errore %d", n); bad++; continue; }
            if (!ended) { unf++; continue; }
            if (!valid_json(out, n) || !chk_obj(out, n)) { CHECK(0, "json_object: invalido: %s", out); bad++; continue; }
            ok++;
        }
        printf("%-22s validi=%d  non_validi=%d  non_terminati=%d\n", "json_object", ok, bad, unf);
        CHECK(ok > 0, "json_object: nessun doc");
        llm_grammar_free(g);
    }
    {   /* negativi */
        const char* sch = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}},\"required\":[\"a\"]}";
        char err[100]; LlmGrammar* g = llm_grammar_new_schema(sch, strlen(sch), err, sizeof err);
        uint8_t* tk[NV]; int tl[NV]; for (int i = 0; i < g_nv; i++) { tk[i] = g_tok[i]; tl[i] = g_len[i]; }
        int eos[1] = { g_eos }; llm_grammar_set_vocab(g, g_nv, (const uint8_t* const*)tk, tl, eos, 1);
        CHECK(llm_grammar_accept(g, 'x') == -1, "token illegale accettato");
        CHECK(llm_grammar_accept(g, g_eos) == -1, "EOS accettato a documento incompleto");
        const char* doc = "{\"a\":42}"; int okd = 1;
        for (const char* p = doc; *p; p++) if (llm_grammar_accept(g, (unsigned char)*p) != 0) okd = 0;
        CHECK(okd, "documento valido rifiutato");
        CHECK(llm_grammar_is_done(g), "non completo dopo il documento");
        CHECK(llm_grammar_accept(g, g_eos) == 0, "EOS rifiutato a documento completo");
        llm_grammar_free(g);
        CHECK(llm_grammar_new_schema("{bad", 4, err, sizeof err) == NULL, "schema invalido accettato");
    }
    {   /* tempo di fill_mask su un vocabolario da 150k (token casuali 1..8 byte) */
        const int NB = 150000; uint8_t** tk = (uint8_t**)malloc(sizeof(uint8_t*) * NB); int* tl = (int*)malloc(sizeof(int) * NB);
        srand(5);
        for (int i = 0; i < NB; i++) { int l = i < 256 ? 1 : 1 + rand() % 8; tk[i] = (uint8_t*)malloc((size_t)l); tl[i] = l; for (int k = 0; k < l; k++) tk[i][k] = i < 256 ? (uint8_t)i : (uint8_t)(32 + rand() % 95); }
        int eos[1] = { NB - 1 }; tl[NB - 1] = 0;
        const char* sch = "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"age\":{\"type\":\"integer\"},\"tags\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}},\"required\":[\"name\",\"age\"]}";
        char err[100]; LlmGrammar* g = llm_grammar_new_schema(sch, strlen(sch), err, sizeof err);
        llm_grammar_set_vocab(g, NB, (const uint8_t* const*)tk, tl, eos, 1);
        uint32_t* w = (uint32_t*)malloc((size_t)g->n_words * 4);
        const char* doc = "{\"name\": \"Mario\", \"age\": 42, \"tags\": [\"a\", \"b\"]}";
        double tmax = 0, ttot = 0; int n = 0;
        for (const char* p = doc; *p; p++) {
            struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
            int cnt = llm_grammar_fill_mask(g, w);
            clock_gettime(CLOCK_MONOTONIC, &b);
            double ms = (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) * 1e-6; ttot += ms; n++; if (ms > tmax) tmax = ms;
            CHECK(cnt > 0, "fill_mask: 0 token ammessi");
            CHECK(llm_grammar_accept(g, (unsigned char)*p) == 0, "accept byte '%c'", *p);
        }
        printf("fill_mask su vocab 150k: media %.2f ms, max %.2f ms su %d passi (cache hit=%d miss=%d)\n", ttot / n, tmax, n, g->hits, g->misses);
        llm_grammar_free(g);
    }
    printf(fails ? "FAIL (%d)\n" : "PASS tutti\n", fails);
    return fails ? 1 : 0;
}
