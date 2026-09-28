/* llmengine_moe_json.c — implementazione dell'helper JSON minimale (vedi include/llmengine_moe_json.h). */
#include "llmengine_moe_json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* llmengine_moe_slurp(const char* path, size_t* n_out) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char* b = (char*)malloc((size_t)n + 1);
    if (!b) { fclose(f); return NULL; }
    if (n > 0 && fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return NULL; }
    b[n] = 0; fclose(f);
    if (n_out) *n_out = (size_t)n;
    return b;
}

static int is_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

const char* llmengine_moe_json_find(const char* s, const char* key) {
    if (!s || !key) return NULL;
    char pat[192];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    for (const char* p = s; (p = strstr(p, pat)) != NULL; p += strlen(pat)) {
        const char* q = p + strlen(pat);
        while (is_ws(*q)) ++q;
        if (*q == ':') { ++q; while (is_ws(*q)) ++q; return q; }
    }
    return NULL;
}

int llmengine_moe_json_int(const char* p, long long* out) {
    if (!p || !out) return 0;
    char* end = NULL;
    long long v = strtoll(p, &end, 10);
    if (end == p) return 0;
    *out = v; return 1;
}

int llmengine_moe_json_double(const char* p, double* out) {
    if (!p || !out) return 0;
    char* end = NULL;
    double v = strtod(p, &end);
    if (end == p) return 0;
    *out = v; return 1;
}

int llmengine_moe_json_str(const char* p, char* out, size_t cap) {
    if (!p || !out || cap == 0 || *p != '"') return 0;
    ++p; size_t i = 0;
    while (*p && *p != '"' && i + 1 < cap) out[i++] = *p++;
    out[i] = 0;
    return *p == '"';
}

int llmengine_moe_json_bool(const char* p, int* out) {
    if (!p || !out) return 0;
    if (strncmp(p, "true", 4) == 0)  { *out = 1; return 1; }
    if (strncmp(p, "false", 5) == 0) { *out = 0; return 1; }
    long long v;
    if (llmengine_moe_json_int(p, &v)) { *out = (v != 0); return 1; }
    return 0;
}

long long llmengine_moe_json_num(const char* json, const char* key, long long def) {
    long long v; return llmengine_moe_json_int(llmengine_moe_json_find(json, key), &v) ? v : def;
}
double llmengine_moe_json_flt(const char* json, const char* key, double def) {
    double v; return llmengine_moe_json_double(llmengine_moe_json_find(json, key), &v) ? v : def;
}
int llmengine_moe_json_flag(const char* json, const char* key, int def) {
    int v; return llmengine_moe_json_bool(llmengine_moe_json_find(json, key), &v) ? v : def;
}
