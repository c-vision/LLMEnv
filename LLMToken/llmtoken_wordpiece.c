/* ============================================================================
 * llmtoken_wordpiece.c
 * Tokenizer BERT WordPiece nel runtime C condiviso — Fase 4b parita' C#/C.
 * Porting 1:1 di common/Databases/VectorDB/BertWordPieceTokenizer.cs.
 * Dati Unicode autosufficienti in llmtoken_wp_unicode_data.inc (generati da
 * tools/gen_wordpiece_unicode_data.py, Unicode 16.0.0 == runtime .NET 10).
 *
 * Punti di parita' chiave (cfr. la controparte C#):
 *   - BasicTokenize: clean (C* eccetto \t\n\r), Zs->spazio, NFD -> scarta Mn ->
 *     lowercase, split punteggiatura (P* o range ASCII, '_' incluso = Pc),
 *     isolamento ideogrammi CJK, parola >100 codepoint -> [UNK].
 *   - WordPiece: greedy longest-match, "##" per pezzi non in testa, se UN
 *     qualunque pezzo fallisce -> l'INTERA parola e' UN SOLO [UNK] (is_bad).
 *   - Encode/EncodePair: troncamento che preserva [CLS]/[SEP], token_type_ids
 *     assegnati DOPO il taglio, padding con id 0 e mask 0.
 * ============================================================================ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llmtoken_wordpiece.h"
#include "llmtoken_wp_unicode_data.inc"

#define LLMTOKEN_WP_MAX_INPUT_CHARS_PER_WORD 100

/* ---------------------------------------------------------------------------
 * UTF-8 decode / encode (schema meccanico, identico a llmtoken.c).
 * ------------------------------------------------------------------------- */
static uint32_t wp_utf8_next(const unsigned char* s, size_t len, size_t* i) {
    unsigned char c0 = s[*i];
    if (c0 < 0x80) { *i += 1; return c0; }
    else if ((c0 & 0xE0) == 0xC0 && *i + 1 < len) {
        uint32_t cp = ((uint32_t)(c0 & 0x1Fu) << 6) | (s[*i + 1] & 0x3Fu);
        *i += 2; return cp;
    } else if ((c0 & 0xF0) == 0xE0 && *i + 2 < len) {
        uint32_t cp = ((uint32_t)(c0 & 0x0Fu) << 12) | ((uint32_t)(s[*i + 1] & 0x3Fu) << 6) | (s[*i + 2] & 0x3Fu);
        *i += 3; return cp;
    } else if (*i + 3 < len) {
        uint32_t cp = ((uint32_t)(c0 & 0x07u) << 18) | ((uint32_t)(s[*i + 1] & 0x3Fu) << 12)
                      | ((uint32_t)(s[*i + 2] & 0x3Fu) << 6) | (s[*i + 3] & 0x3Fu);
        *i += 4; return cp;
    }
    *i += 1; return c0;
}

static int wp_utf8_encode(uint32_t cp, char* buf) {
    if (cp < 0x80) { buf[0] = (char)cp; return 1; }
    else if (cp < 0x800) {
        buf[0] = (char)(0xC0 | (cp >> 6)); buf[1] = (char)(0x80 | (cp & 0x3F)); return 2;
    } else if (cp < 0x10000) {
        buf[0] = (char)(0xE0 | (cp >> 12)); buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (cp & 0x3F)); return 3;
    } else {
        buf[0] = (char)(0xF0 | (cp >> 18)); buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); buf[3] = (char)(0x80 | (cp & 0x3F)); return 4;
    }
}

/* n byte di un codepoint UTF-8 (1-4), dato il primo byte. */
static int wp_utf8_len_of(unsigned char c0) {
    if (c0 < 0x80) return 1;
    else if ((c0 & 0xE0) == 0xC0) return 2;
    else if ((c0 & 0xF0) == 0xE0) return 3;
    else return 4;
}

/* ---------------------------------------------------------------------------
 * Struttura tokenizer: vocabolario (parola -> id), [CLS]/[SEP]/[UNK].
 * Hash semplice: FNV-1a 32-bit su stringa -> lista aperta di voci.
 * ------------------------------------------------------------------------- */
#define LLMTOKEN_WP_VOCAB_BUCKETS 4096

typedef struct wp_vocab_entry {
    char* word;
    int id;
    struct wp_vocab_entry* next;
} wp_vocab_entry;

struct llmtoken_wp_tok {
    wp_vocab_entry* buckets[LLMTOKEN_WP_VOCAB_BUCKETS];
    int cls_id, sep_id, unk_id, pad_id;
    size_t count;
};

static int wp_hash(const char* s) {
    uint32_t h = 2166136261u;
    for (; *s; s++) { h ^= (unsigned char)*s; h *= 16777619u; }
    return (int)(h & (LLMTOKEN_WP_VOCAB_BUCKETS - 1));
}

llmtoken_wp_tok* llmtoken_wp_tok_load(const char* vocab_path) {
    if (!vocab_path) return NULL;
    FILE* f = fopen(vocab_path, "r");
    if (!f) return NULL;

    llmtoken_wp_tok* t = (llmtoken_wp_tok*)calloc(1, sizeof(*t));
    if (!t) { fclose(f); return NULL; }
    t->cls_id = 101; t->sep_id = 102; t->unk_id = 100; t->pad_id = 0;

    char line[8192];
    int id = 0;
    while (fgets(line, sizeof(line), f)) {
        size_t n = strlen(line);
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
        if (n == 0) { /* vocab.txt di BERT non ha righe vuote; le saltiamo ma restano indicizzate come C# ? no: C# non salta */ }
        char* word = (char*)malloc(n + 1);
        if (!word) { llmtoken_wp_tok_free(t); fclose(f); return NULL; }
        memcpy(word, line, n + 1);
        int b = wp_hash(word);
        wp_vocab_entry* e = (wp_vocab_entry*)malloc(sizeof(*e));
        if (!e) { free(word); llmtoken_wp_tok_free(t); fclose(f); return NULL; }
        e->word = word; e->id = id; e->next = t->buckets[b];
        t->buckets[b] = e;
        t->count++;
        if (strcmp(word, "[CLS]") == 0) t->cls_id = id;
        else if (strcmp(word, "[SEP]") == 0) t->sep_id = id;
        else if (strcmp(word, "[UNK]") == 0) t->unk_id = id;
        id++;
    }
    fclose(f);
    if (t->count == 0) { llmtoken_wp_tok_free(t); return NULL; }
    return t;
}

void llmtoken_wp_tok_free(llmtoken_wp_tok* t) {
    if (!t) return;
    for (int i = 0; i < LLMTOKEN_WP_VOCAB_BUCKETS; i++) {
        wp_vocab_entry* e = t->buckets[i];
        while (e) { wp_vocab_entry* nx = e->next; free(e->word); free(e); e = nx; }
    }
    free(t);
}

int llmtoken_wp_tok_cls(const llmtoken_wp_tok* t) { return t ? t->cls_id : -1; }
int llmtoken_wp_tok_sep(const llmtoken_wp_tok* t) { return t ? t->sep_id : -1; }
int llmtoken_wp_tok_unk(const llmtoken_wp_tok* t) { return t ? t->unk_id : -1; }
int llmtoken_wp_tok_pad(const llmtoken_wp_tok* t) { return t ? t->pad_id : 0; }

static int wp_vocab_lookup(const llmtoken_wp_tok* t, const char* word) {
    for (wp_vocab_entry* e = t->buckets[wp_hash(word)]; e; e = e->next)
        if (strcmp(e->word, word) == 0) return e->id;
    return -1;
}

/* ---------------------------------------------------------------------------
 * Tabelle Unicode (intervalli) — lookup.
 * ------------------------------------------------------------------------- */
static int wp_in_ranges(uint32_t cp, const llmtoken_wp_range_t* ranges, unsigned n) {
    /* binary search */
    unsigned lo = 0, hi = n, mid;
    while (lo < hi) {
        mid = (lo + hi) >> 1;
        if (cp < ranges[mid].a) hi = mid;
        else if (cp > ranges[mid].b) lo = mid + 1;
        else return 1;
    }
    return 0;
}

#define WP_RANGE_IN(cp, arr) wp_in_ranges(cp, arr, sizeof(arr) / sizeof(arr[0]))

static int wp_is_clean(uint32_t cp) { return WP_RANGE_IN(cp, llmtoken_wp_clean_ranges); }
static int wp_is_space(uint32_t cp) { return WP_RANGE_IN(cp, llmtoken_wp_space_ranges); }
static int wp_is_mn(uint32_t cp) { return WP_RANGE_IN(cp, llmtoken_wp_mn_ranges); }
static int wp_is_punct(uint32_t cp) {
    /* range ASCII dei P* o categoria */
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126))
        return 1;
    return WP_RANGE_IN(cp, llmtoken_wp_punct_ranges);
}

/* Range HF _is_chinese_char (da C# IsCjkIdeograph). */
static int wp_is_cjk(uint32_t cp) {
    if ((cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
        (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
        (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
        (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F))
        return 1;
    return 0;
}

/* lookup punto-punto con binary search su array ordinato per cp. */
static int wp_nfd_find(uint32_t cp, unsigned* n_parts, const unsigned** parts, int* max_parts) {
    /* llmtoken_wp_nfd_map: row {cp, n, p0, p1, ...} , max columns LLMTOKEN_WP_NFD_MAX_PARTS */
    unsigned lo = 0, hi = LLMTOKEN_WP_NFD_MAP_COUNT, mid;
    while (lo < hi) {
        mid = (lo + hi) >> 1;
        unsigned key = llmtoken_wp_nfd_map[mid][0];
        if (cp < key) hi = mid;
        else if (cp > key) lo = mid + 1;
        else {
            *n_parts = llmtoken_wp_nfd_map[mid][1];
            *parts = &llmtoken_wp_nfd_map[mid][2];
            *max_parts = LLMTOKEN_WP_NFD_MAX_PARTS;
            return 1;
        }
    }
    return 0;
}

static int wp_lower_find(uint32_t cp, unsigned* n_parts, const unsigned** parts, int* max_parts) {
    unsigned lo = 0, hi = LLMTOKEN_WP_LOWER_MAP_COUNT, mid;
    while (lo < hi) {
        mid = (lo + hi) >> 1;
        unsigned key = llmtoken_wp_lower_map[mid][0];
        if (cp < key) hi = mid;
        else if (cp > key) lo = mid + 1;
        else {
            *n_parts = llmtoken_wp_lower_map[mid][1];
            *parts = &llmtoken_wp_lower_map[mid][2];
            *max_parts = LLMTOKEN_WP_LOWER_MAX_PARTS;
            return 1;
        }
    }
    return 0;
}

/* ---------------------------------------------------------------------------
 * Buffer di appoggio "string builder" (capacita' esplicita, no realloc spinto).
 * ------------------------------------------------------------------------- */
typedef struct {
    char* data;
    size_t len, cap;
} wp_sb;

static int wp_sb_reserve(wp_sb* b, size_t need) {
    if (b->cap >= need) return 0;
    size_t nc = b->cap ? b->cap * 2 : 64;
    while (nc < need) nc *= 2;
    char* nd = (char*)realloc(b->data, nc);
    if (!nd) return -1;
    b->data = nd; b->cap = nc;
    return 0;
}
static int wp_sb_putc(wp_sb* b, char c) {
    if (wp_sb_reserve(b, b->len + 1)) return -1;
    b->data[b->len++] = c;
    return 0;
}
static int wp_sb_utf8(wp_sb* b, uint32_t cp) {
    char buf[8]; int n = wp_utf8_encode(cp, buf);
    if (wp_sb_reserve(b, b->len + (size_t)n)) return -1;
    memcpy(b->data + b->len, buf, (size_t)n); b->len += (size_t)n;
    return 0;
}

/* ---------------------------------------------------------------------------
 * Decomposizione NFD di una stringa UTF-8: scrive in out gli id dei pezzi.
 * Ogni codepoint viene (a) scomposto secondo la mappa NFD canonica, oppure
 * (b) tenuto com'e'. Ritorna il numero di pezzi. 'pegs' buffer di size len*4.
 * ------------------------------------------------------------------------- */
static int wp_nfd_string(const unsigned char* s, size_t len, uint32_t* out, size_t out_cap) {
    size_t k = 0, i = 0;
    while (i < len) {
        size_t start = i;
        uint32_t cp = wp_utf8_next(s, len, &i);
        unsigned n = 0; const unsigned* parts = NULL; int mx = 0;
        if (wp_nfd_find(cp, &n, &parts, &mx)) {
            for (unsigned j = 0; j < n; j++) {
                if (k >= out_cap) return -1;
                out[k++] = parts[j];
            }
        } else {
            if (k >= out_cap) return -1;
            out[k++] = cp;
        }
        (void)start;
    }
    return (int)k;
}

/* conta codepoint UTF-8. */
static int wp_runes(const unsigned char* s, size_t len) {
    int n = 0; size_t i = 0;
    while (i < len) { wp_utf8_next(s, len, &i); n++; }
    return n;
}

/* ---------------------------------------------------------------------------
 * WordPiece greedy (una parola pre-tokenizzata). cb emette gli id dei pezzi.
 * ------------------------------------------------------------------------- */
int llmtoken_wp_wordpiece(const llmtoken_wp_tok* t, const char* token, llmtoken_wp_piece_cb cb, void* user) {
    if (!t || !token) return -1;
    size_t len = strlen(token);
    const unsigned char* s = (const unsigned char*)token;

    if (len == 0) return 0;

    /* HF: >100 codepoint -> [UNK] diretto */
    if (wp_runes(s, len) > LLMTOKEN_WP_MAX_INPUT_CHARS_PER_WORD) {
        if (cb) cb(t->unk_id, user);
        return 1;
    }

    int emitted = 0;
    size_t start = 0;

    /* Buffer dei pezzi matchati: NON li emettiamo finche' l'intera parola non ha
     * avuto successo. Se un qualunque segmento fallisce (is_bad), scartiamo tutto
     * quello accumulato ed emettiamo UN SOLO [UNK] — parita' col C#/HF
     * (is_bad -> return [unk_token]). Max 100 pezzi (una parola >100 codepoint e'
     * gia' [UNK] diretto sopra, e ogni pezzo contiene >=1 codepoint). */
    int pieces[LLMTOKEN_WP_MAX_INPUT_CHARS_PER_WORD];
    int n_pieces = 0;

    /* greedy longest-match: per ogni posizione, trova il vocab piu' lungo. */
    while (start < len) {
        size_t end = len;
        int found = -1;

        while (start < end) {
            /* costruisci pezzo UTF-8: [start, end) o con prefisso "##" */
            int is_head = (start == 0);
            size_t piece_len = (is_head ? 0 : 2) + (end - start);
            char* piece = (char*)malloc(piece_len + 1);
            if (!piece) return -1;
            size_t pl = 0;
            if (!is_head) { piece[0] = '#'; piece[1] = '#'; pl = 2; }
            memcpy(piece + pl, token + start, end - start);
            piece[pl + (end - start)] = '\0';

            int id = wp_vocab_lookup(t, piece);
            free(piece);

            if (id >= 0) { found = id; break; }
            /* retrocedi `end` di UN codepoint (in C# e' un code unit UTF-16 == 1
             * codepoint BMP; qui ragioniamo in byte UTF-8). */
            size_t p = start;
            while (p < end) {
                const unsigned char c0 = s[p];
                size_t cl = (size_t)wp_utf8_len_of(c0);
                if (p + cl >= end) break;   /* questo codepoint arriva a end: e' l'ultimo */
                p += cl;
            }
            end = p;
        }

        if (found < 0) {
            /* is_bad: l'INTERA parola e' [UNK], scarta i pezzi gia' bufferizzati */
            if (cb) cb(t->unk_id, user);
            return 1;
        }

        if (n_pieces >= LLMTOKEN_WP_MAX_INPUT_CHARS_PER_WORD) { pieces[0] = t->unk_id; n_pieces = 1; break; }
        pieces[n_pieces++] = found;
        start = end;
    }

    for (int i = 0; i < n_pieces; i++) if (cb) cb(pieces[i], user);
    return n_pieces;
}

/* ---------------------------------------------------------------------------
 * BasicTokenizer (fase di pre-tokenizzazione). cb riceve i token utf-8.
 * ------------------------------------------------------------------------- */
int llmtoken_wp_basic_tokenize(const char* text, llmtoken_wp_basic_cb cb, void* user) {
    if (!text) return -1;
    size_t len = strlen(text);
    const unsigned char* s = (const unsigned char*)text;

    /* (1)+(2) clean + isolamento CJK in un passaggio (per codepoint, surrogate-safe). */
    wp_sb cleaned = {0};
    size_t i = 0;
    while (i < len) {
        size_t cstart = i;
        uint32_t cp = wp_utf8_next(s, len, &i);

        if (cp == 0 || cp == 0xFFFD) continue;
        if (cp == '\t' || cp == '\n' || cp == '\r') { wp_sb_putc(&cleaned, ' '); continue; }

        if (wp_is_clean(cp)) continue;              /* categorie C* (escl. \t\n\r) */
        if (wp_is_space(cp)) { wp_sb_putc(&cleaned, ' '); continue; }

        if (wp_is_cjk(cp)) {
            wp_sb_putc(&cleaned, ' ');
            wp_sb_utf8(&cleaned, cp);
            wp_sb_putc(&cleaned, ' ');
        } else {
            /* copia il codepoint originale (puo' essere multi-byte) */
            for (size_t b = cstart; b < i; b++) wp_sb_putc(&cleaned, (char)s[b]);
        }
    }
    wp_sb_putc(&cleaned, '\0');
    cleaned.len--; /* rimuovi il byte NUL dalla lunghezza logica, non dalla cap */

    /* (3) split su whitespace + (4)(5) NFD->strip Mn->lower; split punteggiatura. */
    int emitted = 0;
    const char* p = cleaned.data;
    /* token di una parola (accumulatore per la punteggiatura) */
    wp_sb word = {0};

    size_t ci = 0;
    size_t clen = strlen(cleaned.data);
    while (ci < clen) {
        /* salta spazi di "separazione" (quelli che abbiamo introdotto noi come byte ' ') */
        if (cleaned.data[ci] == ' ') { ci++; continue; }

        /* una "ward" = [ci, ciEnd) fino al prossimo spazio */
        size_t wstart = ci;
        while (ci < clen && cleaned.data[ci] != ' ') ci++;
        const unsigned char* ws = (const unsigned char*)(cleaned.data + wstart);
        size_t wlen = ci - wstart;

        /* NFD */
        uint32_t nfd_buf[2048];
        int nn = wp_nfd_string(ws, wlen, nfd_buf, 2048);
        if (nn < 0) { free(cleaned.data); free(word.data); return -1; }

        /* strip Mn + lowercase, rispetto all'ordine C#: NFD -> strip Mn -> lower.
         * (In C# e' StripCombiningMarks(nfd).ToLowerInvariant()). */
        wp_sb norm = {0};
        for (int k = 0; k < nn; k++) {
            uint32_t cp = nfd_buf[k];
            if (wp_is_mn(cp)) continue;                 /* strip combining marks */
            /* lowercase */
            unsigned ln = 0; const unsigned* lp = NULL; int lmx = 0;
            if (wp_lower_find(cp, &ln, &lp, &lmx)) {
                for (unsigned j = 0; j < ln; j++) wp_sb_utf8(&norm, lp[j]);
            } else {
                wp_sb_utf8(&norm, cp);
            }
        }
        wp_sb_putc(&norm, '\0'); norm.len--;

        /* split punteggiatura: costruisce i token separando su IsPunctuation */
        size_t ni = 0;
        size_t nlen = strlen(norm.data);
        const unsigned char* ns = (const unsigned char*)norm.data;
        while (ni < nlen) {
            size_t cstart = ni;
            uint32_t cp = wp_utf8_next(ns, nlen, &ni);
            if (wp_is_punct(cp)) {
                if (word.len > 0) {
                    wp_sb_putc(&word, '\0'); word.len--;
                    if (cb && cb(word.data, user)) { free(cleaned.data); free(word.data); free(norm.data); return emitted; }
                    emitted++;
                    word.len = 0;
                }
                /* emetti il carattere di punteggiatura come token proprio */
                char pb[8]; int pn = wp_utf8_encode(cp, pb);
                /* reusiamo 'word' temporaneamente per la punteggiatura mono-char */
                for (int b = 0; b < pn; b++) wp_sb_putc(&word, pb[b]);
                wp_sb_putc(&word, '\0'); word.len -= (size_t)pn;
                if (cb && cb(word.data, user)) { free(cleaned.data); free(word.data); free(norm.data); return emitted; }
                emitted++;
                word.len = 0;
                (void)cstart;
            } else {
                for (size_t b = cstart; b < ni; b++) wp_sb_putc(&word, (char)ns[b]);
            }
        }
        if (word.len > 0) {
            wp_sb_putc(&word, '\0'); word.len--;
            if (cb && cb(word.data, user)) { free(cleaned.data); free(word.data); free(norm.data); return emitted; }
            emitted++;
            word.len = 0;
        }
        free(norm.data);
    }

    free(cleaned.data);
    free(word.data);
    return emitted;
}

/* ---------------------------------------------------------------------------
 * Encode / EncodePair con troncamento che preserva [CLS]/[SEP].
 * ------------------------------------------------------------------------- */

/* colleziona ids di una BasicTokenize+WordPiece into 'ids' (List<int> C#). */
typedef struct {
    int* data; size_t len, cap;
} wp_ids;

static int wp_ids_push(wp_ids* v, int x) {
    if (v->len >= v->cap) {
        size_t nc = v->cap ? v->cap * 2 : 64;
        int* nd = (int*)realloc(v->data, nc * sizeof(int));
        if (!nd) return -1;
        v->data = nd; v->cap = nc;
    }
    v->data[v->len++] = x;
    return 0;
}

typedef struct { wp_ids* ids; int piece; } wp_piece_ctx;

static int wp_piece_cb(int id, void* user) {
    wp_piece_ctx* c = (wp_piece_ctx*)user;
    return wp_ids_push(c->ids, id) ? 1 : 0;
}

typedef struct { llmtoken_wp_tok const* t; wp_ids* out; } wp_words_ctx;
static int wp_word_cb(const char* token, void* user) {
    wp_words_ctx* c = (wp_words_ctx*)user;
    int r = llmtoken_wp_wordpiece(c->t, token, wp_piece_cb, &(wp_piece_ctx){ c->out, 0 });
    return r < 0 ? 1 : 0;
}

static int wp_collect(const llmtoken_wp_tok* t, const char* text, wp_ids* out) {
    wp_words_ctx wc = { t, out };
    int r = llmtoken_wp_basic_tokenize(text, wp_word_cb, &wc);
    return r < 0 ? -1 : 0;
}

/* Padding C#: inputIds[i]=tokens[i], attentionMask[i]=1; il resto resta 0. */
static int wp_pad(wp_ids* tokens, int max_len, int32_t* input_ids, int32_t* attention_mask) {
    for (int i = 0; i < max_len; i++) {
        if ((size_t)i < tokens->len) { input_ids[i] = tokens->data[i]; attention_mask[i] = 1; }
        else { input_ids[i] = 0; attention_mask[i] = 0; }
    }
    return (int)tokens->len;
}

static int wp_ids_remove_range(wp_ids* v, int pos, int count) {
    memmove(v->data + pos, v->data + pos + count, (v->len - (size_t)pos - (size_t)count) * sizeof(int));
    v->len -= (size_t)count;
    return 0;
}

int llmtoken_wp_encode(const llmtoken_wp_tok* t, const char* text, int max_len,
                  int32_t* input_ids, int32_t* attention_mask, int32_t* token_type_ids) {
    if (!t || !text || max_len < 0 || !input_ids || !attention_mask) return -1;
    wp_ids tokens = {0};
    if (wp_ids_push(&tokens, t->cls_id)) { free(tokens.data); return -1; }
    if (wp_collect(t, text, &tokens) < 0) { free(tokens.data); return -1; }
    if (wp_ids_push(&tokens, t->sep_id)) { free(tokens.data); return -1; }

    if ((int)tokens.len > max_len) {
        if (max_len >= 2) {
            int keep = max_len - 2;
            wp_ids_remove_range(&tokens, 1 + keep, (int)tokens.len - 2 - keep);
        } else {
            wp_ids_remove_range(&tokens, max_len, (int)tokens.len - max_len);
        }
    }

    int active = wp_pad(&tokens, max_len, input_ids, attention_mask);
    if (token_type_ids) for (int i = 0; i < max_len; i++) token_type_ids[i] = 0;
    free(tokens.data);
    return active;
}

int llmtoken_wp_encode_pair(const llmtoken_wp_tok* t, const char* query, const char* document, int max_len,
                       int32_t* input_ids, int32_t* attention_mask, int32_t* token_type_ids) {
    if (!t || !query || !document || max_len < 0 || !input_ids || !attention_mask || !token_type_ids) return -1;

    wp_ids qt = {0}, dt = {0};
    int ok = 1;
    ok &= !wp_ids_push(&qt, t->cls_id);
    if (ok) { if (wp_collect(t, query, &qt) < 0) ok = 0; }
    if (ok) ok = !wp_ids_push(&qt, t->sep_id);

    if (ok) { if (wp_collect(t, document, &dt) < 0) ok = 0; }
    if (ok) ok = !wp_ids_push(&dt, t->sep_id);
    if (!ok) { free(qt.data); free(dt.data); return -1; }

    /* Budget del documento (incl. il suo [SEP]): taglia il contenuto, preserva [SEP] finale. */
    int maxDocTokens = (int)(max_len - qt.len); if (maxDocTokens < 1) maxDocTokens = 1;
    if ((int)dt.len > maxDocTokens)
        wp_ids_remove_range(&dt, maxDocTokens - 1, (int)dt.len - maxDocTokens);

    /* Query oltre il budget (doc minima): longest-first, preserva [CLS] e [SEP]. */
    int maxQueryTokens = (int)(max_len - dt.len); if (maxQueryTokens < 2) maxQueryTokens = 2;
    if ((int)qt.len > maxQueryTokens)
        wp_ids_remove_range(&qt, maxQueryTokens - 1, (int)qt.len - maxQueryTokens);

    int queryLen = (int)qt.len;
    wp_ids all = {0};
    for (int i = 0; i < (int)qt.len; i++) wp_ids_push(&all, qt.data[i]);
    for (int i = 0; i < (int)dt.len; i++) wp_ids_push(&all, dt.data[i]);
    if ((int)all.len > max_len)
        wp_ids_remove_range(&all, max_len, (int)all.len - max_len);   /* guardia */

    int active = wp_pad(&all, max_len, input_ids, attention_mask);
    for (int i = queryLen; i < (int)all.len; i++) token_type_ids[i] = 1;
    /* padding -> tipo 0 (default) */
    free(qt.data); free(dt.data); free(all.data);
    return active;
}
