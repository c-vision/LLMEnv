/* ============================================================================
 * llmtoken.c -- implementazione, vedi llmtoken.h.
 *
 * Riscrittura in C puro (2026-08-24) della precedente implementazione C++
 * (std::string/std::vector/std::unordered_map + nlohmann::json). Stesso
 * identico algoritmo, stessa superficie pubblica (llmtoken.h non è
 * stata toccata). Vedi Fase0b-LlamaCpp-Wrapper-C-Rewrite-Plan.md per il piano
 * complessivo (questo è stato il primo file convertito: il più piccolo e
 * autonomo dei moduli AI nativi C++).
 *
 * Dipendenze vendorizzate: PCRE2 (già C, invariata), cJSON (sostituisce
 * nlohmann::json), sds (sostituisce std::string -- append ammortizzato O(1),
 * binary-safe). Hash table string->int32 scritta a mano qui sotto (open
 * addressing, FNV-1a) per vocab/merge_rank: la llmtoken_dict_t di llm_runtime
 * è una lista lineare (O(n) per lookup) e non è adatta a un vocabolario di
 * ~100k+ voci consultato ad ogni passo del merge BPE.
 * ============================================================================ */
#include "llmtoken.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <cJSON.h>
#include <sds.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --------------------------------------------------------------------------
 * StrI32Map: hash table string(chiave posseduta, sds) -> int32_t, open
 * addressing con linear probing, hash FNV-1a. Usata per vocab e merge_rank
 * (fino a ~100k-300k voci nella pratica, lookup ripetuti nel loop caldo del
 * merge BPE -- deve essere O(1) medio, non una scansione lineare).
 * -------------------------------------------------------------------------- */
typedef struct {
    sds key;       /* NULL = slot libero */
    int32_t value;
} StrI32Slot;

typedef struct {
    StrI32Slot* slots;
    size_t capacity; /* sempre potenza di 2 */
    size_t count;
} StrI32Map;

static uint64_t fnv1a(const char* data, size_t len) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= (unsigned char)data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static void strimap_init(StrI32Map* m, size_t initial_capacity_pow2) {
    m->capacity = initial_capacity_pow2;
    m->count = 0;
    m->slots = (StrI32Slot*)calloc(m->capacity, sizeof(StrI32Slot));
}

static void strimap_free(StrI32Map* m) {
    if (!m->slots) return;
    for (size_t i = 0; i < m->capacity; i++) {
        if (m->slots[i].key) sdsfree(m->slots[i].key);
    }
    free(m->slots);
    m->slots = NULL;
    m->capacity = 0;
    m->count = 0;
}

/* Inserisce senza duplicare la chiave (assume che non esista già -- vero per
 * come questo file popola vocab/merge_rank, un solo inserimento per chiave). */
static void strimap_put_raw(StrI32Map* m, sds owned_key, int32_t value);

static void strimap_grow(StrI32Map* m) {
    StrI32Slot* old_slots = m->slots;
    size_t old_cap = m->capacity;
    m->capacity = old_cap * 2;
    m->count = 0;
    m->slots = (StrI32Slot*)calloc(m->capacity, sizeof(StrI32Slot));
    for (size_t i = 0; i < old_cap; i++) {
        if (old_slots[i].key) strimap_put_raw(m, old_slots[i].key, old_slots[i].value);
    }
    free(old_slots);
}

static void strimap_put_raw(StrI32Map* m, sds owned_key, int32_t value) {
    if (m->count * 10 >= m->capacity * 7) strimap_grow(m); /* load factor 0.7 */
    size_t mask = m->capacity - 1;
    size_t idx = (size_t)fnv1a(owned_key, sdslen(owned_key)) & mask;
    while (m->slots[idx].key != NULL) {
        idx = (idx + 1) & mask;
    }
    m->slots[idx].key = owned_key;
    m->slots[idx].value = value;
    m->count++;
}

/* Copia key/key_len in una sds posseduta dalla mappa. */
static void strimap_put(StrI32Map* m, const char* key, size_t key_len, int32_t value) {
    strimap_put_raw(m, sdsnewlen(key, key_len), value);
}

/* Ritorna 1 se trovata (scrive *out_value), 0 altrimenti. */
static int strimap_get(const StrI32Map* m, const char* key, size_t key_len, int32_t* out_value) {
    if (m->capacity == 0) return 0;
    size_t mask = m->capacity - 1;
    size_t idx = (size_t)fnv1a(key, key_len) & mask;
    for (;;) {
        sds k = m->slots[idx].key;
        if (k == NULL) return 0;
        if (sdslen(k) == key_len && memcmp(k, key, key_len) == 0) {
            *out_value = m->slots[idx].value;
            return 1;
        }
        idx = (idx + 1) & mask;
    }
}

/* --------------------------------------------------------------------------
 * UTF-8 e mappa byte<->unicode (schema GPT-2/GPT-4, invariato dalla versione
 * C++: replica esatta di bytes_to_unicode() dell'encoder GPT-2 originale).
 * -------------------------------------------------------------------------- */

/* Scrive la codifica UTF-8 di cp in buf (>=4 byte), ritorna la lunghezza. */
static int utf8_encode_cp(uint32_t cp, char* buf) {
    if (cp < 0x80) {
        buf[0] = (char)cp;
        return 1;
    } else if (cp < 0x800) {
        buf[0] = (char)(0xC0 | (cp >> 6));
        buf[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp < 0x10000) {
        buf[0] = (char)(0xE0 | (cp >> 12));
        buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    } else {
        buf[0] = (char)(0xF0 | (cp >> 18));
        buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
}

/* Decodifica UN codepoint UTF-8 a partire da s[*i] (s ha lunghezza len),
 * avanza *i. Nessuna tabella Unicode necessaria -- solo lo schema meccanico
 * di codifica UTF-8. */
static uint32_t utf8_decode_next(const char* s, size_t len, size_t* i) {
    unsigned char c0 = (unsigned char)s[*i];
    if (c0 < 0x80) {
        *i += 1;
        return c0;
    } else if ((c0 & 0xE0) == 0xC0 && *i + 1 < len) {
        uint32_t cp = ((uint32_t)(c0 & 0x1Fu) << 6) | ((unsigned char)s[*i + 1] & 0x3Fu);
        *i += 2;
        return cp;
    } else if ((c0 & 0xF0) == 0xE0 && *i + 2 < len) {
        uint32_t cp = ((uint32_t)(c0 & 0x0Fu) << 12) | (((unsigned char)s[*i + 1] & 0x3Fu) << 6) |
                      ((unsigned char)s[*i + 2] & 0x3Fu);
        *i += 3;
        return cp;
    } else if (*i + 3 < len) {
        uint32_t cp = ((uint32_t)(c0 & 0x07u) << 18) | (((unsigned char)s[*i + 1] & 0x3Fu) << 12) |
                      (((unsigned char)s[*i + 2] & 0x3Fu) << 6) | ((unsigned char)s[*i + 3] & 0x3Fu);
        *i += 4;
        return cp;
    }
    *i += 1;
    return c0; /* byte malformato isolato -- pass-through, non dovrebbe capitare su UTF-8 valido */
}

/* out[256][5]: ogni voce è NUL-terminated, lunghezza reale in out_len[256]. */
static void build_byte_to_unicode(char out[256][5], int out_len[256]) {
    int in_bs[256] = {0};
    for (int i = (int)'!'; i <= (int)'~'; i++) in_bs[i] = 1;
    for (int i = 0xA1; i <= 0xAC; i++) in_bs[i] = 1;
    for (int i = 0xAE; i <= 0xFF; i++) in_bs[i] = 1;

    int byte_to_cp[256];
    for (int b = 0; b < 256; b++) byte_to_cp[b] = in_bs[b] ? b : -1;

    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (byte_to_cp[b] < 0) {
            byte_to_cp[b] = 256 + n;
            n++;
        }
    }

    for (int b = 0; b < 256; b++) {
        int len = utf8_encode_cp((uint32_t)byte_to_cp[b], out[b]);
        out[b][len] = '\0';
        out_len[b] = len;
    }
}

/* Scrive a + separatore ('\x1F', mai prodotto da build_byte_to_unicode) + b
 * in buf (capacità buf_cap); ritorna la lunghezza totale (troncata se non
 * capisse -- non dovrebbe mai accadere per pezzi realistici, i simboli BPE
 * restano corti anche dopo molti merge su una singola parola). */
static size_t make_pair_key(char* buf, size_t buf_cap, const char* a, size_t alen, const char* b, size_t blen) {
    size_t total = alen + 1 + blen;
    if (total > buf_cap) total = buf_cap;
    size_t pa = alen < buf_cap ? alen : buf_cap;
    memcpy(buf, a, pa);
    if (pa < buf_cap) { buf[pa] = '\x1F'; pa++; }
    size_t remaining = buf_cap > pa ? buf_cap - pa : 0;
    size_t pb = blen < remaining ? blen : remaining;
    if (pb > 0) memcpy(buf + pa, b, pb);
    return total;
}

/* --------------------------------------------------------------------------
 * Struttura tokenizer e caricamento
 * -------------------------------------------------------------------------- */

typedef struct {
    sds content;
    int32_t id;
} SpecialToken;

/* Passo di normalizzazione dichiarato dal file (sezione "normalizer"). Qui
 * stanno SOLO i due tipi che l'encoder metaspace serve davvero: Prepend (il
 * marcatore di inizio parola) e Replace (lo spazio -> marcatore). `from`/`to`
 * stanno in 8 byte: i marcatori sono UTF-8 corti (U+2581 = 3 byte, spazio = 1). */
#define NORM_STR_MAX 8
typedef struct {
    int kind;                          /* 1 = Prepend, 2 = Replace */
    char from[NORM_STR_MAX];
    size_t from_len;
    char to[NORM_STR_MAX];
    size_t to_len;
} NormStep;

typedef struct {
    StrI32Map vocab;             /* stringa byte-unicode -> id */
    sds* id_to_token;            /* id -> stringa byte-unicode (NULL se non in vocab) */
    size_t id_to_token_len;
    StrI32Map merge_rank;        /* pair_key(left,right) -> rango (minore = priorita' maggiore) */
    sds* merges;                 /* merges in ORDINE di rango ("left right"), per l'export GGUF */
    size_t merges_len, merges_cap;
    const char** export_tokens;  /* cache id->stringa densa (buchi riempiti coi speciali) per l'export */
    size_t export_tokens_len;
    char byte_to_unicode[256][5];
    int byte_to_unicode_len[256];
    uint32_t unicode_cp[256];    /* per il decode: cp -> byte (inverso di byte_to_unicode) */
    uint8_t unicode_byte[256];
    int unicode_map_count;
    SpecialToken* special_tokens; /* ordinati per lunghezza contenuto decrescente */
    size_t special_tokens_len;
    /* id -> contenuto token speciale (lista lineare piccola, non serve hash) */
    int32_t* special_ids;
    sds* special_contents;
    size_t special_id_map_len;
    pcre2_code* pre_re[4];       /* regex Split del pre-tokenizer, in ordine (Sequence) */
    size_t pre_re_len;
    /* --- encoder metaspace/SentencePiece (Llama-1/2, Mistral/Mixtral) vs
     * byte-level (GPT-2/Llama-3/Qwen/GLM). La famiglia la dice il vocabolario,
     * non una supposizione: il byte-level contiene lo spazio mirrorato "Ġ", il
     * metaspace no ma contiene il marcatore del normalizer. Misurato:
     * Mixtral-8x22B voci=32000, Ġ assente, "▁" presente; GLM-4.5-Air (151329) e
     * Qwen3-Coder-30B (151643) l'opposto. -- */
    int metaspace;               /* 1 = i simboli iniziali sono CARATTERI UTF-8 */
    NormStep norm[4];
    size_t norm_len;
    char ws_marker[NORM_STR_MAX]; /* marcatore di parola (normalizer: "▁") -> spazio in decode */
    size_t ws_marker_len;
    int byte_fallback;           /* model.byte_fallback: carattere fuori vocab -> <0xHH> */
    sds unk_token;               /* model.unk_token, se presente (fallback finale) */
} LlmTokenizer;

/* bpe_merge: unisce iterativamente la coppia adiacente col rango più basso,
 * finché non ce ne sono più. word/word_len in-out (array di sds, capacità
 * word_cap sufficiente -- il merge non fa mai crescere il conteggio di
 * simboli, solo l'algoritmo classico BPE). Ritorna la nuova lunghezza. */
static size_t bpe_merge(const LlmTokenizer* t, sds* word, size_t word_len) {
    if (word_len < 2) return word_len;
    char key_buf[512];
    for (;;) {
        int32_t best_rank = INT32_MAX;
        size_t best_i = (size_t)-1;
        int found = 0;
        for (size_t i = 0; i + 1 < word_len; i++) {
            size_t klen = make_pair_key(key_buf, sizeof(key_buf), word[i], sdslen(word[i]), word[i + 1], sdslen(word[i + 1]));
            int32_t rank;
            if (strimap_get(&t->merge_rank, key_buf, klen, &rank) && rank < best_rank) {
                best_rank = rank;
                best_i = i;
                found = 1;
            }
        }
        if (!found) break;

        /* Fonde OGNI occorrenza non sovrapposta della coppia vincente in un
         * solo passaggio sinistra->destra -- stesso comportamento
         * dell'originale C++ (e di GPT-2/HuggingFace): non solo la prima
         * occorrenza trovata. Bug reale trovato con un test a 3+ merge
         * concatenati (2026-08-24): una prima versione fondeva solo
         * l'occorrenza a best_i, producendo una tokenizzazione diversa da
         * quella attesa su parole con la stessa coppia ripetuta più volte.
         * win_first/win_second sono copie possedute (sdsdup), non alias su
         * word[best_i]/word[best_i+1]: quegli slot vengono liberati durante
         * questo stesso passaggio quando i raggiunge best_i, quindi un alias
         * sarebbe un puntatore pendente per i confronti successivi. */
        sds win_first = sdsdup(word[best_i]);
        sds win_second = sdsdup(word[best_i + 1]);
        size_t win_first_len = sdslen(win_first);
        size_t win_second_len = sdslen(win_second);
        size_t new_len = 0;
        size_t i = 0;
        while (i < word_len) {
            if (i + 1 < word_len &&
                sdslen(word[i]) == win_first_len && memcmp(word[i], win_first, win_first_len) == 0 &&
                sdslen(word[i + 1]) == win_second_len && memcmp(word[i + 1], win_second, win_second_len) == 0) {
                sds merged = sdsdup(word[i]);
                merged = sdscatsds(merged, word[i + 1]);
                sdsfree(word[i]);
                sdsfree(word[i + 1]);
                word[new_len++] = merged;
                i += 2;
            } else {
                word[new_len++] = word[i];
                i += 1;
            }
        }
        sdsfree(win_first);
        sdsfree(win_second);
        word_len = new_len;
        if (word_len == 1) break;
    }
    return word_len;
}

/* Applica i passi di normalizzazione dichiarati dal file, in ordine (metaspace:
 * Prepend del marcatore + Replace dello spazio col marcatore).
 * Solo il percorso metaspace la chiama: il byte-level non ha normalizer da
 * servire (la sua trasposizione byte->unicode e' dentro encode_piece). */
static sds normalize_metaspace(const LlmTokenizer* t, const char* text, size_t text_len) {
    sds cur = sdsnewlen(text, text_len);
    for (size_t s = 0; s < t->norm_len; s++) {
        const NormStep* st = &t->norm[s];
        sds out = sdsempty();
        if (st->kind == 1) {                                  /* Prepend */
            out = sdscatlen(out, st->to, st->to_len);
            out = sdscatlen(out, cur, sdslen(cur));
        } else {                                              /* Replace */
            size_t i = 0;
            while (i < sdslen(cur)) {
                if (st->from_len && i + st->from_len <= sdslen(cur)
                    && memcmp(cur + i, st->from, st->from_len) == 0) {
                    out = sdscatlen(out, st->to, st->to_len);
                    i += st->from_len;
                } else {
                    out = sdscatlen(out, cur + i, 1);
                    i++;
                }
            }
        }
        sdsfree(cur);
        cur = out;
    }
    return cur;
}

/* push_piece: accoda un sds a un array dinamico (capacita' potenza di 2). */
static void push_piece(sds** arr, size_t* len, size_t* cap, sds v) {
    if (*len >= *cap) {
        *cap = (*cap == 0) ? 16 : (*cap * 2);
        *arr = (sds*)realloc(*arr, (*cap) * sizeof(sds));
    }
    (*arr)[(*len)++] = v;
}

/* Validatore UTF-8 O(n): se il buffer e' valido si puo' usare PCRE2_NO_UTF_CHECK (pcre2_match
 * altrimenti rivalida l'intero subject ad ogni chiamata -> O(n^2) sul pre-tokenizer). */
static int llmtoken_utf8_valid(const unsigned char* p, size_t n) {
    size_t i = 0;
    while (i < n) {
        unsigned char c = p[i];
        if (c < 0x80) { i++; continue; }
        int len = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC2) ? 2 : 0;
        if (len == 0 || i + (size_t)len > n) return 0;
        for (int j = 1; j < len; j++) if ((p[i + (size_t)j] & 0xC0) != 0x80) return 0;
        i += (size_t)len;
    }
    return 1;
}

/* pretokenize: split via il pattern regex Unicode-aware del pre-tokenizer
 * (PCRE2, comportamento "Isolated": i pezzi sono le porzioni matchate
 * stesse). Scrive i pezzi (sds posseduti dal chiamante) in *out_pieces,
 * *out_len; il chiamante alloca/rialloca l'array. Ritorna 0 su errore. */
static int pretokenize(const LlmTokenizer* t, const char* text, size_t text_len, sds** out_pieces, size_t* out_len, size_t* out_cap) {
    if (text_len == 0) return 1;
    if (t->metaspace) {
        /* Nessuna regex: il confine di parola lo porta il marcatore, quindi il
         * segmento normalizzato intero e' UN pezzo BPE (esattamente cio' che fa
         * tokenizers con pre_tokenizer assente). */
        sds norm = normalize_metaspace(t, text, text_len);
        if (*out_len >= *out_cap) {
            *out_cap = (*out_cap == 0) ? 16 : (*out_cap * 2);
            *out_pieces = (sds*)realloc(*out_pieces, (*out_cap) * sizeof(sds));
        }
        (*out_pieces)[(*out_len)++] = norm;
        return 1;
    }
    /* Nessuna regex: un solo pezzo (tutto il testo). */
    if (t->pre_re_len == 0) {
        push_piece(out_pieces, out_len, out_cap, sdsnewlen(text, text_len));
        return 1;
    }
    /* Applica le regex Split in SEQUENZA (pre-tokenizer type=Sequence):
     * comportamento "Isolated" = ogni match E ogni gap sono pezzi, in ordine. */
    sds* cur = (sds*)malloc(sizeof(sds));
    size_t cur_len = 1, cur_cap = 1;
    cur[0] = sdsnewlen(text, text_len);
    for (size_t k = 0; k < t->pre_re_len; k++) {
        pcre2_match_data* md = pcre2_match_data_create_from_pattern(t->pre_re[k], NULL);
        if (!md) { for (size_t i = 0; i < cur_len; i++) sdsfree(cur[i]); free(cur); return 0; }
        sds* nxt = NULL; size_t nxt_len = 0, nxt_cap = 0;
        for (size_t i = 0; i < cur_len; i++) {
            const char* buf = cur[i]; size_t len = sdslen(cur[i]);
            PCRE2_SIZE offset = 0;
            /* PERFORMANCE (2026-10-01): senza PCRE2_NO_UTF_CHECK, pcre2_match rivalida l'UTF-8
             * di TUTTO il subject a OGNI match; con l'offset che avanza e' O(len^2) (a 15MB=2.5M
             * token: >64 min CPU). Validiamo l'UTF-8 UNA volta (O(len)) e poi saltiamo il check. */
            const int mopts = llmtoken_utf8_valid((const unsigned char*)buf, len) ? PCRE2_NO_UTF_CHECK : 0;
            while (offset < len) {
                int rc = pcre2_match(t->pre_re[k], (PCRE2_SPTR)buf, len, offset, mopts, md, NULL);
                if (rc < 0) break;
                PCRE2_SIZE* ov = pcre2_get_ovector_pointer(md);
                PCRE2_SIZE ms = ov[0], me = ov[1];
                if (ms > offset) push_piece(&nxt, &nxt_len, &nxt_cap, sdsnewlen(buf + offset, ms - offset));
                if (me > ms) push_piece(&nxt, &nxt_len, &nxt_cap, sdsnewlen(buf + ms, me - ms));
                offset = (me > offset) ? me : offset + 1; /* guardia anti-loop su match a lunghezza zero */
            }
            if (offset < len) push_piece(&nxt, &nxt_len, &nxt_cap, sdsnewlen(buf + offset, len - offset));
        }
        pcre2_match_data_free(md);
        for (size_t i = 0; i < cur_len; i++) sdsfree(cur[i]);
        free(cur);
        cur = nxt; cur_len = nxt_len; cur_cap = nxt_cap;
        if (cur_len == 0) { cur = (sds*)malloc(sizeof(sds)); cur_len = 0; cur_cap = 0; }
    }
    for (size_t i = 0; i < cur_len; i++) push_piece(out_pieces, out_len, out_cap, cur[i]);
    free(cur);
    return 1;
}

/* encode_piece: appende gli id prodotti per un singolo pezzo pretokenizzato
 * a *out_ids e *out_len (array gestito dal chiamante). Ritorna 0 su errore
 * (bug reale: byte-level BPE non dovrebbe mai fallire il lookup vocab). */
static int encode_piece(const LlmTokenizer* t, const char* piece, size_t piece_len, int32_t** out_ids, size_t* out_len, size_t* out_cap) {
    sds* symbols = (sds*)malloc(piece_len * sizeof(sds));
    size_t n_symbols = 0;
    int build_ok = 1;
    if (t->metaspace) {
        /* Metaspace: i simboli iniziali sono i CARATTERI UTF-8, non i byte
         * trasposti. Un carattere che non sta nel vocabolario viene espanso nei
         * suoi byte scritti come <0xHH> se il file dichiara byte_fallback (e' il
         * simmetrico del decoder ByteFallback della sezione "decoder"), altrimenti
         * diventa il token unk di model.unk_token; senza nessuno dei due e' un
         * errore vero, non un default inventato. */
        size_t p = 0;
        while (p < piece_len) {
            size_t start = p;
            utf8_decode_next(piece, piece_len, &p);
            sds ch = sdsnewlen(piece + start, p - start);
            int32_t probe;
            if (strimap_get(&t->vocab, ch, sdslen(ch), &probe)) {
                symbols[n_symbols++] = ch;
            } else if (t->byte_fallback) {
                sdsfree(ch);
                for (size_t q = start; q < p; q++) {
                    char hex[8];
                    int hlen = snprintf(hex, sizeof hex, "<0x%02X>", (unsigned char)piece[q]);
                    symbols[n_symbols++] = sdsnewlen(hex, (size_t)(hlen > 0 ? hlen : 0));
                }
            } else if (t->unk_token) {
                sdsfree(ch);
                symbols[n_symbols++] = sdsdup(t->unk_token);
            } else {
                sdsfree(ch);
                build_ok = 0;
                break;
            }
        }
    } else {
        for (size_t i = 0; i < piece_len; i++) {
            unsigned char b = (unsigned char)piece[i];
            symbols[n_symbols++] = sdsnewlen(t->byte_to_unicode[b], (size_t)t->byte_to_unicode_len[b]);
        }
    }
    if (!build_ok) {
        for (size_t i = 0; i < n_symbols; i++) sdsfree(symbols[i]);
        free(symbols);
        return 0;
    }
    size_t merged_len = bpe_merge(t, symbols, n_symbols);

    int ok = 1;
    for (size_t i = 0; i < merged_len; i++) {
        int32_t id;
        if (!strimap_get(&t->vocab, symbols[i], sdslen(symbols[i]), &id)) { ok = 0; break; }
        if (*out_len >= *out_cap) {
            *out_cap = (*out_cap == 0) ? 64 : (*out_cap * 2);
            *out_ids = (int32_t*)realloc(*out_ids, (*out_cap) * sizeof(int32_t));
        }
        (*out_ids)[(*out_len)++] = id;
    }
    /* Libera SOLO [0, merged_len): bpe_merge compatta in-place spostando i
     * puntatori verso il basso senza azzerare gli slot residui oltre
     * merged_len -- quegli slot sono duplicati di puntatori già liberati (o
     * ancora vivi in [0, merged_len)) durante il merge; liberarli di nuovo
     * qui sarebbe un double-free (bug reale trovato con lldb, 2026-08-24:
     * SIGTRAP in libsystem_malloc dentro encode_piece). */
    for (size_t i = 0; i < merged_len; i++) sdsfree(symbols[i]);
    free(symbols);
    return ok;
}

static int special_token_cmp(const void* pa, const void* pb) {
    const SpecialToken* a = (const SpecialToken*)pa;
    const SpecialToken* b = (const SpecialToken*)pb;
    size_t la = sdslen(a->content), lb = sdslen(b->content);
    if (la != lb) return la > lb ? -1 : 1; /* lunghezza decrescente */
    return 0;
}

/* Registra un passo del normalizer del file (Prepend o Replace con pattern
 * letterale). Gli altri tipi (Nfkd, BertNormalizer, Replace con pattern Regex)
 * NON vengono registrati: il percorso metaspace ha bisogno solo di questi due, e
 * un normalizer che chiede altro resta ignorato come e' stato ignorato finora --
 * non e' una promessa di fedelta' per quei passi, e' il gap gia' documentato
 * nell'header per la NFC. Ritorna senza aggiungere nulla se la stringa non ci
 * sta nel buffer fisso (passo troppo lungo per l'encoder che serviamo). */
static void parse_norm_step(cJSON* step, LlmTokenizer* t) {
    if (t->norm_len >= sizeof(t->norm) / sizeof(t->norm[0])) return;
    cJSON* ty = cJSON_GetObjectItemCaseSensitive(step, "type");
    if (!cJSON_IsString(ty)) return;
    cJSON* src = NULL;
    cJSON* dst = NULL;
    NormStep ns;
    memset(&ns, 0, sizeof ns);
    if (strcmp(ty->valuestring, "Prepend") == 0) {
        ns.kind = 1;
        dst = cJSON_GetObjectItemCaseSensitive(step, "prepend");
    } else if (strcmp(ty->valuestring, "Replace") == 0) {
        cJSON* pat = cJSON_GetObjectItemCaseSensitive(step, "pattern");
        ns.kind = 2;
        src = pat ? cJSON_GetObjectItemCaseSensitive(pat, "String") : NULL;
        dst = cJSON_GetObjectItemCaseSensitive(step, "content");
    } else {
        return;
    }
    if (!cJSON_IsString(dst)) return;
    size_t dlen = strlen(dst->valuestring);
    if (dlen > NORM_STR_MAX - 1) return;
    memcpy(ns.to, dst->valuestring, dlen);
    ns.to_len = dlen;
    if (ns.kind == 2) {
        if (!cJSON_IsString(src)) return;
        size_t slen = strlen(src->valuestring);
        if (slen > NORM_STR_MAX - 1) return;
        memcpy(ns.from, src->valuestring, slen);
        ns.from_len = slen;
    }
    t->norm[t->norm_len++] = ns;
}

static llmtoken_handle llmtoken_build_json(const char* json, size_t len) {
    cJSON* j = cJSON_ParseWithLength(json, len);
    if (!j) {
        fprintf(stderr, "llmtoken: JSON malformato\n");
        return NULL;
    }

    cJSON* model = cJSON_GetObjectItemCaseSensitive(j, "model");
    cJSON* model_type = model ? cJSON_GetObjectItemCaseSensitive(model, "type") : NULL;
    if (!model || !cJSON_IsString(model_type) || strcmp(model_type->valuestring, "BPE") != 0) {
        fprintf(stderr, "llmtoken_load: model.type non e' \"BPE\" -- schema non supportato\n");
        cJSON_Delete(j);
        return NULL;
    }
    cJSON* pre_tokenizer = cJSON_GetObjectItemCaseSensitive(j, "pre_tokenizer");
    /* NON e' un errore se manca o e' null: un checkpoint SentencePiece (Mixtral) non ha
     * nessuna regex di split, perche' il confine di parola lo porta gia' il marcatore del
     * normalizer. Nessuna guardia qui: get_object_item() del cJSON in _deps restituisce NULL
     * se l'oggetto e' NULL, quindi il pattern sotto non si trova e il file viene comunque
     * accettato SE l'encoder e' metaspace; altrimenti resta l'errore originale. */

    LlmTokenizer* t = (LlmTokenizer*)calloc(1, sizeof(LlmTokenizer));
    build_byte_to_unicode(t->byte_to_unicode, t->byte_to_unicode_len);
    strimap_init(&t->vocab, 1 << 17);       /* ~131k slot iniziali, load factor 0.7 -> ~91k voci senza resize */
    strimap_init(&t->merge_rank, 1 << 18);  /* i merge sono tipicamente più numerosi del vocab */

    /* vocab */
    int32_t max_id = -1;
    cJSON* vocab_obj = cJSON_GetObjectItemCaseSensitive(model, "vocab");
    cJSON* entry;
    cJSON_ArrayForEach(entry, vocab_obj) {
        if (!cJSON_IsNumber(entry)) continue;
        int32_t id = (int32_t)entry->valuedouble;
        strimap_put(&t->vocab, entry->string, strlen(entry->string), id);
        if (id > max_id) max_id = id;
    }

    /* merges -- formato confermato dal file reale: lista di coppie [left,right]
     * o, in alcuni tokenizer.json, stringhe "left right". */
    cJSON* merges = cJSON_GetObjectItemCaseSensitive(model, "merges");
    int32_t merge_idx = 0;
    cJSON* m;
    cJSON_ArrayForEach(m, merges) {
        char key_buf[512];
        size_t klen;
        sds merged = NULL;
        if (cJSON_IsArray(m) && cJSON_GetArraySize(m) == 2) {
            cJSON* left = cJSON_GetArrayItem(m, 0);
            cJSON* right = cJSON_GetArrayItem(m, 1);
            if (!cJSON_IsString(left) || !cJSON_IsString(right)) { merge_idx++; continue; }
            klen = make_pair_key(key_buf, sizeof(key_buf), left->valuestring, strlen(left->valuestring),
                                  right->valuestring, strlen(right->valuestring));
            merged = sdscat(sdscat(sdsnew(left->valuestring), " "), right->valuestring);
        } else if (cJSON_IsString(m)) {
            const char* s = m->valuestring;
            const char* sp = strchr(s, ' ');
            if (!sp) { merge_idx++; continue; }
            klen = make_pair_key(key_buf, sizeof(key_buf), s, (size_t)(sp - s), sp + 1, strlen(sp + 1));
            merged = sdsnew(s);
        } else {
            merge_idx++;
            continue;
        }
        strimap_put(&t->merge_rank, key_buf, klen, merge_idx);
        if (t->merges_len == t->merges_cap) {
            t->merges_cap = t->merges_cap ? t->merges_cap * 2 : 4096;
            t->merges = (sds*)realloc(t->merges, t->merges_cap * sizeof(sds));
        }
        t->merges[t->merges_len++] = merged;
        merge_idx++;
    }

    /* added_tokens (token speciali -- match letterale, non passano per BPE) */
    cJSON* added_tokens = cJSON_GetObjectItemCaseSensitive(j, "added_tokens");
    size_t n_added = added_tokens ? (size_t)cJSON_GetArraySize(added_tokens) : 0;
    t->special_tokens = n_added ? (SpecialToken*)malloc(n_added * sizeof(SpecialToken)) : NULL;
    t->special_ids = n_added ? (int32_t*)malloc(n_added * sizeof(int32_t)) : NULL;
    t->special_contents = n_added ? (sds*)malloc(n_added * sizeof(sds)) : NULL;
    size_t n_special = 0;
    if (added_tokens) {
        cJSON* at;
        cJSON_ArrayForEach(at, added_tokens) {
            cJSON* content_j = cJSON_GetObjectItemCaseSensitive(at, "content");
            cJSON* id_j = cJSON_GetObjectItemCaseSensitive(at, "id");
            const char* content = cJSON_IsString(content_j) ? content_j->valuestring : "";
            int32_t id = cJSON_IsNumber(id_j) ? (int32_t)id_j->valuedouble : -1;
            if (content[0] == '\0' || id < 0) continue;
            sds content_sds = sdsnew(content);
            t->special_tokens[n_special].content = content_sds;
            t->special_tokens[n_special].id = id;
            t->special_ids[n_special] = id;
            t->special_contents[n_special] = sdsdup(content_sds);
            n_special++;
            if (id > max_id) max_id = id;
        }
        qsort(t->special_tokens, n_special, sizeof(SpecialToken), special_token_cmp);
    }
    t->special_tokens_len = n_special;
    t->special_id_map_len = n_special;

    /* id_to_token per il decode */
    t->id_to_token_len = (size_t)max_id + 1;
    t->id_to_token = (sds*)calloc(t->id_to_token_len, sizeof(sds));
    for (size_t i = 0; i < t->vocab.capacity; i++) {
        if (t->vocab.slots[i].key) {
            int32_t id = t->vocab.slots[i].value;
            if ((size_t)id < t->id_to_token_len) t->id_to_token[id] = sdsdup(t->vocab.slots[i].key);
        }
    }

    /* inverso byte<->unicode per il decode (al massimo 256 voci, scansione
     * lineare in decode è accettabile: non è il loop caldo, a differenza di
     * vocab/merge_rank durante l'encode). */
    t->unicode_map_count = 0;
    for (int b = 0; b < 256; b++) {
        size_t dummy = 0;
        uint32_t cp = utf8_decode_next(t->byte_to_unicode[b], (size_t)t->byte_to_unicode_len[b], &dummy);
        t->unicode_cp[t->unicode_map_count] = cp;
        t->unicode_byte[t->unicode_map_count] = (uint8_t)b;
        t->unicode_map_count++;
    }

    /* ------------------------------------------------------------------------
     * Famiglia di encoder: la dice il VOCABOLARIO, non una supposizione sul
     * modello. Byte-level (GPT-2/Llama-3/Qwen/GLM) ha il gruppo di token che
     * inizia con lo spazio "mirrorato" Ġ che DOMINA il vocab, e i suoi simboli
     * iniziali sono byte trasposti; metaspace/SentencePiece (Llama-1/2,
     * Mistral/Mixtral, gemma) usa il marcatore di parola dichiarato dal
     * normalizer e ha il gruppo ▁-prefissato che domina. Non basta "Ġ esiste":
     * gemma-4 (262144 voci) ha UN solo Ġ spurio contro 137541 token ▁ — decide
     * la dominanza del gruppo, non la presenza. Misurato sui checkpoint in casa:
     * Mixtral-8x22B voci=32000, Ġ ASSENTE, "▁" presente, pre_tokenizer null,
     * normalizer Sequence[Prepend("▁"), Replace(" "->"▁")], byte_fallback true,
     * unk_token "<unk>"; gemma-4-31b, normalizer Replace(" "->"▁") + pre_tokenizer
     * Split su " " (letterale, non regex) con MergedWithPrevious: il marcatore
     * ▁ domina il vocab; GLM-4.5-Air (151329) e Qwen3-Coder-30B (151643)
     * l'opposto (gruppo Ġ dominante, decoder ByteLevel, pre_tokenizer Regex).
     * ------------------------------------------------------------------------ */
    cJSON* norm_j = cJSON_GetObjectItemCaseSensitive(j, "normalizer");
    cJSON* norm_type = norm_j ? cJSON_GetObjectItemCaseSensitive(norm_j, "type") : NULL;
    if (cJSON_IsString(norm_type) && strcmp(norm_type->valuestring, "Sequence") == 0) {
        cJSON* steps = cJSON_GetObjectItemCaseSensitive(norm_j, "normalizers");
        cJSON* st;
        cJSON_ArrayForEach(st, steps) parse_norm_step(st, t);
    } else if (norm_type) {
        parse_norm_step(norm_j, t);
    }
    for (size_t s = 0; s < t->norm_len; s++) {
        if (t->norm[s].to_len && (t->norm[s].kind == 1 || t->norm[s].from[0] == ' ')) {
            memcpy(t->ws_marker, t->norm[s].to, t->norm[s].to_len);
            t->ws_marker_len = t->norm[s].to_len;
            break;
        }
    }
    {
        int32_t probe;
        static const char bytelevel_space[] = "\xC4\xA0";   /* U+0120 "Ġ" in UTF-8 */
        /* Dominanza, non presenza. Un vocab byte-level (GPT-2/Llama-3/Qwen/GLM)
         * ha DECINE DI MIGLIAIA di token che iniziano con Ġ; uno metaspace
         * (Mixtral/gemma) ne ha ~zero e usa il marcatore di parola (▁) che gli
         * passa il normalizer. La vecchia prova "basta che Ġ esista" si rompe su
         * gemma-4: 262144 voci con UN solo Ġ spurio vs 137541 ▁-prefissati.
         * Contiamo entrambi i gruppi e lascia vincere chi domina: cosi' un token
         * Ġ isolato non inverte la famiglia, e un vero byte-level (dove il
         * marcatore o e' vuoto o coincide con Ġ, quindi le due cifre sono pari)
         * continua a NON attivare il metaspace. */
        size_t n_bl = 0, n_ws = 0;
        for (size_t i = 0; i < t->vocab.capacity; i++) {
            sds k = t->vocab.slots[i].key;
            if (!k) continue;
            size_t kl = sdslen(k);
            if (kl >= sizeof(bytelevel_space) - 1
                && memcmp(k, bytelevel_space, sizeof(bytelevel_space) - 1) == 0) {
                n_bl++;
            } else if (t->ws_marker_len && kl >= t->ws_marker_len
                && memcmp(k, t->ws_marker, t->ws_marker_len) == 0) {
                n_ws++;
            }
        }
        if (n_ws > n_bl && t->ws_marker_len
            && strimap_get(&t->vocab, t->ws_marker, t->ws_marker_len, &probe)) {
            t->metaspace = 1;
            fprintf(stderr, "llmtoken_load: encoder metaspace/SentencePiece (marcatore %s, %zu token vs %zu byte-level)\n",
                    t->ws_marker, n_ws, n_bl);
        }
    }
    t->byte_fallback = cJSON_IsTrue(cJSON_GetObjectItemCaseSensitive(model, "byte_fallback"));
    {
        cJSON* unk = cJSON_GetObjectItemCaseSensitive(model, "unk_token");
        int32_t uid;
        if (cJSON_IsString(unk) && unk->valuestring[0]
            && strimap_get(&t->vocab, unk->valuestring, strlen(unk->valuestring), &uid)) {
            t->unk_token = sdsnew(unk->valuestring);
        }
    }

    /* pattern regex del pre-tokenizer -- letto dal file, non hardcodato "a
     * memoria": cerchiamo il primo pretokenizer di tipo Split dentro la
     * Sequence (schema Qwen3/GPT-2: Split con pattern Regex + ByteLevel). */
    const char* patterns[4];
    size_t npat = 0;
    cJSON* pt_type = cJSON_GetObjectItemCaseSensitive(pre_tokenizer, "type");
    if (cJSON_IsString(pt_type) && strcmp(pt_type->valuestring, "Sequence") == 0) {
        cJSON* subs = cJSON_GetObjectItemCaseSensitive(pre_tokenizer, "pretokenizers");
        cJSON* sub;
        cJSON_ArrayForEach(sub, subs) {
            cJSON* sub_type = cJSON_GetObjectItemCaseSensitive(sub, "type");
            cJSON* sub_pattern = cJSON_GetObjectItemCaseSensitive(sub, "pattern");
            if (cJSON_IsString(sub_type) && strcmp(sub_type->valuestring, "Split") == 0 && sub_pattern) {
                cJSON* regex = cJSON_GetObjectItemCaseSensitive(sub_pattern, "Regex");
                if (cJSON_IsString(regex) && npat < 4) patterns[npat++] = regex->valuestring;
            }
        }
    } else if (cJSON_IsString(pt_type) && strcmp(pt_type->valuestring, "Split") == 0) {
        cJSON* sub_pattern = cJSON_GetObjectItemCaseSensitive(pre_tokenizer, "pattern");
        cJSON* regex = sub_pattern ? cJSON_GetObjectItemCaseSensitive(sub_pattern, "Regex") : NULL;
        if (cJSON_IsString(regex) && npat < 4) patterns[npat++] = regex->valuestring;
    }
    if (npat == 0 && !t->metaspace) {
        /* Nessun pattern Split esplicito: byte-level BPE (ByteLevel/use_regex, o pre_tokenizer
         * assente) -> la regex GPT-2 di default, come fa HF. Evita TOK LOAD FAIL su tokenizer
         * convertiti che omettono lo Split (es. OLMoE-1B-7B-0924). */
        (void)pt_type;
        patterns[npat++] = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
    }

    for (size_t pk = 0; pk < npat; pk++) {
        int errorcode;
        PCRE2_SIZE erroroffset;
        pcre2_code* re = pcre2_compile((PCRE2_SPTR)patterns[pk], PCRE2_ZERO_TERMINATED, PCRE2_UTF | PCRE2_UCP,
                                       &errorcode, &erroroffset, NULL);
        if (!re) {
            PCRE2_UCHAR errbuf[256];
            pcre2_get_error_message(errorcode, errbuf, sizeof(errbuf));
            fprintf(stderr, "llmtoken_load: pcre2_compile fallita a offset %zu: %s\n",
                    (size_t)erroroffset, (const char*)errbuf);
            cJSON_Delete(j);
            llmtoken_free(t);
            return NULL;
        }
        t->pre_re[t->pre_re_len++] = re;
    }

    cJSON_Delete(j);
    return t;
}

extern llmtoken_handle llmtoken_load_json(const char* json, size_t len) {
    if (!json || len == 0) return NULL;
    return llmtoken_build_json(json, len);
}

extern llmtoken_handle llmtoken_load(const char* tokenizer_json_path) {
    if (tokenizer_json_path == NULL) return NULL;
    FILE* f = fopen(tokenizer_json_path, "rb");
    if (!f) {
        fprintf(stderr, "llmtoken_load: impossibile aprire %s\n", tokenizer_json_path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize < 0) { fclose(f); return NULL; }
    char* buf = (char*)malloc((size_t)fsize + 1);
    size_t nread = fread(buf, 1, (size_t)fsize, f);
    fclose(f);
    buf[nread] = '\0';
    llmtoken_handle h = llmtoken_build_json(buf, nread);
    free(buf);
    return h;
}

/* --------------------------------------------------------------------------
 * Tokenizer TIKTOKEN (modelli Kimi: kimi_linear / Kimi-Linear) -- vedi header.
 *
 * Differenze dal percorso tokenizer.json:
 *  - il vocabolario NON e' JSON: e' `tiktoken.model`, righe "base64(token) SP rank";
 *    id del token = rank (rank densi 0..N-1 sui token base).
 *  - NON c'e' una lista "merges" ordinata: tiktoken fonde, ad ogni passo, la coppia
 *    adiacente il cui token CONCATENATO ha rank MINIMO. E' esattamente la semantica di
 *    bpe_merge() se merge_rank[pair] = rank(token concatenato); deriviamo merge_rank da
 *    ogni token spezzabile in due token esistenti.
 *  - il pre-tokenizer e' il pat_str di Kimi, che usa intersezione di classi
 *    (`[\p{Lu}...&&[^\p{Han}]]`): PCRE2 NON la supporta -> riscrittura equivalente con
 *    lookahead `(?![\p{Han}])[\p{Lo}]`, verificata byte-exact contro tiktoken.
 * -------------------------------------------------------------------------- */

/* Pattern di pre-tokenizzazione Kimi/TikToken, riscritto per PCRE2 (niente `&&`). */
static const char LLMTOKEN_TIKTOKEN_KIMI_PAT[] =
    "[\\p{Han}]+"
    "|[^\\r\\n\\p{L}\\p{N}]?(?:[\\p{Lu}\\p{Lt}\\p{Lm}\\p{M}]|(?![\\p{Han}])[\\p{Lo}])*"
    "(?:[\\p{Ll}\\p{Lm}\\p{M}]|(?![\\p{Han}])[\\p{Lo}])+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    "|[^\\r\\n\\p{L}\\p{N}]?(?:[\\p{Lu}\\p{Lt}\\p{Lm}\\p{M}]|(?![\\p{Han}])[\\p{Lo}])+"
    "(?:[\\p{Ll}\\p{Lm}\\p{M}]|(?![\\p{Han}])[\\p{Lo}])*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    "|\\p{N}{1,3}"
    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
    "|\\s*[\\r\\n]+"
    "|\\s+(?!\\S)"
    "|\\s+";

static int b64_index(int c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

/* Decodifica base64 di in[0..inlen) in out (capacita' >= 3*inlen/4+4). Ritorna la
 * lunghezza decodificata, o -1 su input malformato. */
static long b64_decode(const char* in, size_t inlen, unsigned char* out) {
    int quad[4];
    int qn = 0;
    long olen = 0;
    for (size_t i = 0; i < inlen; i++) {
        int c = (unsigned char)in[i];
        if (c == '=') break;
        if (c == '\r' || c == '\n' || c == ' ' || c == '\t') continue;
        int v = b64_index(c);
        if (v < 0) return -1;
        quad[qn++] = v;
        if (qn == 4) {
            out[olen++] = (unsigned char)((quad[0] << 2) | (quad[1] >> 4));
            out[olen++] = (unsigned char)(((quad[1] & 0xF) << 4) | (quad[2] >> 2));
            out[olen++] = (unsigned char)(((quad[2] & 0x3) << 6) | quad[3]);
            qn = 0;
        }
    }
    if (qn == 1) return -1;
    if (qn == 2) {
        out[olen++] = (unsigned char)((quad[0] << 2) | (quad[1] >> 4));
    } else if (qn == 3) {
        out[olen++] = (unsigned char)((quad[0] << 2) | (quad[1] >> 4));
        out[olen++] = (unsigned char)(((quad[1] & 0xF) << 4) | (quad[2] >> 2));
    }
    return olen;
}

static llmtoken_handle llmtoken_build_tiktoken(const char* model_path, const char* config_path) {
    FILE* f = fopen(model_path, "rb");
    if (!f) {
        fprintf(stderr, "llmtoken_load_tiktoken: impossibile aprire %s\n", model_path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize <= 0) { fclose(f); return NULL; }
    char* buf = (char*)malloc((size_t)fsize + 1);
    size_t nread = fread(buf, 1, (size_t)fsize, f);
    fclose(f);
    buf[nread] = '\0';

    LlmTokenizer* t = (LlmTokenizer*)calloc(1, sizeof(LlmTokenizer));
    build_byte_to_unicode(t->byte_to_unicode, t->byte_to_unicode_len);
    strimap_init(&t->vocab, 1 << 18);
    strimap_init(&t->merge_rank, 1 << 19);

    sds* raw_tokens = NULL;
    int32_t* raw_ranks = NULL;
    size_t n_raw = 0, raw_cap = 0;
    int32_t max_id = -1;

    /* --- pass 1: righe "base64 rank" -> vocab (stringa byte-level -> rank) --- */
    unsigned char* dec = (unsigned char*)malloc((size_t)fsize + 4);
    if (!dec) { free(buf); free(t); return NULL; }
    char* p = buf;
    char* end = buf + nread;
    while (p < end) {
        char* nl = (char*)memchr(p, '\n', (size_t)(end - p));
        char* line_end = nl ? nl : end;
        char* sp = (char*)memchr(p, ' ', (size_t)(line_end - p));
        if (sp) {
            long rank = strtol(sp + 1, NULL, 10);
            long dl = b64_decode(p, (size_t)(sp - p), dec);
            if (dl > 0 && rank >= 0) {
                sds mapped = sdsempty();
                for (long i = 0; i < dl; i++) {
                    unsigned char b = dec[i];
                    mapped = sdscatlen(mapped, t->byte_to_unicode[b], (size_t)t->byte_to_unicode_len[b]);
                }
                strimap_put(&t->vocab, mapped, sdslen(mapped), (int32_t)rank);
                if (n_raw >= raw_cap) {
                    raw_cap = raw_cap ? raw_cap * 2 : 4096;
                    raw_tokens = (sds*)realloc(raw_tokens, raw_cap * sizeof(sds));
                    raw_ranks = (int32_t*)realloc(raw_ranks, raw_cap * sizeof(int32_t));
                }
                raw_tokens[n_raw] = sdsnewlen(dec, (size_t)dl);
                raw_ranks[n_raw] = (int32_t)rank;
                n_raw++;
                if (rank > max_id) max_id = (int32_t)rank;
                sdsfree(mapped);
            }
        }
        if (!nl) break;
        p = nl + 1;
    }

    /* --- pass 2: merge_rank[pair] = rank(token concatenato). Per ogni token di lunghezza
     * >= 2, ogni split in due pezzi ANCHE token esistenti e' un merge valido con priorita' =
     * rank del token: copre esattamente l'insieme dei merge possibili di tiktoken. --- */
    char* mbuf = NULL;
    size_t mbuf_cap = 0;
    for (size_t i = 0; i < n_raw; i++) {
        const unsigned char* rb = (const unsigned char*)raw_tokens[i];
        size_t L = sdslen(raw_tokens[i]);
        if (L < 2) continue;
        size_t need = L * 4 + 1;
        if (need > mbuf_cap) { mbuf_cap = need * 2; mbuf = (char*)realloc(mbuf, mbuf_cap); }
        size_t* plen = (size_t*)malloc((L + 1) * sizeof(size_t));
        size_t ml = 0;
        plen[0] = 0;
        for (size_t b = 0; b < L; b++) {
            memcpy(mbuf + ml, t->byte_to_unicode[rb[b]], (size_t)t->byte_to_unicode_len[rb[b]]);
            ml += (size_t)t->byte_to_unicode_len[rb[b]];
            plen[b + 1] = ml;
        }
        for (size_t s = 1; s < L; s++) {
            int32_t tmp;
            if (strimap_get(&t->vocab, mbuf, plen[s], &tmp)
                && strimap_get(&t->vocab, mbuf + plen[s], ml - plen[s], &tmp)) {
                char key_buf[512];
                size_t klen = make_pair_key(key_buf, sizeof(key_buf), mbuf, plen[s], mbuf + plen[s], ml - plen[s]);
                strimap_put(&t->merge_rank, key_buf, klen, raw_ranks[i]);
            }
        }
        free(plen);
    }
    free(mbuf);
    for (size_t i = 0; i < n_raw; i++) sdsfree(raw_tokens[i]);
    free(raw_tokens);
    free(raw_ranks);
    free(dec);
    free(buf);

    /* --- special token da tokenizer_config.json -> added_tokens_decoder --- */
    if (config_path) {
        FILE* cf = fopen(config_path, "rb");
        if (cf) {
            fseek(cf, 0, SEEK_END);
            long csize = ftell(cf);
            fseek(cf, 0, SEEK_SET);
            if (csize > 0) {
                char* cbuf = (char*)malloc((size_t)csize + 1);
                size_t cn = fread(cbuf, 1, (size_t)csize, cf);
                cbuf[cn] = '\0';
                cJSON* cj = cJSON_ParseWithLength(cbuf, cn);
                if (cj) {
                    cJSON* deco = cJSON_GetObjectItemCaseSensitive(cj, "added_tokens_decoder");
                    size_t n_added = deco ? (size_t)cJSON_GetArraySize(deco) : 0;
                    t->special_tokens = n_added ? (SpecialToken*)malloc(n_added * sizeof(SpecialToken)) : NULL;
                    t->special_ids = n_added ? (int32_t*)malloc(n_added * sizeof(int32_t)) : NULL;
                    t->special_contents = n_added ? (sds*)malloc(n_added * sizeof(sds)) : NULL;
                    size_t n_special = 0;
                    cJSON* at;
                    cJSON_ArrayForEach(at, deco) {
                        if (!cJSON_IsObject(at)) continue;
                        cJSON* content_j = cJSON_GetObjectItemCaseSensitive(at, "content");
                        if (!cJSON_IsString(content_j) || content_j->valuestring[0] == '\0') continue;
                        int32_t id = at->string ? (int32_t)strtol(at->string, NULL, 10) : -1;
                        if (id < 0) continue;
                        sds content_sds = sdsnew(content_j->valuestring);
                        t->special_tokens[n_special].content = content_sds;
                        t->special_tokens[n_special].id = id;
                        t->special_ids[n_special] = id;
                        t->special_contents[n_special] = sdsdup(content_sds);
                        n_special++;
                        if (id > max_id) max_id = id;
                    }
                    qsort(t->special_tokens, n_special, sizeof(SpecialToken), special_token_cmp);
                    t->special_tokens_len = n_special;
                    t->special_id_map_len = n_special;
                    cJSON_Delete(cj);
                }
                free(cbuf);
            }
            fclose(cf);
        }
    }

    /* --- id_to_token (decode) --- */
    t->id_to_token_len = (size_t)max_id + 1;
    t->id_to_token = (sds*)calloc(t->id_to_token_len, sizeof(sds));
    for (size_t i = 0; i < t->vocab.capacity; i++) {
        if (t->vocab.slots[i].key) {
            int32_t id = t->vocab.slots[i].value;
            if ((size_t)id < t->id_to_token_len) t->id_to_token[id] = sdsdup(t->vocab.slots[i].key);
        }
    }

    /* --- inverso byte<->unicode per il decode (byte-level) --- */
    t->unicode_map_count = 0;
    for (int b = 0; b < 256; b++) {
        size_t dummy = 0;
        uint32_t cp = utf8_decode_next(t->byte_to_unicode[b], (size_t)t->byte_to_unicode_len[b], &dummy);
        t->unicode_cp[t->unicode_map_count] = cp;
        t->unicode_byte[t->unicode_map_count] = (uint8_t)b;
        t->unicode_map_count++;
    }

    /* --- pre-tokenizer: pattern tiktoken riscritto per PCRE2 --- */
    {
        int errorcode;
        PCRE2_SIZE erroroffset;
        pcre2_code* re = pcre2_compile((PCRE2_SPTR)LLMTOKEN_TIKTOKEN_KIMI_PAT, PCRE2_ZERO_TERMINATED,
                                       PCRE2_UTF | PCRE2_UCP, &errorcode, &erroroffset, NULL);
        if (!re) {
            PCRE2_UCHAR errbuf[256];
            pcre2_get_error_message(errorcode, errbuf, sizeof(errbuf));
            fprintf(stderr, "llmtoken_load_tiktoken: pcre2_compile fallita a offset %zu: %s\n",
                    (size_t)erroroffset, (const char*)errbuf);
            llmtoken_free(t);
            return NULL;
        }
        t->pre_re[t->pre_re_len++] = re;
    }

    fprintf(stderr, "llmtoken_load_tiktoken: caricati %zu token (max id %d), %zu special\n",
            t->vocab.count, max_id, t->special_tokens_len);
    return t;
}

extern llmtoken_handle llmtoken_load_tiktoken(const char* tiktoken_model_path, const char* tokenizer_config_json_path) {
    if (tiktoken_model_path == NULL) return NULL;
    return llmtoken_build_tiktoken(tiktoken_model_path, tokenizer_config_json_path);
}

extern void llmtoken_free(llmtoken_handle handle) {
    if (!handle) return;
    LlmTokenizer* t = (LlmTokenizer*)handle;
    for (size_t pk = 0; pk < t->pre_re_len; pk++) if (t->pre_re[pk]) pcre2_code_free(t->pre_re[pk]);
    strimap_free(&t->vocab);
    strimap_free(&t->merge_rank);
    for (size_t i = 0; i < t->merges_len; i++) sdsfree(t->merges[i]);
    free(t->merges);
    free(t->export_tokens);
    for (size_t i = 0; i < t->id_to_token_len; i++) {
        if (t->id_to_token[i]) sdsfree(t->id_to_token[i]);
    }
    free(t->id_to_token);
    for (size_t i = 0; i < t->special_tokens_len; i++) sdsfree(t->special_tokens[i].content);
    free(t->special_tokens);
    for (size_t i = 0; i < t->special_id_map_len; i++) sdsfree(t->special_contents[i]);
    free(t->special_ids);
    free(t->special_contents);
    if (t->unk_token) sdsfree(t->unk_token);
    free(t);
}

/* ── Export per i METADATA GGUF (tokens/merges/model) ────────────────────────
 * Unica sorgente di tokenizer.json: il converter MLX->GGUF NON ri-parsa il file,
 * ma prende i dati da qui. I puntatori sono posseduti dall'handle: validi fino
 * a llmtoken_free (il converter li scrive mentre l'handle e' vivo). */
extern const char* const* llmtoken_export_tokens(llmtoken_handle handle, size_t* out_n) {
    LlmTokenizer* t = (LlmTokenizer*)handle;
    if (!t) { if (out_n) *out_n = 0; return NULL; }
    if (!t->export_tokens) {
        size_t n = t->id_to_token_len;
        const char** arr = (const char**)calloc(n ? n : 1, sizeof(char*));
        for (size_t i = 0; i < n; i++) if (t->id_to_token[i]) arr[i] = t->id_to_token[i];
        for (size_t i = 0; i < t->special_id_map_len; i++) {   /* riempi i buchi (added_tokens) */
            int32_t id = t->special_ids[i];
            if (id >= 0 && (size_t)id < n && !arr[id]) arr[id] = t->special_contents[i];
        }
        t->export_tokens = arr;
        t->export_tokens_len = n;
    }
    if (out_n) *out_n = t->export_tokens_len;
    return t->export_tokens;
}
extern const char* const* llmtoken_export_merges(llmtoken_handle handle, size_t* out_n) {
    LlmTokenizer* t = (LlmTokenizer*)handle;
    if (!t) { if (out_n) *out_n = 0; return NULL; }
    if (out_n) *out_n = t->merges_len;
    return (const char* const*)t->merges;
}
extern const char* llmtoken_gguf_model(llmtoken_handle handle) {
    LlmTokenizer* t = (LlmTokenizer*)handle;
    if (!t) return NULL;
    return t->metaspace ? "llama" : "gpt2";
}

extern int llmtoken_encode(llmtoken_handle handle, const char* text, int32_t* out_ids, int max_ids) {
    if (handle == NULL || text == NULL) return -1;
    const LlmTokenizer* t = (const LlmTokenizer*)handle;

    int32_t* ids = NULL;
    size_t ids_len = 0, ids_cap = 0;
    int ok = 1;

    size_t n = strlen(text);
    size_t pos = 0;
    sds normal_buf = sdsempty();

    while (pos < n && ok) {
        int matched = 0;
        for (size_t si = 0; si < t->special_tokens_len; si++) {
            size_t sclen = sdslen(t->special_tokens[si].content);
            if (n - pos >= sclen && memcmp(text + pos, t->special_tokens[si].content, sclen) == 0) {
                if (sdslen(normal_buf) > 0) {
                    sds* pieces = NULL; size_t pieces_len = 0, pieces_cap = 0;
                    if (!pretokenize(t, normal_buf, sdslen(normal_buf), &pieces, &pieces_len, &pieces_cap)) { ok = 0; }
                    for (size_t pi = 0; ok && pi < pieces_len; pi++) {
                        if (!encode_piece(t, pieces[pi], sdslen(pieces[pi]), &ids, &ids_len, &ids_cap)) ok = 0;
                    }
                    for (size_t pi = 0; pi < pieces_len; pi++) sdsfree(pieces[pi]);
                    free(pieces);
                    sdsclear(normal_buf);
                }
                if (ok) {
                    if (ids_len >= ids_cap) { ids_cap = ids_cap ? ids_cap * 2 : 64; ids = (int32_t*)realloc(ids, ids_cap * sizeof(int32_t)); }
                    ids[ids_len++] = t->special_tokens[si].id;
                    pos += sclen;
                }
                matched = 1;
                break;
            }
        }
        if (!matched) {
            normal_buf = sdscatlen(normal_buf, text + pos, 1);
            pos++;
        }
    }
    if (ok && sdslen(normal_buf) > 0) {
        sds* pieces = NULL; size_t pieces_len = 0, pieces_cap = 0;
        if (!pretokenize(t, normal_buf, sdslen(normal_buf), &pieces, &pieces_len, &pieces_cap)) ok = 0;
        for (size_t pi = 0; ok && pi < pieces_len; pi++) {
            if (!encode_piece(t, pieces[pi], sdslen(pieces[pi]), &ids, &ids_len, &ids_cap)) ok = 0;
        }
        for (size_t pi = 0; pi < pieces_len; pi++) sdsfree(pieces[pi]);
        free(pieces);
    }
    sdsfree(normal_buf);

    if (!ok) { free(ids); return -1; }

    int count = (int)ids_len;
    if (out_ids != NULL) {
        int to_copy = count < max_ids ? count : max_ids;
        for (int i = 0; i < to_copy; i++) out_ids[i] = ids[i];
    }
    free(ids);
    return count;
}

extern int llmtoken_decode(llmtoken_handle handle, const int32_t* ids, int n_ids, char* out_text, int max_bytes) {
    return llmtoken_decode_ex(handle, ids, n_ids, out_text, max_bytes, 1);
}
/* Come llmtoken_decode ma con `strip_leading_metaspace` esplicito: il decode PER-TOKEN
 * (streaming) NON deve applicare lo Strip(" ", start=1) del metaspace, altrimenti ogni token
 * perde il suo spazio iniziale e le parole si incollano (bug gemma4: "Lacapitaled'Italia"). */
extern int llmtoken_decode_ex(llmtoken_handle handle, const int32_t* ids, int n_ids, char* out_text, int max_bytes, int strip_leading_metaspace) {
    if (handle == NULL || (ids == NULL && n_ids > 0)) return -1;
    const LlmTokenizer* t = (const LlmTokenizer*)handle;

    sds out = sdsempty();
    int ok = 1;
    for (int i = 0; ok && i < n_ids; i++) {
        int32_t id = ids[i];
        int is_special = 0;
        for (size_t si = 0; si < t->special_id_map_len; si++) {
            if (t->special_ids[si] == id) {
                out = sdscatsds(out, t->special_contents[si]);
                is_special = 1;
                break;
            }
        }
        if (is_special) continue;

        if (id < 0 || (size_t)id >= t->id_to_token_len || !t->id_to_token[id]) { ok = 0; break; }
        sds tok = t->id_to_token[id];
        if (t->metaspace) {
            /* Inverso del metaspace: le stringhe del vocabolario sono gia' UTF-8
             * reale, quindi basta rimandare il marcatore a spazio e sciogliere
             * <0xHH> nel byte che rappresenta (l'inverso del byte_fallback in
             * encode). Lo Strip(" ", start=1) che il file mette in coda al decoder
             * vale sull'intera sequenza e sta dopo il ciclo, non qui. */
            size_t q = 0, tl = sdslen(tok);
            while (q < tl) {
                int hex_done = 0;
                if (q + 6 <= tl && tok[q] == '<' && tok[q + 1] == '0' && (tok[q + 2] == 'x' || tok[q + 2] == 'X')
                    && tok[q + 5] == '>') {
                    char nib[3] = { tok[q + 3], tok[q + 4], '\0' };
                    char* endp = NULL;
                    long v = strtol(nib, &endp, 16);
                    if (endp && *endp == '\0' && v >= 0 && v <= 255) {
                        char b = (char)v;
                        out = sdscatlen(out, &b, 1);
                        q += 6;
                        hex_done = 1;
                    }
                }
                if (hex_done) continue;
                size_t start = q;
                utf8_decode_next(tok, tl, &q);
                if (t->ws_marker_len && q - start == t->ws_marker_len
                    && memcmp(tok + start, t->ws_marker, t->ws_marker_len) == 0) {
                    out = sdscatlen(out, " ", 1);
                } else {
                    out = sdscatlen(out, tok + start, q - start);
                }
            }
            continue;
        }
        size_t p = 0, toklen = sdslen(tok);
        while (p < toklen) {
            uint32_t cp = utf8_decode_next(tok, toklen, &p);
            int found = 0;
            for (int k = 0; k < t->unicode_map_count; k++) {
                if (t->unicode_cp[k] == cp) {
                    char b = (char)t->unicode_byte[k];
                    out = sdscatlen(out, &b, 1);
                    found = 1;
                    break;
                }
            }
            if (!found) { ok = 0; break; }
        }
    }

    if (!ok) { sdsfree(out); return -1; }
    if (strip_leading_metaspace && t->metaspace && sdslen(out) > 0 && out[0] == ' ') {
        /* Strip(" ", start=1, stop=0), l'ultimo passo della sezione "decoder" del
         * file: ricade sulla sequenza INTERA, e l'unico consumer (decode() di
         * llm_server) passa la sequenza completa generata -- non il token
         * singolo, che sarebbe il caso in cui togliere lo spazio fra due parole. */
        sdsrange(out, 1, -1);
    }

    int needed = (int)sdslen(out);
    if (out_text != NULL && max_bytes > 0) {
        int to_copy = needed < (max_bytes - 1) ? needed : (max_bytes - 1);
        if (to_copy > 0) memcpy(out_text, out, (size_t)to_copy);
        out_text[to_copy] = '\0';
    }
    sdsfree(out);
    return needed;
}
