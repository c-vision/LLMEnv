/* ============================================================================
 * Dominio pesi/shard/quant-synth: wcache, multishard, load_all_shards, awq/gptq-nf4 synth, kimi MLA synth, pretouch, stack_moe_experts, fuse_mixtral_gate_up.
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include "llmcodec_fp16.h"
#include "llmcodec_quant_awq.h"
#include "llmcodec_quant_gptq.h"
#include "llmcodec_quant_nf4.h"
#include "llmcodec_rolemap.h"
#include "llm_mlx_tq_decode.h"   /* decode TurboQuant CONDIVISO (anche dal path SWAP) */
#include <cJSON.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Pesi caricati da TUTTI gli shard safetensors del modello (a differenza
 * delle funzioni di checkpoint sopra, che leggono solo il primo file). Ogni
 * shard resta una mappa separata; il lookup prova ogni shard in ordine --
 * più semplice e più robusto del fondere le mappe con l'API iteratore di
 * mlx-c (mai esercitata prima in questo codebase). */

/* Memo dei lookup pesi: (shards, name) -> mlx_array. I pesi sono IMMUTABILI per la vita
 * del modello, ma i blocchi li ricercano per stringa a OGNI token (mlx_map_string_to_array_get
 * su tutti gli shard) -- misurato: ~4% del tempo CPU nel decode (sample). La cache evita la
 * ricerca ripetuta. Chiave = puntatore dell'array di shard (unico per modello) + nome.
 * Svuotata a ogni load_model. */
#define LLM_WCACHE_SZ 32768u
static LlmWCacheEnt g_wcache[LLM_WCACHE_SZ];
void llm_mlx_wcache_clear(void) {
    for (unsigned i = 0; i < LLM_WCACHE_SZ; i++) {
        if (g_wcache[i].key != NULL) {
            mlx_array_free(g_wcache[i].val);
            /* FIX 2026-09-16: mlx_array_free NON azzera l'handle -> `.ctx` restava DANGLING.
             * Al fill successivo `mlx_array_set(&g_wcache[j].val, *out)` fa il release del ctx
             * vecchio (gia' liberato) -> EXC_BAD_ACCESS (SIGSEGV). La cache si svuota a ogni
             * load_model, quindi bastava caricare un SECONDO modello (es. il draft DFlash2 in
             * dflash2_create) per far crashare il primo multishard_get su un nome gia' in cache:
             * e' la ragione per cui la speculative decode DFlash2 non era mai stata misurata. */
            g_wcache[i].val = mlx_array_new();
            g_wcache[i].key = NULL;
        }
    }
}
/* Traduce un nome pesi HF (es. "model.layers.0.self_attn.q_proj.weight") nel nome canonico GGUF
 * (es. "blk.0.attn_q.weight") via la role map `.llmarch`. 1 se tradotto, 0 se nessuna regola.
 * La mappa HF<->GGUF resta UNA SOLA (il `.llmarch`); qui c'e' solo la meccanica del `%d`. */
/* Un tensore di BIAS GGUF (`blk.N.attn_q.bias`) mappa sulla stessa regola del peso, il cui
 * template finisce in `.weight`: per una richiesta `.bias` si sostituisce il suffisso. */
static void llm_bias_suffix(char* s) {
    size_t n = strlen(s);
    if (n > 7 && strcmp(s + n - 7, ".weight") == 0) memcpy(s + n - 7, ".bias", 6);
}
static int llm_mlx_hf_to_canonical(const LlmCodecRoleMap* rm, const char* hf, char* out, size_t cap) {
    if (rm == NULL || hf == NULL) return 0;
    /* Solo i nomi che la role map mappa davvero (`.weight`/`.bias`). Le richieste di artefatti di
     * quantizzazione (`.scales`/`.biases`) NON vanno tradotte: restituirebbero il nome del peso e il
     * forward crederebbe di avere scales. Cosi' i pesi "plain" (GGUF) seguono il path non-quantizzato. */
    size_t hl = strlen(hf);
    int is_w = (hl > 7 && strcmp(hf + hl - 7, ".weight") == 0);
    int is_b = (hl > 5 && strcmp(hf + hl - 5, ".bias") == 0);
    if (!is_w && !is_b) return 0;
    const char* g = llmcodec_rolemap_gguf(rm, hf);   /* template della regola che combacia (o NULL) */
    if (g == NULL || g[0] == 0) return 0;
    char t[192];
    /* `%s` = gate/up/down (esperti ROUTED stacked: il blocco chiede `mlp.switch_mlp.{gate,up,down}_proj`). */
    const char* ps = strstr(g, "%s");
    if (ps != NULL) {
        const char* sub = strstr(hf, "gate_proj") ? "gate" : strstr(hf, "up_proj") ? "up" : strstr(hf, "down_proj") ? "down" : NULL;
        if (sub == NULL) return 0;
        snprintf(t, sizeof t, "%.*s%s%s", (int)(ps - g), g, sub, ps + 2);
        g = t;
    }
    /* `%d` = indice layer. */
    const char* pd = strstr(g, "%d");
    if (pd != NULL) {
        const char* p = strstr(hf, "layers.");
        p = p ? p + 7 : hf;
        while (*p && (*p < '0' || *p > '9')) p++;
        if (*p < '0' || *p > '9') return 0;
        int L = atoi(p);
        char t2[224];
        snprintf(t2, sizeof t2, "%.*s%d%s", (int)(pd - g), g, L, pd + 2);
        snprintf(out, cap, "%s", t2);
        if (is_b) llm_bias_suffix(out);
        return 1;
    }
    snprintf(out, cap, "%s", g);
    if (is_b) llm_bias_suffix(out);
    return 1;
}
bool multishard_get(const MultiShardWeights* mw, const char* name, mlx_array* out) {
    if (mw == NULL || name == NULL) return false;
    unsigned h = 2166136261u;
    for (const char* p = name; *p; ++p) { h ^= (unsigned char)*p; h *= 16777619u; }
    unsigned idx = (h ^ (unsigned)(uintptr_t)mw->shards) & (LLM_WCACHE_SZ - 1);
    for (unsigned probe = 0; probe < 16; ++probe) {
        unsigned j = (idx + probe) & (LLM_WCACHE_SZ - 1);
        if (g_wcache[j].key == (const void*)mw->shards && strcmp(g_wcache[j].name, name) == 0) {
            if (g_wcache[j].val.ctx == NULL) return false;          /* memoizzato ASSENTE */
            return mlx_array_set(out, g_wcache[j].val) == 0;
        }
        if (g_wcache[j].key == NULL) break;
    }
    for (int i = 0; i < mw->count; i++) {
        if (mlx_map_string_to_array_get(out, mw->shards[i], name) == 0) {
            for (unsigned probe = 0; probe < 16; ++probe) {
                unsigned j = (idx + probe) & (LLM_WCACHE_SZ - 1);
                if (g_wcache[j].key == NULL) {
                    g_wcache[j].key = (const void*)mw->shards;
                    llm_mlx_mode_set(g_wcache[j].name, sizeof g_wcache[j].name, name);
                    mlx_array_set(&g_wcache[j].val, *out);
                    break;
                }
            }
            return true;
        }
    }
    /* Fallback GGUF: i pesi sono chiavati col nome canonico GGUF; traduci il nome HF del forward. */
    if (mw->gguf_rm != NULL) {
        char cn[256];
        if (llm_mlx_hf_to_canonical(mw->gguf_rm, name, cn, sizeof cn)) {
            for (int i = 0; i < mw->count; i++) {
                if (mlx_map_string_to_array_get(out, mw->shards[i], cn) == 0) {
                    for (unsigned probe = 0; probe < 16; ++probe) {
                        unsigned j = (idx + probe) & (LLM_WCACHE_SZ - 1);
                        if (g_wcache[j].key == NULL) {
                            g_wcache[j].key = (const void*)mw->shards;
                            llm_mlx_mode_set(g_wcache[j].name, sizeof g_wcache[j].name, name);
                            mlx_array_set(&g_wcache[j].val, *out);
                            break;
                        }
                    }
                    return true;
                }
            }
        }
    }
    for (unsigned probe = 0; probe < 16; ++probe) {
        unsigned j = (idx + probe) & (LLM_WCACHE_SZ - 1);
        if (g_wcache[j].key == NULL) {
            g_wcache[j].key = (const void*)mw->shards;
            llm_mlx_mode_set(g_wcache[j].name, sizeof g_wcache[j].name, name);
            g_wcache[j].val = mlx_array_new();                      /* marker ASSENTE */
            break;
        }
    }
    return false;
}


/* Copia un handle MLX opzionale dalla cache. Gli array assenti sono rappresentati
 * da ctx==NULL e devono restare tali: mlx_array_set dereferenzia il sorgente e su
 * un handle vuoto solleva "expected a non-empty mlx_array". */
bool mlx_array_set_optional(mlx_array* out, mlx_array value) {
    return value.ctx == NULL || mlx_array_set(out, value) == 0;
}

/* Inserisce/sostituisce nella memo un array per (shards, name): serve a PRE-RIEMPIRE la cache
 * con copie GPU-resident dei pesi (vedi llm_mlx_shards_to_gpu). */
static void llm_wcache_put(const void* shards, const char* name, mlx_array val) {
    if (shards == NULL || name == NULL || val.ctx == NULL) return;
    unsigned h = 2166136261u;
    for (const char* p = name; *p; ++p) { h ^= (unsigned char)*p; h *= 16777619u; }
    unsigned idx = (h ^ (unsigned)(uintptr_t)shards) & (LLM_WCACHE_SZ - 1);
    for (unsigned probe = 0; probe < 16; ++probe) {
        unsigned j = (idx + probe) & (LLM_WCACHE_SZ - 1);
        if (g_wcache[j].key == shards && strcmp(g_wcache[j].name, name) == 0) {
            mlx_array_set(&g_wcache[j].val, val);
            return;
        }
        if (g_wcache[j].key == NULL) {
            g_wcache[j].key = shards;
            llm_mlx_mode_set(g_wcache[j].name, sizeof g_wcache[j].name, name);
            g_wcache[j].val = mlx_array_new();
            mlx_array_set(&g_wcache[j].val, val);
            return;
        }
    }
}

/* COPIA TUTTI I PESI DI UN MODELLO SULLA GPU E LI MEMOIZZA.
 * Perche': `load_all_shards` carica con `mlx_default_cpu_stream_new()` -> gli array restano
 * CPU-residenti; ad ogni op che li usa (es. il matmul bf16 non quantizzato del draft DFlash2)
 * MLX ri-trasferisce il peso CPU->GPU: misurato 22 MB / ~13 ms per una proiezione MLP del draft
 * (40x l'atteso) -> il backbone del draft costava 55 ms/round. Stessa classe del fix MTP
 * (`mlx_load_safetensors li lascia su CPU`, 35->3.8 ms). */
void llm_mlx_shards_to_gpu(MultiShardWeights* mw, mlx_stream gpu) {
    if (mw == NULL || mw->shards == NULL || gpu.ctx == NULL) return;
    long n_copied = 0;
    for (int i = 0; i < mw->count; i++) {
        mlx_map_string_to_array map = mw->shards[i];
        if (map.ctx == NULL) continue;
        mlx_map_string_to_array_iterator it = mlx_map_string_to_array_iterator_new(map);
        if (it.ctx == NULL) continue;
        const char* key = NULL;
        mlx_array val = mlx_array_new();
        while (mlx_map_string_to_array_iterator_next(&key, &val, it) == 0) {
            if (key == NULL || val.ctx == NULL) continue;
            mlx_array g = mlx_array_new();
            if (mlx_copy(&g, val, gpu) == 0 && mlx_array_eval(g) == 0) {
                llm_wcache_put((const void*)mw->shards, key, g);
                n_copied++;
            }
            mlx_array_free(g);
        }
        mlx_array_free(val);
        mlx_map_string_to_array_iterator_free(it);
    }
}

bool multishard_get_triple(const MultiShardWeights* mw, const char* base, mlx_array* w, mlx_array* scales, mlx_array* biases) {
    /* Nome costruito SENZA snprintf/__vfprintf: questa funzione e' nel percorso caldo (3 nomi
     * per chiamata, ~1200 formattazioni/token su un MoE) e il printf-engine pesava ~3% del
     * ciclo di decode nel profilo `sample` (157 frame vs 0 di mlx_lm, che usa attributi Python).
     * Suffissi di lunghezza fissa (7 char + NUL) copiati dopo `base`. */
    char name[256];
    size_t bl = strlen(base);
    if (bl + 8 > sizeof(name)) return false;
    memcpy(name, base, bl);
    memcpy(name + bl, ".weight", 8);
    if (!multishard_get(mw, name, w)) return false;
    /* Tolleranza tensori PLAIN (non quantizzati): scales/biases possono mancare (es. le
     * proiezioni bf16 di gpt-oss-120b-mxfp4-bf16). Il chiamante che riceve scales/biases con
     * ctx==NULL passa a quantized_linear_cfg, che ora instrada a plain_linear quando scales e'
     * assente (fix 15c21156). Solo il weight e' obbligatorio: un NOME sbagliato fallisce lo
     * stesso (weight non trovato). */
    memcpy(name + bl, ".scales", 8);
    if (!multishard_get(mw, name, scales)) { /* scales mancante: tensore plain, si prosegue */ }
    memcpy(name + bl, ".biases", 8);
    if (!multishard_get(mw, name, biases)) {
        /* GGUF attn/mlp bias in F32 (`blk.N.attn_q.bias`, singolare). */
        memcpy(name + bl, ".bias", 6);
        if (!multishard_get(mw, name, biases)) { /* biases mancante: quant bias-less o plain */ }
    }
    return true;
}

void multishard_free_all(MultiShardWeights* mw) {
    for (int i = 0; i < mw->count; i++) mlx_map_string_to_array_free(mw->shards[i]);
    free(mw->shards);
    mw->shards = NULL;
    mw->count = 0;
    mw->cap = 0;
}

/* Dedup dei nomi file di shard (era std::set<std::string>): un modello ha
 * tipicamente poche decine di shard al massimo, una ricerca lineare prima di
 * ogni inserimento e' ampiamente sufficiente e non richiede una hash table
 * dedicata come per il vocabolario BPE del tokenizer. */

static bool strset_contains(const StrSet* s, const char* name) {
    for (int i = 0; i < s->count; i++) if (strcmp(s->names[i], name) == 0) return true;
    return false;
}

static void strset_add(StrSet* s, const char* name) {
    if (strset_contains(s, name)) return;
    if (s->count >= s->cap) {
        s->cap = s->cap ? s->cap * 2 : 8;
        s->names = (char**)realloc(s->names, (size_t)s->cap * sizeof(char*));
    }
    s->names[s->count++] = strdup(name);
}

static void strset_free(StrSet* s) {
    for (int i = 0; i < s->count; i++) free(s->names[i]);
    free(s->names);
    s->names = NULL;
    s->count = 0;
    s->cap = 0;
}

bool load_all_shards(const char* model_dir, mlx_stream stream, MultiShardWeights* weights) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors.index.json", model_dir);
    FILE* f = fopen(path, "rb");
    if (!f) {
        /* §13.6: modello SINGOLO-shard senza index.json (es. draft BF16 Qwen3.8-27B-DFlash2
         * che ha solo model.safetensors). Carica direttamente il file come unico shard. */
        mlx_map_string_to_array m = mlx_map_string_to_array_new();
        mlx_map_string_to_string meta = mlx_map_string_to_string_new();
        char sp[1024];
        snprintf(sp, sizeof(sp), "%s/model.safetensors", model_dir);
        bool ok = mlx_load_safetensors(&m, &meta, sp, stream) == 0;
        mlx_map_string_to_string_free(meta);
        if (!ok) {
            fprintf(stderr, "load_all_shards: mlx_load_safetensors fallita per %s (niente index.json)\n", sp);
            mlx_map_string_to_array_free(m);
            return false;
        }
        weights->shards = (mlx_map_string_to_array*)realloc(weights->shards, sizeof(mlx_map_string_to_array));
        weights->shards[0] = m;
        weights->count = 1;
        weights->cap = 1;
        return true;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsize < 0) { fclose(f); return false; }
    char* buf = (char*)malloc((size_t)fsize + 1);
    size_t nread = fread(buf, 1, (size_t)fsize, f);
    fclose(f);
    buf[nread] = '\0';

    cJSON* j = cJSON_ParseWithLength(buf, nread);
    free(buf);
    if (!j) {
        fprintf(stderr, "load_all_shards: JSON malformato in %s\n", path);
        return false;
    }
    cJSON* weight_map = cJSON_GetObjectItemCaseSensitive(j, "weight_map");
    if (!weight_map) {
        fprintf(stderr, "load_all_shards: \"weight_map\" assente nell'index\n");
        cJSON_Delete(j);
        return false;
    }

    StrSet shard_files = {0};
    cJSON* entry;
    cJSON_ArrayForEach(entry, weight_map) {
        if (cJSON_IsString(entry)) strset_add(&shard_files, entry->valuestring);
    }
    cJSON_Delete(j);

    bool ok = true;
    for (int i = 0; ok && i < shard_files.count; i++) {
        mlx_map_string_to_array m = mlx_map_string_to_array_new();
        mlx_map_string_to_string meta = mlx_map_string_to_string_new();
        char shard_path[1024];
        snprintf(shard_path, sizeof(shard_path), "%s/%s", model_dir, shard_files.names[i]);
        if (mlx_load_safetensors(&m, &meta, shard_path, stream) != 0) {
            fprintf(stderr, "load_all_shards: mlx_load_safetensors fallita per %s\n", shard_path);
            mlx_map_string_to_array_free(m);
            mlx_map_string_to_string_free(meta);
            ok = false;
            break;
        }
        mlx_map_string_to_string_free(meta);
        if (weights->count >= weights->cap) {
            weights->cap = weights->cap ? weights->cap * 2 : 8;
            weights->shards = (mlx_map_string_to_array*)realloc(weights->shards, (size_t)weights->cap * sizeof(mlx_map_string_to_array));
        }
        weights->shards[weights->count++] = m;
    }
    strset_free(&shard_files);
    return ok;
}

/* ── AWQ (AutoAWQ GEMM) — dequant a load ────────────────────────────────────────────────────────
 * I checkpoint AWQ (`quantization_config.quant_method="awq"`, `version="gemm"`) NON hanno
 * `<base>.weight`: hanno `<base>.qweight` (I32 [in,out/8]) + `<base>.qzeros` (I32
 * [in/group,out/8]) + `<base>.scales` (F16 [in/group,out]). Il resto del motore cerca
 * `.weight`, quindi senza questo passo il load fallisce. Qui i pesi vengono DEQUANTIZZATI UNA
 * VOLTA a load (llmcodec_awq_dequant, nel codec condiviso — §0bis) in un tensore PLAIN F16
 * [out,in] e iniettati come `<base>.weight` in uno shard SINTETICO (stesso pattern di
 * stack_moe_experts / kimi_linear_synthesize_absorbed_mla). multishard_get_triple li trova
 * senza modifiche; poiche' `quantization_config` NON e' la chiave `quantization` letta dal
 * config, quant_mode resta "none" -> quantized_linear_cfg instrada a plain_linear (x @ w.T).
 *
 * Va chiamata SUBITO dopo load_all_shards e PRIMA di qualunque multishard_get: quelle funzioni
 * MEMOIZZANO i miss, quindi un lookup di `<base>.weight` fatto prima dell'iniezione resterebbe
 * "assente" in cache. Ritorna il numero di tensori dequantizzati (0 = nessun qweight, no-op),
 * -1 su errore. */
int awq_synthesize_weights(MultiShardWeights* weights, mlx_stream stream) {
    if (weights == NULL || weights->count <= 0) return 0;
    mlx_map_string_to_array synth = mlx_map_string_to_array_new();
    if (synth.ctx == NULL) return -1;
    int n_dequant = 0;
    bool ok = true;
    char base[240], nm[272];
    for (int si = 0; ok && si < weights->count; si++) {
        mlx_map_string_to_array map = weights->shards[si];
        if (map.ctx == NULL) continue;
        mlx_map_string_to_array_iterator it = mlx_map_string_to_array_iterator_new(map);
        if (it.ctx == NULL) continue;
        const char* key = NULL;
        mlx_array val = mlx_array_new();
        while (ok && mlx_map_string_to_array_iterator_next(&key, &val, it) == 0) {
            size_t kl = (key != NULL) ? strlen(key) : 0;
            /* Solo le chiavi `<base>.qweight`. */
            if (key == NULL || kl <= 8 || strcmp(key + kl - 8, ".qweight") != 0) {
                mlx_array_free(val); val = mlx_array_new();
                continue;
            }
            size_t bl = kl - 8;
            if (bl >= sizeof(base)) { fprintf(stderr, "[awq] nome troppo lungo: %s\n", key); ok = false; break; }
            memcpy(base, key, bl); base[bl] = '\0';

            /* GPTQ condivide i nomi `.qweight/.qzeros/.scales` ma ha anche `.g_idx`:
             * lo dequantizza gptq_nf4_synthesize_weights. Qui lo si salta, altrimenti
             * verrebbe interpretato (male) col layout AWQ. */
            snprintf(nm, sizeof nm, "%s.g_idx", base);
            mlx_array gprobe = mlx_array_new();
            if (multishard_get(weights, nm, &gprobe)) {
                mlx_array_free(gprobe); mlx_array_free(val); val = mlx_array_new();
                continue;
            }

            mlx_array qz = mlx_array_new(), sc = mlx_array_new();
            snprintf(nm, sizeof nm, "%s.qzeros", base);
            bool have_z = multishard_get(weights, nm, &qz);
            snprintf(nm, sizeof nm, "%s.scales", base);
            bool have_s = multishard_get(weights, nm, &sc);
            if (!have_z || !have_s ||
                mlx_array_eval(val) != 0 || mlx_array_eval(qz) != 0 || mlx_array_eval(sc) != 0) {
                fprintf(stderr, "[awq] %s: qzeros/scales assenti o eval fallita (z=%d s=%d)\n",
                        base, (int)have_z, (int)have_s);
                ok = false;
                mlx_array_free(qz); mlx_array_free(sc); mlx_array_free(val); val = mlx_array_new();
                break;
            }
            if (mlx_array_ndim(val) != 2 || mlx_array_ndim(qz) != 2 || mlx_array_ndim(sc) != 2 ||
                mlx_array_dtype(val) != MLX_INT32 || mlx_array_dtype(qz) != MLX_INT32 ||
                mlx_array_dtype(sc) != MLX_FLOAT16) {
                fprintf(stderr, "[awq] %s: dtype/shape inattesi (qweight/qzeros I32 2D, scales F16 2D)\n", base);
                ok = false;
                mlx_array_free(qz); mlx_array_free(sc); mlx_array_free(val); val = mlx_array_new();
                break;
            }
            const int* qws = mlx_array_shape(val);
            const int* qzs = mlx_array_shape(qz);
            const int* scs = mlx_array_shape(sc);
            const int in = qws[0], op = qws[1], out = op * 8;
            const int group = (qzs[0] > 0) ? in / qzs[0] : 0;
            if (qws[1] != qzs[1] || group <= 0 || (in % group) != 0 || scs[0] != qzs[0] || scs[1] != out) {
                fprintf(stderr, "[awq] %s: shape incoerenti in=%d out=%d op=%d group=%d sc=[%d,%d]\n",
                        base, in, out, op, group, scs[0], scs[1]);
                ok = false;
                mlx_array_free(qz); mlx_array_free(sc); mlx_array_free(val); val = mlx_array_new();
                break;
            }
            float* buf = (float*)malloc((size_t)out * (size_t)in * sizeof(float));
            int rc = (buf == NULL) ? -1
                 : llmcodec_awq_dequant((const uint32_t*)mlx_array_data_int32(val),
                                   (const uint32_t*)mlx_array_data_int32(qz),
                                   (const uint16_t*)(const void*)mlx_array_data_float16(sc),
                                   4, group, out, in, buf);
            if (rc != 0) {
                fprintf(stderr, "[awq] %s: llmcodec_awq_dequant rc=%d\n", base, rc);
                free(buf); ok = false;
                mlx_array_free(qz); mlx_array_free(sc); mlx_array_free(val); val = mlx_array_new();
                break;
            }
            const int wshape[2] = {out, in};
            mlx_array w32 = mlx_array_new_data(buf, wshape, 2, MLX_FLOAT32);
            free(buf);
            mlx_array w16 = mlx_array_new();
            bool aok = (w32.ctx != NULL) && mlx_astype(&w16, w32, MLX_FLOAT16, stream) == 0 && mlx_array_eval(w16) == 0;
            mlx_array_free(w32);
            snprintf(nm, sizeof nm, "%s.weight", base);
            if (aok) aok = mlx_map_string_to_array_insert(synth, nm, w16) == 0;
            if (!aok) { fprintf(stderr, "[awq] %s: cast/insert F16 fallita\n", base); ok = false; }
            mlx_array_free(w16);
            mlx_array_free(qz); mlx_array_free(sc);
            mlx_array_free(val); val = mlx_array_new();
            if (ok) n_dequant++;
        }
        mlx_array_free(val);
        mlx_map_string_to_array_iterator_free(it);
    }
    if (ok && n_dequant > 0) {
        mlx_map_string_to_array* ns =
            (mlx_map_string_to_array*)realloc(weights->shards,
                                              (size_t)(weights->count + 1) * sizeof(mlx_map_string_to_array));
        if (ns == NULL) { mlx_map_string_to_array_free(synth); return -1; }
        weights->shards = ns;
        weights->shards[weights->count++] = synth;
        weights->cap = weights->count;
    } else {
        mlx_map_string_to_array_free(synth);
    }
    return ok ? n_dequant : -1;
}

/* ── GPTQ / bitsandbytes-NF4 — dequant a load ──────────────────────────────────────────────────
 * Stesso hook di awq_synthesize_weights: chiamata SUBITO dopo load_all_shards e PRIMA di
 * qualunque multishard_get (che MEMOIZZA i miss), dequantizza UNA VOLTA i checkpoint 4-bit privi
 * di `<base>.weight` plain in F16 [out,in], iniettandoli come `<base>.weight` in uno shard
 * SINTETICO (poi multishard_get_triple li trova senza modifiche; quant_mode resta "none").
 *  - GPTQ (`quantization_config.quant_method="gptq"`, AutoGPTQ/gptqmodel): `<base>.qweight` I32
 *    [in/(32/bits),out] + `<base>.qzeros` I32 [in/group,out/(32/bits)] + `<base>.scales` F16
 *    [in/group,out] + `<base>.g_idx` I32 [in] -> llmcodec_gptq_dequant (llmcodec_quant_gptq.h). La presenza
 *    di `.g_idx` distingue GPTQ da AWQ (che condivide qweight/qzeros/scales ma non ha g_idx).
 *  - NF4 (`quant_method="bitsandbytes"`, `bnb_4bit_quant_type="nf4"`): `<base>.weight` U8 packed
 *    (2 nibble/byte, nibble ALTO = elemento pari) + `.weight.absmax` U8 (indici double-quant) +
 *    `.weight.nested_absmax`/`.weight.nested_quant_map` (double-quant a blocchi da 256) +
 *    `.weight.quant_map` (codebook NF4 16 voci) + `.weight.quant_state.bitsandbytes__nf4`
 *    (tensore U8 = JSON con shape/blocksize/nested_offset) -> llmcodec_nf4_dequant (llmcodec_quant_nf4.h).
 * Ritorna il numero di tensori dequantizzati (0 = nessun match, no-op), -1 su errore. */
int gptq_nf4_synthesize_weights(MultiShardWeights* weights, mlx_stream stream) {
    static const char QW_SUF[] = ".qweight";
    static const char QS_NF4[] = ".weight.quant_state.bitsandbytes__nf4";
    const size_t QW_L = sizeof(QW_SUF) - 1;      /* 8 */
    const size_t QS_L = sizeof(QS_NF4) - 1;      /* 37 */
    if (weights == NULL || weights->count <= 0) return 0;
    mlx_map_string_to_array synth = mlx_map_string_to_array_new();
    if (synth.ctx == NULL) return -1;
    int n_dequant = 0;
    bool ok = true;
    char base[240], nm[320];

    for (int si = 0; ok && si < weights->count; si++) {
        mlx_map_string_to_array map = weights->shards[si];
        if (map.ctx == NULL) continue;
        mlx_map_string_to_array_iterator it = mlx_map_string_to_array_iterator_new(map);
        if (it.ctx == NULL) continue;
        const char* key = NULL;
        mlx_array val = mlx_array_new();
        while (ok && mlx_map_string_to_array_iterator_next(&key, &val, it) == 0) {
            size_t kl = (key != NULL) ? strlen(key) : 0;

            /* ── GPTQ: `<base>.qweight` con sibling `<base>.g_idx` ─────────────────────────── */
            if (key != NULL && kl > QW_L && strcmp(key + kl - QW_L, QW_SUF) == 0) {
                size_t bl = kl - QW_L;
                if (bl >= sizeof(base)) { fprintf(stderr, "[gptq] nome troppo lungo: %s\n", key); ok = false; break; }
                memcpy(base, key, bl); base[bl] = '\0';
                snprintf(nm, sizeof nm, "%s.g_idx", base);
                mlx_array gidx = mlx_array_new();
                if (!multishard_get(weights, nm, &gidx)) {         /* senza g_idx e' AWQ */
                    mlx_array_free(gidx); mlx_array_free(val); val = mlx_array_new();
                    continue;
                }
                mlx_array qz = mlx_array_new(), sc = mlx_array_new();
                snprintf(nm, sizeof nm, "%s.qzeros", base);
                bool have_z = multishard_get(weights, nm, &qz);
                snprintf(nm, sizeof nm, "%s.scales", base);
                bool have_s = multishard_get(weights, nm, &sc);
                if (!have_z || !have_s || mlx_array_eval(val) != 0 || mlx_array_eval(qz) != 0 ||
                    mlx_array_eval(sc) != 0 || mlx_array_eval(gidx) != 0) {
                    fprintf(stderr, "[gptq] %s: qzeros/scales/g_idx assenti o eval fallita\n", base);
                    ok = false;
                    mlx_array_free(qz); mlx_array_free(sc); mlx_array_free(gidx);
                    mlx_array_free(val); val = mlx_array_new(); break;
                }
                if (mlx_array_ndim(val) != 2 || mlx_array_ndim(qz) != 2 || mlx_array_ndim(sc) != 2 ||
                    mlx_array_dtype(val) != MLX_INT32 || mlx_array_dtype(qz) != MLX_INT32 ||
                    mlx_array_dtype(sc) != MLX_FLOAT16 || mlx_array_dtype(gidx) != MLX_INT32) {
                    fprintf(stderr, "[gptq] %s: dtype inattesi (qweight/qzeros/g_idx I32, scales F16)\n", base);
                    ok = false;
                    mlx_array_free(qz); mlx_array_free(sc); mlx_array_free(gidx);
                    mlx_array_free(val); val = mlx_array_new(); break;
                }
                const int* qws = mlx_array_shape(val);
                const int* qzs = mlx_array_shape(qz);
                const int* scs = mlx_array_shape(sc);
                const int in = (int)mlx_array_dim(gidx, 0);
                const int out = qws[1];
                if (in <= 0 || out <= 0 || qws[0] <= 0 || qzs[0] <= 0 || scs[1] != out ||
                    scs[0] != qzs[0] || (in % qws[0]) != 0) {
                    fprintf(stderr, "[gptq] %s: shape incoerenti in=%d out=%d qw=[%d,%d] qz=[%d,%d] sc=[%d,%d]\n",
                            base, in, out, qws[0], qws[1], qzs[0], qzs[1], scs[0], scs[1]);
                    ok = false;
                    mlx_array_free(qz); mlx_array_free(sc); mlx_array_free(gidx);
                    mlx_array_free(val); val = mlx_array_new(); break;
                }
                const int per = in / qws[0];                       /* 32/bits */
                const int bits = 32 / per;
                const int group = in / qzs[0];
                if ((32 % per) != 0 || (out % per) != 0 || qzs[1] != out / per ||
                    (in % group) != 0 || (bits != 2 && bits != 4 && bits != 8)) {
                    fprintf(stderr, "[gptq] %s: packing incoerente per=%d bits=%d group=%d\n", base, per, bits, group);
                    ok = false;
                    mlx_array_free(qz); mlx_array_free(sc); mlx_array_free(gidx);
                    mlx_array_free(val); val = mlx_array_new(); break;
                }
                float* buf = (float*)malloc((size_t)out * (size_t)in * sizeof(float));
                int rc = (buf == NULL) ? -1
                     : llmcodec_gptq_dequant((const uint32_t*)mlx_array_data_int32(val),
                                        (const uint32_t*)mlx_array_data_int32(qz),
                                        (const uint16_t*)(const void*)mlx_array_data_float16(sc),
                                        (const uint32_t*)mlx_array_data_int32(gidx),
                                        bits, group, out, in, buf);
                mlx_array_free(qz); mlx_array_free(sc); mlx_array_free(gidx);
                if (rc != 0) {
                    fprintf(stderr, "[gptq] %s: llmcodec_gptq_dequant rc=%d\n", base, rc);
                    free(buf); ok = false; mlx_array_free(val); val = mlx_array_new(); break;
                }
                const int wshape[2] = {out, in};
                mlx_array w32 = mlx_array_new_data(buf, wshape, 2, MLX_FLOAT32);
                free(buf);
                mlx_array w16 = mlx_array_new();
                bool aok = (w32.ctx != NULL) && mlx_astype(&w16, w32, MLX_FLOAT16, stream) == 0 &&
                           mlx_array_eval(w16) == 0;
                mlx_array_free(w32);
                snprintf(nm, sizeof nm, "%s.weight", base);
                if (aok) aok = mlx_map_string_to_array_insert(synth, nm, w16) == 0;
                if (!aok) { fprintf(stderr, "[gptq] %s: cast/insert F16 fallita\n", base); ok = false; }
                mlx_array_free(w16);
                mlx_array_free(val); val = mlx_array_new();
                if (ok) n_dequant++;
                continue;
            }

            /* ── NF4/BNB: `<base>.weight.quant_state.bitsandbytes__nf4` ────────────────────── */
            if (key != NULL && kl > QS_L && strcmp(key + kl - QS_L, QS_NF4) == 0) {
                size_t bl = kl - QS_L;
                if (bl >= sizeof(base)) { fprintf(stderr, "[nf4] nome troppo lungo: %s\n", key); ok = false; break; }
                memcpy(base, key, bl); base[bl] = '\0';

                mlx_array wq = mlx_array_new(), ab = mlx_array_new();
                mlx_array nam = mlx_array_new(), ncm = mlx_array_new(), cm = mlx_array_new();
                snprintf(nm, sizeof nm, "%s.weight", base);
                bool have_w = multishard_get(weights, nm, &wq);
                snprintf(nm, sizeof nm, "%s.weight.absmax", base);
                bool have_ab = multishard_get(weights, nm, &ab);
                snprintf(nm, sizeof nm, "%s.weight.nested_absmax", base);
                bool have_nam = multishard_get(weights, nm, &nam);
                snprintf(nm, sizeof nm, "%s.weight.nested_quant_map", base);
                bool have_ncm = multishard_get(weights, nm, &ncm);
                snprintf(nm, sizeof nm, "%s.weight.quant_map", base);
                bool have_cm = multishard_get(weights, nm, &cm);
                const bool nested = have_nam && have_ncm;
                if (!have_w || !have_ab || !have_cm ||
                    mlx_array_eval(val) != 0 || mlx_array_eval(wq) != 0 ||
                    mlx_array_eval(ab) != 0 || mlx_array_eval(cm) != 0 ||
                    (nested && (mlx_array_eval(nam) != 0 || mlx_array_eval(ncm) != 0)) ||
                    mlx_array_dtype(val) != MLX_UINT8 || mlx_array_dtype(wq) != MLX_UINT8 ||
                    mlx_array_dtype(cm) != MLX_FLOAT32 ||
                    (nested ? (mlx_array_dtype(ab) != MLX_UINT8 ||
                               mlx_array_dtype(nam) != MLX_FLOAT32 ||
                               mlx_array_dtype(ncm) != MLX_FLOAT32)
                            : (mlx_array_dtype(ab) != MLX_FLOAT32))) {
                    fprintf(stderr, "[nf4] %s: tensori/dtype inattesi (packed/absmax U8, codebook F32)\n", base);
                    ok = false;
                    mlx_array_free(wq); mlx_array_free(ab); mlx_array_free(nam);
                    mlx_array_free(ncm); mlx_array_free(cm);
                    mlx_array_free(val); val = mlx_array_new(); break;
                }
                /* quant_state = JSON impacchettato in un tensore U8 (bitsandbytes pack_dict_to_tensor). */
                const uint8_t* jb = mlx_array_data_uint8(val);
                size_t jl = mlx_array_size(val);
                char* js = (char*)malloc(jl + 1);
                int out = 0, in = 0, blocksize = 64, nbs = 256;
                float offset = 0.0f;
                bool parsed = false;
                if (js != NULL) { memcpy(js, jb, jl); js[jl] = '\0';
                    cJSON* qs = cJSON_Parse(js);
                    free(js);
                    if (qs != NULL) {
                        cJSON* sh = cJSON_GetObjectItemCaseSensitive(qs, "shape");
                        cJSON* bs = cJSON_GetObjectItemCaseSensitive(qs, "blocksize");
                        cJSON* nb = cJSON_GetObjectItemCaseSensitive(qs, "nested_blocksize");
                        cJSON* of = cJSON_GetObjectItemCaseSensitive(qs, "nested_offset");
                        if (cJSON_IsArray(sh) && cJSON_GetArraySize(sh) == 2) {
                            cJSON* d0 = cJSON_GetArrayItem(sh, 0);
                            cJSON* d1 = cJSON_GetArrayItem(sh, 1);
                            if (cJSON_IsNumber(d0) && cJSON_IsNumber(d1)) {
                                out = (int)d0->valuedouble; in = (int)d1->valuedouble; parsed = true;
                            }
                        }
                        if (cJSON_IsNumber(bs)) blocksize = (int)bs->valuedouble;
                        if (cJSON_IsNumber(nb)) nbs = (int)nb->valuedouble;
                        if (cJSON_IsNumber(of)) offset = (float)of->valuedouble;
                        cJSON_Delete(qs);
                    }
                }
                if (!parsed || out <= 0 || in <= 0 || blocksize <= 0) {
                    fprintf(stderr, "[nf4] %s: quant_state non parsato (shape/blocksize)\n", base);
                    ok = false;
                    mlx_array_free(wq); mlx_array_free(ab); mlx_array_free(nam);
                    mlx_array_free(ncm); mlx_array_free(cm);
                    mlx_array_free(val); val = mlx_array_new(); break;
                }
                const size_t n = (size_t)out * (size_t)in;
                float* buf = (float*)malloc(n * sizeof(float));
                int rc = (buf == NULL) ? -1
                     : llmcodec_nf4_dequant(mlx_array_data_uint8(wq),
                                       nested ? (const void*)mlx_array_data_uint8(ab)
                                              : (const void*)mlx_array_data_float32(ab),
                                       nested ? mlx_array_data_float32(nam) : NULL,
                                       nested ? mlx_array_data_float32(ncm) : NULL,
                                       mlx_array_data_float32(cm),
                                       offset, blocksize, nbs, n, buf);
                mlx_array_free(wq); mlx_array_free(ab); mlx_array_free(nam);
                mlx_array_free(ncm); mlx_array_free(cm);
                if (rc != 0) {
                    fprintf(stderr, "[nf4] %s: llmcodec_nf4_dequant rc=%d\n", base, rc);
                    free(buf); ok = false; mlx_array_free(val); val = mlx_array_new(); break;
                }
                const int wshape[2] = {out, in};
                mlx_array w32 = mlx_array_new_data(buf, wshape, 2, MLX_FLOAT32);
                free(buf);
                mlx_array w16 = mlx_array_new();
                bool aok = (w32.ctx != NULL) && mlx_astype(&w16, w32, MLX_FLOAT16, stream) == 0 &&
                           mlx_array_eval(w16) == 0;
                mlx_array_free(w32);
                snprintf(nm, sizeof nm, "%s.weight", base);
                if (aok) aok = mlx_map_string_to_array_insert(synth, nm, w16) == 0;
                if (!aok) { fprintf(stderr, "[nf4] %s: cast/insert F16 fallita\n", base); ok = false; }
                mlx_array_free(w16);
                mlx_array_free(val); val = mlx_array_new();
                if (ok) n_dequant++;
                continue;
            }

            mlx_array_free(val); val = mlx_array_new();
        }
        mlx_array_free(val);
        mlx_map_string_to_array_iterator_free(it);
    }
    if (ok && n_dequant > 0) {
        /* PREPEND (non append): per NF4 il checkpoint ha GIA' una chiave `<base>.weight` (il
         * blob U8 packed). multishard_get scandisce gli shard in ordine e restituisce il PRIMO
         * match: mettendo lo shard sintetico per primo, il `.weight` F16 dequantizzato vince
         * sul packed. (Per GPTQ/AWQ il packed non esiste, quindi prepend e' comunque corretto.) */
        mlx_map_string_to_array* ns =
            (mlx_map_string_to_array*)realloc(weights->shards,
                                              (size_t)(weights->count + 1) * sizeof(mlx_map_string_to_array));
        if (ns == NULL) { mlx_map_string_to_array_free(synth); return -1; }
        memmove(ns + 1, ns, (size_t)weights->count * sizeof(mlx_map_string_to_array));
        ns[0] = synth;
        weights->shards = ns;
        weights->count += 1;
        weights->cap = weights->count;
    } else {
        mlx_map_string_to_array_free(synth);
    }
    return ok ? n_dequant : -1;
}

/* ── TurboQuant (PolarQuant-MLX) — dequant a load ──────────────────────────────────────────────
 * I checkpoint `mode:"turboquant"` NON hanno un `.weight` plain: hanno `<base>.weight` U32 (indici)
 * + `<base>.scales` F16 [.., in/64] + `<base>.signs` F16 [in] + `<base>.codebook` F16 (8 = 3-bit
 * Lloyd-Max N(0,1), 3 = ternary {-c,0,+c}, c=1.22401). I pesi sono in dominio RUOTATO:
 * `W_rot = Hadamard(W·signs)`. Qui si DEQUANTIZZA UNA VOLTA (indici→codebook→scala→Hadamard
 * inversa→signs) in `.weight` F16 [out,in] e si inietta in uno shard SINTETICO in TESTA (come
 * gptq_nf4) così vince il lookup su `<base>.weight` U32. Formato di
 * github.com/manjunathshiva/turboquant-mlx (decode validato identico alla libreria). Il decode
 * per-riga (Hadamard inversa inclusa) vive in `llm_mlx_tq_decode.h`, condiviso col path SWAP. */

int turboquant_synthesize_weights(MultiShardWeights* weights, mlx_stream stream) {
    if (weights == NULL || weights->count <= 0) return 0;
    mlx_map_string_to_array synth = mlx_map_string_to_array_new();
    if (synth.ctx == NULL) return -1;
    int n_dec = 0; bool ok = true;
    char base[240], nm[288];
    for (int si = 0; ok && si < weights->count; si++) {
        mlx_map_string_to_array map = weights->shards[si];
        if (map.ctx == NULL) continue;
        char** bases = NULL; int nb = 0, capb = 0;
        mlx_map_string_to_array_iterator it = mlx_map_string_to_array_iterator_new(map);
        if (it.ctx == NULL) continue;
        const char* key = NULL; mlx_array val = mlx_array_new();
        while (mlx_map_string_to_array_iterator_next(&key, &val, it) == 0) {
            size_t kl = key ? strlen(key) : 0;
            if (kl > 9 && strcmp(key + kl - 9, ".codebook") == 0) {
                /* SWAP esperti ATTIVO (ingest aperto): gli esperti ROUTED (switch_mlp) NON si
                 * dequantizzano a load (sarebbero tutti residenti -> nessun risparmio); li decodifica
                 * on-the-fly il path di streaming (`llm_mlx_tq_stack`). shared_expert/attention
                 * restano dequantizzati. `swap_active` (non `swap_on`) garantisce che se l'ingest
                 * non apre si resti resident, mai spazzatura. */
                if (llm_mlx_moe_swap_active() && strstr(key, "switch_mlp") != NULL) {
                    mlx_array_free(val); val = mlx_array_new();
                    continue;
                }
                if (nb == capb) { capb = capb ? capb * 2 : 16; bases = (char**)realloc(bases, (size_t)capb * sizeof(char*)); }
                bases[nb++] = strndup(key, kl - 9);
            }
            mlx_array_free(val); val = mlx_array_new();
        }
        mlx_array_free(val); mlx_map_string_to_array_iterator_free(it);
        for (int b = 0; ok && b < nb; b++) {
            snprintf(base, sizeof base, "%s", bases[b]);
            mlx_array w = mlx_array_new(), sc = mlx_array_new(), sg = mlx_array_new(), cb = mlx_array_new();
            snprintf(nm, sizeof nm, "%s.weight", base);   bool gw = multishard_get(weights, nm, &w);
            snprintf(nm, sizeof nm, "%s.scales", base);   bool gs = multishard_get(weights, nm, &sc);
            snprintf(nm, sizeof nm, "%s.signs", base);    bool gg = multishard_get(weights, nm, &sg);
            snprintf(nm, sizeof nm, "%s.codebook", base); bool gc = multishard_get(weights, nm, &cb);
            if (!gw || !gs || !gg || !gc ||
                mlx_array_eval(w) || mlx_array_eval(sc) || mlx_array_eval(sg) || mlx_array_eval(cb)) {
                fprintf(stderr, "[turboquant] %s: weight/scales/signs/codebook mancanti o eval fallita (w=%d s=%d g=%d c=%d)\n",
                        base, (int)gw, (int)gs, (int)gg, (int)gc);
                ok = false; mlx_array_free(w); mlx_array_free(sc); mlx_array_free(sg); mlx_array_free(cb); break;
            }
            int ndw = mlx_array_ndim(w);
            const int* ws = mlx_array_shape(w);
            int E = (ndw == 3) ? ws[0] : 1;
            int rows = (ndw == 3) ? ws[1] : ws[0];
            int ngrp = mlx_array_shape(sc)[mlx_array_ndim(sc) - 1];
            int in_dim = ngrp * 64;
            int cbn = mlx_array_shape(cb)[0];
            int trit = (cbn == 3);
            int bits = trit ? 0 : (cbn == 8 ? 3 : (cbn == 4 ? 2 : 4));
            int perw = trit ? 20 : (32 / bits);
            int packed = ws[ndw - 1];
            int need_cols = trit ? (in_dim + 19) / 20 : (in_dim + perw - 1) / perw;
            if (!llm_mlx_tq_pow2(in_dim) || need_cols > packed) {
                fprintf(stderr, "[turboquant] %s: shape inattese in_dim=%d packed=%d cbn=%d\n", base, in_dim, packed, cbn);
                ok = false; mlx_array_free(w); mlx_array_free(sc); mlx_array_free(sg); mlx_array_free(cb); break;
            }
            /* ── TQ DENSO (2-D): NON si dequantizza a load. `.weight` resta PACKED (usato dal kernel
             * TQ a runtime); si inserisce solo `.biases`=signs (per la rotazione input nel decode).
             * Prima si materializzava F16 (es. lm_head [248320,2048] = ~2GB f32 + 1GB f16 SPRECATI). */
            if (llm_mlx_tq_enabled() && ndw == 2) {
                const mlx_dtype odt = g_bf16_all ? g_act16 : MLX_FLOAT16;
                mlx_array sgo = mlx_array_new();
                bool okd = true;
                if (mlx_array_dtype(sg) == odt) mlx_array_set(&sgo, sg);
                else if (mlx_astype(&sgo, sg, odt, stream) != 0) okd = false;
                if (okd) { snprintf(nm, sizeof nm, "%s.biases", base); okd = mlx_map_string_to_array_insert(synth, nm, sgo) == 0; }
                mlx_array_free(sgo);
                mlx_array_free(w); mlx_array_free(sc); mlx_array_free(sg); mlx_array_free(cb);
                if (!okd) { fprintf(stderr, "[turboquant] %s: insert biases fallita\n", base); ok = false; }
                else n_dec++;
                continue;
            }
            const uint32_t* wd = mlx_array_data_uint32(w);
            const uint16_t* scd = (const uint16_t*)(const void*)mlx_array_data_float16(sc);
            const uint16_t* sgd = (const uint16_t*)(const void*)mlx_array_data_float16(sg);
            const uint16_t* cbd = (const uint16_t*)(const void*)mlx_array_data_float16(cb);
            if (!wd || !scd || !sgd || !cbd) { fprintf(stderr, "[turboquant] %s: data NULL\n", base); ok = false; mlx_array_free(w); mlx_array_free(sc); mlx_array_free(sg); mlx_array_free(cb); break; }
            /* Hoist F32 di codebook/signs: nel decode per-riga evita la conversione f16 per elemento. */
            float cbf[16];
            if (cbn < 1 || cbn > 16) { fprintf(stderr, "[turboquant] %s: cbn=%d inatteso\n", base, cbn); ok = false; mlx_array_free(w); mlx_array_free(sc); mlx_array_free(sg); mlx_array_free(cb); break; }
            for (int i = 0; i < cbn; i++) cbf[i] = llmcodec_f16_to_f32(cbd[i]);
            float* sgf = (float*)malloc((size_t)in_dim * sizeof(float));
            if (!sgf) { ok = false; mlx_array_free(w); mlx_array_free(sc); mlx_array_free(sg); mlx_array_free(cb); break; }
            for (int i = 0; i < in_dim; i++) sgf[i] = llmcodec_f16_to_f32(sgd[i]);
            float* ob = (float*)malloc((size_t)E * rows * in_dim * sizeof(float));
            if (!ob) { free(sgf); ok = false; mlx_array_free(w); mlx_array_free(sc); mlx_array_free(sg); mlx_array_free(cb); break; }
            for (int e = 0; e < E; e++) {
                for (int r = 0; r < rows; r++) {
                    const uint32_t* wrow = wd + ((size_t)e * rows + r) * packed;
                    float* orow = ob + ((size_t)e * rows + r) * in_dim;
                    const uint16_t* scrow = scd + ((size_t)e * rows + r) * ngrp;
                    llm_mlx_tq_decode_row(wrow, scrow, cbf, sgf, in_dim, ngrp, trit, bits, orow);
                }
            }
            free(sgf);
            int oshape3[3] = { E, rows, in_dim }, oshape2[2] = { rows, in_dim };
            mlx_array of = mlx_array_new_data(ob, ndw == 3 ? oshape3 : oshape2, ndw, MLX_FLOAT32); free(ob);
            mlx_array o16 = mlx_array_new();
            /* Nel dtype di ATTIVAZIONE (come i synth AWQ/GPTQ sopra): evita l'AsType per-token dei
             * pesi tq nel path lineare (attention/shared), che oggi dominava il decode. */
            const mlx_dtype odt = g_bf16_all ? g_act16 : MLX_FLOAT16;
            bool aok = of.ctx && mlx_astype(&o16, of, odt, stream) == 0 && mlx_array_eval(o16) == 0;
            mlx_array_free(of);
            snprintf(nm, sizeof nm, "%s.weight", base);
            if (aok) aok = mlx_map_string_to_array_insert(synth, nm, o16) == 0;
            if (!aok) { fprintf(stderr, "[turboquant] %s: insert fallita\n", base); ok = false; }
            mlx_array_free(o16);
            mlx_array_free(w); mlx_array_free(sc); mlx_array_free(sg); mlx_array_free(cb);
            if (ok) n_dec++;
        }
        for (int b = 0; b < nb; b++) free(bases[b]);
        free(bases);
    }
    if (ok && n_dec > 0) {
        mlx_map_string_to_array* ns = (mlx_map_string_to_array*)malloc((size_t)(weights->count + 1) * sizeof(mlx_map_string_to_array));
        if (ns == NULL) { mlx_map_string_to_array_free(synth); return -1; }
        memmove(ns + 1, weights->shards, (size_t)weights->count * sizeof(mlx_map_string_to_array));
        ns[0] = synth;
        weights->shards = ns; weights->count += 1; weights->cap = weights->count;
    } else {
        mlx_map_string_to_array_free(synth);
    }
    return ok ? n_dec : -1;
}

/* LLM_PRETOUCH_WEIGHTS (2026-09-07, default ON dal 2026-09-07 pomeriggio -- misurato su
 * GLM-4.5-Air reale a macchina idle: decode 24 token 9.264s (2.6 tok/s) -> 3.354s (7.2 tok/s,
 * 2.76x), tempo totale processo 31.21s -> 23.63s (il pretouch stesso costa solo 4.156s per
 * 1698 tensori -- il totale load+decode e' PIU' basso, non solo spostato), output greedy
 * bit-esatto identico, picco RSS 52GB su 128GB (zero swap). LLM_NO_PRETOUCH_WEIGHTS=1 per
 * disattivare (stesso schema opt-out di LLM_NO_ASYNC_DECODE) se necessario per un caso non
 * ancora misurato (es. RAM molto piu' stretta, dove materializzare l'intero modello invece
 * del sottoinsieme instradato potrebbe non starci). Vedi
 * Knowledge/LLMEnv-RapidMLX-Gap-Closure-Strategy.md §3.4/P1.1 per i dettagli del meccanismo:
 * mlx_load_safetensors
 * (via load_all_shards sopra) non chiama mai mlx_array_eval -- i tensori restano nodi lazy
 * `Load` (mlx/backend/common/load.cpp, Load::eval_cpu fa la vera read() da disco). Per un
 * modello denso questo non cambia nulla: il primo forward tocca TUTTI i pesi comunque. Per
 * un modello MoE con molti esperti instradati (GLM-4.5-Air: 128), il routing top-k tocca
 * pochi esperti per token -- ogni esperto MAI toccato prima forza una materializzazione al
 * primo uso, concentrando un costo fisso (misurato: C~11s su GLM-4.5-Air) sui primi token
 * di OGNI sessione fresca invece che sul caricamento. Stesso identico meccanismo del bug
 * gia' risolto per la tabella N-gram di qwen4_exp (vedi commento sopra
 * qwen4_exp_ngram_table_open, MAI mlx_load_safetensors su quel file per lo stesso motivo --
 * ma li' la tabella (30GB) va tenuta lazy apposta, qui i pesi del modello sono comunque
 * destinati a stare residenti per servire l'inferenza, quindi materializzarli subito e'
 * corretto, non uno spreco).
 *
 * Valuta TUTTI gli array di TUTTI gli shard in una sola mlx_eval (batch, cosi' le singole
 * read() vengono accodate insieme sul thread pool IO di MLX invece che una alla volta) --
 * sposta il costo one-time dal primo decode al load, dove l'utente gia' si aspetta un'attesa.
 * Zero rischio di correttezza (non cambia NESSUN valore, solo QUANDO viene letto da disco). */
bool llm_mlx_pretouch_weights_enabled(void) {
    /* NIENTE cache statica: l'env e' letto ad OGNI load, cosi' l'opzione BASIC "swap" per-modello
     * (che setta LLM_NO_PRETOUCH_WEIGHTS) ha effetto anche se nel processo sono stati caricati
     * altri modelli prima. Con LLM_NO_PRETOUCH_WEIGHTS set, i pesi restano LAZY/file-backed
     * (mmap) -> paginati dal SO da disco = swap dei pesi su disco.
     *
     * Auto-OFF sotto streaming esperti (offload): pretouchare TUTTI i
     * tensori dello shard (inclusi gli switch_mlp.* degli esperti) vanifica lo SWAP —
     * materializza le decine di GB di esperti in RAM, cioe' esattamente ciò che lo streaming
     * evita (OOM/lockup su modelli >wired-limit tipo Qwen3.8-Flash-Next 104GB). Port del fix
     * GMBasic 87e75432b. */
    if (llm_mlx_moe_swap_on()) return false;
    return llm_ai_mlx_opt_env("LLM_NO_PRETOUCH_WEIGHTS") == NULL;
}


/* Kimi Linear (kimi_linear): MLA ASSORBITA. Il checkpoint salva `kv_b_proj` (proiezione
 * condivisa kv_lora -> [H, qk_nope+v_head]), NON i tensori assorbiti. mlx_lm li deriva in
 * Model.sanitize (kimi_linear.py:556-589):
 *   v  = dequant(kv_b_proj)             [H*(qn+vhd), kvl]
 *   v  = v.reshape(H, qn+vhd, kvl)
 *   wk = v[:, :qn, :].swapaxes(-1,-2)   [H, kvl, qn]   -> embed_q
 *   wv = v[:, qn:, :]                   [H, vhd, kvl]  -> unembed_out
 * poi ri-quantizza wk/wv (stessi bits/group). Replichiamo UNA VOLTA a load e iniettiamo uno
 * shard SINTETICO in coda a weights->shards[] (multishard_get li trova senza modifiche, come
 * per stack_mixtral_moe_experts). Solo layer full (MLA); i layer KDA non li hanno. */
bool kimi_linear_synthesize_absorbed_mla(const ModelConfig* cfg, MultiShardWeights* weights, mlx_stream stream) {
    const int H = cfg->n_heads, qn = cfg->qk_nope_head_dim, vhd = cfg->v_head_dim, kvl = cfg->kv_lora_rank;
    const int head_dim = qn + vhd;
    if (H <= 0 || qn <= 0 || vhd <= 0 || kvl <= 0) {
        fprintf(stderr, "kimi_absorbed_mla: config MLA incompleta (H=%d qn=%d vhd=%d kvl=%d)\n", H, qn, vhd, kvl);
        return false;
    }
    mlx_map_string_to_array synth = mlx_map_string_to_array_new();
    mlx_array null_gs = mlx_array_new();
    bool ok = true;
    char nb[256], nw[320];
    for (int l = 0; ok && l < cfg->n_layers; l++) {
        if (cfg->is_linear_layer != NULL && cfg->is_linear_layer[l]) continue;   /* KDA: niente MLA */
        tname(cfg->layer_prefix, l, "self_attn.kv_b_proj.weight", nb, sizeof nb);
        mlx_array kw = mlx_array_new();
        if (!multishard_get(weights, nb, &kw)) {
            fprintf(stderr, "kimi_absorbed_mla: %s mancante\n", nb);
            mlx_array_free(kw); ok = false; break;
        }
        mlx_array ks = mlx_array_new(), kb = mlx_array_new();
        tname(cfg->layer_prefix, l, "self_attn.kv_b_proj.scales", nb, sizeof nb);
        bool quant = multishard_get(weights, nb, &ks);
        if (quant) {
            tname(cfg->layer_prefix, l, "self_attn.kv_b_proj.biases", nb, sizeof nb);
            if (!multishard_get(weights, nb, &kb)) quant = false;
        }
        int bits = cfg->quant_bits, gsz = cfg->quant_group_size;
        int kw_last = (int)mlx_array_dim(kw, mlx_array_ndim(kw) - 1);
        if (kw_last > 0) bits = (kw_last * 32) / kvl;
        if (quant) { int sl = (int)mlx_array_dim(ks, mlx_array_ndim(ks) - 1); if (sl > 0) gsz = kvl / sl; }
        if (bits != cfg->quant_bits || gsz != cfg->quant_group_size)
            fprintf(stderr, "kimi_absorbed_mla[layer=%d]: gs/bits derivati %d/%d != config %d/%d\n",
                    l, gsz, bits, cfg->quant_group_size, cfg->quant_bits);
        mlx_array full = mlx_array_new();
        if (quant) {
            mlx_optional_int gso = {gsz, true}, bto = {bits, true};
            mlx_optional_dtype dto = {g_bf16_all ? g_act16 : MLX_FLOAT16, true};  /* scale nel dtype di attivazione: niente AsType per-token */
            ok = mlx_dequantize(&full, kw, ks, kb, gso, bto, "affine", null_gs, dto, stream) == 0;
        } else ok = mlx_array_set(&full, kw) == 0;
        mlx_array_free(kw); mlx_array_free(ks); mlx_array_free(kb);
        if (!ok) { fprintf(stderr, "kimi_absorbed_mla[layer=%d]: dequant fallita\n", l); mlx_array_free(full); break; }
        int sh3[3] = {H, head_dim, kvl};
        mlx_array r3 = mlx_array_new(), kn = mlx_array_new(), vv = mlx_array_new(), wk = mlx_array_new(), wv = mlx_array_new();
        ok = mlx_reshape(&r3, full, sh3, 3, stream) == 0;
        mlx_array_free(full);
        if (ok) { int st[3]={0,0,0}, sp[3]={H,qn,kvl}, sv[3]={1,1,1}; ok = mlx_slice(&kn, r3, st,3,sp,3,sv,3, stream) == 0; }
        if (ok) { int st[3]={0,qn,0}, sp[3]={H,head_dim,kvl}, sv[3]={1,1,1}; ok = mlx_slice(&vv, r3, st,3,sp,3,sv,3, stream) == 0; }
        mlx_array_free(r3);
        if (ok) ok = mlx_swapaxes(&wk, kn, 1, 2, stream) == 0;   /* [H,kvl,qn] */
        if (ok) ok = mlx_array_set(&wv, vv) == 0;                 /* [H,vhd,kvl] */
        mlx_array_free(kn); mlx_array_free(vv);
        const char* dst[2] = {"self_attn.embed_q", "self_attn.unembed_out"};
        mlx_array src[2] = {wk, wv};
        for (int t = 0; ok && t < 2; t++) {
            mlx_vector_array qv = mlx_vector_array_new();
            mlx_optional_int gso = {gsz, true}, bto = {bits, true};
            ok = mlx_quantize(&qv, src[t], gso, bto, "affine", null_gs, stream) == 0;
            if (ok && mlx_vector_array_size(qv) >= 2) {
                mlx_array qw = mlx_array_new(), qs = mlx_array_new(), qb = mlx_array_new();
                mlx_vector_array_get(&qw, qv, 0);
                mlx_vector_array_get(&qs, qv, 1);
                if (mlx_vector_array_size(qv) >= 3) mlx_vector_array_get(&qb, qv, 2);
                tname(cfg->layer_prefix, l, dst[t], nb, sizeof nb);
                snprintf(nw, sizeof nw, "%s.weight", nb);
                ok = mlx_map_string_to_array_insert(synth, nw, qw) == 0;
                snprintf(nw, sizeof nw, "%s.scales", nb);
                ok = ok && mlx_map_string_to_array_insert(synth, nw, qs) == 0;
                if (qb.ctx != NULL) { snprintf(nw, sizeof nw, "%s.biases", nb); ok = ok && mlx_map_string_to_array_insert(synth, nw, qb) == 0; }
                mlx_array_free(qw); mlx_array_free(qs); mlx_array_free(qb);
            }
            mlx_vector_array_free(qv);
            if (!ok) fprintf(stderr, "kimi_absorbed_mla[layer=%d]: quantize/insert %s fallita\n", l, dst[t]);
        }
        mlx_array_free(wk); mlx_array_free(wv);
    }
    if (ok) {
        weights->shards = (mlx_map_string_to_array*)realloc(weights->shards, (size_t)(weights->count + 1) * sizeof(mlx_map_string_to_array));
        if (weights->shards == NULL) { fprintf(stderr, "kimi_absorbed_mla: realloc shards fallita\n"); ok = false; }
        else { weights->shards[weights->count] = synth; weights->count += 1; }
    }
    if (!ok) mlx_map_string_to_array_free(synth);
    mlx_array_free(null_gs);
    return ok;
}

void llm_mlx_pretouch_weights(const MultiShardWeights* weights) {
    mlx_vector_array all = mlx_vector_array_new();
    size_t n_tensors = 0;
    for (int i = 0; i < weights->count; i++) {
        mlx_map_string_to_array_iterator it = mlx_map_string_to_array_iterator_new(weights->shards[i]);
        const char* key = NULL;
        mlx_array val = mlx_array_new();
        while (mlx_map_string_to_array_iterator_next(&key, &val, it) == 0 && key != NULL) {
            /* Streaming esperti: NON pretouchare gli esperti (switch_mlp.*) -> restano lazy e
             * vengono letti on-demand dallo store; il resto (attn/embed/norm/lm_head) va residente. */
            if (llm_mlx_moe_swap_on() && strstr(key, "switch_mlp")) continue;
            mlx_vector_array_append_value(all, val);
            n_tensors++;
        }
        mlx_map_string_to_array_iterator_free(it);
    }
    if (n_tensors > 0) {
        int rc = mlx_eval(all);
        (void)rc;
    }
    mlx_vector_array_free(all);
}

/* FASE 1b.3 (Mixtral): il modello reale tiene i pesi degli esperti separati per
 * indice ("block_sparse_moe.experts.{e}.w1/w2/w3.{weight,scales,biases}"), a
 * differenza del layout "stacked" di Qwen3-MoE (un solo tensore switch_mlp.* con
 * una dimensione esperto in più) -- confermato leggendo Model.sanitize() in
 * mlx_lm/models/mixtral.py: PRIMA di costruire il modello, mlx_lm impila (mx.stack)
 * gli 8 tensori per-esperto in uno stacked, e SOLO DOPO il resto del forward
 * (MixtralSparseMoeBlock/SwitchGLU) usa esattamente lo stesso meccanismo gather_qmm
 * di Qwen3-MoE. Replichiamo lo stesso passaggio qui, UNA SOLA VOLTA dopo il
 * caricamento degli shard (non ad ogni forward, che ricreerebbe la stessa copia ad
 * ogni token) -- il risultato viene inserito come uno shard SINTETICO aggiuntivo in
 * coda a weights->shards[], cosi' multishard_get/multishard_get_triple lo trovano
 * senza alcuna modifica a quelle funzioni. Mapping dei nomi verificato in
 * sanitize(): w1->gate_proj, w3->up_proj, w2->down_proj (attenzione: w2/w3 non
 * nell'ordine "intuitivo" -- verificato leggendo il sorgente, non assunto). */
bool stack_moe_experts(const ModelConfig* cfg, MultiShardWeights* weights, mlx_stream stream,
                              const char* base, const char* s0, const char* s1, const char* s2) {
    mlx_map_string_to_array synth = mlx_map_string_to_array_new();
    const struct { const char* src; const char* dst; } kinds[3] = {
        {s0, "gate_proj"}, {s1, "up_proj"}, {s2, "down_proj"},
    };
    static const char* parts[3] = {"weight", "scales", "biases"};
    bool ok = true;
    char name[256];
    for (int l = 0; ok && l < cfg->n_layers; l++) {
        for (int k = 0; ok && k < 3; k++) {
            for (int p = 0; ok && p < 3; p++) {
                mlx_vector_array vec = mlx_vector_array_new();
                bool part_present = true;
                for (int e = 0; ok && e < cfg->n_experts; e++) {
                    snprintf(name, sizeof(name), "model.layers.%d.%s.experts.%d.%s.%s",
                             l, base, e, kinds[k].src, parts[p]);
                    mlx_array t = mlx_array_new();
                    if (!multishard_get(weights, name, &t)) {
                        mlx_array_free(t);
                        if (p == 0) {   /* .weight obbligatorio; scales/biases opzionali (checkpoint F16 non quantizzato) */
                            fprintf(stderr, "stack_moe_experts: %s mancante\n", name);
                            ok = false;
                        } else {
                            part_present = false;
                        }
                        break;
                    }
                    mlx_vector_array_append_value(vec, t);
                    mlx_array_free(t);
                }
                mlx_array stacked = mlx_array_new();
                if (ok && part_present) {
                    ok = mlx_stack(&stacked, vec, stream) == 0;
                    if (!ok) fprintf(stderr, "stack_moe_experts: mlx_stack fallita layer=%d %s.%s\n", l, kinds[k].dst, parts[p]);
                }
                mlx_vector_array_free(vec);
                if (ok && part_present) {
                    snprintf(name, sizeof(name), "model.layers.%d.%s.switch_mlp.%s.%s",
                             l, base, kinds[k].dst, parts[p]);
                    ok = mlx_map_string_to_array_insert(synth, name, stacked) == 0;
                    if (!ok) fprintf(stderr, "stack_moe_experts: insert fallito per %s\n", name);
                }
                mlx_array_free(stacked);
            }
        }
    }
    if (ok) {
        /* Layout 1:1 con mlx_lm (sanitize fa weights.pop sugli esperti originali): dopo lo stack
         * gli originali per-esperto NON servono. La map c-API non ha erase, quindi ricostruisco
         * le shard tenendo SOLO i non-esperti + synth, poi libero le shard originali (senza questo
         * il motore tiene 2x il modello: Mixtral ~148GB di GPU -> memory thrash -> 0.2 tok/s). */
        mlx_map_string_to_array kept = mlx_map_string_to_array_new();
        mlx_map_string_to_array* old = weights->shards;
        int old_n = weights->count;
        mlx_map_string_to_array* news = NULL;
        int new_n = 0, new_cap = 2;
        news = (mlx_map_string_to_array*)malloc((size_t)new_cap * sizeof(mlx_map_string_to_array));
        news[new_n++] = kept;
        news[new_n++] = synth;
        char excl[64];
        snprintf(excl, sizeof(excl), ".%s.experts.", base);
        for (int si = 0; si < old_n; si++) {
            const char* key = NULL;
            mlx_array val = mlx_array_new();
            mlx_map_string_to_array_iterator it = mlx_map_string_to_array_iterator_new(old[si]);
            while (mlx_map_string_to_array_iterator_next(&key, &val, it) == 0) {
                if (key && strstr(key, excl) == NULL)
                    mlx_map_string_to_array_insert(kept, key, val);
                mlx_array_free(val);
                val = mlx_array_new();
            }
            mlx_array_free(val);
            mlx_map_string_to_array_iterator_free(it);
            mlx_map_string_to_array_free(old[si]);
        }
        free(old);
        /* multishard_get ha memoizzato tutti gli expert sorgente durante lo stack.
         * Quelle entry hanno come chiave il vecchio mw->shards e restano irraggiungibili
         * dopo la sostituzione, ma i loro handle trattengono ~76 GB di pesi originali.
         * Svuotarle QUI e' indispensabile: senza, kept+synth occupano 155.49 GB invece
         * dei 79.39 GB di mlx_lm e il forward finisce in swap. */
        llm_mlx_wcache_clear();
        weights->shards = news;
        weights->count = new_n;
        weights->cap = new_cap;
        /* ESPERIMENTO (LLM_STACK_EVAL, default OFF): materializza qui gli stack invece che al
         * primo forward. Misurato il 2026-09-09 con la timeline dell'allocatore: su Mixtral-8x22B
         * i pesi caricati occupano 79,39 GB e il PREFILL li porta a 130,41 su un limite di 130,57
         * (G0 nomina gli autori: Concatenate/ExpandDims da 384 MB degli stack, 24.192 MB nel grafo
         * del prefill). Gli stack sono pigri, quindi le loro 8 inputs (i pesi per-esperto originali,
         * che la mappe appena ricostruita NON contiene piu') restano vive finche' lo stack non e'
         * valutato; MLX libera gli inputs di un nodo subito dopo averlo_evalutato (transforms.cpp
         * "arr.detach()"). Se questa ipotesi e' vera, eval-per-stack a load riporta l'attivo verso
         * ~79 GB invece di lasciarlo salire a 130 al primo forward. */
        /* Il vecchio eval UNO-STACK-ALLA-VOLTA e' ora solo diagnostico. Con MLX corrente
         * conserva i buffer sorgente Load mentre materializza ogni copia stacked: sul
         * Mixtral-8x22B porta l'attivo 0 -> 152.20 GB (o 79.39 -> 155.49 GB se preceduto
         * dal pretouch) e manda la macchina in swap. Il loader valuta invece l'INTERA
         * mappa sanitizzata con una sola mlx_eval, esattamente come mlx_lm. */
        if (ok && llm_env_isset("LLM_STACK_EVAL_LEGACY")) {
            int done = 0, total = 0;
            size_t act0 = 0;
            mlx_get_active_memory(&act0);
            mlx_map_string_to_array_iterator cnt = mlx_map_string_to_array_iterator_new(synth);
            {
                const char* k = NULL;
                mlx_array v = mlx_array_new();
                while (mlx_map_string_to_array_iterator_next(&k, &v, cnt) == 0) { total++; mlx_array_free(v); v = mlx_array_new(); }
                mlx_array_free(v);
            }
            mlx_map_string_to_array_iterator_free(cnt);
            mlx_map_string_to_array_iterator it = mlx_map_string_to_array_iterator_new(synth);
            const char* key = NULL;
            mlx_array val = mlx_array_new();
            while (mlx_map_string_to_array_iterator_next(&key, &val, it) == 0) {
                if (mlx_array_eval(val) != 0) { ok = false; break; }
                done++;
                if (done % 27 == 0 || done == total) {
                    size_t act = 0, cac = 0;
                    mlx_get_active_memory(&act);
                    mlx_get_cache_memory(&cac);
                    fprintf(stderr, "[stack-eval] %d/%d attivi=%.2f GB cache=%.2f GB (base=%.2f GB) "
                                    "ultimos=%s\n", done, total, (double)act / 1e9, (double)cac / 1e9,
                            (double)act0 / 1e9, key ? key : "?");
                    fflush(stderr);
                }
                mlx_array_free(val);
                val = mlx_array_new();
            }
            mlx_array_free(val);
            mlx_map_string_to_array_iterator_free(it);
            mlx_clear_cache();
            fprintf(stderr, "[stack-eval] fine: ok=%d eval=%d/%d\n", ok, done, total);
            fflush(stderr);
        }
    } else {
        mlx_map_string_to_array_free(synth);
    }
    return ok;
}

/* Copia 1:1 di rapidMLX moe_fusion.py (vllm_mlx/moe_fusion.py, Apache-2.0):
 * fuse le proiezioni gate+up degli esperti in UN sol weighted-stack
 * [E, 2*inter, in] (concat lungo l'asse output = axis 1, lo stesso di rapid),
 * cosi' il decode emette UNA gather_qmm invece di due. Bit-identico perche'
 * la quantizzazione affine impacchetta ogni RIGA di output in modo indipendente
 * (stesso argomento del docstring rapido). I risultati vengono inseriti come
 * chiavi "...switch_mlp.gate_up_proj.{weight,scales[,biases]}" in uno shard
 * SINTETICO aggiuntivo, trovato da multishard_get_triple senza toccare nulla.
 * Layer non idonei (shape/dtype diversi, plain) restano NON fusi (fallback).
 * ⚠️ MISURATO 2026-09-03: su questo engine la versione fusa e' ~40x PIU' LENTA
 * (gen8 ctx128: 249s vs 6s) -- la split produce input non-contigui che fanno
 * degradare la gather_qmm successiva. Default quindi OFF: abilitare solo con
 * LLM_GU_FUSE=1 (sperimentale, non per produzione). */
static int g_gu_fuse_state = -1;
static bool llm_mlx_gu_fuse_enabled(void) {
    if (g_gu_fuse_state < 0) g_gu_fuse_state = (llm_ai_mlx_opt_env("LLM_GU_FUSE") != NULL) ? 1 : 0;
    return g_gu_fuse_state == 1;
}
bool fuse_mixtral_gate_up(const ModelConfig* cfg, MultiShardWeights* weights, mlx_stream stream) {
    if (!llm_mlx_gu_fuse_enabled()) return true; /* disabilitato: nessun cambiamento */
    if (cfg == NULL || weights == NULL || cfg->n_layers <= 0) return true;
    mlx_map_string_to_array fused = mlx_map_string_to_array_new();
    bool ok = true;
    int fused_layers = 0;
    for (int l = 0; ok && l < cfg->n_layers; l++) {
        char base[256];
        snprintf(base, sizeof(base), "model.layers.%d.block_sparse_moe.switch_mlp.gate_proj", l);
        mlx_array gw = mlx_array_new(), gs = mlx_array_new(), gb = mlx_array_new();
        if (!multishard_get_triple(weights, base, &gw, &gs, &gb)) { mlx_array_free(gw); mlx_array_free(gs); mlx_array_free(gb); ok = false; break; }
        snprintf(base, sizeof(base), "model.layers.%d.block_sparse_moe.switch_mlp.up_proj", l);
        mlx_array uw = mlx_array_new(), us = mlx_array_new(), ub = mlx_array_new();
        if (!multishard_get_triple(weights, base, &uw, &us, &ub)) { mlx_array_free(gw); mlx_array_free(gs); mlx_array_free(gb); mlx_array_free(uw); mlx_array_free(us); mlx_array_free(ub); ok = false; break; }
        /* Guardie strutturali (specchio di _can_fuse): quantizzati entrambi,
         * stesso group/bits/mode (dalla coppia scales), weight con stessa shape. */
        bool fuse = (gs.ctx != NULL) && (us.ctx != NULL);
        if (fuse) {
            const int* sg = mlx_array_shape(gw); const int* su = mlx_array_shape(uw);
            int ndg = (int)mlx_array_ndim(gw), ndu = (int)mlx_array_ndim(uw);
            fuse = (ndg == 3 && ndu == 3);
            if (fuse) for (int d = 0; d < ndg; d++) if (sg[d] != su[d]) { fuse = false; break; }
            if (fuse && mlx_array_dtype(gw) != mlx_array_dtype(uw)) fuse = false;
            bool gbias = (gb.ctx != NULL), ubias = (ub.ctx != NULL);
            if (gbias != ubias) fuse = false;
        }
        if (fuse) {
            mlx_array fw = mlx_array_new(), fs = mlx_array_new();
            mlx_vector_array vw = mlx_vector_array_new();
            mlx_vector_array_append_value(vw, gw);
            mlx_vector_array_append_value(vw, uw);
            ok = mlx_concatenate_axis(&fw, vw, /*axis=*/1, stream) == 0;
            mlx_vector_array_free(vw);
            if (ok) {
                mlx_vector_array vs = mlx_vector_array_new();
                mlx_vector_array_append_value(vs, gs);
                mlx_vector_array_append_value(vs, us);
                ok = mlx_concatenate_axis(&fs, vs, 1, stream) == 0;
                mlx_vector_array_free(vs);
            }
            mlx_array fb = mlx_array_new();
            if (ok && gb.ctx != NULL) {
                mlx_vector_array vb = mlx_vector_array_new();
                mlx_vector_array_append_value(vb, gb);
                mlx_vector_array_append_value(vb, ub);
                ok = mlx_concatenate_axis(&fb, vb, 1, stream) == 0;
                mlx_vector_array_free(vb);
            }
            if (ok) {
                snprintf(base, sizeof(base), "model.layers.%d.block_sparse_moe.switch_mlp.gate_up_proj.weight", l);
                ok = mlx_map_string_to_array_insert(fused, base, fw) == 0;
            }
            if (ok) {
                snprintf(base, sizeof(base), "model.layers.%d.block_sparse_moe.switch_mlp.gate_up_proj.scales", l);
                ok = mlx_map_string_to_array_insert(fused, base, fs) == 0;
            }
            if (ok && fb.ctx != NULL) {
                snprintf(base, sizeof(base), "model.layers.%d.block_sparse_moe.switch_mlp.gate_up_proj.biases", l);
                ok = mlx_map_string_to_array_insert(fused, base, fb) == 0;
            }
            mlx_array_free(fw); mlx_array_free(fs); mlx_array_free(fb);
            if (ok) fused_layers++;
        }
        mlx_array_free(gw); mlx_array_free(gs); mlx_array_free(gb);
        mlx_array_free(uw); mlx_array_free(us); mlx_array_free(ub);
    }
    if (!ok) { mlx_map_string_to_array_free(fused); return false; }
    if (fused_layers > 0) {
        /* append come shard sintetico in coda (multishard_get lo prova per ultimo). */
        if (weights->count >= weights->cap) {
            int ncap = weights->cap ? weights->cap * 2 : 8;
            mlx_map_string_to_array* ns = (mlx_map_string_to_array*)realloc(weights->shards, (size_t)ncap * sizeof(mlx_map_string_to_array));
            if (!ns) { mlx_map_string_to_array_free(fused); return false; }
            weights->shards = ns;
            weights->cap = ncap;
        }
        weights->shards[weights->count++] = fused;
        fprintf(stderr, "[moe_fusion] gate+up fusion applicata: %d layer Mixtral\n", fused_layers);
    } else {
        mlx_map_string_to_array_free(fused);
    }
    return true;
}
