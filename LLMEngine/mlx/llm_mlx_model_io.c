/* ============================================================================
 * Dominio model IO: write_mlx_meta_file, read_mlx_meta_file (swap meta), llm_ai_mlx_load_model, llm_ai_mlx_free_model.
 * Estratto VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llm_mlx_memgov.h"      /* B9: budget memoria (reserve/grow_min) */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llmengine_plugin.h"
#include "llm_mlx_internal.h"
#include "llm_mlx_ops.h"
#include "llm_mlx_kvpage.h"
#include "llm_mlx_kvdisk.h"
#include "llm_mlx_ngram.h"
#include "llm_mlx_qwen4exp.h"
#include "llm_dsv4_engine.h"
#include "llm_dsv41_engine.h"
#include "llm_mlx_which_path.h"
#include "llm_mlx_dispatch_profile.h"
#include <cJSON.h>
#include "llm_mlx_bonsai_hadamard.h"

#include <stdio.h>
#include <stdlib.h>
#include <mach/mach.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>
#include <sys/sysctl.h>

/* Store/cache esperti DS4 (LRU): bundle (gate+up+down) tenuti in RAM quando l'offload esperti è
 * attivo (parametro --offload-experts -> set_expert_swap). Nessuna variabile d'ambiente. */
#define LLM_DSV4_STORE_BUNDLES 2048

extern const LlmEnginePluginPrimMLX* g_primmlx_norm;
extern void* g_primmlx_norm_h;
/* FASE 2b (MLX): formato meta per lo swap su disco -- stesso spirito di LLMKV02 in
 * llmengine_llama.cpp, ma senza model_handle: il model handle di MLX e' un puntatore
 * in-processo (non un intero da registro come in llama.cpp), non ha senso serializzarlo --
 * il chiamante C# lo passa esplicitamente a llm_ai_mlx_session_swap_in (ha gia'
 * _modelHandle come campo). n_layers/hidden_size sono un controllo di coerenza col modello
 * a swap-in, non dati funzionali -- un modello diverso da quello di swap-out deve fallire
 * esplicitamente, non produrre stato corrotto. */
static const char kMlxMetaMagic[8] = {'G', 'M', 'B', 'M', 'X', '0', '1', '\0'};


bool write_mlx_meta_file(const char* path, const MlxSwapMeta* meta) {
    FILE* f = fopen(path, "wb");
    if (f == NULL) return false;
    bool ok = true;
    ok = ok && fwrite(kMlxMetaMagic, 1, sizeof(kMlxMetaMagic), f) == sizeof(kMlxMetaMagic);
    ok = ok && fwrite(&meta->cache_offset, sizeof(int32_t), 1, f) == 1;
    ok = ok && fwrite(&meta->pending_next_id, sizeof(int32_t), 1, f) == 1;
    uint8_t has_pending_byte = meta->has_pending ? 1 : 0;
    ok = ok && fwrite(&has_pending_byte, 1, 1, f) == 1;
    ok = ok && fwrite(&meta->n_layers, sizeof(int32_t), 1, f) == 1;
    ok = ok && fwrite(&meta->hidden_size, sizeof(int32_t), 1, f) == 1;
    fclose(f);
    return ok;
}

bool read_mlx_meta_file(const char* path, MlxSwapMeta* out) {
    FILE* f = fopen(path, "rb");
    if (f == NULL) return false;
    bool ok = true;
    char magic[sizeof(kMlxMetaMagic)] = {0};
    ok = ok && fread(magic, 1, sizeof(magic), f) == sizeof(magic);
    if (!ok || memcmp(magic, kMlxMetaMagic, sizeof(kMlxMetaMagic)) != 0) { fclose(f); return false; }
    int32_t cache_offset = 0, pending_next_id = 0, n_layers = 0, hidden_size = 0;
    ok = ok && fread(&cache_offset, sizeof(int32_t), 1, f) == 1;
    ok = ok && fread(&pending_next_id, sizeof(int32_t), 1, f) == 1;
    uint8_t has_pending_byte = 0;
    ok = ok && fread(&has_pending_byte, 1, 1, f) == 1;
    ok = ok && fread(&n_layers, sizeof(int32_t), 1, f) == 1;
    ok = ok && fread(&hidden_size, sizeof(int32_t), 1, f) == 1;
    fclose(f);
    if (!ok) return false;
    out->cache_offset = cache_offset;
    out->pending_next_id = pending_next_id;
    out->has_pending = has_pending_byte != 0;
    out->n_layers = n_layers;
    out->hidden_size = hidden_size;
    return true;
}





llm_ai_mlx_model_handle llm_ai_mlx_load_model(const char* model_dir) {
    if (getenv("LLM_MLX_ERRTRACE")) mlx_set_error_handler(llm_mlx_err_trace_handler, NULL, NULL);
    if (model_dir == NULL) return NULL;

    LlmMlxModel* m = (LlmMlxModel*)malloc(sizeof(LlmMlxModel));
    llm_mlx_model_init(m);
    { size_t _n = strlen(model_dir); if (_n >= sizeof(m->model_dir)) _n = sizeof(m->model_dir)-1; memcpy(m->model_dir, model_dir, _n); m->model_dir[_n] = '\0'; }
    llm_mlx_wcache_clear();   /* le chiavi del memo sono per-modello */
    llm_mlx_set_layer_ops(NULL, 0);    /* pipeline del layer: reset (canonica) finche' il .llmarch non la dichiara */
    llm_dsv4_set_layer_ops(NULL, 0);

    /* WIRING PLUGIN: la `.llmarch` puo' fornire la PRIMITIVA (MLX-aware) al posto del codice
     * built-in. Qui l'engine CONSULTA la descrizione e carica le primitive dichiarate (handle
     * opachi, zero copie); un plugin assente lascia il built-in — gap visibile, mai silenzioso. */
    {
        LlmDesc arch;
        if (llm_mlx_resolve_arch(model_dir, &arch) == 0) {
            for (int i = 0; i < arch.n_struct; i++) {
                const char* pr = arch.structures[i].prim;
                if (pr != NULL && *pr != 0) {
                    const LlmEnginePluginPrimMLX* p = llmengine_plugin_prim_mlx_get(pr);
                    if (p != NULL) {
                        fprintf(stderr, "[arch] primitiva '%s' (kind=%s) dal plugin per struttura '%s'\n",
                                p->name, p->kind, arch.structures[i].key);
                        if (strcmp(arch.structures[i].key, "norm") == 0) {
                            g_primmlx_norm = p; g_primmlx_norm_h = p->create(NULL);
                        }
                    }
                    else
                        fprintf(stderr, "[arch] primitiva '%s' dichiarata per '%s' ma plugin assente\n",
                                pr, arch.structures[i].key);
                }

                /* COMPOSIZIONE (livello 3): la struttura puo' essere una PIPELINE ordinata di
                 * primitive (`steps`). L'engine la risolve passo-passo (plugin MLX -> reference);
                 * un passo vuoto e' il built-in. Cosi' una struttura nuova = una lista di primitive. */
                const LlmDescStruct* S = &arch.structures[i];
                if (S->n_steps > 0) {
                    fprintf(stderr, "[arch] pipeline '%s' (%s):", S->key, S->type[0] ? S->type : "-");
                    for (int k = 0; k < S->n_steps; k++) {
                        const char* spr = S->steps[k].prim;
                        const char* src = "-";
                        if (spr != NULL && *spr) {
                            if (llmengine_plugin_prim_mlx_get(spr) != NULL) src = "mlx";
                            else if (llmengine_plugin_prim_get(spr) != NULL) src = "prim";
                            else if (llmengine_plugin_codec_get(spr) != NULL) src = "codec";
                            else src = "ASSENTE";
                        }
                        fprintf(stderr, " %s->%s(%s)", S->steps[k].op, (spr && *spr) ? spr : "-", src);
                    }
                    fprintf(stderr, "\n");
                    /* COMPOSIZIONE del LAYER: la pipeline dichiarata guida il forward DS4. */
                    if (strcmp(S->key, "layer") == 0) {
                        const char* ops[LLM_DSV4_MAX_OPS]; int no = 0;
                        for (int k = 0; k < S->n_steps && no < LLM_DSV4_MAX_OPS; k++) ops[no++] = S->steps[k].op;
                        llm_dsv4_set_layer_ops(ops, no);   /* pipeline del forward DS4 (whole-model) */
                        llm_mlx_set_layer_ops(ops, no);    /* pipeline del forward generico (layer_forward) */
                    }
                }
            }
        }
    }

    /* DS4 (DeepSeek V4 Flash): modello in un SINGOLO file GGUF (path passato come model_dir). */
    {
        size_t dn = strlen(model_dir);
        if (dn > 5 && strcmp(model_dir + dn - 5, ".gguf") == 0) {
            LlmDsv4Engine* e = llm_dsv4_engine_load(model_dir, 4096);
            if (e != NULL) {
                /* Store/cache esperti DS4 attivato dal PARAMETRO di offload (--offload-experts ->
                 * set_expert_swap), non da una variabile d'ambiente. LRU di N bundle in RAM. */
                /* DS4 e' 97.6 GB: lo streaming degli esperti e' il DEFAULT (nessun flag).
                 * `offload=0` esplicito lo disattiva. La cache usa offload_cache_mb. */
                int want_swap = llm_mlx_moe_swap_on();
                if (llm_ai_mlx_opt_is_set("offload") != 1) want_swap = 1;
                if (want_swap) {
                    int mb = 0, bundles = LLM_DSV4_STORE_BUNDLES;
                    if (llm_ai_mlx_opt_is_set("offload_cache_mb") == 1 &&
                        llm_ai_mlx_opt_get_int("offload_cache_mb", &mb) == 0 && mb > 0) {
                        bundles = mb / 7;   /* un bundle gate+up+down DS4 ~ 7 MB */
                    } else {
                        /* AUTO-SIZING (come ds4 originale): tutta la RAM libera DOPO i pesi
                         * NON-esperto va alla cache esperti. Prima era fissa a 2048 (~14 GB):
                         * ogni token rileggeva da SSD -> ~1 tok/s. */
                        const LlmDsv4Config* cc = &e->w.cfg;
                        uint64_t eb = 0;
                        for (int il = 0; il < cc->n_layer && e->w.exps; il++) {
                            const LlmDsv4PackedExperts* pe = &e->w.exps[il];
                            uint64_t rg = (uint64_t)llm_ggml_nbytes(pe->gate_type, pe->in_dim);
                            uint64_t rd = (uint64_t)llm_ggml_nbytes(pe->down_type, pe->ff_dim);
                            eb += (uint64_t)cc->n_expert *
                                  ((uint64_t)cc->n_ff_exp * rg * 2ull + (uint64_t)cc->n_embd * rd);
                        }
                        uint64_t maplen = (uint64_t)e->maplen;
                        uint64_t nonexp = (eb > 0 && maplen > eb) ? (maplen - eb) : maplen;
                        uint64_t ram = 0; size_t sz = sizeof ram;
                        if (sysctlbyname("hw.memsize", &ram, &sz, NULL, 0) != 0) ram = 0;
                        uint64_t limit = ram ? (ram / 4ull) * 3ull : 0;
                        int32_t wired = 0; sz = sizeof wired;
                        if (sysctlbyname("iogpu.wired_limit_mb", &wired, &sz, NULL, 0) == 0 && wired > 0)
                            limit = (uint64_t)wired * 1024ull * 1024ull;
                        LlmMemGov _mg = llm_memgov_from_registry();
                        uint64_t reserve = _mg.reserve_bytes;   /* B9: dal registro (default 2 GiB) */
                        uint64_t avail = (limit > nonexp + reserve) ? (limit - nonexp - reserve) : 0;
                        /* SICUREZZA: la cache non deve superare la memoria REALMENTE disponibile ora (libera + inattiva, meno un margine di
                         * 12 GiB per il resto del sistema e gli altri processi): sul 128 GB con altri agenti/app attivi il tetto fisso da 86 GB
                         * ha gia' portato la macchina al riavvio. Opzione `dsv4_cache_mb` (0 = automatico) per fissarla a mano. */
                        {
                            vm_statistics64_data_t vs; mach_msg_type_number_t vc = HOST_VM_INFO64_COUNT;
                            if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vs, &vc) == KERN_SUCCESS) {
                                uint64_t pg = (uint64_t)vm_page_size;
                                uint64_t freeish = ((uint64_t)vs.free_count + (uint64_t)vs.inactive_count + (uint64_t)vs.speculative_count) * pg;
                                uint64_t margin = 12ull << 30;
                                uint64_t cap_now = freeish > margin ? freeish - margin : 0;
                                if (avail > cap_now) avail = cap_now;
                            }
                            int mb = 0; if (llm_ai_mlx_opt_get_int("dsv4_cache_mb", &mb) == 0 && mb > 0) avail = (uint64_t)mb * 1024ull * 1024ull;
                        }
                        int b = (int)(avail / (7ull * 1024 * 1024));
                        if (b > 256) bundles = b;
                        fprintf(stderr, "DSV4 cache: esperti=%.1fGB non-esperto=%.1fGB limit=%.1fGB -> %d bundle (~%.1fGB)\n",
                                eb / 1073741824.0, nonexp / 1073741824.0, limit / 1073741824.0,
                                bundles, (double)bundles * 7.0 / 1024.0);
                    }
                    llm_dsv4_engine_enable_swap(e, bundles);
                    /* Lo swap (cache esperti) e' ora attivo: re-init GPU per usare la cache. */
                    llm_dsv4_gpu_ctx_reinit(&e->w.cfg, e->w.exps);
                }
                m->arch = find_arch_ops("deepseek_v4");
                m->dsv4 = e;
                m->gguf_mmap_bytes = (uint64_t)e->maplen;   /* pesi zero-copy: il planner li conti */
                m->cfg.n_layers = e->w.cfg.n_layer;
                m->cfg.vocab_size = e->w.cfg.n_vocab;
                m->cfg.hidden_size = e->w.cfg.n_embd;
                m->cfg.n_heads = e->w.cfg.n_head;
                m->cfg.head_dim = e->w.cfg.n_head_dim;
                m->cfg.n_experts = e->w.cfg.n_expert;
                m->cfg.top_k = e->w.cfg.n_expert_used;
                m->cfg.moe_intermediate_size = e->w.cfg.n_ff_exp;
                m->cfg.intermediate_size = e->w.cfg.n_ff_exp;
                snprintf(m->cfg.model_type, sizeof m->cfg.model_type, "%s", "deepseek_v4");
                fprintf(stderr, "llm_ai_mlx_load_model: deepseek_v4 (DS4) caricato (%d layer, vocab %d)\n",
                        e->w.cfg.n_layer, e->w.cfg.n_vocab);
                return m;
            }
        }
    }

    /* GGUF GENERICO (llama/qwen2/qwen3/…): arch nella ARCH_TABLE standard (non DS4). Riusa il
     * forward per-arch esistente; costruisce cfg/pesi dal GGUF (llm_mlx_load_generic_gguf). */
    {
        size_t dn = strlen(model_dir);
        if (dn > 5 && strcmp(model_dir + dn - 5, ".gguf") == 0) {
            LlmEngineMoeGguf g; void* gmap = NULL; size_t gmaplen = 0;
            if (llm_gguf_open_file(model_dir, &g, &gmap, &gmaplen)) {
                const LlmGgufVal* v = llm_gguf_find_val(&g, "general.architecture");
                char at[64] = "";
                if (v != NULL && v->type == GGSTR && v->s != NULL && v->slen > 0) {
                    int n = (int)v->slen; if (n > 63) n = 63; memcpy(at, v->s, (size_t)n); at[n] = 0;
                }
                llm_gguf_close_file(&g, gmap, gmaplen);
                if (at[0] != 0 && strcmp(at, "deepseek4") != 0 && strcmp(at, "deepseek_v41") != 0 &&
                    find_arch_ops(at) != NULL) {
                    if (llm_mlx_load_generic_gguf(model_dir, m, at)) { m->arch = find_arch_ops(at); return m; }
                    fprintf(stderr, "llm_ai_mlx_load_model: load GGUF generico fallito (%s)\n", model_dir);
                    llm_mlx_model_destroy(m); free(m); return NULL;
                }
            }
        }
    }

    /* DeepSeek V4.1 (whole-model): il config.json del release e' in formato HF root
     * (`text_config` annidato) che `load_config` generico non accetta. Rileva PRIMA di
     * load_config e carica via adattatore (mHC+CSA2+Engram, cache 7 slot). */
    {
        char cpath[4096];
        if ((size_t)snprintf(cpath, sizeof cpath, "%s/config.json", model_dir) < sizeof cpath) {
            FILE* cf = fopen(cpath, "rb");
            if (cf) {
                fseek(cf, 0, SEEK_END); long cn = ftell(cf); fseek(cf, 0, SEEK_SET);
                char* cb = cn > 0 ? (char*)malloc((size_t)cn + 1) : NULL;
                if (cb) {
                    size_t rd = fread(cb, 1, (size_t)cn, cf); cb[rd] = '\0';
                    bool is_dsv41 = strstr(cb, "\"deepseek_v41\"") != NULL;
                    free(cb);
                    fclose(cf);
                    if (is_dsv41) {
                        mlx_device gpu = mlx_device_new_type(MLX_GPU, 0);
                        bool av = false; mlx_device_is_available(&av, gpu);
                        mlx_device chosen = av ? gpu : mlx_device_new_type(MLX_CPU, 0);
                        if (!av) mlx_device_free(gpu);
                        mlx_set_default_device(chosen);
                        mlx_device_free(m->device); m->device = chosen;
                        if (mlx_get_default_stream(&m->stream, m->device) != 0) {
                            fprintf(stderr, "llm_ai_mlx_load_model: deepseek_v41 stream fallito\n");
                            llm_mlx_model_destroy(m); free(m); return NULL;
                        }
                        m->arch = find_arch_ops("deepseek_v41");
                        m->dsv41 = llm_dsv41_engine_load(model_dir, m->stream);
                        if (m->dsv41 == NULL) {
                            fprintf(stderr, "llm_ai_mlx_load_model: deepseek_v41 load fallito (%s)\n", model_dir);
                            llm_mlx_model_destroy(m); free(m); return NULL;
                        }
                        LlmDsv41Config* dc = &((LlmDsv41Engine*)m->dsv41)->w.cfg;
                        {
                            const char* mtp = llm_ai_mlx_opt_env("LLM_DSV41_MTP");
                            if (mtp && mtp[0] && mtp[0] != '0' && ((LlmDsv41Engine*)m->dsv41)->has_dspark)
                                ((LlmDsv41Engine*)m->dsv41)->mtp_enabled = true;
                        }
                        m->cfg.n_layers = dc->n_layers;
                        m->cfg.vocab_size = dc->vocab_size;
                        m->cfg.hidden_size = dc->dim;
                        m->cfg.n_heads = dc->n_heads;
                        m->cfg.n_kv_heads = dc->n_heads;   /* dsv41: KV per-testina, non GQA */
                        m->cfg.head_dim = dc->head_dim;
                        m->cfg.n_experts = dc->n_routed_experts;
                        m->cfg.top_k = dc->n_activated_experts;
                        m->cfg.moe_intermediate_size = dc->moe_inter_dim;
                        m->cfg.intermediate_size = dc->moe_inter_dim;
                        snprintf(m->cfg.model_type, sizeof m->cfg.model_type, "%s", "deepseek_v41");
                        fprintf(stderr, "llm_ai_mlx_load_model: deepseek_v41 whole-model caricato "
                                        "(%d layer, vocab %d)\n", m->cfg.n_layers, m->cfg.vocab_size);
                        return m;
                    }
                } else { fclose(cf); }
            }
        }
    }

    if (!load_config(model_dir, &m->cfg)) {
        llm_mlx_model_destroy(m);
        free(m);
        return NULL;
    }

    /* FASE 1b.0: risolve model_type -> ARCH_TABLE. Un model_type non ancora
     * supportato (nessuna entry combacia) fallisce qui esplicitamente, non a
     * metà del primo forward -- stesso principio delle altre validazioni di
     * load_config (fallire prima, non silenziosamente più tardi). */
    m->arch = find_arch_ops(m->cfg.model_type);
    /* Ternary-Bonsai 2 27B: il contratto Hadamard del pack (block, sign per width, nomi da
     * ruotare, embed inverso). Serve agli hook llm_bonsai_rotate nel forward. */
    {
        /* Il contratto Hadamard si rileva dal FILE (il model_type e' normalizzato a qwen3_5 in
         * load_config): e' l'unico marker affidabile. Se il file non c'e', non e' un pack Prism. */
        LlmBonsaiHadamard* bh = (LlmBonsaiHadamard*)calloc(1, sizeof(*bh));
        if (bh != NULL && llm_bonsai_hadamard_load(model_dir, bh) == 0) {
            m->bonsai_had = bh;
            /* Il base e' qwen3_5: NORMALIZZO il model_type perche' tutti i controlli interni del
             * motore (rope, naming, dispatch per-layer, chunk, ...) confrontano "qwen3_5" e col
             * nome del pack cadrebbero nel ramo sbagliato. La provenienza Prism resta visibile in
             * m->bonsai_had != NULL (ed e' quello che abilita gli hook). */
            snprintf(m->cfg.model_type, sizeof(m->cfg.model_type), "qwen3_5");
            /* STATO 2026-09-23: gli hook (llm_bonsai_rotate) sono cablati e girano su tutte le
             * proiezioni elencate + embed/lm_head, ma la trasformata applicata NON riproduce il
             * modello (output incoerente in tutte le 4 combinazioni sign/embed-inv). Resta da
             * determinare la variante esatta della Hadamard (ordinamento Walsh/sequency vs
             * natural/Sylvester) con l'A/B empirico. Finche' non e' risolta si FALLISCE FORTE:
             * mai un output sbagliato in silenzio. Vedi il todo Ternary-Bonsai (P1). */
            /* RISOLTO 2026-09-25: la de-rotazione Hadamard e' ora fedele (segni PER-BLOCCO +
             * embed-inverse + niente permutazione GDN) e verificata bit-per-bit col reference
             * mlx-vlm (IDS 11751 13 198 ... == reference). Il vecchio rifiuto e' rimosso. */
            if (getenv("LLM_BONSAI_DBG"))
            fprintf(stderr, "llm_ai_mlx_load_model: contratto Hadamard del pack letto "
                            "(block=%d, widths=%d, weight_names=%d, inverse=%d)\n",
                    bh->block, bh->n_widths, bh->n_names, bh->n_inv);
        } else { free(bh); }
    }
    if (m->arch != NULL &&
        (strcmp(m->cfg.model_type, "deepseek_v3") == 0 || strcmp(m->cfg.model_type, "kimi_k2") == 0 ||
         strcmp(m->cfg.model_type, "joyai_llm_flash") == 0 ||
         (strcmp(m->cfg.model_type, "deepseek_v2") == 0 && (m->cfg.q_lora_rank > 0 || m->cfg.n_group > 1))))
        fprintf(stderr, "ATTENZIONE: %s (MLA): percorso implementato ma NON validato (solo "
                        "DeepSeek-V2-Lite lo e'); l'output puo' essere errato.\n", m->cfg.model_type);
    if (!m->arch) {
        if (strcmp(m->cfg.model_type, "deepseek_v41") == 0)
            fprintf(stderr, "llm_ai_mlx_load_model: deepseek_v41 — port MATEMATICO completo e "
                            "verificato (mHC/CSA2/MoE/Engram/RoPE + packing FP8, logits identici "
                            "al riferimento tiny), ma il FORWARD nel motore MLX non e' ancora "
                            "integrato (model_type non in ARCH_TABLE). "
                            "Vedi Todo/(open) - DeepSeek-V41-Port.\n");
        else
            fprintf(stderr, "llm_ai_mlx_load_model: model_type \"%s\" non supportato (nessuna entry in ARCH_TABLE)\n", m->cfg.model_type);
        llm_mlx_model_destroy(m);
        free(m);
        return NULL;
    }

    /* Attivazioni a 16 bit di DEFAULT (come mlx_lm) dove VALIDATO e vantaggioso:
     *  - gpt_oss con quant per-tensore mista (20b MXFP4-Q8): +18% prefill, output identico.
     *  - MoE affine (qwen3_moe/qwen3_5_moe root affine, es. 35B-A3B OptiQ-4bit, Coder-30B 8bit):
     *    +40-50% decode, token identici. I mxfp4 PURI (A10B, gpt-oss-120b) con bf16 sono 10x PIU'
     *    lenti o crashano -> restano fp32.
     *  - denso qwen3_5 4-bit/8-bit affine (Qwen3.6-27B-4bit, Qwen3.8-27B-8bit): +24,5% / +20%
     *    decode, qualita' a parita' (gate loss/token corso oggi, vedi il commento del ramo).
     * Il ramo qwen2/DeepSeek NON e' qui: il suo flavor (f16) dipende dal dtype delle SCALE, che e'
     * nei safetensors e non nel config -> si decide dopo load_all_shards, vedi
     * llm_mlx_auto_pair_act_dtype. llama resta ESCLUSA (bf16 BOTH corrompe device-side, task #5). */
    /* I due bit a 16 bit sono GLOBALI DI FILE (scelti cosi' perche' i siti che li leggono sono ~20
     * e nessun handle arriva fin li'), quindi sopravvivono a uno swap di modello nello STESSO
     * processo: il secondo modello ereditava `g_bf16_all` e il flavor del primo. Ora che anche
     * l'accoppiamento automatico legge quello stato, l'eredita' non e' piu' innocua (un qwen2
     * caricato dopo un mixtral si fosse creduto "gia' promosso"). Azzerare QUI, prima del default
     * per-architettura; `bfloat16_input` invece e' per-modello (vive in m->cfg, gia' fresco). */
    g_bf16_all = false;
    g_act16 = MLX_BFLOAT16;

    bool want_bf16 = false;
    bool want_f16 = false;   /* FIX 2026-09-17 Mixtral: flavour ACCOPPIATO alle scale F16 del checkpoint */
    /* bf16 PROMOSSO a default per gpt-oss mxfp4 (causa misurata §L, vault). Il 20b-gpt_oss gira
     * GIÀ gli esperti mxfp4 sotto bf16 (want_bf16 via per_tensor_quant) ed e' a ~parita' decode
     * con rapid (123 vs 134): la stessa config e' quindi validata per l'architettura. Il 120b e'
     * mxfp4-PURO solo perche' NON ha override per-tensore nel config, non perche' il bf16 gli
     * faccia male; lasciarlo in fp32 paga 1,4x@8K e 1,7x@32K (fp32 vs bf16, misurato sul 20b) --
     * e' l'unica leva residua dopo che GEMV (v2/v4 a soffitto) e ring in-place (piu' lento) sono
     * stati esclusi da misura. Le guardie plain-tensor sotto bf16 (:445-451) coprono l'attenzione
     * plain del 120b. Escape hatch per A/B e kill-switch: LLM_GPTOSS_BF16=0 forza fp32 (= stato
     * pre-promozione). payoff definitivo da confermare on-load sul 120b. */
    const char* goss_bf16_env = llm_ai_mlx_opt_env("LLM_GPTOSS_BF16");
    const bool goss_bf16_off = (goss_bf16_env != NULL && goss_bf16_env[0] == '0');
    if (strncmp(m->cfg.model_type, "gpt_oss", 7) == 0)
        want_bf16 = m->cfg.per_tensor_quant
                 || (strcmp(m->cfg.quant_mode, "mxfp4") == 0 && !goss_bf16_off);
    /* glm4_moe DENTRO dal 2026-09-09: NON c'era, e la cosa era invisibile perche' la riga
     * "[bf16:default] applicato" compariva per mixtral e non per GLM -- cioe' GLM-4.5-Air girava
     * gli ESPERTI in fp32 con le scale BF16 del checkpoint, che e' esattamente il "leak fp32" gia'
     * misurato sui densi. Misura nella STESSA finestra, 2 rep alternate, decodectx N=4096 M=32,
     * stesso dylib, solo bf16= both vs niente: decode 10,4 -> 27,8 tok/s (x2,67), prefill
     * 548 -> 961 tok/s (x1,75). Stessa finestra ha anche falsificato un'altra cosa: il flag
     * sperimentale LLMENGINE_MOE_QQ16 (allineare x al dtype delle scale del gather esperti; ELIMINATO
     * il 2026-09-09 perche' questa misura ne ha reso inutile l'esistenza) sul GLM
     * PROMOSSO non vale nulla (27,8 -> 27,5 decode, 961 -> 943 prefill = rumore), perche' con il
     * flavor x arriva BF16 e le scale sono BF16: la promozione mancante era la causa, il cast no.
     * Gate di qualita' (braccia off/both nello stesso processo, teacher forcing su 1500 posizioni
     * di testo reale, 2 rep, ordine alternato): vedi la sezione "L28bis" di
     * AiWiki/01-Projects/LLMEnv/Knowledge/LLMEnv-vs-RapidMLX-Comparison.md. */
    else if (strcmp(m->cfg.model_type, "qwen3_moe") == 0 || strcmp(m->cfg.model_type, "qwen3_5_moe") == 0
             || strcmp(m->cfg.model_type, "glm4_moe") == 0)
        /* PROMOSSO 2026-09-17 (bug 122B-A10B-mxfp4): il ramo mxfp4 era tenuto fp32 sulla fede di
         * una nota obsoleta ("mxfp4 puri con bf16 sono 10x piu' lenti o crashano"). MISURATO oggi,
         * 122B ctx4096 gen32: fp32 101,05 ms/tok -> bf16 35,05 ms/tok = **2,88x**, IDS BIT-IDENTICI.
         * E il probe MLX mostra gather_qmm mxfp4 a piena banda (87-90 GB/s) con lhs f32/bf16/f16 ->
         * il "10x o crash" NON era di MLX. gpt-oss mxfp4 era GIA' promosso (policy sotto): stessa
         * regola. Kill-switch per A/B e rollback: LLM_MXFP4_PURE_FP32=1 riporta a fp32. */
        want_bf16 = strcmp(m->cfg.quant_mode, "affine") == 0
                    || (strcmp(m->cfg.quant_mode, "mxfp4") == 0 && llm_ai_mlx_opt_env("LLM_MXFP4_PURE_FP32") == NULL);
    else if (strcmp(m->cfg.model_type, "mixtral") == 0) {
        /* FIX 2026-09-17 (bug MLX-Mixtral-8x22B-Prefill-Decode-Spazzatura): le scale/biases del
         * checkpoint Mixtral sono **F16** (3138 tensori verificati) -> il flavour ACCOPPIATO e' **f16**.
         * Con bf16 si ha `promote(bf16,f16)=f32` a OGNI quantized_matmul = "leak fp32": numerica
         * divergente dall'oracolo mlx_lm (che gira f16) e BYTE DOPPI (2,0 tok/s misurati vs ~22 attesi
         * dalla banda sui pesi attivi). Stessa regola del caso GLM (scale BF16 -> bf16) e qwen3_5
         * 6-bit: "niente promozioni per analogia, il flavour si ACCOPPIA alle scale". */
        want_bf16 = strcmp(m->cfg.quant_mode, "affine") == 0;
        want_f16 = want_bf16;
    }
    else if (strcmp(m->cfg.model_type, "qwen3_5") == 0)
        /* PROMOSSO 2026-09-08, denso qwen3_5 affine 4-bit, 6-bit E 8-bit (Qwen3.6-27B-4bit,
         * Qwen3.8-27B-8bit, Opus-Distilled-27B-6bit): era il "leak fp32" dei densi. Gate di
         * QUALITA' su 600 posizioni di
         * testo reale (teacher forcing, braccia nello stesso processo, ordine alternato):
         *   4bit  loss 0,6677 bf16 vs 0,6711 fp32 | top-1 326/600 vs 324/600 | nessun P>1
         *   8bit  loss 0,6872 bf16 vs 0,6877 fp32 | top-1 327/600 vs 325/600 | nessun P>1
         * Velocita' decode (gen=128, 3 rep adiacenti, ordine alternato):
         *   4bit  25,3 -> 31,5 tok/s = 1,245x  |  8bit  13,9 -> 16,7 tok/s = 1,20x
         * Riferimento rapidMLX misurato lo stesso giorno, stesso modello: 4bit 28, 8bit 15-16
         * tok/s -> il denso qwen3_5 passa da 0,90x/0,92x a 1,13x/1,07x contro rapid.
         * 6-bit DENTRO dal 2026-09-08 (era "escluso perche' mai misurato"): misurato su
         * Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit (affine, group 64, `.scales` = **BF16**
         * -> bf16 e' anche il flavor ACCOPPIATO, non solo quello che va bene): decode
         * 14,1/14,8/14,8 -> 17,9/17,7/17,6 = **+21%**, prefill 4096 405 -> 544 = **+37%** (3 rep
         * alternate, load 4,1-4,5). Gate a 1800 posizioni (3 processi indipendenti, teacher
         * forcing): loss 0,6007 -> 0,6013 (**+0,0006**), top-1 1052 -> 1051 (**−1**), argmax
         * diversi **99/5400 = 1,83%**.
         * CONTROLLO che ha deciso la promozione: lo STESSO gate sullo STESSO testo sulla famiglia
         * 8-bit gia' promossa di default (Qwen3.8-27B-8bit, scale BF16) dava loss 0,6125 -> 0,6123
         * (**−0,0002**), top-1 1105 -> 1099 (**−6**), argmax diversi **38/1800 = 2,11%** — cioe'
         * DIVARICA MAGGIORE di quella del 6-bit, e gia' dentro il default. Il 6-bit non aggiunge
         * rischio qualitativo rispetto a cio' che questo motore gia' fa.
         * Nota di onesta' sul gate vecchio: i 600 posizioni qui sopra non hanno MAI misurato la
         * divergenza di argmax (guardano loss, top-1 e "nessun P>1"), quindi non esiste un numero
         * "IDS identici" per 4/8-bit da confrontare; la frase "IDS greedy identici" che circola
         * nella doc appartiene al gate f16 di DeepSeek, non a questa promozione.
         * Guardie che restano: su llama embed+matmul
         * INSIEME danno spazzatura device-side dal primo token, e su qwen2 denso 8bit
         * (DeepSeek-R1-32B) bf16 non vale nulla (1,03x, misurato) -> il win e' dipendente
         * dall'architettura, non dal solo "bf16": niente promozioni per analogia. */
        want_bf16 = strcmp(m->cfg.quant_mode, "affine") == 0
                    && (m->cfg.quant_bits == 4 || m->cfg.quant_bits == 6 || m->cfg.quant_bits == 8);
    else if (strcmp(m->cfg.model_type, "kimi_linear") == 0)
        /* kimi_linear: le scale del checkpoint sono BF16 e i layer KDA/MLA-assorbita/MoE usano
         * quantized_matmul affine, che su Metal RICHIEDE attivazioni f16/bf16 — con input f32
         * fallisce ("prefill fallito", verificato). bf16 e' anche il dtype naturale del modello.
         * Non e' una promozione per analogia: e' necessaria per correttezza. */
        want_bf16 = strcmp(m->cfg.quant_mode, "affine") == 0;
    else if (strcmp(m->cfg.model_type, "qwen3") == 0)
        /* qwen3 DENSO (Qwen3-8B; NON qwen3_moe ne' qwen3_5): era il "leak fp32" dei densi non
         * promossi -- girava fp32 con scale BF16, cioe' il caso peggiore per la GEMV.
         * MISURA 2026-09-16 (Qwen3-8B-4bit, gen=256, 3 rep alternate, load 1,8-2,2, stesso dylib):
         *   fp32 2,64/2,64/2,66 s  ->  bf16 2,38/2,38/2,38 s = **-10%** wall (~+11% tok/s)
         *   (f16 2,35 s = -11%, ma bf16 e' il flavor ACCOPPIATO: le scale del checkpoint sono
         *    BF16 -> promote(bf16,bf16)=bf16, zero AsType per-token e nessuna conversione al load.)
         * IDS greedy IDENTICI a fp32 su 192 token, sia bf16 sia f16 (verificato prima di promuovere).
         * NON e' una promozione per analogia: e' misurata su QUESTA architettura (la regola qui
         * sopra). Kill-switch: LLM_BF16_DENSE=off. Limite dichiarato: il gate di loss/top-1 su
         * testo reale non e' stato rifatto in questa sessione (evidenza disponibile = IDS).
         * BIT 4/6/8 come la famiglia densa sorella qwen3_5: il caso **2-bit NON entra** (non
         * misurato qui; Ternary-Bonsai-4B e' qwen3 affine 2-bit e resta sul comportamento attuale). */
        want_bf16 = strcmp(m->cfg.quant_mode, "affine") == 0
                    && (m->cfg.quant_bits == 4 || m->cfg.quant_bits == 6 || m->cfg.quant_bits == 8);
    if (want_bf16) {
        m->cfg.bfloat16_input = true;
        g_bf16_all = true;
        g_act16 = want_f16 ? MLX_FLOAT16 : MLX_BFLOAT16; /* flavour accoppiato alle scale (Mixtral=F16) */
        /* Il dtype di calcolo di un modello caricato NON puo' cambiare in silenzio: stampa sempre
         * (stesso formato di llm_mlx_apply_bf16_mode, cosi' un log dice quale bit e' acceso e per
         * perche'). Kill-switch: LLM_BF16_DENSE=off, oppure mlxComputeDtype="off" nel config. */
        fprintf(stderr, "[%s:default] applicato: model_type=%s quant_mode=%s quant_bits=%d "
                        "mode=both dt=%s embed=1 matmul=1\n",
                want_f16 ? "f16" : "bf16", m->cfg.model_type, m->cfg.quant_mode, m->cfg.quant_bits,
                want_f16 ? "f16" : "bf16");
    }

    /* G2 (piano antigravity 2026-09-08): diagnostica bf16 forzata sui modelli DENSE non promossi
     * dal default (llama/qwen2/qwen3 denso/gemma4, e i qwen3_5 con quant diverse da affine-4bit che
     * non sono stati misurati). Regola di ammissione e stampa del rifiuto stanno in
     * llm_mlx_apply_bf16_mode, condivisa con la config esplicita. Da quando i densi promossi
     * esistono, l'ambiente è anche KILL-SWITCH (LLM_BF16_DENSE=off spegne qwen3_5 4bit).
     * BISECTION (G2 passo 1): ciascun bit DA SOLO e' token-identico al default su llama-1B-4bit e
     * Qwen3.6-27B-4bit; la COMBINAZIONE produce spazzatura silenziosa SOLO su llama (head_dim 64),
     * mentre su qwen3_5 (head_dim 256) resta identica -> il difetto e' locale a
     * llama_attention_block, NON un "bf16 denso rotto". */
    const char* bf16_dense = llm_ai_mlx_opt_env("LLM_BF16_DENSE");
    /* La richiesta esplicita dell'ospite VINCE sull'accoppiamento automatico del flavor (che
     * comunque non si applica se i bit a 16 bit sono gia' accesi): questo flag vive fino al punto
     * dopo load_all_shards dove il dtype delle scale diventa leggibile. */
    const bool dtype_explicit = bf16_dense != NULL;
    if (bf16_dense != NULL) {
        /* Mode nota usata alla lettera (incluso "off"), "altro/vuoto" = entrambi, per non rompere
         * le misure dell'08/09 prese con LLM_BF16_DENSE=1. NON e' piu' condizionato a !want_bf16:
         * con il default che accende i densi, l'ambiente deve poterli anche SPENERE (kill-switch);
         * apply_bf16_mode e' idempotente sullo stato gia' richiesto. */
        llm_mlx_apply_bf16_mode(m, llm_mlx_bf16_mode_known(bf16_dense) ? bf16_dense : "both",
                                "env LLM_BF16_DENSE");
    }

    mlx_device gpu = mlx_device_new_type(MLX_GPU, 0);
    bool gpu_available = false;
    mlx_device_is_available(&gpu_available, gpu);
    mlx_device chosen = gpu_available ? gpu : mlx_device_new_type(MLX_CPU, 0);
    if (!gpu_available) mlx_device_free(gpu);
    if (mlx_set_default_device(chosen) != 0) {
        fprintf(stderr, "llm_ai_mlx_load_model: mlx_set_default_device fallita\n");
        mlx_device_free(chosen);
        llm_mlx_model_destroy(m);
        free(m);
        return NULL;
    }
    mlx_device_free(m->device);
    m->device = chosen;

    if (mlx_get_default_stream(&m->stream, m->device) != 0) {
        fprintf(stderr, "llm_ai_mlx_load_model: mlx_get_default_stream fallita\n");
        llm_mlx_model_destroy(m);
        free(m);
        return NULL;
    }
    /* Limite alla cache buffer MLX. PROMOSSO a 8 GiB (2026-09-12): con 2 GiB il modello
     * piu' esigente in buffer temporanei (gpt-oss-120b, 128 esperti) finiva in churn
     * alloc/free ad ogni eval -> decode ~1,8x piu' lento (misurato: 11,2 -> 20,3 tok/s,
     * 3/3 rep, parita' IDS bit-identica, maxRSS +0,6GB). 8 GiB satura (16/64 GiB uguali) e
     * resta neutro su 20b e llama-1B. Limite VOLUTO (bounded): senza, i buffer liberati
     * restano nel cache allocator e la memoria active cresce a dismisura (Mixtral: +2,7GB/
     * layer -> ~150GB -> thrash). Override diagnostico: LLM_MLX_CACHE_LIMIT_MB. */
    size_t cache_lim = (size_t)8 << 30; /* 8 GiB */
    const char* clim_env = llm_ai_mlx_opt_env("LLM_MLX_CACHE_LIMIT_MB");
    if (clim_env && clim_env[0]) {
        long mb = atol(clim_env);
        if (mb > 0) cache_lim = (size_t)mb << 20;
    }
    mlx_set_cache_limit(&cache_lim, cache_lim);
    /* mlx_set_wired_limit — pinna N GB di memoria unificata (come fa rapidMLX: "asks the OS to
     * keep model resident"). POLICY (misurata 2026-09-12): NON e' un default globale — aiuta i
     * modelli grandi sotto pressione di memoria (Mixtral: prefill -27%, ordine-invariante; il
     * 120b a load alto: prefill 5,7->3,9s) ma a load BASSO il 120b peggiora (sovra-pin senza
     * pressione). Quindi: (a) `LLM_MLX_WIRED_GB=N` forza N GB (diagnostica/opt-in); (b)
     * `LLM_NO_WIRE=1` disabilita; (c) AUTO solo per `model_type=="mixtral"`.
     *
     * FIX 2026-09-17 — CAUSA del gap 27-40x di Mixtral (prefill 5-10 tok/s vs 272 di mlx_lm).
     * Il cap AUTO fisso a 64 GB era SOTTO il footprint di Mixtral (73,9 GB = 68,8 GiB): il wired
     * veniva SALTATO -> modello NON pinnato -> thrash. mlx_lm non ha il problema perche'
     * `wired_limit()` (mlx_lm/generate.py:257) pinna a `max_recommended_working_set_size`, che su
     * questo M5 Max vale 96 GiB (`sysctl iogpu.wired_limit_mb` = 98304) -> Mixtral interamente
     * pinnato. Prova che chiude TUTTE le ipotesi: (1) i modelli <64 GB (gpt-oss-120b 61 GB,
     * GLM-4.5-Air 56 GB) erano pinnati e VINCONO mlx_lm; solo Mixtral (l'unico >64 GB) perde;
     * (2) un Mixtral SINTETICO alle dimensioni reali (h6144/i16384, L8=11 GB, L16=22,6 GB) gira
     * a PARITA' (L8: 2980 vs 2680 tok/s) o 2,2x PIU' VELOCE (L16: 1434 vs 650) di mlx_lm, e scala
     * LINEARMENTE con i layer -> il codice/path non c'entra. Non era quindi ne' le forme
     * gather_qmm, ne' sorted/unsorted, ne' i kernel: era la RESIDENZA.
     * Ora il cap AUTO = min(massimo OS, footprint+8 GiB), con floor 64 GiB: si alza quel tanto che
     * serve a pinnare il modello senza sovra-pin. Si salta SOLO se nemmeno il massimo OS copre il
     * footprint (allora il wired non e' applicabile). */
    { const char* we = llm_ai_mlx_opt_env("LLM_MLX_WIRED_GB");
      const char* nw = llm_ai_mlx_opt_env("LLM_NO_WIRE");
      const size_t mbytes = llm_mlx_dir_safetensors_bytes(model_dir);
      const long fp_gib = mbytes > 0 ? (long)(mbytes >> 30) : 0;
      long gb = 0;
      if (we && we[0]) gb = atol(we);
      /* PROMOSSO 2026-09-17: AUTO per TUTTI i modelli (era solo mixtral). Il fix Mixtral del
       * 17/09 aveva ristretto l'AUTO a mixtral -> tutti gli altri grandi restavano NON pinnati ->
       * pressione memoria (free ~0,08 GB!) -> CLIFF. MISURATO, 3-6 run per modello, ctx4096:
       * 122B-mxfp4 34,9 -> 21,2 ms/tok (1,65x); gpt-oss-120b 20,8 -> 15,2 (1,37x);
       * GLM-4.5-Air 34,0 -> 28,5 (1,19x); Qwen3-8B 12,39 -> 12,04 (neutro). Nessun danno trovato.
       * Kill-switch: LLM_NO_WIRE=1 (o LLM_MLX_WIRED_GB=N per forzare). */
      else if (nw == NULL && fp_gib >= 64) {
          /* Massimo consentito dall'OS/MLX = lo stesso valore che pinna mlx_lm. */
          long maxrec_gb = 0;
          { mlx_device_info info = mlx_device_info_new();
            size_t mrs = 0;
            if (mlx_device_info_get(&info, m->device) == 0 &&
                mlx_device_info_get_size(&mrs, info, "max_recommended_working_set_size") == 0 && mrs > 0)
                maxrec_gb = (long)(mrs >> 30);
            mlx_device_info_free(info); }
          if (maxrec_gb <= 0) {
              size_t ramsz = 0; size_t sz = sizeof(ramsz);
              if (sysctlbyname("hw.memsize", &ramsz, &sz, NULL, 0) == 0 && ramsz > 0)
                  maxrec_gb = (long)((ramsz * 6 / 10) >> 30);   /* fallback 60% RAM */
          }
          gb = 64;                                  /* floor storico (min(60% RAM,64)) */
          if (maxrec_gb > 0 && gb > maxrec_gb) gb = maxrec_gb;
          if (fp_gib > 0) {                         /* alza il cap per coprire il footprint */
              long need = fp_gib + 8;
              if (maxrec_gb > 0 && need > maxrec_gb) need = maxrec_gb;
              if (gb < need) gb = need;
          }
      }
      if (gb > 0 && we == NULL && fp_gib > 0 && gb < fp_gib) {
          fprintf(stderr, "[wired] skip: modello %.1f GiB > cap OS %ld GiB (nemmeno il massimo OS "
                          "copre il footprint; LLM_MLX_WIRED_GB per forzare)\n",
                  (double)mbytes / 1073741824.0, gb);
          gb = 0;
      }
      if (gb > 0) { size_t prev = 0;
          int rc = mlx_set_wired_limit(&prev, (size_t)gb << 30);
          fprintf(stderr, "[wired] model_type=%s GB=%ld (footprint=%ld GiB) rc=%d prev=%zu\n",
                  m->cfg.model_type, gb, fp_gib, rc, prev); } }

    /* AUTO-POLICY leve model-dependent (2026-09-12) — vedi
     * Todo/(open) - MLX-Leve-Model-Dependent-Fine-Tuning.md. Abilita leve misurate
     * order-invarianti + parità IDS, SOLO per il modello che ne beneficia (mai globale).
     * `setenv(..., overwrite=0)`: non sovrascrive una scelta esplicita dell'utente.
     * Kill-switch: LLM_NO_<LEVA> (il loader salta l'abilitazione). */
    if (strcmp(m->cfg.model_type, "gemma4") == 0 && llm_ai_mlx_opt_env("LLMMETAL_NO_FUSE_GEGLU_METAL") == NULL) {
        setenv("LLMMETAL_FUSE_GEGLU_METAL", "1", 0);  /* +1,2% decode (unico dense_geglu_ffn) */
        fprintf(stderr, "[auto-policy] gemma4 -> LLMMETAL_FUSE_GEGLU_METAL=1\n");
    }
    /* gemma4: bf16 "matmul" (g_bf16_all, NON bfloat16_input). Il checkpoint e' dtype=bfloat16
     * → bf16 e' il dtype NATIVO (LLM girava f32 = "leak fp32"; rapid/mlx-lm usano bf16). Misurato
     * +21% decode; il bit `embed` e' piu' lento e `both` aveva argmax 9,0% -> solo `matmul`
     * (argmax 7,4%, acc 2,0% = livello accettato; gate §44). Kill-switch LLM_NO_BF16. */
    if (strcmp(m->cfg.model_type, "gemma4") == 0 && llm_ai_mlx_opt_env("LLM_NO_BF16") == NULL) {
        llm_mlx_apply_bf16_mode(m, "matmul", "auto-policy gemma4");
        fprintf(stderr, "[auto-policy] gemma4 -> bf16 matmul\n");
    }
    /* oQ A8 (prefill INT8 x Q4/Q5 sui tensor unit M5) e' MODEL-DEPENDENT, misurato con
     * llm_batch_scale_probe (sessione nuova + warm-up):
     *   qwen3 denso (Qwen3-8B, hidden 4096): N=1024/2048/4096 -> ON/OFF 0,72x/0,84x/0,77x
     *                                        = **ON vince +16..28%**
     *   llama (Llama-1B, hidden 2048):       N=16..2048 -> ON/OFF 3,38x..1,93x = **ON perde 2-6x**
     *     (l'overhead per-call del bridge eager non si ammortizza sui GEMM piu' piccoli)
     * -> abilitato SOLO per qwen3 denso. setenv(...,overwrite=0): una scelta esplicita
     * dell'utente (LLMMETAL_OQ_A8=1 o =0) vince SEMPRE su questa auto-policy. */
    /* AUTO-POLICY (riattivata 2026-09-19 sera, dopo la diagnosi): il "bug qualita'" NON era il
     * W8A8 ma la combo con bf16-mode OFF (guard in llm_mlx_oq_a8_active_for). Con bf16 ON la
     * qualita' e' identica a MLX (score 256 pos: W8A8 89,8% vs MLX 90,6%) -> il W8A8 resta
     * attivo di default sui qwen3 come deciso il 2026-09-18. */
    if (strcmp(m->cfg.model_type, "qwen3") == 0 && llm_ai_mlx_opt_env("LLMMETAL_OQ_A8") == NULL) {
        setenv("LLMMETAL_OQ_A8", "1", 0);
        fprintf(stderr, "[auto-policy] qwen3 -> LLMMETAL_OQ_A8=1 (misurato +16-28% prefill N>=1024; "
                        "su llama sarebbe -2/-6x)\n");
    }
    if (strcmp(m->cfg.model_type, "qwen3_next") == 0 && llm_ai_mlx_opt_env("LLM_NO_MXFP4_ACT_F16") == NULL) {
        setenv("LLM_MXFP4_ACT_F16", "1", 0);     /* +7% prefill (misurato neutro su qwen3_5_moe) */
        fprintf(stderr, "[auto-policy] qwen3_next -> LLM_MXFP4_ACT_F16=1\n");
    }
    /* B7 (2026-09-29): opzione di REGISTRO `kvquant` = off|8[:finestra]|auto[:soglia]|16.
     * `16` = KV in bf16 esplicito (nessuna quant). `auto` = attiva l'int8 Splash da sola quando
     * il contesto supera la soglia (default 8192, il crossover misurato). Default del registro:
     * `auto`. Precedenza: opzione di registro > env legacy > config.json. */
    if (!m->cfg.kv_quantize) {
        char kvs[64];
        if (llm_ai_mlx_opt_get_str("kvquant", kvs, sizeof kvs) == 0 && kvs[0]) {
            if (strncmp(kvs, "off", 3) == 0) {
                /* resta bf16 (default) */
            } else if (strncmp(kvs, "auto", 4) == 0) {
                int trig = 8192; const char* c = strchr(kvs, ':');
                if (c) trig = atoi(c + 1);
                if (trig < 512) trig = 512;
                m->cfg.kv_quantize_auto = true;
                m->cfg.kv_quantize_auto_trigger = trig;
                fprintf(stderr, "[kvquant] auto: int8 Splash oltre %d token (finestra 2048)\n", trig);
            } else {
                int bits = atoi(kvs);
                int thr = 0;
                const char* c = strchr(kvs, ':');
                if (c) thr = atoi(c + 1);
                if (bits == 16) {
                    m->cfg.kv_quantize = false; m->cfg.kv_quantize_bits = 16; m->cfg.kv_quantize_threshold = 0;
                    fprintf(stderr, "[kvquant] bf16 (nessuna quant, opzione di registro)\n");
                } else if (bits == 8) {
                    /* B7 (2026-09-29): KV-quant SPLASH int8 (pool int8 storage primario + read fused). */
                    m->cfg.kv_quantize = true;
                    m->cfg.kv_quantize_bits = 8;
                    m->cfg.kv_quantize_threshold = thr > 0 ? thr : 2048;
                    fprintf(stderr, "[kvquant] int8 Splash attivata threshold=%d\n", m->cfg.kv_quantize_threshold);
                } else if (bits == 4) {
                    fprintf(stderr, "[kvquant] bits=4 non supportato dal pool Splash (solo 8)\n");
                } else {
                    fprintf(stderr, "[kvquant] valore '%s' non valido (off|8[:finestra]|auto[:soglia]|16)\n", kvs);
                }
            }
        }
    }
    /* LLM_KVQUANT (=1 attiva; _BITS default 8; _THRESH default 2048): abilita la KV-quant T3.2
     * (cold block affine) via env, senza editare il config.json dei modelli (nessun config lo
     * imposta oggi). NB: il path e' DEQUANT-ON-READ (layer_forward dequantizza la cold all'inizio
     * e la ri-quantizza alla fine) -> riduce la memoria a contesto lungo ma costa 2xO(pos)/token. */
    if (!m->cfg.kv_quantize && 0 && llm_ai_mlx_opt_env("LLM_KVQUANT") != NULL) {   /* B7: env disattivata (port Splash incompleto, vedi TODO) */
        const char* b = llm_ai_mlx_opt_env("LLM_KVQUANT_BITS");
        const char* t = llm_ai_mlx_opt_env("LLM_KVQUANT_THRESH");
        m->cfg.kv_quantize = true;
        m->cfg.kv_quantize_bits = b ? atoi(b) : 8;
        m->cfg.kv_quantize_threshold = t ? atoi(t) : 2048;
        fprintf(stderr, "[kvquant] attivato bits=%d threshold=%d\n",
                m->cfg.kv_quantize_bits, m->cfg.kv_quantize_threshold);
    }

    /* mlx_load_safetensors non ha implementazione GPU ("Load::eval_gpu Not
     * implemented", verificato con un errore runtime reale) -- il
     * caricamento dei pesi da disco va sempre fatto su uno stream CPU,
     * indipendentemente dallo stream (GPU-preferred) usato per il calcolo
     * dei layer sotto. Memoria unificata Apple Silicon: gli array caricati
     * su CPU sono utilizzabili senza copie esplicite dalle op successive su
     * stream GPU. */
    mlx_stream cpu_stream = mlx_default_cpu_stream_new();
    bool shards_ok = load_all_shards(model_dir, cpu_stream, &m->weights);
    mlx_stream_free(cpu_stream);
    if (!shards_ok) {
        llm_mlx_model_destroy(m);
        free(m);
        return NULL;
    }
    /* AWQ (autoawq GEMM): dequantizza `<base>.qweight/qzeros/scales` in `<base>.weight` F16
     * PRIMA di qualunque lookup (le wcache memorizzano i miss) e PRIMA di auto_pair/pretouch. */
    {
        int awq_n = awq_synthesize_weights(&m->weights, m->stream);
        if (awq_n < 0) {
            fprintf(stderr, "llm_ai_mlx_load_model: awq_synthesize_weights fallita\n");
            llm_mlx_model_destroy(m);
            free(m);
            return NULL;
        }
        if (awq_n > 0) {
            fprintf(stderr, "[awq] dequantizzati %d tensori qweight/qzeros/scales -> <base>.weight F16\n", awq_n);
            llm_mlx_wcache_clear();
        }
    }
    /* GPTQ (autogptq/gptqmodel) e bitsandbytes NF4: stesso hook dell'AWQ, per i checkpoint
     * 4-bit che non hanno `<base>.weight` plain. Va PRIMA di auto_pair/pretouch. */
    {
        int qn = gptq_nf4_synthesize_weights(&m->weights, m->stream);
        if (qn < 0) {
            fprintf(stderr, "llm_ai_mlx_load_model: gptq_nf4_synthesize_weights fallita\n");
            llm_mlx_model_destroy(m);
            free(m);
            return NULL;
        }
        if (qn > 0) {
            fprintf(stderr, "[quant] dequantizzati %d tensori GPTQ/NF4 -> <base>.weight F16\n", qn);
            llm_mlx_wcache_clear();
        }
    }
    /* TurboQuant (PolarQuant-MLX, mode:"turboquant"): dequant `<base>.weight U32 + scales + signs
     * + codebook` in F16 plain. Stesso hook (PRIMA di auto_pair/pretouch). */
    {
        int tqn = turboquant_synthesize_weights(&m->weights, m->stream);
        if (tqn < 0) {
            fprintf(stderr, "llm_ai_mlx_load_model: turboquant_synthesize_weights fallita\n");
            llm_mlx_model_destroy(m);
            free(m);
            return NULL;
        }
        if (tqn > 0) {
            fprintf(stderr, "[turboquant] dequantizzati %d tensori -> <base>.weight F16\n", tqn);
            llm_mlx_wcache_clear();
        }
    }
    /* Mixtral va sanitizzato PRIMA del pretouch, come mlx_lm: materializzare qui gli
     * expert separati e poi creare gli stack produce due copie residenti del MoE
     * (misurato sul 8x22B: 79.39 -> 155.49 GB attivi). Per le altre architetture il
     * pretouch resta nella posizione storica. */
    const bool is_mixtral = strcmp(m->cfg.model_type, "mixtral") == 0;
    const bool pretouch_weights = llm_mlx_pretouch_weights_enabled();
    if (!is_mixtral && pretouch_weights)
        llm_mlx_pretouch_weights(&m->weights);
    llm_mlx_dp_memory("post-shard-load"); /* prima dello stacking: sono i GB dei soli pesi */
    /* Il dtype delle scale si conosce SOLO qui: e' nel checkpoint, non nel config. */
    llm_mlx_auto_pair_act_dtype(m, dtype_explicit);

    /* Qwen4-Exp N-gram/PLE: la tabella di embedding vive in un file separato
     * (`ngram_table.bin`, formato safetensors), troppo grande per stare nell'index dei pesi
     * principali (30GB su un checkpoint reale). Aperta SOLO se il config ha layer PLE
     * attivi, via qwen4_exp_ngram_table_open -- parsa l'header JSON a mano e tiene il file
     * aperto per pread() mirati a riga, MAI mlx_load_safetensors su questo file (vedi
     * commento su LlmMlxModel.ngram_fd: quel percorso forzerebbe la materializzazione
     * dell'intero array a ogni gather). Se il file manca o l'header e' malformato, i layer
     * PLE continuano a rifiutarsi esplicitamente (nessun crash del load dell'intero modello). */
    if (m->cfg.ple_layer_ids_n > 0) {
        char ngram_path[1200];
        snprintf(ngram_path, sizeof(ngram_path), "%s/ngram_table.bin", model_dir);
        m->ngram_table_loaded = qwen4_exp_ngram_table_open(ngram_path, m);
        if (!m->ngram_table_loaded) {
            /* Checkpoint UFFICIALE mlx-community: niente `ngram_table.bin` monolitico, la
             * tabella e' inline nei safetensors principali in 128 shard row-wise. Stesso API
             * a valle (qwen4_exp_ngram_read_rows), lettura sempre on-demand via pread. */
            m->ngram_table_loaded = qwen4_exp_ngram_table_open_shards(model_dir, m);
            if (m->ngram_table_loaded) {
                fprintf(stderr, "llm_ai_mlx_load_model: N-gram/PLE da %d shard inline "
                                "(rows_per_shard=%lld, bits=%d group=%d)\n",
                        m->ngram_nshards, (long long)m->ngram_rows_per_shard,
                        m->ngram_table_bits, m->ngram_table_group_size);
            }
        }
        if (!m->ngram_table_loaded) {
            fprintf(stderr, "llm_ai_mlx_load_model: ple_layer_ids non vuoto ma %s assente/illeggibile -- "
                            "i layer PLE si rifiuteranno esplicitamente\n", ngram_path);
        }
    }

    /* FASE 1b.3 (Mixtral): impila i pesi per-esperto separati in un unico tensore
     * stacked per gate_proj/up_proj/down_proj, UNA SOLA VOLTA qui (non ad ogni
     * forward) -- vedi commento su stack_mixtral_moe_experts sopra per il perché. */
    const bool is_mixtral_arch = strcmp(m->cfg.model_type, "mixtral") == 0;
    const bool is_olmoe_arch = strcmp(m->cfg.model_type, "olmoe") == 0;
    if (is_mixtral_arch || is_olmoe_arch) {
        bool sok = is_olmoe_arch
            ? stack_moe_experts(&m->cfg, &m->weights, m->stream, "mlp", "gate_proj", "up_proj", "down_proj")
            : stack_moe_experts(&m->cfg, &m->weights, m->stream, "block_sparse_moe", "w1", "w3", "w2");
        if (!sok) {
            fprintf(stderr, "llm_ai_mlx_load_model: stack_moe_experts fallita\n");
            llm_mlx_model_destroy(m);
            free(m);
            return NULL;
        }
        /* Copia rapidMLX moe_fusion.py: dopo lo stack, fonde gate+up in un
         * solo tensore per layer (una gather_qmm nel decode invece di due). */
        if (is_mixtral_arch && !fuse_mixtral_gate_up(&m->cfg, &m->weights, m->stream)) {
            fprintf(stderr, "llm_ai_mlx_load_model: fuse_mixtral_gate_up fallita\n");
        }
        /* Una sola eval di TUTTI i parametri finali (kept + stack), non 504 eval
         * indipendenti: e' la sequenza di mlx_lm load_model(sanitize -> load -> eval).
         * LLM_NO_STACK_EVAL conserva il kill-switch diagnostico storico. */
        if (pretouch_weights && !llm_env_isset("LLM_NO_STACK_EVAL"))
            llm_mlx_pretouch_weights(&m->weights);
    }

    /* Kimi Linear: deriva i tensori MLA assorbiti (embed_q/unembed_out) da kv_b_proj e
     * iniettali come shard sintetico (UNA VOLTA a load). I layer KDA sono saltati. */
    if (strcmp(m->cfg.model_type, "kimi_linear") == 0) {
        if (!kimi_linear_synthesize_absorbed_mla(&m->cfg, &m->weights, m->stream)) {
            fprintf(stderr, "llm_ai_mlx_load_model: kimi_linear_synthesize_absorbed_mla fallita\n");
            llm_mlx_model_destroy(m);
            free(m);
            return NULL;
        }
    }

    /* Hoist il cast bf16/f16->fp32 delle scale fuori dal percorso per-token (default ON;
     * LLM_F32_SCALES=0 opt-out). Prima del pack QKV perche' llm_mlx_attn_quant popola qs/qb dal map. */
    llm_mlx_promote_scales_f32(m);

    /* QKV candidate: pack only the two validated naming/architecture families.
     * Originals remain available for fallback, LoRA and runtime dtype selection. */
    if (llm_mlx_qkv_enabled()) {
        int packed = 0;
        if (strcmp(m->cfg.quant_mode, "affine") == 0 &&
            (strcmp(m->cfg.model_type, "llama") == 0 || strcmp(m->cfg.model_type, "qwen3") == 0)) {
            for (int l = 0; l < m->cfg.n_layers; ++l) {
                GattnQuant* c = llm_mlx_attn_quant(m, l);
                if (!c) continue;
                mlx_array w[3] = {c->qw, c->kw, c->vw};
                mlx_array s[3] = {c->qs, c->ks, c->vs};
                mlx_array b[3] = {c->qb, c->kb, c->vb};
                if (llm_mlx_qkv_pack(&c->packed_qkv, w, s, b, m->stream)) ++packed;
            }
        }
        fprintf(stderr, "[packed-qkv] model=%s packed=%d/%d%s\n", m->cfg.model_type,
                packed, m->cfg.n_layers, packed ? "" : " fallback: architecture/quantization/cache ineligible");
    }
    /* P4 memory plan: calcolato e stampato ORA al load (a pesi materializzati, KV ancora assente),
     * non alla prima `append`. Il piano e' a livello MODELLO (dipende solo da cfg e dai pesi), quindi
     * anticiparlo non cambia i numeri ma fa uscire la diagnostica all'avvio — prima del prompt in
     * `fastllm chat`, non dopo il primo messaggio. Idempotente: `p4_plan_done` rende la chiamata in
     * session_append un no-op. */
    llm_p4_plan_init(m);
    llm_mlx_dp_memory("post-load"); /* dopo lo stacking: il delta da post-shard-load e' nostro */
    /* W1 (2026-09-18) — DUMP DEL RAMO ATTIVO, una volta per load, dietro LLM_WHICH_PATH=1.
     * REGOLA branch-first (AGENTS §0sexies / Hard-Lessons/Ramo-Sbagliato-Prima-Del-Kernel.md):
     * prima di misurare, far dire all'engine QUALE percorso gira e PER QUALE flag. Caso reale:
     * LLM_KVQUANT=1 da solo non attivava il dequant-on-read -> percorso legacy per giorni.
     * PUNTO SICURO: qui, dopo la finalizzazione del cfg (il tentativo 1 moriva chiamandolo prima).
     * `want_bf16` e' la locale di questa funzione, in scope. Nessun costo quando il flag e' assente. */
    if (getenv("LLM_WHICH_PATH")) {
        LlmWhichPath wp; memset(&wp, 0, sizeof(wp));
        wp.model_type = m->cfg.model_type;
        wp.dt_name = want_bf16 ? "bf16" : ((m->cfg.quant_mode && strcmp(m->cfg.quant_mode, "mxfp4") == 0) ? "fp32" : "f16");
        wp.dtype_src = "policy(cfg/per_tensor/LLM_*)";
        wp.quant_name = m->cfg.quant_mode;
        wp.quant_bits = m->cfg.quant_bits;
        wp.kv_quant = m->cfg.kv_quantize;
        wp.kv_quant_bits = m->cfg.kv_quantize_bits;
        wp.kv_quant_thresh = m->cfg.kv_quantize_threshold;
        wp.kv_read = (m->cfg.kv_quantize && llm_ai_mlx_opt_env("LLM_KV_DEQUANT_READ")) ? 1 : 0;
        wp.kv_read_src = llm_ai_mlx_opt_env("LLM_KV_DEQUANT_READ") ? 1 : (m->cfg.kv_quantize ? 0 : 2);
        wp.kv_slice = llm_ai_mlx_opt_env("LLM_KVSLICE") != NULL;
        wp.kv_seg = llm_ai_mlx_opt_env("LLM_KVSEG") != NULL;
        wp.kv_f16 = llm_ai_mlx_opt_env("LLM_KVF16") != NULL;
        wp.kv_metal = llm_ai_mlx_opt_env("LLMMETAL_KVMETAL") != NULL;
        wp.kv_max_tokens = llm_ai_mlx_opt_env("LLM_KV_MAX_TOKENS") ? atoi(llm_ai_mlx_opt_env("LLM_KV_MAX_TOKENS")) : 0;
        wp.moe_fuse = llm_ai_mlx_opt_env("LLMENGINE_MOE_FUSE_MOE_DECODE") != NULL;
        wp.moe_mxfp4 = llm_ai_mlx_opt_env("LLMENGINE_MOE_FUSE_MOE_MXFP4") != NULL;
        wp.moe_gateup = llm_ai_mlx_opt_env("LLMENGINE_MOE_GATEUP_FUSE") != NULL;
        wp.moe_n = m->cfg.n_experts;
        wp.moe_topk = m->cfg.top_k;
        wp.at_flash_decode = llm_ai_mlx_opt_env("LLM_GEMMA_FLASH_DECODE") != NULL;
        wp.at_flash_prefill = llm_ai_mlx_opt_env("LLM_GEMMA_FLASH_PREFILL") != NULL;
        wp.at_fuse_rope = llm_ai_mlx_opt_env("LLM_FUSE_ROPE") != NULL;
        wp.at_sliding = m->cfg.sliding_window > 0;
        wp.w_packed_qkv = (strcmp(m->cfg.quant_mode, "affine") == 0 &&
                           (strcmp(m->cfg.model_type, "llama") == 0 || strcmp(m->cfg.model_type, "qwen3") == 0));
        wp.w_wcache = llm_ai_mlx_opt_env("LLM_WCACHE") != NULL;
        wp.w_reshape_canon = llm_ai_mlx_opt_env("LLM_RESHAPE_CANONICAL") != NULL;
        wp.w_scales_bf16 = want_bf16 ? 1 : 0;
        wp.w_gemvq4 = llm_ai_mlx_opt_env("LLM_GEMVQ4") != NULL;
        wp.w_oq_a8 = llm_ai_mlx_opt_env("LLMMETAL_OQ_A8") != NULL;
        wp.x_compile = llm_ai_mlx_opt_env("LLM_COMPILE") != NULL;
        wp.x_fixkv = llm_ai_mlx_opt_env("LLM_FIXKV") != NULL;
        wp.x_compile_dense = llm_ai_mlx_opt_env("LLM_COMPILE_DENSE") != NULL;
        wp.x_async = llm_ai_mlx_opt_env("LLM_ASYNC_DECODE") != NULL && llm_ai_mlx_opt_env("LLM_NO_ASYNC_DECODE") == NULL;
        wp.x_sampler_legacy = llm_ai_mlx_opt_env("LLM_SAMPLER_LEGACY") != NULL;
        llm_mlx_which_path_dump(&wp);
    }
    /* D15: prefix-cache PERSISTENTE — se l'opzione `prefix_cache` e' impostata, ricarica i
     * prefissi salvati (nome+token) e li registra (radix); con il tier SSD attivo la KV e'
     * restaurabile, quindi il riuso evita il re-prefill (TTFT cached) anche tra riavvii. */
    { char pp[2048];
      if (llm_ai_mlx_opt_get_str("prefix_cache", pp, (int)sizeof pp) == 0 && pp[0]) {
          int nl = llm_ai_mlx_model_prefix_load((llm_ai_mlx_model_handle)m, pp);
          if (nl > 0) fprintf(stderr, "[prefix] %d prefissi persistenti caricati da %s\n", nl, pp);
      }
    }
    return m;
}

void llm_ai_mlx_free_model(llm_ai_mlx_model_handle handle) {
    if (handle == NULL) return;
    LlmMlxModel* m = (LlmMlxModel*)handle;
    llm_mlx_kvseg_reset_all();   /* LLM_KVSEG: libera main/segmento per-layer */
    llm_mlx_pad512_reset();      /* PAD512: i pesi padded referenziano il modello */
    llm_mlx_gateup_reset();      /* FUSE_GATEUP: idem per i pesi gate+up concatenati */
    llm_mlx_model_destroy(m);
    free(m);
}
