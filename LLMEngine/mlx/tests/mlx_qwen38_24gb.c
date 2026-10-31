/* Route B (2026-10-30): Qwen3.8-Flash-Next MLX 2-bit locale con streaming esperti
 * e budget RAM di 24 GB. Nessun download: usa i safetensors già presenti.
 *
 * Obiettivo: capire se il ramo MLX sa ESEGUIRE questo modello con un footprint
 * contenuto (≤ 24 GB) sfruttando il fatto che i pesi sono mmap/file-backed.
 * Non è un benchmark di qualità: sampling greedy, prompt corto.
 *
 * Uso: ./mlx_qwen38_24gb <model_dir> [cache_mb] [ctx] [max_new] [hold_after_load_s]
 */
#include "llmengine.h"
#include "llm_engine_opts.h"
#include "llmtoken.h"
#include "llmjinja.h"
#include "cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/resource.h>
#include <libproc.h>
#include <mach/mach.h>

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }
static double rss_gb(void) { struct rusage ru; if (getrusage(RUSAGE_SELF, &ru) != 0) return -1.0; return (double)ru.ru_maxrss / (1024.0 * 1024.0 * 1024.0); }
/* phys_footprint del processo (proc_pid_rusage): la metrica che conta davvero per la RAM
 * (stessa grandezza di `vmmap` "Physical footprint"). Su Metal include le allocazioni
 * unified/GPU che l'RSS NON vede: misurato 41,4 GB di footprint contro 0,3 GB di RSS sullo
 * stesso processo. `footprint -p` (il CLI) su processi brevi puo' riportare valori assurdi:
 * usare questo valore o `vmmap`. */
static double footprint_gb(void) {
    struct rusage_info_v4 ri;
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t*)&ri) != 0) return -1.0;
    return (double)ri.ri_phys_footprint / (1024.0 * 1024.0 * 1024.0);
}


/* Carica `chat_template` da tokenizer_config.json e rende [{"role":"user","content":…}]
 * con `add_generation_prompt=true`, come fa FastLLM. Se qualcosa manca, ritorna NULL e il
 * chiamante usa il prompt grezzo (dichiarandolo). Un prompt SENZA template cambia la
 * distribuzione del modello: era l'ipotesi principale dell'output degradato a 31 token. */
static char* render_chat(const char* dir, const char* user_text) {
    char path[1200];
    /* Preferenza: chat_template.jinja (file a sé, come nei checkpoint mlx-community) e poi
     * tokenizer_config.json["chat_template"]. */
    snprintf(path, sizeof path, "%s/chat_template.jinja", dir);
    FILE* jf = fopen(path, "rb");
    if (jf != NULL) {
        fseek(jf, 0, SEEK_END); long jsz = ftell(jf); fseek(jf, 0, SEEK_SET);
        if (jsz > 0 && jsz < (1 << 20)) {
            char* tpl = (char*)malloc((size_t)jsz + 1);
            if (tpl != NULL) {
                size_t rj = fread(tpl, 1, (size_t)jsz, jf); tpl[rj] = 0; fclose(jf);
                llmjinja_val* a = llmjinja_arr();
                llmjinja_val* m0 = llmjinja_obj();
                llmjinja_obj_set(m0, "role", llmjinja_str("user"));
                llmjinja_obj_set(m0, "content", llmjinja_str(user_text));
                llmjinja_arr_push(a, m0);
                llmjinja_val* v = llmjinja_obj();
                llmjinja_obj_set(v, "messages", a);
                llmjinja_obj_set(v, "add_generation_prompt", llmjinja_bool(1));
                char e[256] = {0};
                char* o = llmjinja_render(tpl, v, e, sizeof e);
                if (o == NULL) fprintf(stderr, "[routeB] jinja(.jinja): %s\n", e[0] ? e : "render fallito");
                llmjinja_val_free(v); free(tpl);
                if (o != NULL) return o;
            }
        } else fclose(jf);
    }
    snprintf(path, sizeof path, "%s/tokenizer_config.json", dir);
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > (1 << 22)) { fclose(f); return NULL; }
    char* js = (char*)malloc((size_t)sz + 1);
    if (!js) { fclose(f); return NULL; }
    size_t rd = fread(js, 1, (size_t)sz, f); js[rd] = 0; fclose(f);

    cJSON* root = cJSON_Parse(js); free(js);
    if (!root) return NULL;
    cJSON* tpl = cJSON_GetObjectItemCaseSensitive(root, "chat_template");
    if (cJSON_IsObject(tpl)) tpl = cJSON_GetObjectItemCaseSensitive(tpl, "default");
    if (!cJSON_IsString(tpl)) { cJSON_Delete(root); return NULL; }

    llmjinja_val* arr = llmjinja_arr();
    llmjinja_val* msg = llmjinja_obj();
    llmjinja_obj_set(msg, "role", llmjinja_str("user"));
    llmjinja_obj_set(msg, "content", llmjinja_str(user_text));
    llmjinja_arr_push(arr, msg);

    llmjinja_val* vars = llmjinja_obj();
    llmjinja_obj_set(vars, "messages", arr);
    llmjinja_obj_set(vars, "add_generation_prompt", llmjinja_bool(1));

    char err[256] = {0};
    char* out = llmjinja_render(tpl->valuestring, vars, err, sizeof err);
    if (out == NULL) fprintf(stderr, "[routeB] jinja: %s\n", err[0] ? err : "render fallito");
    llmjinja_val_free(vars);
    cJSON_Delete(root);
    return out;
}

static const char* PROMPT = "Explain in two sentences why streaming model weights from disk lets a "
                            "large mixture-of-experts model run on a laptop with less RAM than the model.";

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "uso: %s <model_dir> [cache_mb] [ctx] [max_new] [hold_s]\n", argv[0]); return 2; }
    const char* dir = argv[1];
    int cache_mb = (argc > 2) ? atoi(argv[2]) : 16384;
    int ctx      = (argc > 3) ? atoi(argv[3]) : 4096;
    int max_new  = (argc > 4) ? atoi(argv[4]) : 24;
    int hold_s   = (argc > 5) ? atoi(argv[5]) : 0;
    const char* prompt = (argc > 6) ? argv[6] : PROMPT;
    float temp   = (argc > 7) ? (float)atof(argv[7]) : 0.0f;

    /* Parametri SOLO dal registro opzioni (mai env). */
    char buf[32];
    snprintf(buf, sizeof buf, "%d", cache_mb);
    llm_ai_mlx_opt_set("offload", "on");
    llm_ai_mlx_opt_set("offload_cache_mb", buf);
    llm_ai_mlx_opt_set("memgov", "1");
    llm_ai_mlx_opt_set("kvquant", "8");
    snprintf(buf, sizeof buf, "%d", ctx);
    llm_ai_mlx_opt_set("context", buf);
    /* 24 GB di budget: NON pinnare i pesi (83 GB) e non materializzarli in anticipo.
     * Sono nomi del registro avanzato, quindi passano senza env. */
    llm_ai_mlx_opt_set("LLM_NO_WIRE", "1");
    llm_ai_mlx_opt_set("LLM_NO_PRETOUCH_WEIGHTS", "1");

    const LlmEngineLlmEngine* e = llmengine_llm_engine_mlx();
    if (e == NULL) { fprintf(stderr, "FAIL: engine mlx\n"); return 1; }
    fprintf(stderr, "[routeB] engine=%s abi=%u\n", e->name, (unsigned)llmengine_abi_version());
    if (e->set_expert_swap != NULL) e->set_expert_swap(dir);

    double t0 = now_s();
    LlmEngineLlmModel m = e->load_model(dir);
    if (m == NULL) { fprintf(stderr, "FAIL: load (rss_peak=%.2f GB)\n", rss_gb()); return 1; }
    fprintf(stderr, "[routeB] load OK in %.1fs (rss_peak=%.2f GB)\n", now_s() - t0, rss_gb());

    if (hold_s > 0) { printf("pid=%d fase=post-load rss=%.2f GB\n", (int)getpid(), rss_gb()); fflush(stdout); sleep(hold_s); }

    if (e->kv_set_max != NULL) (void)e->kv_set_max(m, ctx);
    LlmEngineLlmSession s = e->create_session(m);
    if (s == NULL) { fprintf(stderr, "FAIL: session (rss_peak=%.2f GB)\n", rss_gb()); return 1; }

    llmtoken_handle tok = NULL;
    char tpath[1024];
    snprintf(tpath, sizeof tpath, "%s/tokenizer.json", dir);
    tok = llmtoken_load(tpath);
    if (tok == NULL) { fprintf(stderr, "FAIL: tokenizer.json non caricabile\n"); return 1; }

    int32_t ids[512]; int n_ids = 0;
    char* rendered = render_chat(dir, prompt);
    if (rendered != NULL) fprintf(stderr, "[routeB] chat template applicato (%zu char)\n", strlen(rendered));
    else fprintf(stderr, "[routeB] ATTENZIONE: chat template non applicato, prompt grezzo\n");
    n_ids = llmtoken_encode(tok, rendered ? rendered : prompt, ids, 512);
    free(rendered);
    if (n_ids <= 0) {
        fprintf(stderr, "FAIL: encode\n"); return 1;
    }
    fprintf(stderr, "[routeB] prompt = %d token (rss_peak=%.2f GB)\n", n_ids, rss_gb());

    LlmEngineLlmSampling sp = {0};
    sp.temperature = temp; sp.top_p = 0.95f; sp.top_k = 0; sp.seed = 42;
    if (e->set_sampling) (void)e->set_sampling(s, &sp);

    double tp = now_s();
    if (e->append_ids(s, ids, n_ids) != 0) { fprintf(stderr, "FAIL: append_ids (rss_peak=%.2f GB)\n", rss_gb()); return 1; }
    double t_prefill = now_s() - tp;
    fprintf(stderr, "[routeB] prefill %.1f token in %.2fs = %.1f tok/s (rss_peak=%.2f GB)\n",
            (double)n_ids, t_prefill, n_ids / t_prefill, rss_gb());

    int32_t out[256];
    double tg = now_s();
    int n = e->generate_ids ? e->generate_ids(s, max_new, -1, out) : -1;
    double t_gen = now_s() - tg;
    if (n <= 0) { fprintf(stderr, "FAIL: generate_ids (%d)\n", n); return 1; }
    fprintf(stderr, "[routeB] decode %d token in %.2fs = %.2f tok/s (rss_peak=%.2f GB, footprint=%.2f GB)\n",
            n, t_gen, n / t_gen, rss_gb(), footprint_gb());

    char txt[4096];
    int nb = llmtoken_decode(tok, out, n, txt, (int)sizeof txt); /* ritorna i BYTE, non 0 */
    if (nb >= 0) printf("OUT (%d byte): %s\n", nb, txt);
    printf("IDS:"); for (int i = 0; i < n; ++i) printf(" %d", out[i]); printf("\n");

    e->destroy_session(s);
    e->free_model(m);
    llmtoken_free(tok);
    printf("FINE rss_peak=%.2f GB footprint=%.2f GB (metrica valida: RSS sottostima su Metal)\n", rss_gb(), footprint_gb());
    return 0;
}
