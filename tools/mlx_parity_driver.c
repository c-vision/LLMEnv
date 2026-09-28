/* mlx_parity_driver.c — driver di PARITA' del motore MLX estratto.
 *
 * Carica una dylib del motore (LLMEnv `libllm_ai_mlx` oppure GMBasic
 * `libgmb_ai_mlx`) VIA dlopen, risolve l'accessor della vtable
 * (`llmengine_llm_engine_mlx` oppure `gmb_llm_engine_mlx`) e usa l'ABI comune
 * `LLMEngine/include/llmengine.h` (handle opachi) per:
 *   load modelo -> create session -> prefill di ids dati -> generate greedy.
 *
 * L'ABI NON espone la tokenizzazione (stabilmente a monte, in C#): il prompt e'
 * quindi passato come LISTA DI TOKEN ID, identica per i due motori. La parita'
 * e' percio' dimostrata a parita' di prompt ids (piu' forte del solo testo).
 *
 * Uso:
 *   mlx_parity_driver <dylib> <accessor> <model_dir> <ids_csv> <N> [temperature]
 *
 * Stampa (una riga chiave=valore, parse-friendly):
 *   ENGINE=mlx
 *   APPEND_RC=<int>
 *   N=<token generati>
 *   IDS=<id1,id2,...>
 *   OK / ERROR: <msg>
 */
#include "llmengine.h"

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef const LlmEngineLlmEngine* (*accessor_fn)(void);

static int parse_ids(const char* csv, int32_t** out, int* n_out) {
    int cap = 64, n = 0;
    int32_t* ids = (int32_t*)malloc((size_t)cap * sizeof(int32_t));
    if (!ids) return -1;
    const char* p = csv;
    while (*p) {
        char* end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        if (n == cap) {
            cap *= 2;
            int32_t* t = (int32_t*)realloc(ids, (size_t)cap * sizeof(int32_t));
            if (!t) { free(ids); return -1; }
            ids = t;
        }
        ids[n++] = (int32_t)v;
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    if (n == 0) { free(ids); return -1; }
    *out = ids;
    *n_out = n;
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr,
                "uso: %s <dylib> <accessor> <model_dir> <ids_csv> <N> [temperature]\n",
                argv[0]);
        return 2;
    }
    const char* dylib_path = argv[1];
    const char* accessor = argv[2];
    const char* model_dir = argv[3];
    const char* ids_csv = argv[4];
    int max_new = atoi(argv[5]);
    float temperature = (argc > 6) ? (float)atof(argv[6]) : 0.0f;

    void* lib = dlopen(dylib_path, RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        fprintf(stderr, "ERROR: dlopen(%s): %s\n", dylib_path, dlerror());
        return 3;
    }
    accessor_fn fn = (accessor_fn)dlsym(lib, accessor);
    if (!fn) {
        fprintf(stderr, "ERROR: dlsym(%s): %s\n", accessor, dlerror());
        return 4;
    }
    const LlmEngineLlmEngine* eng = fn();
    if (!eng || !eng->load_model || !eng->create_session || !eng->append_ids ||
        !eng->generate_ids) {
        fprintf(stderr, "ERROR: vtable incompleta\n");
        return 5;
    }
    printf("ENGINE=%s\n", eng->name ? eng->name : "?");

    int32_t* ids = NULL;
    int n_ids = 0;
    if (parse_ids(ids_csv, &ids, &n_ids) != 0) {
        fprintf(stderr, "ERROR: ids non valide\n");
        return 6;
    }

    LlmEngineLlmModel model = eng->load_model(model_dir);
    if (!model) {
        fprintf(stderr, "ERROR: load_model(%s) fallita\n", model_dir);
        free(ids);
        return 7;
    }

    LlmEngineLlmSession sess = eng->create_session(model);
    if (!sess) {
        fprintf(stderr, "ERROR: create_session fallita\n");
        eng->free_model(model);
        free(ids);
        return 8;
    }

    /* Greedy deterministico: temp<=0 => argmax. Penalita' neutre. */
    LlmEngineLlmSampling sp;
    memset(&sp, 0, sizeof(sp));
    sp.temperature = temperature;
    sp.top_p = 1.0f;
    sp.top_k = 0;
    sp.repetition_penalty = 1.0f;
    sp.frequency_penalty = 0.0f;
    sp.presence_penalty = 0.0f;
    sp.seed = 0;
    if (eng->set_sampling) eng->set_sampling(sess, &sp);

    int append_rc = eng->append_ids(sess, ids, n_ids);
    printf("APPEND_RC=%d\n", append_rc);

    int32_t* out = (int32_t*)malloc((size_t)max_new * sizeof(int32_t));
    if (!out) {
        fprintf(stderr, "ERROR: out alloc\n");
        return 9;
    }
    /* eos=-1 disattiva lo stop anticipato: si vogliono esattamente max_new id. */
    int got = eng->generate_ids(sess, max_new, -1, out);
    printf("N=%d\n", got);
    printf("IDS=");
    for (int i = 0; i < got; i++) printf("%s%d", i ? "," : "", out[i]);
    printf("\n");

    if (got == max_new) {
        printf("OK\n");
    } else {
        printf("ERROR: generati %d su %d richiesti\n", got, max_new);
    }

    free(out);
    eng->destroy_session(sess);
    eng->free_model(model);
    free(ids);
    dlclose(lib);
    return (got == max_new) ? 0 : 1;
}
