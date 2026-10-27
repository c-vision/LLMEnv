/* engine_select.c — scelta del motore in base al modello.
 * Convenzione: directory (o path .mlx) -> motore MLX; file GGUF (magic "GGUF") -> motore ds4.
 * Il motore ds4 e' opzionale: se non linkato, la selezione ritorna NULL per i GGUF. */
#include "llmengine.h"
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <limits.h>

/* Fornita dal backend ds4 (LLMEngine/ds4/bridge) quando linkato. */
extern const LlmEngineLlmEngine* llmengine_llm_engine_ds4n(void) __attribute__((weak));

const LlmEngineLlmEngine* llmengine_engine_for_model(const char* model_path) {
    if (model_path == NULL || model_path[0] == '\0') return NULL;
    struct stat st;
    if (stat(model_path, &st) == 0 && S_ISDIR(st.st_mode)) return llmengine_llm_engine_mlx();
    FILE* f = fopen(model_path, "rb");
    if (!f) return NULL;
    unsigned char m[4] = {0,0,0,0};
    size_t n = fread(m, 1, 4, f);
    fclose(f);
    if (n == 4 && memcmp(m, "GGUF", 4) == 0)
        return llmengine_llm_engine_ds4n ? llmengine_llm_engine_ds4n() : NULL;
    return NULL;
}
