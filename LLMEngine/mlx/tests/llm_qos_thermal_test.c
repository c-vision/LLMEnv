/* llm_qos_thermal_test (2026-10-31) — test LOAD-FREE del backoff termico (F1).
 *
 * Verifica: (a) la policy pura (stato -> divisore/chunk), (b) le opzioni di registro
 * `qos`/`qos_thermal`/`qos_thermal_sim` esistono e leggono i valori attesi, (c) la lettura dello
 * stato termico non crasha (valore reale o -1). Nessun modello: compila e gira in millisecondi.
 *
 *   clang -O2 -std=c11 -I LLMEngine/mlx -I LLMEngine/Options \
 *         -o build/llm_qos_thermal_test LLMEngine/mlx/tests/llm_qos_thermal_test.c \
 *         -L build/LLMEngine/mlx -lllm_ai_mlx -Wl,-rpath,$PWD/build/LLMEngine/mlx
 */
#include "llm_mlx_qos.h"
#include "llm_engine_opts.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
static void check(int cond, const char* what) {
    printf("%s %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) fails++;
}

int main(void) {
    /* (a) policy pura */
    check(llm_mlx_qos_chunk_divisor(-1) == 4, "stato sconosciuto -> divisore 4 (nessun peggioramento)");
    check(llm_mlx_qos_chunk_divisor(0)  == 4, "nominal -> 4");
    check(llm_mlx_qos_chunk_divisor(1)  == 4, "fair -> 4");
    check(llm_mlx_qos_chunk_divisor(2)  == 8, "serious -> 8");
    check(llm_mlx_qos_chunk_divisor(3)  == 16, "critical -> 16");
    check(llm_mlx_qos_wants_background(3) == 1, "critical -> QoS BACKGROUND");
    check(llm_mlx_qos_wants_background(2) == 0, "serious -> resta UTILITY");
    check(llm_mlx_qos_apply_thermal(2048, 0) == 512, "chunk 2048 nominale -> 512");
    check(llm_mlx_qos_apply_thermal(2048, 2) == 256, "chunk 2048 serious -> 256");
    check(llm_mlx_qos_apply_thermal(2048, 3) == 128, "chunk 2048 critical -> 128");
    check(llm_mlx_qos_apply_thermal(1024, 3) == 128, "floor 128 rispettato (1024/16=64 -> 128)");

    /* (b) opzioni di registro */
    check(llm_ai_mlx_opt_is_set("qos_thermal") >= 0, "opzione qos_thermal esiste");
    check(llm_ai_mlx_opt_is_set("qos_thermal_sim") >= 0, "opzione qos_thermal_sim esiste");
    int v = -999; check(llm_ai_mlx_opt_get_int("qos_thermal", &v) == 0 && v == 1, "qos_thermal default 1");
    check(llm_ai_mlx_opt_get_int("qos_thermal_sim", &v) == 0 && v == -1, "qos_thermal_sim default -1");
    check(llm_ai_mlx_opt_set("qos_thermal_sim", "3") == 0, "set qos_thermal_sim=3");
    check(llm_ai_mlx_opt_get_int("qos_thermal_sim", &v) == 0 && v == 3, "riletto 3");
    check(llm_ai_mlx_opt_set("qos_thermal_sim", "-1") == 0, "reset qos_thermal_sim=-1");

    printf("\n%s (%d fail)\n", fails ? "FALLITO" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
