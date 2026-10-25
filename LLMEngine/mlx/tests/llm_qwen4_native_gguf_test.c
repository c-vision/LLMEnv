/* llm_qwen4_native_gguf_test.c — gate model-free P7e: adapter qwen4exp sul
 * loader GGUF esistente (llm_gguf + writer del codec), con GGUF sintetico. */
#include "mlx/c/mlx.h"
#include "qwen4_native/qwen4_native_gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int q4n_write_synth_gguf(const char* path);

int main(void) {
    char path[] = "/tmp/q4n_gguf_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) { printf("FAIL mkstemp\n"); return 1; }
    close(fd);
    if (q4n_write_synth_gguf(path) != 0) { printf("FAIL write\n"); unlink(path); return 1; }

    LlmEngineMoeGguf g; void* map = NULL; size_t maplen = 0;
    if (!llm_gguf_open_file(path, &g, &map, &maplen)) { printf("FAIL open\n"); unlink(path); return 1; }
    Q4nHp hp; char err[256] = {0};
    int bad = 0;
    if (q4n_hparams(&g, &hp, err, sizeof err) != 0) { printf("hparams FAIL: %s\n", err); bad = 1; }
    else if (hp.E != 8 || hp.L != 2 || hp.H != 2 || hp.Hkv != 1 || hp.D != 4 || hp.NE != 4 || hp.K != 2 ||
             hp.hc != 2 || hp.rank != 3 || hp.lin_Hk != 2 || hp.lin_Hv != 3 || hp.lin_D != 4 ||
             hp.ple_ng != 3 || hp.n_rot != 4) { printf("hparams valori errati\n"); bad = 1; }

    Q4nLayerRefs r0; memset(&r0, 0, sizeof r0);
    if (q4n_layer_load(&g, 0, true, false, &r0, err, sizeof err) != 0) { printf("layer0 FAIL: %s\n", err); bad = 1; }
    else if (!r0.ssm_a || !r0.attn_qkv || !r0.gate_exps || r0.type_gate_exps != 0 || r0.ssm_a[0] <= 0.0f) {
        printf("layer0 refs errati\n"); bad = 1;
    }
    Q4nLayerRefs r1; memset(&r1, 0, sizeof r1);
    if (q4n_layer_load(&g, 1, false, true, &r1, err, sizeof err) != 0) { printf("layer1 FAIL: %s\n", err); bad = 1; }
    else if (!r1.attn_q || !r1.ple_key || !r1.idx_q_proj) { printf("layer1 refs errati\n"); bad = 1; }
    Q4nLayerRefs rx;
    if (q4n_layer_load(&g, 9, true, false, &rx, err, sizeof err) == 0) { printf("layer9 doveva fallire\n"); bad = 1; }

    printf("%s\n", bad ? "P7e GGUF ADAPTER: FALLITO" : "P7e GGUF ADAPTER: OK");
    llm_gguf_close_file(&g, map, maplen);
    unlink(path);
    return bad ? 1 : 0;
}
