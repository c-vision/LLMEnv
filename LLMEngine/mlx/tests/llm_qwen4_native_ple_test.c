/* test hash + row-read PLE su GGUF sintetico. */
#include "mlx/c/mlx.h"
#include "qwen4_native/qwen4_native_ple.h"
#include <stdio.h>
#include <string.h>
int q4n_write_ple_synth(const char* path);
static int rp_check(const float* v, int s, int r);
int main(void) {
    char path[] = "/tmp/q4n_ple_XXXXXX"; int fd = mkstemp(path);
    if (fd < 0 || q4n_write_ple_synth(path) != 0) { printf("FAIL write\n"); return 1; }
    close(fd);
    LlmEngineMoeGguf g; void* map = NULL; size_t len = 0;
    if (!llm_gguf_open_file(path, &g, &map, &len)) { printf("FAIL open\n"); return 1; }
    Q4nHp hp; memset(&hp, 0, sizeof hp); hp.ple_ng = 3; hp.ple_heads = 8;
    Q4nPleMeta pm; memset(&pm, 0, sizeof pm);
    if (q4n_ple_meta(&g, &hp, &pm) != 0) { printf("FAIL meta\n"); return 1; }
    printf("n_heads=%d off0=%g voc0=%g mult0=%g\n", pm.n_heads, pm.offsets[0], pm.vocab[0], pm.mult[0]);
    int prev[4] = {0,0,0,0};
    uint32_t rows[64];
    q4n_ple_rows(5, prev, &pm, 3, 0, rows);
    printf("rows:");
    for (int i = 0; i < pm.n_heads; i++) printf(" %u", rows[i]);
    printf("\n");
    int bad = 0;
    for (int i = 0; i < pm.n_heads; i++) {
        float v[160];
        if (q4n_ple_row_read(&g, rows[i], v) != 0) { printf("read fail %d\n", i); bad = 1; continue; }
        uint32_t r = rows[i];
        int s = (int)(r / 4), rr = (int)(r % 4);
        if (rp_check(v, s, rr) != 0) bad = 1;
    }
    printf("%s\n", bad ? "PLE ROWS: FALLITO" : "PLE ROWS: OK");
    llm_gguf_close_file(&g, map, len); unlink(path);
    return bad;
}
static int rp_check(const float* v, int s, int r) {
    for (int d = 0; d < 160; d++) if (v[d] != (float)(s*1000 + r*10 + d)) return 1;
    return 0;
}
