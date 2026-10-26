/* GGUF sintetico per il test PLE: offsets/vocab/multipliers + 8 shard da 4 righe. */
#include "llmcodec_gguf_writer.h"
#include <stdio.h>
#include <stdint.h>
#define NTS 16
static char names[NTS][64]; static float data[NTS][4*160]; static uint64_t dims[NTS][2];
static LlmCodecGgufTensor ts[NTS]; static int nts = 0;
static void add(const char* nm, int d0, int d1, const float* v) {
    snprintf(names[nts], sizeof names[nts], "%s", nm);
    dims[nts][0] = (uint64_t)d0; dims[nts][1] = (uint64_t)d1;
    int n = d0 * d1; for (int i = 0; i < n; i++) data[nts][i] = v[i];
    ts[nts].name = names[nts]; ts[nts].ggml_type = LLMCODEC_GGML_F32;
    ts[nts].dims = dims[nts]; ts[nts].ndim = 2; ts[nts].data = data[nts]; ts[nts]._offset = 0; nts++;
}
int q4n_write_ple_synth(const char* path) {
    LlmCodecGgufMetaKV kv[8]; int nkv = 0;
    kv[nkv].key = "general.architecture"; kv[nkv].type = LLMCODEC_GGUF_MT_STRING; kv[nkv].str = "qwen4exp"; nkv++;
    float off[16], voc[16], mul[3] = {1.0f, 1.0f, 1.0f};
    for (int h = 0; h < 16; h++) { off[h] = (float)(h * 2); voc[h] = 2.0f; }
    add("ple.head_offsets", 16, 1, off);
    add("ple.head_vocab_sizes", 16, 1, voc);
    add("blk.1.ple_layer_multipliers", 3, 1, mul);
    float row[4 * 160];
    for (int s = 0; s < 8; s++) {
        for (int r = 0; r < 4; r++) for (int d = 0; d < 160; d++) row[r*160+d] = (float)(s*1000 + r*10 + d);
        char nm[32]; snprintf(nm, sizeof nm, "ple.shard.%d.weight", s);
        add(nm, 160, 4, row);
    }
    return llmcodec_gguf_write(path, kv, nkv, ts, nts, 0);
}
