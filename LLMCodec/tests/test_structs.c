/* Verifica la lettura delle `structures` + ruoli DS4.1 + il check "non supportata". */
#include "llmcodec_rolemap.h"
#include <stdio.h>
#include <string.h>

#ifndef LLMCODEC_ABI_DIR
#define LLMCODEC_ABI_DIR "../../ABI"
#endif

int main(void) {
    char path[512]; snprintf(path, sizeof path, "%s/deepseek_v41.gmbarch", LLMCODEC_ABI_DIR);
    LlmCodecRoleMap rm;
    if (llmcodec_rolemap_load(path, &rm) != 0) { printf("FAIL load\n"); return 1; }
    const char* attn = llmcodec_rolemap_struct_type(&rm, "attention");
    const char* eng  = llmcodec_rolemap_struct_type(&rm, "engram");
    const char* vis  = llmcodec_rolemap_struct_type(&rm, "vision");
    const char* csa  = llmcodec_rolemap_struct_type(&rm, "compression");
    const char* idx  = llmcodec_rolemap_struct_type(&rm, "indexer");
    int ea = llmcodec_rolemap_expert_axis(&rm);
    char ub[512] = {0}; int nbad = llmcodec_rolemap_unsupported(&rm, ub, sizeof ub);
    printf("[structs] n=%d attn=%s engram=%s vision=%s compression=%s indexer=%s expert_axis=%d\n",
           rm.n_st, attn ? attn : "-", eng ? eng : "-", vis ? vis : "-", csa ? csa : "-", idx ? idx : "-", ea);
    printf("[structs] non supportate=%d  riporto:%s\n", nbad, ub);

    const char* r_comp = llmcodec_rolemap_of(&rm, "layers.0.attn.compressor.weight");
    const char* r_idx  = llmcodec_rolemap_of(&rm, "layers.0.attn.indexer.proj.weight");
    const char* r_eng  = llmcodec_rolemap_of(&rm, "layers.1.engram.table");
    const char* r_hc   = llmcodec_rolemap_of(&rm, "layers.0.hc_proj.weight");
    const char* r_mtp  = llmcodec_rolemap_of(&rm, "mtp.0.main_proj.weight");
    printf("[structs] ruoli: compressor=%s indexer=%s engram=%s hc=%s mtp=%s\n", r_comp, r_idx, r_eng, r_hc, r_mtp);

    int ok = attn && strcmp(attn,"mla")==0 && eng && strcmp(eng,"engram")==0 && vis && strcmp(vis,"vit")==0
             && csa && strcmp(csa,"csa")==0 && idx && strcmp(idx,"indexer")==0
             && ea == 3 && nbad == 0
             && strcmp(r_comp,"ATTN_COMPRESSOR")==0 && strcmp(r_idx,"ATTN_INDEXER")==0
             && strcmp(r_eng,"ENGRAM")==0 && strcmp(r_hc,"MHC")==0 && strcmp(r_mtp,"MTP")==0;
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
