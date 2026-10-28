/* llm_mlx_wpacked.c — implementazione M1 (mmap + risoluzione weight/scales/biases). */
#include "llm_mlx_wpacked.h"
#include "llmcodec_safetensors.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

static int find_off(const LlmCodecStTensorInfo* ti, int n, const char* nm, uint64_t* off, uint64_t* len, const char** dt) {
    for (int i = 0; i < n; i++) {
        if (strcmp(ti[i].name, nm) != 0) continue;
        *off = ti[i].begin; *len = ti[i].end - ti[i].begin; *dt = ti[i].dtype; return 0;
    }
    return -1;
}

int llm_mlx_packed_open(const char* path, const char* base, int bits, int group, LlmMlxPackedW* out) {
    if (!path || !base || !out || bits <= 0 || group <= 0) return -1;
    memset(out, 0, sizeof *out);
    LlmCodecStTensorInfo* ti = NULL; int n = 0; uint64_t base_off = 0;
    if (llmcodec_safetensors_read(path, &ti, &n, &base_off) != 0) { fprintf(stderr, "[wpacked3d] read header FALLITA: %s\n", path); return -1; }
    char nm[1024];
    uint64_t wo = 0, wl = 0, so = 0, sl = 0, bo = 0, bl = 0; const char *dtw = NULL, *dts = NULL, *dtb = NULL;
    snprintf(nm, sizeof nm, "%s.weight", base);
    int ok = find_off(ti, n, nm, &wo, &wl, &dtw) == 0;
    if (ok) { snprintf(nm, sizeof nm, "%s.scales", base); ok = find_off(ti, n, nm, &so, &sl, &dts) == 0; }
    int has_b = 0;
    if (ok) { snprintf(nm, sizeof nm, "%s.biases", base); has_b = find_off(ti, n, nm, &bo, &bl, &dtb) == 0; }
    /* dimensioni dall'header: weight U32 [rows][cols*bits/32] (cols = dim1*32/bits), scales [rows][cols/group] */
    uint64_t rows = 0, wcols = 0, scols = 0;
    if (ok) {
        snprintf(nm, sizeof nm, "%s.weight", base);
        for (int i = 0; i < n; i++) if (strcmp(ti[i].name, nm) == 0) { rows = ti[i].shape[0]; wcols = ti[i].shape[1]; }
        snprintf(nm, sizeof nm, "%s.scales", base);
        for (int i = 0; i < n; i++) if (strcmp(ti[i].name, nm) == 0) { scols = ti[i].shape[1]; }
    }
    llmcodec_safetensors_info_free(ti);
    if (!ok || rows == 0 || wcols == 0) return -1;
    uint64_t cols = wcols * 32ull / (uint64_t)bits;
    if (scols != cols / (uint64_t)group) return -1;

    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st; if (fstat(fd, &st) != 0) { close(fd); return -1; }
    size_t mlen = (size_t)st.st_size;
    void* map = mmap(NULL, mlen, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) return -1;
    out->w = (const uint32_t*)((const uint8_t*)map + base_off + wo);
    out->scales = (const uint16_t*)((const uint8_t*)map + base_off + so);
    out->biases = has_b ? (const uint16_t*)((const uint8_t*)map + base_off + bo) : NULL;
    out->rows = (uint32_t)rows; out->cols = (uint32_t)cols; out->bits = bits; out->group = group;
    out->w_bytes = (size_t)wl; out->s_bytes = (size_t)sl; out->b_bytes = has_b ? (size_t)bl : 0;
    out->map = map; out->map_bytes = mlen;
    return 0;
}

void llm_mlx_packed_close(LlmMlxPackedW* w) {
    if (!w || !w->map) return;
    munmap(w->map, w->map_bytes);
    memset(w, 0, sizeof *w);
}

/* ---- vista 3D (esperti): [E, rows, cols] ---- */
int llm_mlx_packed_open3d(const char* path, const char* base, int bits, int group, LlmMlxPackedW3* out) {
    fprintf(stderr, "[wpacked3d] entry path=%s base=%s\n", path?path:"(null)", base?base:"(null)");
    if (!path || !base || !out) return -1;
    memset(out, 0, sizeof *out);
    LlmCodecStTensorInfo* ti = NULL; int n = 0; uint64_t base_off = 0;
    if (llmcodec_safetensors_read(path, &ti, &n, &base_off) != 0) return -1;
    char nm[1024]; uint64_t wo=0,wl=0,so=0,sl=0,bo=0,bl=0; const char* dtw=NULL; const char* dts=NULL; const char* dtb=NULL;
    snprintf(nm,sizeof nm,"%s.weight",base); int ok = (find_off(ti,n,nm,&wo,&wl,&dtw)==0);
    if (ok) { snprintf(nm,sizeof nm,"%s.scales",base); ok = (find_off(ti,n,nm,&so,&sl,&dts)==0); }
    int has_b=0; if (ok) { snprintf(nm,sizeof nm,"%s.biases",base); has_b=(find_off(ti,n,nm,&bo,&bl,&dtb)==0); }
    uint64_t E=0, rows=0, wcols=0, scols=0;
    if (ok) {
        snprintf(nm,sizeof nm,"%s.weight",base);
        for (int i=0;i<n;i++) if (strcmp(ti[i].name,nm)==0) { E=ti[i].shape[0]; rows=ti[i].shape[1]; wcols=ti[i].shape[2]; }
        snprintf(nm,sizeof nm,"%s.scales",base);
        for (int i=0;i<n;i++) if (strcmp(ti[i].name,nm)==0) scols=ti[i].shape[2];
    }
    llmcodec_safetensors_info_free(ti);
    if (!ok || E==0 || rows==0 || wcols==0) { fprintf(stderr, "[wpacked3d] %s: E=%llu rows=%llu wcols=%llu has_b=%d\n", base,(unsigned long long)E,(unsigned long long)rows,(unsigned long long)wcols,has_b); return -1; }
    uint64_t cols = wcols*32ull/(uint64_t)bits;
    if (scols != cols/(uint64_t)group) {
        fprintf(stderr, "[wpacked3d] %s: wcols=%llu scols=%llu cols=%llu group=%d (E=%llu rows=%llu)\n",
                base, (unsigned long long)wcols, (unsigned long long)scols,
                (unsigned long long)cols, group, (unsigned long long)E, (unsigned long long)rows);
        return -1;
    }
    int fd = open(path, O_RDONLY); if (fd<0) return -1;
    struct stat st; if (fstat(fd,&st)!=0){close(fd);return -1;}
    void* map = mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    if (map==MAP_FAILED) return -1;
    out->w=(const uint32_t*)((const uint8_t*)map+base_off+wo);
    out->scales=(const uint16_t*)((const uint8_t*)map+base_off+so);
    out->biases=has_b?(const uint16_t*)((const uint8_t*)map+base_off+bo):NULL;
    out->experts=(uint32_t)E; out->rows=(uint32_t)rows; out->cols=(uint32_t)cols;
    out->bits=bits; out->group=group;
    out->w_stride=(size_t)rows*wcols*4u; out->s_stride=(size_t)rows*scols*2u;
    out->map=map; out->map_bytes=(size_t)st.st_size;
    return 0;
}
void llm_mlx_packed_close3d(LlmMlxPackedW3* w) { if (!w||!w->map) return; munmap(w->map,w->map_bytes); memset(w,0,sizeof *w); }
