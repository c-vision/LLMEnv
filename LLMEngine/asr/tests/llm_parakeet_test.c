/* llm_parakeet_test.c — trascrive un WAV con Parakeet-TDT e, se dato un testo atteso, confronta le parole.
 * Uso: llm_parakeet_test <parakeet.gguf> <file.wav> [testo-atteso.txt]
 * Esce 77 (skip) se il GGUF o il WAV mancano; 1 se la trascrizione e' lontana dal testo atteso (<85% delle parole). */
#include "llm_parakeet.h"
#include "gmb_asr.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* parole minuscole senza punteggiatura */
static int words(const char* s, char out[][64], int cap) {
    int n = 0, k = 0;
    char cur[64];
    for (const char* p = s;; p++) {
        if (isalnum((unsigned char)*p) || (*p & 0x80)) { if (k < 63) cur[k++] = (char)tolower((unsigned char)*p); }
        else {
            if (k > 0 && n < cap) { cur[k] = '\0'; memcpy(out[n++], cur, (size_t)k + 1); }
            k = 0;
            if (!*p) break;
        }
    }
    return n;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "uso: %s <parakeet.gguf> <file.wav> [atteso.txt]\n", argv[0]); return 2; }
    if (access(argv[1], R_OK) != 0 || access(argv[2], R_OK) != 0) { printf("SKIP: modello o audio assenti\n"); return 77; }
    char err[256] = "";
    LlmParakeet* m = llm_parakeet_load(argv[1], err, sizeof err);
    if (!m) { fprintf(stderr, "load fallito: %s\n", err); return 1; }
    int n = 0;
    float* pcm = gmb_asr_wav_load(argv[2], &n);
    if (!pcm) { fprintf(stderr, "wav non leggibile\n"); llm_parakeet_free(m); return 1; }
    char* text = llm_parakeet_transcribe(m, pcm, n, err, sizeof err);
    if (!text) { fprintf(stderr, "trascrizione fallita: %s\n", err); return 1; }
    printf("%s\n", text);
    int rc = 0;
    if (argc > 3) {
        FILE* f = fopen(argv[3], "rb");
        static char exp_txt[8192];
        size_t k = f ? fread(exp_txt, 1, sizeof exp_txt - 1, f) : 0;
        if (f) fclose(f);
        exp_txt[k] = '\0';
        static char a[400][64], b[400][64];
        int na = words(exp_txt, a, 400), nb = words(text, b, 400), hit = 0;
        for (int i = 0; i < na; i++)
            for (int j = 0; j < nb; j++) if (!strcmp(a[i], b[j])) { hit++; break; }
        printf("parole attese ritrovate: %d/%d\n", hit, na);
        if (na == 0 || hit * 100 < na * 85) rc = 1;
    }
    free(text);
    free(pcm);
    llm_parakeet_free(m);
    return rc;
}
