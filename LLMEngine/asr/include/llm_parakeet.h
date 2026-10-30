/* llm_parakeet.h — riconoscimento vocale con NVIDIA Parakeet-TDT (FastConformer + transducer con durate), da GGUF.
 *
 * Il GGUF (`general.architecture = parakeet`) porta tutto: finestra e banco di filtri mel, pesi dell'encoder
 * (sottocampionamento convoluzionale x8 + 24 blocchi Conformer con attenzione a posizione relativa), rete di
 * predizione LSTM, rete di giunzione e vocabolario SentencePiece. CPU, C11, matrici con Accelerate (cblas) su macOS.
 * Decodifica greedy TDT: ogni passo sceglie un simbolo e una durata (in fotogrammi) e salta avanti. */
#ifndef LLM_PARAKEET_H
#define LLM_PARAKEET_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LlmParakeet LlmParakeet;

/* 1 se `path` e' un GGUF con general.architecture = "parakeet". */
int llm_parakeet_is_gguf(const char* path);

/* Carica il modello (i pesi quantizzati si dequantizzano in f32). NULL e messaggio in `err` se fallisce. */
LlmParakeet* llm_parakeet_load(const char* gguf_path, char* err, size_t errcap);

/* Trascrive audio mono f32 a 16 kHz in [-1,1]. Ritorna il testo (malloc, liberare con free) o NULL.
 * L'audio piu' lungo di ~4 minuti si divide in segmenti tagliati nei punti di minima energia. */
char* llm_parakeet_transcribe(LlmParakeet* m, const float* samples, int n_samples, char* err, size_t errcap);

void llm_parakeet_free(LlmParakeet* m);

#ifdef __cplusplus
}
#endif

#endif
