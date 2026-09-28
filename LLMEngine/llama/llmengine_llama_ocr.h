/* llmengine_llama_ocr.h — OCR/visione embedded via llama.cpp mtmd (FASE 5B): API C pura
 * (llm_ai_ocr_load/read_image/free_string/unload) sopra il wrapper llm_ai_llamacpp,
 * per IVisionCompleter. Vedi sotto per uso da BASIC e note su contesto/KV. */
#ifndef LLMENGINE_LLAMA_HOST_AI_OCR_H
#define LLMENGINE_LLAMA_HOST_AI_OCR_H

// FASE 5B: OCR/visione embedded via llama.cpp mtmd (vendored in
// native/llm_host_ai/build/_deps/llama_cpp-src/tools/mtmd/). Stesso pattern
// del wrapper testuale llmengine_llama: una libreria C++ (llm_ai_llamacpp)
// che già carica un llama_model, qui estesa con l'encoder visione (mmproj)
// tramite mtmd. API C pura, P/Invoke-friendly.
//
// Uso tipico (da BASIC, dietro IVisionCompleter):
//   llm_ai_ocr_load(model_gguf, mmproj_gguf, n_ctx, n_gpu_layers)
//   llm_ai_ocr_read_image(handle, image_path, prompt, max_tokens) -> char*
//   llm_ai_ocr_free_string / llm_ai_ocr_unload
//
// Il modello testo è lo stesso di llmengine_llama (llama_model), caricato
// qui in un contesto dedicato -- l'OCR non condivide il context della chat
// (niente sessioni persistenti: l'OCR è usa-e-getta per immagine, il KV-cache
// viene ricreato ad ogni chiamata, coerente con il fatto che l'input è
// un'immagine nuova ogni volta, non una conversazione).
//
// Il file immagine viene letto da path (mtmd_helper_bitmap_init_from_file, che
// usa stb_image vendored -- nessuna pipeline di decodifica da scrivere in C#).

#ifdef __cplusplus
extern "C" {
#endif

// llm_ai_ocr_handle è un puntatore opaco al contesto (modello testo + mtmd ctx).
typedef struct llm_ai_ocr_ctx* llm_ai_ocr_handle;

// Carica il modello testo (GGUF) e il vision encoder (mmproj GGUF). Ritorna NULL
// su fallimento (file mancanti/corrotti, architettura non supportata da mtmd).
llm_ai_ocr_handle llm_ai_ocr_load(const char* model_path, const char* mmproj_path,
                                  int n_ctx, int n_gpu_layers);

// Esegue l'OCR/visione sull'immagine al path indicato. image_prompt è il testo
// della richiesta (per PaddleOCR basta la domanda in lingua; il template chat
// del modello viene applicato internamente). Ritorna una stringa UTF-8
// heap-allocata da liberare con llm_ai_ocr_free_string, o NULL su errore
// (immagine illeggibile, decode fallito).
char* llm_ai_ocr_read_image(llm_ai_ocr_handle handle, const char* image_path,
                            const char* image_prompt, int max_tokens);

// Esegue la trascrizione audio sul file al path indicato (wav/mp3/flac via
// miniaudio, auto-detect dal magic bytes). Pensato per modelli multimodali
// audio come Qwen2.5-Omni (mmproj con projector audio): il prompt va dato nel
// formato ChatML del modello. Ritorna stringa heap-allocata, o NULL su errore.
char* llm_ai_ocr_read_audio(llm_ai_ocr_handle handle, const char* audio_path,
                            const char* audio_prompt, int max_tokens);

// Libera una stringa ritornata da llm_ai_ocr_read_image.
void llm_ai_ocr_free_string(char* s);

// Rilascia il contesto OCR (modello testo + mtmd ctx).
void llm_ai_ocr_unload(llm_ai_ocr_handle handle);

#ifdef __cplusplus
}
#endif

#endif /* LLMENGINE_LLAMA_HOST_AI_OCR_H */
