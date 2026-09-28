/* llm_mlx_chattpl.h — §D R4.3: CHAT TEMPLATE condiviso con l'inferenza (rendering in C).
 *
 * Mirror byte-per-byte di `ChatTranscripts.BuildLlama3` (C#, verificato contro
 * `apply_chat_template` di mlx-lm): `<|begin_of_text|>` una volta, poi per turno
 * `<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>`; opener assistant
 * `<|start_header_id|>assistant<|end_header_id|>\n\n`.
 *
 * Per l'SFT: la loss va SOLO sull'ultima risposta assistant -> `llm_chattpl_llama3_split`
 * produce (prompt, completion): prompt = BOS + turni fino all'ultimo assistant + opener;
 * completion = contenuto dell'ultimo assistant + `<|eot_id|>`. Poi si tokenizza con lo STESSO
 * tokenizer dell'inferenza (llmtoken) e si usa la mask di llm_mlx_dataset.h.
 *
 * PURO (solo stringhe). */
#ifndef LLM_MLX_CHATTPL_H
#define LLM_MLX_CHATTPL_H

#include <stdio.h>
#include <string.h>

#define LLM_CHAT_BOS   "<|begin_of_text|>"
#define LLM_CHAT_HDRS  "<|start_header_id|>"
#define LLM_CHAT_HDRP  "<|end_header_id|>"
#define LLM_CHAT_EOT   "<|eot_id|>"

static inline void llm_sb_add(char* out, int cap, int* len, const char* s) {
    while (s && *s) { if (*len < cap - 1) out[(*len)++] = *s; s++; }
}
static inline void llm_sb_turn(char* out, int cap, int* len, const char* role, const char* content) {
    llm_sb_add(out, cap, len, LLM_CHAT_HDRS);
    llm_sb_add(out, cap, len, role);
    llm_sb_add(out, cap, len, LLM_CHAT_HDRP "\n\n");
    llm_sb_add(out, cap, len, content);
    llm_sb_add(out, cap, len, LLM_CHAT_EOT);
}

/* Renderizza l'intero transcript (come inferenza, senza opener finale): BOS + tutti i turni. */
static inline void llm_chattpl_llama3(const char* const* roles, const char* const* contents, int n,
                                      char* out, int cap) {
    int len = 0; if (cap > 0) out[0] = 0;
    llm_sb_add(out, cap, &len, LLM_CHAT_BOS);
    for (int i = 0; i < n; i++) llm_sb_turn(out, cap, &len, roles[i], contents[i]);
    out[len < cap ? len : cap - 1] = 0;
}

/* Split SFT: prompt (fino all'opener assistant dell'ULTIMO turno assistant) + completion
 * (contenuto ultimo assistant + eot). Ritorna l'indice dell'ultimo assistant, o -1 se assente. */
static inline int llm_chattpl_llama3_split(const char* const* roles, const char* const* contents, int n,
                                           char* prompt, int pcap, char* completion, int ccap) {
    int last = -1;
    for (int i = 0; i < n; i++) if (strcmp(roles[i], "assistant") == 0) last = i;
    int pl = 0; if (pcap > 0) prompt[0] = 0;
    int cl = 0; if (ccap > 0) completion[0] = 0;
    if (last < 0) {   /* nessuna risposta: prompt = transcript completo, completion vuota */
        llm_chattpl_llama3(roles, contents, n, prompt, pcap);
        return -1;
    }
    llm_sb_add(prompt, pcap, &pl, LLM_CHAT_BOS);
    for (int i = 0; i < last; i++) llm_sb_turn(prompt, pcap, &pl, roles[i], contents[i]);
    llm_sb_add(prompt, pcap, &pl, LLM_CHAT_HDRS "assistant" LLM_CHAT_HDRP "\n\n");   /* opener */
    prompt[pl < pcap ? pl : pcap - 1] = 0;
    llm_sb_add(completion, ccap, &cl, contents[last]);
    llm_sb_add(completion, ccap, &cl, LLM_CHAT_EOT);
    completion[cl < ccap ? cl : ccap - 1] = 0;
    return last;
}

#endif /* LLM_MLX_CHATTPL_H */
