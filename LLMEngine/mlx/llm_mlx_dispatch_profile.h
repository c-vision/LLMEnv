/* llm_mlx_dispatch_profile.h — G0: contatore DETERMINISTICO dei nodi di dispatch di un
 * passo di decode (campagna rapidMLX, Todo L3 + L26, 2026-09-09).
 *
 * A cosa serve. Le fusioni candidate (L1 RoPE, L2 QKV, L6 gate+up MoE, L7 kernel mancanti)
 * promettono di "ridurre i dispatch per layer", e il cluster C (GLM R5,9x, 122B R4,2x,
 * gpt-oss-120b R4,1x, Mixtral R3,2x) veniva archiviato come "overhead di dispatch" senza
 * che quel numero fosse mai stato misurato. Questo modulo conta i NODI CON PRIMITIVA che il
 * grafo di un passo di decode realmente costruisce, e la loro composizione per tipo di
 * primitiva: e' il denominatore che rende falsificabile "abbiamo N nodi per token, di cui M
 * di tipo X", e che dice subito se una fusione proposta eliminerebbe 8 nodi o 800.
 *
 * Perche' deterministico e non cronometrato: il conteggio non dipende dal carico della
 * macchina, quindi funziona anche nelle finestre condivise dove i tok/s non sono interpretabili
 * (regola §4 del Todo).
 *
 * Come conta. DFS iterativa dalle radici del passo, con un insieme `seen` NUOVO A OGNI PASSO
 * sulle `primitive_id()` (una Primitive = un dispatch: gli output multipli la condividono, e la
 * primitiva gia' visitata non si riattraversa nemmeno da un'altra radice -- i ~2*L+1 output del
 * passo condividono l'attaccatura con la cache); le foglie senza primitiva (pesi, scalari) sono
 * deduplicate per `id()` per non far esplodere la camminata. Il conteggio del passo viene
 * sommato al totale e la dedup NON e' cumulativa tra i passi: vedi il "PERCHE' PER PASSO" sotto.
 *
 * REGOLA OPERATIVA (misurata, non dedotta -- prima stesura di questo strumento diceva il
 * contrario ed era falsa): il conteggio va fatto alla COSTRUZIONE del grafo, PRIMA del
 * dispatch. Instrumentando il dopo-`mlx_async_eval` le radici del passo risultano
 * `has_primitive()==false`, `inputs()` e `siblings()` VUOTI: il grafo di un array valutato non
 * e' piu' camminabile. MECCANISMO NON STABILITO: `array::detach()` (l'unico posto che azzera
 * `primitive`) e' chiamato solo in transforms.cpp, quindi NON sappiamo perche' l'eval produca
 * quello stato; l'unico indizio nei sorgenti e' il commento di `Status::evaluated` in
 * `mlx/array.h` ("if it is not a tracer then it will be detached from the graph"). Nessuna
 * dipendenza da quella spiegazione: qui serve il numero, e il numero si prende prima del dispatch.
 *
 * PERCHE' PER PASSO, NON CUMULATIVO (misurato, era il design iniziale ed era sbagliato): con un
 * `seen` cumulativo sulle `primitive_id()` il primo passo contava 705 nodi e TUTTI i successivi
 * zero. Quando il passo k+1 builda, le `Primitive` del passo k sono morte e malloc restituisce
 * gli STESSI indirizzi: un insieme cumulativo scambia ogni nodo nuovo per "gia' visto".
 * LETTURA DEL NUMERO: "nodi di grafo raggiungibili dal passo", non "nodi nuovi creati dal passo".
 * Se una parte di antenati ancora vivi venga ricontata a ogni passo NON e' stabilito; quel che e'
 * misurato e' che il conteggio per passo non cresce con l'indice (2124 a 8 token come a 16, e
 * `min/passo == max/passo` su tutti i modelli), quindi qualunque sia l'effetto, e' limitato e non
 * cumulativo. Strumento di diagnosi, non numero da release.
 *
 * RISULTATO (2026-09-09, 9 modelli, decode greedy 8 token a contesto 72 -- perche' sia un
 * numero spendibile serve il confronto col reference, non c'e' ancora): 704 nodi/token su
 * llama-1B denso (44,0/layer), 2124 su DeepSeek-32B e su Qwen3-Coder-30B-A3B, 3346 su
 * Qwen3.6-27B, 4193-4437 sui cinque che perdono (gemma-4-31b denso incluso: falsifica la lettura
 * "e' il router MoE"). Le celle e l'arbitrato rimasto: vault
 * `Knowledge/LLMEnv-vs-RapidMLX-Comparison.md` §23ter + Todo item 22 (L26).
 *
 * NESSUN ASSORBIMENTO DEL PREFILL (non e' un'ipotesi, e' misurato): una baseline camminata dalle
 * radici vive alla fine del prefill (cache di sessione + pending_h) marca 0 nodi con primitiva,
 * per lo stesso motivo qui sopra. Il prefill quindi NON entra nel conteggio da se': i numeri di
 * questo strumento sono nodi del DECODE per token.
 *
 * Perimetro: il conteggio copre il path greedy asincrono (default del decode greedy, tutte le
 * architetture). Il path con sampling attivo e la closure compilata (LLM_COMPILE, solo mixtral)
 * NON sono strumentati: nel secondo caso il passo e' UN solo nodo `Compiled`, e il conteggio non
 * misurerebbe nulla di cio' che serve.
 *
 * ABI C, implementazione C++ (deve toccare l'API C++ di mlx, come llm_mlx_fused_rope_append.cpp):
 * llmengine_mlx.c resta C puro.
 */
#ifndef LLM_MLX_DISPATCH_PROFILE_H
#define LLM_MLX_DISPATCH_PROFILE_H

#include <stdbool.h>
#include <stddef.h>

#include "mlx/c/array.h"

#ifdef __cplusplus
extern "C" {
#endif

/* true se LLM_DISPATCH e' impostato. Letto UNA volta: mai getenv nel percorso caldo. */
bool llm_mlx_dp_enabled(void);

/* Azzera il set "gia' visti" e l'istogramma. Va chiamato all'inizio della generazione. */
void llm_mlx_dp_reset(void);

/* MoE: per correggere `MB_in` dei nodi `GatherQMM` (che leggono l'intero banco esperti) si passa
 * il numero di esperti totali e quanti ne attiva il routing (`config.json`). 0 = nessun MoE. */
void llm_mlx_dp_set_moe(int n_experts, int top_k);

/* Speculazione: registra i token ACCETTATI di una passata (verify). La velocita' reale per token
 * = `MB_in` per PASSATA / (token accettati per passata). Deterministico (nessuna GPU). */
void llm_mlx_dp_spec_add(int accepted_tokens);

/* Cammina le radici e conta i nodi con primitive NON gia' visti, accumulandoli nell'istogramma
 * per tipo di primitiva. Da chiamare a grafo del passo COSTRUITO ma PRIMA del dispatch (vedi la
 * REGOLA OPERATIVA sopra: dopo l'eval le radici non sono piu' camminabili). */
void llm_mlx_dp_count(const mlx_array* roots, int n_roots);

/* Attribuzione del TEMPO di un passo, il mezzo che il solo conteggio non ha: `build_sec` e' il
 * tempo di CPU per costruire il grafo (dall'inizio del passo a dopo la camminata di count),
 * `eval_sec` il tempo dal dispatch alla lettura del token (quindi il lavoro GPU accodato, che nel
 * pipeline asincrono e' quello del passo precedente). Senza questa separazione "ns/nodo = 630000"
 * non dice se stiamo lenti sulla CPU o sul Metal, e le due cure sono opposte. */
void llm_mlx_dp_stage(double build_sec, double eval_sec);

/* Stampa su stderr lo stato dell'allocatore Metal (attivi / cache / picco / limite) con un tag di
 * fase. Serve a rispondere DOVE nasce il working set: misurato su Mixtral-8x22B decode attivo =
 * 130,47 GB con limite 130,57 e modello di 74 GB su disco, cioe' ~56 GB di buffer vivi sopra il
 * modello. Chiamato alla fine del caricamento e dopo il prefill, dice se quei GB sono gia' li'
 * (stacking dei pesi) o li creano i passi di decode. Nessuna azione se LLM_DISPATCH non e' attivo. */
void llm_mlx_dp_memory(const char* where);

/* Stampa una riga di riepilogo su stderr: nodi/token, nodi/layer, MB/token (somma dei `nbytes()`
 * degli output), ns/nodo e GB/s_scritto (dal tempo passato), e le prime `top_k` famiglie di
 * primitive per conteggio con i rispettivi MB/token. `steps` e `layers` servono solo a
 * normalizzare; se layers<=0 il campo nodi/layer viene omesso. */
void llm_mlx_dp_report(const char* tag, int steps, int layers, double seconds, int top_k);

#ifdef __cplusplus
}
#endif

#endif /* LLM_MLX_DISPATCH_PROFILE_H */
