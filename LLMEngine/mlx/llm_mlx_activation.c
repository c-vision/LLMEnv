/* ============================================================================
 * llm_mlx_activation.c — dominio attivazioni compilate: flag LLM_FUSE_GEGLU(_METAL)/SWIGLU,
 * closure pure GeGLU/SwiGLU (compile-mode) usate da dense_geglu_ffn / ramo sorted MoE. Estratto
 * VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_engine_opts.h"   /* opt_env: registro */
#include "llm_mlx_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* LLM_FUSE_GEGLU (2026-09-06, vedi piano Proposta-614GBs-Banda-Metal-DeepDive.md §9
 * "RapidMLX -- analisi codice reale" punto 6): l'attivazione GeGLU sotto (gelu_approx(gate)*up)
 * e' scomposta in 9 dispatch MLX separati (power/multiply/add/multiply/tanh/add/multiply x2 +
 * la moltiplicazione finale per `up`), ripetuti per OGNI layer -- 60 layer su gemma-4-31b =
 * 540 dispatch/token solo per questa attivazione, mentre rapidMLX la fonde in un solo kernel
 * via `@partial(mx.compile, shapeless=True)` (`gemma4_vendored/language.py:29-31`, verificato
 * leggendo il sorgente). Stessa tecnica qui: una closure PURA (nessuno stato di sessione, nessun
 * peso catturato -- input gate/up, output act, sicura da condividere fra TUTTE le sessioni/
 * modelli) compilata una sola volta con `MLX_COMPILE_MODE_ENABLED` (fusione REALE richiesta,
 * a differenza del decode-step di RMLX10 dove la fusione altera i risultati -- qui la catena e'
 * puramente elementwise, nessuna riduzione, rischio di deriva numerica molto piu' basso, ma
 * ANCORA da trattare come tale: verificare parita' greedy prima di fidarsi, stesso standard delle
 * altre feature sperimentali di questo file). Il compile-mode globale viene ripristinato a
 * NO_FUSE subito dopo (usato altrove da RMLX10 Fase B, §9, per un motivo diverso e incompatibile
 * con la fusione reale). */
static int g_fuse_geglu_state = -1;
bool llm_mlx_fuse_geglu_enabled(void) {
    if (g_fuse_geglu_state < 0) g_fuse_geglu_state = (llm_ai_mlx_opt_env("LLM_FUSE_GEGLU") != NULL) ? 1 : 0;
    return g_fuse_geglu_state == 1;
}
/* PROVATO CONTROPRODUCENTE (2026-09-06, verificato col profiler xctrace/Time Profiler):
 * iokit_user_client_trap passa da 129ms a 1855ms (14x) con LLM_FUSE_GEGLU=1 -- il
 * meccanismo mlx_compile/mlx_closure_apply sopra NON riduce i dispatch reali, li
 * aumenta. Lasciato per riferimento (dietro flag OFF, mai proporlo come base per
 * altre architetture) -- vedi llm_mlx_fuse_geglu_metal_enabled sotto per il tentativo
 * corretto (kernel Metal diretto, no compile/closure). */
/* LLMMETAL_FUSE_GEGLU_METAL: come sopra ma via kernel Metal reale (llm_mlx_geglu_kernel.h,
 * mlx_fast_metal_kernel -- stesso meccanismo del kernel GDN gia' in produzione, NESSUN
 * passaggio da mlx_compile/closure). Un dispatch invece di 9, thread-per-elemento. */
static int g_fuse_geglu_metal_state = -1;
bool llm_mlx_fuse_geglu_metal_enabled(void) {
    if (g_llm_mlx_train_composable) return false;   /* training: no kernel Metal (no vjp) */
    if (g_fuse_geglu_metal_state < 0) g_fuse_geglu_metal_state = (llm_ai_mlx_opt_env("LLMMETAL_FUSE_GEGLU_METAL") != NULL) ? 1 : 0;
    return g_fuse_geglu_metal_state == 1;
}
static int llm_mlx_geglu_fn(mlx_vector_array* out, const mlx_vector_array in, void* payload) {
    (void)payload;
    mlx_array gate = mlx_array_new(), up = mlx_array_new();
    mlx_vector_array_get(&gate, in, 0);
    mlx_vector_array_get(&up, in, 1);
    mlx_array three = mlx_array_new_float32(3.0f);
    mlx_array c0044715 = mlx_array_new_float32(0.044715f);
    mlx_array sqrt2_pi = mlx_array_new_float32(0.7978845608028654f);
    mlx_array one = mlx_array_new_float32(1.0f);
    mlx_array half = mlx_array_new_float32(0.5f);
    mlx_array gate3 = mlx_array_new(), gate3_scaled = mlx_array_new(), inner = mlx_array_new();
    mlx_array inner_scaled = mlx_array_new(), tanh_out = mlx_array_new(), one_plus_tanh = mlx_array_new();
    mlx_array gelu_out = mlx_array_new(), act = mlx_array_new();
    mlx_stream stream = mlx_default_gpu_stream_new();
    bool ok = mlx_power(&gate3, gate, three, stream) == 0;
    if (ok) ok = mlx_multiply(&gate3_scaled, gate3, c0044715, stream) == 0;
    if (ok) ok = mlx_add(&inner, gate, gate3_scaled, stream) == 0;
    if (ok) ok = mlx_multiply(&inner_scaled, inner, sqrt2_pi, stream) == 0;
    if (ok) ok = mlx_tanh(&tanh_out, inner_scaled, stream) == 0;
    if (ok) ok = mlx_add(&one_plus_tanh, tanh_out, one, stream) == 0;
    if (ok) ok = mlx_multiply(&gelu_out, gate, one_plus_tanh, stream) == 0;
    if (ok) ok = mlx_multiply(&gelu_out, gelu_out, half, stream) == 0;
    if (ok) ok = mlx_multiply(&act, gelu_out, up, stream) == 0;
    int rc = 1;
    if (ok) {
        mlx_vector_array built = mlx_vector_array_new_data(&act, 1);
        rc = mlx_vector_array_set(out, built) == 0 ? 0 : 1;
        mlx_vector_array_free(built);
    }
    mlx_stream_free(stream);
    mlx_array_free(gate); mlx_array_free(up);
    mlx_array_free(three); mlx_array_free(c0044715); mlx_array_free(sqrt2_pi);
    mlx_array_free(one); mlx_array_free(half);
    mlx_array_free(gate3); mlx_array_free(gate3_scaled); mlx_array_free(inner);
    mlx_array_free(inner_scaled); mlx_array_free(tanh_out); mlx_array_free(one_plus_tanh);
    mlx_array_free(gelu_out); mlx_array_free(act);
    return rc;
}
static mlx_closure g_geglu_closure;
static bool g_geglu_closure_valid = false;
bool llm_mlx_geglu_compiled(mlx_array gate_out, mlx_array up_out, mlx_array* out_act, mlx_stream stream) {
    (void)stream;
    if (!g_geglu_closure_valid) {
        mlx_closure raw = mlx_closure_new_func_payload(llm_mlx_geglu_fn, NULL, NULL);
        mlx_set_compile_mode(MLX_COMPILE_MODE_ENABLED);
        int rc = mlx_compile(&g_geglu_closure, raw, /*shapeless=*/true);
        mlx_set_compile_mode(MLX_COMPILE_MODE_NO_FUSE);
        mlx_closure_free(raw);
        if (rc != 0) return false;
        g_geglu_closure_valid = true;
    }
    mlx_array ins[2]; ins[0] = gate_out; ins[1] = up_out;
    mlx_vector_array in = mlx_vector_array_new_data(ins, 2);
    mlx_vector_array res = mlx_vector_array_new();
    int rc = mlx_closure_apply(&res, g_geglu_closure, in);
    bool ok = (rc == 0) && mlx_vector_array_size(res) == 1;
    if (ok) mlx_vector_array_get(out_act, res, 0);
    mlx_vector_array_free(in);
    mlx_vector_array_free(res);
    return ok;
}

/* LLM_FUSE_SWIGLU_PREFILL (2026-09-07, roadmap P3 "graph-reuse su prefill MoE batched" --
 * Knowledge/LLMEnv-RapidMLX-Gap-Closure-Strategy.md P3 punto 1): l'ipotesi MAI testata di
 * D16ter ("il costo fisso di mlx_closure_apply si ammortizza su N righe nel prefill batched")
 * applicata QUI, non al decode dove e' gia' falsificata (LLM_FUSE_GEGLU, -57%/14x dispatch in
 * piu', vedi sopra). Attivazione SwiGLU (sigmoid+2 multiply, 3 dispatch) del ramo "sorted" di
 * qwen3_moe_ffn_block (M=n_tokens*top_k>=64, ovvero SOLO prefill batched, mai decode a
 * n_tokens==1) -- stessa architettura di chiusura di LLM_FUSE_GEGLU (closure pura, shapeless,
 * nessun peso catturato, condivisibile fra sessioni), stesso obbligo di verifica prima di
 * fidarsi: default OFF, misurare parita' greedy PRIMA di misurare la velocita'. */
static int g_fuse_swiglu_prefill_state = -1;
bool llm_mlx_fuse_swiglu_prefill_enabled(void) {
    /* mlx_lm attiva SEMPRE la SwiGLU compilata (`activations.py`: @partial(mx.compile,
     * shapeless=True)) sia in prefill sia in decode. In mlx-c pero' `mlx_closure_apply` ha un
     * costo fisso CPU per applicazione che a n_tokens==1 NON si ammortizza: A/B V2-Lite
     * (2026-09-13) eager 169,1 vs compiled 163,9 tok/s (+3,2% decode), IDS greedy identici.
     * Quindi ora vale SOLO per il prefill battezzato (sorted, n_tokens>1); il decode usa
     * l'eager. Default ON; kill-switch LLM_FUSE_SWIGLU_PREFILL=0 per l'A/B. */
    if (g_fuse_swiglu_prefill_state < 0) {
        const char* e = llm_ai_mlx_opt_env("LLM_FUSE_SWIGLU_PREFILL");
        g_fuse_swiglu_prefill_state = (e != NULL && strcmp(e, "0") == 0) ? 0 : 1;
    }
    return g_fuse_swiglu_prefill_state == 1;
}
static int llm_mlx_swiglu_fn(mlx_vector_array* out, const mlx_vector_array in, void* payload) {
    (void)payload;
    mlx_array gate = mlx_array_new(), up = mlx_array_new();
    mlx_vector_array_get(&gate, in, 0);
    mlx_vector_array_get(&up, in, 1);
    mlx_array sig = mlx_array_new(), silu = mlx_array_new(), act = mlx_array_new();
    mlx_stream stream = mlx_default_gpu_stream_new();
    bool ok = mlx_sigmoid(&sig, gate, stream) == 0;
    if (ok) ok = mlx_multiply(&silu, gate, sig, stream) == 0;
    if (ok) ok = mlx_multiply(&act, silu, up, stream) == 0;
    int rc = 1;
    if (ok) {
        mlx_vector_array built = mlx_vector_array_new_data(&act, 1);
        rc = mlx_vector_array_set(out, built) == 0 ? 0 : 1;
        mlx_vector_array_free(built);
    }
    mlx_stream_free(stream);
    mlx_array_free(gate); mlx_array_free(up);
    mlx_array_free(sig); mlx_array_free(silu); mlx_array_free(act);
    return rc;
}
/* Closure per-RANK (0..7). Una sola closure `shapeless` NON basta: le forme che la SwiGLU
 * riceve nel decode DeepSeek alternano ogni layer fra 4D (routed `(1,6,1,1408)`) e 3D
 * (shared/denso `(1,1,2816)`/`(1,1,10944)`). `mx.compile(shapeless=True)` fissa la
 * STRUTTURA (incluso il numero di dimensioni): al cambio di rank la closure viene ri-tracciata
 * e per quel passo espone i 3 op interni (Sigmoid+Multiply+Multiply) invece del nodo fuso ->
 * misurato: 2 Sigmoid/token residui (1 routed + 1 shared), gli altri 51 applicati/token fusi.
 * Con una closure PER RANK ogni classe mantiene il proprio grafo compilato: 0 Sigmoid.
 * `shapeless=True` dentro la rank (la dimensione M varia in prefill) senza ricompilare. */
/* Compilazione PER-SHAPE (`shapeless=false`) con cache per firma. La `shapeless=true` unica
 * NON basta: con forme che variano (routed 4D `(1,6,1,1408)` / shared 3D `(1,1,2816)`, e M
 * variabile in prefill) alcune applicazioni degenerano e MLX riespone i 3 op interni invece
 * del grafo fuso -> misurato: 2 Sigmoid/token (1 routed + 1 shared), le altre 51 fuse.
 * Compilando per-shape ESATTA ogni forma usa il suo grafo fuso: 0 Sigmoid. Cache limitata
 * (32): oltre, fallback e non si compila piu' (nessun costo illimitato in prefill). */
/* NOTA (2026-09-13): provate e SCARTATE (misurate, non migliorano) la closure per-rank
 * (4D routed / 3D shared: Sigmoid resta 2/token) e la compilazione per-shape
 * (`shapeless=false` + cache per firma: PEGGIORA a 3/token, fa deoptare anche la densa
 * 10944). La closure `shapeless` unica qui sotto e' lo stato migliore: 51/53 applicazioni
 * fuse, 2 espongono i 3 op interni (Sigmoid residua) -- deopt interno di MLX, non codice LLM. */
static mlx_closure g_swiglu_closure;
static bool g_swiglu_closure_valid = false;

bool llm_mlx_swiglu_compiled(mlx_array gate_out, mlx_array up_out, mlx_array* out_act, mlx_stream stream) {
    (void)stream;
    if (!g_swiglu_closure_valid) {
        mlx_closure raw = mlx_closure_new_func_payload(llm_mlx_swiglu_fn, NULL, NULL);
        mlx_set_compile_mode(MLX_COMPILE_MODE_ENABLED);
        int rc = mlx_compile(&g_swiglu_closure, raw, /*shapeless=*/true);
        mlx_set_compile_mode(MLX_COMPILE_MODE_NO_FUSE);
        mlx_closure_free(raw);
        if (rc != 0) return false;
        g_swiglu_closure_valid = true;
    }
    mlx_array ins[2]; ins[0] = gate_out; ins[1] = up_out;
    mlx_vector_array in = mlx_vector_array_new_data(ins, 2);
    mlx_vector_array res = mlx_vector_array_new();
    /* FUSIONE REALE (2026-10-16, TODO MLX-Trace-Matmul-Quantizzata): la fusione di MLX si decide
     * a APPLY-time (traccia + compile_fuse al primo apply per firma), NON a mlx_compile-time. Col
     * modo NO_FUSE globale la closure restava l'eager (sigmoid+mul+mul separati, ciascuno arrotondato
     * a bf16), mentre mlx_lm (`@mx.compile(shapeless=True) swiglu`) esegue il kernel FUSO: differenza
     * di 1 ulp bf16 su qualche elemento (Coder-30B 8bit, layer 2: 8/30720), che si propaga ai
     * logits. Quindi ENABLED solo attorno all'apply, poi si ripristina NO_FUSE. */
    mlx_set_compile_mode(MLX_COMPILE_MODE_ENABLED);
    int rc = mlx_closure_apply(&res, g_swiglu_closure, in);
    mlx_set_compile_mode(MLX_COMPILE_MODE_NO_FUSE);
    bool ok = (rc == 0) && mlx_vector_array_size(res) == 1;
    if (ok) mlx_vector_array_get(out_act, res, 0);
    mlx_vector_array_free(in);
    mlx_vector_array_free(res);
    return ok;
}
