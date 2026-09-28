/* ============================================================================
 * llm_mlx_bf16.c — dominio politica dtype: ammissione/parse della mode bf16/f16, apply,
 * auto-pair attivazione<->scales, hoisting (promote) delle scales/bias/norm. Estratto
 * VERBATIM da llmengine_mlx.c (split).
 * ============================================================================ */
#include "llmengine_mlx.h"
#include "llm_mlx_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* bf16 chiesto dall'UTENTE: una sola regola di ammissione, usata dai due percorsi non
 * validati -- la diagnostica LLM_BF16_DENSE (qui sotto) e la config esplicita via
 * llm_ai_mlx_model_set_bf16_mode. Prima di qui la allow-list esisteva SOLO nel ramo env:
 * l'API pubblica llm_ai_mlx_model_set_bfloat16 non controllava nulla, quindi un chiamante
 * poteva chiedere bf16 su un mxfp4/mxfp8/nvfp4 -- cio' che il commento del default
 * architetturale poco sopra dichiara "10x PIU' lenti o crashano". NON qui dentro il default
 * per-architettura: quello e' una whitelist gia' validata con misure, e passarcelo
 * cambierebbe un comportamento misurato (gpt_oss+per_tensor_quant e' il caso che non
 * combacia con la allow-list). mode: "both"|"embed"|"matmul"|"off". 0 applicato,
 * -1 quant_mode non ammesso, -2 argomento non valido. Mai silenzioso. */
static bool llm_mlx_bf16_mode_allowed(const char* quant_mode) {
    return strcmp(quant_mode, "affine") == 0 || strcmp(quant_mode, "none") == 0;
}

/* Una sola definizione di "mode accettata", condivisa tra l'helper e il setter esportato: senza di
 * questa il setter doveva distinguere "mode sconosciuta" da "architettura vietata" e le
 * confondeva (una mode inesistente su llama veniva rifiutata col messaggio di llama). */
bool llm_mlx_bf16_mode_known(const char* mode) {
    if (mode == NULL) return false;
    /* L19: "bf16" e' sinonimo esplicito di "both" (nessun nuovo stato); "f16"/"fp16" = both +
     * float16, il dtype che si accoppia con le SCALE F16 dei checkpoint DeepSeek-R1-distill
     * (vedi g_act16: la GEMV 8-bit prende il kernel veloce solo se x e le scale sono stessi bit). */
    return strcmp(mode, "both") == 0 || strcmp(mode, "embed") == 0 || strcmp(mode, "matmul") == 0
        || strcmp(mode, "off") == 0 || strcmp(mode, "fp32") == 0
        || strcmp(mode, "bf16") == 0 || strcmp(mode, "f16") == 0 || strcmp(mode, "fp16") == 0;
}

/* Un solo punto che scompone una mode nello STATO COMPLETO (due bit + flavor). Il flavor e' bf16
 * per ogni mode storica (both/embed/matmul/off/fp32): chiedere "matmul" NON eredita un "f16"
 * richiesto prima, per la stessa ragione per cui i due bit si assegnano sempre entrambi. */
static bool llm_mlx_bf16_mode_parse(const char* mode, bool* on_embed, bool* on_matmul,
                                    mlx_dtype* dt) {
    if (!llm_mlx_bf16_mode_known(mode)) return false;
    const bool want_f16 = strcmp(mode, "f16") == 0 || strcmp(mode, "fp16") == 0;
    const char* pair = (want_f16 || strcmp(mode, "bf16") == 0) ? "both" : mode;
    *on_embed = strcmp(pair, "matmul") != 0;
    *on_matmul = strcmp(pair, "embed") != 0;
    *dt = want_f16 ? MLX_FLOAT16 : MLX_BFLOAT16;
    return true;
}

/* I bit chiesti da `mode` sono GIA' accesi? (mode ASSUNTA nota e non-off). Serve perche' la
 * allow-list e' una regola di AMMISSIONE, non uno stato: il caso reale e' gpt-oss-20b MXFP4-Q8,
 * quant_mode=mxfp4 ma bf16 acceso dal default architettuale validato (want_bf16 via
 * per_tensor_quant, +18% prefill). Lì "both" ricadeva nel rifiuto "formati a blocchi: 10x piu'
 * lento o crash" mentre lo stato richiesto era gia' quello in corso: un NO-OP spacciato per
 * diniego. Misurato 2026-09-08. */
bool llm_mlx_bf16_bits_on(const LlmMlxModel* m, const char* mode) {
    bool want_embed, want_matmul;
    mlx_dtype dt;
    if (!llm_mlx_bf16_mode_parse(mode, &want_embed, &want_matmul, &dt)) return false;
    if (want_embed != m->cfg.bfloat16_input || want_matmul != g_bf16_all) return false;
    /* Il FLAVOR e' parte dello stato: "f16" su un modello gia' in bf16 NON e' un no-op, altrimenti
     * il dtype giusto per DeepSeek resterebbe irraggiungibile. Con entrambi i bit spenti il
     * flavor non esiste: nessuna attivazione a 16 bit da confrontare. */
    return (!want_embed && !want_matmul) || dt == g_act16;
}

int llm_mlx_apply_bf16_mode(LlmMlxModel* m, const char* mode, const char* origin) {
    if (m == NULL || !llm_mlx_bf16_mode_known(mode)) {
        fprintf(stderr, "[bf16:%s] mode \"%s\" non valida (attese: both|embed|matmul|bf16|f16|fp16|off|fp32)\n",
                origin, mode ? mode : "(null)");
        return -2;
    }
    if (strcmp(mode, "off") == 0 || strcmp(mode, "fp32") == 0) {
        /* Spegnere e' sempre lecito, anche dove il default architetturale ha acceso bf16. */
        m->cfg.bfloat16_input = false;
        g_bf16_all = false;
        g_act16 = MLX_BFLOAT16; /* la mode e' lo stato COMPLETO: non lasciare un flavor orfano */
        fprintf(stderr, "[bf16:%s] fp32 esplicito: model_type=%s quant_mode=%s\n",
                origin, m->cfg.model_type, m->cfg.quant_mode);
        return 0;
    }
    /* Chiedere bit gia' accesi NON e' una richiesta da passare al vaglio: e' un no-op. Senza
     * questo corto circuito la allow-list negava "both" a gpt-oss-20b MXFP4-Q8 che quel bit lo
     * ha gia' accesa di default (vedi llm_mlx_bf16_bits_on). */
    if (llm_mlx_bf16_bits_on(m, mode)) {
        fprintf(stderr, "[bf16:%s] gia' attivo: model_type=%s quant_mode=%s mode=%s embed=%d "
                        "matmul=%d (nessuna modifica)\n", origin, m->cfg.model_type,
                m->cfg.quant_mode, mode, (int)m->cfg.bfloat16_input, (int)g_bf16_all);
        return 0;
    }
    if (!llm_mlx_bf16_mode_allowed(m->cfg.quant_mode)) {
        fprintf(stderr, "[bf16:%s] rifiutato: quant_mode=%s non e' affine/none (formati a blocchi: "
                        "bf16 documentato 10x piu' lento o crash)\n", origin, m->cfg.quant_mode);
        return -1;
    }
    /* Bisection G2 passo 1: i due bit NON sono la stessa cosa -- bfloat16_input governa
     * embed/router/norm/dtype-costanti, g_bf16_all governa matmul+rms_norm. La mode e' lo STATO
     * COMPLETO dei due bit, non una richiesta di accensione: si assegnano SEMPRE entrambi.
     * Assegnare solo i bit "on" rendeva "embed" dopo "matmul" un both (e un cambio a caldo su un
     * modello gia' carico lasciava acceso il bit che l'utente aveva appena tolto): MISURATO
     * 2026-09-08, la bisection in-process dava top-1 163 = il numero di "both", non 162. */
    bool on_embed, on_matmul;
    mlx_dtype dt;
    llm_mlx_bf16_mode_parse(mode, &on_embed, &on_matmul, &dt); /* mode gia' valida sopra */
    m->cfg.bfloat16_input = on_embed;
    g_bf16_all = on_matmul;
    g_act16 = dt;
    fprintf(stderr, "[bf16:%s] applicato: model_type=%s quant_mode=%s mode=%s dt=%s embed=%d matmul=%d\n",
            origin, m->cfg.model_type, m->cfg.quant_mode, mode, llm_mlx_act16_name(),
            (int)on_embed, (int)on_matmul);
    return 0;
}

/* Il dtype delle `.scales` del checkpoint: vive nei safetensors, NON nel config, quindi si puo'
 * sapere solo dopo load_all_shards. Campiona la prima proiezione quantizzata del layer 0 (uno
 * script di quantizzazione usa lo stesso dtype di scala in tutto il file). false se il modello
 * non ha nessuna tripla quantizzata nei nomi provati. */
static bool llm_mlx_scales_dtype(const MultiShardWeights* weights, mlx_dtype* out) {
    static const char* bases[] = {"model.layers.0.self_attn.q_proj",
                                  "model.layers.0.mlp.gate_proj",
                                  "model.layers.0.attention.wq",
                                  "model.layers.0.feed_forward.w1"};
    for (size_t i = 0; i < sizeof(bases) / sizeof(bases[0]); i++) {
        mlx_array w = mlx_array_new(), s = mlx_array_new(), b = mlx_array_new();
        /* multishard_get_triple tollera i tensori PLAIN (scales assente, ctx==NULL): qui e'
         * proprio "non e' la tripla giusta", si passa al nome successivo. */
        bool quant = multishard_get_triple(weights, bases[i], &w, &s, &b) && s.ctx != NULL;
        if (quant) *out = mlx_array_dtype(s);
        mlx_array_free(w); mlx_array_free(s); mlx_array_free(b);
        if (quant) return true;
    }
    return false;
}

/* L19, la regola della COPPIA applicata come default. `mlx_quantized_matmul`/`mlx_gather_qmm`
 * prendono il kernel veloce solo se i bit di `x` sono gli stessi delle `scales`: sul denso qwen2
 * affine 8-bit con scale F16 (DeepSeek-R1-Distill-Qwen-32B-8bit) questo vale +27% di decode, con
 * gate di qualita' verde (loss +0,0003 nats/token, top-1 -3/1800 su 1800, IDS greedy identici) e
 * parita' misurata contro rapidMLX (14,60 vs 14,80 tok/s nella stessa finestra). Lo STESSO default
 * vale anche sul prefill (L21, N~9000 di testo reale): 1,67x sul fp32, dove bf16 resta SOTTO il fp32.
 *
 * Dal 2026-09-09 la regola copre DUE casi, perche' il default architetturale indovina i BIT ma non
 * il FLAVOR (`g_act16 = MLX_BFLOAT16` e' scritto li', il dtype delle scale sta nel checkpoint):
 *  - bit SPENTI (il caso storico): solo il denso qwen2 affine 8-bit, come sopra;
 *  - bit ACCESI dal default per-architettura: se le scale sono F16, `x` BF16 non coincide e
 *    `promote(bf16,f16) = FLOAT32` (type_rules, mlx/dtype.cpp:34-49) -> la "promozione" dei MoE
 *    degenera in matematica fp32 e paga anche la conversione delle scale.
 * La regola e' auto-limitante e lo si e' MISURATO sugli header safetensors: fra i MoE affine
 * promossi le scale sono TUTTE BF16 tranne Mixtral (tutte 1569 le sue `.scales` = F16; GLM 459,
 * Coder-30B 386, Qwen3.6-35B 512 = BF16), quindi qui cambia SOLO Mixtral. Misura su quello:
 * decode x1,97/x3,61/x2,90 e prefill x1,11/x2,16/x1,48 su 3 rep alternate (canone decodectx
 * 4096/32), gate di qualita' in-process both vs f16: loss 0,44291 -> 0,44205, top-1 298 -> 300
 * su 400, argmax diversi 10/800 = 1,25% (le promozioni GIA' dentro il default stanno a 2,11% e
 * 2,33%) con 0 ref diversi. Nota per i misuratori: x non si deduce dai nodi AsType del grafo G0,
 * che portano l'out_type cioe' il RISULTATO della promozione, non l'input.
 *
 * Una cosa che NON fa: non tocca nulla se l'ospite ha chiesto un dtype esplicitamente (precede
 * sempre la richiesta, che arriva da LLM_BF16_DENSE o da mlxComputeDtype). */
void llm_mlx_auto_pair_act_dtype(LlmMlxModel* m, bool explicit_requested) {
    if (explicit_requested || m == NULL) return;
    /* Bit gia' accesi SENZA richiesta esplicita = acceso dal default per-architettura: qui la
     * scelta non e' "se" accendere ma "quale" flavor, e apply_bf16_mode lo sa (il flavor e'
     * parte dello stato, vedi llm_mlx_bf16_bits_on: "f16" su un modello in bf16 NON e' no-op). */
    const bool promoted_by_default = m->cfg.bfloat16_input && g_bf16_all;
    if (m->cfg.bfloat16_input || g_bf16_all) {
        if (!promoted_by_default || g_act16 != MLX_BFLOAT16) return; /* flavor scelto da altri */
    } else {
        bool is_llama = (strcmp(m->cfg.model_type, "llama") == 0);
        bool is_deepseek = (strcmp(m->cfg.model_type, "deepseek_v2") == 0);
        bool is_qwen2_8 = (strcmp(m->cfg.model_type, "qwen2") == 0 && m->cfg.quant_bits == 8);
        /* 2-bit affine (Ternary-Bonsai-4B-mlx-2bit): f16 batte fp32 di +29% sul prefill e
         * +1% sul decode, IDS identici (512/2048). La vecchia "f16 peggiora" era rumore.
         * Kill-switch LLM_BF16_DENSE=off (o la richiesta esplicita) ha precedenza. */
        bool is_2bit = (m->cfg.quant_bits == 2);
        if (!is_llama && !is_deepseek && !is_qwen2_8 && !is_2bit) return;
        if (is_llama || is_deepseek) {
            /* llama: attivazioni f16 (NON bf16) quando le scale sono f16 -- la cache KV e' f16,
             * tenerle f16 rimuove ~4 AsType/layer. bf16 BOTH corrompe llama device-side (task #5),
             * ma f16 e' verificato bit-identico (IDS N=512/8192/32768) e +12%/+7%/+2% decode,
             * +35% prefill. Kill-switch LLMENGINE_LLAMA_F16=0. NON generalizzato: su qwen3 denso
             * (Ternary-Bonsai-4B-2bit) f16 PEGGIORA (83-146 vs 157-165 tok/s) -> resta fp32. */
            const char* e = getenv("LLMENGINE_LLAMA_F16");
            if (e != NULL && strcmp(e, "0") == 0) return;
        }
    }
    if (strcmp(m->cfg.quant_mode, "affine") != 0) return;
    mlx_dtype scales_dt;
    if (!llm_mlx_scales_dtype(&m->weights, &scales_dt) || scales_dt != MLX_FLOAT16) return;
    llm_mlx_apply_bf16_mode(m, "f16", promoted_by_default ? "auto pair scale=F16 (MoE promosso)"
                                                          : "auto pair scales=F16");
}

/* LLM_F32_SCALES (default ON; opt-out LLM_F32_SCALES=0) -- hoisting del cast delle scale quantizzate.
 * mlx_quantized_matmul promote le scale/bias di ogni peso quantizzato al dtype
 * dell'attivazione A OGNI CALL. Su un modello a attivazione fp32 (llama: escluso dal
 * bf16 per il task #5) con scale del checkpoint in bf16/f16, questo emette 2 nodi AsType
 * per proiezione, per token. Misurato (probe deterministico sull'mlx pinnato, e
 * LLM_DISPATCH su Llama-3.2-1B): fp32x/bf16s -> 2 AsType, fp32x/fp32s -> 0; decode
 * 261 AsType/token = 147,58 MB = 94% dei byte scritti (GB/s 53,6 contro roofline ~500).
 * Qui materializziamo le scale/bias in fp32 UNA VOLTA al load: il matmul le trova gia'
 * del dtype giusto e non ricasta piu' per token. bf16/f16 -> fp32 e' un allargamento
 * ESATTO (conserva ogni valore): l'output greedy resta bit-identico, cambia solo DOVE
 * avviene lo STESSO cast. Agisce SOLO se l'attivazione sara' fp32; per i modelli bf16/f16
 * lo stesso hoisting sarebbe lossy (bf16<->f16) e viene omesso (ritorno 0). */
static bool llm_mlx_f32_scales_enabled(void) {
    static int v = -1;
    if (v < 0) { const char* e = getenv("LLM_F32_SCALES"); v = !(e && strcmp(e, "0") == 0); }  /* default ON, opt-out =0 */
    return v != 0;
}

/* ".scales" e ".biases" sono entrambi 7 byte: match sul suffisso del nome del tensore. */
static bool llm_mlx_is_scales_or_bias(const char* k) {
    size_t n = strlen(k);
    return n >= 7 && (strcmp(k + n - 7, ".scales") == 0 || strcmp(k + n - 7, ".biases") == 0);
}

/* Peso di una normalizzazione (RMSNorm/LayerNorm): suffisso ".weight" con "norm" nel nome.
 * Con attivazioni fp32 questi pesi (spesso fp16 nel checkpoint) generano 2 AsType/layer
 * (input_layernorm + post_attention_layernorm) a OGNI token: promuoverli a fp32 al load li
 * elimina (fp16->fp32 e' un allargamento ESATTO -> output bit-identico). */
static bool llm_mlx_is_norm_weight(const char* k) {
    size_t n = strlen(k);
    if (n < 7 || strcmp(k + n - 7, ".weight") != 0) return false;
    return strstr(k, "norm") != NULL || strstr(k, "layernorm") != NULL;
}

/* LLM_F16_SCALES (default ON): quando l'attivazione e' f16, allinea al load le scale/bias/norm
 * a f16. Senza, una scala bf16 con x f16 fa `promote(f16,bf16)=f32` a OGNI quantized_matmul ->
 * AsType per-token (misurato V2-Lite: 156 AsType/token; `LLM_BF16_DENSE=off` ne lascia 27 con IDS
 * identico). Con le scale f16, `promote(f16,f16)=f16` -> 0 AsType e si mantiene la velocita' f16
 * (bf16/f16 attivazione sarebbero ~2x piu' lente). Cast bf16->f16: piu' mantissa, range minore;
 * le scale reali stanno in [~1e-6,10] (dentro il range f16 non-subnormale). Verifica: IDS. */
static bool llm_mlx_f16_scales_enabled(void) {
    static int v = -1;
    if (v < 0) { const char* e = getenv("LLM_F16_SCALES"); v = !(e && strcmp(e, "0") == 0); }
    return v != 0;
}

/* LLM_BF16_SCALES (default OFF, opt-in): con attivazione bf16 allinea al load scale/bias/norm a
 * bf16. Senza, una scala a 16-bit di dtype diverso (tipico: f16 nei checkpoint) fa
 * `promote(bf16,f16)=f32` a OGNI quantized_matmul -> AsType per-token (misurato Qwen3.6-35B-A3B:
 * 314 AsType/token, 10% dei nodi). Al load il cast a bf16 e' una perdita di mantissa rispetto al
 * promote per-call (che va in f32) -> OPT-IN finche' non e' validato con IDS/gate. */
static bool llm_mlx_bf16_scales_enabled(void) {
    static int v = -1;
    if (v < 0) { const char* e = getenv("LLM_BF16_SCALES"); v = (e != NULL && strcmp(e, "0") != 0); }
    return v != 0;
}

int llm_mlx_promote_scales_f32(LlmMlxModel* m) {
    if (!llm_mlx_f32_scales_enabled()) return 0;
    mlx_dtype tgt;
    if (m->cfg.bfloat16_input || g_bf16_all) {
        /* Attivazione a 16-bit: solo f16 + scale allineate a f16 azzera il promote per-token.
         * Con attivazione bf16 (o flavor bf16) si esce: la storia qui e' misurata in A/B. */
        if (g_bf16_all && g_act16 == MLX_BFLOAT16 && llm_mlx_bf16_scales_enabled()) tgt = MLX_BFLOAT16;
        else if (g_bf16_all && g_act16 == MLX_FLOAT16 && llm_mlx_f16_scales_enabled()) tgt = MLX_FLOAT16;
        else return 0;
    } else {
        tgt = MLX_FLOAT32;   /* solo attivazione fp32: allargamento esatto bf16/f16->f32 */
    }
    int promoted = 0;
    for (int s = 0; s < m->weights.count; ++s) {
        mlx_map_string_to_array map = m->weights.shards[s];
        mlx_map_string_to_array_iterator it = mlx_map_string_to_array_iterator_new(map);
        const char* key = NULL;
        for (;;) {
            mlx_array val = mlx_array_new();
            if (mlx_map_string_to_array_iterator_next(&key, &val, it) != 0) { mlx_array_free(val); break; }
            mlx_dtype dt = val.ctx ? mlx_array_dtype(val) : MLX_FLOAT32;
            if (key && (llm_mlx_is_scales_or_bias(key) || llm_mlx_is_norm_weight(key)) && dt != tgt &&
                (dt == MLX_FLOAT16 || dt == MLX_BFLOAT16 ||
                 (tgt == MLX_FLOAT16 && dt == MLX_FLOAT32))) {
                /* insert_or_assign su una CHIAVE ESISTENTE di unordered_map riassegna il solo
                 * value in loco: nessun nuovo nodo, nessun rehash, l'iteratore resta valido. */
                mlx_array cvt = mlx_array_new();
                if (mlx_astype(&cvt, val, tgt, m->stream) == 0 &&
                    mlx_array_eval(cvt) == 0 &&
                    mlx_map_string_to_array_insert(map, key, cvt) == 0)
                    ++promoted;
                mlx_array_free(cvt);
            }
            mlx_array_free(val);
        }
        mlx_map_string_to_array_iterator_free(it);
    }
    if (promoted) fprintf(stderr, "[scales-hoist] promoted %d scale/bias/norm tensors to %s "
                                  "(hoists the per-token AsType out of quantized_matmul/norms)\n",
                          promoted, tgt == MLX_FLOAT16 ? "f16" : (tgt == MLX_BFLOAT16 ? "bf16" : "fp32"));
    return promoted;
}
