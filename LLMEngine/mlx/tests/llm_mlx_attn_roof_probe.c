/* llm_mlx_attn_roof_probe.c — banda dell'attenzione DECODE al crescere di Nk, SENZA modello.
 *
 * Perche' esiste: Bugs/(open) - MLX-Decode-Costo-Contesto-KV-Attention.md. Il gap vs mlx_lm
 * cresce col contesto (pendenza +31%), quindi il costo extra e' PER TOKEN DI CONTESTO.
 * Il dispatch Metal di MLX (mlx/backend/metal/scaled_dot_product_attention.cpp) sceglie il
 * kernel `sdpa_vector_2pass` quando `k.shape(1) < q.shape(1) && k.shape(2) >= 4096`: con GQA
 * (n_kv < n_heads) sopra 4096 si cambia kernel, e il 2-pass rilegge la KV. Questa probe misura
 * il costo reale del path di produzione per vedere DOVE la curva si stacca dal tetto.
 *
 * Chiama il WRAPPER REALE (llm_mlx_gqa_sdpa, window=0 = path denso di qwen3) su 3 layout K/V:
 *   A contig   buffer [1,nkv,Nk,hd] esatto     (riferimento "nessun padding")
 *   B v256     VISTA [1,nkv,Nk,hd] da cap=ceil256(Nk)  <-- layout REALE del decode (step 256)
 *   C vpow2    VISTA da cap=pow2 >= Nk         (caso peggiore di padding)
 * e il tetto di SOLA LETTURA (sum sull'intero K: legge Nk*hd*nkv per element access, scrive 1).
 *
 * Build (non entra in CMake, come llm_mlx_pagesel_test.c; eseguibile da native/llmengine_mlx/):
 *   cc -O2 -Wno-unused-function -I. -I../../libs/mlx-deps/mlx-c-src -I../../libs/mlx-deps/mlx-src \
 *      -Ibuild/_deps/metal_cpp-src/include llm_mlx_attn_roof_probe.c -o /tmp/llm_attn_roof_probe \
 *      /usr/bin/c++ ... (vedi nota in fondo per la riga completa)
 * Uso: /tmp/llm_attn_roof_probe [n_heads] [n_kv] [head_dim] [reps]
 *   LLM_ATTN_PROBE_NK="512 4096 4192 16384" per una lista di Nk personalizzata.
 */
#include "llm_mlx_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int ceil256(int n) { return ((n + 255) / 256) * 256; }
static int pow2_at_least(int n) { int p = 256; while (p < n) p <<= 1; return p; }

/* Costruisce K e V nella geometria [1,nkv,nk,hd] bf16. Se cap>nk restituisce una VISTA
 * (mlx_slice) del buffer da cap righe, come fa llm_mlx_kv_append_inplace a decode.
 * I buffer genitori restano vivi in *parents (2 handle) e vanno liberati dal chiamante. */
static bool build_kv(mlx_array* k, mlx_array* v, int nkv, int nk, int cap, int hd,
                     mlx_array* parents, mlx_stream st) {
    const int bsh[4] = { 1, nkv, cap, hd };
    parents[0] = mlx_array_new();
    parents[1] = mlx_array_new();
    if (mlx_zeros(&parents[0], bsh, 4, MLX_BFLOAT16, st) != 0) return false;
    if (mlx_zeros(&parents[1], bsh, 4, MLX_BFLOAT16, st) != 0) return false;
    /* eval: una slice su un nodo `zeros` non valutato resta nel grafo; a decode la cache e'
     * gia' valutata (viene scritta in-place), quindi eval qui reproduce la condizione reale. */
    if (mlx_array_eval(parents[0]) != 0 || mlx_array_eval(parents[1]) != 0) return false;
    if (cap == nk) {
        if (mlx_array_set(k, parents[0]) != 0) return false;
        if (mlx_array_set(v, parents[1]) != 0) return false;
        /* parents non possiedono piu' nulla: mlx_array_set trasferisce, li rimettiamo vuoti */
        parents[0] = mlx_array_new();
        parents[1] = mlx_array_new();
        return true;
    }
    const int p0[4] = { 0, 0, 0, 0 }, ps[4] = { 1, 1, 1, 1 }, shp[4] = { 1, nkv, nk, hd };
    if (mlx_slice(k, parents[0], p0, 4, shp, 4, ps, 4, st) != 0) return false;
    if (mlx_slice(v, parents[1], p0, 4, shp, 4, ps, 4, st) != 0) return false;
    return true;
}

/* Tempo medio (ms) di reps chiamate. DUE modalità di timing:
 *   WALL    : una eval per chiamata -> include il costo di SUBMIT CPU (pavimento ~0,15 ms,
 *             che a Nk piccoli DOMINA e produce bande impossibili: non leggerla come GPU).
 *   BATCHED : accoda `reps` nodi SDPA indipendenti e fa UNA sola mlx_eval -> il submit si
 *             amortizza, il tempo residuo e' l'esecuzione seriale dei kernel sullo stream
 *             (≈ GPU pura). E' la colonna da usare per la curva banda-vs-Nk.
 * llm_mlx_gqa_sdpa ritorna TRUE se ok (e' bool): il self-test che lo leggeva "!= 0 = errore"
 * era il perche' del falso FAIL registrato nel bug. */
static double bench_sdpa_mode(mlx_array q, mlx_array k, mlx_array v, int hd, int reps,
                              mlx_stream st, bool raw, bool* ok) {
    const float scale = 1.0f / sqrtf((float)hd);
    if (!raw) { /* warmup del wrapper (costruisce/riusa la mask cache se window>0) */
        mlx_array o = mlx_array_new();
        for (int w = 0; w < 2 && *ok; w++) {
            if (!llm_mlx_gqa_sdpa(&o, q, k, v, scale, 0, st)) *ok = false;
            else if (mlx_array_eval(o) != 0) *ok = false;
        }
        mlx_array_free(o);
    }
    if (!*ok) return 0.0;
    mlx_array no_mask = mlx_array_new(), no_sinks = mlx_array_new();
    mlx_array* outs = (mlx_array*)calloc((size_t)reps, sizeof(mlx_array));
    mlx_vector_array vec = mlx_vector_array_new();
    double ms = 0.0;
    if (outs != NULL && vec.ctx != NULL) {
        for (int r = 0; r < reps && *ok; r++) {
            outs[r] = mlx_array_new();
            if (raw) {
                if (mlx_fast_scaled_dot_product_attention(&outs[r], q, k, v, scale, "",
                                                          no_mask, no_sinks, st) != 0) *ok = false;
            } else if (!llm_mlx_gqa_sdpa(&outs[r], q, k, v, scale, 0, st)) {
                *ok = false;
            }
            if (*ok) mlx_vector_array_append_value(vec, outs[r]);
        }
        if (*ok) {
            const double t0 = now_s();
            if (mlx_eval(vec) != 0) *ok = false;
            ms = (now_s() - t0) * 1e3 / (double)reps;
        }
    } else { *ok = false; }
    for (int r = 0; r < reps; r++) if (outs != NULL && outs[r].ctx != NULL) mlx_array_free(outs[r]);
    free(outs);
    mlx_vector_array_free(vec);
    mlx_array_free(no_mask); mlx_array_free(no_sinks);
    return ms;
}

static double bench_sdpa(mlx_array q, mlx_array k, mlx_array v, int hd, int reps,
                         mlx_stream st, bool* ok) {
    mlx_array o = mlx_array_new();
    const float scale = 1.0f / sqrtf((float)hd);
    for (int w = 0; w < 3; w++) {
        if (!llm_mlx_gqa_sdpa(&o, q, k, v, scale, 0, st)) { *ok = false; break; }
        if (mlx_array_eval(o) != 0) { *ok = false; break; }
    }
    double ms = 0.0;
    if (*ok) {
        const double t0 = now_s();
        for (int r = 0; r < reps; r++) {
            if (!llm_mlx_gqa_sdpa(&o, q, k, v, scale, 0, st)) { *ok = false; break; }
            if (mlx_array_eval(o) != 0) { *ok = false; break; }
        }
        ms = (now_s() - t0) * 1e3 / (double)reps;
    }
    mlx_array_free(o);
    return ms;
}

/* Tetto: somma dell'intero K = lettura pura di nkv*nk*hd elementi bf16 (batched: submit ammortizzato). */
static double bench_readroof(mlx_array k, int reps, mlx_stream st, bool* ok) {
    mlx_array* outs = (mlx_array*)calloc((size_t)reps, sizeof(mlx_array));
    mlx_vector_array vec = mlx_vector_array_new();
    double ms = 0.0;
    if (outs != NULL && vec.ctx != NULL) {
        for (int r = 0; r < reps && *ok; r++) {
            outs[r] = mlx_array_new();
            if (mlx_sum(&outs[r], k, false, st) != 0) *ok = false;
            else mlx_vector_array_append_value(vec, outs[r]);
        }
        if (*ok) {
            const double t0 = now_s();
            if (mlx_eval(vec) != 0) *ok = false;
            ms = (now_s() - t0) * 1e3 / (double)reps;
        }
    } else { *ok = false; }
    for (int r = 0; r < reps; r++) if (outs != NULL && outs[r].ctx != NULL) mlx_array_free(outs[r]);
    free(outs);
    mlx_vector_array_free(vec);
    return ms;
}

/* ── MODE C: working set FRESCO (il fix del difetto mortale del bench warm) ───────
 * Perche' esiste: le colonne G/R/B/C di questo file rileggono SEMPRE LO STESSO buffer
 * `reps` volte. A Nk=16100 una coppia K+V di una layer e' 67 MB: 40 letture dello stesso
 * buffer restano nella L2 del GPU (infatti il "tetto" usciva 807 GB/s, SOPRA il tetto reale
 * di DRAM ~620-747). Un passo di decode production invece streama `layers` cache DISTINTE
 * (36 x 67 MB = 2,4 GB a 16k): la L2 non c'entra niente. Col warm si misura la L2 e si
 * "dimostra" che il kernel e' al tetto: era l'artefatto che mi ha fatto escludere
 * l'attenzione per il bug (open) - MLX-Decode-Costo-Contesto-KV-Attention.md.
 *
 * Qui il working set e' quello VERO: `nbufs` coppie indipendenti (una per layer), una
 * chiamata SDPA per coppia, UNA eval per giro. `roof_cold` e' la lettura di solo K sulle
 * STESSE nbufs coppie (stesso cold), quindi il rapporto G_cold/roof_cold non ha termine
 * di confronto sbilanciato.
 * Memoria: nbufs x 2 x Nk x nkv x hd x 2 B  (a Nk=32768, nbufs=36: ~4,8 GB). */
static int bench_cold(int nkv, int nh, int hd, int nk, int nbufs, int reps,
                      mlx_stream st, double* ms_call, double* roof_ms_call, double* peak_gb) {
    int rc = 0;
    const int bsh[4] = { 1, nkv, nk, hd };
    mlx_array* kk = (mlx_array*)calloc((size_t)nbufs, sizeof(mlx_array));
    mlx_array* vv = (mlx_array*)calloc((size_t)nbufs, sizeof(mlx_array));
    mlx_array q = mlx_array_new();
    const int qsh[4] = { 1, nh, 1, hd };
    if (kk == NULL || vv == NULL) { rc = -1; }
    if (rc == 0 && (mlx_zeros(&q, qsh, 4, MLX_BFLOAT16, st) != 0 || mlx_array_eval(q) != 0)) rc = -2;
    /* Alloca e VALUTA le nbufs coppie: devono essere buffer reali distinti, non nodi del grafo. */
    for (int b = 0; b < nbufs && rc == 0; b++) {
        kk[b] = mlx_array_new(); vv[b] = mlx_array_new();
        if (mlx_zeros(&kk[b], bsh, 4, MLX_BFLOAT16, st) != 0 ||
            mlx_zeros(&vv[b], bsh, 4, MLX_BFLOAT16, st) != 0) rc = -2;
        else if (mlx_array_eval(kk[b]) != 0 || mlx_array_eval(vv[b]) != 0) rc = -2;
    }
    const float scale = 1.0f / sqrtf((float)hd);
    mlx_array no_mask = mlx_array_new(), no_sinks = mlx_array_new();
    const long n_calls = (long)reps * (long)nbufs;
    for (int arm = 0; arm < 2 && rc == 0; arm++) {
        /* arm 0 = SDPA (wrapper di produzione), arm 1 = tetto di lettura (sum sul solo K) */
        mlx_vector_array vec = mlx_vector_array_new();
        mlx_array* outs = (mlx_array*)calloc((size_t)n_calls, sizeof(mlx_array));
        if (vec.ctx == NULL || outs == NULL) { rc = -1; }
        if (rc == 0 && arm == 0) {  /* warmup del wrapper su una coppia */
            mlx_array o = mlx_array_new();
            if (!llm_mlx_gqa_sdpa(&o, q, kk[0], vv[0], scale, 0, st) || mlx_array_eval(o) != 0) rc = -3;
            mlx_array_free(o);
        }
        long i = 0;
        const double t0 = now_s();
        for (int r = 0; r < reps && rc == 0; r++) {
            for (int b = 0; b < nbufs && rc == 0; b++, i++) {
                outs[i] = mlx_array_new();
                if (arm == 0) {
                    if (!llm_mlx_gqa_sdpa(&outs[i], q, kk[b], vv[b], scale, 0, st)) rc = -3;
                } else {
                    if (mlx_sum(&outs[i], kk[b], false, st) != 0) rc = -4;
                }
                if (rc == 0) mlx_vector_array_append_value(vec, outs[i]);
            }
        }
        if (rc == 0 && mlx_eval(vec) != 0) rc = -5;
        const double ms = (now_s() - t0) * 1e3 / (double)n_calls;
        if (rc == 0) { if (arm == 0) *ms_call = ms; else *roof_ms_call = ms; }
        for (long j = 0; j < n_calls; j++) if (outs != NULL && outs[j].ctx) mlx_array_free(outs[j]);
        free(outs);
        mlx_vector_array_free(vec);
    }
    mlx_array_free(no_mask); mlx_array_free(no_sinks);
    if (peak_gb != NULL) {
        size_t pk = 0, act = 0;
        if (mlx_get_peak_memory(&pk) == 0 && mlx_get_active_memory(&act) == 0)
            *peak_gb = ((double)pk - (double)act) / 1e9;
    }
    for (int b = 0; b < nbufs; b++) { if (kk[b].ctx) mlx_array_free(kk[b]); if (vv[b].ctx) mlx_array_free(vv[b]); }
    free(kk); free(vv);
    mlx_array_free(q);
    return rc;
}

/* ── MODE F: INTERFERENZA fra il flusso KV e il flusso PESI (l'ultimo residuo) ──────
 * Perche' esiste: dopo MODE C (working set freddo) e MODE D/E (copia SliceUpdate,
 * refutata), il conto non torna ancora. A Nk=16100 il kernel SDPA da solo, sulle 36 cache
 * DISTINCTE come in produzione, costa 5,05 ms/token; il passo di decode intero ne costa
 * ~18,9 (e ~10,7 a ctx 512). La differenza ctx-proporzionale (~4,4 ms) NON e' nella KV e
 * NON e' nella copia dell'append: resta il flusso dei PESI (Qwen3-8B-4bit: ~3 GB/token,
 * ~5,4 ms al tetto di 558 GB/s) che nella lettura e' SERIALE con la KV sullo stesso
 * stream. Se le due letture si sommano, il costo atteso e' (KV + pesi); se INTERFERISCONO
 * (thrash della gerarchia di cache / perdita di coalescing sulle code lunghe), la stessa
 * mole di byte costa PIU' della somma delle due esecuzioni isolate.
 *
 * TRE braccia, stesso working set freddo, stessa geometria di submissione (un grafo, UNA
 * eval per "token"):
 *   A = 36 chiamate SDPA (solo KV)                       -> ms_sdpa
 *   B = 36 letture di "peso" (mlx_sum, sola lettura)    -> ms_w
 *   C = le DUE accodate per layer nello STESSO ordine del forward (w, sdpa, w, sdpa...) -> ms_both
 * Superlinearita' = ms_both / (ms_sdpa + ms_w): 1,00 = additivo (nessuna interferenza),
 * >1 = interferenza quantizzata in ms/token. `n_wbufs` buffer di peso DISTINTI ruotati
 * (b % n_wbufs) per non misurare la cache di secondo livello: con wmb=82 MB e n_wbufs=4
 * il set dei pesi e' 328 MB, ben oltre la L2, e la lettura e' DRAM-bound come in produzione.
 * MEMORIA: nbufs x 2 x Nk x nkv x hd x 2 B (KV) + n_wbufs x wmb MB  (a Nk=16100, nbufs=36:
 * ~4,8 GB + 0,33 GB).  NON tocca il modello: nessun caricamento di checkpoint. */
static int bench_interf(int nkv, int nh, int hd, int nk, int nbufs, int wmb, int n_wbufs,
                        int reps, mlx_stream st,
                        double* ms_sdpa, double* ms_w, double* ms_both) {
    int rc = 0;
    const int bsh[4] = { 1, nkv, nk, hd };
    const int qsh[4] = { 1, nh, 1, hd };
    const long w_elems = (long)wmb * 1000000L / 2;      /* bf16 */
    const int wsh[4] = { 1, 1, (int)w_elems, 1 };
    mlx_array* kk = (mlx_array*)calloc((size_t)nbufs, sizeof(mlx_array));
    mlx_array* vv = (mlx_array*)calloc((size_t)nbufs, sizeof(mlx_array));
    mlx_array* ww = (mlx_array*)calloc((size_t)n_wbufs, sizeof(mlx_array));
    mlx_array q = mlx_array_new();
    if (!kk || !vv || !ww) rc = -1;
    if (rc == 0 && (mlx_zeros(&q, qsh, 4, MLX_BFLOAT16, st) != 0 || mlx_array_eval(q) != 0)) rc = -2;
    for (int b = 0; b < nbufs && rc == 0; b++) {
        kk[b] = mlx_array_new(); vv[b] = mlx_array_new();
        if (mlx_zeros(&kk[b], bsh, 4, MLX_BFLOAT16, st) != 0 ||
            mlx_zeros(&vv[b], bsh, 4, MLX_BFLOAT16, st) != 0) rc = -2;
        else if (mlx_array_eval(kk[b]) != 0 || mlx_array_eval(vv[b]) != 0) rc = -2;
    }
    for (int b = 0; b < n_wbufs && rc == 0; b++) {
        ww[b] = mlx_array_new();
        if (mlx_zeros(&ww[b], wsh, 4, MLX_BFLOAT16, st) != 0) rc = -2;
        else if (mlx_array_eval(ww[b]) != 0) rc = -2;
    }
    const float scale = 1.0f / sqrtf((float)hd);
    const long n_calls = (long)reps * (long)nbufs;
    for (int arm = 0; arm < 3 && rc == 0; arm++) {
        mlx_vector_array vec = mlx_vector_array_new();
        mlx_array* outs = (mlx_array*)calloc((size_t)n_calls * 2, sizeof(mlx_array));
        if (vec.ctx == NULL || outs == NULL) { rc = -1; }
        if (rc == 0 && arm == 0) {           /* warmup del wrapper su una coppia */
            mlx_array o = mlx_array_new();
            if (!llm_mlx_gqa_sdpa(&o, q, kk[0], vv[0], scale, 0, st) || mlx_array_eval(o) != 0) rc = -3;
            mlx_array_free(o);
        }
        long i = 0;
        const double t0 = now_s();
        for (int r = 0; r < reps && rc == 0; r++) {
            for (int b = 0; b < nbufs && rc == 0; b++, i++) {
                if (arm != 0) {              /* ramo peso SEMPRE prima: ordine del forward */
                    outs[i] = mlx_array_new();
                    if (mlx_sum(&outs[i], ww[b % n_wbufs], false, st) != 0) rc = -4;
                    else mlx_vector_array_append_value(vec, outs[i]);
                    i++;
                }
                if (arm != 1) {
                    outs[i] = mlx_array_new();
                    if (!llm_mlx_gqa_sdpa(&outs[i], q, kk[b], vv[b], scale, 0, st)) rc = -3;
                    else mlx_vector_array_append_value(vec, outs[i]);
                }
            }
        }
        if (rc == 0 && mlx_eval(vec) != 0) rc = -5;
        const double ms = (now_s() - t0) * 1e3 / (double)n_calls;
        if (rc == 0) {
            double* dst = (arm == 0) ? ms_sdpa : (arm == 1) ? ms_w : ms_both;
            *dst = ms;
        }
        for (long j = 0; j < n_calls * 2; j++) if (outs != NULL && outs[j].ctx) mlx_array_free(outs[j]);
        free(outs);
        mlx_vector_array_free(vec);
    }
    for (int b = 0; b < nbufs; b++) { if (kk[b].ctx) mlx_array_free(kk[b]); if (vv[b].ctx) mlx_array_free(vv[b]); }
    for (int b = 0; b < n_wbufs; b++) if (ww[b].ctx) mlx_array_free(ww[b]);
    free(kk); free(vv); free(ww);
    mlx_array_free(q);
    return rc;
}

/* ── MODE P: il PATTERN REALE del decode, non solo il kernel ─────────────────────
 * La colonna G misura il kernel SDPA da solo. Ma a decode la produzione, per ogni layer e
 * ogni token, fa: append IN-PLACE della riga KV (mlx_slice_update_inplace) + Vista (mlx_slice)
 * + SDPA. Se l'append ha un costo che cresce col contesto (copia/lettura del parent), la
 * pendenza misurata in produzione (3,45e-4 ms per token-di-contesto) supera quella del solo
 * kernel. Questo modo separa i due.
 * NB (2026-09-17, autocorrezione): il commento qui sotto sosteneva che il kernel del probe
 * valeva ~2,7e-4 "gia' pari a mlx_lm" -> **FALSO**: era il warm su un solo buffer (vedi MODE C
 * sopra), quindi misurava la L2. Il censimento LLM_DISPATCH sulla produzione dice che il costo
 * per token di contesto e' ~2x il tetto di una sola lettura della KV.
 *
 * Costruisce L cache [1,nkv,cap,hd] (cap=ceil256(Nk), come step=256 del motore), poi fa S
 * STEP: ogni step = per ogni layer (append in-place 1 token + view + SDPA) e UNA sola eval,
 * esattamente come un forward di decode (un grafo, una eval per token).
 * Memoria: L x 2 x Nk x nkv x hd x 2 B  (a Nk=32768, L=4: ~1,07 GB).
 *
 * ── MODE D = MODE P + `hold_refs` (LA LEVA CHE CERCAVO) ──────────────────────────
 * `mlx_slice_update_inplace` -> `mlx::core::slice_update` -> primitiva `SliceUpdate`, il cui
 * `eval_gpu` (mlx/backend/metal/indexing.cpp:731) fa' PRIMA `copy_gpu(in, out, Vector)` e SOLO
 * DOPO scrive la riga nuova. `copy_gpu` evita la copia SOLO se il buffer dell'input e'
 * DONABILE (`set_copy_output_data` -> `is_donatable(in,out)` -> `array::is_donatable()`:
 * `array_desc_.use_count()==1 && data.use_count()==1`, mlx/array.h:294). Una VISTA ancora viva
 * della cache condivide il `Data` della cache -> `use_count>1` -> DONAZIONE NEGATA ->
 * `out.set_data(malloc(data_size))` + **copia dell'INTERA cache** (lettura + scrittura) per
 * (layer,token). E' un costo che NON compare nel grafo (nessun nodo in piu': il censimento
 * LLM_DISPATCH conta 9521 nodi identici a 512 e a 16k) e che il self-test C3 NON poteva vedere:
 * C3 misurava la CRESCITA di memoria, e l'allocatore riusa lo stesso blocco -> crescita 0,00 MB
 * anche copiando tutto a ogni passo.
 * Nel decode reale il passo k+1 viene buildato MENTRE gli output del passo k sono ancora vivi
 * (pipeline asincrona: le viste k/v sono `inputs` della primitiva SDPA del passo k), quindi la
 * condizione "non donabile" e' la NORMA, non l'eccezione. `hold_refs=1` riproduce esattamente
 * quello: le viste+output del passo precedente restano vivi attraverso gli append del passo
 * successivo. `hold_refs=0` (il comportamento storico del probe) li libera a fine passo -> cache
 * donabile -> zero copia: per questo il delta misurato era ~0,012 ms/(layer token) invece del
 * ~0,115 di una copia piena.
 * Conferma diretta e DETERMINISTICA (senza modello): l'API C patchata stampa gia'
 * `[SU] donatable=%d` se LLM_SLICEUP_DBG=1 -> con count = 0 in hold_refs=1, >0 in hold_refs=0.
 *
 * ── `concat_origin` (MODE E): la cache costruita COME IN PRODUZIONE ───────────────
 * Nel prefill, quando l'append in-place non e' disponibile, la cache NASCE DA UNA CONCATENATE
 * (llmengine_mlx.c:4172 `mlx_concatenate_axis`). `concatenate_gpu`
 * (mlx/backend/metal/slicing.cpp:13) imposta ESPLICITAMENTE
 * `row_contiguous=col_contiguous=contiguous=false` sull'output. E `SliceUpdate::eval_gpu`
 * calcola `ctype = in.flags().contiguous && in.size()==in.data_size() ? Vector : General`:
 * con ctype != Vector, `set_copy_output_data` NON passa nemmeno dal test di donazione ->
 * **copia sempre l'intera cache**, a prescindere da `is_donatable`. Peggio: il ramo non-Vector
 * fa `out.set_data(malloc, in.data_size(), in.strides(), in.flags())`, cioe' EREDITA i flags ->
 * lo stato non-contiguo e' PERMANENTE per tutta la sessione.
 * COROLLO (importante, e' il mio stesso errore di lettura): il censimento `[SU] donatable=1`
 * sull'engine reale (194/216 a ctx 16384) NON prova che non si copia -- misura il predicato
 * SBAGLIATO. `concat_origin=1` misura il ramo reale; `is_contiguous`/`is_row_contiguous`
 * (`_mlx_array_is_contiguous`, exported da mlx/c/array.cpp) danno il verdetto DETERMINISTICO. */
static int bench_pattern(int nkv, int nh, int hd, int nk, int steps, int layers, int do_append,
                         int hold_refs, int concat_origin, mlx_stream st,
                         double* out_ms, double* peak_mb) {
    const int cap = ceil256(nk);
    if (cap < nk) return -1;
    const int base = cap - steps;             /* si scrive dalla riga `base` in poi */
    mlx_array* kc = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
    mlx_array* vc = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
    if (!kc || !vc) { free(kc); free(vc); return -1; }
    const int bsh[4] = { 1, nkv, cap, hd }, nsh[4] = { 1, nkv, 1, hd }, qsh[4] = { 1, nh, 1, hd };
    mlx_array kn = mlx_array_new(), vn = mlx_array_new(), q = mlx_array_new();
    int rc = 0;
    /* kn/vn/q PRIMA delle cache: nel ramo concat_origin la riga nuova entra nella CONCATENATE
     * che costruisce la cache, quindi deve gia' esistere. */
    if (mlx_zeros(&kn, nsh, 4, MLX_BFLOAT16, st) != 0 ||
        mlx_zeros(&vn, nsh, 4, MLX_BFLOAT16, st) != 0 ||
        mlx_zeros(&q,  qsh, 4, MLX_BFLOAT16, st) != 0 ||
        mlx_array_eval(kn) != 0 || mlx_array_eval(vn) != 0 || mlx_array_eval(q) != 0) rc = -1;
    for (int l = 0; l < layers && rc == 0; l++) {
        kc[l] = mlx_array_new(); vc[l] = mlx_array_new();
        if (concat_origin) {
            /* come nel prefill reale: cache piu' corta + CONCATENATE della riga nuova -> cap */
            const int psh[4] = { 1, nkv, cap - 1, hd };
            mlx_array kp = mlx_array_new(), vp = mlx_array_new();
            mlx_vector_array kv = mlx_vector_array_new(), vv = mlx_vector_array_new();
            if (mlx_zeros(&kp, psh, 4, MLX_BFLOAT16, st) != 0 ||
                mlx_zeros(&vp, psh, 4, MLX_BFLOAT16, st) != 0 ||
                mlx_array_eval(kp) != 0 || mlx_array_eval(vp) != 0) rc = -1;
            if (rc == 0) {
                mlx_vector_array_append_value(kv, kp);
                mlx_vector_array_append_value(kv, kn);
                mlx_vector_array_append_value(vv, vp);
                mlx_vector_array_append_value(vv, vn);
                if (mlx_concatenate_axis(&kc[l], kv, 2, st) != 0 ||
                    mlx_concatenate_axis(&vc[l], vv, 2, st) != 0) rc = -1;
                if (rc == 0 && (mlx_array_eval(kc[l]) != 0 || mlx_array_eval(vc[l]) != 0)) rc = -1;
                if (l == 0 && getenv("LLM_ATTN_PROBE_CONTIG") != NULL) {
                    bool zc = false, zr = false, cc = false, cr = false;
                    _mlx_array_is_contiguous(&zc, kp); _mlx_array_is_row_contiguous(&zr, kp);
                    _mlx_array_is_contiguous(&cc, kc[l]); _mlx_array_is_row_contiguous(&cr, kc[l]);
                    /* NB: 9 specificatori per 9 argomenti. La prima stesura ne aveva 8 e i
                     * valori slittavano (leggevo "cap 8" = nkv e "contiguous=16128"). */
                    printf("# CONTIG zeros[1,%d,%d,%d] -> contiguous=%d row_contiguous=%d   "
                           "CONCAT[1,%d,%d,%d] -> contiguous=%d row_contiguous=%d\n",
                           nkv, cap - 1, hd, (int)zc, (int)zr, nkv, cap, hd, (int)cc, (int)cr);
                    fflush(stdout);
                }
            }
            mlx_vector_array_free(kv); mlx_vector_array_free(vv);
            mlx_array_free(kp); mlx_array_free(vp);
            if (rc != 0) break;
            continue;
        }
        if (mlx_zeros(&kc[l], bsh, 4, MLX_BFLOAT16, st) != 0 ||
            mlx_zeros(&vc[l], bsh, 4, MLX_BFLOAT16, st) != 0 ||
            mlx_array_eval(kc[l]) != 0 || mlx_array_eval(vc[l]) != 0) rc = -1;
    }
    const float scale = 1.0f / sqrtf((float)hd);
    /* A/B SULLE STEDESSE CACHE: `do_append`=1 esegue anche la scrittura in-place della riga
     * nuova (path reale); =0 salta SOLO quella scrittura e costruisce LA STESSA VISTA, con lo
     * STESSO `need` di righe. Il DELTA fra le due passate e' quindi SOLO il costo di
     * `mlx_slice_update_inplace` per (layer,token).
     * CORREZIONE (2026-09-17): la prima stesura tagliava le viste di controllo a `cap` righe
     * mentre il ramo append ne passava `need` alla SDPA -> le due braccia NON leggevano la
     * stessa mole di dati e il delta usciva NEGATIVO a 512/4096 (misurato). Ora `need` e'
     * la stessa espressione in entrambi i rami. */
    mlx_array* hk = NULL; mlx_array* hv = NULL; mlx_array* ho = NULL;
    if (rc == 0 && hold_refs) {
        hk = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
        hv = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
        ho = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
        if (!hk || !hv || !ho) rc = -1;
    }
    for (int pass = 0; pass < 2 && rc == 0; pass++) {   /* pass 0 = warmup non cronometrato */
        const int n = (pass == 0) ? 2 : steps;
        int have_held = 0;
        if (pass == 1) { mlx_reset_peak_memory(); if (peak_mb) *peak_mb = -1.0; }
        const double t0 = now_s();
        for (int s = 0; s < n && rc == 0; s++) {
            mlx_array* kv = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
            mlx_array* vv = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
            mlx_array* oo = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
            mlx_vector_array vec = mlx_vector_array_new();
            if (!kv || !vv || !oo) { rc = -1; }
            for (int l = 0; l < layers && rc == 0; l++) {
                kv[l] = mlx_array_new(); vv[l] = mlx_array_new(); oo[l] = mlx_array_new();
                if (do_append) {
                    if (!llm_mlx_kv_append_inplace(&kc[l], &vc[l], kn, vn, base + s, 1, 256,
                                                   &kv[l], &vv[l], st)) { rc = -2; break; }
                } else {
                    /* stessa vista, stesso `need` (need = off + n_tokens del ramo append) */
                    const int need = base + s + 1;
                    const int p0[4] = { 0, 0, 0, 0 }, ps[4] = { 1, 1, 1, 1 };
                    const int pk[4] = { 1, nkv, need, hd };
                    if (mlx_slice(&kv[l], kc[l], p0, 4, pk, 4, ps, 4, st) != 0 ||
                        mlx_slice(&vv[l], vc[l], p0, 4, pk, 4, ps, 4, st) != 0) { rc = -5; break; }
                }
                if (!llm_mlx_gqa_sdpa(&oo[l], q, kv[l], vv[l], scale, 0, st)) { rc = -4; break; }
                mlx_vector_array_append_value(vec, oo[l]);
            }
            if (rc == 0 && mlx_eval(vec) != 0) rc = -3;
            for (int l = 0; l < layers; l++) {
                if (hold_refs) {
                    /* il passo k+1 si builda con il passo k ANCORA VIVO: trasferisci viste e
                     * output nel `held` e libera il held vecchio SOLO qui, cioè DOPO gli
                     * append del passo corrente. `mlx_array` e' `{void* ctx}`: copiare il
                     * struct e azzerare `.ctx` trasferisce la proprieta' senza doppio free. */
                    if (have_held) {
                        if (hk[l].ctx) mlx_array_free(hk[l]);
                        if (hv[l].ctx) mlx_array_free(hv[l]);
                        if (ho[l].ctx) mlx_array_free(ho[l]);
                    }
                    hk[l] = kv[l]; hv[l] = vv[l]; ho[l] = oo[l];
                    kv[l].ctx = NULL; vv[l].ctx = NULL; oo[l].ctx = NULL;
                } else {
                    if (kv && kv[l].ctx) mlx_array_free(kv[l]);
                    if (vv && vv[l].ctx) mlx_array_free(vv[l]);
                    if (oo && oo[l].ctx) mlx_array_free(oo[l]);
                }
            }
            have_held = 1;
            free(kv); free(vv); free(oo);
            mlx_vector_array_free(vec);
        }
        for (int l = 0; hold_refs && l < layers; l++) {
            if (ho[l].ctx) mlx_array_free(ho[l]);
            if (hk[l].ctx) mlx_array_free(hk[l]);
            if (hv[l].ctx) mlx_array_free(hv[l]);
        }
        if (pass == 1 && rc == 0) {
            *out_ms = (now_s() - t0) * 1e3 / ((double)n * (double)layers);
            if (peak_mb != NULL) {
                size_t pk = 0, act = 0;
                if (mlx_get_peak_memory(&pk) == 0 && mlx_get_active_memory(&act) == 0)
                    *peak_mb = ((double)pk - (double)act) / 1e6;  /* cio' che il passaggio ha
                                                                    * materializzato in piu'
                                                                    * del residente */
            }
        }
    }
    for (int l = 0; l < layers; l++) { if (kc[l].ctx) mlx_array_free(kc[l]); if (vc[l].ctx) mlx_array_free(vc[l]); }
    free(kc); free(vc);
    free(hk); free(hv); free(ho);   /* gli array trattenuti sono gia' liberati a fine pass */
    mlx_array_free(kn); mlx_array_free(vn); mlx_array_free(q);
    return rc;
}

/* ── MODE H: append "RAW" nel buffer ESISTENTE via kernel Metal custom ─────────────
 * Domanda a cui risponde (Bugs/(open) - MLX-Decode-Costo-Contesto-KV-Attention.md §8.13(v)):
 * si puo' scrivere la riga KV nuova DIRETTAMENTE nel buffer della cache, senza passare da
 * `slice_update` (copia O(cap) quando il `Data` non e' donabile) ne' da `concatenate`
 * (copia sempre)?
 * Vincolo letto al sorgente: la firma dei kernel custom la GENERA MLX e per ogni input emette
 * `const device T*` (mlx/backend/common/metal_kernel.cpp:103-107); le output sono `device` ma
 * sono SEMPRE allocazioni nuove (backend/metal/custom_kernel.cpp:37). Senza toccare MLX resta
 * SOLO `const_cast<device InT*>(cache)` nel CORPO: il binding e' un normale MTLBuffer e la
 * `const` e' solo un qualificador MSL — SE il runtime lo lascia fare, il costo dell'append
 * diventa O(nkv*hd) (~2 KB) per (layer,token), INDIPENDENTE da `is_donatable`, e l'attenzione
 * resta UNA sola lettura di K+V (nessun merge, nessun pezzo).
 * Bracci (A/B nello STESSO processo, stesse cache — robusto alla contesa):
 *   arm 0 = nessuna scrittura (vista + SDPA)                         -> baseline
 *   arm 1 = mlx_slice_update_inplace + vista + SDPA con le viste del  -> regime di COPIA (MODE D)
 *           passo k vive nel passo k+1 (hold_refs=1)
 *   arm 2 = raw write + vista + SDPA, hold_refs come sopra
 * + VERIFICA DETERMINISTICA (niente wall-time): scrivi 1.0 alla riga `off`, fai `sum` della
 *   riga e leggi lo scalare (l'eval sincronizza): se il write finisse in una copia, la somma
 *   resterebbe 0. Controllo anche la riga `off+1` (deve restare 0: acchiappa stride sbagliati). */
static const char* kRawHeader =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n";

static const char* kRawSrc =
    "  const int d   = thread_position_in_grid.x;\n"
    "  const int h   = thread_position_in_grid.y;\n"
    "  const int nkv = kcache_shape[1];\n"
    "  const int cap = kcache_shape[2];\n"
    "  const int hd  = kcache_shape[3];\n"
    "  if (d >= hd || h >= nkv || row < 0 || row >= cap) return;\n"
    "  const int64_t hs  = kcache_strides[1];\n"
    "  const int64_t rs  = kcache_strides[2];\n"
    "  const int64_t nhs = knew_strides[1];\n"
    "  device InT* dk = const_cast<device InT*>(kcache);\n"
    "  device InT* dv = const_cast<device InT*>(vcache);\n"
    "  const size_t dst = (size_t)row * (size_t)rs + (size_t)h * (size_t)hs + (size_t)d;\n"
    "  const size_t src = (size_t)h * (size_t)nhs + (size_t)d;\n"
    "  dk[dst] = knew[src];\n"
    "  dv[dst] = vnew[src];\n"
    "  if (h == 0 && d == 0) dummy[0] = (float)cap;\n";

static char g_raw_err[512] = { 0 };
static void g_raw_err_cb(const char* msg, void* data) {
    (void)data;
    if (msg != NULL) { snprintf(g_raw_err, sizeof g_raw_err, "%s", msg); }
}

typedef struct { mlx_fast_metal_kernel k; bool built; int rc; } llm_raw_kernel;
static llm_raw_kernel g_raw = { { NULL }, false, 0 };

static int raw_kernel_build(void) {
    if (g_raw.built) return g_raw.rc;
    const char* in_names[]  = { "kcache", "vcache", "knew", "vnew", "row" };
    const char* out_names[] = { "dummy" };
    mlx_vector_string iv = mlx_vector_string_new();
    for (int i = 0; i < 5; i++) mlx_vector_string_append_value(iv, in_names[i]);
    mlx_vector_string ov = mlx_vector_string_new();
    mlx_vector_string_append_value(ov, out_names[0]);
    g_raw_err[0] = '\0';
    mlx_set_error_handler(g_raw_err_cb, NULL, NULL);
    g_raw.k = mlx_fast_metal_kernel_new("llm_kv_raw_row", iv, ov, kRawSrc, kRawHeader,
                                        /*ensure_row_contiguous=*/false, /*atomic_outputs=*/false);
    mlx_vector_string_free(iv); mlx_vector_string_free(ov);
    g_raw.built = true;
    g_raw.rc = (g_raw.k.ctx == NULL) ? -1 : 0;
    return g_raw.rc;
}

/* Scrive la riga `row` di kcache/vcache prendendo i valori da knew/vnew [1,nkv,1,hd].
 * *dummy (opzionale) riceve l'array-output da 1 float, che il chiamante libera. */
static bool raw_row_write(mlx_array kcache, mlx_array vcache, mlx_array knew, mlx_array vnew,
                          int row, mlx_array* dummy, mlx_stream st) {
    if (raw_kernel_build() != 0) return false;
    const int* ksh = mlx_array_shape(kcache);
    if (ksh == NULL || mlx_array_ndim(kcache) != 4) return false;
    const int nkv = ksh[1], hd = ksh[3];
    int32_t r32 = (int32_t)row;
    mlx_array rowarr = mlx_array_new_data(&r32, NULL, 0, MLX_INT32);
    if (rowarr.ctx == NULL) return false;
    mlx_vector_array in = mlx_vector_array_new();
    mlx_vector_array_append_value(in, kcache);
    mlx_vector_array_append_value(in, vcache);
    mlx_vector_array_append_value(in, knew);
    mlx_vector_array_append_value(in, vnew);
    mlx_vector_array_append_value(in, rowarr);
    const int osh[1] = { 1 };
    mlx_fast_metal_kernel_config cfg = mlx_fast_metal_kernel_config_new();
    mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", mlx_array_dtype(kcache));
    mlx_fast_metal_kernel_config_add_output_arg(cfg, osh, 1, MLX_FLOAT32);
    mlx_fast_metal_kernel_config_set_thread_group(cfg, hd < 32 ? hd : 32, nkv < 32 ? nkv : 32, 1);
    mlx_fast_metal_kernel_config_set_grid(cfg, hd, nkv, 1);
    if (getenv("LLM_ATTN_PROBE_RAW_VERBOSE") != NULL)
        mlx_fast_metal_kernel_config_set_verbose(cfg, true);
    mlx_vector_array outs = mlx_vector_array_new();
    const int rc = mlx_fast_metal_kernel_apply(&outs, g_raw.k, in, cfg, st);
    bool ok = (rc == 0) && mlx_vector_array_size(outs) == 1;
    if (ok && dummy != NULL) {
        mlx_array a0 = mlx_array_new();
        if (mlx_vector_array_get(&a0, outs, 0) == 0) { *dummy = a0; }
        else { mlx_array_free(a0); ok = false; }
    }
    mlx_fast_metal_kernel_config_free(cfg);
    mlx_vector_array_free(outs);
    mlx_vector_array_free(in);
    mlx_array_free(rowarr);
    if (!ok) fprintf(stderr, "[raw] apply rc=%d err=%s\n", rc, g_raw_err);
    return ok;
}

/* Somma di una riga della cache, valutata (l'eval sincronizza la GPU).
 * ATTENZIONE (lezione di questa sessione, §8.13(w)): il terzo elenco di `mlx_slice` e'
 * l'elenco degli STOP (esclusi), NON la forma della fetta. Passare {1,nkv,1,hd} come stop
 * significa fetta VUOTA e la somma viene 0 anche quando il write e' arrivato: per anni
 * questo avrebbe "refutato" un fix funzionante. */
static double raw_row_sum(mlx_array cache, int row, int nkv, int hd, bool* ok, mlx_stream st) {
    const int p0[4] = { 0, 0, row, 0 }, ps[4] = { 1, 1, 1, 1 }, pk[4] = { 1, nkv, row + 1, hd };
    mlx_array rv = mlx_array_new(), s = mlx_array_new(), s32 = mlx_array_new();
    double out = -1.0;
    bool k = mlx_slice(&rv, cache, p0, 4, pk, 4, ps, 4, st) == 0;
    if (k) k = mlx_sum(&s, rv, false, st) == 0;
    if (k) k = mlx_astype(&s32, s, MLX_FLOAT32, st) == 0;
    if (k) k = mlx_array_eval(s32) == 0;
    float item = -1.0f;
    if (k) k = mlx_array_item_float32(&item, s32) == 0;
    if (k) out = (double)item; else *ok = false;
    mlx_array_free(rv); mlx_array_free(s); mlx_array_free(s32);
    return out;
}

/* Somma di TUTTI gli elementi di un array (valutato). Indipendente dagli indici: e' il
 * controllo che distingue "il write non e' arrivato" da "lo leggo nel punto sbagliato". */
static double full_sum(mlx_array a, bool* ok, mlx_stream st) {
    mlx_array s = mlx_array_new(), s32 = mlx_array_new();
    double out = -1.0;
    bool k = mlx_sum(&s, a, false, st) == 0;
    if (k) k = mlx_astype(&s32, s, MLX_FLOAT32, st) == 0;
    if (k) k = mlx_array_eval(s32) == 0;
    float item = -1.0f;
    if (k) k = mlx_array_item_float32(&item, s32) == 0;
    if (k) out = (double)item; else *ok = false;
    mlx_array_free(s); mlx_array_free(s32);
    return out;
}

/* 1 = write raw CORRETTO (riga `off` = nkv*hd, riga `off+1` = 0); 0 = fallito; -1 = errore. */
static int raw_verify_write(int nkv, int hd, mlx_stream st) {
    const int cap = 64, off = 7;
    const int bsh[4] = { 1, nkv, cap, hd }, nsh[4] = { 1, nkv, 1, hd };
    mlx_array kc = mlx_array_new(), vc = mlx_array_new();
    if (mlx_zeros(&kc, bsh, 4, MLX_BFLOAT16, st) != 0 ||
        mlx_zeros(&vc, bsh, 4, MLX_BFLOAT16, st) != 0 ||
        mlx_array_eval(kc) != 0 || mlx_array_eval(vc) != 0) return -1;
    float one[1] = { 1.0f };
    mlx_array val1 = mlx_array_new_data(one, NULL, 0, MLX_FLOAT32);
    mlx_array kn = mlx_array_new(), vn = mlx_array_new();
    if (mlx_full(&kn, nsh, 4, val1, MLX_BFLOAT16, st) != 0 ||
        mlx_full(&vn, nsh, 4, val1, MLX_BFLOAT16, st) != 0) {
        mlx_array_free(val1); mlx_array_free(kc); mlx_array_free(vc); return -1;
    }
    mlx_array_free(val1);
    mlx_array dm = mlx_array_new();
    const bool wok = raw_row_write(kc, vc, kn, vn, off, &dm, st);
    int verdict = 0;
    if (wok) {
        bool ok = true;
        /* Il kernel e' girato? `dummy[0]` viene SCRITTO dal kernel stesso: se vale `cap`
         * la griglia e' partita e il problema e' SOLO la scrittura sulla cache. */
        float dval = -1.0f;
        const bool dev_ok = dm.ctx != NULL && mlx_array_eval(dm) == 0 &&
                            mlx_array_item_float32(&dval, dm) == 0;
        printf("#   dummy(out scrivibile)=%.0f atteso=%d  -> kernel %s\n",
               dev_ok ? dval : -9.0f, cap, dev_ok && dval == (float)cap ? "ESEGUITO" : "NON ESEGUITO");
        const double got  = raw_row_sum(kc, off,     nkv, hd, &ok, st);
        /* Lettura DIRETTA del buffer (memoria unificata): distingue "il write non e'
         * arrivato" da "il write e' arrivato ma MLX ha rivalutato lo zeros".
         * INDICE LINEARE di [1,nkv,cap,hd] = h*cap*hd + row*hd + d  (NON row*nkv*hd). */
        const uint16_t* hp = mlx_array_data_uint16(kc);
        const long hidx = (long)off * (long)hd;
        const long hidx1 = (long)cap * (long)hd + hidx;
        printf("#   byte del buffer testa0 riga %d: 0x%04x (atteso 0x3f80)   testa1: 0x%04x   "
               "riga %d: 0x%04x\n",
               off, hp != NULL ? hp[hidx] : 0xFFFFu, hp != NULL ? hp[hidx1] : 0xFFFFu,
               off + 1, hp != NULL ? hp[hidx + hd] : 0xFFFFu);
        const double gotv = raw_row_sum(vc, off,     nkv, hd, &ok, st);
        const double near = raw_row_sum(kc, off + 1, nkv, hd, &ok, st);
        const double prev = raw_row_sum(kc, off - 1, nkv, hd, &ok, st);
        const double want = (double)((long)nkv * (long)hd);
        bool oks = true;
        const double skc = full_sum(kc, &oks, st), svc = full_sum(vc, &oks, st);
        printf("#   sum(INTERO) K=%.0f V=%.0f (atteso %.0f se il write e' arrivato da qualche parte)\n",
               skc, svc, want);
        printf("#   verify: riga %d K=%.0f V=%.0f (atteso %.0f)  riga %d=%.0f riga %d=%.0f (atteso 0)  %s\n",
               off, got, gotv, want, off + 1, near, off - 1, prev,
               (ok && got == want && gotv == want && near == 0.0 && prev == 0.0 &&
                skc == want && svc == want) ? "OK" : "MISMATCH");
        if (ok && got == want && gotv == want && near == 0.0 && prev == 0.0 &&
            skc == want && svc == want) verdict = 1;
    } else {
        printf("#   verify: apply FALLITO err=%s\n", g_raw_err);
    }
    if (dm.ctx) mlx_array_free(dm);
    mlx_array_free(kn); mlx_array_free(vn); mlx_array_free(kc); mlx_array_free(vc);
    fflush(stdout);
    return verdict;
}

/* ── MODE I: FOGLIA ALIAS sullo stesso buffer MTL della cache ───────────────────
 * See llm_mlx_attn_roof_probe_alias.cpp perche' esiste questa strada (costruttore
 * pubblico `array(allocator::Buffer, ...)` + accessore pubblico `array::buffer()`).
 * Se la foglia e' donabile, `set_copy_output_data` fa `copy_shared_buffer(in)` e
 * SALTAla copia: l'append costa O(riga) SENZA materializzare nulla (nessun drain, a
 * differenza di LLM_KV_MATERIALIZE che costa +2,98 ms/token). */
extern int llm_probe_alias_leaf(mlx_array* out, mlx_array src, const int* shape,
                                int dim, long long* diag);
extern int llm_probe_donatable(mlx_array a);
extern long long llm_probe_data_uses(mlx_array a);
/* Ancoraggio del buffer MTL (arm 4): lo slot tiene viva il Data che POSSIEDE la memoria,
 * perche' la foglia alias ha un deleter no-op. Vedi lo shim .cpp. */
extern long long llm_probe_holder_put(mlx_array a);
extern int llm_probe_alias_leaf_at(mlx_array* out, long long slot, const int* shape,
                                   int dim, long long* nbytes_out);
extern int llm_probe_holder_same_buffer(long long slot, mlx_array a);
extern void llm_probe_holder_clear(void);

/* 1 = il fix a foglia alias FUNZIONA (write visibile nella RADICE e nessun buffer nuovo
 * allocato); 0 = non funziona; -1 = errore. */
static int alias_verify(int nkv, int hd, mlx_stream st) {
    const int cap = 64, off = 7;
    const int bsh[4] = { 1, nkv, cap, hd }, nsh[4] = { 1, nkv, 1, hd };
    const int* psh = bsh;
    mlx_array kc = mlx_array_new(), vc = mlx_array_new();
    if (mlx_zeros(&kc, bsh, 4, MLX_BFLOAT16, st) != 0 ||
        mlx_zeros(&vc, bsh, 4, MLX_BFLOAT16, st) != 0 ||
        mlx_array_eval(kc) != 0 || mlx_array_eval(vc) != 0) return -1;
    float one[1] = { 1.0f };
    mlx_array val1 = mlx_array_new_data(one, NULL, 0, MLX_FLOAT32);
    mlx_array kn = mlx_array_new(), vn = mlx_array_new();
    if (mlx_full(&kn, nsh, 4, val1, MLX_BFLOAT16, st) != 0 ||
        mlx_full(&vn, nsh, 4, val1, MLX_BFLOAT16, st) != 0) {
        mlx_array_free(val1); mlx_array_free(kc); mlx_array_free(vc); return -1;
    }
    mlx_array_free(val1);

    long long dg[8] = { -9, -9, -9, -9, -9, -9, -9, -9 };
    printf("# MODE I — foglia alias sullo STESSO buffer MTL della cache\n");
    {   /* CONTROLLO dell'input: se `mlx_full` col valore float32 non produce 1.0 in bf16
         * ogni verdetto (QUI e in MODE H) e' vuoto: va misurato prima di leggere i write. */
        bool okc = true;
        const double want0 = (double)((long)nkv * (long)hd);
        const double sk = full_sum(kn, &okc, st), sv = full_sum(vn, &okc, st);
        printf("#   CONTROLLO input: sum(knew)=%.0f sum(vnew)=%.0f (atteso %.0f) -> %s\n",
               sk, sv, want0, (okc && sk == want0 && sv == want0) ? "input OK" : "INPUT GUASTO");
        fflush(stdout);
    }
    printf("#   radice: donabile=%d  data_uses=%lld\n",
           llm_probe_donatable(kc), llm_probe_data_uses(kc));
    mlx_array lk = mlx_array_new(), lv = mlx_array_new();
    if (llm_probe_alias_leaf(&lk, kc, psh, 4, dg) != 0 ||
        llm_probe_alias_leaf(&lv, vc, psh, 4, NULL) != 0 ||
        lk.ctx == NULL || lv.ctx == NULL) {
        printf("#   alias_leaf: FALLITO (rc/ctx nulli) diag=%lld,%lld,%lld,%lld,%lld\n",
               dg[0], dg[1], dg[2], dg[3], dg[4]);
        mlx_array_free(kn); mlx_array_free(vn); mlx_array_free(kc); mlx_array_free(vc);
        return -1;
    }
    printf("#   foglia: offset_src=%lld donabile_src=%lld DONABILE_FOGLIA=%lld "
           "data_uses=%lld nbytes_uguali=%lld\n",
           dg[0], dg[1], dg[2], dg[3], dg[4]);
    const void* proot = (const void*)mlx_array_data_uint16(kc);
    if (mlx_array_eval(lk) != 0 || mlx_array_eval(lv) != 0) {
        printf("#   alias_leaf: eval FALLITO\n");
        mlx_array_free(lk); mlx_array_free(lv);
        mlx_array_free(kn); mlx_array_free(vn); mlx_array_free(kc); mlx_array_free(vc);
        return -1;
    }
    printf("#   dopo la costruzione: puntatore foglia %s quello della radice (%p vs %p)\n",
           (const void*)mlx_array_data_uint16(lk) == proot ? "UGUALE a" : "DIVERSO da",
           (const void*)mlx_array_data_uint16(lk), proot);
    fflush(stdout);

    /* STESSO helper di produzione (llm_mlx_ops.h), ma sulla FOGLIA. */
    mlx_array kv = mlx_array_new(), vv = mlx_array_new();
    const bool ok_app = llm_mlx_kv_append_inplace(&lk, &lv, kn, vn, off, 1, 256,
                                                  &kv, &vv, st);
    int verdict = 0;
    if (!ok_app) {
        printf("#   append sulla foglia: FALLITO\n");
    } else {
        /* L'append e' pigro: SENZA eval il descrittore di `lk` non ha ancora un buffer e
         * leggere i puntatori/le righe sarebbe privo di senso (e deref di un Data nullo). */
        if (mlx_array_eval(lk) != 0 || mlx_array_eval(lv) != 0) {
            printf("#   append sulla foglia: eval FALLITO\n");
            verdict = -1;
        } else {
            bool ok = true;
            const double got  = raw_row_sum(kc, off,     nkv, hd, &ok, st);
            const double gotl = raw_row_sum(lk, off,     nkv, hd, &ok, st);
            const double gotv = raw_row_sum(vc, off,     nkv, hd, &ok, st);
            const double near = raw_row_sum(kc, off + 1, nkv, hd, &ok, st);
            const double prev = raw_row_sum(kc, off - 1, nkv, hd, &ok, st);
            const double want = (double)((long)nkv * (long)hd);
            const void* pafter = (const void*)mlx_array_data_uint16(lk);
            bool oks = true;
            const double sfo = full_sum(lk, &oks, st), sra = full_sum(kc, &oks, st);
            printf("#   append: riga %d RADICE K=%.0f  FOGLIA K=%.0f  RADICE V=%.0f (atteso %.0f)  "
                   "riga %d=%.0f riga %d=%.0f (atteso 0)\n",
                   off, got, gotl, gotv, want, off + 1, near, off - 1, prev);
            printf("#   sum(FOGLIA)=%.0f  sum(RADICE)=%.0f  (atteso %.0f se il write e' arrivato)\n",
                   sfo, sra, want);
            printf("#   buffer dopo l'append: foglia %p radice %p -> %s (copia=%s)\n",
                   pafter, proot, pafter == proot ? "STESSO buffer" : "buffer NUOVO",
                   pafter == proot ? "NO" : "SI'");
            printf("#   data_uses radice=%lld foglia=%lld   vista K: donabile=%d\n",
                   llm_probe_data_uses(kc), llm_probe_data_uses(lk), llm_probe_donatable(kv));
            fflush(stdout);
            {   /* CONTROLLO POSITIVO: stesso update sul PERCORSO NORMALE (slice_update che
                 * alloca un buffer nuovo). Se qui la riga non torna 1024, e' guasto il
                 * meccanismo di verifica (full/slice/sum), non il fix. */
                mlx_array kr = mlx_array_new(), ku = mlx_array_new();
                bool cok = true;
                const double cwant = want;
                double cgot = -1.0, csum = -1.0;
                if (mlx_zeros(&kr, bsh, 4, MLX_BFLOAT16, st) == 0 &&
                    mlx_array_eval(kr) == 0) {
                    const int s0[4] = { 0, 0, off, 0 }, st1[4] = { 1, 1, 1, 1 };
                    const int sk[4] = { 1, nkv, off + 1, hd };
                    if (mlx_slice_update(&ku, kr, kn, s0, 4, sk, 4, st1, 4, st) == 0 &&
                        mlx_array_eval(ku) == 0) {
                        cgot = raw_row_sum(ku, off, nkv, hd, &cok, st);
                        csum = full_sum(ku, &cok, st);
                    }
                }
                printf("#   CONTROLLO positivo (slice_update normale, buffer nuovo): "
                       "riga %d=%.0f sum(TUTTO)=%.0f (atteso %.0f) -> %s\n", off, cgot, csum, cwant,
                       (cok && cgot == cwant) ? "verifica OK"
                               : (csum == cwant ? "IL WRITE C'E': e la LETTURA della riga che guasta"
                                                : "VERIFICA GUASTA"));
                fflush(stdout);
                mlx_array_free(kr); mlx_array_free(ku);
            }
            if (ok && got == want && gotl == want && gotv == want && near == 0.0 &&
                prev == 0.0 && pafter == proot) verdict = 1;
        }
    }
    mlx_array_free(kv); mlx_array_free(vv);
    mlx_array_free(lk); mlx_array_free(lv);
    mlx_array_free(kn); mlx_array_free(vn); mlx_array_free(kc); mlx_array_free(vc);
    fflush(stdout);
    return verdict;
}

/* (a) delle tre condizioni di sicurezza di §8.13(w): il passo k+1 SCRIVE la riga k+1 nello
 * stesso buffer MTL che il passo k, ancora in coda, LEGGE nelle righe <=k.
 * Firma = somma PONDERATA per step delle uscite SDPA (q != 0: dipende dal contenuto ESATTO
 * delle righe lette) + somma della cache. Confronto DETERMINISTICO (niente wall-time).
 *   regime: 1 = copia di produzione, 3 = foglia alias (la radice NON viene sostituita)
 *   build : 0 = serializzato (una eval per token)  1 = ASINCRONO (24 step in coda, eval alla
 *           fine: e' il caso in cui la scrittura del passo k+1 puo' finire prima che il
 *           lettore del passo k sia partito)
 *   ctrl  : 0 = normale.  1 = contenuto delle righe x3  2 = righe scritte con off-by-one.
 *           Sono i CONTROLLI POSITIVI: DEVONO cambiare la firma, altrimenti lo strumento e'
 *           cieco e nessun "OK" di equivalenza vale nulla (lezione §8.13(w)).
 * *pos (opzionale) riceve SOLO il termine POSIZIONE-SENSIBILE: una somma di un attention
 * output `window`-ridotto (solo le ultime `window` righe visibili) + le righe della cache
 * pesate PER INDICE. Senza questo termine la firma e' una funzione SIMMETRICA del
 * multiset di valori (l'attention su un prefisso intero non distingue una PERMUTAZIONE di
 * righe: misurato, il controllo off-by-one non muoveva nulla) e non vede un off-by-one.
 * Ritorna 0 OK con la firma in *sig, -1 in caso di errore.
 * Due effetti COLLATERALI letti dalla MODE J (non passano dalla firma):
 *   g_eq_samebuf  = 1 se dopo tutti gli append la radice punta ANCORA il buffer
 *                   dell'ancora (donazione riuscita, zero allocazioni per append),
 *                   0 = buffer diverso (copia), -1 = non valutato / non eseguito.
 *   g_eq_active   = bytes active alla fine del run (contesto per *mempeak). */
static int g_eq_samebuf = -1;
static long long g_eq_active = 0;

static int eq_run(int arm, int async_build, int ctrl, int nkv, int nh, int hd, int nk, int steps,
                  int layers, mlx_stream st, double* sig, double* pos, size_t* mempeak) {
    const int cap = ceil256(nk);
    /* riga di partenza:
     *   ctrl 0/1/2 -> `cap - steps - 4`: TUTTE le scritture restano dentro la capacita',
     *                cosi' il controllo off-by-one non fa scattare il grow (che cambierebbe
     *                la firma per un motivo diverso da una lettura storta);
     *   ctrl 3     -> `cap - 3`: gli ultimi step chiedono `need > cap`, cioe' il GROW cosi'
     *                come avviene in produzione (off <= cap al momento del grow: e' il caso
     *                reale, non una scrittura fuori range a priori). */
    const int row0 = (ctrl == 3) ? (cap - 3) : (cap - steps - 4);
    const int base = row0;
    /* Misurato: con la radice che NON adotta (arm 3) il grow e' impossibile per contratto
     * (la radice resta alla cap vecchia e lo step successivo chiede off > cap): MLX lancia
     * `broadcast_shapes` e il profilo default del binding C fa `exit(-1)`; qui si e' visto
     * anche un SIGSEGV subito dopo l'eccezione. Non lo rilanciamo: il verdetto e' gia'
     * registrato (vedi §8.13(w)). */
    if (ctrl == 3 && arm == 3) { printf("#   (arm 3 + GROW: saltato, gia' misurato in fallimento)\n");
                                 fflush(stdout); return -3; }
    const int bsh[4] = { 1, nkv, cap, hd }, nsh[4] = { 1, nkv, 1, hd }, qsh[4] = { 1, nh, 1, hd };
    const float scale = 1.0f / sqrtf((float)hd);
    mlx_array q = mlx_array_new();
    float qv[1] = { 0.01f };
    mlx_array qval = mlx_array_new_data(qv, NULL, 0, MLX_FLOAT32);
    if (mlx_full(&q, qsh, 4, qval, MLX_BFLOAT16, st) != 0 || mlx_array_eval(q) != 0) return -1;
    mlx_array_free(qval);
    mlx_array* kc = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
    mlx_array* vc = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
    /* arm 4: lo slot e' l'ANCORA del buffer MTL della cache (la foglia alias ha un deleter
     * no-op: se nessuno tiene vivo il Data che possiede la memoria, il MetalAllocator la
     * rimette in circolo e le scritture dei passi successivi finiscono su un buffer riusato.
     * Senza ancora il regime NON e' misurabile: per questo va misurato CON l'ancora. */
    long long* slotk = (long long*)calloc((size_t)layers, sizeof(long long));
    long long* slotv = (long long*)calloc((size_t)layers, sizeof(long long));
    for (int i = 0; i < layers; i++) { slotk[i] = -1; slotv[i] = -1; }
    llm_probe_holder_clear();                      /* rilascia le ancore del run precedente */
    g_eq_samebuf = -1; g_eq_active = 0;
    int grew = 0;
    for (int l = 0; l < layers; l++) {
        kc[l] = mlx_array_new(); vc[l] = mlx_array_new();
        if (mlx_zeros(&kc[l], bsh, 4, MLX_BFLOAT16, st) != 0 ||
            mlx_zeros(&vc[l], bsh, 4, MLX_BFLOAT16, st) != 0 ||
            mlx_array_eval(kc[l]) != 0 || mlx_array_eval(vc[l]) != 0) {
            free(kc); free(vc); free(slotk); free(slotv); mlx_array_free(q); return -1;
        }
        if (arm == 4 || arm == 5) {
            slotk[l] = llm_probe_holder_put(kc[l]);
            slotv[l] = llm_probe_holder_put(vc[l]);
            if (slotk[l] < 0 || slotv[l] < 0) { free(kc); free(vc); free(slotk); free(slotv);
                                               mlx_array_free(q); return -1; }
        }
    }
    if (mempeak) { mlx_clear_cache(); mlx_reset_peak_memory(); *mempeak = 0; }
    double acc = 0.0;
    mlx_vector_array keep = mlx_vector_array_new();   /* nel ramo asincrono tutto resta in coda */
    for (int s = 0; s < steps; s++) {
        const float mul = (ctrl == 1) ? 3.0f : 1.0f;        /* controllo contenuto */
        const int   off = base + s + ((ctrl == 2) ? 1 : 0);
        float kf[1] = { mul * (float)(s + 1) }, vf[1] = { mul * -0.5f * (float)(s + 1) };
        mlx_array ak = mlx_array_new_data(kf, NULL, 0, MLX_FLOAT32);
        mlx_array av = mlx_array_new_data(vf, NULL, 0, MLX_FLOAT32);
        mlx_array kn = mlx_array_new(), vn = mlx_array_new();
        if (mlx_full(&kn, nsh, 4, ak, MLX_BFLOAT16, st) != 0 ||
            mlx_full(&vn, nsh, 4, av, MLX_BFLOAT16, st) != 0) {
            mlx_array_free(ak); mlx_array_free(av); mlx_array_free(kn); mlx_array_free(vn);
            mlx_vector_array_free(keep); mlx_array_free(q); free(kc); free(vc); return -1;
        }
        mlx_array_free(ak); mlx_array_free(av);
        mlx_vector_array stepvec = mlx_vector_array_new();
        const size_t keep_mark = mlx_vector_array_size(keep);   /* ctrl 4: primo oo del passo */
        for (int l = 0; l < layers; l++) {
            mlx_array kv = mlx_array_new(), vv = mlx_array_new(), oo = mlx_array_new();
            mlx_array q_use = mlx_array_new();   /* arm 6: q con catena di dipendenza dal write */
            bool okb;
            if (arm == 1) {
                okb = llm_mlx_kv_append_inplace(&kc[l], &vc[l], kn, vn, off, 1, 256,
                                                &kv, &vv, st);
            } else if (arm == 5) {
                /* arm 5 = foglia alias FRESCA per ogni step + `mlx_slice_update` FUNZIONALE dal
                 * leaf, con le VISTE derivate dal RISULTATO dell'update.
                 *  - donazione: il leaf e' nuovo (Data no-op, use_count 1) => `slice_update`
                 *    dona SEMPRE, anche con le viste del passo precedente ancora vive nel grafo
                 *    (a differenza della catena sul risultato, che condividerebbe il Data);
                 *  - ordine: la vista dipende dal proprio nodo di update (arco read<-write);
                 *    i write fra loro sono su righe DISGIUNTE => nessun hazard di ordine;
                 *  - grow: si delega all'helper (ramo funzionale) + eval + ri-ancoraggio (raro). */
                const int* sk = mlx_array_shape(kc[l]);
                const int* sv = mlx_array_shape(vc[l]);
                int lksh[4] = { 1, nkv, cap, hd }, lvsh[4] = { 1, nkv, cap, hd };
                if (sk != NULL && mlx_array_ndim(kc[l]) == 4)
                    for (int d = 0; d < 4; d++) lksh[d] = sk[d];
                if (sv != NULL && mlx_array_ndim(vc[l]) == 4)
                    for (int d = 0; d < 4; d++) lvsh[d] = sv[d];
                const int cap_before = lksh[2];
                const int need5 = off + 1;
                if (need5 > cap_before) {
                    okb = llm_mlx_kv_append_inplace(&kc[l], &vc[l], kn, vn, off, 1, 256,
                                                    &kv, &vv, st);
                    if (okb) {
                        const int* sk2 = mlx_array_shape(kc[l]);
                        if (sk2 != NULL && sk2[2] != cap_before) {
                            grew++;
                            if (mlx_array_eval(kc[l]) != 0 || mlx_array_eval(vc[l]) != 0 ||
                                (slotk[l] = llm_probe_holder_put(kc[l])) < 0 ||
                                (slotv[l] = llm_probe_holder_put(vc[l])) < 0) okb = false;
                        }
                    }
                } else {
                    mlx_array lk = mlx_array_new(), lv = mlx_array_new();
                    mlx_array rk = mlx_array_new(), rv = mlx_array_new();
                    int rck = llm_probe_alias_leaf_at(&lk, slotk[l], lksh, 4, NULL);
                    int rcv = llm_probe_alias_leaf_at(&lv, slotv[l], lvsh, 4, NULL);
                    int s0[4] = {0,0,off,0}, st4[4] = {1,1,1,1};
                    int wk[4] = {1,nkv,need5,hd}, wv[4] = {1,nkv,need5,hd};
                    okb = rck == 0 && rcv == 0 &&
                          mlx_slice_update(&rk, lk, kn, s0,4, wk,4, st4,4, st) == 0 &&
                          mlx_slice_update(&rv, lv, vn, s0,4, wv,4, st4,4, st) == 0;
                    if (okb) {
                        mlx_array_set(&kc[l], rk);
                        mlx_array_set(&vc[l], rv);
                        int p0[4] = {0,0,0,0}, ps[4] = {1,1,1,1};
                        int pk[4] = {1,nkv,need5,hd}, pv[4] = {1,nkv,need5,hd};
                        okb = mlx_slice(&kv, kc[l], p0,4, pk,4, ps,4, st) == 0 &&
                              mlx_slice(&vv, vc[l], p0,4, pv,4, ps,4, st) == 0;
                    }
                    mlx_array_free(rk); mlx_array_free(rv);
                    mlx_array_free(lk); mlx_array_free(lv);
                }
            } else if (arm == 6) {
                /* arm 6 = CANDIDATO DI PRODUZIONE (MODE K): scrittura RAW in-place (kernel
                 * custom, const_cast — MODE H) nel buffer della RADICE, che NON viene mai
                 * sostituita; viste del prefisso dal root; CATENA DI DIPENDENZA ESPLICITA
                 * write -> SDPA (q_dep = q + (dummy - dummy): valore esattamente 0 ma arco
                 * nel grafo). Senza l'arco la SDPA del passo k puo' leggere la riga k prima
                 * che il write la scriva — e' la classe di race di §8.13x; con l'arco il
                 * write e' un ANTENATO dell'uscita del passo. I write fra passi sono su righe
                 * disgiunte e il grow resta il ramo funzionale esistente (one-time). */
                const int need6 = off + 1;
                const int* sk6 = mlx_array_shape(kc[l]);
                const int cap6 = (sk6 != NULL && mlx_array_ndim(kc[l]) == 4) ? sk6[2] : cap;
                if (need6 > cap6) {
                    okb = llm_mlx_kv_append_inplace(&kc[l], &vc[l], kn, vn, off, 1, 256,
                                                    &kv, &vv, st);
                    if (okb) {
                        const int* sk2 = mlx_array_shape(kc[l]);
                        if (sk2 != NULL && sk2[2] != cap6) grew++;
                    }
                } else {
                    /* ctrl 4 = modello degli archi di readback: il write del passo k+1 (kn/vn)
                     * dipende dall'USCITA del passo k dello stesso layer (in produzione il
                     * token k+1 deriva dall'ultimo layer di k, che dipende da ogni layer).
                     * Senza questo arco il write e' indipendente dalla lettura precedente e
                     * puo' correre con essa (modello di un lag > 1 o di archi assenti). */
                    mlx_array kuse = mlx_array_new(), vuse = mlx_array_new();
                    if (ctrl == 4 && s > 0) {
                        mlx_array prev = mlx_array_new();
                        const size_t pidx = keep_mark - (size_t)(3 * layers) + (size_t)l * 3 + 2;
                        if (mlx_vector_array_size(keep) > pidx &&
                            mlx_vector_array_get(&prev, keep, (int)pidx) == 0) {
                            mlx_array kd = mlx_array_new(), kq = mlx_array_new(), kn2 = mlx_array_new();
                            mlx_array vd = mlx_array_new(), vq = mlx_array_new(), vn2 = mlx_array_new();
                            if (mlx_subtract(&kd, prev, prev, st) == 0 &&
                                mlx_astype(&kq, kd, mlx_array_dtype(kn), st) == 0 &&
                                mlx_add(&kn2, kn, kq, st) == 0 &&
                                mlx_subtract(&vd, prev, prev, st) == 0 &&
                                mlx_astype(&vq, vd, mlx_array_dtype(vn), st) == 0 &&
                                mlx_add(&vn2, vn, vq, st) == 0) {
                                mlx_array_set(&kuse, kn2); mlx_array_set(&vuse, vn2);
                            }
                            mlx_array_free(kd); mlx_array_free(kq); mlx_array_free(kn2);
                            mlx_array_free(vd); mlx_array_free(vq); mlx_array_free(vn2);
                        }
                        mlx_array_free(prev);
                    }
                    mlx_array knew_use = (kuse.ctx != NULL) ? kuse : kn;
                    mlx_array vnew_use = (vuse.ctx != NULL) ? vuse : vn;
                    mlx_array dummy = mlx_array_new();
                    okb = raw_row_write(kc[l], vc[l], knew_use, vnew_use, off, &dummy, st);
                    if (okb) {
                        int p0[4] = {0,0,0,0}, ps[4] = {1,1,1,1};
                        int pk[4] = {1,nkv,need6,hd}, pv[4] = {1,nkv,need6,hd};
                        okb = mlx_slice(&kv, kc[l], p0,4, pk,4, ps,4, st) == 0 &&
                              mlx_slice(&vv, vc[l], p0,4, pv,4, ps,4, st) == 0;
                    }
                    if (okb) {
                        mlx_array dd = mlx_array_new(), qd = mlx_array_new(), ddq = mlx_array_new();
                        okb = mlx_subtract(&dd, dummy, dummy, st) == 0 &&
                              mlx_astype(&ddq, dd, mlx_array_dtype(q), st) == 0 &&
                              mlx_add(&qd, q, ddq, st) == 0;
                        if (okb) mlx_array_set(&q_use, qd);
                        mlx_array_free(dd); mlx_array_free(ddq); mlx_array_free(qd);
                    }
                    mlx_array_free(dummy);
                    mlx_array_free(kuse); mlx_array_free(vuse);
                }
            } else {
                /* arm 3 = la RADICE non si tocca (ogni step riparte da lei con una foglia
                 * nuova: niente vincolo d'ordine fra le scritture -> messo alla prova dal
                 * ramo asincrono).  arm 4 = la foglia nasce dall'ANCORA di buffer e viene
                 * ADOTTATA nella radice, come fa la MODE I: la catena del grafo conserva
                 * l'ordine fra lo step k e il k+1 e la forma della radice segue il grow. */
                const int* sk = mlx_array_shape(kc[l]);
                const int* sv = mlx_array_shape(vc[l]);
                int lksh[4] = { 1, nkv, cap, hd }, lvsh[4] = { 1, nkv, cap, hd };
                if (sk != NULL && mlx_array_ndim(kc[l]) == 4)
                    for (int d = 0; d < 4; d++) lksh[d] = sk[d];
                if (sv != NULL && mlx_array_ndim(vc[l]) == 4)
                    for (int d = 0; d < 4; d++) lvsh[d] = sv[d];
                const int cap_before = lksh[2];
                mlx_array lk = mlx_array_new(), lv = mlx_array_new();
                int rck = (arm == 4) ? llm_probe_alias_leaf_at(&lk, slotk[l], lksh, 4, NULL)
                                     : llm_probe_alias_leaf(&lk, kc[l], lksh, 4, NULL);
                int rcv = (arm == 4) ? llm_probe_alias_leaf_at(&lv, slotv[l], lvsh, 4, NULL)
                                     : llm_probe_alias_leaf(&lv, vc[l], lvsh, 4, NULL);
                okb = rck == 0 && rcv == 0 &&
                      llm_mlx_kv_append_inplace(&lk, &lv, kn, vn, off, 1, 256,
                                                &kv, &vv, st);
                if (arm == 4 && okb) {
                    mlx_array_set(&kc[l], lk); mlx_array_set(&vc[l], lv);
                    const int* sk2 = mlx_array_shape(kc[l]);
                    if (sk2 != NULL && sk2[2] != cap_before) {
                        /* GROW: il buffer e' NUOVO (branch zeros+slice_update). L'ancora
                         * vecchia punterebbe la memoria del regime precedente: va ricaricata
                         * sulla radice appena adottata, che pero' NON e' valutata -> qui (e
                         * SOLO qui, non a ogni step) serve una eval. */
                        grew++;
                        if (mlx_array_eval(kc[l]) != 0 || mlx_array_eval(vc[l]) != 0 ||
                            (slotk[l] = llm_probe_holder_put(kc[l])) < 0 ||
                            (slotv[l] = llm_probe_holder_put(vc[l])) < 0) okb = false;
                    }
                }
                mlx_array_free(lk); mlx_array_free(lv);
            }
            if (!okb || !llm_mlx_gqa_sdpa(&oo, q_use.ctx != NULL ? q_use : q, kv, vv, scale, 0, st)) {
                printf("#   run arm %d fallito al passo %d layer %d (okb=%d)\n",
                       arm, s, l, okb);
                mlx_array_free(kv); mlx_array_free(vv); mlx_array_free(oo);
                mlx_array_free(q_use);
                mlx_vector_array_free(stepvec); mlx_array_free(kn); mlx_array_free(vn);
                mlx_vector_array_free(keep); mlx_array_free(q);
                free(slotk); free(slotv); free(kc); free(vc); return -1;
            }
            mlx_vector_array_append_value(stepvec, oo);
            mlx_vector_array_append_value(keep, kv);
            mlx_vector_array_append_value(keep, vv);
            /* il vector possiede una COPIA dell'handle: i nostri si possono liberare */
            mlx_array_free(kv); mlx_array_free(vv); mlx_array_free(oo);
            mlx_array_free(q_use);
        }
        if (!async_build) {
            if (mlx_eval(stepvec) != 0) {
                mlx_vector_array_free(stepvec); mlx_array_free(kn); mlx_array_free(vn);
                mlx_vector_array_free(keep); mlx_array_free(q); free(kc); free(vc);
                return -1;
            }
            for (size_t i = 0; i < mlx_vector_array_size(stepvec); i++) {
                mlx_array e = mlx_array_new();
                bool ook = true;
                if (mlx_vector_array_get(&e, stepvec, (int)i) == 0) {
                    acc += (double)(s + 1) * full_sum(e, &ook, st);
                } else ook = false;
                mlx_array_free(e);
                if (!ook) { mlx_vector_array_free(stepvec); mlx_array_free(kn);
                            mlx_array_free(vn); mlx_vector_array_free(keep);
                            mlx_array_free(q); free(kc); free(vc); return -1; }
            }
        } else {
            for (size_t i = 0; i < mlx_vector_array_size(stepvec); i++) {
                mlx_array e = mlx_array_new();
                if (mlx_vector_array_get(&e, stepvec, (int)i) == 0)
                    mlx_vector_array_append_value(keep, e);
                mlx_array_free(e);
            }
        }
        mlx_vector_array_free(stepvec);
        mlx_array_free(kn); mlx_array_free(vn);
        if (ctrl == 4 && async_build) {
            /* MODELLO DEGLI ARCHI DI READBACK DELLA PRODUZIONE (lag 1): il token del passo
             * k+1 dipende dall'USCITA del passo k (la readback del sampled id sincronizza
             * l'eval). Nel probe, senza questo arco, i passi sono indipendenti e il write
             * k+1 puo' correre con la lettura k. Qui si ricrea l'arco incatenando la q del
             * passo successivo all'output del primo layer (valore: q + 0 = q esatto). */
            mlx_array first = mlx_array_new(), dd = mlx_array_new(), ddq = mlx_array_new(), q2 = mlx_array_new();
            bool okq = mlx_vector_array_size(keep) > keep_mark + 2 &&
                       mlx_vector_array_get(&first, keep, (int)(keep_mark + 2)) == 0;
            if (okq) okq = mlx_subtract(&dd, first, first, st) == 0 &&
                           mlx_astype(&ddq, dd, mlx_array_dtype(q), st) == 0 &&
                           mlx_add(&q2, q, ddq, st) == 0;
            mlx_array_free(first); mlx_array_free(dd); mlx_array_free(ddq);
            if (okq) mlx_array_set(&q, q2);
            mlx_array_free(q2);
        }
    }
    if (async_build) {
        if (mlx_eval(keep) != 0) { mlx_vector_array_free(keep); mlx_array_free(q);
                                   free(kc); free(vc); return -1; }
        const size_t nk_out = mlx_vector_array_size(keep);
        size_t no_out = 0;                   /* gli OUTPUT sono `layers` per step, in coda al step */
        for (size_t i = 0; i < nk_out; i++) {
            mlx_array e = mlx_array_new();
            bool ook = true;
            if (mlx_vector_array_get(&e, keep, (int)i) == 0) {
                /* solo gli OUTPUT [1,nh,1,hd] entrano nella firma: le viste hanno forma [1,nkv,cap,hd] */
                const int* sh = mlx_array_shape(e);
                if (sh != NULL && mlx_array_ndim(e) == 4 && sh[1] == nh) {
                    acc += (double)((int)(no_out / (size_t)layers) + 1) * full_sum(e, &ook, st);
                    no_out++;
                }
            } else ook = false;
            mlx_array_free(e);
            if (!ook) { mlx_vector_array_free(keep); mlx_array_free(q); free(kc); free(vc); return -1; }
        }
    }
    mlx_vector_array_free(keep);
    bool fok = true;
    double cache_acc = 0.0;
    for (int l = 0; l < layers; l++) {
        cache_acc += full_sum(kc[l], &fok, st) * (double)(l + 1);
        cache_acc += full_sum(vc[l], &fok, st) * (double)(l + 1) * 0.001;
    }
    /* Termine POSIZIONE-SENSIBILE: ogni riga letta e pesata per il SUO indice. E' la sola
     * cosa che vede uno spostamento di riga (la somma totale e l'attention su tutto il
     * prefisso sono funzioni simmetriche del multiset: misurato, non lo vedevano). */
    double posw = 0.0;
    const int* sh0 = mlx_array_shape(kc[0]);
    const int livecap = (sh0 != NULL && mlx_array_ndim(kc[0]) == 4) ? sh0[2] : cap;
    if (ctrl == 3) {
        printf("#   GROW arm %s: cap della RADICE dopo %d passi = %d (partiva da %d, "
               "lettura righe fermata a %d), grow=%d%s\n",
               arm == 1 ? "COPIA" : (arm == 4 ? "ALIAS+ADOZIONE" : "ALIAS"),
               steps, livecap, cap, livecap - 1 - base, grew,
               grew ? " (l'ancora e' stata ricaricata sul buffer nuovo)" : "");
        fflush(stdout);
    }
    for (int l = 0; l < layers && fok; l++) {
        for (int k = -2; k < 2 * steps + 6 && fok; k++) {
            if (base + k >= livecap) break;                 /* post-grow: la cap e' cresciuta */
            const double rk = raw_row_sum(kc[l], base + k, nkv, hd, &fok, st);
            const double rv = raw_row_sum(vc[l], base + k, nkv, hd, &fok, st);
            posw += (double)(k + 5) * (rk + 0.001 * rv) * (double)(l + 1);
        }
    }
    /* Condizione (c) - MEMORIA: picco dall'inizio del run. Se la foglia alias DONA il
     * buffer, i `steps` append non allocano nulla: il picco deve stare sul solo raddoppio
     * del grow (o niente, se non cresce). Nel regime di copia ogni append alloca cap*riga. */
    if (mempeak) { size_t pk = 0, ac = 0;
                   if (mlx_get_peak_memory(&pk) == 0) *mempeak = pk;
                   if (mlx_get_active_memory(&ac) == 0) g_eq_active = (long long)ac; }
    /* Verifica strutturale del candidato: dopo `steps` append la RADICE punta ancora il
     * buffer dell'ancora => la donazione ha riusato lo stesso MTLBuffer (zero allocazioni
     * per append). Richiede che kc[0] sia valutato: lo e', ci siamo passati sopra. */
    if (arm == 4 && layers > 0)
        g_eq_samebuf = llm_probe_holder_same_buffer(slotk[0], kc[0]);
    for (int l = 0; l < layers; l++) { mlx_array_free(kc[l]); mlx_array_free(vc[l]); }
    free(kc); free(vc); free(slotk); free(slotv); mlx_array_free(q);
    *sig = acc + cache_acc + posw;
    if (pos != NULL) *pos = posw;
    return fok ? 0 : -1;
}

/* Braccio di timing: arm 0 = nessuna scrittura, 1 = slice_update (regime di copia),
 * 2 = raw write, 3 = foglia alias (MODE I). Le viste del passo k restano vive nel
 * passo k+1 (hold_refs, come in produzione). Ritorna ms/(layer*token). */
static int bench_raw(int nkv, int nh, int hd, int nk, int steps, int layers, int arm,
                     mlx_stream st, double* out_ms) {
    const int cap = ceil256(nk);
    const int base = cap - steps;
    mlx_array* kc = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
    mlx_array* vc = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
    if (!kc || !vc) { free(kc); free(vc); return -1; }
    const int bsh[4] = { 1, nkv, cap, hd }, nsh[4] = { 1, nkv, 1, hd }, qsh[4] = { 1, nh, 1, hd };
    mlx_array kn = mlx_array_new(), vn = mlx_array_new(), q = mlx_array_new();
    int rc = 0;
    if (mlx_zeros(&kn, nsh, 4, MLX_BFLOAT16, st) != 0 ||
        mlx_zeros(&vn, nsh, 4, MLX_BFLOAT16, st) != 0 ||
        mlx_zeros(&q,  qsh, 4, MLX_BFLOAT16, st) != 0 ||
        mlx_array_eval(kn) != 0 || mlx_array_eval(vn) != 0 || mlx_array_eval(q) != 0) rc = -1;
    for (int l = 0; l < layers && rc == 0; l++) {
        kc[l] = mlx_array_new(); vc[l] = mlx_array_new();
        if (mlx_zeros(&kc[l], bsh, 4, MLX_BFLOAT16, st) != 0 ||
            mlx_zeros(&vc[l], bsh, 4, MLX_BFLOAT16, st) != 0 ||
            mlx_array_eval(kc[l]) != 0 || mlx_array_eval(vc[l]) != 0) rc = -1;
    }
    const float scale = 1.0f / sqrtf((float)hd);
    mlx_array* hk = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
    mlx_array* hv = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
    mlx_array* ho = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
    if (!hk || !hv || !ho) rc = -1;
    for (int pass = 0; pass < 2 && rc == 0; pass++) {
        const int n = (pass == 0) ? 2 : steps;
        int have_held = 0;
        const double t0 = now_s();
        for (int s = 0; s < n && rc == 0; s++) {
            mlx_array* kv = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
            mlx_array* vv = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
            mlx_array* oo = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
            mlx_array* dd = (mlx_array*)calloc((size_t)layers, sizeof(mlx_array));
            mlx_vector_array vec = mlx_vector_array_new();
            if (!kv || !vv || !oo || !dd) { rc = -1; }
            const int need = base + s + 1;
            const int p0[4] = { 0, 0, 0, 0 }, ps[4] = { 1, 1, 1, 1 }, pk[4] = { 1, nkv, need, hd };
            for (int l = 0; l < layers && rc == 0; l++) {
                kv[l] = mlx_array_new(); vv[l] = mlx_array_new();
                oo[l] = mlx_array_new(); dd[l] = mlx_array_new();
                if (arm == 1) {
                    if (!llm_mlx_kv_append_inplace(&kc[l], &vc[l], kn, vn, base + s, 1, 256,
                                                   &kv[l], &vv[l], st)) { rc = -2; break; }
                } else if (arm == 3) {
                    /* MODE I: la scrittura parte da una FOGLIA ALIAS (donabile per
                     * costruzione), la radice `kc[l]/vc[l]` non si tocca. */
                    mlx_array lk = mlx_array_new(), lv = mlx_array_new();
                    const bool buk =
                        llm_probe_alias_leaf(&lk, kc[l], bsh, 4, NULL) == 0 && lk.ctx != NULL &&
                        llm_probe_alias_leaf(&lv, vc[l], bsh, 4, NULL) == 0 && lv.ctx != NULL &&
                        llm_mlx_kv_append_inplace(&lk, &lv, kn, vn, base + s, 1, 256,
                                                  &kv[l], &vv[l], st);
                    mlx_array_free(lk); mlx_array_free(lv);
                    if (!buk) { rc = -7; break; }
                } else {
                    if (arm == 2 && !raw_row_write(kc[l], vc[l], kn, vn, base + s, &dd[l], st)) {
                        rc = -6; break;
                    }
                    if (mlx_slice(&kv[l], kc[l], p0, 4, pk, 4, ps, 4, st) != 0 ||
                        mlx_slice(&vv[l], vc[l], p0, 4, pk, 4, ps, 4, st) != 0) { rc = -5; break; }
                }
                if (!llm_mlx_gqa_sdpa(&oo[l], q, kv[l], vv[l], scale, 0, st)) { rc = -4; break; }
                mlx_vector_array_append_value(vec, oo[l]);
            }
            if (rc == 0 && mlx_eval(vec) != 0) rc = -3;
            for (int l = 0; l < layers; l++) {
                if (have_held) {   /* il passo k+1 si builda con il passo k ANCORA VIVO */
                    if (hk[l].ctx) mlx_array_free(hk[l]);
                    if (hv[l].ctx) mlx_array_free(hv[l]);
                    if (ho[l].ctx) mlx_array_free(ho[l]);
                }
                hk[l] = kv[l]; hv[l] = vv[l]; ho[l] = oo[l];
                kv[l].ctx = NULL; vv[l].ctx = NULL; oo[l].ctx = NULL;
                if (dd && dd[l].ctx) mlx_array_free(dd[l]);
            }
            have_held = 1;
            free(kv); free(vv); free(oo); free(dd);
            mlx_vector_array_free(vec);
        }
        for (int l = 0; l < layers; l++) {
            if (ho[l].ctx) mlx_array_free(ho[l]);
            if (hk[l].ctx) mlx_array_free(hk[l]);
            if (hv[l].ctx) mlx_array_free(hv[l]);
            ho[l].ctx = NULL; hk[l].ctx = NULL; hv[l].ctx = NULL;
        }
        if (pass == 1 && rc == 0)
            *out_ms = (now_s() - t0) * 1e3 / ((double)n * (double)layers);
    }
    for (int l = 0; l < layers; l++) { if (kc[l].ctx) mlx_array_free(kc[l]); if (vc[l].ctx) mlx_array_free(vc[l]); }
    free(kc); free(vc); free(hk); free(hv); free(ho);
    mlx_array_free(kn); mlx_array_free(vn); mlx_array_free(q);
    return rc;
}

int main(int argc, char** argv) {
    const int nh  = (argc > 1) ? atoi(argv[1]) : 32;   /* Qwen3-8B: 32 teste query */
    const int nkv = (argc > 2) ? atoi(argv[2]) : 8;    /*            8 teste KV  (GQA 4x) */
    const int hd  = (argc > 3) ? atoi(argv[3]) : 128;  /*            head_dim 128 */
    int reps      = (argc > 4) ? atoi(argv[4]) : 60;

    int nks[16]; int nnk = 0;
    const char* env = getenv("LLM_ATTN_PROBE_NK");
    if (env != NULL) {
        char buf[256]; snprintf(buf, sizeof buf, "%s", env);
        for (char* tok = strtok(buf, " ,"); tok != NULL && nnk < 16; tok = strtok(NULL, " ,"))
            nks[nnk++] = atoi(tok);
    } else {
        const int def[] = { 512, 2048, 4096, 4192, 8192, 16384, 32768 };
        for (size_t i = 0; i < sizeof def / sizeof def[0] && nnk < 16; i++) nks[nnk++] = def[i];
    }

    mlx_device dev = mlx_device_new_type(MLX_GPU, 0);
    if (dev.ctx == NULL) { fprintf(stderr, "FAIL: device GPU\n"); return 1; }
    if (mlx_set_default_device(dev) != 0) { fprintf(stderr, "FAIL: set_default_device\n"); return 1; }
    mlx_stream st = mlx_stream_new();
    if (mlx_get_default_stream(&st, dev) != 0 || st.ctx == NULL) {
        fprintf(stderr, "FAIL: default_stream\n"); return 1;
    }

    const int qsh[4] = { 1, nh, 1, hd };
    mlx_array q = mlx_array_new();
    if (mlx_zeros(&q, qsh, 4, MLX_BFLOAT16, st) != 0 || mlx_array_eval(q) != 0) {
        fprintf(stderr, "FAIL: q\n"); return 1;
    }

    /* ── MODE P (invocazione separata: LLM_ATTN_PROBE_PATTERN=1) ───────────────
     *.pattern = append in-place + vista + SDPA per layer, UNA eval per token: il costo
     * REALE per (layer, token) a confronto col solo kernel (colonna G sopra). */
    if (getenv("LLM_ATTN_PROBE_PATTERN") != NULL) {
        const int layers = (argc > 5) ? atoi(argv[5]) : 4;
        const int steps  = (argc > 6) ? atoi(argv[6]) : 24;
        const int aliased = (getenv("LLM_ATTN_PROBE_ALIASED") != NULL);
        const int concat  = (getenv("LLM_ATTN_PROBE_CONCAT") != NULL);
        printf("# PATTERN decode (append in-place + vista + SDPA), layers=%d steps=%d  "
               "(memoria ~%.2f GB al massimo Nk)  %s\n", layers, steps,
               (layers * 2.0 * 32768 * nkv * hd * 2) / 1e9,
               aliased ? "ALIASED=ON (terza braccia: passo k vivo durante gli append di k+1)"
                       : "ALIASED=off (usa LLM_ATTN_PROBE_ALIASED=1 per il ramo produzione)");
        printf("%8s %12s %12s %12s %10s %12s %12s %12s\n",
               "Nk", "append_ms", "noapp_ms", "delta_ms", "GB/s", "peakA_MB", "peakNoA_MB",
               aliased ? "alias_ms" : "");
        fflush(stdout);
        for (int i = 0; i < nnk; i++) {
            const int nk = nks[i];
            const double kv_bytes = 2.0 * (double)nkv * (double)nk * (double)hd * 2.0;
            double pa = 0.0, pn = 0.0, pkA = -1.0, pkN = -1.0, pz = 0.0, pc = 0.0;
            int prc = bench_pattern(nkv, nh, hd, nk, steps, layers, /*do_append=*/0,
                                    /*hold_refs=*/0, /*concat_origin=*/0, st, &pn, &pkN);
            if (prc == 0) prc = bench_pattern(nkv, nh, hd, nk, steps, layers, /*do_append=*/1,
                                              /*hold_refs=*/0, /*concat_origin=*/0, st, &pa, &pkA);
            if (prc == 0 && aliased)
                prc = bench_pattern(nkv, nh, hd, nk, steps, layers, /*do_append=*/1,
                                    /*hold_refs=*/1, /*concat_origin=*/0, st, &pz, NULL);
            if (prc == 0 && concat)
                prc = bench_pattern(nkv, nh, hd, nk, steps, layers, /*do_append=*/1,
                                    /*hold_refs=*/0, /*concat_origin=*/1, st, &pc, NULL);
            if (prc != 0) { printf("%8d  FAIL rc=%d\n", nk, prc); }
            else if (aliased)
                printf("%8d %12.4f %12.4f %12.4f %10.0f %12.1f %12.1f %12.4f  alias-vs-donabile=+%8.4f ms/layer (rapporto x%.2f)\n",
                       nk, pa, pn, pa - pn, kv_bytes / (pa * 1e-3) / 1e9, pkA, pkN, pz,
                       pz - pa, pa > 0 ? pz / pa : 0.0);
            else printf("%8d %12.4f %12.4f %12.4f %10.0f %12.1f %12.1f\n",
                        nk, pa, pn, pa - pn, kv_bytes / (pa * 1e-3) / 1e9, pkA, pkN);
            if (prc == 0 && concat)
                printf("         CONCAT-origin (prefill reale): append_ms=%12.4f  vs zeros-origin %12.4f   "
                       "Delta_copia=+%8.4f ms/layer = +%6.2f ms/token @%d layer  (x%.2f)\n",
                       pc, pa, pc - pa, (pc - pa) * layers, layers, pa > 0 ? pc / pa : 0.0);
            fflush(stdout);
        }
        mlx_array_free(q);
        mlx_stream_free(st);
        mlx_device_free(dev);
        return 0;
    }

    /* ── MODE H (invocazione separata: LLM_ATTN_PROBE_RAW=1) ────────────────────
     * Prezzi per (layer,token): nessuna scrittura vs `slice_update` nel regime di copia vs
     * `raw write`. 3 round INTERLACCIATI, si tiene il MIN (robusto alla contesa: mai fidarsi
     * del valore assoluto su questa macchina condivisa). */
    if (getenv("LLM_ATTN_PROBE_RAW") != NULL) {
        const int layers = (argc > 5) ? atoi(argv[5]) : 4;
        const int steps  = (argc > 6) ? atoi(argv[6]) : 24;
        const int rounds = (argc > 7) ? atoi(argv[7]) : 3;
        const int vw = raw_verify_write(nkv, hd, st);
        printf("# MODE H — append RAW nel buffer ESISTENTE (const_cast nel kernel custom)\n");
        printf("#   layers=%d steps=%d rounds=%d  memoria ~%.2f GB al massimo Nk\n",
               layers, steps, rounds, (layers * 2.0 * 32768 * nkv * hd * 2) / 1e9);
        printf("#   VERIFY WRITE (deterministico, non wall-time) = %s\n",
               vw == 1 ? "OK (il write arriva nel buffer della cache)"
                       : (vw == 0 ? "FALLITO (il write NON arriva / stride sbagliato)"
                                  : "ERRORE (apply fallita: vedi stderr)"));
        printf("%8s %12s %12s %12s %14s %14s %10s\n",
               "Nk", "none_ms", "su_ms", "raw_ms", "copia_su", "copia_raw", "raw/none");
        fflush(stdout);
        for (int i = 0; i < nnk; i++) {
            const int nk = nks[i];
            double bn = 1e30, bs = 1e30, br = 1e30;
            int prc = 0;
            for (int r = 0; r < rounds && prc == 0; r++) {
                double a = 0.0;
                if ((prc = bench_raw(nkv, nh, hd, nk, steps, layers, 0, st, &a)) != 0) break;
                if (a < bn) bn = a;
                if ((prc = bench_raw(nkv, nh, hd, nk, steps, layers, 1, st, &a)) != 0) break;
                if (a < bs) bs = a;
                if ((prc = bench_raw(nkv, nh, hd, nk, steps, layers, 2, st, &a)) != 0) break;
                if (a < br) br = a;
            }
            if (prc != 0) { printf("%8d  FAIL rc=%d\n", nk, prc); fflush(stdout); continue; }
            printf("%8d %12.4f %12.4f %12.4f %14.4f %14.4f %10.2f\n",
                   nk, bn, bs, br, bs - bn, br - bn, bn > 0 ? br / bn : 0.0);
            printf("#   @%d: copia slice_update = %+.4f ms/layer (%+.2f ms/token x%d layer)   "
                   "raw = %+.4f ms/layer (%+.2f ms/token)   recupero = %.1f%%\n",
                   nk, bs - bn, (bs - bn) * layers, layers,
                   br - bn, (br - bn) * layers,
                   (bs - bn) > 1e-9 ? 100.0 * (1.0 - (br - bn) / (bs - bn)) : 0.0);
            fflush(stdout);
        }
        mlx_array_free(q); mlx_stream_free(st); mlx_device_free(dev);
        if (g_raw.built && g_raw.k.ctx) mlx_fast_metal_kernel_free(g_raw.k);
        return 0;
    }

    /* ── MODE I (invocazione separata: LLM_ATTN_PROBE_ALIAS=1) ───────────────────
     * Prezzo dell'append se parte da una FOGLIA ALIAS (stesso buffer MTL, donabile per
     * costruzione): arm 0 = nessuna scrittura, arm 1 = regime di copia (produzione),
     * arm 3 = foglia alias. atteso: arm3 - arm0 ~ 0 (la copia O(cap) sparisce). */
    if (getenv("LLM_ATTN_PROBE_ALIAS") != NULL) {
        const int layers = (argc > 5) ? atoi(argv[5]) : 4;
        const int steps  = (argc > 6) ? atoi(argv[6]) : 24;
        const int rounds = (argc > 7) ? atoi(argv[7]) : 3;
        const int vw = alias_verify(nkv, hd, st);
        printf("#   VERIFY ALIAS (deterministico, non wall-time) = %s\n",
               vw == 1 ? "OK (scrive nella cache conDIVISA, nessun buffer nuovo)"
                       : (vw == 0 ? "FALLITO (copia ancora in atto o write non visibile)"
                                  : "ERRORE (foglia alias non costruibile)"));
        printf("#   layers=%d steps=%d rounds=%d  memoria ~%.2f GB al massimo Nk\n",
               layers, steps, rounds, (layers * 2.0 * 32768 * nkv * hd * 2) / 1e9);
        printf("%8s %12s %12s %12s %14s %14s %10s\n",
               "Nk", "none_ms", "copia_ms", "alias_ms", "copia-su-none", "alias-su-none", "recup%");
        fflush(stdout);
        for (int i = 0; i < nnk; i++) {
            const int nk = nks[i];
            double bn = 1e30, bs = 1e30, ba = 1e30;
            int prc = 0;
            for (int r = 0; r < rounds && prc == 0; r++) {
                double a = 0.0;
                if ((prc = bench_raw(nkv, nh, hd, nk, steps, layers, 0, st, &a)) != 0) break;
                if (a < bn) bn = a;
                if ((prc = bench_raw(nkv, nh, hd, nk, steps, layers, 1, st, &a)) != 0) break;
                if (a < bs) bs = a;
                if ((prc = bench_raw(nkv, nh, hd, nk, steps, layers, 3, st, &a)) != 0) break;
                if (a < ba) ba = a;
            }
            if (prc != 0) { printf("%8d  FAIL rc=%d\n", nk, prc); fflush(stdout); continue; }
            printf("%8d %12.4f %12.4f %12.4f %14.4f %14.4f %10.1f\n",
                   nk, bn, bs, ba, bs - bn, ba - bn,
                   (bs - bn) > 1e-9 ? 100.0 * (1.0 - (ba - bn) / (bs - bn)) : 0.0);
            printf("#   @%d: copia produzione = %+.4f ms/layer (%+.2f ms/token x%d layer)   "
                   "alias = %+.4f ms/layer (%+.2f ms/token)\n",
                   nk, bs - bn, (bs - bn) * layers, layers,
                   ba - bn, (ba - bn) * layers);
            fflush(stdout);
        }
        mlx_array_free(q); mlx_stream_free(st); mlx_device_free(dev);
        return 0;
    }

    /* ── MODE J (invocazione separata: LLM_ATTN_PROBE_EQ=1) ─────────────────────
     * EQUIVALENZA deterministica (firma numerica, niente wall-time) fra il regime di
     * COPIA di produzione e la FOGLIA ALIAS, sia serializzato sia asincrono (tutti gli
     * step in coda, una sola eval alla fine). E' la condizione (a) di §8.13(w): il passo
     * k+1 scrive la riga k+1 nello stesso MTLBuffer che il passo k, ancora in coda, legge
     * nelle righe <=k.  I due CONTROLLI POSITIVI devono cambiare la firma: se non lo
     * fanno lo strumento e' cieco e l'"uguale" non vale nulla. */
    if (getenv("LLM_ATTN_PROBE_EQ") != NULL) {
        const int layers = (argc > 5) ? atoi(argv[5]) : 2;
        const int steps  = (argc > 6) ? atoi(argv[6]) : 8;
        const char* sne  = getenv("LLM_ATTN_PROBE_EQ_NK");
        const int nk     = sne ? atoi(sne) : 2048;
        struct { int arm, asy, ctrl; const char* name; } R[] = {
            { 1, 0, 0, "copia  serializzato  (riferimento)" },
            { 3, 0, 0, "alias  serializzato  (candidato)"   },
            { 1, 1, 0, "copia  asincrono     (24 step in coda)" },
            { 3, 1, 0, "alias  asincrono     (candidato + pipeline)" },
            { 1, 0, 1, "CONTROLLO contenuto righe x3 (deve DIVERSI)" },
            { 1, 0, 2, "CONTROLLO righe scritte +1   (deve DIVERSI)" },
            { 1, 1, 1, "CONTROLLO asincrono: contenuto x3 (deve DIVERSI)" },
            { 1, 1, 2, "CONTROLLO asincrono: righe +1     (deve DIVERSI)" },
            { 1, 0, 3, "GROW copia   (serve oltre la cap)" },
            { 3, 0, 3, "GROW alias   (la radice non puo' crescere)" },
            { 4, 0, 0, "alias+ADOZIONE serializzato  (= MODE I)" },
            { 4, 1, 0, "alias+ADOZIONE asincrono     (= MODE I + pipeline)" },
            { 4, 0, 3, "GROW alias+ADOZIONE" },
            { 4, 1, 2, "CONTROLLO alias+ADOZIONE asincrono: righe +1" },
            { 4, 1, 3, "GROW alias+ADOZIONE asincrono" },
            { 4, 0, 2, "CONTROLLO alias+ADOZIONE serializzato: righe +1" },
            { 5, 0, 0, "alias FUNZIONALE serializzato (leaf fresco + vista dal risultato)" },
            { 5, 1, 0, "alias FUNZIONALE asincrono     (idem + pipeline)" },
            { 5, 0, 3, "GROW alias FUNZIONALE" },
            { 5, 1, 3, "GROW alias FUNZIONALE asincrono" },
            { 5, 1, 2, "CONTROLLO alias FUNZIONALE asincrono: righe +1" },
            { 1, 1, 3, "GROW copia asincrono (riferimento)" },
            { 6, 0, 0, "RAW+DEP serializzato (candidato di produzione)" },
            { 6, 1, 0, "RAW+DEP asincrono   (candidato + pipeline)" },
            { 6, 0, 2, "CONTROLLO RAW+DEP serializzato: righe +1" },
            { 6, 1, 2, "CONTROLLO RAW+DEP asincrono: righe +1" },
            { 6, 0, 3, "GROW RAW+DEP (fallback funzionale)" },
            { 6, 1, 3, "GROW RAW+DEP asincrono" },
            { 1, 1, 4, "copia asincrono + archi readback (riferimento)" },
            { 4, 1, 4, "alias+ADOZIONE asincrono + archi readback" },
            { 6, 1, 4, "RAW+DEP asincrono + archi readback (modello produzione)" },
        };
        const int nr = (int)(sizeof(R) / sizeof(R[0]));
        double sg[40], psg[40]; int rcs[40]; size_t mps[40]; int sbf[40];
        /* Il DEFAULT handler di mlx-c fa `printf + exit(-1)`: senza questo, la prima
         * eccezione MLX uccide il processo e la tabella non viene stampata (misurato: exit
         * 255). In produzione NON viene installato nessun handler: un'eccezione MLX ammazza
         * il motore, quindi QUI serve anche il TESTO dell'errore come evidenza. */
        char errs[40][512];
        mlx_set_error_handler(g_raw_err_cb, NULL, NULL);
        for (int i = 0; i < nr; i++) {
            sg[i] = 0.0; psg[i] = 0.0; g_raw_err[0] = '\0'; mps[i] = 0;
            rcs[i] = eq_run(R[i].arm, R[i].asy, R[i].ctrl,
                            nkv, nh, hd, nk, steps, layers, st, &sg[i], &psg[i], &mps[i]);
            snprintf(errs[i], sizeof errs[i], "%s", g_raw_err);
            sbf[i] = g_eq_samebuf;
        }
        mlx_set_error_handler(NULL, NULL, NULL);   /* torna al default (exit) */
        printf("# MODE J — equivalenza copia vs foglia alias (nk=%d layers=%d steps=%d, "
               "memoria ~%.2f GB, NESSUN modello)\n", nk, layers, steps,
               (layers * 2.0 * (double)ceil256(nk) * nkv * hd * 2 * (steps + 2)) / 1e9);
        printf("%2s %-46s %24s %14s %11s %5s %8s\n",
               "#", "run", "firma", "firma-pos", "peak MB", "sbuf", "rc");
        for (int i = 0; i < nr; i++)
            printf("%2d %-46s %24.17g %14.17g %11.1f %5d %8d\n", i, R[i].name,
                   rcs[i] == 0 ? sg[i] : 0.0 / 0.0, rcs[i] == 0 ? psg[i] : 0.0 / 0.0,
                   (double)mps[i] / 1e6, sbf[i], rcs[i]);
        const double ref = sg[0];
        for (int i = 0; i < nr; i++)
            printf("#   run%d: delta firma = %+.6e   delta firma-pos = %+.6e%s%s\n", i,
                   sg[i] - ref, psg[i] - psg[0],
                   (rcs[i] != 0 && errs[i][0]) ? "  ERRORE MLX: " : "",
                   (rcs[i] != 0 && errs[i][0]) ? errs[i] : "");
        const int ok_run = (rcs[0] == 0 && rcs[1] == 0 && rcs[4] == 0 && rcs[5] == 0 &&
                            rcs[6] == 0 && rcs[7] == 0);
        const int sens_c = ok_run && fabs(sg[4] - sg[0]) > 0.0 && fabs(sg[6] - sg[2]) > 0.0;
        const int sens_o = ok_run && fabs(sg[5] - sg[0]) > 0.0 && fabs(sg[7] - sg[2]) > 0.0;
        printf("# STRUMENTO SENSIBILE (controlli positivi, serializzato + asincrono): "
               "contenuto=%s off-by-one=%s\n",
               sens_c ? "SI'" : "NO (cieco: verdetto non pubblicabile)",
               sens_o ? "SI'" : "NO (cieco: verdetto non pubblicabile)");
        if (!ok_run) printf("# ESITO: run falliti, non misurabile\n");
        else {
            printf("# alias==copia (serializzato): %s\n", sg[1] == sg[0] ? "IDENTICO" : "DIVERSO");
            if (rcs[2] == 0 && rcs[3] == 0)
                printf("# alias==copia (asincrono):  %s\n", sg[3] == sg[2] ? "IDENTICO" : "DIVERSO");
            else printf("# alias==copia (asincrono):  non eseguito (rc %d/%d)\n", rcs[2], rcs[3]);
            if (rcs[2] == 0)
                printf("# copia asincrona == copia serializzata: %s  <- se DIVERSO, e' la "
                       "PRODUZIONE ad avere l'hazard, non la foglia alias\n",
                       sg[2] == sg[0] ? "IDENTICO" : "DIVERSO");
            if (nr > 9 && rcs[8] == 0 && rcs[9] == 0)
                printf("# GROW (ctrl 3): alias == copia: %s  <- finche' DIVERSO, la foglia "
                       "alias DEVE essere usata solo nel ramo `cap >= need` (in produzione "
                       "il grow resta il percorso esistente, one-time ogni 256 token)\n",
                       sg[9] == sg[8] ? "IDENTICO" : "DIVERSO");
        }
        if (nr > 13) {
            printf("# ── ARM 4 (foglia alias + ADOZIONE nella radice: il candidato di MODE I) ──\n");
            if (rcs[10] == 0)
                printf("# alias+adozione serializzato == copia: %s   (stesso buffer "
                       "dell'ancora: sbuf=%d)\n",
                       sg[10] == sg[0] ? "IDENTICO" : "DIVERSO", sbf[10]);
            else printf("# alias+adozione serializzato: rc %d %s\n", rcs[10], errs[10]);
            if (rcs[11] == 0 && rcs[2] == 0)
                printf("# alias+adozione ASINCRONO == copia asincrono: %s   <- LA condizione "
                       "(a): senza vincolo d'ordine (arm 3) qui e' risultato DIVERSO\n",
                       sg[11] == sg[2] ? "IDENTICO" : "DIVERSO");
            else printf("# alias+adozione asincrono: rc %d/%d %s\n", rcs[11], rcs[2], errs[11]);
            if (rcs[12] == 0 && rcs[8] == 0)
                printf("# alias+adozione GROW == copia GROW: %s   <- condizione (b) (serializzato)\n",
                       sg[12] == sg[8] ? "IDENTICO" : "DIVERSO");
            else printf("# alias+adozione grow: rc %d/%d %s\n", rcs[12], rcs[8], errs[12]);
            if (rcs[14] == 0 && rcs[2] == 0)
                printf("# alias+adozione GROW asincrono == copia asincrono: %s   <- condizione "
                       "(b) nel regime di pipeline\n",
                       sg[14] == sg[2] ? "IDENTICO" : "DIVERSO");
            else printf("# alias+adozione grow asincrono: rc %d/%d %s\n", rcs[14], rcs[2], errs[14]);
            if (rcs[15] == 0 && rcs[10] == 0)
                printf("# controllo off-by-one arm4 serializzato: %s\n",
                       sg[15] != sg[10] ? "DIVERSO (strumento sensibile)" : "IDENTICO (CIECO!)");
            if (rcs[13] == 0 && rcs[11] == 0)
                printf("# controllo off-by-one arm4 asincrono: %s   <- deve DIVERSO\n",
                       sg[13] != sg[11] ? "DIVERSO (strumento sensibile)" : "IDENTICO (CIECO!)");
        }
        if (nr > 21) {
            printf("# ── ARM 5 (foglia alias FRESCA + slice_update FUNZIONALE + vista dal risultato) ──\n");
            if (rcs[16] == 0)
                printf("# alias-funzionale serializzato == copia: %s\n",
                       sg[16] == sg[0] ? "IDENTICO" : "DIVERSO");
            if (rcs[17] == 0 && rcs[2] == 0)
                printf("# alias-funzionale ASINCRONO == copia asincrono: %s   <- LA condizione (a)\n",
                       sg[17] == sg[2] ? "IDENTICO" : "DIVERSO");
            if (rcs[18] == 0 && rcs[8] == 0)
                printf("# alias-funzionale GROW == copia GROW: %s   <- condizione (b) serializzata\n",
                       sg[18] == sg[8] ? "IDENTICO" : "DIVERSO");
            if (rcs[19] == 0 && rcs[21] == 0)
                printf("# alias-funzionale GROW asincrono == GROW copia asincrono: %s   <- condizione (b) pipeline\n",
                       sg[19] == sg[21] ? "IDENTICO" : "DIVERSO");
            if (rcs[20] == 0 && rcs[17] == 0)
                printf("# controllo off-by-one arm5 asincrono: %s   <- deve DIVERSO\n",
                       sg[20] != sg[17] ? "DIVERSO (strumento sensibile)" : "IDENTICO (CIECO!)");
        }
        if (nr > 27) {
            printf("# ── ARM 6 / MODE K (RAW write in-place + catena di dipendenza write->SDPA) ──\n");
            if (rcs[22] == 0)
                printf("# RAW+DEP serializzato == copia: %s\n",
                       sg[22] == sg[0] ? "IDENTICO" : "DIVERSO");
            if (rcs[23] == 0 && rcs[2] == 0)
                printf("# RAW+DEP ASINCRONO == copia asincrono: %s   <- LA condizione (a) di MODE K\n",
                       sg[23] == sg[2] ? "IDENTICO" : "DIVERSO");
            if (rcs[24] == 0 && rcs[22] == 0)
                printf("# controllo off-by-one arm6 serializzato: %s   <- deve DIVERSO\n",
                       sg[24] != sg[22] ? "DIVERSO (strumento sensibile)" : "IDENTICO (CIECO!)");
            if (rcs[25] == 0 && rcs[23] == 0)
                printf("# controllo off-by-one arm6 asincrono: %s   <- deve DIVERSO\n",
                       sg[25] != sg[23] ? "DIVERSO (strumento sensibile)" : "IDENTICO (CIECO!)");
            if (rcs[26] == 0 && rcs[8] == 0)
                printf("# RAW+DEP GROW == copia GROW: %s   <- condizione (b) serializzata\n",
                       sg[26] == sg[8] ? "IDENTICO" : "DIVERSO");
            if (rcs[27] == 0 && rcs[21] == 0)
                printf("# RAW+DEP GROW asincrono == GROW copia asincrono: %s   <- condizione (b) pipeline\n",
                       sg[27] == sg[21] ? "IDENTICO" : "DIVERSO");
        }
        if (nr > 30) {
            printf("# ── ARCHI DI READBACK (modello produzione lag 1: token k+1 <- uscita k) ──\n");
            if (rcs[28] == 0 && rcs[2] == 0)
                printf("# copia+archi == copia asincrono: %s   <- gli archi NON cambiano i valori\n",
                       sg[28] == sg[2] ? "IDENTICO" : "DIVERSO");
            if (rcs[29] == 0 && rcs[28] == 0)
                printf("# alias+ADOZIONE+archi == copia+archi: %s\n",
                       sg[29] == sg[28] ? "IDENTICO" : "DIVERSO");
            if (rcs[30] == 0 && rcs[28] == 0)
                printf("# RAW+DEP+archi == copia+archi: %s   <- SE IDENTICO: il design e' sicuro "
                       "nel regime di produzione (lag 1); la divergenza senza archi e' il "
                       "modello di un lag > 1\n",
                       sg[30] == sg[28] ? "IDENTICO" : "DIVERSO");
        }
        /* Condizione (c): MEMORY. Il picco e' misurato dall'inizio del run (reset prima dei
         * passi). Il regime di copia alloca un buffer da cap*righe_per_token OGNI append:
         * se l'alias dona, il picco crolla a ~1 cache per layer. */
        printf("# ── MEMORIA (peak MB dall'inizio del run; 11=copia ser 12=... vedi tabella) ──\n");
        printf("#   copia  serializzato peak %8.1f MB   alias(no-adz) %8.1f   "
               "alias+adozione %8.1f   alias-funzionale %8.1f\n",
               (double)mps[0] / 1e6, (double)mps[1] / 1e6, (double)mps[10] / 1e6,
               (double)mps[16] / 1e6);
        printf("#   copia  asincrono    peak %8.1f MB   alias(no-adz) %8.1f   "
               "alias+adozione %8.1f   alias-funzionale %8.1f   (active fine run: %lld MB)\n",
               (double)mps[2] / 1e6, (double)mps[3] / 1e6, (double)mps[11] / 1e6,
               (double)mps[17] / 1e6, g_eq_active / (1000 * 1000));
        fflush(stdout);
        mlx_array_free(q); mlx_stream_free(st); mlx_device_free(dev);
        return 0;
    }

    /* ── MODE F (invocazione separata: LLM_ATTN_PROBE_INTERF=<nbufs>) ──────────────
     * KV e pesi separatamente e poi insieme, sullo stesso working set freddo: vedi
     * il commento di bench_interf. `x_interf` = both/(sdpa+w): 1,00 = additivo. */
    if (getenv("LLM_ATTN_PROBE_INTERF") != NULL) {
        const int nbufs = atoi(getenv("LLM_ATTN_PROBE_INTERF"));
        const int wmb   = getenv("LLM_ATTN_PROBE_WMB") ? atoi(getenv("LLM_ATTN_PROBE_WMB")) : 82;
        const int nwb   = getenv("LLM_ATTN_PROBE_NWB") ? atoi(getenv("LLM_ATTN_PROBE_NWB")) : 4;
        const int freps = (argc > 7) ? atoi(argv[7]) : 3;
        if (nbufs <= 0) { fprintf(stderr, "FAIL: nbufs\n"); return 1; }
        printf("# INTERFERENZA KV<->pesi: nbufs=%d (=layer), peso %d MB/layer x %d buffer distinti\n",
               nbufs, wmb, nwb);
        printf("#   memoria a Nk=16100: ~%.2f GB (KV) + ~%.2f GB (pesi) = ~%.2f GB  [nessun modello caricato]\n",
               (nbufs * 2.0 * 16100 * nkv * hd * 2) / 1e9, (nwb * (double)wmb * 1e6) / 1e9,
               (nbufs * 2.0 * 16100 * nkv * hd * 2 + nwb * (double)wmb * 1e6) / 1e9);
        printf("%8s %10s %10s %10s %10s %9s %11s %12s\n",
               "Nk", "sdpa_ms", "w_ms", "sum_ms", "both_ms", "x_interf", "both_GB/s", "excess_tok");
        fflush(stdout);
        for (int i = 0; i < nnk; i++) {
            const int nk = nks[i];
            double a = 0.0, b = 0.0, c = 0.0;
            const int frc = bench_interf(nkv, nh, hd, nk, nbufs, wmb, nwb, freps, st, &a, &b, &c);
            if (frc != 0 || a <= 0.0 || b <= 0.0 || c <= 0.0) { printf("%8d  FAIL rc=%d\n", nk, frc); }
            else {
                const double bytes = 2.0 * (double)nkv * (double)nk * (double)hd * 2.0
                                   + (double)wmb * 1e6;              /* KV(K+V) + peso, per layer */
                const double sum = a + b;
                printf("%8d %10.4f %10.4f %10.4f %10.4f %9.3f %11.0f %12.2f\n",
                       nk, a, b, sum, c, c / sum, bytes / (c * 1e-3) / 1e9,
                       (c - sum) * (double)nbufs);
            }
            fflush(stdout);
        }
        mlx_array_free(q);
        mlx_stream_free(st);
        mlx_device_free(dev);
        return 0;
    }

    /* ── MODE C (invocazione separata: LLM_ATTN_PROBE_COLD=<nbufs>) ───────────────
     * Working set = nbufs cache DISTINTE (default 36 = i layer di Qwen3-8B), cosi' il
     * kernel non pu' rileggere dalla L2 come fa il bench warm qui sopra. */
    if (getenv("LLM_ATTN_PROBE_COLD") != NULL) {
        const int nbufs = atoi(getenv("LLM_ATTN_PROBE_COLD"));
        const int coldreps = (argc > 7) ? atoi(argv[7]) : 4;
        printf("# COLD working set: nbufs=%d (=%.2f GB di KV a Nk=16100), reps=%d\n",
               nbufs, (nbufs * 2.0 * 16100 * nkv * hd * 2) / 1e9, coldreps);
        printf("%8s %10s %9s %10s %9s %9s %9s\n",
               "Nk", "cold_ms", "cold_GB/s", "roof_ms", "roof_GB/s", "x_passi", "peakGB");
        fflush(stdout);
        for (int i = 0; i < nnk; i++) {
            const int nk = nks[i];
            const double kbytes = (double)nkv * (double)nk * (double)hd * 2.0;   /* solo K */
            double cms = 0.0, rms = 0.0, pgb = 0.0;
            const int crc = bench_cold(nkv, nh, hd, nk, nbufs, coldreps, st, &cms, &rms, &pgb);
            if (crc != 0 || cms <= 0.0 || rms <= 0.0) { printf("%8d  FAIL rc=%d\n", nk, crc); }
            else printf("%8d %10.3f %9.0f %10.3f %9.0f %9.2f %9.2f\n",
                        nk, cms, 2.0 * kbytes / (cms * 1e-3) / 1e9,
                        rms, kbytes / (rms * 1e-3) / 1e9,
                        /* x_passi = quante letture complete della KV il kernel paga davvero:
                         * 1,0 = ottimo (una passata al tetto), 2,0 = la KV viene riletta. */
                        cms / (2.0 * rms), pgb);
            fflush(stdout);
        }
        mlx_array_free(q);
        mlx_stream_free(st);
        mlx_device_free(dev);
        return 0;
    }

    printf("# attention decode roof probe  nh=%d nkv=%d hd=%d bf16  reps=%d  (timing BATCHED: submit ammortizzato)\n",
           nh, nkv, hd, reps);
    printf("# G=wrapper produzione llm_mlx_gqa_sdpa   R=call cruda mlx_fast_sdpa(mask=\"\")   W=wall 1-eval-per-call\n");
    printf("# B=vista da cap=ceil256 (layout REALE del decode)   C=vista da cap=pow2   roof=sum(K)\n");
    printf("%8s %9s %8s %9s %8s %9s %8s %9s %8s\n",
           "Nk", "G_ms", "G_GB/s", "R_ms", "R_GB/s", "W_ms", "W_GB/s", "roof_ms", "roof_GB/s");
    fflush(stdout);

    for (int i = 0; i < nnk; i++) {
        const int nk = nks[i];
        const double kv_bytes = 2.0 * (double)nkv * (double)nk * (double)hd * 2.0; /* K+V bf16 */
        const double k_bytes  = (double)nkv * (double)nk * (double)hd * 2.0;
        double g[3] = { 0, 0, 0 };   /* A,B,C — wrapper batched */
        double rraw = 0, rwall = 0, rms_roof = 0;
        bool ok = true;

        for (int variant = 0; variant < 3 && ok; variant++) {
            const int cap = (variant == 0) ? nk : (variant == 1) ? ceil256(nk) : pow2_at_least(nk);
            if (variant == 1 && cap == nk) { g[1] = g[0]; continue; }
            mlx_array k = mlx_array_new(), v = mlx_array_new();
            mlx_array parents[2] = { mlx_array_new(), mlx_array_new() };
            if (build_kv(&k, &v, nkv, nk, cap, hd, parents, st)) {
                g[variant] = bench_sdpa_mode(q, k, v, hd, reps, st, /*raw=*/false, &ok);
                if (ok && variant == 0) {
                    rraw = bench_sdpa_mode(q, k, v, hd, reps, st, /*raw=*/true, &ok);
                    if (ok) rwall = bench_sdpa(q, k, v, hd, reps, st, &ok);
                    if (ok) rms_roof = bench_readroof(k, reps, st, &ok);
                }
            } else { ok = false; }
            mlx_array_free(k); mlx_array_free(v);
            mlx_array_free(parents[0]); mlx_array_free(parents[1]);
        }

        if (!ok) { printf("%8d  FAIL (path rifiutato: vedere messaggio mlx sopra)\n", nk); }
        else {
            const double gbs = (g[0] > 0) ? kv_bytes / (g[0] * 1e-3) / 1e9 : 0;
            const double rbs = (rraw > 0) ? kv_bytes / (rraw * 1e-3) / 1e9 : 0;
            const double wbs = (rwall > 0) ? kv_bytes / (rwall * 1e-3) / 1e9 : 0;
            const double ofs = (rms_roof > 0) ? k_bytes / (rms_roof * 1e-3) / 1e9 : 0;
            printf("%8d %9.3f %8.0f %9.3f %8.0f %9.3f %8.0f %9.3f %8.0f\n",
                   nk, g[0], gbs, rraw, rbs, rwall, wbs, rms_roof, ofs);
            /* B/C in riga separata: stesso dato, solo layout diverso */
            printf("         B(ceil256)=%.3f ms  C(pow2)=%.3f ms  ->  delta_vs_A=%+.1f%% / %+.1f%%\n",
                   g[1], g[2],
                   g[0] > 0 ? 100.0 * (g[1] - g[0]) / g[0] : 0.0,
                   g[0] > 0 ? 100.0 * (g[2] - g[0]) / g[0] : 0.0);
        }
        fflush(stdout);
    }
    mlx_array_free(q);
    mlx_stream_free(st);
    mlx_device_free(dev);
    return 0;
}
/* Riga completa di link (dal build/ gia' configurato, niente ri-build della dylib condivisa):
 *   cd native/llmengine_mlx && /usr/bin/c++ -std=c++20 -O2 -Wno-unused-function \
 *     -I. -I../../libs/mlx-deps/mlx-c-src -I../../libs/mlx-deps/mlx-src \
 *     -Ibuild/_deps/metal_cpp-src/include \
 *     llm_mlx_attn_roof_probe.c build/_deps/mlx_c-build/libmlxc.a \
 *     build/_deps/mlx-build/libmlx.a build/_deps/cjson-build/libcjson.a \
 *     build/_deps/mlx-build/mlx/io/libgguflib.a build/jaccl/libjaccl.a \
 *     -framework Metal -framework Foundation -framework QuartzCore -framework Accelerate \
 *     -o /tmp/llm_attn_roof_probe
 * (il file e' C: -x c con il driver c++ per il link, oppure compilare con cc e linkare c++.) */
