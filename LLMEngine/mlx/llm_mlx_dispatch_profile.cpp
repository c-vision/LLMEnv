/* llm_mlx_dispatch_profile.cpp — implementazione di G0 (vedi il rationale nel header).
 *
 * E' C++ perche' deve leggere il grafo di MLX con l'API C++ (mlx::core::array::inputs(),
 * siblings(), has_primitive(), primitive().name(), primitive_id()): mlx-c non espone NESSUN
 * accesso al grafo in ABI C. L'ABI esportata resta C (llm_mlx_dispatch_profile.h), quindi
 * llmengine_mlx.c continua a essere C puro.
 *
 * Nota sul perche' NON si usa mlx::core::export_to_dot / print_graph: mlx-c v0.6.0 non esporta
 * quei binding e i sorgenti MLX li mettono dietro il target `mlx` che qui e' gia' linkato
 * (statico dentro libmlxc.a), ma le loro classi di simboli non sono dichiarate in ABI C --
 * un riuso dall'esterno sarebbe un secondo ABI break. La camminata qui sotto usa SOLO metodi
 * inline pubblici del header array.h: nessun simbolo nuovo da linkare.
 */
#include "llm_mlx_dispatch_profile.h"

#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mlx/array.h"
#include "mlx/primitives.h"
#include "mlx/c/memory.h" /* mlx_get_active_memory/cache/peak/limit: stato dell'allocatore Metal */

namespace {

struct Fam {
    long long nodes = 0;
    /* somma dei `nbytes()` degli OUTPUT della primitiva: proxy del materiale scritto. Non e'
     * il traffico totale (non conta le letture) e su una primitiva a output multipli conta
     * solo l'output visitato per primo -- basta pero' a distinguere il cast di un vettore
     * (8 KB) dal cast di un peso (14 MB), che e' la domanda a cui il solo conteggio di nodi
     * non risponde. */
    long long bytes = 0;
    /* somma dei `nbytes()` degli INPUT (pesi + KV + attivazioni LETTI): e' il numeratore di
     * EFFECTIVE_WEIGHT_BW, il traffico REALE. Complementare a `bytes` (gli output scritti):
     * per le op in-place (SliceUpdate KV) `bytes` sovrastima, `bytes_in` no. */
    long long bytes_in = 0;
    /* Il PIU' GROSSO output visto per questa famiglia, con la sua forma: e' cio' che risponde
     * "quale operando e'" senza doverlo indovinare dal conteggio. */
    long long max_bytes = 0;
    std::string max_shape;
};

struct Profile {
    bool flag_read = false;
    bool on = false;
    std::unordered_map<std::string, Fam> hist;
    /* STESSA contabilita', ma con la forma nella chiave: risponde "quale tensore e'" invece di
     * "quanti nodi", che e' l'unico modo di distinguere 336 cast da 48 MB (un peso per layer,
     * o uno stacked da 8 esperti) da 1519 cast da 8 KB (vettori di attivazione). */
    std::unordered_map<std::string, Fam> by_shape;
    /* AsType: chi e' l'INPUT della primitiva (nome primitiva o "leaf") + forma output. Serve a
     * rispondere "quale cast e' dispatching per token" invece di indovinarlo dal solo conteggio. */
    std::unordered_map<std::string, long long> astype_src;
    std::unordered_map<std::string, long long> sig_src;
    std::unordered_map<std::string, long long> sq_src;
    std::unordered_map<std::string, long long> reshape_src;
    long long total = 0;
    long long bytes_total = 0;
    long long bytes_in_total = 0;
    long long last_new = 0;
    /* attribuzione del tempo di passo: CPU per buildare il grafo vs dispatch+lettura del token */
    double build_sec = 0;
    double eval_sec = 0;
    /* stabilita' passo-per-passo: un contatore che oscilla non e' un costo per token. */
    long long step_min = -1;
    long long step_max = 0;
};

/* Insieme di dedup di UN SINGOLO passo. NON e' cumulativo tra i passi perche' misurato: i
 * nodi del passo k non esistono piu' quando parte il passo k+1 (l'eval ne azzera la
 * primitiva, le Primitive muoiono e malloc restituisce gli STESSI indirizzi al passo dopo) --
 * con il set cumulativo il primo passo contava 705 nodi e tutti i successivi 0. Dentro il
 * passo la dedup serve davvero: lo stesso nodo e' raggiungibile da piu' radici (le ~2·L+1
 * radici del passo condividono l'attaccatura con la cache), e una primitiva a output multipli
 * va contata UNA (dispatch = primitiva, non array). */
struct StepSeen {
    std::unordered_set<uintptr_t> prim;
    std::unordered_set<uintptr_t> leaf;
};

Profile& p() {
    static Profile inst;
    return inst;
}

/* MoE: config del modello per correggere `MB_in` dei GatherQMM (default 0 = nessun MoE / non noto). */
static int g_dp_n_experts = 0;
static int g_dp_top_k = 0;

/* Banda DRAM di riferimento per il ROOFLINE (GB/s). I byte letti sono DETERMINISTICI (walk a
 * build-time), quindi `bytes_in/BW` = il floor di tempo per-token di ogni funzione, INDIPENDENTE
 * dal carico della macchina: e' la parte "senza lanciare il modello". Override: LLM_DISPATCH_BW. */
/* Speculazione: token accettati accumulati + passate, per la velocita' reale per token. */
static long long g_dp_spec_tok = 0;
static long long g_dp_spec_pass = 0;

static double dp_ceiling_gbs(void) {
    static double v = -1.0;
    if (v < 0.0) {
        const char* e = getenv("LLM_DISPATCH_BW");
        v = (e != nullptr && atof(e) > 0.0) ? atof(e) : 540.0;
    }
    return v;
}

/* DFS iterativa da una radice: conta i nodi con primitiva NUOVI per questo passo e li
 * accumula nell'istogramma. Iterativa per scelta: il grafo di un passo di decode e' profondo
 * ~(n_layer × nodi-per-layer) e una ricorsione su un modello 96-layer supererebbe lo stack di
 * un thread non principale. */
long long walk(const mlx::core::array* root, StepSeen& seen, long long& bytes_out) {
    if (root == nullptr) return 0;
    Profile& d = p();
    long long counted = 0;
    std::vector<const mlx::core::array*> st;
    st.push_back(root);
    while (!st.empty()) {
        const mlx::core::array* a = st.back();
        st.pop_back();
        if (!a->has_primitive()) {
            /* foglia (peso, scalare host, o nodo gia' dispatchato): nessun dispatch da contare;
             * la dedup evita di riattraversarla per ogni consumatore */
            if (!seen.leaf.insert(a->id()).second) continue;
            const std::vector<mlx::core::array>& ins = a->inputs();
            for (const mlx::core::array& in : ins) st.push_back(&in);
            continue;
        }
        if (!seen.prim.insert(a->primitive_id()).second) continue; /* gia' contato in questo passo */
        counted += 1;
        const long long nb = (long long)a->nbytes();
        bytes_out += nb;
        /* forma + larghezza del dtype: distingue "cast di un vettore" da "cast di un peso" e
         * dice se il risultato e' 16-bit (itemsize 2) o 32-bit (4). */
        const mlx::core::Shape& sh = a->shape();
        std::string shp = "(";
        for (size_t i = 0; i < sh.size(); i++) {
            if (i) shp += ",";
            shp += std::to_string((long long)sh[i]);
        }
        shp += ") x";
        shp += std::to_string((int)mlx::core::size_of(a->dtype()));
        const std::string name(a->primitive().name());
        if (name == "AsType" || name == "Sigmoid" || name == "Squeeze" || name == "Reshape") {
            const std::vector<mlx::core::array>& ai = a->inputs();
            std::string src = ai.empty() ? std::string("(none)")
                          : (ai[0].has_primitive() ? std::string(ai[0].primitive().name())
                                                   : std::string("leaf"));
            if (name == "AsType") d.astype_src[src + " ->" + shp] += 1;
            else if (name == "Squeeze") d.sq_src[src + " ->" + shp] += 1;
            else if (name == "Reshape") d.reshape_src[src + " ->" + shp] += 1;
            else                  d.sig_src[src + " ->" + shp] += 1;
        }
        Fam& f = d.hist[name];
        f.nodes += 1;
        f.bytes += nb;
        if (nb > f.max_bytes) { f.max_bytes = nb; f.max_shape = shp; }
        Fam& g = d.by_shape[name + " " + shp];
        g.nodes += 1;
        g.bytes += nb;
        if (nb > g.max_bytes) { g.max_bytes = nb; g.max_shape = shp; }
        /* BYTE LETTI = somma degli `nbytes()` degli input (pesi+KV+attivazioni): per un
         * QuantizedMatmul e' ~il peso quantizzato + le scale + l'attivazione, cioe' il traffico
         * reale del nodo. Una primitiva e' UN dispatch (dedup sopra), quindi ogni input e'
         * contato una volta per passo -> `bytes_in/steps` = EFFECTIVE_WEIGHT_BW per token. */
        long long nbi = 0;
        const std::vector<mlx::core::array>& ains = a->inputs();
        /* MoE: un nodo `GatherQMM`/`GatherMM` LEGGE logicamente l'INTERO banco esperti (l'input con
         * piu' byte), ma in memoria tocca solo gli esperti ATTIVI. Il routing e' runtime, quindi la
         * frazione si prende dal config (`n_experts`/`top_k`, via llm_mlx_dp_set_moe): senza di esso
         * `MB_in` dei MoE resta un limite superiore (es. glm4_moe ~57 GB/token, GB/s_letto >> ceiling). */
        const bool is_gather = (name == "GatherQMM" || name == "GatherMM");
        double gather_frac = 1.0;
        size_t gather_bank = (size_t)-1;
        if (is_gather && !ains.empty()) {
            long long best = -1;
            for (size_t j = 0; j < ains.size(); j++) {
                const long long nb = (long long)ains[j].nbytes();
                if (nb > best) { best = nb; gather_bank = j; }
            }
            if (g_dp_n_experts > 0 && g_dp_top_k > 0) {
                gather_frac = (double)g_dp_top_k / (double)g_dp_n_experts;
                if (gather_frac > 1.0) gather_frac = 1.0;
            }
        }
        for (size_t ii = 0; ii < ains.size(); ii++) {
            const mlx::core::array& in = ains[ii];
            /* Ogni array LETTO una volta per passo (in mlx i pesi NON sono leaf: hanno una
             * primitiva, quindi il dedup e' per `id()` su TUTTI gli input, non solo le foglie).
             * Cosi' `bytes_in` = l'insieme di lettura unico del passo = pesi + KV (+ le
             * attivazioni condivise una volta) = numeratore di EFFECTIVE_WEIGHT_BW. */
            if (!seen.leaf.insert(in.id()).second) continue;
            bool is_bank = false;
            if (is_gather) {
                if (g_dp_n_experts > 0) {
                    const mlx::core::Shape& ish = in.shape();
                    is_bank = (ish.size() >= 1 && (long long)ish[0] == (long long)g_dp_n_experts);
                } else {
                    is_bank = (ii == gather_bank);
                }
            }
            const double scale = is_bank ? gather_frac : 1.0;
            nbi += (long long)((double)in.nbytes() * scale);
        }
        f.bytes_in += nbi; g.bytes_in += nbi; d.bytes_in_total += nbi;
        /* Non si spingono i siblings: in mlx::core::array::make_arrays tutti gli output di
         * una primitiva condividono STESSO vettore di input, e la dedup e' per primitiva. */
        const std::vector<mlx::core::array>& ins = a->inputs();
        for (const mlx::core::array& in : ins) st.push_back(&in);
    }
    return counted;
}

} /* namespace */

extern "C" {

bool llm_mlx_dp_enabled(void) {
    Profile& d = p();
    if (!d.flag_read) {
        d.flag_read = true;
        d.on = (getenv("LLM_DISPATCH") != nullptr);
    }
    return d.on;
}

void llm_mlx_dp_set_moe(int n_experts, int top_k) {
    g_dp_n_experts = (n_experts > 0) ? n_experts : 0;
    g_dp_top_k = (top_k > 0) ? top_k : 0;
}

void llm_mlx_dp_spec_add(int accepted_tokens) {
    if (accepted_tokens <= 0) return;
    g_dp_spec_tok += accepted_tokens;
    g_dp_spec_pass += 1;
}

void llm_mlx_dp_reset(void) {
    if (!llm_mlx_dp_enabled()) return;
    Profile& d = p();
    d.hist.clear();
    d.by_shape.clear();
    /* Le mappe per-SORGENTE (Reshape/AsType/Squeeze/Sigmoid) vanno azzerate come l'istogramma:
     * senza, si accumulano prefill+decode e il per-SORGENTE di un report e' falsato. */
    d.astype_src.clear();
    d.sig_src.clear();
    d.sq_src.clear();
    d.reshape_src.clear();
    d.total = 0;
    d.bytes_total = 0;
    d.bytes_in_total = 0;
    /* NB: i contatori spec NON si azzerano qui (il generate_ngram annidato richiama il reset e
     * cancellerebbe l'amortizzazione accumulata): li azzera il report dopo averli stampati. */
    d.build_sec = 0;
    d.eval_sec = 0;
    d.last_new = 0;
    d.step_min = -1;
    d.step_max = 0;
}

void llm_mlx_dp_count(const mlx_array* roots, int n_roots) {
    if (!llm_mlx_dp_enabled() || roots == nullptr || n_roots <= 0) return;
    Profile& d = p();
    StepSeen seen; /* insieme NUOVO a ogni passo: vedi il commento su StepSeen */
    long long step = 0;
    long long step_bytes = 0;
    for (int i = 0; i < n_roots; i++) {
        if (roots[i].ctx != nullptr) {
            step += walk(static_cast<const mlx::core::array*>(roots[i].ctx), seen, step_bytes);
        }
    }
    d.total += step;
    d.bytes_total += step_bytes;
    d.last_new = step;
    if (d.step_min < 0 || step < d.step_min) d.step_min = step;
    if (step > d.step_max) d.step_max = step;
}

void llm_mlx_dp_stage(double build_sec, double eval_sec) {
    if (!llm_mlx_dp_enabled()) return;
    Profile& d = p();
    d.build_sec += build_sec;
    d.eval_sec += eval_sec;
}

void llm_mlx_dp_memory(const char* where) {
    if (!llm_mlx_dp_enabled()) return;
    /* `attivi` incollato a `limite` = il modello tocca il tetto del working set: il tempo non e'
     * piu' tempo di kernel ma di allocatore (eviction + ricalcolo). E' la differenza tra "perde
     * perche' scrive tanto" e "perde perche' non ha dove mettere i buffer". */
    size_t act = 0, cac = 0, lim = 0, pk = 0;
    mlx_get_active_memory(&act);
    mlx_get_cache_memory(&cac);
    mlx_get_memory_limit(&lim);
    mlx_get_peak_memory(&pk);
    fprintf(stderr, "LLM_DISPATCH[%s]:   metal GB attivi=%.2f cache=%.2f picco=%.2f limite=%.2f\n",
            where ? where : "", (double)act / 1e9, (double)cac / 1e9, (double)pk / 1e9,
            (double)lim / 1e9);
}

void llm_mlx_dp_report(const char* tag, int steps, int layers, double seconds, int top_k) {
    if (!llm_mlx_dp_enabled()) return;
    Profile& d = p();
    const long long total = d.total;
    if (steps < 1) steps = 1;
    const double per_token = (double)total / (double)steps;
    fprintf(stderr, "LLM_DISPATCH[%s]: nodi_totali=%lld token=%d nodi/token=%.1f",
            tag ? tag : "", total, steps, per_token);
    if (layers > 0) {
        fprintf(stderr, " nodi/layer=%.1f", per_token / (double)layers);
    }
    fprintf(stderr, " MB/token=%.2f", (double)d.bytes_total / (double)steps / 1048576.0);
    fprintf(stderr, " MB_in/token=%.2f", (double)d.bytes_in_total / (double)steps / 1048576.0);
    if (seconds > 0.0 && total > 0) {
        fprintf(stderr, " ns/nodo=%.0f", seconds * 1e9 / (double)total);
    }
    if (seconds > 0.0 && d.bytes_in_total > 0) {
        /* EFFECTIVE_WEIGHT_BW: byte LETTI (pesi+KV+attivazioni) / secondo. E' il numero che il
         * roofline confronta (~540 GB/s raggiungibili su M5 Max). */
        fprintf(stderr, " GB/s_letto=%.1f", (double)d.bytes_in_total / 1e9 / seconds);
    }
    if (seconds > 0.0 && d.bytes_total > 0) {
        /* banda DOMANDATA dai soli output: GB/s = MB_totali / secondi. Non e' la banda del
         * modello (manca il letto) e non e' la banda raggiunta dal chip: e' il denominatore che
         * permette di dire "questa famiglia di cast chiede da sola X GB/s", cioe' di convertire
         * un conteggio di nodi in una quantita' di traffico confrontabile col roofline. */
        fprintf(stderr, " GB/s_scritto=%.1f", (double)d.bytes_total / 1e9 / seconds);
    }
    {
        /* ROOFLINE (macchina-indipendente): floor di tempo per-token = byte-LETTI / BW_ceil.
         * E' il minimo che la memoria impone a questo passo; `meas - floor` = overhead (launch,
         * gap di banda). Vale anche a macchina carica perche' i byte non dipendono dal carico. */
        const double bw = dp_ceiling_gbs();
        const double floor_us = ((double)d.bytes_in_total / 1e9 / bw) / (double)steps * 1e6;
        fprintf(stderr, " floor_us/tok=%.0f@%.0fGBs", floor_us, bw);
        if (seconds > 0.0) {
            const double meas_us = seconds / (double)steps * 1e6;
            const double overhead = meas_us - floor_us;
            fprintf(stderr, " meas_us/tok=%.0f overhead=%.0f%%",
                    meas_us, 100.0 * overhead / (meas_us > 1e-9 ? meas_us : 1.0));
        }
        if (g_dp_spec_pass > 0) {
            /* Spec: il floor e' per PASSATA; diviso per i token accettati/passata da la velocita'
             * reale per token (l'amortizzazione dei pesi). Deterministico. */
            const double tpp = (double)g_dp_spec_tok / (double)g_dp_spec_pass;
            fprintf(stderr, " spec_tok/pass=%.2f floor_us/tokREAL=%.0f",
                    tpp, floor_us / (tpp > 1e-9 ? tpp : 1.0));
        }
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "LLM_DISPATCH[%s]:   diag nodi_ultimo_passo=%lld min/passo=%lld max/passo=%lld\n",
            tag ? tag : "", d.last_new, d.step_min, d.step_max);
    if (d.build_sec > 0.0 || d.eval_sec > 0.0) {
        const double bp = d.build_sec / (double)steps * 1000.0;
        const double ep = d.eval_sec / (double)steps * 1000.0;
        fprintf(stderr, "LLM_DISPATCH[%s]:   tempo ms/passo build=%.1f eval=%.1f (build = %.0f%% del totale)\n",
                tag ? tag : "", bp, ep, 100.0 * d.build_sec / (d.build_sec + d.eval_sec));
    }
    {
        /* Stato dell'allocatore Metal al momento del report: vedi llm_mlx_dp_memory. */
        llm_mlx_dp_memory(tag);
    }

    if (top_k <= 0) top_k = 12;
    /* selection-sort parziale: top_k elementi su una mappa che puo' avere poche decine di
     * chiavi; niente qsort su callback, niente allocazione extra oltre al vettore righe. */
    auto top_by = [](std::vector<std::pair<std::string, Fam>>& rows, size_t lim, bool by_bytes) {
        const size_t n = rows.size();
        const auto val = [&](const Fam& f) { return by_bytes ? f.bytes : f.nodes; };
        for (size_t i = 0; i < lim; i++) {
            size_t best = i;
            for (size_t j = i + 1; j < n; j++) {
                if (val(rows[j].second) > val(rows[best].second)) best = j;
            }
            if (best != i) std::swap(rows[i], rows[best]);
        }
    };
    std::vector<std::pair<std::string, Fam>> rows;
    rows.reserve(d.hist.size());
    for (const auto& kv : d.hist) rows.push_back(kv);
    size_t lim = (size_t)top_k < rows.size() ? (size_t)top_k : rows.size();
    top_by(rows, lim, false);
    const double rep_bw = dp_ceiling_gbs();
    for (size_t i = 0; i < lim; i++) {
        /* `us_floor` = floor roofline della famiglia (byte-LETTI / BW_ceil, per token):
         * macchina-indipendente, dice quanto la memoria IMPONE a quella famiglia. */
        const double us_floor = (double)rows[i].second.bytes_in / (double)steps / 1e9 / rep_bw * 1e6;
        fprintf(stderr, "LLM_DISPATCH[%s]:   %-24s %8lld  %6.1f/token  %9.2f MBout %9.2f MBin/token %7.1f usfloor  max %.2f MB %s\n",
                tag ? tag : "", rows[i].first.c_str(), rows[i].second.nodes,
                (double)rows[i].second.nodes / (double)steps,
                (double)rows[i].second.bytes / (double)steps / 1048576.0,
                (double)rows[i].second.bytes_in / (double)steps / 1048576.0,
                us_floor,
                (double)rows[i].second.max_bytes / 1048576.0,
                rows[i].second.max_shape.c_str());
    }
    {
        /* RIPARTIZIONE pesi vs altro (L5 Lightning: weight_bytes vs KV/activation): i byte LETTI
         * dai nodi che leggono PESI (Matmul/QuantizedMatmul/GatherQMM/GatherMM) vs il resto
         * (KV, attivazioni, layout). Deterministico. Dice se il decode e' weight-bound o no. */
        long long w_nodes = 0, o_nodes = 0, w_bin = 0, o_bin = 0;
        for (const auto& kv : d.hist) {
            const std::string& nm = kv.first;
            const bool is_w = (nm == "QuantizedMatmul" || nm == "Matmul" ||
                               nm == "GatherQMM" || nm == "GatherMM" ||
                               nm == "GatherQuantizedMatmul" || nm == "Dequantize");
            if (is_w) { w_nodes += kv.second.nodes; w_bin += kv.second.bytes_in; }
            else      { o_nodes += kv.second.nodes; o_bin += kv.second.bytes_in; }
        }
        fprintf(stderr, "LLM_DISPATCH[%s]:   -- PESI vs ALTRO -- pesi: %.1f nodi/tok, %.2f MBin/tok | altro: %.1f nodi/tok, %.2f MBin/tok\n",
                tag ? tag : "",
                (double)w_nodes / (double)steps, (double)w_bin / (double)steps / 1048576.0,
                (double)o_nodes / (double)steps, (double)o_bin / (double)steps / 1048576.0);
    }

    /* TOP PER VOLUME: le forme con il byte-count maggiore. E' la lista che trova i bug, perche'
     * 1519 nodi non dicono nulla e "336 x AsType (8,6144,256) = 16 GB/token" dice tutto. */
    rows.clear();
    rows.reserve(d.by_shape.size());
    for (const auto& kv : d.by_shape) rows.push_back(kv);
    lim = (size_t)top_k < rows.size() ? (size_t)top_k : rows.size();
    top_by(rows, lim, true);
    fprintf(stderr, "LLM_DISPATCH[%s]:   -- TOP %zu PER VOLUME (forma = operando) --\n",
            tag ? tag : "", lim);
    for (size_t i = 0; i < lim; i++) {
        fprintf(stderr, "LLM_DISPATCH[%s]:   %-52s %6lld nodi %9.2f MB/token\n",
                tag ? tag : "", rows[i].first.c_str(), rows[i].second.nodes,
                (double)rows[i].second.bytes / (double)steps / 1048576.0);
    }

    /* AsType: sorgente (input) -> forma/dtype output. Trova i cast evitabili (mlx_lm ne fa 0). */
    if (!d.astype_src.empty()) {
        std::vector<std::pair<std::string, long long>> av(d.astype_src.begin(), d.astype_src.end());
        for (size_t i = 0; i < av.size(); i++) {
            size_t best = i;
            for (size_t j = i + 1; j < av.size(); j++) if (av[j].second > av[best].second) best = j;
            if (best != i) std::swap(av[i], av[best]);
        }
        fprintf(stderr, "LLM_DISPATCH[%s]:   -- AsType per SORGENTE (%zu) --\n", tag ? tag : "", av.size());
        for (size_t i = 0; i < av.size() && i < 24; i++) {
            fprintf(stderr, "LLM_DISPATCH[%s]:   %-52s %6lld  %6.1f/token\n",
                    tag ? tag : "", av[i].first.c_str(), av[i].second,
                    (double)av[i].second / (double)steps);
        }
    }
    /* Reshape: sorgente (input) -> forma. Ogni Reshape e' un dispatch in piu' (mlx lo fa spesso
     * inline): trovare le forme evitabili e' riduzione diretta dei nodi/token. */
    if (!d.reshape_src.empty()) {
        std::vector<std::pair<std::string, long long>> rv(d.reshape_src.begin(), d.reshape_src.end());
        for (size_t i = 0; i < rv.size(); i++) {
            size_t best = i;
            for (size_t j = i + 1; j < rv.size(); j++) if (rv[j].second > rv[best].second) best = j;
            if (best != i) std::swap(rv[i], rv[best]);
        }
        fprintf(stderr, "LLM_DISPATCH[%s]:   -- Reshape per SORGENTE (%zu) --\n", tag ? tag : "", rv.size());
        for (size_t i = 0; i < rv.size() && i < 16; i++) {
            fprintf(stderr, "LLM_DISPATCH[%s]:   %-52s %6lld  %6.1f/token\n",
                    tag ? tag : "", rv[i].first.c_str(), rv[i].second,
                    (double)rv[i].second / (double)steps);
        }
    }
    if (!d.sig_src.empty()) {
        std::vector<std::pair<std::string, long long>> sv(d.sig_src.begin(), d.sig_src.end());
        for (size_t i = 0; i < sv.size(); i++) {
            size_t best = i;
            for (size_t j = i + 1; j < sv.size(); j++) if (sv[j].second > sv[best].second) best = j;
            if (best != i) std::swap(sv[i], sv[best]);
        }
        fprintf(stderr, "LLM_DISPATCH[%s]:   -- Sigmoid per SORGENTE (%zu) --\n", tag ? tag : "", sv.size());
        for (size_t i = 0; i < sv.size() && i < 12; i++) {
            fprintf(stderr, "LLM_DISPATCH[%s]:   %-52s %6lld  %6.1f/token\n",
                    tag ? tag : "", sv[i].first.c_str(), sv[i].second,
                    (double)sv[i].second / (double)steps);
        }
    }
    if (!d.sq_src.empty()) {
        std::vector<std::pair<std::string, long long>> qv(d.sq_src.begin(), d.sq_src.end());
        for (size_t i = 0; i < qv.size(); i++) {
            size_t best = i;
            for (size_t j = i + 1; j < qv.size(); j++) if (qv[j].second > qv[best].second) best = j;
            if (best != i) std::swap(qv[i], qv[best]);
        }
        fprintf(stderr, "LLM_DISPATCH[%s]:   -- Squeeze per SORGENTE (%zu) --\n", tag ? tag : "", qv.size());
        for (size_t i = 0; i < qv.size() && i < 12; i++) {
            fprintf(stderr, "LLM_DISPATCH[%s]:   %-52s %6lld  %6.1f/token\n",
                    tag ? tag : "", qv[i].first.c_str(), qv[i].second,
                    (double)qv[i].second / (double)steps);
        }
    }
}

} /* extern "C" */
