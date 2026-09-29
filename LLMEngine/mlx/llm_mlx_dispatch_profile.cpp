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
    long long total = 0;
    long long bytes_total = 0;
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
        if (name == "AsType" || name == "Sigmoid" || name == "Squeeze") {
            const std::vector<mlx::core::array>& ai = a->inputs();
            std::string src = ai.empty() ? std::string("(none)")
                          : (ai[0].has_primitive() ? std::string(ai[0].primitive().name())
                                                   : std::string("leaf"));
            if (name == "AsType") d.astype_src[src + " ->" + shp] += 1;
            else if (name == "Squeeze") d.sq_src[src + " ->" + shp] += 1;
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
        d.on = 0;
    }
    return d.on;
}

void llm_mlx_dp_reset(void) {
    if (!llm_mlx_dp_enabled()) return;
    Profile& d = p();
    d.hist.clear();
    d.by_shape.clear();
    d.total = 0;
    d.bytes_total = 0;
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
    if (seconds > 0.0 && total > 0) {
        fprintf(stderr, " ns/nodo=%.0f", seconds * 1e9 / (double)total);
    }
    if (seconds > 0.0 && d.bytes_total > 0) {
        /* banda DOMANDATA dai soli output: GB/s = MB_totali / secondi. Non e' la banda del
         * modello (manca il letto) e non e' la banda raggiunta dal chip: e' il denominatore che
         * permette di dire "questa famiglia di cast chiede da sola X GB/s", cioe' di convertire
         * un conteggio di nodi in una quantita' di traffico confrontabile col roofline. */
        fprintf(stderr, " GB/s_scritto=%.1f", (double)d.bytes_total / 1e9 / seconds);
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
    for (size_t i = 0; i < lim; i++) {
        fprintf(stderr, "LLM_DISPATCH[%s]:   %-24s %8lld  %6.1f/token  %9.2f MB/token  max %.2f MB %s\n",
                tag ? tag : "", rows[i].first.c_str(), rows[i].second.nodes,
                (double)rows[i].second.nodes / (double)steps,
                (double)rows[i].second.bytes / (double)steps / 1048576.0,
                (double)rows[i].second.max_bytes / 1048576.0,
                rows[i].second.max_shape.c_str());
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
