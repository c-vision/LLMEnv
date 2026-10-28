#!/usr/bin/env python3
"""kvseg_refactor.py — completa il refactor di GMB_KVSEG (#61) lasciato a metà.

Problema: `llm_mlx_kvseg_step` e' definita PRIMA di `struct GmbMlxModel` ma usa
`m->cfg`/`m->stream` (tipo incompleto) -> 20 errori di compilazione. Inoltre chiama
`llm_mlx_kvseg_flush(m, layer_idx)` con la firma vecchia (ora e' `(stream,nkv,hd,layer)`).

Fix minimo (NON sposta 277 righe): la funzione prende `stream/nkv/hd` espliciti.
Aggiorna: firma + guardie + call flush + forward decl + chiamante. Le `m->stream`
dentro il corpo della funzione diventano `stream`.

Uso: python3 tools/kvseg_refactor.py <path-gmb_host_ai_mlx.c>
Idempotente: se la firma nuova e' gia' presente, esce senza toccare nulla.
"""
import sys

NEW_SIG = "static bool llm_mlx_kvseg_step(mlx_stream stream, int nkv, int hd, int layer_idx,"
OLD_SIG = "static bool llm_mlx_kvseg_step(GmbMlxModel* m, int layer_idx,"

OLD_HEAD = """static bool llm_mlx_kvseg_step(GmbMlxModel* m, int layer_idx,
                               int cache_offset, int n_tokens,
                               mlx_array k_rope, mlx_array v_t,
                               mlx_array* k_cache_out, mlx_array* v_cache_out,
                               mlx_array* k_piece1, mlx_array* v_piece1,
                               mlx_array* k_piece2, mlx_array* v_piece2,
                               int* n_pieces) {
    if (m == NULL || k_piece1 == NULL || v_piece1 == NULL || n_pieces == NULL) return false;
    if (layer_idx < 0 || layer_idx >= MAX_KVSEG_LAYERS) return false;
    if (k_cache_out == NULL || v_cache_out == NULL) return false;
    if ((*k_cache_out).ctx == NULL || (*v_cache_out).ctx == NULL) return false;
    const int nkv = m->cfg.n_kv_heads, hd = m->cfg.head_dim;
    if (nkv <= 0 || hd <= 0 || cache_offset < 0 || n_tokens != 1) return false;"""

NEW_HEAD = """static bool llm_mlx_kvseg_step(mlx_stream stream, int nkv, int hd, int layer_idx,
                               int cache_offset, int n_tokens,
                               mlx_array k_rope, mlx_array v_t,
                               mlx_array* k_cache_out, mlx_array* v_cache_out,
                               mlx_array* k_piece1, mlx_array* v_piece1,
                               mlx_array* k_piece2, mlx_array* v_piece2,
                               int* n_pieces) {
    if (k_piece1 == NULL || v_piece1 == NULL || n_pieces == NULL) return false;
    if (layer_idx < 0 || layer_idx >= MAX_KVSEG_LAYERS) return false;
    if (k_cache_out == NULL || v_cache_out == NULL) return false;
    if ((*k_cache_out).ctx == NULL || (*v_cache_out).ctx == NULL) return false;
    if (nkv <= 0 || hd <= 0 || cache_offset < 0 || n_tokens != 1) return false;"""

OLD_FWD = """static bool llm_mlx_kvseg_step(GmbMlxModel* m, int layer_idx, int cache_offset, int n_tokens,
                               mlx_array k_rope, mlx_array v_t,"""
NEW_FWD = """static bool llm_mlx_kvseg_step(mlx_stream stream, int nkv, int hd, int layer_idx, int cache_offset, int n_tokens,
                               mlx_array k_rope, mlx_array v_t,"""

OLD_CALL = """            if (llm_mlx_kvseg_step(m, layer_idx, cache_offset, n_tokens,
                                   k_rope, v_t, k_cache, v_cache,
                                   &seg_k1, &seg_v1, &seg_k2, &seg_v2, &seg_pieces)) {"""
NEW_CALL = """            if (llm_mlx_kvseg_step(m->stream, nkv, hd, layer_idx, cache_offset, n_tokens,
                                   k_rope, v_t, k_cache, v_cache,
                                   &seg_k1, &seg_v1, &seg_k2, &seg_v2, &seg_pieces)) {"""


def main():
    if len(sys.argv) != 2:
        print("uso: kvseg_refactor.py <path>", file=sys.stderr)
        return 2
    p = sys.argv[1]
    src = open(p, encoding="utf-8").read()

    if NEW_SIG in src and "llm_mlx_kvseg_flush(m, layer_idx)" not in src:
        print("gia' refactorato: nessuna modifica")
        return 0

    for old, new, name in ((OLD_HEAD, NEW_HEAD, "head"), (OLD_FWD, NEW_FWD, "fwd"),
                           (OLD_CALL, NEW_CALL, "call")):
        if old not in src:
            print(f"ERRORE: blocco {name} non trovato", file=sys.stderr)
            return 1
        src = src.replace(old, new, 1)

    if "llm_mlx_kvseg_flush(m, layer_idx)" not in src:
        print("ERRORE: call flush non trovata", file=sys.stderr)
        return 1
    src = src.replace("llm_mlx_kvseg_flush(m, layer_idx)",
                      "llm_mlx_kvseg_flush(stream, nkv, hd, layer_idx)")

    start = src.index(NEW_SIG)
    end = src.index("/* GMB_KVSEG fine")
    body = src[start:end].replace("m->stream", "stream")
    src = src[:start] + body + src[end:]

    open(p, "w", encoding="utf-8").write(src)
    print("refactor applicato")
    return 0


if __name__ == "__main__":
    sys.exit(main())
