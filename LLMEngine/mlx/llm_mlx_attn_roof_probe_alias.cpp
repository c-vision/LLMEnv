// Shim C++ del SOLO probe llm_mlx_attn_roof_probe (MODE I). Non entra nella build di
// produzione: viene compilato a mano insieme al probe (vedi tools/attn_decode_bench.sh).
//
// Domanda a cui risponde (Bugs/(open) - MLX-Decode-Costo-Contesto-KV-Attention.md §8.13(w)):
// perche' l'append della KV costi O(cap) invece di O(riga). Letto al sorgente:
//   mlx/backend/common/copy.h:25-47  set_copy_output_data -> SE is_donatable(in,out) e il
//     ctype e' Vector, l'output `copy_shared_buffer(in)` (NESSUNA copia); ALTRIMENTI
//     `out.set_data(malloc(in.data_size()*itemsize))` = la copia dell'INTERO buffer.
//   mlx/array.h:295  is_donatable() = array_desc_.use_count()==1 && data.use_count()==1.
//   mlx/backend/metal/indexing.cpp:731+  SliceUpdate::eval_gpu -> copy_gpu(in,out,Vector)
//     quando `in.flags().contiguous && in.size()==in.data_size()`.
// Quindi: se la DESTRAZIONE della riga parte da una FOGLIA FRESCA (descrittori e Data con
// use_count 1) che Pero' condivide il buffer MTL della cache, la donazione riesce SEMPRE e
// il costo dell'append diventa O(riga) senza dover materializzare nulla (niente drain).
// Le due strade per avere quella foglia, senza patchare MLX:
//   (1) C API `mlx_array_new_data_managed`  -> make_buffer = newBufferWithBytes = COPIA
//       (mlx/backend/metal/allocator.cpp:207) => scartata a lettura.
//   (2) costruttore pubblico `array(allocator::Buffer, Shape, Dtype, Deleter)`
//       (mlx/array.h:71-76) + accessore pubblico `array::buffer()` (mlx/array.h:352):
//       zero copie, deleter no-op perche' il buffer resta del proprietario originale.
// Questa riga di codice E' il candidato fix; qui viene misurata prima di toccare qualsiasi
// file condiviso.

#include <mlx/c/error.h>
#include <mlx/c/private/array.h>

#include <vector>

extern "C" {

// Costruisce in *out una FOGLIA (nessun primitivo, quindi gia' "available") di `shape` con
// il dtype di `src`, che condivide SENZA COPIARE il buffer di `src`.
// Ritorna 0 OK; 1 = src vuoto/sfiduciato; 2 = eccezione;
//         3 = src NON ANCORA VALUTATO: `array::buffer()` fa `array_desc_->data->buffer` e
//             su un nodo non valutato `data` e' nullptr -> SIGSEGV su indirizzo 0 (misurato:
//             exit 139 dentro eq_run quando la radice adottava l'output di slice_update).
//             Un output di slice_update NON puo' mai essere la sorgente della foglia: da qui
//             la need di un ANCORAGGIO separato (vedi llm_probe_holder_put).
// `diag`: [0]=offset di src, [1]=is_donatable di src, [2]=is_donatable della foglia,
//         [3]=data use_count della foglia, [4]=nbytes foglia vs src (1 se uguali).
static int leaf_from(
    mlx::core::array& a, mlx_array* out, const int* shape, int dim, long long* diag) {
  if (a.ndim() == 0 || a.data_shared_ptr() == nullptr) return 3;
  if (a.buffer().ptr() == nullptr) return 1;
  // La foglia che costruiamo parte dall'inizio del buffer: valida solo se la sorgente
  // e' un array contiguo con offset 0 (la cache radice della KV e' cosi').
  if (a.offset() != 0 || !a.flags().contiguous || !a.flags().row_contiguous) return 1;
  if (diag) { diag[0] = (long long)a.offset(); diag[1] = (long long)a.is_donatable(); }
  mlx::core::allocator::Buffer b = a.buffer();
  mlx::core::Shape sh(shape, shape + dim);
  mlx::core::array leaf(b, sh, a.dtype(), [](mlx::core::allocator::Buffer) {});
  mlx_array tmp = mlx_array_new_(std::move(leaf));
  if (diag) {
    mlx::core::array& L = mlx_array_get_(tmp);
    diag[2] = (long long)L.is_donatable();
    diag[3] = (long long)L.data_shared_ptr().use_count();
    diag[4] = (long long)(L.nbytes() == (size_t)a.nbytes() ? 1 : 0);
  }
  if (out->ctx == NULL) *out = tmp;
  else { mlx_array_set_(*out, mlx_array_get_(tmp)); mlx_array_free_(tmp); }
  return 0;
}

int llm_probe_alias_leaf(
    mlx_array* out, mlx_array src, const int* shape, int dim, long long* diag) {
  if (out == NULL || src.ctx == NULL) return 1;
  try {
    return leaf_from(mlx_array_get_(src), out, shape, dim, diag);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 2;
  }
}

/* ── ANCORAGGIO del buffer (arm 4 = MODE I reale) ─────────────────────────────────
 * `allocator::Buffer` e' solo un wrapper around a raw pointer (mlx/allocator.h:13-31):
 * chi possiede davvero la memoria e' il `Data` (mlx/array.h:231-246) che nel distruttore
 * invoca il deleter (`allocator::free` per i buffer allocati da MLX). La foglia alias ha
 * un deleter NO-OP, quindi il buffer resta vivo SOLO finche' vive il Data del suo
 * proprietario originale.
 * In produzione quell'ancora NON c'e': `llm_mlx_kv_append_inplace` sostituisce
 * `*k_cache` in place (mlx_slice_update_inplace) e il vecchio array viene rilasciato ->
 * l'append successivo (e ogni scrittura sulle foglie) finirebbe su un buffer rilasciato in
 * cache dal MetalAllocator (che LO RIUTILIZZA: corruzione silenziosa, non crash).
 * Lo slot qui sotto FA da ancora: conserva una COPIA dell'array valutato (quindi un
 * riferimento al suo Data) e le foglie si costruiscono da quella, mai dall'output non
 * valutato di slice_update. E' il pezzo di stato che il fix deve introdurre nel motore. */
static std::vector<mlx::core::array>* g_holders = nullptr;

// Mette un'ancora sullo slot; richiede `a` GIA' valutato. Ritorna lo slot (>=0) o -1.
long long llm_probe_holder_put(mlx_array a) {
  if (a.ctx == NULL) return -1;
  try {
    mlx::core::array& x = mlx_array_get_(a);
    if (x.data_shared_ptr() == nullptr) return -1;   // non valutato: niente buffer da ancorare
    if (!g_holders) g_holders = new std::vector<mlx::core::array>();
    g_holders->push_back(x);                          // copia = condivide il Data (ancora)
    return (long long)(g_holders->size() - 1);
  } catch (std::exception& e) { mlx_error(e.what()); return -1; }
}

// Foglia dal buffer ancorato in `slot`, con la forma `shape` che gli passiamo NOI
// (la capacita' viva della cache: puo' essere diversa da quella dell'ancora dopo un grow).
int llm_probe_alias_leaf_at(
    mlx_array* out, long long slot, const int* shape, int dim, long long* nbytes_out) {
  if (out == NULL || slot < 0 || !g_holders || (size_t)slot >= g_holders->size()) return 1;
  try {
    mlx::core::array& a = (*g_holders)[(size_t)slot];
    int rc = leaf_from(a, out, shape, dim, NULL);
    if (rc == 0 && nbytes_out) *nbytes_out = (long long)a.nbytes();
    return rc;
  } catch (std::exception& e) { mlx_error(e.what()); return 2; }
}

// 1 = l'ancora in `slot` punta lo STESSO buffer MTL di `a` (e' la verifica che dopo un
// grow l'ancora va ricaricata, altrimenti scriveremmo sulla memoria vecchia).
int llm_probe_holder_same_buffer(long long slot, mlx_array a) {
  if (slot < 0 || !g_holders || (size_t)slot >= g_holders->size()) return -1;
  if (a.ctx == NULL) return -1;
  try {
    mlx::core::array& x = mlx_array_get_(a);
    if (x.data_shared_ptr() == nullptr) return -1;
    return (*g_holders)[(size_t)slot].buffer().ptr() == x.buffer().ptr() ? 1 : 0;
  } catch (...) { return -1; }
}

void llm_probe_holder_clear(void) {
  if (g_holders) { g_holders->clear(); }
}

// 1 = la donazione riesce su `a` (quindi l'append che parte da qui NON copia il buffer).
int llm_probe_donatable(mlx_array a) {
  if (a.ctx == NULL) return -1;
  try { return mlx_array_get_(a).is_donatable() ? 1 : 0; }
  catch (...) { return -1; }
}

// Quanti `Data` diversi puntano lo STESSO buffer MTL di `a`: e' l'aliasing che il fix
// introduce (ogni step crea un Data nuovo sulla stessa memoria).
long long llm_probe_data_uses(mlx_array a) {
  if (a.ctx == NULL) return -1;
  try { return (long long)mlx_array_get_(a).data_shared_ptr().use_count(); }
  catch (...) { return -1; }
}

} // extern "C"
