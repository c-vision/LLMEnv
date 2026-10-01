// Copyright © 2023-2024 Apple Inc.
#include <memory>

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

/* ── GMBasic patch (diagnostica GPU per-primitiva) ─────────────────────────────
 * Con GMB_MLX_PEROP=1 ogni primitiva viene chiusa in un PROPRIO command buffer e
 * cronometrata con GPUEndTime-GPUStartTime; a fine processo si stampa la classifica
 * per tempo GPU. Serve a trovare quale kernel Metal consuma (niente Xcode/attach).
 * Default OFF: zero overhead. */
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace {
struct GmbPerOp {
    std::map<std::string, double> ms;
    std::map<std::string, long> n;
    std::mutex mu;
};
GmbPerOp& gmb_perop() { static GmbPerOp s; return s; }
long gmb_perop_count = 0;
const char* gmb_dtname(const mlx::core::Dtype& d) {
    if (d == mlx::core::float16) return "f16";
    if (d == mlx::core::bfloat16) return "bf16";
    if (d == mlx::core::float32) return "f32";
    return "?";
}
bool gmb_perop_on() {
    static int v = -1;
    if (v < 0) v = (getenv("GMB_MLX_PEROP") != nullptr) ? 1 : 0;
    return v == 1;
}
int gmb_perop_mode() {
    static int v = -2;
    if (v == -2) {
        const char* e = getenv("GMB_MLX_PEROP");
        v = e ? atoi(e) : 0;
    }
    return v;
}
void gmb_perop_dump() {
    GmbPerOp& s = gmb_perop();
    std::vector<std::pair<std::string, double>> v(s.ms.begin(), s.ms.end());
    std::sort(v.begin(), v.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    double tot = 0;
    for (auto& p : v) tot += p.second;
    fprintf(stderr, "[GMB_MLX_PEROP] n_prim=%ld distinct=%zu total_gpu=%.2f ms\n",
            gmb_perop_count, v.size(), tot);
    for (auto& p : v)
        if (p.second > 0.01)
            fprintf(stderr, "  %-44s %8.3f ms  n=%ld\n", p.first.c_str(), p.second,
                    s.n[p.first]);
}
void gmb_perop_maybe_dump() {
    if (!gmb_perop_on()) return; /* silenzioso di default: il dump esce solo con GMB_MLX_PEROP */
    if (++gmb_perop_count % 1000 == 0) gmb_perop_dump();
}
struct GmbPerOpAtExit {
    GmbPerOpAtExit() { if (gmb_perop_on()) atexit(gmb_perop_dump); }
};
GmbPerOpAtExit gmb_perop_atexit;
} // namespace

namespace mlx::core::gpu {

void init() {}

void new_stream(Stream s) {
  assert(s.device == Device::gpu);
  auto& encoders = metal::get_command_encoders();
  auto& d = metal::device(s.device);
  encoders.try_emplace(s.index, d, s.index, d.residency_set());
}

void new_thread_unsafe_stream(Stream s) {
  assert(s.device == Device::gpu);
  auto& encoders = metal::get_global_command_encoders();
  auto& d = metal::device(s.device);
  encoders.try_emplace(s.index, d, s.index, d.residency_set());
}

void eval(array& arr) {
  auto pool = metal::new_scoped_memory_pool();
  auto s = arr.primitive().stream();
  auto& encoder = metal::get_command_encoder(s);
  auto* command_buffer = encoder.get_command_buffer();

  auto outputs = arr.outputs();
  {
    // If the array is a tracer hold a reference
    // to its inputs so they don't get donated
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }

    debug_set_primitive_buffer_label(command_buffer, arr.primitive());
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }
  std::unordered_set<std::shared_ptr<array::Data>> buffers;
  for (auto& in : arr.inputs()) {
    buffers.insert(in.data_shared_ptr());
  }
  for (auto& s : arr.siblings()) {
    buffers.insert(s.data_shared_ptr());
  }
  // Remove the output if it was donated to by an input
  if (auto it = buffers.find(arr.data_shared_ptr()); it != buffers.end()) {
    buffers.erase(it);
  }

  if (gmb_perop_mode() == 1) {
    /* GMBasic: una primitiva = un command buffer, cronometrato. Nome + shape + dtype. */
    std::string pname = arr.primitive().name();
    {
      auto o = arr.outputs();
      if (!o.empty()) {
        pname += " ";
        pname += gmb_dtname(o[0].dtype());
        pname += " in=";
        auto in = arr.inputs();
        for (size_t k = 0; k < in.size() && k < 3; k++)
          pname += in[k].flags().row_contiguous ? "C" : "N";
        pname += " [";
        for (int i = 0; i < o[0].ndim(); i++) {
          if (i) pname += ",";
          pname += std::to_string(o[0].shape(i));
        }
        pname += "]";
      }
    }
    MTL::CommandBuffer* cb = command_buffer;
    encoder.end_encoding();
    scheduler::notify_new_task(s);
    encoder.commit([s, cb, pname, buffers = std::move(buffers)]() {
      double dt = (cb->GPUEndTime() - cb->GPUStartTime()) * 1e3;
      GmbPerOp& st = gmb_perop();
      {
        std::lock_guard<std::mutex> lk(st.mu);
        st.ms[pname] += dt;
        st.n[pname] += 1;
      }
      gmb_perop_maybe_dump();
      scheduler::notify_task_completion(s);
    });
  } else if (encoder.needs_commit()) {
    MTL::CommandBuffer* cb = command_buffer;
    encoder.end_encoding();
    scheduler::notify_new_task(s);
    if (gmb_perop_mode() == 2) {
      encoder.commit([s, cb, buffers = std::move(buffers)]() {
        std::string lbl = "?";
        if (auto l = cb->label(); l) lbl = l->utf8String();
        if (lbl.size() > 60) lbl = lbl.substr(0, 60);
        double dt = (cb->GPUEndTime() - cb->GPUStartTime()) * 1e3;
        GmbPerOp& st = gmb_perop();
        {
          std::lock_guard<std::mutex> lk(st.mu);
          st.ms[lbl] += dt;
          st.n[lbl] += 1;
        }
        gmb_perop_maybe_dump();
        scheduler::notify_task_completion(s);
      });
    } else {
      encoder.commit([s, buffers = std::move(buffers)]() {
        scheduler::notify_task_completion(s);
      });
    }
  } else {
    command_buffer->addCompletedHandler(
        [buffers = std::move(buffers)](MTL::CommandBuffer* cbuf) {});
  }
}

void finalize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  auto& encoder = metal::get_command_encoder(s);
  auto* cb = encoder.get_command_buffer();
  encoder.end_encoding();
  if (gmb_perop_mode() == 2) {
    encoder.commit([cb]() {
      std::string lbl = "?";
      if (auto l = cb->label(); l) lbl = l->utf8String();
      if (lbl.size() > 60) lbl = lbl.substr(0, 60);
      double dt = (cb->GPUEndTime() - cb->GPUStartTime()) * 1e3;
      GmbPerOp& st = gmb_perop();
      { std::lock_guard<std::mutex> lk(st.mu); st.ms[lbl] += dt; st.n[lbl] += 1; }
    });
    gmb_perop_dump();  /* cumulativo: l'ultima riga e' il totale */
  } else {
    encoder.commit();
  }
}

void synchronize(Stream s) {
  metal::get_command_encoder(s).synchronize();
}

void clear_streams() {
  metal::get_command_encoders().clear();
  if (is_main_thread()) {
    metal::get_global_command_encoders().clear();
  }
}

} // namespace mlx::core::gpu
