cmake_minimum_required(VERSION 3.16)
# C E CXX entrambi abilitati: llmengine_mlx.c e' C puro (riscritto 2026-08-24,
# vedi Fase0b-LlamaCpp-Wrapper-C-Rewrite-Plan.md), ma mlx-c stessa e' C++
# internamente (Metal/C++, esposta via API C) -- il linker finale deve usare
# il driver C++ per risolvere i simboli libstdc++/libc++ di libmlxc.a, quindi
# CXX resta necessario come linguaggio del progetto anche se nessun file .cpp
# e' compilato qui.
# OBJC aggiunto (2026-09-06) per llm_kvmetal_bridge.m -- bridge ABI-C verso
# native/llmmetal_kv/llmmetal_kv.h (Objective-C puro, usa <Metal/Metal.h> direttamente,
# non incluibile da llmengine_mlx.c che resta C). Vedi llm_kvmetal_bridge.h.
project(llmengine_mlx C CXX OBJC OBJCXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ds4engine (motore ds4 VENDORIZZATO) come subdirectory SOLO quando mlx e' il progetto TOP-LEVEL
# (build standalone, es. GMBasic `build-gmbasic`): i consumer che lo aggiungono da soli (FastLLM)
# non lo duplicano. Se il target esiste, in fondo `llm_ai_mlx` lo linka + `LLM_HAVE_DS4ENGINE`,
# cosi' un DS4 caricato via `llmengine_llm_engine_mlx` usa il VENDORED (piu' veloce), non ds4-slow.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
   AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../ds4engine/CMakeLists.txt")
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../ds4engine" "${CMAKE_BINARY_DIR}/ds4engine")
endif()

# Vendoring mlx-c via FetchContent -- stesso approccio "no submodule, nessun
# checkout manuale" già usato per llama.cpp in native/llm_host_ai/. Pin a un
# tag di release, mai a `master`, per build riproducibili.
include(FetchContent)

set(MLX_C_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)  # mlx-c statico, il nostro
                                                  # wrapper è l'unica SHARED

# Xcode completo installato e attivato (xcode-select -s .../Xcode.app/...)
# il 2026-08-16 -- Metal ripristinato. Se questa build fallisce di nuovo
# con "unable to find utility metal", verificare `xcrun --find metal`.
set(MLX_BUILD_METAL ON CACHE BOOL "" FORCE)

# Cache locale persistente (2026-09-07): la sorgente patchata di mlx-c/mlx veniva
# prima cacheata in /tmp (FETCHCONTENT_SOURCE_DIR_MLX), effimero per natura -- una
# pulizia di sistema a mezzanotte l'ha svuotata e ha bloccato la build (vedi Todo
# C17 e Knowledge Base MLX-Architetture). Ora la cache vive in <root>/libs/mlx-deps/
# (fuori da git, vedi .gitignore e Infrastruttura-LLMEnv.md §libs/) e sopravvive a
# pulizie di /tmp e a build/ cancellate. Se presente, FetchContent la usa DIRETTAMENTE
# (nessun fetch di rete, nessuna ri-applicazione della patch -- gia' applicata alla
# copia salvata): rigenerare la cache con `rsync` da build/_deps/{mlx,mlx_c}-src dopo
# un fetch pulito se la patch cambia. Se assente, fallback INVARIATO al fetch di rete.
set(LLM_MLX_LOCAL_DEPS "${CMAKE_CURRENT_SOURCE_DIR}/../../libs/mlx-deps")
if(EXISTS "${LLM_MLX_LOCAL_DEPS}/mlx-c-src/CMakeLists.txt")
    message(STATUS "llmengine_mlx: uso la cache locale ${LLM_MLX_LOCAL_DEPS}/mlx-c-src (niente fetch di rete)")
    set(FETCHCONTENT_SOURCE_DIR_MLX_C "${LLM_MLX_LOCAL_DEPS}/mlx-c-src" CACHE PATH "" FORCE)
endif()
if(EXISTS "${LLM_MLX_LOCAL_DEPS}/mlx-src/CMakeLists.txt")
    message(STATUS "llmengine_mlx: uso la cache locale ${LLM_MLX_LOCAL_DEPS}/mlx-src (niente fetch di rete)")
    set(FETCHCONTENT_SOURCE_DIR_MLX "${LLM_MLX_LOCAL_DEPS}/mlx-src" CACHE PATH "" FORCE)
endif()

# MLX Metal-4.1 NAX fix (2026-09-28, idempotente). I kernel NAX (tensor unit M5) di MLX 0.32.x
# NON compilano col compilatore Metal 4.1 (`32023.921`): rende ESPLICITO il qualificatore
# implicito `thread` -> errori address-space in `steel/{attn,gemm}/nax.h`. Fix upstream
# ml-explore/mlx #3963 (presente in 0.32.1+), ma 0.32.1+ cambia l'API `compile` (rompe
# mlx-c v0.6.0): si applica SOLO il fix Metal, mantenendo l'API 0.32.0.
# Va applicato PRIMA di FetchContent_MakeAvailable(mlx_c) perche' tocca anche
# `mlx/backend/metal/kernels/CMakeLists.txt` (aggiunge il flag `-Wmetal-addr-spaces`).
if(EXISTS "${LLM_MLX_LOCAL_DEPS}/mlx-src/mlx/backend/metal/kernels")
    execute_process(
        COMMAND patch -p1 --forward --dry-run --silent -i
                ${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx_metal41_nax.patch
        WORKING_DIRECTORY "${LLM_MLX_LOCAL_DEPS}/mlx-src"
        RESULT_VARIABLE LLM_MLX_NAX_DRYRUN OUTPUT_QUIET ERROR_QUIET)
    if(LLM_MLX_NAX_DRYRUN EQUAL 0)
        execute_process(
            COMMAND patch -p1 --forward -i
                    ${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx_metal41_nax.patch
            WORKING_DIRECTORY "${LLM_MLX_LOCAL_DEPS}/mlx-src"
            RESULT_VARIABLE LLM_MLX_NAX_APPLY OUTPUT_QUIET ERROR_QUIET)
        if(LLM_MLX_NAX_APPLY EQUAL 0)
            message(STATUS "llmengine_mlx: applicata patch MLX Metal-4.1 NAX")
        else()
            message(WARNING "llmengine_mlx: patch MLX Metal-4.1 NAX FALLITA (rc=${LLM_MLX_NAX_APPLY})")
        endif()
    else()
        message(STATUS "llmengine_mlx: patch MLX Metal-4.1 NAX gia' applicata")
    endif()
endif()

FetchContent_Declare(
    mlx_c
    GIT_REPOSITORY https://github.com/ml-explore/mlx-c.git
    GIT_TAG v0.6.0  # PIN ESPLICITO -- verificato reale (tag GitHub più recente
                     # al 2026-08-16, https://github.com/ml-explore/mlx-c/tags,
                     # commit 0726ca922fc902c4c61ef9c27d94132be418e945).
                     # mlx-c a sua volta vendorizza mlx via il proprio
                     # FetchContent -- vedi il suo CMakeLists.txt.
    GIT_SHALLOW TRUE
    # UPGRADE MLX 0.31.1 -> 0.32.0 (2026-09-02): la patch forza il GIT_TAG di mlx a
    # v0.32.0 (dentro il CMakeLists di mlx-c) e adatta il wrapper C fft.cpp all'API
    # 0.32 (parametro FFTNorm). Migliora il decode (misurato ~1.9x su gpt-oss-20b).
    # Rigenerare la patch con: diff -u <mlx_c-src> <mlx_c-32 patched>.
    # NON applicata se FETCHCONTENT_SOURCE_DIR_MLX_C punta alla cache locale sopra
    # (gia' patchata li' dentro) -- FetchContent salta il PATCH_COMMAND quando la
    # sorgente e' gia' popolata da un SOURCE_DIR esplicito.
    PATCH_COMMAND patch -p1 -i ${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx_c_mlx032.patch
)
FetchContent_MakeAvailable(mlx_c)

# ── LLMEnv patch mlx-c (2026-09-13) — slice-update IN-PLACE ───────────────────
# Aggiunge `mlx_slice_update_inplace` = overwrite_descriptor(slice_update(...)) nel core MLX,
# la stessa cosa che fa `keys[..., a:b, :] = x` di mlx_lm (python/src/indexing.cpp
# mlx_set_item). Dà l'append KV O(1) (nessuna ricopia O(cap) del concatenate) -> risolve il
# crollo del decode a contesto lungo. Applicazione idempotente (dry-run poi apply); la cache
# locale (libs/mlx-deps/mlx-c-src) è già patchata -> il dry-run fallisce e si salta.
if(TARGET mlxc)
    get_target_property(LLM_MLXC_SRC_DIR mlxc SOURCE_DIR)
    if(LLM_MLXC_SRC_DIR AND EXISTS "${LLM_MLXC_SRC_DIR}/mlx/c/ops.cpp")
        # Guardia idempotente robusta al rename (2026-09-28): il `--forward --dry-run`
        # NON riconosce piu' la patch come "gia' applicata" se la cache locale e' stata
        # patchata con una variante il cui commento differisce (il rename dei simboli
        # verso il namespace llm/llmengine cambia le righe aggiunte), quindi la patch
        # si riapplicava producendo una
        # DEFINIZIONE DUPLICATA di mlx_slice_update_inplace (redefinition a compile).
        # Si salta in base alla presenza REALE del simbolo nel sorgente, non al testo
        # delle righe aggiunte dal patch.
        file(READ "${LLM_MLXC_SRC_DIR}/mlx/c/ops.cpp" LLM_MLXC_OPS_CPP)
        string(FIND "${LLM_MLXC_OPS_CPP}" "mlx_slice_update_inplace" LLM_MLXC_ALREADY)
        if(NOT LLM_MLXC_ALREADY EQUAL -1)
            message(STATUS "llmengine_mlx: mlx_slice_update_inplace gia' presente in ${LLM_MLXC_SRC_DIR} (patch saltata)")
        else()
            execute_process(
                COMMAND patch -p1 --forward --dry-run --silent -i
                        "${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx_c_inplace_slice_update.patch"
                WORKING_DIRECTORY "${LLM_MLXC_SRC_DIR}"
                RESULT_VARIABLE LLM_MLXC_PATCH_DRYRUN
                OUTPUT_QUIET ERROR_QUIET)
            if(LLM_MLXC_PATCH_DRYRUN EQUAL 0)
                execute_process(
                    COMMAND patch -p1 --forward -i
                            "${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx_c_inplace_slice_update.patch"
                    WORKING_DIRECTORY "${LLM_MLXC_SRC_DIR}"
                    RESULT_VARIABLE LLM_MLXC_PATCH_APPLY)
                if(LLM_MLXC_PATCH_APPLY EQUAL 0)
                    message(STATUS "llmengine_mlx: applicata patch mlx-c slice_update_inplace")
                else()
                    message(WARNING "llmengine_mlx: patch mlx-c slice_update_inplace FALLITA (rc=${LLM_MLXC_PATCH_APPLY})")
                endif()
            endif()
        endif()
    endif()
endif()

# ── LLMEnv patch mlx-c (2026-10-03) — slice_update_inplace VERO O(n) ──────────
# La patch precedente aggiungeva `mlx_slice_update_inplace` ma chiamando lo `slice_update`
# FUNZIONALE (+ overwrite_descriptor) => COPIA O(cap) del buffer. Questa la rende una VERA
# scrittura in-place O(n) via l'op MLX `slice_update_inplace` (patch mlx_inplace_slice.patch):
# elimina la ricopia O(pos) dell'append KV (misurato 1,6x@16k / 4,4x@32k sul decode, GEN_IDS
# identici). Idempotente (guard su GMB_SLICEUP_COPY); A/B parity con GMB_SLICEUP_COPY=1.
if(TARGET mlxc)
    get_target_property(LLM_MLXC_SRC_DIR mlxc SOURCE_DIR)
    if(LLM_MLXC_SRC_DIR AND EXISTS "${LLM_MLXC_SRC_DIR}/mlx/c/ops.cpp")
        file(READ "${LLM_MLXC_SRC_DIR}/mlx/c/ops.cpp" LLM_MLXC_OPS2)
        string(FIND "${LLM_MLXC_OPS2}" "GMB_SLICEUP_COPY" LLM_MLXC2_ALREADY)
        if(NOT LLM_MLXC2_ALREADY EQUAL -1)
            message(STATUS "llmengine_mlx: inplace slice_update gia' presente in ${LLM_MLXC_SRC_DIR} (patch saltata)")
        else()
            execute_process(
                COMMAND patch -p1 --forward --dry-run --silent -i
                        "${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx_c_inplace_slice.patch"
                WORKING_DIRECTORY "${LLM_MLXC_SRC_DIR}"
                RESULT_VARIABLE LLM_MLXC2_DRYRUN OUTPUT_QUIET ERROR_QUIET)
            if(LLM_MLXC2_DRYRUN EQUAL 0)
                execute_process(
                    COMMAND patch -p1 --forward -i
                            "${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx_c_inplace_slice.patch"
                    WORKING_DIRECTORY "${LLM_MLXC_SRC_DIR}"
                    RESULT_VARIABLE LLM_MLXC2_APPLY)
                if(LLM_MLXC2_APPLY EQUAL 0)
                    message(STATUS "llmengine_mlx: applicata patch mlx-c inplace slice_update")
                else()
                    message(WARNING "llmengine_mlx: patch mlx-c inplace slice_update FALLITA (rc=${LLM_MLXC2_APPLY})")
                endif()
            endif()
        endif()
    endif()
endif()

# ── LLMEnv patch MLX (2026-10-03) — op `slice_update_inplace` (append KV O(n)) ─
# Aggiunge al core MLX la primitiva `SliceUpdate` con flag `inplace` + l'op
# `slice_update_inplace`: ALIASA il buffer di `src` e scrive solo la regione [start,stop) (O(n)),
# senza la copia O(cap) dello `slice_update` funzionale. Idempotente (guard su
# `out.copy_shared_buffer(in);` in indexing.cpp).
if(TARGET mlx)
    get_target_property(LLM_MLX_SRC_DIR mlx SOURCE_DIR)
    if(LLM_MLX_SRC_DIR AND EXISTS "${LLM_MLX_SRC_DIR}/mlx/backend/metal/indexing.cpp")
        file(READ "${LLM_MLX_SRC_DIR}/mlx/backend/metal/indexing.cpp" LLM_MLX_IDX)
        string(FIND "${LLM_MLX_IDX}" "out.copy_shared_buffer(in);" LLM_MLX_INP_ALREADY)
        if(NOT LLM_MLX_INP_ALREADY EQUAL -1)
            message(STATUS "llmengine_mlx: slice_update_inplace gia' presente in ${LLM_MLX_SRC_DIR} (patch saltata)")
        else()
            execute_process(
                COMMAND patch -p1 --forward --dry-run --silent -i
                        "${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx_inplace_slice.patch"
                WORKING_DIRECTORY "${LLM_MLX_SRC_DIR}"
                RESULT_VARIABLE LLM_MLX_INP_DRYRUN OUTPUT_QUIET ERROR_QUIET)
            if(LLM_MLX_INP_DRYRUN EQUAL 0)
                execute_process(
                    COMMAND patch -p1 --forward -i
                            "${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx_inplace_slice.patch"
                    WORKING_DIRECTORY "${LLM_MLX_SRC_DIR}"
                    RESULT_VARIABLE LLM_MLX_INP_APPLY)
                if(LLM_MLX_INP_APPLY EQUAL 0)
                    message(STATUS "llmengine_mlx: applicata patch MLX slice_update_inplace")
                else()
                    message(WARNING "llmengine_mlx: patch MLX slice_update_inplace FALLITA (rc=${LLM_MLX_INP_APPLY})")
                endif()
            endif()
        endif()
    endif()
endif()

# ── LLMEnv patch MLX (2026-09-12) ─────────────────────────────────────────────
# Il kernel `qmv_wide` di MLX (usato per M>=2) NON e' bit-identico al `qmv`
# (M==1): riduzione a shuffle-ladder su k_lanes=8 invece del simd_sum su 32.
# Sul decode batch (n_sess>=2) la matmul quantizzata differisce di ~1 ulp e
# rompe la bit-parity batch-vs-sequenziale (misurato su gpt-oss bf16: diverge
# al token 8). patches/mlx_disable_qmv_wide.patch disabilita qmv_wide di
# default (opt-in con MLX_QMV_WIDE_OPT); il costo del passthrough a qmv e' nel
# rumore. Si applica idempotentemente alla sorgente MLX effettiva (cache locale
# libs/mlx-deps/mlx-src o fetch di rete), cosi' un wipe della cache non perde
# il fix. Se gia' applicata il dry-run fallisce e si salta.
if(TARGET mlx)
    get_target_property(LLM_MLX_SRC_DIR mlx SOURCE_DIR)
    if(LLM_MLX_SRC_DIR AND EXISTS "${LLM_MLX_SRC_DIR}/mlx/backend/metal/quantized.cpp")
        execute_process(
            COMMAND patch -p1 --forward --dry-run --silent -i
                    "${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx_disable_qmv_wide.patch"
            WORKING_DIRECTORY "${LLM_MLX_SRC_DIR}"
            RESULT_VARIABLE LLM_MLX_PATCH_DRYRUN
            OUTPUT_QUIET ERROR_QUIET)
        if(LLM_MLX_PATCH_DRYRUN EQUAL 0)
            execute_process(
                COMMAND patch -p1 --forward -i
                        "${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx_disable_qmv_wide.patch"
                WORKING_DIRECTORY "${LLM_MLX_SRC_DIR}"
                RESULT_VARIABLE LLM_MLX_PATCH_APPLY)
            if(LLM_MLX_PATCH_APPLY EQUAL 0)
                message(STATUS "llmengine_mlx: applicata patch MLX disable qmv_wide")
            else()
                message(WARNING "llmengine_mlx: patch MLX disable qmv_wide FALLITA (rc=${LLM_MLX_PATCH_APPLY})")
            endif()
        endif()
    endif()
endif()

# ── LLMEnv patch MLX (2026-09-12) — SDPA vector head_dim=512 ──────────────────
# MLX supporta la sdpa_vector solo per head_dim {64,96,128,256} (e 192/128); per
# gemma-4 (global_head_dim=512, 10/60 layer full) cade nel path unfused (materializza
# Broadcast GQA O(N)). patches/mlx_sdpa_hd512.patch istanzia il kernel (512,512)
# (qk_per_thread=16, register-fittable) + lo abilita nel check. Misurato su gemma-4:
# decode +5% (bit-identico, IDS 64/64), prefill neutro. Idempotente (dry-run -> skip
# se gia' applicata). Vedi Todo/(open) - MLX-Leve-Model-Dependent-Fine-Tuning.md.
if(TARGET mlx)
    get_target_property(LLM_MLX_SRC_DIR mlx SOURCE_DIR)
    if(LLM_MLX_SRC_DIR AND EXISTS "${LLM_MLX_SRC_DIR}/mlx/backend/metal/scaled_dot_product_attention.cpp")
        execute_process(
            COMMAND patch -p1 --forward --dry-run --silent -i
                    "${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx_sdpa_hd512.patch"
            WORKING_DIRECTORY "${LLM_MLX_SRC_DIR}"
            RESULT_VARIABLE LLM_MLX_PATCH512_DRYRUN
            OUTPUT_QUIET ERROR_QUIET)
        if(LLM_MLX_PATCH512_DRYRUN EQUAL 0)
            execute_process(
                COMMAND patch -p1 --forward -i
                        "${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx_sdpa_hd512.patch"
                WORKING_DIRECTORY "${LLM_MLX_SRC_DIR}"
                RESULT_VARIABLE LLM_MLX_PATCH512_APPLY)
            if(LLM_MLX_PATCH512_APPLY EQUAL 0)
                message(STATUS "llmengine_mlx: applicata patch MLX sdpa hd512")
            else()
                message(WARNING "llmengine_mlx: patch MLX sdpa hd512 FALLITA (rc=${LLM_MLX_PATCH512_APPLY})")
            endif()
        endif()
    endif()
endif()

# cJSON: parser JSON C puro, per config.json e model.safetensors.index.json.
# Sostituisce nlohmann/json (C++ header-only) nella riscrittura in C puro del
# 2026-08-24 -- stesso pin gia' usato in native/llmtoken/CMakeLists.txt.
set(ENABLE_CJSON_TEST OFF CACHE BOOL "" FORCE)
set(ENABLE_CJSON_UTILS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_AND_STATIC_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    cjson
    GIT_REPOSITORY https://github.com/DaveGamble/cJSON.git
    GIT_TAG v1.7.19  # PIN ESPLICITO -- stesso di native/llmtoken/CMakeLists.txt
    GIT_SHALLOW TRUE
)
# v1.7.19 dichiara cmake_minimum_required troppo vecchio per CMake attuale.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_MakeAvailable(cjson)

# ABI del forward differenziabile (training PUL). È Apache (infereza), ma è ESPERIMENTALE/
# separabile: FastLLM e un engine "puro" possono compilare con -DLLM_ENABLE_TRAIN=OFF senza
# includere llm_mlx_train.c (nessuna dipendenza inversa: chi non lo linka non lo vede).
option(LLM_ENABLE_TRAIN "Includi l'ABI del forward differenziabile (llm_mlx_train.c)" ON)
set(LLM_MLX_TRAIN_SRC "")
if(LLM_ENABLE_TRAIN)
    set(LLM_MLX_TRAIN_SRC llm_mlx_diff_forward.c)
endif()

# ds4-slow (nostro engine DS4 nativo) e' OPZIONALE: se disattivato (LLM_WITH_DS4SLOW=OFF) o se la
# cartella non c'e', il motore degrada sul vendored `ds4` / su `ds41`. Cosi' la cartella e'
# cancellabile senza errori di configurazione/compilazione, e un consumer che non vuole il nativo
# (es. GMBasic, che ha il vendored) non lo compila affatto.
option(LLM_WITH_DS4SLOW "Compila il motore DS4 nativo (ds4-slow) in llm_ai_mlx" ON)
set(GMB_DS4SLOW_SRC "")
set(GMB_DS4SLOW_INC "")
if(LLM_WITH_DS4SLOW AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ds4-slow)
  list(APPEND GMB_DS4SLOW_SRC
       ds4-slow/llm_dsv4_metal.mm
       ds4-slow/llm_dsv4_gpu_ctx.mm
       ds4-slow/llm_ds4slow_abi.c)
  set(GMB_DS4SLOW_INC ds4-slow)
endif()

add_library(llm_ai_mlx SHARED
    llmengine_mlx.c
    llm_mlx_p4.c
    llm_mlx_cb.c
    llm_mlx_lora.c
    ${LLM_MLX_TRAIN_SRC}
    llm_mlx_prefix.c
    llm_mlx_dflash2.c
    llm_mlx_kvcompress.c
    llm_mlx_api.c
    llm_mlx_mtp.c
    llm_mlx_selftest.c
    llm_mlx_stats.c
    llm_mlx_rope.c
    llm_mlx_gptoss.c
    llm_mlx_bf16.c
    llm_mlx_gdn_cache.c
    llm_mlx_activation.c
    llm_mlx_moe.c
    llm_mlx_kimi.c
    llm_mlx_qwen4exp_ops.c
    llm_mlx_mixtral.c
    llm_mlx_glm_ds_ffn.c
    llm_mlx_gptoss_ffn.c
    llm_mlx_selftest_eval.c
    llm_mlx_sensitivity.c
    llm_mlx_model_ops.c
    llm_mlx_model_config.c
    llm_mlx_session_swap.c
    llm_mlx_gguf_io.c
    llm_mlx_session_spec.c
    llm_mlx_session_logprob.c
    llm_mlx_embed.c
    llm_mlx_session_generate.c
    llm_mlx_attn_batch.c
    llm_mlx_gdn.c
    llm_mlx_qwen35.c
    llm_mlx_llama.c
    llm_mlx_gemma_mixtral.c
    llm_mlx_mla_glm4.c
    llm_mlx_layer_forward.c
    llm_mlx_execplan.c
    llm_mlx_expolicy.c
    llm_mlx_qwen3_attn.c
    llm_mlx_weights_io.c
    llm_mlx_session_append.c
    llm_mlx_final_step.c
    llm_mlx_gdn_compiled.c
    llm_mlx_dense_geglu.c
    llm_mlx_dense_swiglu.c
    llm_mlx_kvmetal.c
    llm_mlx_session_lifecycle.c
    llm_mlx_model_io.c
    llm_mlx_compiled_decode.c
    llm_mlx_qwen35_moe_ffn.c
    llm_mlx_qwen2_attn.c
    llm_mlx_load_config.c
    llm_mlx_imatrix.c
    llm_mlx_model_lifecycle.c
    llm_kvmetal_bridge.m
    llm_gemvq4_bridge.m
    # ds4-slow (Metal DS4 IQ2/Q8, contesto GPU, adapter ABI): OPZIONALE (vedi GMB_DS4SLOW_SRC).
    # Se presente: Metal DS4 zero-copy + adapter llmengine_llm_engine_ds4slow.
    ${GMB_DS4SLOW_SRC}
    # Adapter ds41-native -> ABI LlmEngine (stessi 9 metodi di ds4/ds4-slow).
    ds41-native/llm_ds41_abi.c
    # P10: GEMM int8-activation x Q4/Q5 sui tensor unit M5 (port oMLX; bridge ABI-C, ObjC/Metal)
    ${CMAKE_CURRENT_SOURCE_DIR}/../../LLMMetal/oq_a8/llmmetal_oq_a8_bridge.mm
    # MPP q4 affine (tensor unit Apple): bridge ObjC++ zero-copy sui buffer MLX (metallib build-time).
    ${CMAKE_CURRENT_SOURCE_DIR}/../../LLMMetal/mpp_q4/llmmetal_mpp_q4_bridge.mm
    # Kernel Metal fuso layout+RoPE per il decode (G1 piano antigravity 2026-09-08).
    # E' C++ perche' usa l'API C++ di mlx (mlx::core::fast) attraverso mlx-c: la
    # API e' comunque esposta extern "C" (llm_mlx_fused_rope_append.h), quindi
    # llmengine_mlx.c resta C puro. Adaptato da mlxcel (Apache-2.0).
    llm_mlx_fused_rope_append.cpp
    # Kernel Metal fuso "residual + RMSNorm" (y = x + w*rmsnorm(r)): toglie 1 nodo/layer.
    # C++ per mlx_fast_metal_kernel; ABI C (llm_mlx_fused_residual_rms.h).
    llm_mlx_fused_residual_rms.cpp
    # Kernel Metal custom per il decode SDPA a head_dim che fora il fused MLX (gemma4 hd=512).
    # C++ perche' usa mlx_fast_metal_kernel (API C++ di mlx via mlx-c); ABI esportata C
    # (llm_mlx_flash_decode_kernel.h) cosi' llmengine_mlx.c resta C puro.
    llm_mlx_flash_decode_kernel.cpp
    # G0 (Todo L3/L26): contatore deterministico dei nodi di dispatch del passo di decode.
    # E' C++ perche' il grafo MLX (array::inputs/primitive_id) non ha alcun binding in ABI C;
    # l'ABI esportata resta C (llm_mlx_dispatch_profile.h). Serve gli header C++ di mlx, quindi
    # sotto viene propagato il loro include dir.
    llm_mlx_dispatch_profile.cpp
    # ABI plugin + descrizione .llmarch: progetti autonomi (non piu' nel runtime).
    ${CMAKE_CURRENT_SOURCE_DIR}/../plugin/llmengine_plugin.c
    # LLMDesc/llmdesc.c NON piu' compilato qui: l'engine lo USA (llm_mlx_gguf_io.c) quindi si
    # linka il target `llmdesc` (sotto), cosi' l'oggetto esiste una volta sola (libllmdesc.a)
    # invece di essere ricompilato anche dentro libllm_ai_mlx.
    # ENGINE container reader: container `GGUF` -> f32 (denso + esperti GMQF) per il forward.
    # Delega i DEQUANT di formato al codec (§0bis); il container runtime dà record+byte.
    llm_engine_container.c
    # Adapter all'ABI comune llmengine (vtable llmengine_llm_engine_mlx).
    llm_engine_mlx.c
)

# Gli header C++ di mlx NON sono reachabili via `mlxc` (che li linka PRIVATE), ma il target
# `mlx` esiste nel grafo di build (FetchContent dentro mlx-c) e la sua include dir di build e'
# letta dalla proprieta'. Lettura da proprieta' invece di un path hardcoded: funziona sia con
# la cache locale libs/mlx-deps/mlx-src sia con un fetch di rete fresco in build/_deps.
if(TARGET mlx)
    target_include_directories(llm_ai_mlx PRIVATE $<TARGET_PROPERTY:mlx,INTERFACE_INCLUDE_DIRECTORIES> ${CMAKE_CURRENT_SOURCE_DIR}/../plugin ${CMAKE_CURRENT_SOURCE_DIR}/../../LLMDesc ${CMAKE_CURRENT_SOURCE_DIR}/../include)
endif()

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

target_link_libraries(llm_ai_mlx PRIVATE mlxc cjson ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
target_include_directories(llm_ai_mlx PRIVATE "${cjson_SOURCE_DIR}")
target_include_directories(llm_ai_mlx PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../LLMMetal/oq_a8")
target_include_directories(llm_ai_mlx PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../LLMMetal/mpp_q4")
target_include_directories(llm_ai_mlx PUBLIC . ${GMB_DS4SLOW_INC} tokenizers ds41-native ../Options)

# FIX dipendenze: gli header ds4-slow/ds41-native sono header-only (NON sono sorgenti) e il tracking
# automatico non li vede -> una modifica a *.h NON ricompila i TU consumatori (build "verde" ma binario
# STALE, molto insidioso). Aggiungendoli come SOURCES del target, CMake li traccia come dipendenze.
file(GLOB LLM_DS4SLOW_HDRS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ds4-slow/*.h)
file(GLOB LLM_DS41_HDRS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ds41-native/*.h)
if(LLM_DS4SLOW_HDRS OR LLM_DS41_HDRS)
  target_sources(llm_ai_mlx PRIVATE ${LLM_DS4SLOW_HDRS} ${LLM_DS41_HDRS})
endif()

# Se il motore ds4 VENDORIZZATO (ds4engine) è nel build, llm_ai_mlx lo linka: la dylib esporta
# allora anche `llmengine_llm_engine_ds4`, e i consumatori (GMBasic/FastLLM) possono scegliere tra
# engine e slow via la STESSA dylib. Se ds4engine non c'è, resta solo `..._ds4slow` (adapter nativo).
if(TARGET ds4engine)
  target_link_libraries(llm_ai_mlx PRIVATE ds4engine)
  # Riferimento diretto all'accessor vendored (default DS4): porta il simbolo DENTRO libllm_ai_mlx.
  target_compile_definitions(llm_ai_mlx PRIVATE LLM_HAVE_DS4ENGINE=1)
endif()

# ENGINE container reader (llm_engine_container.c): include + link del container runtime (GGUF) e del
# codec (dequant di formato). Modifica di infrastruttura autorizzata dall'utente 2026-09-24.
target_include_directories(llm_ai_mlx PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../moe/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../LLMCodec/include)
# Dipendenze come target CMake (build autonomo, niente path a .a/.dylib prebuildati).
if(NOT TARGET llmcodec)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../LLMCodec llmcodec_mlx_build)
endif()
if(NOT TARGET llmengine_moe)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../moe llmengine_moe_mlx_build)
endif()
# Registro opzioni condiviso (LLMEngine/Options): core, dipendenza di mlx/moe/llama.
if(NOT TARGET llmengine_opts)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../Options llmengine_opts_build)
endif()
# LLMDesc (descrizione .llmarch, llmdesc.h/.c): progetto autonomo — l'engine lo USA
# (llm_mlx_gguf_io.c) quindi lo LINK, non lo ricompila (dedup sorgente).
if(NOT TARGET llmdesc)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../LLMDesc llmdesc_mlx_build)
endif()
target_link_libraries(llm_ai_mlx PRIVATE
    llmengine_moe llmcodec llmengine_opts llmdesc
    # i DEQUANT di formato (llmcodec_dsv41_dequant_fp8_block/_fp8_row/_fp4_expert) vivono nel plugIN fp8;
    # il layer ggml-type (llmcodec_ggml_dequant_row/block_vals/type_name/q*_value) nel plugIN ggml.
    llmcodec_fp8 llmcodec_ggml
    -lcompression)   # llmengine_moe_compress (llmengine_moe) usa libcompression macOS
set_source_files_properties(llm_kvmetal_bridge.m llm_gemvq4_bridge.m ${CMAKE_CURRENT_SOURCE_DIR}/../../LLMMetal/oq_a8/llmmetal_oq_a8_bridge.mm ${CMAKE_CURRENT_SOURCE_DIR}/../../LLMMetal/mpp_q4/llmmetal_mpp_q4_bridge.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")

set_target_properties(llm_ai_mlx PROPERTIES
    OUTPUT_NAME "llm_ai_mlx"
    POSITION_INDEPENDENT_CODE ON
    LINKER_LANGUAGE CXX
)

# ── Test DS4 (OPZIONALE) — dipendenze CMake TRACCIATE: MAI stale ──────────────
# REGOLA (utente): non deve esistere il concetto di "stale". I test DS4 erano compilati ad-hoc
# con clang: un binario compilato prima di una modifica a un header condiviso (es.
# LlmCodecGgufTensorInfo in LLMCodec) restava VECCHIO mentre la dylib era nuova -> mismatch ABI
# silenzioso ("manca blk..."). Qui i test diventano target CMake: si ricompilano automaticamente
# quando cambia QUALSIASI header transitivo. Abilita con -DLLM_BUILD_DSV4_TESTS=ON.
option(LLM_BUILD_DSV4_TESTS "Compila i test/bench DS4 come target CMake (dipendenze tracciate, niente stale)" OFF)
if(LLM_BUILD_DSV4_TESTS)
    file(GLOB DSV4_TEST_SRCS CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/ds4-slow/tests/llm_dsv4_*_test.c
        ${CMAKE_CURRENT_SOURCE_DIR}/ds4-slow/tests/llm_dsv4_prefill_bench.c
        ${CMAKE_CURRENT_SOURCE_DIR}/ds41-native/tests/llm_dsv41_*_test.c
        ${CMAKE_CURRENT_SOURCE_DIR}/ds41-native/tests/llm_dsv41_*_bench.c)
    foreach(_src ${DSV4_TEST_SRCS})
        get_filename_component(_t ${_src} NAME_WE)
        add_executable(${_t} ${_src})
        target_link_libraries(${_t} PRIVATE llm_ai_mlx llmcodec_ggml)
        target_include_directories(${_t} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/ds4-slow
            ${CMAKE_CURRENT_SOURCE_DIR}/ds41-native
            ${CMAKE_CURRENT_SOURCE_DIR}/../Options
            ${CMAKE_CURRENT_SOURCE_DIR}/../../LLMCodec/include)
        set_target_properties(${_t} PROPERTIES LINKER_LANGUAGE CXX)
    endforeach()

    # §9.A M10 EXECUTION PLANNER: test model-free (parse + per-layer + skip + reset).
    add_executable(llm_mlx_execplan_test tests/llm_mlx_execplan_test.c)
    target_link_libraries(llm_mlx_execplan_test PRIVATE llm_ai_mlx cjson)
    target_include_directories(llm_mlx_execplan_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} "${cjson_SOURCE_DIR}" ${CMAKE_CURRENT_SOURCE_DIR}/../Options)
    set_target_properties(llm_mlx_execplan_test PROPERTIES LINKER_LANGUAGE CXX)

    # §9.A EXPERT_RESIDENCY_POLICY: test model-free (parse + derive + apply).
    add_executable(llm_mlx_expolicy_test tests/llm_mlx_expolicy_test.c)
    target_link_libraries(llm_mlx_expolicy_test PRIVATE llm_ai_mlx cjson)
    target_include_directories(llm_mlx_expolicy_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} "${cjson_SOURCE_DIR}" ${CMAKE_CURRENT_SOURCE_DIR}/../Options)
    set_target_properties(llm_mlx_expolicy_test PROPERTIES LINKER_LANGUAGE CXX)
endif()
