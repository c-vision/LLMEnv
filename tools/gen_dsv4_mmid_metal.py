#!/usr/bin/env python3
"""gen_dsv4_mmid_metal.py — estrae dal ds4 (`metal/moe.metal`) i kernel del MoE BATCHED (mul_mm_id) per
IQ2_XXS / Q2_K / Q4_K e li emette come stringa Metal per il backend GPU DS4 (llm_dsv4_metal.mm). Port FEDELE:
i kernel non sono riscritti, solo estratti (+ lo scarto tp/addr non usato). Le tabelle IQ2 restano quelle del codec.

Uso (da LLMEnv/):  python3 tools/gen_dsv4_mmid_metal.py [/Users/gaetano/ds4/metal/moe.metal]
Output:            LLMEngine/mlx/llm_dsv4_metal_mmid.h  (generato, non editare a mano)
"""
import re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/Users/gaetano/ds4/metal/moe.metal")
OUT = ROOT / "LLMEngine/mlx/llm_dsv4_metal_mmid.h"
s = SRC.read_text()


def block(start_pat: str, open_ch="{", close_ch="}", include_prefix: str = "") -> str:
    """Testo da `start_pat` fino alla parentesi graffa chiusa corrispondente (+ ';' se struct)."""
    m = re.search(start_pat, s)
    if not m:
        raise SystemExit(f"non trovato: {start_pat}")
    i = m.start()
    j = s.index(open_ch, m.end() - 1 if s[m.end() - 1] == open_ch else m.end())
    depth, k = 0, j
    while True:
        c = s[k]
        if c == open_ch: depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0: break
        k += 1
    end = k + 1
    if end < len(s) and s[end] == ';': end += 1
    return include_prefix + s[i:end]


parts = []
parts.append("""#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define QK8_0 32
#ifndef QK_K
#define QK_K 256
#endif
#define N_SIMDWIDTH 32
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define QK_NL 16
constant bool FC_mul_mm_bc_inp = false;
""")
for name in ("block_q2_K", "block_q4_K", "block_iq2_xxs"):
    parts.append(block(r"struct " + name + r"\s*\{"))
parts.append("struct block_q8_0 {\n    half d;\n    int8_t qs[QK8_0];\n};")   # preambolo embedded di ds4_metal.m
parts.append(block(r"static inline uchar2 get_scale_min_k4_just2\([^)]*\)\s*\{"))
for fn in ("dequantize_q2_K", "dequantize_q4_K", "dequantize_iq2_xxs"):
    parts.append("template <typename type4x4>\n" + block(r"void " + fn + r"\("))
DENSE = pathlib.Path(str(SRC).replace("moe.metal", "dense.metal")).read_text()
_m = re.search(r"template <typename type4x4>\s*void dequantize_q8_0\(", DENSE)
_i = DENSE.index("{", _m.end()); _d = 0; _k = _i
while True:
    _d += (DENSE[_k] == "{") - (DENSE[_k] == "}")
    if _d == 0 and DENSE[_k] == "}": break
    _k += 1
parts.append(DENSE[_m.start():_k + 1])
parts.append(block(r"static inline bool ds4_tp_owns_expert\("))
parts.append(block(r"struct ds4_metal_args_mul_mm_id_map0\s*\{"))
parts.append(block(r"struct ds4_metal_args_mul_mm_id\s*\{"))
parts.append("template<short ne20>\n" + block(r"kernel void kernel_mul_mm_id_map0\("))
for _n in (1, 6, 8):
    parts.append(f'template [[host_name("kernel_mul_mm_id_map0_ne20_{_n}")]] kernel decltype(kernel_mul_mm_id_map0<{_n}>) kernel_mul_mm_id_map0<{_n}>;')
mm = re.search(r"template<short NR1, typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void \(\*dequantize_func\)\(device const block_q \*, short, thread S0_4x4 &\), typename T0, typename T0_4x4, typename T1, typename T1_2x4, bool CULL_TAIL_SIMDGROUPS = false, bool EXPERT_ADDRESSES = false>\s*kernel void kernel_mul_mm_id\(", s)
if not mm: raise SystemExit("kernel_mul_mm_id non trovato")
sub = s[mm.start():]
parts.append(block(re.escape(sub[:mm.end() - mm.start()])))
parts.append("typedef decltype(kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, block_q2_K, QK_NL, dequantize_q2_K, float, float4x4, float, float2x4>) mul_mm_id;")
for host, blk, deq in (("q8_0", "block_q8_0", "dequantize_q8_0"), ("iq2_xxs", "block_iq2_xxs", "dequantize_iq2_xxs"), ("q2_K", "block_q2_K", "dequantize_q2_K"), ("q4_K", "block_q4_K", "dequantize_q4_K")):
    parts.append(f'template [[host_name("kernel_mul_mm_id_{host}_f32")]] kernel mul_mm_id kernel_mul_mm_id<32, half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, {blk}, {'2' if host == 'q8_0' else 'QK_NL'}, {deq}, float, float4x4, float, float2x4>;')

body = "\n\n".join(parts)
# stringa C: raw literal (MSL contiene virgolette e backslash)
assert ')MSL"' not in body
out = ("/* GENERATO da tools/gen_dsv4_mmid_metal.py — NON modificare a mano.\n"
       " * Kernel MoE batched (mul_mm_id) estratti FEDELMENTE da ds4 metal/moe.metal (IQ2_XXS, Q2_K, Q4_K). */\n"
       "#ifndef LLM_DSV4_METAL_MMID_H\n#define LLM_DSV4_METAL_MMID_H\n\n"
       "static const char* const LLM_DSV4_METAL_MMID = R\"MSL(\n" + body + "\n)MSL\";\n\n#endif /* LLM_DSV4_METAL_MMID_H */\n")
OUT.write_text(out)
print("scritto", OUT, len(body), "byte MSL")
