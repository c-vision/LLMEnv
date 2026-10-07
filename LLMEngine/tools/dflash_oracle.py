# Golden oracle per il port DFlash2 nel motore MLX di LLMEnv.
# Esegue il round-loop DFlash REALE (mlx-vlm generate_step + vendored drafter DFlash2)
# su un prompt fisso greedy, e registra per OGNI draft_block:
#   - anchor (ultimo token verificato, int)
#   - ctx feature hidden in ingresso [S, 25600] (concat hidden di layer [5,19,33,47,61])
#   - blocco proposto dal draft (token int)
# Registra anche il transcript emesso (bonus + token accettati, greedy del target).
# Output: /tmp/dflash_goldens.npz (arrays) + /tmp/dflash_goldens.json (metadati).
import sys, json, time
import numpy as np
import mlx.core as mx

sys.path.insert(0, "/Users/gaetano/Documents/Projects/Rapid-MLX-NextGen")
from vllm_mlx.speculative.dflash.drafters import load_drafter, is_dflash2_repo

TARGET = "/Users/gaetano/ai/Models/Qwen3.8-27B-8bit"
DRAFT = "/Users/gaetano/ai/Models/Qwen3.8-27B-DFlash2"
PROMPT = "The capital of France is"
MAX_TOKENS = 20
BLOCK = 8
OUT_NPZ = "/tmp/dflash_goldens_deep.npz"
OUT_JSON = "/tmp/dflash_goldens_deep.json"

t0 = time.time()
print("[oracle] loading target...", flush=True)
from mlx_vlm import load as vlm_load
model, processor = vlm_load(TARGET)
print(f"[oracle] target loaded {time.time()-t0:.0f}s", flush=True)

tok = processor.tokenizer
ids = tok.encode(PROMPT)
print(f"[oracle] prompt ids ({len(ids)}): {ids}", flush=True)
assert max(ids) < 248320

print("[oracle] loading draft...", flush=True)
assert is_dflash2_repo(DRAFT), "not dflash2"
draft, _kind = load_drafter(DRAFT)
print(f"[oracle] draft loaded {time.time()-t0:.0f}s", flush=True)

# ---- instrumentation: registra ogni draft_block chiamato (anchor, ctx hidden, blocco) ----
rounds_log = []
orig_draft_block = draft.draft_block

def record_draft_block(last_bonus, hidden, cache, block_size, sampler, token_dtype=mx.int32):
    # materializza ctx hidden e anchor per il golden
    anchor = int(last_bonus)
    h = mx.array(hidden)
    h32 = h.astype(mx.float32)
    mx.eval(h32)
    ctx = np.array(h32)  # [1, S, 25600]
    if ctx.ndim == 3 and ctx.shape[0] == 1:
        ctx = ctx[0]
    rec = {"anchor": anchor, "ctx_feature": ctx, "block_size": int(block_size)}
    block = orig_draft_block(last_bonus, hidden, cache, block_size, sampler, token_dtype)
    bl = mx.array(block)
    mx.eval(bl)
    toks = np.array(bl).reshape(-1).tolist()
    rec["draft_tokens"] = toks
    snaps = [layer_snaps[i] for i in range(len(layer_snaps))]
    rec["snaps"] = snaps
    layer_snaps.clear()
    rounds_log.append(rec)
    print(f"[oracle] round {len(rounds_log)}: anchor={anchor} ctxS={ctx.shape[0]} draft={toks[:block_size]}", flush=True)
    return block

layer_snaps = []
orig_norm = draft.norm
def snap_norm(x):
    y = orig_norm(x)
    yy = mx.array(y).astype(mx.float32); mx.eval(yy)
    layer_snaps.append(("final_norm", np.array(yy)))
    return y
draft.norm = snap_norm
orig_hidden = draft._hidden
def snap_hidden(inputs, target_hidden, cache):
    h = draft._embed_input_tokens(inputs)
    h_ctx = draft.hidden_norm(draft.fc(target_hidden))
    for li, (layer, c) in enumerate(zip(draft.layers, cache)):
        h = layer(h, h_ctx, draft.rope, c)
        yy = mx.array(h).astype(mx.float32); mx.eval(yy)
        layer_snaps.append((f"layer{li}", np.array(yy)))
    return draft.norm(h)
draft._hidden = snap_hidden

draft.draft_block = record_draft_block

from mlx_vlm.generate.ar import generate_step

print("[oracle] running generate_step (dflash greedy)...", flush=True)
transcript = []
t_gen = time.time()
gen = generate_step(
    mx.array([ids], dtype=mx.int32),
    model,
    None, None,
    max_tokens=MAX_TOKENS,
    temperature=0.0,
    draft_model=draft,
    draft_kind="dflash",
    draft_block_size=BLOCK,
    verbose=False,
)
for toks, _state in gen:
    if isinstance(toks, (int, np.integer)):
        transcript.append(int(toks))
    elif hasattr(toks, "tolist"):
        transcript.extend(int(x) for x in toks.tolist())
    else:
        transcript.extend(int(x) for x in toks)
print(f"[oracle] generation done {time.time()-t_gen:.0f}s total {time.time()-t0:.0f}s", flush=True)

# decoded transcript per ispezione
dec = []
for x in transcript:
    try:
        dec.append(processor.decode([x]))
    except Exception as e:
        dec.append(f"<?{e}>")
print("[oracle] transcript tokens:", transcript)
print("[oracle] transcript text :", processor.decode(transcript))

# salva golden: ctx feature di ogni round + draft tokens + anchor + transcript
data = {"transcript": np.array(transcript, dtype=np.int64)}
meta = {"prompt": PROMPT, "prompt_ids": ids, "target": TARGET, "draft": DRAFT,
        "block_size": BLOCK, "max_tokens": MAX_TOKENS,
        "rounds": [], "transcript": transcript}
for i, r in enumerate(rounds_log):
    for si,(tag,arr) in enumerate(r["snaps"]):
        data[f"round{i}_{tag}"] = np.ascontiguousarray(arr, dtype=np.float32)
    data[f"round{i}_ctx"] = np.ascontiguousarray(r["ctx_feature"], dtype=np.float32)
    data[f"round{i}_draft"] = np.array(r["draft_tokens"], dtype=np.int64)
    data[f"round{i}_anchor"] = np.array([r["anchor"]], dtype=np.int64)
    meta["rounds"].append({"anchor": r["anchor"], "ctxS": int(r["ctx_feature"].shape[0]), "snaps": [t for t,_ in r["snaps"]],
                           "block_size": r["block_size"], "draft_tokens": r["draft_tokens"]})
np.savez(OUT_NPZ, **data)
json.dump(meta, open(OUT_JSON, "w"), indent=1)
print(f"[oracle] saved {OUT_NPZ} + {OUT_JSON} ({len(rounds_log)} rounds)")
