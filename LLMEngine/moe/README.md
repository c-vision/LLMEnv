# moe

A runtime for mixture-of-experts models that are **too big to fit in memory**. A mixture-of-experts
model has many "experts" but uses only a few for each token. This runtime keeps the experts on disk and
brings in only the ones needed, keeping the busy ones in a cache and fetching likely ones ahead of time.

It keeps four concerns separate: the model (architecture), the runtime (where experts live), the cache
(what stays in memory) and the prefetcher (what to load next).

- `include/`: public headers (`llmengine_moe.h` is the entry point).
- `src/`: the runtime.
- `tests/`: tests.
- `tools/`: helper tools.
- `llm_engine_moe.c`: the adapter to the common interface. This runtime works one request at a time:
  the session collects the tokens and generation runs when asked.
