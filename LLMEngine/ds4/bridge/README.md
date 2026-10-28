# bridge

Our own code around ds4 (not part of upstream, so it can be changed freely):

- `ds4_bridge.{c,h}`: a token-level interface (open a model, sync tokens, get the best next token or
  the logits) on top of ds4's engine and session objects.
- the adapter that exposes ds4 through the common interface in `../../include/llmengine.h`, so a
  program can use it like any other back end.
