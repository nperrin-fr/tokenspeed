# Draft-tree speculation

This document records the invariants of draft-tree speculative decoding
(`--speculative-eagle-topk > 1`, EAGLE3). A deviation from the rules here is a
bug unless this document is updated in the same change.

## The problem this solves

A chain drafter proposes one token per depth; a draft tree keeps the best
`K` candidates per step and verifies up to `N` nodes (`N <= 64`) in one target
forward, so a request accepts whichever root path the target agrees with. The
rest of the runtime — commit (`vc += accept_len`), the output processor, draft
step 0, KV and hidden-state bookkeeping — is written for a chain. Trees must
not grow a second copy of any of it.

## Invariants

### The tree is a parameter of the chain path

After verify, the accepted path is compacted into a chain and everything
downstream sees a chain:

* target KV of the path moves to the window's leading slots
  (`TreeSpec.compact_kv`, one kernel for every layer);
* target hidden rows move to the front of each request's window
  (`TreeSpec.compact_rows`);
* `predict` is packed along the path by the verify kernel itself.

A chain is the tree `parent[i] = i - 1`. With `topk == 1` no tree state exists
and the chain path runs unchanged; with `topk > 1` there is no tree-only commit,
output or drafting-step-0 code.

### Position follows depth, slot follows node index

Node `i` of a request's verify window has RoPE position `vc + depth[i]` and KV
slot `write_locations[b * N + i]`. Only positions change for a tree:
`TreeSpec.depth_positions` shifts the window's `vc + i` to `vc + depth` before
the forward, and `TreeSpec.chain_positions` shifts them back inside the forward
after compaction (row `i` then holds the path node at depth `i`).

The two shifts must cancel for any `depth_buf`, including graph warmup, which
replays the forward without the step prep: `depth_buf` and `mask_buf` start as
the chain the initial parents describe (`test_fresh_spec_is_the_chain`).

### One verify kernel

Target verify is `tree_decode_attention`: split-KV over the paged cache, every
row sees the committed prefix, and the 64-bit ancestor mask applies only to the
last `N` keys. There is no size-dependent second path (no xqa branch, no
prefix + tree cascade). The draft lanes keep the cascade (prefix context kernel
+ `tree_attention` over a side buffer) because their ancestors' K/V never enter
the paged cache.

### Draft lanes never write the paged cache

Drafting steps `1 .. S-1` run `K` lane rows per request. Their K/V go to a
per-layer side buffer of `(S - 1) * K` slots per request and are valid only
while the round's tree is drafted; KV prewrite is off on lane steps
(`support_kv_cache_prewrite`).

### Tree construction is deterministic

`DraftTree` keeps the best `N - 1` of the `K + (S - 1) K^2` scored candidates;
a per-depth tie-break puts every parent strictly ahead of its children, so the
kept set is a tree. Nodes are numbered depth first with each node's best child
first, so the most likely path is `0, 1, 2, ...`. NaN scores (padded requests)
rank last, so lanes and nodes are always fully written.

### Sampled verify keys noise by position

The triton sampling backend draws a verify row's target token by Gumbel-max
keyed by `(seed, position)`. For a tree the key is the node's position
`vc + depth`, never its row: each node row samples with its request's
parameters and its offset advanced by its depth, so the token accepted at every
position is the one plain decoding samples there. Acceptance is the greedy tree
walk over those draws (a child is accepted when it equals its parent's draw).
Backends that cannot do this keep `supports_tree_verify = False` and the server
refuses tree drafting with them.

## Scope

EAGLE3 drafters; `greedy` and `triton` sampling backends; the `trtllm`
attention backend with bf16 KV and one cache group; no structured output, no
mixed batches, no sliding window or attention sinks in the target.

## Tests

* `tokenspeed-kernel/test/ops/test_tree_speculative.py` — tree kernels against
  fp32 references (masks up to 64 nodes, short prefixes, splits).
* `test/runtime/test_draft_tree.py` — tree construction against a per-request
  EAGLE-2 reference, depth-first order, strided roots, NaN scores.
* `test/runtime/test_tree_spec.py` — KV compaction; the fresh-spec chain.
* `test/runtime/sampling/test_tree_sampling.py` — a chain-shaped tree verifies
  like the chain; every node samples the chain's draw at its depth.
