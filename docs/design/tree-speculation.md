# Draft-tree speculation

This document records the invariants of draft-tree speculative decoding
(`--speculative-eagle-topk > 1`, EAGLE3 and EAGLE-style MTP drafters). A deviation from the rules here is a
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

After verify, `TreeSpec.verify` (the `verify_tree` kernel) packs `predict`
along the accepted path and the sampling backend stores the TP-agreed path with
`TreeSpec.record_path`. The executor then calls `TreeSpec.compact` once, a
single launch that moves, for every request:

* target KV of the path (every layer) to the window's leading slots;
* target hidden rows to the front of the window;
* the window positions back from `vc + depth` to `vc + i`.

Everything downstream sees a chain.

A chain is the tree `parent[i] = i - 1`. With `topk == 1` no tree state exists
and the chain path runs unchanged; with `topk > 1` there is no tree-only commit,
output or drafting-step-0 code.

### Position follows depth, slot follows node index

Node `i` of a request's verify window has RoPE position `vc + depth[i]` and KV
slot `write_locations[b * N + i]`. Only positions change for a tree:
`TreeSpec.depth_positions` shifts the window's `vc + i` to `vc + depth` before
the forward, and `TreeSpec.compact` shifts them back after the forward (row `i`
then holds the path node at depth `i`).

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

### Recurrent state follows the parent

Linear-attention (GDN) layers keep one conv window and one recurrent state per
verify node in the backend's verify scratch. Node `t` starts from the state
after its parent, not after node `t - 1`: `causal_conv1d_update` and
`gdn_decode_mtp` take `parent_indices` and reload the parent's scratch row at
branch points (a chain never reloads). The commit copies the scratch row of the
last accepted node, `1 + path[accept_len - 1]`, which for a chain is the
familiar `accept_len`. ReplaySSM keeps no per-node state and refuses trees; the
fused KDA verify kernel follows a chain and refuses them too.

KV compaction covers the attention layers only (`history_group_by_layer`) and
moves each physical region once.

### Draft lanes never write the paged cache

Drafting steps `1 .. S-1` run `K` lane rows per request. Their K/V go to a
per-layer side buffer of `(S - 1) * K` slots per request and are valid only
while the round's tree is drafted; KV prewrite is off on lane steps
(`support_kv_cache_prewrite`).

Lane attention reads `TreeDraftInputs` (prefix lengths over the accepted
frontier, the lanes' ancestor masks, and the step index, a Python value fixed
per lane forward), which the drafter owns and writes within the round: prefix
lengths once per round, masks by `draft_tree_expand` for the next step. This is the one exception to the
refresh-only draft metadata contract of `unified_path.md`; it is graph-safe
because the buffers are bound once at fixed addresses and written by in-graph
ops before the lane forward reads them.

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

EAGLE3 and EAGLE-style MTP drafters (the `Eagle` drafter; the multi-depth `Mtp`
drafter refuses trees at startup); `greedy` and `triton` sampling backends; the `trtllm`
attention backend with bf16 KV and one KV cache group, alone or inside the
hybrid linear-attention backend (GDN, without ReplaySSM); no structured output,
no mixed batches, no sliding window or attention sinks in the target.

## Tests

* `tokenspeed-kernel/test/ops/test_tree_speculative.py` — tree kernels against
  fp32 references (masks up to 64 nodes, short prefixes, splits).
* `test/runtime/test_draft_tree.py` — tree construction against a per-request
  EAGLE-2 reference, depth-first order, strided roots, NaN scores.
* `test/runtime/test_tree_spec.py` — KV compaction; the fresh-spec chain.
* `test/runtime/sampling/test_tree_sampling.py` — a chain-shaped tree verifies
  like the chain; every node samples the chain's draw at its depth.
* `tokenspeed-kernel/test/ops/test_attention_gdn.py`,
  `test/runtime/test_causal_conv1d_tree.py` — per-node GDN states and conv
  windows against a per-path reference.
