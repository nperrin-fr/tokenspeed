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

The sampling backend verifies the step's trees (`TreeVerifyBatch`: parents and
depths) with the `verify_tree` kernel, packing `predict` along the accepted
path. The TP-agreed path has one owner, the backend's packed verify output
(`SamplingBackend.accepted_path`); every consumer receives it explicitly:

* the attention backend moves the path's target KV (every layer) to the
  window's leading slots (`compact_verify_window`; the cache-group router owns
  the K/V address table and the `compact_window_rows` kernel copies words, so
  it is dtype-agnostic);
* `TreeSpec.compact_rows` moves the target hidden rows to the front of the
  window and the positions back from `vc + depth` to `vc + i`;
* the linear-attention commit reads the last accepted node through
  `commit_speculative_state_after_verify(accepted_path=...)` (`None` for a chain).

Everything downstream sees a chain.

A chain is the tree `parent[i] = i - 1`. With `topk == 1` no tree state exists
and the chain path runs unchanged; with `topk > 1` there is no tree-only commit,
output or drafting-step-0 code.

### Position follows depth, slot follows node index

Node `i` of a request's verify window has RoPE position `vc + depth[i]` and KV
slot `write_locations[b * N + i]`. Only positions change for a tree:
`TreeSpec.depth_positions` shifts the window's `vc + i` to `vc + depth` before
the forward, and `TreeSpec.compact_rows` shifts them back after the forward
(row `i` then holds the path node at depth `i`).

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

KV compaction covers the attention layers only (`history_group_by_layer`, read
by the router from its bound pool) and moves each physical region once.

### Draft lanes never write the paged cache

Drafting steps `1 .. S-1` run `K` lane rows per request. Their K/V go to the
trtllm leaf's per-layer side buffers of `(S - 1) * K` slots per request,
allocated from the leaf's own bound pool, and are valid only while the round's
tree is drafted; KV prewrite is off on lane steps (`support_kv_cache_prewrite`).

Lane attention reads `TreeDraftInputs` (prefix lengths over the accepted
frontier, the lanes' ancestor masks, and the step index, a Python value fixed
per lane forward), which the drafter owns and writes within the round: prefix
lengths once per round, masks by `draft_tree_expand` for the next step. This is the one exception to the
refresh-only draft metadata contract of `unified_path.md`; it is graph-safe
because the buffers are bound once at fixed addresses and written by in-graph
ops before the lane forward reads them.

### The drafter scores, the tree selects

`Eagle._score_candidates` is the one place a drafter decides how candidates are
scored (today `logprob_topk` over the full draft vocabulary); `DraftTree` only
records `(scores, tokens)` and selects. A child's score is at most its parent's
(`draft_tree_expand` clamps child log-probabilities at 0). A new drafter changes
the scorer, not the tree machinery.

### Next round's tree rides with next round's tokens

The drafter's parents for a pool slot live in `RuntimeStates.future_parent_map`
next to its candidate tokens in `future_input_map`; rows reset to dummy tokens
(bootstrap, recovery) reset to the chain.

### Tree construction is deterministic

`DraftTree` keeps the best `N - 1` of the `K + (S - 1) K^2` scored candidates;
a child's cumulative log-probability never exceeds its parent's and ties go to
the lower candidate id (the parent's), so the kept set is a tree. Nodes are numbered depth first with each node's best child
first, so the most likely path is `0, 1, 2, ...`. NaN scores (padded requests)
rank last, so lanes and nodes are always fully written.

### Sampled verify keys noise by position

The triton sampling backend draws a verify row's target token by Gumbel-max
keyed by `(seed, position)`. For a tree the key is the node's position
`vc + depth`, never its row: each node row samples with its request's
parameters and its offset advanced by its depth, so the token accepted at every
position is the one plain decoding samples there. Acceptance is the greedy tree
walk over those draws (a child is accepted when it equals its parent's draw).
Backends that cannot do this keep `supports_tree_verify = False` and the
executor refuses tree drafting with them at startup.

## Scope

EAGLE3 and EAGLE-style MTP drafters (the `Eagle` drafter; the multi-depth `Mtp`
drafter refuses trees at startup); `greedy` and `triton` sampling backends; the `trtllm`
attention backend with bf16 KV and one KV cache group, alone or inside the
hybrid linear-attention backend (GDN, without ReplaySSM); no structured output,
no mixed batches, no pipeline parallelism, no sliding window or attention
sinks in the target or draft layers, and no
target that reads request token history or n-gram (Engram) input history.

Each attention backend node declares its own part through `tree_support()`
(verify and lanes, each supported or refused with a reason);
`resolve_tree_support` composes it over the target and draft backend trees
through `child_backends()` once at startup, before any bind, and reports every
blocker together. Composites never forward the question, so a new composite
cannot silently skip a child. Sliding window and attention sinks are the named
exception: they are per layer and per call, so the trtllm forward refuses them.

## Not scheduler or cache state

The only per-request fact that crosses steps is the next round's `parent[N]`,
an attribute of the candidate block `future_input_map` already carries on the
executor side. Lane K/V are round-local scratch (never prefix-matched,
transferred or freed with blocks) and the per-node GDN states are the existing
verify scratch, so nothing here is a cache group for the C++ scheduler to own.

## Intended direction

* One drafter loop: fold the lane steps into the chain's multi-step loop with
  `K` as the row multiplier (keep the fused distributed argmax at `K == 1`).
* Tree refresh inside `refresh_decode_metadata`, and leaves branching on a
  tree mask in the verify metadata instead of the query shape.
* The parallel tree conv kernel as the only verify conv kernel if it benches at
  or above the serial chain kernel.

## Tests

* `tokenspeed-kernel/test/ops/test_tree_speculative.py` — tree kernels against
  fp32 references (masks up to 64 nodes, short prefixes, splits).
* `test/runtime/test_draft_tree.py` — tree construction against a per-request
  EAGLE-2 reference, depth-first order, strided roots, NaN scores.
* `test/runtime/test_tree_spec.py` — hidden-row and position compaction; the
  fresh-spec chain. `test_tree_speculative.py::test_compact_window_rows_moves_every_buffer`
  — KV-row compaction over bf16 and fp8 planes.
* `test/runtime/test_tree_support_resolution.py` — backend capability
  resolution: supported trees, a draft with linear layers, each named blocker.
* `test/runtime/sampling/test_tree_sampling.py` — a chain-shaped tree verifies
  like the chain; every node samples the chain's draw at its depth.
* `tokenspeed-kernel/test/ops/test_attention_gdn.py`,
  `test/runtime/test_causal_conv1d_tree.py` — per-node GDN states and conv
  windows against a per-path reference.
