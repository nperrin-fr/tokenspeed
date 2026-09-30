# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Select and number the nodes of a dynamic draft tree (EAGLE-2 style).

Drafting records ``E`` scored candidates per request: token, cumulative
log-probability, parent candidate (``-1`` under the root) and depth. The tree
keeps the best ``N - 1`` of them under the root and numbers them depth first,
each node's children best first, so the most likely path is ``0, 1, 2, ..``.
One program per request does the whole selection.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["draft_tree_expand", "draft_tree_finalize", "tree_ancestry"]


@triton.jit
def _draft_tree_finalize_kernel(
    scores_ptr,  # [bs, E] float32
    parent_ptr,  # [bs, E] int64 parent candidate, -1 under the root
    depth_ptr,  # [bs, E] int64, 1 under the root
    tokens_ptr,  # [bs, E] int64
    root_ptr,  # [bs] root token
    out_tokens_ptr,  # [bs, N] int32
    out_parent_ptr,  # [bs, N] int32
    rank_ptr,  # [bs, E_PAD] int32 scratch
    sibling_ptr,  # [bs, E_PAD] int32 scratch
    key_ptr,  # [bs, E_PAD] int64 scratch
    pos_ptr,  # [bs, E_PAD] int32 scratch
    root_stride,
    tie_break,
    E: tl.constexpr,
    N: tl.constexpr,
    MAX_DEPTH: tl.constexpr,
    RANK_BITS: tl.constexpr,
    E_PAD: tl.constexpr,
    CHUNK: tl.constexpr,
):
    req = tl.program_id(0)
    base = req * E
    scratch = req * E_PAD
    ent = tl.arange(0, E_PAD)
    ent_ok = ent < E
    depth = tl.load(depth_ptr + base + ent, mask=ent_ok, other=0)
    score = tl.load(scores_ptr + base + ent, mask=ent_ok, other=float("-inf"))
    score = score - tie_break * depth.to(tl.float32)
    score = tl.where(score == score, score, float("-inf"))
    parent = tl.load(parent_ptr + base + ent, mask=ent_ok, other=-1)

    # Global rank: entries strictly better, ties to the lower entry id.
    rank = tl.zeros([E_PAD], dtype=tl.int32)
    for start in tl.static_range(0, E_PAD, CHUNK):
        other = start + tl.arange(0, CHUNK)
        other_ok = other < E
        other_depth = tl.load(depth_ptr + base + other, mask=other_ok, other=0)
        other_score = tl.load(
            scores_ptr + base + other, mask=other_ok, other=float("-inf")
        )
        other_score = other_score - tie_break * other_depth.to(tl.float32)
        other_score = tl.where(other_score == other_score, other_score, float("-inf"))
        better = (other_score[None, :] > score[:, None]) | (
            (other_score[None, :] == score[:, None]) & (other[None, :] < ent[:, None])
        )
        rank += tl.sum((better & other_ok[None, :]).to(tl.int32), axis=1)
    # A child scores strictly below its parent, so kept nodes form a tree.
    kept = ent_ok & (rank < N - 1)
    tl.store(rank_ptr + scratch + ent, rank)
    tl.debug_barrier()

    # Sibling rank among kept nodes, in global rank order.
    sibling = tl.zeros([E_PAD], dtype=tl.int32)
    for start in tl.static_range(0, E_PAD, CHUNK):
        other = start + tl.arange(0, CHUNK)
        other_ok = other < E
        other_parent = tl.load(parent_ptr + base + other, mask=other_ok, other=-2)
        other_rank = tl.load(rank_ptr + scratch + other, mask=other_ok, other=N)
        ahead = (
            (other_parent[None, :] == parent[:, None])
            & (other_rank[None, :] < rank[:, None])
            & (other_rank[None, :] < N - 1)
            & other_ok[None, :]
        )
        sibling += tl.sum(ahead.to(tl.int32), axis=1)
    tl.store(sibling_ptr + scratch + ent, sibling)
    tl.debug_barrier()

    # Depth-first key: (sibling rank + 1) per level from the root down.
    key = tl.zeros([E_PAD], dtype=tl.int64)
    cursor = tl.where(kept, ent.to(tl.int64), -1)
    for _ in tl.static_range(MAX_DEPTH):
        live = cursor >= 0
        safe = tl.where(live, cursor, 0)
        level = tl.load(depth_ptr + base + safe, mask=live, other=0)
        digit = (
            tl.load(sibling_ptr + scratch + safe, mask=live, other=0).to(tl.int64) + 1
        )
        key += tl.where(live, digit << ((MAX_DEPTH - level) * RANK_BITS), 0)
        cursor = tl.where(
            live, tl.load(parent_ptr + base + safe, mask=live, other=-1), -1
        )
    tl.store(key_ptr + scratch + ent, key)
    tl.debug_barrier()

    pos = tl.zeros([E_PAD], dtype=tl.int32)
    for start in tl.static_range(0, E_PAD, CHUNK):
        other = start + tl.arange(0, CHUNK)
        other_ok = other < E
        other_key = tl.load(key_ptr + scratch + other, mask=other_ok, other=0)
        other_rank = tl.load(rank_ptr + scratch + other, mask=other_ok, other=N)
        before = (
            (other_key[None, :] < key[:, None])
            & (other_rank < N - 1)[None, :]
            & other_ok[None, :]
        )
        pos += tl.sum(before.to(tl.int32), axis=1)
    tl.store(pos_ptr + scratch + ent, pos)
    tl.debug_barrier()

    has_parent = kept & (parent >= 0)
    parent_pos = tl.load(
        pos_ptr + scratch + tl.where(has_parent, parent, 0), mask=has_parent, other=-1
    )
    node = pos + 1
    tokens = tl.load(tokens_ptr + base + ent, mask=kept, other=0)
    tl.store(out_tokens_ptr + req * N + node, tokens.to(tl.int32), mask=kept)
    tl.store(out_parent_ptr + req * N + node, parent_pos + 1, mask=kept)
    tl.store(
        out_tokens_ptr + req * N, tl.load(root_ptr + req * root_stride).to(tl.int32)
    )
    tl.store(out_parent_ptr + req * N, -1)


@triton.jit
def _tree_ancestry_kernel(
    parent_ptr,  # [bs, N] int32
    depth_ptr,  # [bs, N] int32
    mask_ptr,  # [bs, N] int64
    N: tl.constexpr,
    N_PAD: tl.constexpr,
    MAX_DEPTH: tl.constexpr,
):
    req = tl.program_id(0)
    nodes = tl.arange(0, N_PAD)
    ok = nodes < N
    mask = tl.full([N_PAD], 1, tl.int64) << nodes.to(tl.int64)
    depth = tl.zeros([N_PAD], dtype=tl.int32)
    cursor = tl.load(parent_ptr + req * N + nodes, mask=ok, other=-1)
    for _ in tl.static_range(MAX_DEPTH):
        live = cursor >= 0
        safe = tl.where(live, cursor, 0)
        mask |= tl.where(live, tl.full([N_PAD], 1, tl.int64) << safe.to(tl.int64), 0)
        depth += live.to(tl.int32)
        cursor = tl.where(
            live, tl.load(parent_ptr + req * N + safe, mask=live, other=-1), -1
        )
    tl.store(depth_ptr + req * N + nodes, depth, mask=ok)
    tl.store(mask_ptr + req * N + nodes, mask, mask=ok)


def tree_ancestry(
    parent: torch.Tensor, max_depth: int, depth: torch.Tensor, mask: torch.Tensor
) -> None:
    """Depth and ancestor-or-self mask of every tree node.

    Args:
        parent: ``[bs, N]`` int32 parent node, ``-1`` for the root; ``N <= 64``.
        max_depth: deepest node depth in the tree.
        depth: ``[bs, N]`` int32 output.
        mask: ``[bs, N]`` int64 output; bit ``j`` marks node ``j`` as an
            ancestor of, or equal to, the row's node.
    """
    bs, num_nodes = parent.shape
    if num_nodes > 64:
        raise ValueError(f"a 64-bit mask holds at most 64 nodes, got {num_nodes}")
    if bs == 0:
        return
    _tree_ancestry_kernel[(bs,)](
        parent,
        depth,
        mask,
        N=num_nodes,
        N_PAD=max(16, triton.next_power_of_2(num_nodes)),
        MAX_DEPTH=max_depth,
    )


@triton.jit
def _draft_tree_expand_kernel(
    child_scores_ptr,  # [bs, K * K] float32, lane-major
    child_tokens_ptr,  # [bs, K * K] int64
    lane_scores_ptr,  # [bs, K] float32, updated in place
    lane_entry_ptr,  # [bs, K] int64, updated in place
    entry_scores_ptr,  # [bs, E] float32
    entry_parent_ptr,  # [bs, E] int64
    entry_depth_ptr,  # [bs, E] int64
    entry_tokens_ptr,  # [bs, E] int64
    lane_tokens_ptr,  # [bs, K] int64 out
    parent_lane_ptr,  # [bs, K] int64 out
    start,
    depth,
    num_entries,
    K: tl.constexpr,
    KK_PAD: tl.constexpr,
):
    req = tl.program_id(0)
    idx = tl.arange(0, KK_PAD)
    ok = idx < K * K
    lane = idx // K
    child = tl.load(child_scores_ptr + req * K * K + idx, mask=ok, other=float("-inf"))
    token = tl.load(child_tokens_ptr + req * K * K + idx, mask=ok, other=0)
    score = tl.load(lane_scores_ptr + req * K + lane, mask=ok, other=0.0) + child
    # NaN (e.g. a padded request's garbage logits) ranks last, so ranks stay a permutation.
    score = tl.where(score == score, score, float("-inf"))
    parent = tl.load(lane_entry_ptr + req * K + lane, mask=ok, other=-1)

    entry = req * num_entries + start + idx
    tl.store(entry_scores_ptr + entry, score, mask=ok)
    tl.store(entry_parent_ptr + entry, parent, mask=ok)
    tl.store(entry_depth_ptr + entry, tl.full([KK_PAD], depth, tl.int64), mask=ok)
    tl.store(entry_tokens_ptr + entry, token, mask=ok)

    # Rank among the K * K children, ties to the lower index; the best K become the lanes.
    rank = tl.zeros([KK_PAD], dtype=tl.int32)
    for j in tl.static_range(K * K):
        other = tl.load(child_scores_ptr + req * K * K + j) + tl.load(
            lane_scores_ptr + req * K + j // K
        )
        other = tl.where(other == other, other, float("-inf"))
        rank += ((other > score) | ((other == score) & (j < idx))).to(tl.int32)
    tl.debug_barrier()
    best = ok & (rank < K)
    tl.store(lane_scores_ptr + req * K + rank, score, mask=best)
    tl.store(lane_entry_ptr + req * K + rank, (start + idx).to(tl.int64), mask=best)
    tl.store(lane_tokens_ptr + req * K + rank, token, mask=best)
    tl.store(parent_lane_ptr + req * K + rank, lane.to(tl.int64), mask=best)


def draft_tree_expand(
    child_scores: torch.Tensor,
    child_tokens: torch.Tensor,
    lane_scores: torch.Tensor,
    lane_entry: torch.Tensor,
    entry_scores: torch.Tensor,
    entry_parent: torch.Tensor,
    entry_depth: torch.Tensor,
    entry_tokens: torch.Tensor,
    *,
    start: int,
    depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Record every lane's ``K`` children and keep the best ``K`` as new lanes.

    Args:
        child_scores: ``[bs, K * K]`` float32 child log-probabilities, lane-major.
        child_tokens: ``[bs, K * K]`` int64 child tokens.
        lane_scores: ``[bs, K]`` float32 cumulative lane scores, updated in place.
        lane_entry: ``[bs, K]`` int64 candidate id of each lane, updated in place.
        entry_scores: ``[bs, E]`` float32 candidate record; children land at
            ``[start, start + K * K)``, as do ``entry_parent`` (int64),
            ``entry_depth`` (int64) and ``entry_tokens`` (int64).
        start: first candidate id of this step's children.
        depth: depth of this step's children.

    Returns:
        ``(lane_tokens, parent_lane)``: ``[bs, K]`` int64 token of each new lane
        (best first) and the lane it descends from.
    """
    bs, topk = lane_scores.shape
    lane_tokens = torch.empty((bs, topk), dtype=torch.int64, device=lane_scores.device)
    parent_lane = torch.empty_like(lane_tokens)
    if bs == 0:
        return lane_tokens, parent_lane
    _draft_tree_expand_kernel[(bs,)](
        child_scores,
        child_tokens,
        lane_scores,
        lane_entry,
        entry_scores,
        entry_parent,
        entry_depth,
        entry_tokens,
        lane_tokens,
        parent_lane,
        start,
        depth,
        entry_scores.shape[1],
        K=topk,
        KK_PAD=max(16, triton.next_power_of_2(topk * topk)),
    )
    return lane_tokens, parent_lane


def draft_tree_finalize(
    scores: torch.Tensor,
    parent: torch.Tensor,
    depth: torch.Tensor,
    tokens: torch.Tensor,
    root_tokens: torch.Tensor,
    *,
    num_nodes: int,
    max_depth: int,
    tie_break: float,
    rank_bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the best ``num_nodes - 1`` candidates under the root, depth first.

    Args:
        scores: ``[bs, E]`` float32 cumulative log-probability per candidate.
        parent: ``[bs, E]`` int64 parent candidate, ``-1`` under the root; a
            parent precedes its children.
        depth: ``[bs, E]`` int64 candidate depth, 1 under the root.
        tokens: ``[bs, E]`` int64 candidate token.
        root_tokens: ``[bs]`` token of node 0, any stride.
        num_nodes: ``N``, nodes including the root; ``N - 1 <= E``.
        max_depth: deepest candidate depth.
        tie_break: per-depth score penalty that puts every parent strictly
            ahead of its children.
        rank_bits: bits per level of the depth-first key; ``max_depth *
            rank_bits <= 62`` and every sibling rank + 1 fits.

    Returns:
        ``(tokens, parent)``: ``[bs, N]`` int32 node tokens and parent nodes
        (``-1`` for the root).
    """
    bs, num_entries = scores.shape
    if not 1 <= num_nodes - 1 <= num_entries:
        raise ValueError(f"cannot keep {num_nodes - 1} of {num_entries} candidates")
    if max_depth * rank_bits > 62:
        raise ValueError(f"depth {max_depth} x {rank_bits} bits overflows the key")
    device = scores.device
    out_tokens = torch.empty((bs, num_nodes), dtype=torch.int32, device=device)
    out_parent = torch.empty((bs, num_nodes), dtype=torch.int32, device=device)
    if bs == 0:
        return out_tokens, out_parent
    e_pad = max(16, triton.next_power_of_2(num_entries))
    rank = torch.empty((bs, e_pad), dtype=torch.int32, device=device)
    sibling = torch.empty_like(rank)
    pos = torch.empty_like(rank)
    key = torch.empty((bs, e_pad), dtype=torch.int64, device=device)
    _draft_tree_finalize_kernel[(bs,)](
        scores,
        parent,
        depth,
        tokens,
        root_tokens,
        out_tokens,
        out_parent,
        rank,
        sibling,
        key,
        pos,
        root_tokens.stride(0),
        tie_break,
        E=num_entries,
        N=num_nodes,
        MAX_DEPTH=max_depth,
        RANK_BITS=rank_bits,
        E_PAD=e_pad,
        CHUNK=min(e_pad, 32),
    )
    return out_tokens, out_parent
