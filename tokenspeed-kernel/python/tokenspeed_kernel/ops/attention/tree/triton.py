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

"""Attention inside a speculative draft tree.

Each request owns ``R`` query rows and ``M`` key/value slots; row ``r`` sees
slot ``j`` when bit ``j`` of its 64-bit mask is set. Target verify uses
``R == M == N`` over the forward's own K/V; drafting uses ``R == K`` rows over
the per-request buffer of already expanded nodes. The result is one partial
attention state (output and natural-log LSE) that callers merge with the
prefix part.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["tree_attention", "tree_decode_attention"]

MAX_TREE_SLOTS = 64


@triton.jit
def _tree_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mask_ptr,
    prefix_out_ptr,
    prefix_lse_ptr,
    out_ptr,
    lse_ptr,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_ot,
    stride_oh,
    stride_pt,
    stride_ph,
    sm_scale_log2,
    lse_scale,
    ROWS: tl.constexpr,
    SLOTS: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROWS_PAD: tl.constexpr,
    SLOTS_PAD: tl.constexpr,
    DIM_PAD: tl.constexpr,
    HAS_PREFIX: tl.constexpr,
):
    req = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // GROUP

    rows = tl.arange(0, ROWS_PAD)
    slots = tl.arange(0, SLOTS_PAD)
    dims = tl.arange(0, DIM_PAD)
    row_ok = rows < ROWS
    slot_ok = slots < SLOTS
    dim_ok = dims < HEAD_DIM

    q_rows = req * ROWS + rows
    kv_rows = req * SLOTS + slots

    q = tl.load(
        q_ptr + q_rows[:, None] * stride_qt + head * stride_qh + dims[None, :],
        mask=row_ok[:, None] & dim_ok[None, :],
        other=0.0,
    )
    k = tl.load(
        k_ptr + kv_rows[:, None] * stride_kt + kv_head * stride_kh + dims[None, :],
        mask=slot_ok[:, None] & dim_ok[None, :],
        other=0.0,
    )
    v = tl.load(
        v_ptr + kv_rows[:, None] * stride_vt + kv_head * stride_vh + dims[None, :],
        mask=slot_ok[:, None] & dim_ok[None, :],
        other=0.0,
    )

    bits = tl.load(mask_ptr + q_rows, mask=row_ok, other=0)
    visible = ((bits[:, None] >> slots[None, :].to(tl.int64)) & 1) != 0
    visible = visible & slot_ok[None, :] & row_ok[:, None]

    scores = tl.dot(q, tl.trans(k)) * sm_scale_log2
    scores = tl.where(visible, scores, float("-inf"))

    row_max = tl.max(scores, axis=1)
    safe_max = tl.where(row_max == float("-inf"), 0.0, row_max)
    probs = tl.exp2(scores - safe_max[:, None])
    probs = tl.where(visible, probs, 0.0)
    denom = tl.sum(probs, axis=1)
    safe_denom = tl.where(denom > 0.0, denom, 1.0)

    out = tl.dot(probs.to(v.dtype), v) / safe_denom[:, None]
    lse2 = tl.where(denom > 0.0, safe_max + tl.log2(safe_denom), float("-inf"))
    if HAS_PREFIX:
        # Merge with the prefix part's state; its LSE is in the output basis.
        p_lse2 = tl.load(
            prefix_lse_ptr + q_rows * tl.num_programs(1) + head, mask=row_ok, other=0.0
        )
        p_lse2 = p_lse2 / lse_scale
        p_out = tl.load(
            prefix_out_ptr
            + q_rows[:, None] * stride_pt
            + head * stride_ph
            + dims[None, :],
            mask=row_ok[:, None] & dim_ok[None, :],
            other=0.0,
        ).to(tl.float32)
        top = tl.maximum(lse2, p_lse2)
        w_tree = tl.exp2(lse2 - top)
        w_prefix = tl.exp2(p_lse2 - top)
        total = w_tree + w_prefix
        out = (out * w_tree[:, None] + p_out * w_prefix[:, None]) / total[:, None]
        lse2 = top + tl.log2(total)
    lse = lse2 * lse_scale

    tl.store(
        out_ptr + q_rows[:, None] * stride_ot + head * stride_oh + dims[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=row_ok[:, None] & dim_ok[None, :],
    )
    tl.store(lse_ptr + q_rows * tl.num_programs(1) + head, lse, mask=row_ok)


def tree_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    *,
    rows_per_req: int,
    slots_per_req: int,
    sm_scale: float,
    lse_base2: bool = False,
    prefix: tuple[torch.Tensor, torch.Tensor] | None = None,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked attention of each request's query rows over its tree slots.

    Args:
        q: ``[bs * rows_per_req, num_q_heads, head_dim]`` queries.
        k: ``[bs * slots_per_req, num_kv_heads, head_dim]`` keys.
        v: ``[bs * slots_per_req, num_kv_heads, head_dim]`` values.
        mask: ``[bs * rows_per_req]`` int64; bit ``j`` lets a row see slot ``j``.
        rows_per_req: query rows per request.
        slots_per_req: key/value slots per request, at most 64.
        sm_scale: softmax scale applied to ``q . k``.
        lse_base2: return the LSE in base 2 (trtllm-gen's basis) instead of natural log.
        prefix: optional ``(out, lse)`` attention state of the same rows over
            other keys (LSE in the basis ``lse_base2`` selects, every row seeing
            at least one key); the result is then the merged state.
        out: optional ``[bs * rows_per_req, num_q_heads, head_dim]`` output.
        lse: optional ``[bs * rows_per_req, num_q_heads]`` float32 output.

    Returns:
        ``(out, lse)``: the partial attention output in ``q.dtype`` and its
        LSE (natural log unless ``lse_base2``); rows that see no slot get zero output and ``-inf``.
    """
    if slots_per_req > MAX_TREE_SLOTS:
        raise ValueError(f"tree has {slots_per_req} slots, at most {MAX_TREE_SLOTS}")
    num_rows, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    if num_q_heads % num_kv_heads:
        raise ValueError(f"{num_q_heads} query heads over {num_kv_heads} kv heads")
    bs = num_rows // rows_per_req
    if out is None:
        out = torch.empty_like(q)
    if lse is None:
        lse = torch.empty((num_rows, num_q_heads), dtype=torch.float32, device=q.device)
    if bs == 0:
        return out, lse
    _tree_attention_kernel[(bs, num_q_heads)](
        q,
        k,
        v,
        mask,
        q if prefix is None else prefix[0],
        mask if prefix is None else prefix[1],
        out,
        lse,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        out.stride(0),
        out.stride(1),
        0 if prefix is None else prefix[0].stride(0),
        0 if prefix is None else prefix[0].stride(1),
        sm_scale * 1.4426950408889634,
        1.0 if lse_base2 else 0.6931471805599453,
        ROWS=rows_per_req,
        SLOTS=slots_per_req,
        GROUP=num_q_heads // num_kv_heads,
        HEAD_DIM=head_dim,
        ROWS_PAD=max(16, triton.next_power_of_2(rows_per_req)),
        SLOTS_PAD=max(16, triton.next_power_of_2(slots_per_req)),
        DIM_PAD=max(16, triton.next_power_of_2(head_dim)),
        HAS_PREFIX=prefix is not None,
    )
    return out, lse


@triton.jit(do_not_specialize=["num_splits"])
def _tree_decode_split_kernel(
    q_ptr,  # [bs * N, Hq, D]
    k_ptr,  # [slots, Hkv, D] token rows of the paged cache
    v_ptr,
    table_ptr,  # [bs, max_pages] int32
    seq_lens_ptr,  # [bs] int32, including the N window keys
    mask_ptr,  # [bs * N] int64
    part_out_ptr,  # [bs, Hkv, splits, ROWS_PAD, D] float32
    part_lse_ptr,  # [bs, Hkv, splits, ROWS_PAD] float32, base 2
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_table,
    sm_scale_log2,
    num_splits,
    N: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE: tl.constexpr,
    ROWS_PAD: tl.constexpr,
    ROWS_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    kv_head = tl.program_id(1)
    # Query rows are tiled so the live tile stays within on-chip memory at any N x group.
    num_row_blocks: tl.constexpr = ROWS_PAD // ROWS_BLOCK
    split = tl.program_id(2) // num_row_blocks
    row_block = tl.program_id(2) % num_row_blocks

    # Row r is (node r // GROUP, query head kv_head * GROUP + r % GROUP).
    rows = row_block * ROWS_BLOCK + tl.arange(0, ROWS_BLOCK)
    row_ok = rows < N * GROUP
    node = rows // GROUP
    head = kv_head * GROUP + rows % GROUP
    dims = tl.arange(0, HEAD_DIM)
    q = tl.load(
        q_ptr
        + (req * N + node)[:, None] * stride_qt
        + head[:, None] * stride_qh
        + dims[None, :],
        mask=row_ok[:, None],
        other=0.0,
    )
    bits = tl.load(mask_ptr + req * N + node, mask=row_ok, other=0)

    seq_len = tl.load(seq_lens_ptr + req)
    window = seq_len - N
    span = tl.cdiv(tl.cdiv(seq_len, num_splits), BLOCK) * BLOCK
    start = split * span
    end = tl.minimum(seq_len, start + span)

    run_max = tl.full([ROWS_BLOCK], float("-inf"), tl.float32)
    run_sum = tl.zeros([ROWS_BLOCK], dtype=tl.float32)
    acc = tl.zeros([ROWS_BLOCK, HEAD_DIM], dtype=tl.float32)
    cols = tl.arange(0, BLOCK)
    for tile in range(start, end, BLOCK):
        pos = tile + cols
        pos_ok = pos < end
        page = tl.load(
            table_ptr + req * stride_table + pos // PAGE, mask=pos_ok, other=0
        )
        slot = page.to(tl.int64) * PAGE + pos % PAGE
        k = tl.load(
            k_ptr + slot[:, None] * stride_kt + kv_head * stride_kh + dims[None, :],
            mask=pos_ok[:, None],
            other=0.0,
        )
        v = tl.load(
            v_ptr + slot[:, None] * stride_vt + kv_head * stride_vh + dims[None, :],
            mask=pos_ok[:, None],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k)) * sm_scale_log2
        in_window = pos >= window
        bit = (
            (bits[:, None] >> tl.maximum(pos - window, 0).to(tl.int64)[None, :]) & 1
        ) != 0
        visible = pos_ok[None, :] & (~in_window[None, :] | bit)
        scores = tl.where(visible, scores, float("-inf"))
        new_max = tl.maximum(run_max, tl.max(scores, axis=1))
        safe_max = tl.where(new_max == float("-inf"), 0.0, new_max)
        alpha = tl.exp2(run_max - safe_max)
        probs = tl.exp2(scores - safe_max[:, None])
        run_sum = run_sum * alpha + tl.sum(probs, axis=1)
        acc = acc * alpha[:, None] + tl.dot(probs.to(v.dtype), v)
        run_max = new_max

    part = (req * tl.num_programs(1) + kv_head) * num_splits + split
    safe_sum = tl.where(run_sum > 0.0, run_sum, 1.0)
    tl.store(
        part_out_ptr + (part * ROWS_PAD + rows)[:, None] * HEAD_DIM + dims[None, :],
        acc / safe_sum[:, None],
    )
    lse = tl.where(run_sum > 0.0, run_max + tl.log2(safe_sum), float("-inf"))
    tl.store(part_lse_ptr + part * ROWS_PAD + rows, lse)


@triton.jit(do_not_specialize=["num_splits"])
def _tree_decode_reduce_kernel(
    part_out_ptr,
    part_lse_ptr,
    out_ptr,  # [bs * N, Hq, D]
    stride_ot,
    stride_oh,
    num_splits,
    N: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROWS_PAD: tl.constexpr,
    ROWS_BLOCK: tl.constexpr,
    DIM_BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    kv_head = tl.program_id(1)
    num_dim_blocks: tl.constexpr = HEAD_DIM // DIM_BLOCK
    rows = (tl.program_id(2) // num_dim_blocks) * ROWS_BLOCK + tl.arange(0, ROWS_BLOCK)
    row_ok = rows < N * GROUP
    dims = (tl.program_id(2) % num_dim_blocks) * DIM_BLOCK + tl.arange(0, DIM_BLOCK)
    base = (req * tl.num_programs(1) + kv_head) * num_splits
    top = tl.full([ROWS_BLOCK], float("-inf"), tl.float32)
    for s in range(0, num_splits):
        top = tl.maximum(top, tl.load(part_lse_ptr + (base + s) * ROWS_PAD + rows))
    safe_top = tl.where(top == float("-inf"), 0.0, top)
    total = tl.zeros([ROWS_BLOCK], dtype=tl.float32)
    acc = tl.zeros([ROWS_BLOCK, DIM_BLOCK], dtype=tl.float32)
    for s in range(0, num_splits):
        weight = tl.exp2(
            tl.load(part_lse_ptr + (base + s) * ROWS_PAD + rows) - safe_top
        )
        part = tl.load(
            part_out_ptr
            + ((base + s) * ROWS_PAD + rows)[:, None] * HEAD_DIM
            + dims[None, :]
        )
        acc += part * weight[:, None]
        total += weight
    out = acc / tl.where(total > 0.0, total, 1.0)[:, None]
    node = rows // GROUP
    head = kv_head * GROUP + rows % GROUP
    tl.store(
        out_ptr
        + (req * N + node)[:, None] * stride_ot
        + head[:, None] * stride_oh
        + dims[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=row_ok[:, None],
    )


def tree_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_nodes: int,
    page_size: int,
    sm_scale: float,
    num_splits: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Target verify of a draft tree over a paged KV cache in one pass.

    Each request's last ``num_nodes`` keys are its tree window: query row
    ``i`` sees every earlier key and window key ``j`` when bit ``j`` of its
    mask is set. Split-KV (flash decoding) with all of a KV head's query rows
    in one tile.

    Args:
        q: ``[bs * N, num_q_heads, head_dim]`` queries.
        k_cache: ``[slots, num_kv_heads, head_dim]`` token rows of the cache.
        v_cache: laid out like ``k_cache``.
        page_table: ``[bs, max_pages]`` int32 page ids; slot = page * page_size + offset.
        seq_lens: ``[bs]`` int32 keys per request, the window included (>= N).
        mask: ``[bs * N]`` int64 window visibility per query row.
        num_nodes: ``N``, window width, at most 64.
        page_size: tokens per page.
        sm_scale: softmax scale applied to ``q . k``.
        num_splits: KV splits per (request, KV head).
        out: optional ``[bs * N, num_q_heads, head_dim]`` output.

    Returns:
        The attention output in ``q.dtype``.
    """
    if num_nodes > MAX_TREE_SLOTS:
        raise ValueError(f"tree has {num_nodes} nodes, at most {MAX_TREE_SLOTS}")
    num_rows, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    group = num_q_heads // num_kv_heads
    bs = num_rows // num_nodes
    if out is None:
        out = torch.empty_like(q)
    if bs == 0:
        return out
    rows_pad = max(16, triton.next_power_of_2(num_nodes * group))
    part_out = torch.empty(
        (bs, num_kv_heads, num_splits, rows_pad, head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    part_lse = torch.empty(
        (bs, num_kv_heads, num_splits, rows_pad), dtype=torch.float32, device=q.device
    )
    common = dict(N=num_nodes, GROUP=group, HEAD_DIM=head_dim, ROWS_PAD=rows_pad)
    # At most 16K fp32 accumulator elements per program: 128 rows at head_dim 128, 64 at 256.
    rows_block = min(rows_pad, 16384 // head_dim)
    _tree_decode_split_kernel[
        (bs, num_kv_heads, num_splits * (rows_pad // rows_block))
    ](
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        mask,
        part_out,
        part_lse,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        page_table.stride(0),
        sm_scale * 1.4426950408889634,
        num_splits,
        PAGE=page_size,
        BLOCK=64,
        ROWS_BLOCK=rows_block,
        num_warps=8 if rows_block * head_dim >= 128 * 128 else 4,
        **common,
    )
    dim_block = min(head_dim, 64)
    _tree_decode_reduce_kernel[
        (bs, num_kv_heads, rows_pad // 16 * (head_dim // dim_block))
    ](
        part_out,
        part_lse,
        out,
        out.stride(0),
        out.stride(1),
        num_splits,
        ROWS_BLOCK=16,
        DIM_BLOCK=dim_block,
        **common,
    )
    return out
