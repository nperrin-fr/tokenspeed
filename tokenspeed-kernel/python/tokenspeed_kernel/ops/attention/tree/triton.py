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

__all__ = ["tree_attention"]

MAX_TREE_SLOTS = 64


@triton.jit
def _tree_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mask_ptr,
    kv_rows_ptr,
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
    HAS_KV_ROWS: tl.constexpr,
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
    if HAS_KV_ROWS:
        kv_rows = tl.load(kv_rows_ptr + kv_rows, mask=slot_ok, other=0).to(tl.int64)

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
    kv_rows: torch.Tensor | None = None,
    prefix: tuple[torch.Tensor, torch.Tensor] | None = None,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked attention of each request's query rows over its tree slots.

    Args:
        q: ``[bs * rows_per_req, num_q_heads, head_dim]`` queries.
        k: ``[bs * slots_per_req, num_kv_heads, head_dim]`` keys, or any
            ``[rows, num_kv_heads, head_dim]`` store when ``kv_rows`` is given.
        v: values, laid out like ``k``.
        mask: ``[bs * rows_per_req]`` int64; bit ``j`` lets a row see slot ``j``.
        rows_per_req: query rows per request.
        slots_per_req: key/value slots per request, at most 64.
        sm_scale: softmax scale applied to ``q . k``.
        lse_base2: return the LSE in base 2 (trtllm-gen's basis) instead of natural log.
        kv_rows: optional ``[bs * slots_per_req]`` integer row of each slot in
            ``k``/``v`` (e.g. the KV cache slots); slots are dense when omitted.
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
    if kv_rows is not None and kv_rows.stride(0) != 1:
        raise ValueError("kv_rows must be contiguous")
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
        mask if kv_rows is None else kv_rows,
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
        HAS_KV_ROWS=kv_rows is not None,
        HAS_PREFIX=prefix is not None,
    )
    return out, lse
