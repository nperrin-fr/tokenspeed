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

"""Descriptor-table ReplaySSM fold, restructured from the SGLang-derived fold."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def kda_replayssm_exact_fold_batched_kernel(
    addresses,
    read_indices,
    write_indices,
    ring_indices,
    accept_lens,
    B,
    stride_state_slot: tl.constexpr,
    stride_rawv_slot: tl.constexpr,
    stride_rawk_slot: tl.constexpr,
    stride_g_slot: tl.constexpr,
    stride_beta_slot: tl.constexpr,
    layers_per_group: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
):
    i_v, i_n, i_lh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_l, i_hv = i_lh // HV, i_lh % HV
    i_h = i_hv // (HV // H)
    ab = i_l * 5
    state = tl.load(addresses + ab).to(tl.pointer_type(tl.float32))
    rawv = tl.load(addresses + ab + 1).to(tl.pointer_type(tl.bfloat16))
    rawk = tl.load(addresses + ab + 2).to(tl.pointer_type(tl.bfloat16))
    gate = tl.load(addresses + ab + 3).to(tl.pointer_type(tl.float32))
    beta = tl.load(addresses + ab + 4).to(tl.pointer_type(tl.float32))
    group = i_l // layers_per_group
    read_page = tl.load(read_indices + group * B + i_n).to(tl.int64)
    write_page = tl.load(write_indices + group * B + i_n).to(tl.int64)
    ring_page = tl.load(ring_indices + i_n).to(tl.int64)
    steps = tl.load(accept_lens + i_n).to(tl.int32)
    if (read_page <= NULL_BLOCK_ID) or (write_page <= NULL_BLOCK_ID) or (steps <= 0):
        return

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    state_off = i_hv * V * K + o_v[None, :] * K + o_k[:, None]
    b_h = tl.load(
        state + read_page * stride_state_slot + state_off,
        mask=mask_h,
        other=0.0,
    ).to(tl.float32)
    for t in range(0, steps):
        phys = t.to(tl.int64)
        b_k = tl.load(
            rawk
            + ring_page * stride_rawk_slot
            + (i_h * MAX_CACHE_LEN + phys) * K
            + o_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        b_v = tl.load(
            rawv
            + ring_page * stride_rawv_slot
            + (i_hv * MAX_CACHE_LEN + phys) * V
            + o_v,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        b_g = tl.load(
            gate
            + ring_page * stride_g_slot
            + (i_hv * MAX_CACHE_LEN + phys) * K
            + o_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        b_beta = tl.load(
            beta
            + ring_page * stride_beta_slot
            + i_hv * MAX_CACHE_LEN
            + phys
        ).to(tl.float32)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_h *= tl.exp(b_g[:, None])
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
    tl.store(
        state + write_page * stride_state_slot + state_off,
        b_h,
        mask=mask_h,
    )


def commit_kda_replayssm_spec_batched(
    addresses: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    ring_indices: torch.Tensor,
    accept_lens: torch.Tensor,
    *,
    max_cache_len: int,
    num_heads: int,
    num_value_heads: int,
    head_dim: int,
    state_stride: int,
    rawv_stride: int,
    rawk_stride: int,
    gate_stride: int,
    beta_stride: int,
    layers_per_group: int,
    null_block_id: int = -1,
) -> None:
    """Fold all descriptor-table layers in one launch with SGLang-equivalent math."""
    layers = addresses.shape[0]
    batch = accept_lens.numel()
    block_v = min(triton.next_power_of_2(head_dim), 32)
    grid = (triton.cdiv(head_dim, block_v), batch, layers * num_value_heads)
    kda_replayssm_exact_fold_batched_kernel[grid](
        addresses,
        read_indices,
        write_indices,
        ring_indices,
        accept_lens,
        batch,
        stride_state_slot=state_stride,
        stride_rawv_slot=rawv_stride,
        stride_rawk_slot=rawk_stride,
        stride_g_slot=gate_stride,
        stride_beta_slot=beta_stride,
        layers_per_group=layers_per_group,
        H=num_heads,
        HV=num_value_heads,
        K=head_dim,
        V=head_dim,
        BK=triton.next_power_of_2(head_dim),
        BV=block_v,
        MAX_CACHE_LEN=max_cache_len,
        NULL_BLOCK_ID=null_block_id,
        num_warps=1,
        num_stages=3,
    )
