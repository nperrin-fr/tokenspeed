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

"""Top-``k`` log-probabilities of every row of a logits matrix in one pass.

Equivalent to ``torch.topk(torch.log_softmax(logits.float(), -1), k)`` for
small ``k`` without materialising the float log-softmax: each row streams its
vocabulary once, keeping an online log-sum-exp and a running top-``k``.
Draft-tree expansion calls this once per drafting step.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = ["logprob_topk"]


@triton.jit
def _logprob_topk_kernel(
    logits_ptr,
    scores_ptr,
    ids_ptr,
    stride_row,
    vocab,
    K: tl.constexpr,
    K_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    slots = tl.arange(0, K_PAD)
    # Padding slots hold +inf so the running minimum never picks them.
    best_v = tl.where(slots < K, float("-inf"), float("inf"))
    best_i = tl.zeros([K_PAD], dtype=tl.int64)
    run_max = float("-inf")
    run_sum = 0.0
    for start in range(0, vocab, BLOCK):
        cols = start + offs
        x = tl.load(
            logits_ptr + row * stride_row + cols, mask=cols < vocab, other=float("-inf")
        ).to(tl.float32)
        new_max = tl.maximum(run_max, tl.max(x, axis=0))
        safe_max = tl.where(new_max == float("-inf"), 0.0, new_max)
        run_sum = run_sum * tl.exp(run_max - safe_max) + tl.sum(
            tl.exp(x - safe_max), axis=0
        )
        run_max = new_max
        for _ in tl.static_range(K):
            top = tl.max(x, axis=0)
            kth = tl.min(best_v, axis=0)
            take = top > kth
            col = tl.min(tl.where(x == top, cols, vocab), axis=0)
            slot = tl.min(tl.where(best_v == kth, slots, K_PAD), axis=0)
            hit = (slots == slot) & take
            best_v = tl.where(hit, top, best_v)
            best_i = tl.where(hit, col.to(tl.int64), best_i)
            x = tl.where(cols == col, float("-inf"), x)
    lse = run_max + tl.log(run_sum)
    # Emit best first; ``left`` marks slots not yet written.
    left = slots < K
    for j in tl.static_range(K):
        top = tl.max(tl.where(left, best_v, float("-inf")), axis=0)
        slot = tl.min(tl.where(left & (best_v == top), slots, K_PAD), axis=0)
        tl.store(scores_ptr + row * K + j, top - lse)
        tl.store(
            ids_ptr + row * K + j, tl.sum(tl.where(slots == slot, best_i, 0), axis=0)
        )
        left = left & (slots != slot)


def logprob_topk(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-``k`` log-probabilities per row, best first.

    Args:
        logits: ``[rows, vocab]`` logits with unit stride along the vocabulary.
        k: entries to keep per row, at most 16.

    Returns:
        ``(scores, ids)``: ``[rows, k]`` float32 log-probabilities and int64
        vocabulary ids, sorted by descending score.
    """
    if not 1 <= k <= 16:
        raise ValueError(f"logprob_topk keeps 1..16 entries, got {k}")
    if logits.stride(-1) != 1:
        raise ValueError("logprob_topk needs unit stride along the vocabulary")
    rows, vocab = logits.shape
    scores = torch.empty((rows, k), dtype=torch.float32, device=logits.device)
    ids = torch.empty((rows, k), dtype=torch.int64, device=logits.device)
    if rows == 0:
        return scores, ids
    _logprob_topk_kernel[(rows,)](
        logits,
        scores,
        ids,
        logits.stride(0),
        vocab,
        K=k,
        K_PAD=max(2, triton.next_power_of_2(k)),
        BLOCK=min(4096, triton.next_power_of_2(vocab)),
        num_warps=8,
    )
    return scores, ids
