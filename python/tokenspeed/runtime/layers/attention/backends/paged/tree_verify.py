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

"""Draft-tree inputs the attention leaves read.

Target verify: every query row of a request sees the whole committed prefix
and, among the verify window's ``N`` nodes, exactly the ancestors named by its
mask. Drafting: ``K`` lane rows per request over the accepted prefix and a side
buffer of their ancestors' K/V. Leaves without a tree path keep these unset
and the service refuses tree drafting for them.
"""

from __future__ import annotations

import torch

__all__ = ["TreeDraftInputs", "TreeVerifyInputs"]


class TreeVerifyInputs:
    """Per-step tree metadata the verify leaves read (one object, shared)."""

    def __init__(
        self,
        mask: torch.Tensor,
        num_nodes: int,
        parent: torch.Tensor,
    ) -> None:
        """Args:
        mask: ``[max_bs * N]`` int64 ancestor-or-self mask per window row,
            refreshed every step by the executor.
        num_nodes: ``N``, verify window width.
        parent: ``[max_bs, N]`` int32 parent node per window row (``-1`` for
            the root), refreshed every step by the executor.
        """
        self.mask = mask
        self.num_nodes = num_nodes
        self.parent = parent


class TreeDraftInputs:
    """Drafting lanes of a draft tree: ``K`` query rows per request per step.

    A lane row sees the committed prefix (``prefix_lens``, the accepted
    frontier) and, in a per-layer side buffer of ``(S - 1) * K`` slots per
    request, the draft K/V of its own ancestors. Lane K/V never enter the
    paged cache: they are only valid while this round's tree is drafted.
    """

    def __init__(
        self,
        topk: int,
        num_steps: int,
        max_bs: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.topk = topk
        self.num_slots = (num_steps - 1) * topk
        # The drafting step being run (1 .. S - 1); None outside the lane loop.
        self.step: int | None = None
        shape = (max_bs * self.num_slots, num_kv_heads, head_dim)
        self.side_k = [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)
        ]
        self.side_v = [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)
        ]
        self.lane_mask = torch.zeros((max_bs * topk,), dtype=torch.int64, device=device)
        self.prefix_lens = torch.ones((max_bs,), dtype=torch.int32, device=device)
        self.cu_prefix_lens = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=device
        )
        self.cu_query_lens = torch.arange(
            0, (max_bs + 1) * topk, topk, dtype=torch.int32, device=device
        )
        lanes = torch.arange(topk, dtype=torch.int64, device=device)
        requests = torch.arange(max_bs, dtype=torch.int64, device=device)[:, None]
        # Side-buffer row of (request, lane) at each step.
        self.slot_rows = [
            (requests * self.num_slots + (step - 1) * topk + lanes).view(-1)
            for step in range(1, num_steps)
        ]

    def set_prefix(self, bs: int, frontier: torch.Tensor) -> None:
        torch.clamp_min(frontier[:bs].to(torch.int32), 1, out=self.prefix_lens[:bs])
        torch.cumsum(self.prefix_lens[:bs], 0, out=self.cu_prefix_lens[1 : bs + 1])
