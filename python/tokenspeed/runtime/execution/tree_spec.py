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

"""Per-step draft-tree state shared by the executor, attention and drafter.

A decode request verifies ``N`` nodes: node 0 the last verified token, the
rest drafted, linked by ``parent``. Positions follow depth, KV slots follow
node index (the verify write window), and after verification the accepted
path is compacted to the front of the request's window -- target KV, target
hidden rows and the packed predictions -- so everything downstream sees a
chain. A chain is the tree ``parent[i] = i - 1``; with ``topk == 1`` none of
this state exists and the chain path runs unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.sampling.triton.draft_tree import tree_ancestry

from tokenspeed.runtime.utils.triton import tl, triton

__all__ = ["TreeSpec", "TreeSpecConfig"]


@triton.jit
def _compact_kv_kernel(
    buffers_ptr,  # [num_buffers] int64 base addresses of the per-layer K/V buffers
    locs_ptr,  # [bs * N] int32 verify write slots
    path_ptr,  # [bs, N] int32 accepted path, -1 past it
    ROW: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    buf = tl.load(buffers_ptr + tl.program_id(0)).to(tl.pointer_type(tl.bfloat16))
    req = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    # The path is increasing, so row d never overwrites a later row's source.
    for d in range(N):
        src = tl.load(path_ptr + req * N + d)
        if (src >= 0) & (src != d):
            src_loc = tl.load(locs_ptr + req * N + src).to(tl.int64)
            dst_loc = tl.load(locs_ptr + req * N + d).to(tl.int64)
            for start in range(0, ROW, BLOCK):
                cols = start + offs
                row = tl.load(buf + src_loc * ROW + cols, mask=cols < ROW)
                tl.store(buf + dst_loc * ROW + cols, row, mask=cols < ROW)


@dataclass(frozen=True)
class TreeSpecConfig:
    topk: int
    num_steps: int
    num_nodes: int


class TreeSpec:
    """Device buffers for the tree being verified this step."""

    def __init__(
        self, config: TreeSpecConfig, max_bs: int, pool_size: int, device: torch.device
    ) -> None:
        self.config = config
        n = config.num_nodes
        self.chain_parent = torch.arange(-1, n - 1, dtype=torch.int32, device=device)
        # Next round's tree per request pool slot; the drafter writes it next to future_input_map.
        self.future_parent_map = self.chain_parent.repeat(pool_size, 1)
        self.parent_buf = self.chain_parent.repeat(max_bs, 1)
        # The chain the parents describe, so graph warmup's chain_positions (no load_step) is a no-op.
        self.depth_buf = torch.arange(n, dtype=torch.int32, device=device).repeat(
            max_bs, 1
        )
        chain_mask = [(1 << (i + 1)) - 1 for i in range(n)]
        chain_mask = [
            m - (1 << 64) if m >> 63 else m for m in chain_mask
        ]  # bit 63 is the sign
        self.mask_buf = torch.tensor(
            chain_mask, dtype=torch.int64, device=device
        ).repeat(max_bs)
        # Verify writes the accepted path here (root first, -1 past it).
        self.path_buf = torch.full((max_bs, n), -1, dtype=torch.int32, device=device)
        self._node_offsets = torch.arange(n, dtype=torch.int64, device=device)
        # Base addresses of the target's per-layer K/V buffers (bind_kv).
        self.kv_buffer_ptrs: torch.Tensor | None = None
        self.kv_row_elems = 0

    @property
    def num_nodes(self) -> int:
        return self.config.num_nodes

    @property
    def max_depth(self) -> int:
        return self.config.num_steps

    def load_step(self, bs: int, req_pool_indices: torch.Tensor) -> None:
        """Read this step's trees and derive depth and ancestor masks."""
        parent = self.parent_buf[:bs]
        torch.index_select(self.future_parent_map, 0, req_pool_indices, out=parent)
        tree_ancestry(
            parent,
            self.max_depth,
            self.depth_buf[:bs],
            self.mask_buf[: bs * self.num_nodes].view(bs, self.num_nodes),
        )

    def depth_positions(self, bs: int, positions: torch.Tensor) -> None:
        """Turn the verify window's ``vc + i`` positions into ``vc + depth``."""
        view = positions.view(bs, self.num_nodes)
        view.add_(self.depth_buf[:bs] - self._node_offsets.to(view.dtype))

    def chain_positions(self, bs: int, positions: torch.Tensor) -> None:
        """Back to ``vc + i``: after compaction row ``i`` holds the path node at depth ``i``."""
        view = positions.view(bs, self.num_nodes)
        view.add_(self._node_offsets.to(view.dtype) - self.depth_buf[:bs])

    def path_rows(self, bs: int) -> torch.Tensor:
        """``[bs * N]`` source row of each packed row: the accepted path first, then identity."""
        n = self.num_nodes
        path = self.path_buf[:bs].long()
        local = torch.where(path >= 0, path, self._node_offsets)
        return (local + torch.arange(bs, device=path.device)[:, None] * n).view(-1)

    def compact_rows(self, bs: int, rows: torch.Tensor) -> None:
        """Move each request's accepted-path rows to the front of its window, in place."""
        src = self.path_rows(bs)
        rows[: bs * self.num_nodes].copy_(rows.index_select(0, src))

    def bind_kv(self, kv_buffers: list[torch.Tensor]) -> None:
        """Record the target's K/V buffers compaction moves rows in.

        Args:
            kv_buffers: per-layer bf16 K and V buffers of contiguous token rows.
        """
        rows = {buf[0].numel() for buf in kv_buffers}
        if len(rows) != 1 or any(
            buf.dtype != torch.bfloat16 or not buf.is_contiguous() for buf in kv_buffers
        ):
            raise NotImplementedError(
                "draft-tree KV compaction needs contiguous bf16 K/V rows of one width"
            )
        self.kv_row_elems = rows.pop()
        self.kv_buffer_ptrs = torch.tensor(
            [buf.data_ptr() for buf in kv_buffers],
            dtype=torch.int64,
            device=kv_buffers[0].device,
        )

    def compact_kv(self, bs: int, write_locations: torch.Tensor) -> None:
        """Copy the accepted path's KV into the window's leading slots, every layer.

        Args:
            write_locations: ``[bs * N]`` verify write slots of the window.
        """
        if bs == 0:
            return
        _compact_kv_kernel[(self.kv_buffer_ptrs.shape[0], bs)](
            self.kv_buffer_ptrs,
            write_locations,
            self.path_buf,
            ROW=self.kv_row_elems,
            N=self.num_nodes,
            BLOCK=min(triton.next_power_of_2(self.kv_row_elems), 1024),
        )
