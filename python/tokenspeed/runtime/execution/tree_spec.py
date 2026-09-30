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
from tokenspeed_kernel.ops.sampling.triton.tree_verify import verify_tree

from tokenspeed.runtime.utils.triton import tl, triton

__all__ = ["TreeSpec", "TreeSpecConfig"]


@triton.jit
def _compact_tree_kernel(
    buffers_ptr,  # [NUM_KV] int64 base addresses of the per-layer K/V buffers
    locs_ptr,  # [bs * N] int32 verify write slots
    path_ptr,  # [bs, N] int32 accepted path, -1 past it
    hidden_ptr,  # [bs * N, hidden] verify hidden rows
    stride_hidden,
    positions_ptr,  # [bs * N] int64 window positions
    depth_ptr,  # [bs, N] int32 node depth
    NUM_KV: tl.constexpr,
    ROW: tl.constexpr,
    HIDDEN: tl.constexpr,
    N: tl.constexpr,
    N_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Program (i, req): K/V buffer i < NUM_KV, the hidden rows at NUM_KV, the
    positions at NUM_KV + 1 -- each packs request req's accepted path to the
    front of its window."""
    which = tl.program_id(0)
    req = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    if which == NUM_KV + 1:
        # Row i now holds the path node at depth i: back from vc + depth to vc + i.
        nodes = tl.arange(0, N_PAD)
        ok = nodes < N
        depth = tl.load(depth_ptr + req * N + nodes, mask=ok, other=0)
        pos = tl.load(positions_ptr + req * N + nodes, mask=ok, other=0)
        tl.store(
            positions_ptr + req * N + nodes, pos + (nodes - depth).to(tl.int64), mask=ok
        )
    else:
        # The path is increasing, so row d never overwrites a later row's source.
        for d in range(N):
            src = tl.load(path_ptr + req * N + d)
            if (src >= 0) & (src != d):
                if which < NUM_KV:
                    buf = tl.load(buffers_ptr + which).to(tl.pointer_type(tl.bfloat16))
                    src_row = buf + tl.load(locs_ptr + req * N + src).to(tl.int64) * ROW
                    dst_row = buf + tl.load(locs_ptr + req * N + d).to(tl.int64) * ROW
                    width = ROW
                else:
                    src_row = hidden_ptr + (req * N + src).to(tl.int64) * stride_hidden
                    dst_row = hidden_ptr + (req * N + d).to(tl.int64) * stride_hidden
                    width = HIDDEN
                for start in range(0, width, BLOCK):
                    cols = start + offs
                    row = tl.load(src_row + cols, mask=cols < width)
                    tl.store(dst_row + cols, row, mask=cols < width)


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
        # The tree the drafter built this round; the executor publishes it to future_parent_map.
        self.draft_parent_buf = self.chain_parent.repeat(max_bs, 1)
        # The chain the parents describe, so graph warmup's compact (no load_step) is a no-op.
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
        # The accepted path (root first, -1 past it), stored by record_path.
        self.path_buf = torch.full((max_bs, n), -1, dtype=torch.int32, device=device)
        self._node_offsets = torch.arange(n, dtype=torch.int64, device=device)
        # Base addresses of the target's per-layer K/V buffers (bind_kv).
        self.kv_buffer_ptrs: torch.Tensor | None = None
        self.kv_row_elems = 0

    @property
    def num_nodes(self) -> int:
        return self.config.num_nodes

    def load_step(self, bs: int, req_pool_indices: torch.Tensor) -> None:
        """Read this step's trees and derive depth and ancestor masks."""
        parent = self.parent_buf[:bs]
        torch.index_select(self.future_parent_map, 0, req_pool_indices, out=parent)
        tree_ancestry(
            parent,
            self.depth_buf[:bs],
            self.mask_buf[: bs * self.num_nodes].view(bs, self.num_nodes),
        )

    def depth_positions(self, bs: int, positions: torch.Tensor) -> None:
        """Turn the verify window's ``vc + i`` positions into ``vc + depth``."""
        view = positions.view(bs, self.num_nodes)
        view.add_(self.depth_buf[:bs] - self._node_offsets.to(view.dtype))

    def verify(
        self,
        bs: int,
        predict: torch.Tensor,
        accept_length: torch.Tensor,
        accept_index: torch.Tensor,
        candidates: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Accept the longest root path whose drafts match the target's picks.

        Args:
            predict: ``[bs * N]`` int32 output, the picks packed along the path.
            accept_length: ``[bs]`` int32 output, accepted drafts plus the bonus.
            accept_index: ``[bs, N]`` int32 output, the accepted path (root
                first, ``-1`` past it).
            candidates: ``[bs, N]`` node tokens, node 0 the root.
            target: ``[bs * N]`` int32 target pick at every node.
        """
        verify_tree(
            predict,
            accept_length,
            accept_index,
            candidates.to(torch.int32),
            self.parent_buf[:bs],
            target,
        )

    def record_path(self, bs: int, accept_index: torch.Tensor) -> None:
        """Store the accepted path (after the TP broadcast) for compaction."""
        self.path_buf[:bs].copy_(accept_index)

    def path_rows(self, path: torch.Tensor) -> torch.Tensor:
        """``[bs * N]`` source row of each packed row: the accepted path first, then identity.

        Args:
            path: ``[bs, N]`` accepted path, root first, ``-1`` past it.
        """
        bs, n = path.shape
        path = path.long()
        local = torch.where(path >= 0, path, self._node_offsets)
        return (local + torch.arange(bs, device=path.device)[:, None] * n).view(-1)

    def bind_kv(self, kv_buffers: list[torch.Tensor]) -> None:
        """Record the target's K/V buffers compaction moves rows in.

        Args:
            kv_buffers: the attention layers' bf16 K and V buffers of contiguous
                token rows (state layers own none).
        """
        rows = {buf[0].numel() for buf in kv_buffers}
        if len(rows) != 1 or any(
            buf.dtype != torch.bfloat16 or not buf.is_contiguous() for buf in kv_buffers
        ):
            raise NotImplementedError(
                "draft-tree KV compaction needs contiguous bf16 K/V rows of one width"
            )
        self.kv_row_elems = rows.pop()
        # Layers may alias one region through the memory plan; move each region once.
        self.kv_buffer_ptrs = torch.tensor(
            sorted({buf.data_ptr() for buf in kv_buffers}),
            dtype=torch.int64,
            device=kv_buffers[0].device,
        )

    def compact(
        self,
        bs: int,
        write_locations: torch.Tensor,
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Pack each request's accepted path to the front of its verify window,
        in one launch: target KV (every layer), hidden rows, and the positions
        back to ``vc + i`` (row ``i`` then holds the path node at depth ``i``).

        Args:
            write_locations: ``[bs * N]`` verify write slots of the window.
            hidden: ``[bs * N, hidden]`` verify hidden rows, compacted in place.
            positions: ``[bs * N]`` int64 window positions, shifted in place.
        """
        if bs == 0:
            return
        n = self.num_nodes
        _compact_tree_kernel[(self.kv_buffer_ptrs.shape[0] + 2, bs)](
            self.kv_buffer_ptrs,
            write_locations,
            self.path_buf,
            hidden,
            hidden.stride(0),
            positions,
            self.depth_buf,
            NUM_KV=self.kv_buffer_ptrs.shape[0],
            ROW=self.kv_row_elems,
            HIDDEN=hidden.shape[1],
            N=n,
            N_PAD=max(16, triton.next_power_of_2(n)),
            BLOCK=1024,
        )
