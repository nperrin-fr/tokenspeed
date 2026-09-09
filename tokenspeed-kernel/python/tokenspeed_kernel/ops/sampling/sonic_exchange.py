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

"""Symmetric multicast buffers for sonic's vocab-sharded sampling.

With the lm_head vocab-parallel, every rank reduces its logits shard to sonic's packed
top-k slab and publishes it here; the selection then runs on the union of all ranks'
slabs (sonic's sharded scratch layout [W, Z_v, M] per row), so no rank ever gathers
the full logits. Two ping-pong bands: a rank cannot publish round r + 1 before every
peer published round r, because it waited on them to consume round r.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.communication.fabric import fabric_allocation_supported
from tokenspeed_kernel.ops.sampling.triton.exchange import bump_kernel, publish_kernel


def slab_exchange_supported(group: dist.ProcessGroup) -> bool:
    """Whether NVLS multicast can map across group: always within one host, and
    only with the IMEX fabric stack across hosts (where a rendezvous would hang)."""

    if dist.get_world_size(group) <= torch.cuda.device_count():
        return True
    return fabric_allocation_supported(torch.cuda.current_device())


class SlabExchange:
    """The symmetric slab and flag buffers of one TP group, plus the round counter.

    Args:
        group: the TP process group (rendezvous is collective over it).
        rank: this rank within group.
        world: group's size.
        max_rows: the largest row count any call publishes.
        blocks: the largest vocab block count per shard any call publishes.
        max_k: sonic's MAX_K.
        device: the CUDA device owning the buffers.

    Raises:
        RuntimeError: the group has no multicast mapping.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        rank: int,
        world: int,
        max_rows: int,
        blocks: int,
        max_k: int,
        device: torch.device,
    ) -> None:
        self.rank = rank
        self.world = world
        self.band_rows = max_rows
        self.blocks = blocks
        self.max_k = max_k
        self.slab = blocks * max_k
        self.n_flags = triton.next_power_of_2(world * blocks)
        slabs = symm_mem.empty(
            (2, max_rows, world, self.slab), dtype=torch.int32, device=device
        )
        self.flags = symm_mem.empty(
            (2, max_rows, world, blocks), dtype=torch.int32, device=device
        )
        self.flags.zero_()
        self._slab_hdl = symm_mem.rendezvous(slabs, group=group)
        self._flag_hdl = symm_mem.rendezvous(self.flags, group=group)
        if not self._slab_hdl.multicast_ptr or not self._flag_hdl.multicast_ptr:
            raise RuntimeError(
                "the slab exchange needs NVLS multicast across the TP group"
            )
        # Both bands as one [2 * max_rows, W * slab] u32 matrix: the kernels offset by band.
        self.scratch = slabs.view(torch.uint32).view(2 * max_rows, world * self.slab)
        # Flags start at 0, so round 1 is the first one a consumer can observe.
        self.round = torch.ones(1, dtype=torch.int32, device=device)
        dist.barrier(group=group)

    def publish(self, local_slab: torch.Tensor, blocks: int) -> None:
        """Multicast local_slab ([rows, blocks * max_k] u32) into the current band."""

        if blocks > self.blocks:
            raise ValueError(
                f"{blocks} vocab blocks exceed the exchange's {self.blocks}"
            )
        publish_kernel[(local_slab.shape[0], blocks)](
            local_slab,
            self._slab_hdl.multicast_ptr,
            self._flag_hdl.multicast_ptr,
            self.round,
            RANK=self.rank,
            WORLD=self.world,
            BLOCKS=blocks,
            STRIDE_S=self.scratch.stride(0),
            BAND_ROWS=self.band_rows,
            STRIDE_L=local_slab.stride(0),
            MAX_K=self.max_k,
            num_warps=1,
        )

    def bump(self) -> None:
        """Advance the round; call after the step's last consumer launch."""

        bump_kernel[(1,)](self.round, num_warps=1)


__all__ = ["SlabExchange", "slab_exchange_supported"]
