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

"""Kernels of the vocab-sharded candidate exchange: every rank multicast-stores its
packed top-k slab into a symmetric buffer and release-stores a round-stamped flag;
sonic's selection kernels (patched) acquire-poll those flags before merging."""

from __future__ import annotations

from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.communication.triton import multimem_st_128


@triton.jit
def fence_sys(value):
    """System-scope acq_rel fence; returns value so the call is not elided."""

    return tl.inline_asm_elementwise(
        "fence.acq_rel.sys; mov.b32 $0, $1;",
        "=r,r",
        [value],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def mc_st_release_i32(mc_ptr, value):
    """Release-store one int32 through a multicast address (lands on every rank)."""

    return tl.inline_asm_elementwise(
        "multimem.st.release.sys.global.b32 [$1], $2;",
        "=r,l,r",
        [mc_ptr, value],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def publish_kernel(
    local_ptr,
    mc_slab_ptr,
    mc_flag_ptr,
    round_ptr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    BLOCKS: tl.constexpr,
    BAND_ROWS: tl.constexpr,
    STRIDE_S: tl.constexpr,
    STRIDE_L: tl.constexpr,
    MAX_K: tl.constexpr,
):
    """One program per (row, block): multicast the block's MAX_K packed entries into
    band round & 1 at [row, RANK, block] (row stride STRIDE_S u32 in the symmetric slab,
    STRIDE_L in the local one), then release the flag with the round."""

    row = tl.program_id(0)
    blk = tl.program_id(1)
    rnd = tl.load(round_ptr)
    band = rnd & 1
    lane = tl.arange(0, MAX_K // 4)
    src = local_ptr + row * STRIDE_L + blk * MAX_K + lane * 4
    x = tl.load(src)
    y = tl.load(src + 1)
    z = tl.load(src + 2)
    w = tl.load(src + 3)
    elem = (
        (band * BAND_ROWS + row) * STRIDE_S
        + RANK * (BLOCKS * MAX_K)
        + blk * MAX_K
        + lane * 4
    )
    dst = mc_slab_ptr.to(tl.int64).to(tl.pointer_type(tl.uint64)) + elem // 2
    multimem_st_128(dst, x, y, z, w, lane >= 0)
    tl.debug_barrier()
    fenced = fence_sys(rnd)
    flag = ((band * BAND_ROWS + row) * WORLD + RANK) * BLOCKS + blk
    mc_st_release_i32(
        mc_flag_ptr.to(tl.int64).to(tl.pointer_type(tl.int32)) + flag, fenced
    )


@triton.jit
def bump_kernel(round_ptr):
    """Advance the round after every consumer of this step has read it."""

    tl.store(round_ptr, tl.load(round_ptr) + 1)
