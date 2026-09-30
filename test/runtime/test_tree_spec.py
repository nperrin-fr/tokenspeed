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

"""TreeSpec compaction against a plain per-request reference."""

import pytest
import torch

from tokenspeed.runtime.execution.tree_spec import TreeSpec, TreeSpecConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("nodes", [8, 64])
def test_compact_packs_kv_hidden_and_positions(nodes):
    gen = torch.Generator().manual_seed(nodes)
    bs, slots, layers, heads, dim, hidden_size = 3, 4096, 3, 2, 128, 3000
    spec = TreeSpec(
        TreeSpecConfig(topk=4, num_steps=5, num_nodes=nodes), bs, 8, torch.device("cuda")
    )
    buffers = [
        torch.randn(slots, heads, dim, generator=gen).bfloat16().cuda()
        for _ in range(2 * layers)
    ]
    spec.bind_kv(buffers)
    locs = torch.randperm(slots, generator=gen)[: bs * nodes].int().cuda()
    paths = [[0, 2, 5], [0, 1, 2, 3], [0]]  # jump, identity, root only
    spec.path_buf.fill_(-1)
    for b, path in enumerate(paths):
        spec.path_buf[b, : len(path)] = torch.tensor(path, dtype=torch.int32)
    spec.depth_buf[:bs] = torch.randint(0, 5, (bs, nodes), generator=gen).int().cuda()
    hidden = torch.randn(bs * nodes, hidden_size, generator=gen).bfloat16().cuda()
    positions = torch.randint(0, 1000, (bs * nodes,), generator=gen).cuda()

    expected = [buf.clone() for buf in buffers]
    for buf in expected:
        for b, path in enumerate(paths):
            window = locs[b * nodes : (b + 1) * nodes].long()
            buf[window[: len(path)]] = buf[window[torch.tensor(path)]]
    want_hidden = hidden.clone()
    for b, path in enumerate(paths):
        rows = b * nodes + torch.tensor(path)
        want_hidden[b * nodes : b * nodes + len(path)] = hidden[rows]
    offsets = torch.arange(nodes, device="cuda").repeat(bs)
    want_positions = positions + offsets - spec.depth_buf[:bs].reshape(-1).long()

    spec.compact(bs, locs, hidden, positions)

    for got, want in zip(buffers, expected):
        assert torch.equal(got, want)
    assert torch.equal(hidden, want_hidden)
    assert torch.equal(positions, want_positions)


@pytest.mark.parametrize("nodes", [8, 64])
def test_fresh_spec_is_the_chain(nodes):
    """Graph warmup runs compact without a load_step; it must leave positions alone."""
    spec = TreeSpec(
        TreeSpecConfig(topk=1, num_steps=nodes - 1, num_nodes=nodes),
        3,
        8,
        torch.device("cuda"),
    )
    positions = torch.arange(3 * nodes, dtype=torch.int64, device="cuda") + 100
    before = positions.clone()
    spec.bind_kv([torch.zeros(16, 1, 128, dtype=torch.bfloat16, device="cuda")])
    hidden = torch.randn(3 * nodes, 64, device="cuda").bfloat16()
    spec.compact(3, torch.zeros(3 * nodes, dtype=torch.int32, device="cuda"), hidden, positions)
    assert torch.equal(positions, before)
    fresh = (spec.depth_buf.clone(), spec.mask_buf.clone())
    spec.load_step(3, torch.tensor([0, 1, 2], device="cuda"))
    assert torch.equal(spec.depth_buf, fresh[0])
    assert torch.equal(spec.mask_buf, fresh[1])
