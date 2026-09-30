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

"""causal_conv1d_update over a draft tree: each token's window ends at its parent."""

import pytest
import torch

from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
    causal_conv1d_update,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

PARENTS = [
    [-1, 0, 1, 1, 0, 4],
    [-1, 0, 0, 2, 2, 0],
    [-1, 0, 1, 2, 3, 4],
]


def _run(x, conv_state, weight, bias, base, out_rows, parents):
    return causal_conv1d_update(
        x.clone(),
        conv_state,
        weight,
        bias,
        activation="silu",
        conv_state_indices=base,
        output_state_indices=out_rows,
        parent_indices=parents,
    )


def test_tree_windows_match_per_path_reference():
    torch.manual_seed(0)
    bs, dim, width, t = len(PARENTS), 256, 4, len(PARENTS[0])
    x = torch.randn(bs, dim, t, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, width, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
    rows = 1 + bs * (t + 1)
    conv_state = torch.randn(rows, dim, width - 1, device="cuda", dtype=torch.bfloat16)
    base = torch.arange(bs, device="cuda", dtype=torch.int32) * (t + 1) + 1
    out_rows = (
        base[:, None] + 1 + torch.arange(t, device="cuda", dtype=torch.int32)
    ).contiguous()
    parents = torch.tensor(PARENTS, device="cuda", dtype=torch.int32)
    init = conv_state[base.long()].float().clone()

    state = conv_state.clone()
    out = _run(x, state, weight, bias, base, out_rows, parents)

    for b, par in enumerate(PARENTS):
        for i in range(t):
            path, node = [], i
            while node >= 0:
                path.append(node)
                node = par[node]
            seq = torch.cat(
                [init[b], x[b, :, path[::-1]].float()], dim=1
            )  # [dim, 3 + depth + 1]
            ref = (seq[:, -width:] * weight.float()).sum(-1) + bias.float()
            ref = ref * torch.sigmoid(ref)
            torch.testing.assert_close(out[b, :, i].float(), ref, atol=3e-2, rtol=2e-2)
            torch.testing.assert_close(
                state[int(out_rows[b, i])].float(),
                seq[:, -(width - 1) :],
                atol=0,
                rtol=0,
            )

    # A chain given as parents never reloads: identical to the parent-less update.
    chain_state = conv_state.clone()
    chain_out = _run(x, chain_state, weight, bias, base, out_rows, None)
    assert torch.equal(out[2], chain_out[2])
    assert torch.equal(state[out_rows[2].long()], chain_state[out_rows[2].long()])
