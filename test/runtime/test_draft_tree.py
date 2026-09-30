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

"""DraftTree against a plain per-request reference of EAGLE-2 tree drafting."""

import pytest
import torch
from tokenspeed_kernel.ops.sampling.triton.draft_tree import tree_ancestry

from tokenspeed.runtime.execution.drafter.tree import DraftTree

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEVICE = "cuda"


def _drive(bs, topk, steps, nodes, vocab, seed):
    """Run DraftTree on random log-probs; return its output and the entry record."""
    gen = torch.Generator().manual_seed(seed)
    tree = DraftTree(bs, topk, steps, nodes, torch.device(DEVICE))
    record = []  # per request: list of (token path tuple, score)
    lane_paths = [[None] * topk for _ in range(bs)]
    logp = torch.log_softmax(torch.randn(bs, vocab, generator=gen) * 3, -1).to(DEVICE)
    exp = tree.seed(bs, logp)
    for b in range(bs):
        sc, tk = torch.topk(logp[b].float(), topk)
        record.append([((int(t),), float(s)) for t, s in zip(tk, sc)])
        lane_paths[b] = [((int(t),), float(s)) for t, s in zip(tk, sc)]
    for step in range(1, steps):
        logp = torch.log_softmax(
            torch.randn(bs * topk, vocab, generator=gen) * 3, -1
        ).to(DEVICE)
        exp = tree.expand(bs, step, logp)
        for b in range(bs):
            cands = []
            for lane in range(topk):
                path, score = lane_paths[b][lane]
                sc, tk = torch.topk(logp[b * topk + lane].float(), topk)
                for t, s in zip(tk, sc):
                    cands.append((path + (int(t),), score + float(s)))
            record[b].extend(cands)
            best = sorted(range(len(cands)), key=lambda i: -cands[i][1])[:topk]
            lane_paths[b] = [cands[i] for i in best]
            got = exp.lane_tokens[b].tolist()
            assert got == [cands[i][0][-1] for i in best]
    # A strided column, as the drafter passes it.
    roots = (torch.arange(bs, device=DEVICE, dtype=torch.int32) + 7)[:, None].repeat(
        1, 3
    )[:, 0]
    tokens, parent = tree.finalize(bs, roots)
    return tokens.cpu(), parent.cpu(), record


@pytest.mark.parametrize(
    "bs,topk,steps,nodes", [(3, 1, 5, 6), (4, 4, 4, 16), (2, 8, 5, 64), (3, 3, 6, 20)]
)
def test_draft_tree_structure(bs, topk, steps, nodes):
    tokens, parent, record = _drive(bs, topk, steps, nodes, vocab=50, seed=nodes)
    depth = torch.empty_like(parent, device=DEVICE)
    mask = torch.empty(parent.shape, dtype=torch.int64, device=DEVICE)
    tree_ancestry(parent.to(DEVICE), steps, depth, mask)
    depth, mask = depth.cpu(), mask.cpu()
    for b in range(bs):
        assert tokens[b, 0] == 7 + b and parent[b, 0] == -1
        paths = {0: ()}
        for j in range(1, nodes):
            p = int(parent[b, j])
            assert 0 <= p < j, "parents precede children"
            paths[j] = paths[p] + (int(tokens[b, j]),)
            assert depth[b, j] == len(paths[j])
            anc, cur = 0, j
            while cur >= 0:
                anc |= 1 << cur
                cur = int(parent[b, cur])
            assert int(mask[b, j]) & ((1 << 64) - 1) == anc  # bit 63 is the int64 sign
        # Kept nodes are the best N - 1 candidates (depth tie-break as in finalize).
        ranked = sorted(record[b], key=lambda e: -(e[1] - 1e-6 * len(e[0])))
        assert set(paths[j] for j in range(1, nodes)) == {
            e[0] for e in ranked[: nodes - 1]
        }
        # Numbering is depth-first pre-order with children best first.
        score = {e[0]: e[1] for e in record[b]}
        kept = set(paths.values())
        order = []

        def visit(path):
            order.append(path)
            kids = [q for q in kept if len(q) == len(path) + 1 and q[:-1] == path]
            for kid in sorted(kids, key=lambda q: -score[q]):
                visit(kid)

        visit(())
        assert order == [paths[j] for j in range(nodes)]
        # The best path is nodes 1, 2, ...: each node's first kept child is its best.
        node = 0
        while True:
            kids = [j for j in range(1, nodes) if int(parent[b, j]) == node]
            if not kids:
                break
            assert kids[0] == node + 1
            assert max(kids, key=lambda j: score[paths[j]]) == kids[0]
            node = kids[0]
    if topk == 1:
        assert torch.equal(parent, torch.arange(-1, nodes - 1).expand(bs, -1).int())


def test_nan_scores_keep_lanes_and_tree_valid():
    """A padded request's garbage logits must not leave lanes or nodes unset."""
    bs, topk, steps, nodes, vocab = 2, 4, 3, 8, 50
    tree = DraftTree(bs, topk, steps, nodes, torch.device(DEVICE))
    logits = torch.randn(bs, vocab, device=DEVICE)
    logits[1] = float("nan")
    tree.seed(bs, logits)
    lane_logits = torch.randn(bs * topk, vocab, device=DEVICE)
    lane_logits[topk:] = float("nan")
    for step in range(1, steps):
        exp = tree.expand(bs, step, lane_logits)
        assert exp.parent_lane.min() >= 0 and exp.parent_lane.max() < topk
    tokens, parent = tree.finalize(
        bs, torch.zeros(bs, dtype=torch.int32, device=DEVICE)
    )
    for b in range(bs):
        for j in range(1, nodes):
            assert 0 <= int(parent[b, j]) < j
