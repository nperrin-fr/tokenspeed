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

"""Tree speculative decoding kernels against plain references."""

import random

import pytest
import torch
from tokenspeed_kernel.ops.attention import attn_merge_state
from tokenspeed_kernel.ops.attention.tree import tree_attention, tree_decode_attention
from tokenspeed_kernel.ops.sampling.triton.logprob_topk import logprob_topk
from tokenspeed_kernel.ops.sampling.triton.tree_verify import verify_tree_greedy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _reference_tree_attention(q, k, v, mask, rows, slots, scale):
    bs = q.shape[0] // rows
    group = q.shape[1] // k.shape[1]
    kk = k.float().repeat_interleave(group, dim=1).view(bs, slots, q.shape[1], -1)
    vv = v.float().repeat_interleave(group, dim=1).view(bs, slots, q.shape[1], -1)
    qq = q.float().view(bs, rows, q.shape[1], -1)
    scores = torch.einsum("brhd,bshd->bhrs", qq, kk) * scale
    bits = torch.arange(slots, device=q.device)
    visible = ((mask.view(bs, rows, 1) >> bits) & 1).bool()  # [bs, rows, slots]
    scores = scores.masked_fill(~visible[:, None], float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)  # [bs, h, rows]
    probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    out = torch.einsum("bhrs,bshd->brhd", probs, vv)
    return out.reshape(bs * rows, q.shape[1], -1), lse.permute(0, 2, 1).reshape(
        bs * rows, -1
    )


def _random_masks(bs, rows, slots, gen):
    mask = torch.randint(0, 2**62, (bs * rows,), generator=gen, dtype=torch.int64)
    return (mask & ((1 << slots) - 1)).cuda()


@pytest.mark.parametrize("rows,slots", [(4, 4), (17, 17), (64, 64), (4, 40), (1, 7)])
def test_tree_attention_matches_reference(rows, slots):
    gen = torch.Generator().manual_seed(rows * 100 + slots)
    bs, hq, hkv, d = 3, 32, 4, 128
    q = torch.randn(bs * rows, hq, d, generator=gen).bfloat16().cuda()
    k = torch.randn(bs * slots, hkv, d, generator=gen).bfloat16().cuda()
    v = torch.randn(bs * slots, hkv, d, generator=gen).bfloat16().cuda()
    mask = _random_masks(bs, rows, slots, gen)
    mask[0] = 0  # a row that sees nothing
    scale = d**-0.5
    out, lse = tree_attention(
        q, k, v, mask, rows_per_req=rows, slots_per_req=slots, sm_scale=scale
    )
    ref_out, ref_lse = _reference_tree_attention(q, k, v, mask, rows, slots, scale)
    torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=2e-2)
    seen = torch.isfinite(ref_lse)
    assert torch.equal(seen, torch.isfinite(lse))
    torch.testing.assert_close(lse[seen], ref_lse[seen], atol=1e-3, rtol=1e-3)
    assert torch.all(out[0] == 0)


def test_merge_with_prefix_equals_full_attention():
    gen = torch.Generator().manual_seed(7)
    bs, n, prefix, hq, hkv, d = 2, 8, 33, 8, 2, 64
    scale = d**-0.5
    q = torch.randn(bs * n, hq, d, generator=gen).bfloat16().cuda()
    kp = torch.randn(bs, prefix, hkv, d, generator=gen).bfloat16().cuda()
    vp = torch.randn(bs, prefix, hkv, d, generator=gen).bfloat16().cuda()
    kt = torch.randn(bs * n, hkv, d, generator=gen).bfloat16().cuda()
    vt = torch.randn(bs * n, hkv, d, generator=gen).bfloat16().cuda()
    mask = _random_masks(bs, n, n, gen) | (1 << torch.arange(n).repeat(bs)).cuda()
    # Prefix part: every row sees every prefix key (reference in fp32).
    group = hq // hkv
    qq = q.float().view(bs, n, hq, d)
    kk = kp.float().repeat_interleave(group, 2)
    vv = vp.float().repeat_interleave(group, 2)
    s = torch.einsum("bnhd,bphd->bhnp", qq, kk) * scale
    pre_lse = torch.logsumexp(s, -1).permute(0, 2, 1).reshape(bs * n, hq).contiguous()
    pre_out = torch.einsum("bhnp,bphd->bnhd", torch.softmax(s, -1), vv).reshape(
        bs * n, hq, d
    )
    tree_out, tree_lse = tree_attention(
        q, kt, vt, mask, rows_per_req=n, slots_per_req=n, sm_scale=scale
    )
    out, _ = attn_merge_state(
        pre_out.bfloat16().contiguous(), pre_lse, tree_out, tree_lse
    )
    # Full reference over prefix + visible tree slots.
    kt4 = kt.float().repeat_interleave(group, 1).view(bs, n, hq, d)
    vt4 = vt.float().repeat_interleave(group, 1).view(bs, n, hq, d)
    st = torch.einsum("bnhd,bmhd->bhnm", qq, kt4) * scale
    vis = ((mask.view(bs, n, 1) >> torch.arange(n, device="cuda")) & 1).bool()
    st = st.masked_fill(~vis[:, None], float("-inf"))
    full = torch.softmax(torch.cat([s, st], -1), -1)
    ref = torch.einsum("bhnx,bxhd->bnhd", full, torch.cat([vv, vt4], 1)).reshape(
        bs * n, hq, d
    )
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    for base2 in (False, True):
        scale = 1.4426950408889634 if base2 else 1.0
        fused, fused_lse = tree_attention(
            q,
            kt,
            vt,
            mask,
            rows_per_req=n,
            slots_per_req=n,
            sm_scale=d**-0.5,
            lse_base2=base2,
            prefix=(pre_out.bfloat16(), pre_lse * scale),
        )
        torch.testing.assert_close(fused.float(), ref, atol=2e-2, rtol=2e-2)
        full_lse = (
            torch.logsumexp(torch.cat([s, st], -1), -1)
            .permute(0, 2, 1)
            .reshape(bs * n, hq)
        )
        torch.testing.assert_close(fused_lse, full_lse * scale, atol=1e-3, rtol=1e-3)


def _reference_verify(cands, parents, target, max_depth):
    bs, n = cands.shape
    predicts = torch.zeros(bs * n, dtype=torch.int32)
    lengths, paths = [], torch.full((bs, n), -1, dtype=torch.int32)
    for b in range(bs):
        cur, path = 0, [0]
        for _ in range(max_depth):
            pick = int(target[b * n + cur])
            kids = [
                j
                for j in range(n)
                if int(parents[b, j]) == cur and int(cands[b, j]) == pick
            ]
            if not kids:
                break
            predicts[b * n + len(path) - 1] = pick
            cur = kids[0]
            path.append(cur)
        predicts[b * n + len(path) - 1] = int(target[b * n + cur])
        lengths.append(len(path))
        paths[b, : len(path)] = torch.tensor(path, dtype=torch.int32)
    return predicts, torch.tensor(lengths, dtype=torch.int32), paths


def _random_tree(n, max_depth, rng):
    parents, depth = [-1], [0]
    for j in range(1, n):
        choices = [i for i in range(j) if depth[i] < max_depth]
        p = rng.choice(choices)
        parents.append(p)
        depth.append(depth[p] + 1)
    return parents


@pytest.mark.parametrize(
    "n,max_depth,vocab", [(8, 3, 3), (32, 6, 4), (64, 10, 5), (5, 4, 2)]
)
def test_verify_tree_greedy_matches_reference(n, max_depth, vocab):
    rng = random.Random(n)
    bs = 16
    parents = torch.tensor(
        [_random_tree(n, max_depth, rng) for _ in range(bs)], dtype=torch.int32
    )
    cands = torch.randint(0, vocab, (bs, n), dtype=torch.int32)
    target = torch.randint(0, vocab, (bs * n,), dtype=torch.int32)
    # A chain row and an all-accept row.
    parents[0] = torch.arange(-1, n - 1)
    cands[1, 1:] = 0
    target[n : 2 * n] = 0
    ref = _reference_verify(cands, parents, target, max_depth)
    out = [
        torch.zeros(bs * n, dtype=torch.int32),
        torch.zeros(bs, dtype=torch.int32),
        torch.zeros(bs, n, dtype=torch.int32),
    ]
    out = [t.cuda() for t in out]
    verify_tree_greedy(
        *out, cands.cuda(), parents.cuda(), target.cuda(), max_depth=max_depth
    )
    predicts, lengths, paths = [t.cpu() for t in out]
    assert torch.equal(lengths, ref[1])
    assert torch.equal(paths, ref[2])
    for b in range(bs):
        assert torch.equal(
            predicts[b * n : b * n + int(lengths[b])],
            ref[0][b * n : b * n + int(lengths[b])],
        )


def _paged_tree_problem(bs, n, prefix_lens, hq, hkv, d, page, gen):
    """A paged HND cache holding each request's prefix + tree window, plus the reference."""
    scale = d**-0.5
    pages_per_req = max((p + n + page - 1) // page for p in prefix_lens)
    k_cache = torch.zeros(bs * pages_per_req + 1, hkv, page, d, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    tables = torch.zeros(bs, pages_per_req, dtype=torch.int32)
    q = torch.randn(bs * n, hq, d, generator=gen).bfloat16()
    kt = torch.randn(bs * n, hkv, d, generator=gen).bfloat16()
    vt = torch.randn(bs * n, hkv, d, generator=gen).bfloat16()
    mask = torch.zeros(bs * n, dtype=torch.int64)
    refs = []
    for b, plen in enumerate(prefix_lens):
        tables[b] = torch.arange(pages_per_req) + 1 + b * pages_per_req
        kp = torch.randn(plen, hkv, d, generator=gen).bfloat16()
        vp = torch.randn(plen, hkv, d, generator=gen).bfloat16()
        keys = torch.cat([kp, kt[b * n : (b + 1) * n]])
        vals = torch.cat([vp, vt[b * n : (b + 1) * n]])
        for pos in range(plen + n):
            pg, off = tables[b, pos // page], pos % page
            k_cache[pg, :, off] = keys[pos]
            v_cache[pg, :, off] = vals[pos]
        parent = [-1] + [
            int(torch.randint(0, j, (1,), generator=gen)) for j in range(1, n)
        ]
        for i in range(n):
            bits, cur = 0, i
            while cur >= 0:
                bits |= 1 << cur
                cur = parent[cur]
            mask[b * n + i] = (
                bits - (1 << 64) if bits >> 63 else bits
            )  # bit 63 is the sign
        group = hq // hkv
        kk = keys.float().repeat_interleave(group, 1)
        vv = vals.float().repeat_interleave(group, 1)
        qq = q[b * n : (b + 1) * n].float()
        s = torch.einsum("nhd,xhd->hnx", qq, kk) * scale
        vis = torch.ones(n, plen + n, dtype=torch.bool)
        for i in range(n):
            for j in range(n):
                vis[i, plen + j] = bool((int(mask[b * n + i]) >> j) & 1)
        s = s.masked_fill(~vis[None], float("-inf"))
        refs.append(torch.einsum("hnx,xhd->nhd", torch.softmax(s, -1), vv))
    return q, kt, vt, mask, k_cache, v_cache, tables, torch.cat(refs), scale


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="trtllm-gen needs sm100",
)
def test_trtllm_prefix_cascade_matches_reference():
    from tokenspeed_kernel.ops.attention.mha.flashinfer import (
        trtllm_batch_context_with_kv_cache,
    )

    gen = torch.Generator().manual_seed(3)
    bs, n, hq, hkv, d, page = 3, 16, 32, 8, 128, 32
    prefix_lens = [5, 70, 131]
    q, kt, vt, mask, k_cache, v_cache, tables, ref, scale = _paged_tree_problem(
        bs, n, prefix_lens, hq, hkv, d, page, gen
    )
    q, kt, vt, mask = q.cuda(), kt.cuda(), vt.cuda(), mask.cuda()
    prefix = torch.tensor(prefix_lens, dtype=torch.int32).cuda()
    cu_q = torch.arange(0, bs * n + 1, n, dtype=torch.int32).cuda()
    cu_kv = torch.nn.functional.pad(torch.cumsum(prefix, 0, dtype=torch.int32), (1, 0))
    workspace = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda")
    pre_out, pre_lse = trtllm_batch_context_with_kv_cache(
        query=q,
        kv_cache=(k_cache.cuda(), v_cache.cuda()),
        workspace_buffer=workspace,
        block_tables=tables.cuda(),
        seq_lens=prefix,
        max_q_len=n,
        max_kv_len=4096,
        bmm1_scale=scale,
        bmm2_scale=1.0,
        batch_size=bs,
        cum_seq_lens_q=cu_q,
        cum_seq_lens_kv=cu_kv,
        out_dtype=torch.bfloat16,
        causal=False,
        return_lse=True,
    )
    # trtllm-gen returns its LSE in base 2.
    tree_out, tree_lse = tree_attention(
        q, kt, vt, mask, rows_per_req=n, slots_per_req=n, sm_scale=scale, lse_base2=True
    )
    out, _ = attn_merge_state(
        pre_out.contiguous(),
        pre_lse.float().contiguous(),
        tree_out,
        tree_lse,
        lse_scale_log2=1.0,
    )
    torch.testing.assert_close(out.float().cpu(), ref, atol=2e-2, rtol=2e-2)
    # The backend's fused form: the merge runs inside tree_attention, in place.
    fused, _ = tree_attention(
        q,
        kt,
        vt,
        mask,
        rows_per_req=n,
        slots_per_req=n,
        sm_scale=scale,
        lse_base2=True,
        prefix=(pre_out, pre_lse),
        out=pre_out,
    )
    torch.testing.assert_close(fused.float().cpu(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "n,splits", [(8, 1), (8, 5), (16, 3), (4, 8), (32, 4), (64, 2)]
)
@pytest.mark.parametrize("prefix_lens", [[5, 70, 131], [1, 2, 64]])
def test_tree_decode_attention_matches_reference(n, splits, prefix_lens):
    gen = torch.Generator().manual_seed(n * 10 + splits)
    bs, hq, hkv, d, page = 3, 32, 8, 128, 32
    q, _, _, mask, k_cache, v_cache, tables, ref, scale = _paged_tree_problem(
        bs, n, prefix_lens, hq, hkv, d, page, gen
    )
    rows = lambda cache: cache.permute(0, 2, 1, 3).reshape(-1, hkv, d).cuda()
    out = tree_decode_attention(
        q.cuda(),
        rows(k_cache),
        rows(v_cache),
        tables.cuda(),
        torch.tensor(prefix_lens, dtype=torch.int32).cuda() + n,
        mask.cuda(),
        num_nodes=n,
        page_size=page,
        sm_scale=scale,
        num_splits=splits,
    )
    torch.testing.assert_close(out.float().cpu(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "rows,vocab,k", [(1, 32000, 1), (32, 32000, 4), (64, 128256, 8), (5, 10, 8)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_logprob_topk_matches_torch(rows, vocab, k, dtype):
    gen = torch.Generator().manual_seed(rows + vocab + k)
    logits = (torch.randn(rows, vocab, generator=gen) * 4).to(dtype).cuda()
    scores, ids = logprob_topk(logits, k)
    ref_scores, _ = torch.topk(torch.log_softmax(logits.float(), -1), k)
    torch.testing.assert_close(scores, ref_scores, atol=1e-4, rtol=1e-5)
    picked = torch.log_softmax(logits.float(), -1).gather(1, ids)
    torch.testing.assert_close(picked, scores, atol=1e-4, rtol=1e-5)
    assert all(len(set(r)) == k for r in ids.tolist())
