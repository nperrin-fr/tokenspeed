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
from tokenspeed_kernel.ops.kvcache.triton import compact_window_rows
from tokenspeed_kernel.ops.sampling.triton.logprob_topk import logprob_topk
from tokenspeed_kernel.ops.sampling.triton.tree_verify import verify_tree

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
        q,
        k,
        v,
        mask,
        rows_per_req=rows,
        slots_per_req=slots,
        sm_scale=scale,
        lse_base2=False,
    )
    ref_out, ref_lse = _reference_tree_attention(q, k, v, mask, rows, slots, scale)
    torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=2e-2)
    seen = torch.isfinite(ref_lse)
    assert torch.equal(seen, torch.isfinite(lse))
    torch.testing.assert_close(lse[seen], ref_lse[seen], atol=1e-3, rtol=1e-3)
    assert torch.all(out[0] == 0)


def _last_dim_strided(t):
    """The same values viewed with stride 2 along the last dim."""
    backing = torch.zeros(
        *t.shape[:-1], 2 * t.shape[-1], dtype=t.dtype, device=t.device
    )
    backing[..., ::2] = t
    return backing[..., ::2]


def test_tree_attention_last_dim_strided():
    gen = torch.Generator().manual_seed(20260930)
    bs, rows, slots, hq, hkv, d = 2, 17, 17, 8, 2, 128
    q = torch.randn(bs * rows, hq, d, generator=gen).bfloat16().cuda()
    k = torch.randn(bs * slots, hkv, d, generator=gen).bfloat16().cuda()
    v = torch.randn(bs * slots, hkv, d, generator=gen).bfloat16().cuda()
    mask = _random_masks(bs, rows, slots, gen)
    scale = d**-0.5
    out = _last_dim_strided(torch.empty_like(q))
    tree_attention(
        _last_dim_strided(q),
        _last_dim_strided(k),
        _last_dim_strided(v),
        mask,
        rows_per_req=rows,
        slots_per_req=slots,
        sm_scale=scale,
        lse_base2=False,
        out=out,
    )
    ref_out, _ = _reference_tree_attention(q, k, v, mask, rows, slots, scale)
    torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=2e-2)


def test_tree_attention_ignores_nan_in_unseen_slots():
    gen = torch.Generator().manual_seed(5)
    bs, rows, slots, hq, hkv, d = 2, 4, 12, 8, 2, 128
    q = torch.randn(bs * rows, hq, d, generator=gen).bfloat16().cuda()
    k = torch.randn(bs * slots, hkv, d, generator=gen).bfloat16().cuda()
    v = torch.randn(bs * slots, hkv, d, generator=gen).bfloat16().cuda()
    mask = _random_masks(bs, rows, 4, gen) | 1  # rows see slots 0..3 only
    scale = d**-0.5
    ref_out, _ = _reference_tree_attention(q, k, v, mask, rows, slots, scale)
    stale = torch.arange(bs * slots, device="cuda") % slots >= 8
    k[stale] = float("nan")
    v[stale] = float("nan")
    out, _ = tree_attention(
        q,
        k,
        v,
        mask,
        rows_per_req=rows,
        slots_per_req=slots,
        sm_scale=scale,
        lse_base2=False,
    )
    torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=2e-2)


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
        q,
        kt,
        vt,
        mask,
        rows_per_req=n,
        slots_per_req=n,
        sm_scale=scale,
        lse_base2=False,
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
    for base2, strided_lse in ((False, False), (True, False), (True, True)):
        scale = 1.4426950408889634 if base2 else 1.0
        prefix_lse = pre_lse * scale
        if strided_lse:
            prefix_lse = prefix_lse.t().contiguous().t()
        fused, fused_lse = tree_attention(
            q,
            kt,
            vt,
            mask,
            rows_per_req=n,
            slots_per_req=n,
            sm_scale=d**-0.5,
            lse_base2=base2,
            prefix=(pre_out.bfloat16(), prefix_lse),
        )
        torch.testing.assert_close(fused.float(), ref, atol=2e-2, rtol=2e-2)
        full_lse = (
            torch.logsumexp(torch.cat([s, st], -1), -1)
            .permute(0, 2, 1)
            .reshape(bs * n, hq)
        )
        torch.testing.assert_close(fused_lse, full_lse * scale, atol=1e-3, rtol=1e-3)


def _reference_verify(cands, parents, target):
    bs, n = cands.shape
    predicts = torch.zeros(bs * n, dtype=torch.int32)
    lengths, paths = [], torch.full((bs, n), -1, dtype=torch.int32)
    for b in range(bs):
        cur, path = 0, [0]
        while True:
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
def test_verify_tree_matches_reference(n, max_depth, vocab):
    rng = random.Random(n)
    bs = 16
    parents = torch.tensor(
        [_random_tree(n, max_depth, rng) for _ in range(bs)], dtype=torch.int32
    )
    cands = torch.randint(0, vocab, (bs, n), dtype=torch.int32)
    target = torch.randint(0, vocab, (bs * n,), dtype=torch.int32)
    # A chain row deeper than max_depth and an all-accept row.
    parents[0] = torch.arange(-1, n - 1)
    cands[1, 1:] = 0
    target[n : 2 * n] = 0
    ref = _reference_verify(cands, parents, target)
    out = [
        torch.zeros(bs * n, dtype=torch.int32),
        torch.zeros(bs, dtype=torch.int32),
        torch.zeros(bs, n, dtype=torch.int32),
    ]
    out = [t.cuda() for t in out]
    verify_tree(*out, cands.cuda(), parents.cuda(), target.cuda())
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
    _check_tree_decode(bs, n, splits, prefix_lens, hq, hkv, d, page, gen)


@pytest.mark.parametrize(
    "hq,hkv,d,n",
    [
        (16, 2, 256, 64),
        (16, 2, 256, 17),
        (24, 4, 256, 64),
        (32, 4, 128, 64),
        (32, 2, 128, 64),
    ],
)
def test_tree_decode_attention_large_query_tiles(hq, hkv, d, n):
    """N x GQA group beyond one on-chip tile (e.g. GQA 8 at head_dim 256) still launches and matches."""
    gen = torch.Generator().manual_seed(hq * n + d)
    _check_tree_decode(2, n, 3, [40, 97], hq, hkv, d, 64, gen)


def test_tree_decode_attention_last_dim_strided():
    gen = torch.Generator().manual_seed(20260930)
    _check_tree_decode(2, 16, 3, [40, 97], 32, 8, 128, 32, gen, strided=True)


def _check_tree_decode(
    bs, n, splits, prefix_lens, hq, hkv, d, page, gen, strided=False
):
    q, _, _, mask, k_cache, v_cache, tables, ref, scale = _paged_tree_problem(
        bs, n, prefix_lens, hq, hkv, d, page, gen
    )
    layout = _last_dim_strided if strided else (lambda t: t)
    rows = lambda cache: layout(cache.permute(0, 2, 1, 3).reshape(-1, hkv, d).cuda())
    out = tree_decode_attention(
        layout(q.cuda()),
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


def test_logprob_topk_row_offset_beyond_int32():
    """rows x vocab past 2**31 elements must address rows in 64 bits."""
    vocab, rows = 248320, 8700
    logits = torch.zeros(rows, vocab, device="cuda", dtype=torch.bfloat16)
    logits[-1, 12345] = 30.0
    scores, ids = logprob_topk(logits, 2)
    assert int(ids[-1, 0]) == 12345
    assert torch.isfinite(scores).all()


@pytest.mark.parametrize(
    "rows,vocab,k",
    [(1, 32000, 1), (32, 32000, 4), (64, 128256, 8), (5, 10, 8), (4, 248320, 4)],
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


@pytest.mark.parametrize("nodes", [8, 64])
def test_compact_window_rows_moves_every_buffer(nodes):
    """Every buffer (bf16 and fp8 planes alike) packs each accepted path to the window front."""
    gen = torch.Generator().manual_seed(nodes)
    bs, slots, heads, dim = 3, 4096, 2, 128
    buffers = [
        torch.randn(slots, heads, dim, generator=gen).bfloat16().cuda()
        for _ in range(3)
    ] + [
        torch.randn(slots, heads, 2 * dim, generator=gen).to(torch.float8_e4m3fn).cuda()
    ]  # same bytes per token row as the bf16 planes
    locs = torch.randperm(slots, generator=gen)[: bs * nodes].int().cuda()
    paths = [[0, 2, 5], [0, 1, 2, 3], [0]]  # jump, identity, root only
    path = torch.full((bs, nodes), -1, dtype=torch.int32)
    for b, p in enumerate(paths):
        path[b, : len(p)] = torch.tensor(p, dtype=torch.int32)
    expected = [buf.view(torch.uint8).clone() for buf in buffers]
    for want in expected:
        for b, p in enumerate(paths):
            window = locs[b * nodes : (b + 1) * nodes].long()
            want[window[: len(p)]] = want[window[torch.tensor(p)]]

    addresses = torch.tensor(
        [buf.data_ptr() for buf in buffers], dtype=torch.int64, device="cuda"
    )
    compact_window_rows(addresses, locs, path.cuda(), row_bytes=heads * dim * 2)

    for got, want in zip(buffers, expected):
        assert torch.equal(got.view(torch.uint8), want)
