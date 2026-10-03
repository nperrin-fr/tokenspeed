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

"""Persistent trtllm-gen counters: reuse, growth and the kernels' self-reset contract."""

from __future__ import annotations

import math

import pytest
import torch
from tokenspeed_kernel.ops.attention.mha.flashinfer import (
    trtllm_batch_decode_with_kv_cache,
    trtllm_gen_counter_buffer,
)
from tokenspeed_kernel.platform import current_platform

platform = current_platform()

pytestmark = pytest.mark.skipif(
    not platform.is_blackwell, reason="trtllm-gen attention requires Blackwell"
)

HEAD_DIM = 128
PAGE_SIZE = 64


def test_counters_are_reused_and_only_grow() -> None:
    small = trtllm_gen_counter_buffer("cuda:0", 1, 1)
    assert trtllm_gen_counter_buffer("cuda:0", 1, 1) is small
    large = trtllm_gen_counter_buffer("cuda:0", small.numel(), 1)
    assert large is not small and large.numel() >= 4 * small.numel()
    assert not large.any()
    assert trtllm_gen_counter_buffer("cuda:0", 1, 1) is large


def test_counters_refuse_to_grow_during_capture() -> None:
    current = trtllm_gen_counter_buffer("cuda:0", 1, 1)
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        with torch.cuda.graph(graph):
            trtllm_gen_counter_buffer("cuda:0", current.numel(), 1)
    assert trtllm_gen_counter_buffer("cuda:0", 1, 1) is current


def _long_context_decode(batch: int, kv_len: int, q_heads: int, kv_heads: int):
    pages = math.ceil(kv_len / PAGE_SIZE)
    shape = (batch * pages, kv_heads, PAGE_SIZE, HEAD_DIM)
    k_cache = torch.randn(shape, device="cuda").to(torch.float8_e4m3fn)
    v_cache = torch.randn(shape, device="cuda").to(torch.float8_e4m3fn)
    tables = torch.arange(batch * pages, device="cuda", dtype=torch.int32)
    query = torch.randn(batch, q_heads, HEAD_DIM, device="cuda")
    workspace = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda")

    def decode(counters: torch.Tensor | None) -> torch.Tensor:
        return trtllm_batch_decode_with_kv_cache(
            query=query.to(torch.float8_e4m3fn),
            kv_cache=(k_cache, v_cache),
            workspace_buffer=workspace,
            block_tables=tables.view(batch, pages),
            seq_lens=torch.full((batch,), kv_len, device="cuda", dtype=torch.int32),
            max_seq_len=kv_len,
            bmm1_scale=1.0 / math.sqrt(HEAD_DIM),
            bmm2_scale=1.0,
            out_dtype=torch.bfloat16,
            multi_ctas_kv_counter_buffer=counters,
        )

    return decode


def test_multi_cta_decode_resets_reused_counters() -> None:
    decode = _long_context_decode(batch=1, kv_len=16384, q_heads=32, kv_heads=2)
    reference = decode(None)
    counters = trtllm_gen_counter_buffer("cuda:0", 1, 32)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        eager = [decode(counters) for _ in range(3)]
        torch.cuda.synchronize()
    names = {event.name for event in prof.events()}
    assert any("MultiCtasKv" in name for name in names), sorted(names)
    assert not counters.any()
    for out in eager:
        torch.testing.assert_close(out, reference, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = decode(counters)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, reference, rtol=0, atol=0)
        assert not counters.any()
