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

"""Startup-compiled joint BF16 GEMM: a padded serving call equals the exact kernel."""

import pytest
import torch
from tokenspeed_kernel.ops.gemm import flashinfer as fi

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not fi.has_flashinfer_cute_dsl_bf16(),
    reason="the joint BF16 GEMM needs FlashInfer's cute-dsl backend on SM100/SM103",
)

N, K = 1024, 4096


@pytest.mark.parametrize("m", [3, 19, 31])
def test_padded_rows_equal_the_exact_kernel(monkeypatch, m):
    weight = (torch.randn(N, K, device="cuda") * 0.05).to(torch.bfloat16)
    x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(fi, "_bf16_compiled", set())
    monkeypatch.setattr(fi, "_bf16_sealed", False)
    exact = fi.flashinfer_bf16_gemm(x, weight, None)
    bucket = 1 << (m - 1).bit_length()
    fi.flashinfer_bf16_gemm(x.new_zeros((bucket, K)), weight, None)

    monkeypatch.setattr(fi, "_bf16_compiled", {(x.device.index, bucket, N, K)})
    monkeypatch.setattr(fi, "_bf16_sealed", True)
    assert fi.flashinfer_bf16_gemm_ready(x, weight)
    out = torch.empty(m, N, device="cuda", dtype=torch.bfloat16)
    padded = fi.flashinfer_bf16_gemm(x, weight, out)
    assert padded is out
    torch.testing.assert_close(padded, exact, rtol=0, atol=0)
    assert fi._bf16_compiled == {(x.device.index, bucket, N, K)}


def test_precompile_then_seal_serves_every_row_count_without_new_kernels(monkeypatch):
    weight = (torch.randn(N, K, device="cuda") * 0.05).to(torch.bfloat16)
    monkeypatch.setattr(fi, "_bf16_compiled", set())
    monkeypatch.setattr(fi, "_bf16_sealed", False)
    monkeypatch.setattr(fi, "_bf16_projections", {})
    monkeypatch.setattr(fi, "is_autotuning", lambda: True)
    fi.autotune_bf16_gemm(
        torch.zeros(1, K, device="cuda", dtype=torch.bfloat16), weight
    )
    assert fi.precompile_bf16_gemms() == 5
    compiled = set(fi._bf16_compiled)
    monkeypatch.setattr(fi, "_bf16_sealed", True)
    for m in range(1, fi.BF16_GEMM_MAX_M + 1):
        x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
        assert fi.flashinfer_bf16_gemm_ready(x, weight)
        torch.testing.assert_close(
            fi.flashinfer_bf16_gemm(x, weight, None),
            (x.float() @ weight.float().T).to(torch.bfloat16),
            rtol=2e-2,
            atol=2e-2,
        )
    assert fi._bf16_compiled == compiled
    unseen = torch.randn(N + 16, K, device="cuda", dtype=torch.bfloat16)
    assert not fi.flashinfer_bf16_gemm_ready(x, unseen)
