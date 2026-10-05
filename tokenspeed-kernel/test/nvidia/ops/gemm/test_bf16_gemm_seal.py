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

"""Sealed joint BF16 GEMM on FI's warp split-K kernel, which compiles per exact M."""

from dataclasses import astuple

import pytest
import torch
from tokenspeed_kernel.ops.gemm import flashinfer as fi

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not fi.has_flashinfer_cute_dsl_bf16(),
    reason="the joint BF16 GEMM needs FlashInfer's cute-dsl backend on SM100/SM103",
)

N, K = 1024, 4096


def _compiles() -> int:
    from flashinfer.gemm.kernels import dense_bf16_gemm_warp_splitk

    return dense_bf16_gemm_warp_splitk._compile.cache_info().misses


@pytest.fixture
def warp_splitk(monkeypatch):
    """Pin FI's choice to warp split-K with one tactic for every M, as after tuning."""
    from flashinfer.autotuner import AutoTuner

    warp_splitk = pytest.importorskip(
        "flashinfer.gemm.kernels.dense_bf16_gemm_warp_splitk"
    )
    tactic = astuple(warp_splitk.autotune_tactics(fi.BF16_GEMM_MAX_M, N, K)[0])

    def choose_one(tuner, custom_op, runners, tuning_config, inputs, **kwargs):
        assert custom_op == "bf16_gemm"
        runner = next(
            (r for r in runners if type(r).__name__ == "CuteDSLWarpSplitKBf16Runner"),
            None,
        )
        if runner is None:
            pytest.skip("this FlashInfer build omits the warp split-K runner")
        return runner, tactic

    monkeypatch.setattr(AutoTuner, "choose_one", choose_one)
    monkeypatch.setattr(fi, "_bf16_compiled", set())
    monkeypatch.setattr(fi, "_bf16_sealed", False)


def test_sealed_gemm_reruns_startup_rows_and_refuses_new_ones(warp_splitk, monkeypatch):
    weight = (torch.randn(N, K, device="cuda") * 0.05).to(torch.bfloat16)
    x = torch.randn(19, K, device="cuda", dtype=torch.bfloat16)
    expected = fi.flashinfer_bf16_gemm(x, weight, None)
    fi.seal_bf16_gemms()

    compiles = _compiles()
    assert fi.flashinfer_joint_bf16_supported(x, weight, None)
    torch.testing.assert_close(
        fi.flashinfer_bf16_gemm(x, weight, None), expected, rtol=0, atol=0
    )
    assert _compiles() == compiles
    # A row count or projection startup never ran would compile, so it is refused.
    assert not fi.flashinfer_joint_bf16_supported(x[:18], weight, None)
    other = torch.randn(N + 16, K, device="cuda", dtype=torch.bfloat16)
    assert not fi.flashinfer_joint_bf16_supported(x, other, None)
    with pytest.raises(ValueError, match="Unsupported input"):
        fi.flashinfer_bf16_gemm(x[:18], weight, None)
    assert _compiles() == compiles
