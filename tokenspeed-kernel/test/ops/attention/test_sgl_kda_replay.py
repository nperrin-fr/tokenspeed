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

"""Parity for the vendored SGLang KDA verify and ReplaySSM fold kernels."""

import pytest
import torch
from tokenspeed_kernel.ops.attention.triton.kda_dispatch import (
    cutedsl_kda_mtp_verify,
    triton_sgl_replayssm_fold,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.cute_dsl.kda_mtp import is_available
from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
    fused_kda_verify_conv_update,
    fused_recurrent_kda_verify_megafuse,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if not (current_platform().is_nvidia and current_platform().is_hopper_plus):
    pytest.skip("SGLang KDA MTP requires NVIDIA sm90+", allow_module_level=True)
if not is_available():
    pytest.skip("CuTe DSL is not installed", allow_module_level=True)

H, D, WIDTH, T = 4, 128, 4, 4
P = H * D
LOWER_BOUND = -5.0


def _inputs(n: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(100 + n)
    rows = n * T

    def rand(*shape, dtype=torch.bfloat16, scale=1.0):
        return torch.randn(*shape, device="cuda", dtype=dtype) * scale

    data = {
        "qkv": rand(rows, 3 * P),
        "conv_w": rand(3 * P, WIDTH, scale=0.1),
        "conv_pool": rand(n, 3 * P, WIDTH - 1, scale=0.1),
        "f_a": rand(rows, D),
        "f_b": rand(P, D, scale=0.1),
        "beta": rand(rows, H),
        "A_log": rand(H, dtype=torch.float32, scale=0.2),
        "dt_bias": rand(P, dtype=torch.float32, scale=0.1),
        "state": rand(n, H, D, D, dtype=torch.float32, scale=0.01),
        "read": torch.arange(n, device="cuda", dtype=torch.int32),
        "write": torch.arange(rows, device="cuda", dtype=torch.int32),
        "cu": torch.arange(0, (n + 1) * T, T, device="cuda", dtype=torch.int32),
    }
    data["conv_w_fp32"] = data["conv_w"].float()
    data["conv_scratch"] = torch.empty(
        rows, 3 * P, WIDTH - 1, device="cuda", dtype=torch.bfloat16
    )
    data["state_scratch"] = torch.empty(
        rows, H, D, D, device="cuda", dtype=torch.float32
    )
    data["ring_rawv"] = torch.zeros(n, H, T + 1, D, device="cuda", dtype=torch.bfloat16)
    data["ring_rawk"] = torch.zeros_like(data["ring_rawv"])
    data["ring_g"] = torch.zeros(n, H, T + 1, D, device="cuda", dtype=torch.float32)
    data["ring_beta"] = torch.zeros(n, H, T + 1, device="cuda", dtype=torch.float32)
    data["tape_q"] = torch.zeros(
        n, T, P, WIDTH - 1, device="cuda", dtype=torch.bfloat16
    )
    data["tape_k"] = torch.zeros_like(data["tape_q"])
    data["tape_v"] = torch.zeros_like(data["tape_q"])
    return data


def _run_pair(n: int):
    data = _inputs(n)
    conv_qkv = fused_kda_verify_conv_update(
        data["qkv"],
        data["conv_w"],
        data["conv_pool"],
        data["read"],
        num_heads=H,
        head_dim=D,
        draft_token_num=T,
    )
    gate = torch.mm(data["f_a"], data["f_b"].t()).contiguous()
    baseline = fused_recurrent_kda_verify_megafuse(
        data["qkv"],
        data["conv_w"],
        data["conv_pool"],
        data["conv_scratch"],
        data["f_a"],
        data["f_b"],
        data["beta"],
        data["A_log"],
        data["dt_bias"],
        data["state"],
        data["state_scratch"],
        data["read"],
        data["write"],
        num_heads=H,
        head_dim=D,
        draft_token_num=T,
        lower_bound=LOWER_BOUND,
        store_states=True,
        recurrent_layout="k_major",
    )
    v_major = data["state"].transpose(-1, -2).contiguous()
    actual = cutedsl_kda_mtp_verify(
        data["qkv"],
        data["conv_w_fp32"],
        data["conv_pool"],
        data["conv_scratch"],
        data["f_a"],
        data["f_b"],
        data["beta"],
        data["A_log"],
        data["dt_bias"],
        state_pool=v_major,
        state_scratch=None,
        read_indices=data["read"],
        write_indices=data["write"],
        num_heads=H,
        head_dim=D,
        draft_token_num=T,
        lower_bound=LOWER_BOUND,
        cu_seqlens=data["cu"],
        replay_rawv=data["ring_rawv"],
        replay_rawk=data["ring_rawk"],
        replay_g=data["ring_g"],
        replay_beta=data["ring_beta"],
        replay_conv_q=data["tape_q"],
        replay_conv_k=data["tape_k"],
        replay_conv_v=data["tape_v"],
    )
    return data, v_major, actual, baseline


@pytest.mark.parametrize("n", [1, 4])
def test_mtp_verify_matches_kmajor_baseline_and_populates_rings(n: int) -> None:
    data, _, actual, baseline = _run_pair(n)
    torch.testing.assert_close(
        actual.float(), baseline.view_as(actual).float(), atol=2e-2, rtol=2e-2
    )
    for name in ("ring_rawv", "ring_rawk", "ring_g", "ring_beta"):
        assert torch.count_nonzero(data[name][:, :, :T]).item() > 0


@pytest.mark.parametrize("accepted", [T - 1, T])
def test_replayssm_fold_matches_committed_baseline_state(accepted: int) -> None:
    data, checkpoint, _, _ = _run_pair(4)
    accepted_length = torch.full((4,), accepted, device="cuda", dtype=torch.int32)
    triton_sgl_replayssm_fold(
        data["qkv"],
        data["conv_w"],
        data["conv_pool"],
        data["conv_scratch"],
        data["f_a"],
        data["f_b"],
        data["beta"],
        data["A_log"],
        data["dt_bias"],
        state_pool=checkpoint,
        state_out=checkpoint,
        read_indices=data["read"],
        write_indices=data["read"],
        accepted_length=accepted_length,
        num_heads=H,
        head_dim=D,
        draft_token_num=T,
        lower_bound=LOWER_BOUND,
        recurrent_layout="v_major",
        replay_rawv=data["ring_rawv"],
        replay_rawk=data["ring_rawk"],
        replay_g=data["ring_g"],
        replay_beta=data["ring_beta"],
    )
    rows = torch.arange(4, device="cuda") * T + accepted - 1
    expected = data["state_scratch"][rows].transpose(-1, -2)
    torch.testing.assert_close(checkpoint, expected, atol=2e-2, rtol=2e-2)
