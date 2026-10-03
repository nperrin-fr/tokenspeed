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

"""Tests for the MNNVL capability gate on the trtllm one-shot all-reduce path."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform().is_nvidia and torch.cuda.is_available()),
    reason="trtllm MNNVL gate is NVIDIA/CUDA only",
)


def _probe():
    import tokenspeed_kernel.ops.communication.trtllm as trtllm_mod

    return trtllm_mod, trtllm_mod._mnnvl_locally_available


def _two_hosts(monkeypatch, has_fabric):
    import tokenspeed_kernel.ops.communication.fabric as fabric

    monkeypatch.setattr(fabric, "_host_map", [rank // 8 for rank in range(16)])
    monkeypatch.setattr(fabric, "group_has_fabric", has_fabric)


def test_cross_host_group_requires_fabric(monkeypatch):
    """A group spanning hosts needs working fabric memory on every rank.

    Without it, symm_mem.rendezvous() hangs instead of failing, so the gate
    must reject the workspace up front.
    """
    _, probe = _probe()
    _two_hosts(monkeypatch, lambda ranks: False)

    assert probe(list(range(16))) is False


def test_cross_host_group_allowed_with_fabric(monkeypatch):
    _, probe = _probe()
    _two_hosts(monkeypatch, lambda ranks: True)

    # Still subject to the other capability checks, so only assert that the
    # cross-host rule alone no longer vetoes the group.
    assert probe(list(range(16))) == probe(list(range(8)))


def test_a_strided_pair_across_hosts_is_still_probed(monkeypatch):
    """Two ranks on two hosts is fewer ranks than one host holds."""
    from torch._C._distributed_c10d import _SymmetricMemory

    _, probe = _probe()
    probed = []
    _two_hosts(monkeypatch, lambda ranks: probed.append(list(ranks)) or False)
    monkeypatch.setattr(_SymmetricMemory, "has_multicast_support", lambda *a: True)

    assert probe([0, 8]) is False
    assert probed == [[0, 8]]


def test_intra_host_group_ignores_fabric(monkeypatch):
    """Groups inside one host ride NVLS multicast, so fabric must not gate them."""
    _, probe = _probe()
    _two_hosts(
        monkeypatch,
        lambda ranks: pytest.fail("fabric probe must not run for intra-host groups"),
    )

    probe(list(range(8)))


def test_unsupported_world_size_rejected():
    _, probe = _probe()

    assert probe([0, 1, 2]) is False


def test_the_oneshot_cap_follows_the_call_width_not_the_armed_lane():
    """Arming is grow-only, so a retired wide lane must not pin narrow calls."""
    from tokenspeed_kernel.thirdparty.cuda.trtllm import MnnvlAllReduceFusionWorkspace

    def ws(armed, cap, max_token_num=2048):
        return MnnvlAllReduceFusionWorkspace(
            tp_rank=0,
            tp_size=8,
            max_token_num=max_token_num,
            hidden_dim=armed,
            buffer_size_bytes=0,
            multicast_ptr=1,
            peer_ptrs=None,
            local_ptr=1,
            buffer_flags=None,
            oneshot_token_cap=cap,
            refs=(),
        )

    # K3 arms 3584 + 7168 for a lane-norm path that is retired, and every live
    # all-reduce is 7168 wide. At the armed width the cap is 6; at the width in
    # hand it is 9, which is what an eight-token spec-decode step needs.
    k3 = ws(10752, 6)
    assert k3.resolve_use_oneshot(8, None, 10752) is False
    assert k3.resolve_use_oneshot(8, None, 7168) is True
    assert k3.resolve_use_oneshot(10, None, 7168) is False
    assert k3.resolve_use_oneshot(8, False, 7168) is False

    # Scaling never promises more rows than the buffer was armed for.
    assert ws(8192, 4096, max_token_num=64).resolve_use_oneshot(65, None, 4096) is False


def test_every_resolution_passes_the_call_width():
    """Resolution is authoritative wherever it runs, so none may omit the width.

    Upstream wrappers resolve, then the launcher resolves again; a site that
    left the width out would recompute the armed-width answer and undo the
    decision, which is invisible to a unit test calling the method directly.
    """
    import ast
    import pathlib

    import tokenspeed_kernel

    # Walk the package *and* the tests beside it: a stale two-argument call in a
    # distributed test is a TypeError that only a multi-rank run would reach.
    root = pathlib.Path(tokenspeed_kernel.__file__).parents[2]
    sites = 0
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "resolve_use_oneshot"
            ):
                sites += 1
                assert len(node.args) == 3, f"{path}:{node.lineno} omits the width"
    assert sites >= 5, f"expected every resolution site to be checked, saw {sites}"


@pytest.mark.parametrize("on_mnnvl", [True, False])
def test_each_workspace_runs_its_own_oneshot_rule(monkeypatch, on_mnnvl):
    """mnnvl resolves against its armed lane; IPC keeps the traffic heuristic."""
    from types import SimpleNamespace

    trtllm_mod, _ = _probe()
    mnnvl = SimpleNamespace(resolve_use_oneshot=lambda tokens, requested, width: False)
    manager = SimpleNamespace(world_size=8, mnnvl_workspace=mnnvl)
    ipc = object()
    picked = mnnvl if on_mnnvl else ipc
    monkeypatch.setattr(trtllm_mod, "_ar_fusion_workspace", lambda *args: picked)
    monkeypatch.setattr(trtllm_mod, "_ar_should_use_oneshot", lambda *args: True)

    workspace, use_oneshot = trtllm_mod._ar_workspace_and_oneshot(
        manager, 4, 7168, torch.bfloat16, 0, None, False
    )

    assert workspace is picked
    assert use_oneshot is (not on_mnnvl)


def test_ipc_only_collectives_refuse_a_missing_or_mismatched_workspace(monkeypatch):
    from types import SimpleNamespace

    trtllm_mod, _ = _probe()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    ipc = torch.empty(1)
    armed = SimpleNamespace(workspace_tensor=ipc, use_fp32_lamport=False)

    assert trtllm_mod._ipc_workspace(armed, torch.bfloat16, "allgather") is ipc
    with pytest.raises(RuntimeError, match="payload width"):
        trtllm_mod._ipc_workspace(armed, torch.float32, "allgather")
    missing = SimpleNamespace(workspace_tensor=None, use_fp32_lamport=False)
    with pytest.raises(RuntimeError, match="reducescatter fusion requires the IPC"):
        trtllm_mod._ipc_workspace(missing, torch.bfloat16, "reducescatter")
