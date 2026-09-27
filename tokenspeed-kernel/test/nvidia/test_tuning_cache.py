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

"""load_flashinfer_tuning_cache never fails startup.

Every failure mode -- missing file, corrupt JSON, a table swept on a different
GPU model -- must come back as ``False`` (with a warning) so the startup
autotune window tunes those shapes instead of engine startup aborting over a
stale table.
"""

from __future__ import annotations

import ast
import json
import re
from argparse import Namespace
from importlib.util import find_spec
from pathlib import Path

import pytest
from tokenspeed_kernel.ops.tuning import (
    flashinfer_tuning_cache_filename,
    load_flashinfer_tuning_cache,
)

requires_flashinfer = pytest.mark.skipif(
    find_spec("flashinfer") is None, reason="requires flashinfer"
)


def test_flashinfer_tuning_cache_filename_includes_cudnn() -> None:
    assert flashinfer_tuning_cache_filename(
        "kimi-k3",
        8,
        1,
        "NVIDIA B300 SXM6 AC",
        "0.6.16",
        92400,
    ) == (
        "kimi-k3,ep=8,tp=1,device_name=NVIDIA_B300_SXM6_AC,"
        "flashinfer=0.6.16,cudnn=92400.json"
    )


@requires_flashinfer
def test_missing_file_returns_false(tmp_path) -> None:
    assert load_flashinfer_tuning_cache(str(tmp_path / "absent.json")) is False


@requires_flashinfer
def test_corrupt_file_returns_false(tmp_path) -> None:
    path = tmp_path / "corrupt.json"
    path.write_text("{ not json")
    assert load_flashinfer_tuning_cache(str(path)) is False


@requires_flashinfer
def test_packaged_lookup_miss_returns_false() -> None:
    import torch
    from tokenspeed_kernel.ops.tuning import load_packaged_flashinfer_tuning_cache

    if not torch.cuda.is_available():
        pytest.skip("device-name lookup requires CUDA")
    # No table ships for this made-up model; the miss must be a quiet False
    # (INFO log), leaving the startup autotune window to tune these shapes.
    assert (
        load_packaged_flashinfer_tuning_cache("no-such-model-unit-test", 999, 1)
        is False
    )


@requires_flashinfer
def test_mismatched_gpu_metadata_returns_false(tmp_path) -> None:
    # A definite metadata conflict (wrong GPU model) must reject the whole
    # table -- this is the guard that keeps a B300-swept table off other SKUs.
    path = tmp_path / "wrong_gpu.json"
    path.write_text(
        json.dumps(
            {
                "_metadata": {
                    "flashinfer_version": "0.0.1",
                    "cuda_version": "0.0",
                    "cublas_version": "0",
                    "cudnn_version": "0",
                    "cudnn_frontend_version": "0",
                    "gpu": "NVIDIA UnitTest GPU That Does Not Exist",
                },
            }
        )
    )
    assert load_flashinfer_tuning_cache(str(path)) is False


def _packaged_tables() -> list[Path]:
    from tokenspeed_kernel.ops.moe import flashinfer as fi_pkg

    tactics = Path(fi_pkg.__file__).parent / "tactics"
    tables = sorted(tactics.glob("*.json"))
    assert tables, f"no packaged tuning tables under {tactics}"
    return tables


def test_every_packaged_table_is_named_for_its_own_metadata() -> None:
    """A shipped table's filename must restate the environment it was swept on."""
    for path in _packaged_tables():
        meta = json.loads(path.read_text())["_metadata"]
        model, ep, tp = re.match(r"([^,]+),ep=(\d+),tp=(\d+),", path.name).groups()
        assert path.name == flashinfer_tuning_cache_filename(
            model,
            int(ep),
            int(tp),
            meta["gpu"],
            meta["flashinfer_version"],
            meta["cudnn_version"],
        )


def test_kimi_k3_tables_are_keyed_on_their_layouts_rank_geometry() -> None:
    """Runner keys must carry the per-rank experts and intermediate of the named layout.

    FlashInfer keys MoE tactics on the runner's geometry as well as the input
    shapes, so a table swept on another layout's geometry loads cleanly and then
    misses every lookup. Keys without runner extras predate that key format.
    """
    num_experts, moe_intermediate = 896, 3072
    checked = 0
    for path in _packaged_tables():
        if not path.name.startswith("kimi-k3,"):
            continue
        ep, tp = map(int, re.match(r"[^,]+,ep=(\d+),tp=(\d+),", path.name).groups())
        for key in json.loads(path.read_text()):
            if key.startswith("_"):
                continue
            parts = ast.literal_eval(key)
            if len(parts) < 4 or not parts[3]:
                continue
            extras = parts[3]
            assert (extras[2], extras[10]) == (
                num_experts // ep,
                moe_intermediate // tp,
            ), f"{path.name}: key swept on {extras[2]} experts x {extras[10]} intermediate"
            checked += 1
    assert checked, "no packaged Kimi-K3 table carries runner extras"


def test_sweep_derives_rank_geometry_from_the_layout() -> None:
    from tokenspeed_kernel.ops.moe.flashinfer.moe_tactic_sweep import (
        _resolve_rank_geometry,
    )

    def resolve(ep_size: int, tp_size: int) -> tuple[int, int]:
        args = Namespace(
            num_experts=896,
            moe_intermediate_size=3072,
            ep_size=ep_size,
            tp_size=tp_size,
            local_experts=None,
            intermediate_size=None,
        )
        _resolve_rank_geometry(args)
        return args.local_experts, args.intermediate_size

    assert resolve(8, 1) == (112, 3072)
    assert resolve(1, 8) == (896, 384)
    with pytest.raises(ValueError):
        resolve(3, 1)
