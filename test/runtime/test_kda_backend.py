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

from __future__ import annotations

from types import SimpleNamespace

from tokenspeed_kernel.registry import KernelSpec

from tokenspeed.runtime.layers.attention import kda_backend


def _spec(mode: str, layout: str, priority: int, **traits) -> KernelSpec:
    return KernelSpec(
        name=f"{mode}_{layout}_{priority}",
        family="attention",
        mode=mode,
        priority=priority,
        traits={
            **{name: frozenset({value}) for name, value in traits.items()},
            "recurrent_layout": frozenset({layout}),
        },
    )


def _family(layout: str, anchor: str, priority: int = 10) -> list[KernelSpec]:
    return [
        _spec(
            "kda_fused_paged_decode",
            layout,
            priority if anchor == "decode" else 10,
            paged_state=True,
            fused_output_norm=True,
        ),
        _spec(
            "kda_paged_decode",
            layout,
            10,
            indexed_state=True,
            single_token=True,
        ),
        _spec(
            "kda_fused_paged_verify",
            layout,
            priority if anchor == "verify" else 10,
            paged_state=True,
            store_states=False,
            split_producers=True,
            replay_ring=False,
        ),
        _spec(
            "kda_replay_commit",
            layout,
            10,
            flat_state=True,
            replay_ring=False,
        ),
    ]


def _patch_registry(monkeypatch, specs):
    by_mode = {}
    for spec in specs:
        by_mode.setdefault(spec.mode, []).append(spec)
    for values in by_mode.values():
        values.sort(key=lambda spec: spec.priority, reverse=True)
    registry = SimpleNamespace(
        get_for_operator=lambda _family, mode, **_kwargs: by_mode.get(mode, [])
    )
    monkeypatch.setattr(kda_backend.KernelRegistry, "get", lambda: registry)
    return SimpleNamespace(arch="fake")


def test_layout_follows_registry_trait_for_decode_backend(monkeypatch) -> None:
    platform = _patch_registry(monkeypatch, _family("v_major", "decode"))
    assert (
        kda_backend.resolve_kda_state_layout("triton", platform=platform)
        == "v_major"
    )


def test_sgl_backend_follows_replay_ring_layout_trait(monkeypatch) -> None:
    v_major = [
        spec
        for spec in _family("v_major", "decode", 14)
        if spec.mode != "kda_replay_commit"
    ]
    v_major.append(
        _spec(
            "kda_replay_commit",
            "v_major",
            10,
            flat_state=True,
            replay_ring=True,
        )
    )
    platform = _patch_registry(
        monkeypatch, _family("k_major", "decode", 15) + v_major
    )
    assert (
        kda_backend.resolve_kda_state_layout("sgl_mtp", platform=platform)
        == "v_major"
    )


def test_speculation_switches_anchor_to_verify(monkeypatch) -> None:
    specs = _family("k_major", "decode", 15) + _family("v_major", "verify", 16)
    platform = _patch_registry(monkeypatch, specs)
    assert (
        kda_backend.resolve_kda_state_layout("triton", 1, platform=platform)
        == "k_major"
    )
    assert (
        kda_backend.resolve_kda_state_layout("triton", 2, platform=platform)
        == "v_major"
    )


def test_incomplete_consumer_family_falls_back(monkeypatch) -> None:
    preferred = _family("v_major", "decode", 16)
    preferred = [spec for spec in preferred if spec.mode != "kda_paged_decode"]
    specs = preferred + _family("k_major", "decode", 15)
    platform = _patch_registry(monkeypatch, specs)
    assert (
        kda_backend.resolve_kda_state_layout("triton", platform=platform)
        == "k_major"
    )
