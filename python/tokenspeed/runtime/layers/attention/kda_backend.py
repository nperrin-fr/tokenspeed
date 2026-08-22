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

"""Shared KDA decode-backend selection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec
from tokenspeed_kernel.selection import spec_matches_traits

if TYPE_CHECKING:
    from tokenspeed_kernel.platform import PlatformInfo

logger = logging.getLogger(__name__)

_FALLBACK_LOGGED: set[tuple[str, str]] = set()


def default_kda_decode_backend() -> str:
    """Select the fastest available KDA decode backend for this platform."""
    if not current_platform().is_hopper_plus:
        return "triton"

    # Delay CuTe dependency probes until an NVIDIA SM90+ process needs them.
    from tokenspeed_kernel.thirdparty.cute_dsl.kda_mtp import is_available

    return "sgl_mtp" if is_available() else "triton"


def _matching_specs(
    registry: KernelRegistry,
    mode: str,
    platform: PlatformInfo,
    traits: dict[str, object],
) -> list[KernelSpec]:
    return [
        spec
        for spec in registry.get_for_operator("attention", mode, platform=platform)
        if spec_matches_traits(spec, traits) and "recurrent_layout" in spec.traits
    ]


def _layout_is_covered(
    registry: KernelRegistry,
    layout: str,
    platform: PlatformInfo,
    consumers: tuple[tuple[str, dict[str, object]], ...],
) -> bool:
    return all(
        _matching_specs(
            registry,
            mode,
            platform,
            {**traits, "recurrent_layout": layout},
        )
        for mode, traits in consumers
    )


def resolve_kda_state_layout(
    kda_decode_backend: str,
    speculative_num_draft_tokens: int = 1,
    *,
    platform: PlatformInfo | None = None,
) -> str:
    """Resolve one registry-declared recurrent layout for the whole KDA pool."""
    platform = platform or current_platform()
    registry = KernelRegistry.get()
    speculative = speculative_num_draft_tokens > 1
    replay_ring = kda_decode_backend == "sgl_mtp"
    if speculative:
        anchor_mode = "kda_fused_paged_verify"
        anchor_traits = {
            "paged_state": True,
            "store_states": False,
            "split_producers": not replay_ring,
            "replay_ring": replay_ring,
        }
    else:
        anchor_mode = "kda_fused_paged_decode"
        anchor_traits = {"paged_state": True, "fused_output_norm": True}

    # The anchor is whichever kernel dominates step time.
    anchors = _matching_specs(registry, anchor_mode, platform, anchor_traits)
    if not anchors:
        raise RuntimeError(
            f"no KDA layout anchor registered for attention.{anchor_mode} "
            f"on {platform.arch}"
        )

    consumers: list[tuple[str, dict[str, object]]] = [
        (
            "kda_fused_paged_decode",
            {"paged_state": True, "fused_output_norm": True},
        ),
        ("kda_paged_decode", {"indexed_state": True, "single_token": True}),
    ]
    if speculative:
        consumers.extend((("kda_fused_paged_verify", anchor_traits),))
    if speculative or replay_ring:
        consumers.append(
            (
                "kda_replay_commit",
                {"flat_state": True, "replay_ring": replay_ring},
            )
        )
    consumer_family = tuple(consumers)

    anchor_layouts: list[str] = []
    for spec in anchors:
        layouts = spec.traits["recurrent_layout"]
        if len(layouts) != 1:
            raise RuntimeError(
                f"KDA layout anchor {spec.name!r} must declare one " "recurrent_layout"
            )
        layout = next(iter(layouts))
        if layout not in anchor_layouts:
            anchor_layouts.append(layout)

    preferred = anchor_layouts[0]
    for layout in anchor_layouts:
        if _layout_is_covered(registry, layout, platform, consumer_family):
            if layout != preferred:
                reason = f"preferred {preferred} lacks a complete KDA consumer family"
                log_key = (preferred, layout)
                if log_key not in _FALLBACK_LOGGED:
                    logger.info("KDA state layout fallback to %s: %s", layout, reason)
                    _FALLBACK_LOGGED.add(log_key)
            return layout

    raise RuntimeError(
        f"no recurrent layout from attention.{anchor_mode} covers the KDA "
        f"consumer family on {platform.arch}"
    )


__all__ = ["default_kda_decode_backend", "resolve_kda_state_layout"]
