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

"""sonic-sampler entry points for the runtime's ``sonic`` sampling backend.

sonic-sampler (optional dependency, v1.0.0) supplies everything;
``tokenspeed_kernel.thirdparty.sonic`` re-jits its fused kernels and wrappers
from their own source with two asserted patches (batch size as a runtime int,
in-kernel Philox noise) at import time. sonic binds ``triton`` at import time,
so it is imported under the ``tokenspeed_triton`` redirect: every kernel in
the chain compiles with the one vendored Triton.

``available`` is False when sonic-sampler is not installed or its source no
longer matches the patch anchors; every other name is then None. The cause is
logged (debug when the package is absent, warning when it is present but
incompatible).
"""

from __future__ import annotations

import logging

from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton

logger = logging.getLogger(__name__)

available = False

try:
    # sonic binds ``triton`` at import: route the whole chain to the vendored Triton.
    with redirect_triton_to_tokenspeed_triton():
        import sonic_sampler  # noqa: F401
        from sonic_sampler.base.sampler import Selection, Verification
        from sonic_sampler.core.buffer import SamplingBuffers
        from sonic_sampler.core.flags import Indicator, ScopedIndicators
        from sonic_sampler.interface.base import TopKStrategy, TwoStageTiling
        from sonic_sampler.interface.dispatch import (
            BatchBucket,
            PriorityBucket,
            RuntimeConfig,
            ThreeStageWarpConfig,
            TwoStageWarpConfig,
            VocabBucket,
        )
        from sonic_sampler.ops.base import MAX_K
        from tokenspeed_kernel.thirdparty.sonic import fused_multistep, fused_singular

    available = True
except ImportError as exc:
    # Optional dependency: callers gate on ``available``.
    level = logging.DEBUG if isinstance(exc, ModuleNotFoundError) else logging.WARNING
    logger.log(level, "sonic sampling backend unavailable: %s", exc)
    SamplingBuffers = ScopedIndicators = Indicator = None
    TopKStrategy = TwoStageTiling = None
    BatchBucket = PriorityBucket = RuntimeConfig = VocabBucket = None
    ThreeStageWarpConfig = TwoStageWarpConfig = None
    Selection = Verification = None
    fused_multistep = fused_singular = None
    MAX_K = None

# Measured (arch, packaged vocab bucket) -> batch buckets in sonic's TOML schema.
_MEASURED_DISPATCH: dict[tuple[int, int], list[tuple[int, int, str, int, int]]] = {
    # sm100 131k-262k vocab: 2 second-stage warps, 82us -> 24us per row (GB200).
    (100, 262144): [(1 << 16, 4096, "bitonic", 8, 2)],
}


def measured_dispatch(arch: int, bucket: VocabBucket) -> VocabBucket | None:
    """The measured replacement for a packaged vocab bucket on this arch, if any.

    Args:
        arch: compute capability as ``major * 10 + minor``.
        bucket: the packaged ``VocabBucket`` a ``TwoStageTiling`` resolved.

    Returns:
        A ``VocabBucket`` of the same size whose batch buckets carry the measured
        configs, or None when nothing was measured for ``(arch, bucket.size)``.

    Raises:
        ValueError: a measured ``block_n`` is below the packaged bucket's
            smallest, which sized the tiling's scratchpad.
    """
    entries = _MEASURED_DISPATCH.get((arch, bucket.size))
    if entries is None:
        return None
    for _, block_n, _, _, _ in entries:
        if block_n < bucket.block_n:
            raise ValueError(
                f"measured block_n {block_n} < packaged minimum {bucket.block_n} "
                f"for vocab bucket {bucket.size}: scratchpad would be undersized"
            )
    batch = [
        BatchBucket(
            size=size,
            config=[
                RuntimeConfig(
                    priority=0,
                    block_n=block_n,
                    strategy=strategy,
                    first_warps=first,
                    second_warps=second,
                )
            ],
        )
        for size, block_n, strategy, first, second in entries
    ]
    return VocabBucket(size=bucket.size, batch=PriorityBucket(batch))


__all__ = [
    "MAX_K",
    "Indicator",
    "SamplingBuffers",
    "ScopedIndicators",
    "Selection",
    "ThreeStageWarpConfig",
    "TopKStrategy",
    "TwoStageTiling",
    "TwoStageWarpConfig",
    "Verification",
    "available",
    "fused_multistep",
    "fused_singular",
    "measured_dispatch",
]
