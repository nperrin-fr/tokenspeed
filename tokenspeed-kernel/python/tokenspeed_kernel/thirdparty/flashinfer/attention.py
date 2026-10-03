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

"""Persistent multi-CTA KV completion counters for trtllm-gen attention."""

from __future__ import annotations

import torch
from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

_counters: dict[torch.device, torch.Tensor] = {}
# Captured graphs keep pointing at a replaced buffer, so it stays alive.
_retired_counters: list[torch.Tensor] = []


def trtllm_gen_counter_buffer(
    device: torch.device | str, batch_size: int, num_heads: int
) -> torch.Tensor:
    """Return this device's zeroed counters, grown to cover ``batch_size * num_heads``.

    The kernels reset their counters at the end of every launch, so one buffer
    serves every later launch on the device; without it FlashInfer allocates
    and zero-fills a fresh buffer per call. Like the shared workspace, launches
    that use it must not run concurrently.

    Args:
        device: CUDA device of the attention inputs.
        batch_size: Requests in the launch.
        num_heads: Query heads in the launch.

    Returns:
        A zero-initialized uint8 buffer of at least the required size.
    """
    device = torch.device(device)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    required = get_trtllm_gen_multi_ctas_kv_counter_bytes(
        batch_size, num_heads, sm_count
    )
    counters = _counters.get(device)
    if counters is None or counters.numel() < required:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm-gen counters must be sized by an eager launch before "
                "CUDA graph capture"
            )
        if counters is not None:
            _retired_counters.append(counters)
        counters = torch.zeros(required, dtype=torch.uint8, device=device)
        _counters[device] = counters
    return counters
