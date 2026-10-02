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

"""Slot invariants for the hoisted AttnRes mlp-side partial.

Layer L's mlp-side partial is computed on layer L-1's aux sweep, so it lands a
layer before the all-reduce that reads it. That is safe only while a layer's
own slot differs from the one its aux branch writes for the next layer.

Usage:
    cd test/runtime
    python3 -m unittest models.test_kimi_k3_attnres_hoist -v
"""

import os
import sys
import unittest

import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.models.kimi_k3 import _attnres_mlp_slot, _attnres_scratch

register_cuda_ci(est_time=5, suite="runtime-1gpu")

LAYERS = 93


class TestAttnResMlpHoist(unittest.TestCase):
    def test_adjacent_layers_use_different_mlp_slots(self):
        """A layer must not read the slot its own aux branch is writing."""
        for i in range(LAYERS - 1):
            self.assertNotEqual(_attnres_mlp_slot(i), _attnres_mlp_slot(i + 1))

    def test_mlp_slots_never_take_the_attn_side_slot(self):
        """Slot 1 belongs to the attn-side mix."""
        for i in range(LAYERS):
            self.assertNotEqual(_attnres_mlp_slot(i), 1)

    def test_scratch_pool_serves_every_slot_without_aliasing(self):
        """Every slot in use must be a distinct buffer."""
        like = torch.zeros(1, 8)
        used = {1} | {_attnres_mlp_slot(i) for i in range(LAYERS)}
        ptrs = {s: _attnres_scratch(like, slot=s)[2].data_ptr() for s in sorted(used)}
        self.assertEqual(len(set(ptrs.values())), len(ptrs), ptrs)


if __name__ == "__main__":
    unittest.main()
