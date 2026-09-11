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

"""The Lamport gather's work-derived launch geometry."""

from __future__ import annotations

from tokenspeed_kernel.ops.moe.latent_down import (
    _LAMPORT_CTAS,
    _LAMPORT_NARROW_THREADS,
    _LAMPORT_THREADS,
    _LAMPORT_VEC,
    _lamport_geometry,
)

LATENT = 3584


def test_a_narrow_batch_covers_its_fragments_once():
    for m in (1, 8, 64, 128):
        ctas, threads = _lamport_geometry(m, LATENT)
        assert threads == _LAMPORT_NARROW_THREADS
        assert ctas * threads >= m * (LATENT // _LAMPORT_VEC)


def test_past_the_cta_budget_it_yields_to_the_wide_pair():
    # A partial narrow grid measured worse than the wide pair, so the rule must
    # never emit one: above the budget it returns the pair unchanged.
    wide = (_LAMPORT_CTAS, _LAMPORT_THREADS)
    assert _lamport_geometry(256, LATENT) == wide
    assert _lamport_geometry(1280, LATENT) == wide


def test_the_grid_never_exceeds_the_cta_budget():
    for m in range(1, 1281):
        ctas, _ = _lamport_geometry(m, LATENT)
        assert 0 < ctas <= _LAMPORT_CTAS


def test_the_build_stays_small():
    # One kernel per warp-count step, not one per width: an exact per-width
    # grid would ask the JIT for 174 variants at startup.
    distinct = {_lamport_geometry(m, LATENT) for m in range(1, 1281)}
    assert len(distinct) <= 24, len(distinct)
