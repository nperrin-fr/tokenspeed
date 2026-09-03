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

import subprocess
import sys
from pathlib import Path


def _run(repo_root: Path, test_id: str, *, should_pass: bool) -> None:
    result = subprocess.run(
        (sys.executable, "-m", "pytest", "-q", test_id),
        check=False,
        cwd=repo_root,
    )
    passed = result.returncode == 0
    if passed != should_pass:
        expectation = "pass" if should_pass else "fail"
        raise RuntimeError(f"expected {test_id} to {expectation}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    registry_path = repo_root / "python/tokenspeed/runtime/layers/attention/registry.py"
    test_prefix = (
        "test/runtime/test_cache_setup.py::"
        "test_resolve_backend_args_does_not_mutate_caller"
    )
    mutations = (
        (
            "deepseek-v4-target",
            '        resolved_server_args.attention_backend = "deepseek_v4"',
            '        server_args.attention_backend = "deepseek_v4"',
        ),
        (
            "deepseek-v4-draft",
            '        resolved_server_args.drafter_attention_backend = "deepseek_v4"',
            '        server_args.drafter_attention_backend = "deepseek_v4"',
        ),
        (
            "hybrid-linear-model",
            '        resolved_server_args.attention_backend = "hybrid_linear_attn"',
            '        server_args.attention_backend = "hybrid_linear_attn"',
        ),
        (
            "non-hybrid-model-resets-explicit-hybrid-backends",
            "        resolved_server_args.attention_backend = None",
            "        server_args.attention_backend = None",
        ),
    )
    original = registry_path.read_bytes()
    try:
        for case_id, old, new in mutations:
            test_id = f"{test_prefix}[{case_id}]"
            _run(repo_root, test_id, should_pass=True)
            source = original.decode()
            if source.count(old) != 1:
                raise RuntimeError(f"mutation anchor is not unique: {old!r}")
            registry_path.write_text(source.replace(old, new), encoding="utf-8")
            try:
                _run(repo_root, test_id, should_pass=False)
            finally:
                registry_path.write_bytes(original)
    finally:
        registry_path.write_bytes(original)


if __name__ == "__main__":
    main()
