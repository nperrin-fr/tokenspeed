# SPDX-License-Identifier: Apache-2.0

"""Thin availability guard for the vendored SGLang KDA MTP kernel."""

from __future__ import annotations


def is_available() -> bool:
    """Return whether the CuTe DSL dependencies are importable."""
    try:
        import cuda.bindings.driver  # noqa: F401
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
    except ImportError:
        return False
    return True


def fused_kda_decode_mtp_dspark(*args, **kwargs):
    """Load and call the vendored, internally compile-cached wrapper."""
    if not is_available():
        raise ImportError("CuTe DSL and cuda-python are required for KDA MTP verify")
    from tokenspeed_kernel.thirdparty.cute_dsl.kda_mtp._kernel import (
        fused_kda_decode_mtp_dspark as implementation,
    )

    return implementation(*args, **kwargs)


__all__ = ["fused_kda_decode_mtp_dspark", "is_available"]
