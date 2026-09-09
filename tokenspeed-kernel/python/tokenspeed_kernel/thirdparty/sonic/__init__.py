# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026 SonicSampler Team.
# Adapted for tokenspeed-kernel from sonic-sampler v1.0.0
# (https://github.com/tachyontrails/sonic-sampler, Apache-2.0): the anchored
# substitutions below quote and rewrite lines of its kernels and wrappers.

"""sonic-sampler (v1.0.0) with two source-level patches, applied at import.

sonic's kernels are imported as is and re-jitted from their own source with
a handful of asserted, line-anchored substitutions, so this module carries
the *difference* to upstream rather than a copy of it. The anchors are the
v1.0.0 source: an upstream change that moves one raises ``ImportError`` here,
at import, instead of drifting silently.

1. ``batch_size`` becomes a runtime integer instead of a ``tl.constexpr``.
   Upstream compiles a separate kernel per distinct batch size (2-4s each);
   under serving that is a stall for every batch size the eager mixed rounds
   happen to produce.
2. The Gumbel weights and the acceptance coins can be drawn in-kernel from
   Philox(seed[slot], offset[slot]) at the candidate indices the kernels
   actually read (``noise_seeds`` / ``noise_offsets`` / ``noise_steps`` on
   the functional wrappers), so no ``[B, gamma+1, V]`` noise plane exists and
   nothing is refreshed per step. The plane path is untouched when those
   arguments are absent (upstream indexes a caller-supplied plane by slot, so
   it only works without ``slot_mapping``).
3. Vocab-sharded sampling exchanges sonic's packed slabs through a multicast
   ``SlabExchange`` (``exchange=`` on the functional wrappers): the wrappers
   publish after the reduction, and the selection / unpack kernels acquire-poll
   the peers' flags before merging the ``[W, Z_v, M]`` union. Upstream's greedy
   merge decoded the flat ``[W, Z_v']`` index by ``W``; it is fixed here.

sonic binds ``triton`` at import, so its modules are imported under the
``tokenspeed_triton`` redirect here; a sonic that some earlier import already
bound to another Triton cannot host the vendored noise helpers and is refused
(``ImportError``), like every other incompatibility this module detects.
"""

from __future__ import annotations

import inspect
import linecache
import sys
import textwrap
from types import ModuleType
from typing import Any

from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton, triton
from tokenspeed_kernel.ops.sampling.triton.exchange import fence_sys
from tokenspeed_kernel.thirdparty.sonic.noise import gumbel_noise, step_seed


def _source(obj: Any) -> str:
    fn = obj.fn if isinstance(obj, triton.JITFunction) else obj
    return textwrap.dedent(inspect.getsource(fn))


def _patched(
    module: ModuleType,
    name: str,
    subs: list[tuple[str, str]],
    namespace: dict[str, Any],
) -> Any:
    """Re-create ``module.name`` from its source with every ``(old, new)``
    substitution applied exactly once, evaluated in ``namespace`` (the
    module's globals plus any already-patched callees). Binds ``name`` in
    ``namespace`` as a side effect of the ``exec``."""

    if name not in module.__dict__:
        raise ImportError(
            f"{module.__name__} does not define {name}; patch would not apply"
        )
    if inspect.getsourcefile(module) is None:
        raise ImportError(f"{module.__name__} has no source to patch")
    obj = module.__dict__[name]
    src = _source(obj)

    for old, new in subs:
        if src.count(old) != 1:
            raise ImportError(
                f"sonic patch anchor not found exactly once in {module.__name__}."
                f"{name}: {old!r}"
            )
        src = src.replace(old, new)

    # Triton re-reads JIT source via ``inspect.getsource``: serve the patched text from linecache.
    filename = f"<sonic-patch:{module.__name__}.{name}>"
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    exec(compile(src, filename, "exec"), namespace)

    patched = namespace[name]
    if _source(patched) != src:
        raise ImportError(f"patched source of {name} does not round-trip")

    return patched


def _namespace(module: ModuleType, **patched: Any) -> dict[str, Any]:
    """``module``'s globals with the noise helpers and ``patched`` callees
    bound over the names the module imported them under."""

    missing = sorted(set(patched) - set(module.__dict__))
    if missing:
        raise ImportError(
            f"{module.__name__} does not bind {missing}; patch would not apply"
        )
    ns = dict(module.__dict__)
    ns.update(gumbel_noise=gumbel_noise, step_seed=step_seed, fence_sys=fence_sys)
    ns.update(patched)
    return ns


# --------------------------------------------------------------------------- #
# Patch 1: batch_size as a runtime int (prologue + multistep unpack).
# --------------------------------------------------------------------------- #
_BATCH_SIZE_HELPER = [("    batch_size: tl.constexpr,\n", "    batch_size,\n")]
_BATCH_SIZE_KERNEL = [
    (
        "    batch_size: tl.constexpr,           # Total Sequences -> B.\n",
        "    batch_size,                         # Total Sequences -> B.\n",
    )
]

# --------------------------------------------------------------------------- #
# Patch 2: in-kernel noise.
# --------------------------------------------------------------------------- #
_SELECTION_KERNEL = [
    (
        "    g_ptr,                              # Gumbel Noise -> [ B, V_t ].\n",
        "    g_ptr,                              # Gumbel Noise -> [ B, V_t ] (None: in-kernel).\n"
        "    sd_ptr,                             # Noise Seeds -> [ B ].\n"
        "    of_ptr,                             # Noise Offsets -> [ B ].\n",
    ),
    (
        "    probabilities: tl.constexpr,        # Draft Probabilities Flag.\n",
        "    probabilities: tl.constexpr,        # Draft Probabilities Flag.\n"
        "    noise_steps: tl.constexpr,          # Timesteps per slot in the noise stream.\n",
    ),
    (
        "        noise = tl.load(g_ptr + (batch_id * stride_g) + indices)\n",
        "        if g_ptr is None:\n"
        "\n"
        "            noise = gumbel_noise(\n"
        "                step_seed(sd_ptr, of_ptr, batch_id, 0, noise_steps), indices\n"
        "            )\n"
        "\n"
        "        else:\n"
        "\n"
        "            noise = tl.load(g_ptr + (batch_id * stride_g) + indices)\n",
    ),
]

_VERIFY_DRAFTED = [
    (
        "    d_ptr,\n    u_ptr,\n    batch_id,\n",
        "    d_ptr,\n    u_ptr,\n    sd_ptr,\n    of_ptr,\n    batch_id,\n",
    ),
    (
        "    mask,\n    stride_c: tl.constexpr,\n    stride_d: tl.constexpr,\n"
        "    stride_u: tl.constexpr,\n):\n",
        "    mask,\n    block_g: tl.constexpr,\n    noise_steps: tl.constexpr,\n"
        "    stride_c: tl.constexpr,\n    stride_d: tl.constexpr,\n"
        "    stride_u: tl.constexpr,\n):\n",
    ),
    (
        "    uniform = tl.load(u_ptr + (batch_id * stride_u) + span, mask=mask, other=0)\n",
        "    if u_ptr is None:\n"
        "\n"
        "        # In-kernel coins: Philox counter V (Gumbel counters end at V//4), U in (0, 1].\n"
        "\n"
        "        uniform = tl.zeros((block_g,), dtype=tl.float32)\n"
        "\n"
        "        for step in tl.static_range(block_g):\n"
        "\n"
        "            seed = step_seed(sd_ptr, of_ptr, batch_id, step, noise_steps)\n"
        "            coin = tl.maximum(\n"
        "                1.0 - tl.rand(seed, stride_c + tl.zeros((1,), dtype=tl.int32)), 1.0e-7\n"
        "            )\n"
        "\n"
        "            uniform = tl.where(span == step, coin.broadcast_to((block_g,)), uniform)\n"
        "\n"
        "        uniform = tl.where(mask, uniform, 0.0)\n"
        "\n"
        "    else:\n"
        "\n"
        "        uniform = tl.load(u_ptr + (batch_id * stride_u) + span, mask=mask, other=0)\n",
    ),
]

_REJECTION_SAMPLE = [
    (
        "    d_ptr,\n    e_ptr,\n    batch_id,\n",
        "    d_ptr,\n    e_ptr,\n    sd_ptr,\n    of_ptr,\n    batch_id,\n",
    ),
    (
        "    max_k: tl.constexpr,\n    update_token: tl.constexpr,\n",
        "    max_k: tl.constexpr,\n    noise_steps: tl.constexpr,\n    update_token: tl.constexpr,\n",
    ),
    (
        "    gumbel = tl.load(e_ptr + (batch_id * stride_e) + shifts)\n",
        "    if e_ptr is None:\n"
        "\n"
        "        gumbel = gumbel_noise(\n"
        "            step_seed(sd_ptr, of_ptr, batch_id, accepted, noise_steps), positions\n"
        "        )\n"
        "\n"
        "    else:\n"
        "\n"
        "        gumbel = tl.load(e_ptr + (batch_id * stride_e) + shifts)\n",
    ),
]

_VERIFY_KERNEL = [
    (
        "    e_ptr,                              # Gumbel Noise -> [ B • (γ + 1), V ].\n",
        "    e_ptr,                              # Gumbel Noise -> [ B • (γ + 1), V ] (None: in-kernel).\n"
        "    sd_ptr,                             # Noise Seeds -> [ B ].\n"
        "    of_ptr,                             # Noise Offsets -> [ B ].\n",
    ),
    (
        "    logprobs: tl.constexpr,             # Log-Probabilities Flag.\n    # Stride(s).\n",
        "    logprobs: tl.constexpr,             # Log-Probabilities Flag.\n"
        "    noise_steps: tl.constexpr,          # Timesteps per slot in the noise stream.\n"
        "    # Stride(s).\n",
    ),
    (
        "        accepted, targets, matches = verify_drafted(\n            d_ptr,\n            u_ptr,\n"
        "            batch_id,\n",
        "        accepted, targets, matches = verify_drafted(\n            d_ptr,\n            u_ptr,\n"
        "            sd_ptr,\n            of_ptr,\n            batch_id,\n",
    ),
    (
        "            draft_mask,\n            stride_c,\n            stride_d,\n            stride_u,\n        )\n",
        "            draft_mask,\n            block_g,\n            noise_steps,\n            stride_c,\n"
        "            stride_d,\n            stride_u,\n        )\n",
    ),
    (
        "        token, targets, selections = rejection_sample(\n            d_ptr,\n            e_ptr,\n"
        "            batch_id,\n",
        "        token, targets, selections = rejection_sample(\n            d_ptr,\n            e_ptr,\n"
        "            sd_ptr,\n            of_ptr,\n            batch_id,\n",
    ),
    (
        "            block_g,\n            max_k,\n            update_counts,\n            logprobs,\n",
        "            block_g,\n            max_k,\n            noise_steps,\n            update_counts,\n"
        "            logprobs,\n",
    ),
]

_FUSED_SINGULAR = [
    (
        "    gumbel_noise: Tensor | None = None,\n    # Output Buffer(s).\n",
        "    gumbel_noise: Tensor | None = None,\n"
        "    # In-kernel noise: Philox(seed[slot], offset[slot]) instead of the plane.\n"
        "    noise_seeds: Tensor | None = None,\n"
        "    noise_offsets: Tensor | None = None,\n"
        "    noise_steps: int | None = None,\n"
        "    # Output Buffer(s).\n",
    ),
    (
        "    gumbel_weights = resolve_gumbel(gumbel_noise, logits)\n",
        "    if noise_seeds is not None and noise_offsets is None:\n"
        "\n"
        '        raise ValueError("`noise_seeds` requires `noise_offsets`")\n'
        "\n"
        "    gumbel_weights = (\n"
        "        None if noise_seeds is not None else resolve_gumbel(gumbel_noise, logits)\n"
        "    )\n"
        "\n"
        "    noise_steps = noise_steps or 1\n",
    ),
    (
        "    stride_n = gumbel_weights.stride(0)\n",
        "    stride_n = conditional_stride(gumbel_weights)\n",
    ),
    (
        "        decode_counts,\n        gumbel_weights,\n        top_k,\n",
        "        decode_counts,\n        gumbel_weights,\n        noise_seeds,\n        noise_offsets,\n"
        "        top_k,\n",
    ),
    (
        "        return_logprobs,\n        return_probabilities,\n        # Stride(s).\n",
        "        return_logprobs,\n        return_probabilities,\n        noise_steps,\n        # Stride(s).\n",
    ),
]

_FUSED_MULTISTEP = [
    (
        "    uniform_noise: Tensor | None = None,\n    # Output Buffer(s).\n",
        "    uniform_noise: Tensor | None = None,\n"
        "    # In-kernel noise: Philox(seed[slot], offset[slot]) instead of the planes.\n"
        "    noise_seeds: Tensor | None = None,\n"
        "    noise_offsets: Tensor | None = None,\n"
        "    noise_steps: int | None = None,\n"
        "    # Output Buffer(s).\n",
    ),
    (
        "    gumbel_weights = resolve_gumbel(gumbel_noise, logits, batch_size, gamma + 1)\n"
        "    uniform_weights = resolve_uniform(uniform_noise, logits, batch_size, gamma)\n",
        "    if noise_seeds is not None:\n"
        "\n"
        "        if noise_offsets is None:\n"
        "\n"
        '            raise ValueError("`noise_seeds` requires `noise_offsets`")\n'
        "\n"
        "        gumbel_weights = uniform_weights = None\n"
        "\n"
        "    else:\n"
        "\n"
        "        gumbel_weights = resolve_gumbel(gumbel_noise, logits, batch_size, gamma + 1)\n"
        "        uniform_weights = resolve_uniform(uniform_noise, logits, batch_size, gamma)\n"
        "\n"
        "    noise_steps = noise_steps or (gamma + 1)\n",
    ),
    (
        "    stride_n = gumbel_weights.stride(0)\n    stride_u = uniform_weights.stride(0)\n\n"
        "    stride_c = collapse_2d(gumbel_weights).stride(0)\n",
        "    stride_n = conditional_stride(gumbel_weights)\n"
        "    stride_u = conditional_stride(uniform_weights)\n\n"
        "    # Per-step stride of the gumbel plane; with in-kernel noise, the coin\n"
        "    # offset and the decode-counts stride: the target vocab width.\n"
        "    stride_c = (\n"
        "        collapse_2d(gumbel_weights).stride(0)\n"
        "        if gumbel_weights is not None\n"
        "        else vocab_size\n"
        "    )\n",
    ),
    (
        "        uniform_weights,\n        gumbel_weights,\n        decode_counts,\n",
        "        uniform_weights,\n        gumbel_weights,\n        noise_seeds,\n        noise_offsets,\n"
        "        decode_counts,\n",
    ),
    (
        "        update_counts,\n        return_logprobs,\n        # Stride(s).\n        stride_d,\n",
        "        update_counts,\n        return_logprobs,\n        noise_steps,\n        # Stride(s).\n"
        "        stride_d,\n",
    ),
]


# --------------------------------------------------------------------------- #
# Patch 3: vocab-sharded exchange.
# --------------------------------------------------------------------------- #
_SHARDED_MAXIMUM = [
    (
        "        ((index // world_size) * shard_size)\n"
        "        + ((index % world_size) * block_n)\n",
        "        ((index // block_v) * shard_size)\n"
        "        + ((index % block_v) * block_n)\n",
    )
]

# The wait prologue shared by the selection and unpack kernels (sharded scratch only).
_WAIT_PROLOGUE = (
    "    if fl_ptr is not None:\n"
    "\n"
    "        rnd = tl.load(rd_ptr)\n"
    "        band = rnd & 1\n"
    "        flag_ids = tl.arange(0, n_flags)\n"
    "        flag_mask = flag_ids < world_size * vocab_blocks\n"
    "        flag_base = fl_ptr + (band * band_rows + row_id) * (world_size * vocab_blocks)\n"
    "        pending = tl.min(\n"
    "            tl.load(flag_base + flag_ids, mask=flag_mask, other=rnd, volatile=True), axis=0\n"
    "        ) < rnd\n"
    "\n"
    "        while pending:\n"
    "\n"
    "            pending = tl.min(\n"
    "                tl.load(flag_base + flag_ids, mask=flag_mask, other=rnd, volatile=True), axis=0\n"
    "            ) < rnd\n"
    "\n"
    "        fence_sys(rnd)\n"
    "        s_ptr += band * band_rows * stride_s\n"
    "\n"
)
_EXCHANGE_POINTERS = (
    "    s_ptr,                              # Scratch-Pad -> [ L, Z_v • M ].\n",
    "    s_ptr,                              # Scratch-Pad -> [ L, Z_v • M ] (exchange: [ 2 • R, W • Z_v • M ]).\n"
    "    fl_ptr,                             # Exchange Flags -> [ 2, R, W, Z_v ] (None: local scratch).\n"
    "    rd_ptr,                             # Exchange Round -> [ 1 ].\n",
)
_EXCHANGE_CONSTEXPRS = (
    "    band_rows: tl.constexpr,            # Exchange Band Rows -> R.\n"
    "    n_flags: tl.constexpr,              # Next Power-of-2 Exchange Flags per Row -> W • Z_v.\n"
    ") -> None:\n"
)
_SELECTION_EXCHANGE = [
    _EXCHANGE_POINTERS,
    (
        "    block_q: tl.constexpr,              # Next Power-of-2 Top-K Log-Probabilities Targets -> T'.\n"
        ") -> None:\n",
        "    block_q: tl.constexpr,              # Next Power-of-2 Top-K Log-Probabilities Targets -> T'.\n"
        + _EXCHANGE_CONSTEXPRS,
    ),
    (
        "    row_id = tl.program_id(axis=0)\n    batch_id = row_id\n\n",
        "    row_id = tl.program_id(axis=0)\n    batch_id = row_id\n\n"
        + _WAIT_PROLOGUE,
    ),
]
_UNPACK_EXCHANGE = [
    _EXCHANGE_POINTERS,
    (
        "    block_q: tl.constexpr,              # Block Size of Cumulative Length(s).\n) -> None:\n",
        "    block_q: tl.constexpr,              # Block Size of Cumulative Length(s).\n"
        + _EXCHANGE_CONSTEXPRS,
    ),
    (
        "    row_id = tl.program_id(axis=0)\n\n    if enable_pdl:\n",
        "    row_id = tl.program_id(axis=0)\n\n"
        + _WAIT_PROLOGUE
        + "    if enable_pdl:\n",
    ),
]

# Wrapper side: publish between the stages, hand the exchange buffers to stage 2, bump after.
_EXCHANGE_STAGE = (
    "    scratch, flags, rounds = bitpacked, None, None\n"
    "    stride_s, stride_r, stride_b = stride_y, None, None\n"
    "    band_rows, n_flags = 0, 0\n"
    "\n"
    "    if exchange is not None:\n"
    "\n"
    "        exchange.publish(bitpacked, vocab_blocks)\n"
    "        scratch, flags, rounds = exchange.scratch, exchange.flags, exchange.round\n"
    "        stride_s = exchange.scratch.stride(0)\n"
    "        stride_r, stride_b = vocab_blocks * MAX_K, block_v * MAX_K\n"
    "        band_rows, n_flags = exchange.band_rows, exchange.n_flags\n"
    "\n"
)
_FUSED_SINGULAR_EXCHANGE = [
    (
        "    warp_config: TwoStageWarpConfig | None = None,\n) -> Selection:\n",
        "    warp_config: TwoStageWarpConfig | None = None,\n"
        "    # Vocab-sharded exchange (world_size > 1): publish the slab, merge the union.\n"
        "    exchange: object | None = None,\n"
        ") -> Selection:\n",
    ),
    (
        "    cumulative_selection_kernel[grid_1d](\n"
        "        # Input Data Pointer(s).\n"
        "        bitpacked,\n"
        "        indicators,\n",
        _EXCHANGE_STAGE + "    cumulative_selection_kernel[grid_1d](\n"
        "        # Input Data Pointer(s).\n"
        "        scratch,\n"
        "        flags,\n"
        "        rounds,\n"
        "        indicators,\n",
    ),
    (
        "        stride_y,\n        None,\n        None,\n        stride_f,\n",
        "        stride_s,\n        stride_r,\n        stride_b,\n        stride_f,\n",
    ),
    (
        "        block_z,\n"
        "        # Tuning Configuration.\n"
        "        num_warps=config.second,\n"
        "    )\n"
        "\n"
        "    return selection\n",
        "        block_z,\n"
        "        band_rows,\n"
        "        n_flags,\n"
        "        # Tuning Configuration.\n"
        "        num_warps=config.second,\n"
        "    )\n"
        "\n"
        "    if exchange is not None:\n"
        "\n"
        "        exchange.bump()\n"
        "\n"
        "    return selection\n",
    ),
]
_FUSED_MULTISTEP_EXCHANGE = [
    (
        "    warp_config: ThreeStageWarpConfig | None = None,\n) -> Verification:\n",
        "    warp_config: ThreeStageWarpConfig | None = None,\n"
        "    # Vocab-sharded exchange (world_size > 1): publish the slab, merge the union.\n"
        "    exchange: object | None = None,\n"
        ") -> Verification:\n",
    ),
    (
        "    cumulative_unpack_kernel[grid_1d](\n"
        "        # Input Data Pointer(s).\n"
        "        bitpacked,\n"
        "        indicators,\n",
        _EXCHANGE_STAGE + "    cumulative_unpack_kernel[grid_1d](\n"
        "        # Input Data Pointer(s).\n"
        "        scratch,\n"
        "        flags,\n"
        "        rounds,\n"
        "        indicators,\n",
    ),
    (
        "        stride_y,\n"
        "        None,\n"
        "        None,\n"
        "        # Kernel Specific(s).\n"
        "        block_n,\n"
        "        block_v,\n"
        "        block_q,\n"
        "        # Tuning Configuration.\n"
        "        num_warps=config.second,\n"
        "    )\n",
        "        stride_s,\n"
        "        stride_r,\n"
        "        stride_b,\n"
        "        # Kernel Specific(s).\n"
        "        block_n,\n"
        "        block_v,\n"
        "        block_q,\n"
        "        band_rows,\n"
        "        n_flags,\n"
        "        # Tuning Configuration.\n"
        "        num_warps=config.second,\n"
        "    )\n"
        "\n"
        "    if exchange is not None:\n"
        "\n"
        "        exchange.bump()\n",
    ),
]


def _apply() -> tuple[Any, Any]:
    with redirect_triton_to_tokenspeed_triton():
        import sonic_sampler.interface.functional.multistep as f_multistep
        import sonic_sampler.interface.functional.singular as f_singular
        import sonic_sampler.ops.multistep as multistep
        import sonic_sampler.ops.prologue as prologue
        import sonic_sampler.ops.singular as singular
        import sonic_sampler.ops.verify as verify

    for name, module in list(sys.modules.items()):
        bound = name.startswith("sonic_sampler.") and "jit" in module.__dict__
        if bound and module.__dict__["jit"] is not triton.jit:
            raise ImportError(
                f"{name} was imported before tokenspeed_kernel bound sonic-sampler to "
                "tokenspeed_triton"
            )
    for wrapper in (f_singular, f_multistep):
        if "conditional_stride" not in wrapper.__dict__:
            raise ImportError(
                f"{wrapper.__name__} does not bind conditional_stride; patch would not apply"
            )

    # Kernels: patch callees first so the kernels re-jit against them.
    ns = _namespace(prologue)
    _patched(prologue, "resolve_indices", _BATCH_SIZE_HELPER, ns)
    reduction = _patched(prologue, "bitpacked_reduction_kernel", _BATCH_SIZE_KERNEL, ns)

    ns = _namespace(singular)
    _patched(singular, "sharded_maximum", _SHARDED_MAXIMUM, ns)
    greedy = _patched(singular, "greedy_maximum", [], ns)
    selection = _patched(
        singular,
        "cumulative_selection_kernel",
        _SELECTION_KERNEL + _SELECTION_EXCHANGE,
        ns,
    )

    # The unpack kernel imports greedy_maximum from singular: bind the fixed one.
    ns = _namespace(multistep, greedy_maximum=greedy)
    _patched(multistep, "resolve_indices", _BATCH_SIZE_HELPER, ns)
    unpack = _patched(
        multistep, "cumulative_unpack_kernel", _BATCH_SIZE_KERNEL + _UNPACK_EXCHANGE, ns
    )

    ns = _namespace(verify)
    _patched(verify, "verify_drafted", _VERIFY_DRAFTED, ns)
    _patched(verify, "rejection_sample", _REJECTION_SAMPLE, ns)
    verification = _patched(
        verify, "chain_speculative_verification_kernel", _VERIFY_KERNEL, ns
    )

    # Wrappers: re-created against the patched kernels.
    ns = _namespace(
        f_singular,
        bitpacked_reduction_kernel=reduction,
        cumulative_selection_kernel=selection,
    )
    fused_singular = _patched(
        f_singular, "fused_singular", _FUSED_SINGULAR + _FUSED_SINGULAR_EXCHANGE, ns
    )

    ns = _namespace(
        f_multistep,
        bitpacked_reduction_kernel=reduction,
        cumulative_unpack_kernel=unpack,
        chain_speculative_verification_kernel=verification,
    )
    fused_multistep = _patched(
        f_multistep, "fused_multistep", _FUSED_MULTISTEP + _FUSED_MULTISTEP_EXCHANGE, ns
    )

    return fused_singular, fused_multistep


fused_singular, fused_multistep = _apply()

__all__ = ["fused_multistep", "fused_singular"]
