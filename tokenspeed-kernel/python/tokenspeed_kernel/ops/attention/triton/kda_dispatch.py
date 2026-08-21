# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, Songlin Yang, Yu Zhang,
# Zhiyuan Li
#
# The adapters in this file preserve the NVIDIA KDA implementations behind one
# public kernel contract.

"""Registered adapters for KDA implementations."""

from __future__ import annotations

from collections.abc import Callable

import torch
from tokenspeed_kernel.ops.attention.kda_utils import KdaPrefillResult
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_DENSE_HALF_SIGNATURES = format_signatures(
    ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
)
# Descriptor addresses are dereferenced as bf16, so registrations exclude fp16.
_DENSE_BF16_SIGNATURES = format_signatures(("q", "k", "v"), "dense", {torch.bfloat16})


def _nvidia_fused_decode(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None,
    output_gate: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float | None = None,
    recurrent_layout: str,
) -> torch.Tensor:
    """Shared body for the K-major and V-major NVIDIA decode registrations."""
    from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
        fused_recurrent_kda_megafuse,
    )

    return fused_recurrent_kda_megafuse(
        mixed_qkv,
        conv_weights,
        conv_states,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias,
        h_pool=state_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
        output_gate=output_gate,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        recurrent_layout=recurrent_layout,
    ).view(1, -1, num_heads, head_dim)


@register_kernel(
    "attention",
    "kda_fused_paged_decode",
    name="triton_nvidia_kda_fused_paged_decode",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        "paged_state": frozenset({True}),
        "fused_output_norm": frozenset({False, True}),
        "recurrent_layout": frozenset({"k_major"}),
    },
    tags={"nvidia", "paged_cache", "cuda_graph", "fusion"},
)
def triton_nvidia_kda_fused_paged_decode(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None,
    output_gate: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float | None = None,
) -> torch.Tensor:
    """Adapt dev's NVIDIA conv/GEMV/recurrent megafusion."""
    return _nvidia_fused_decode(
        mixed_qkv,
        conv_weights,
        conv_states,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias,
        state_pool=state_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
        output_gate=output_gate,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        recurrent_layout="k_major",
    )


@register_kernel(
    "attention",
    "kda_fused_paged_decode",
    name="triton_nvidia_kda_fused_paged_decode_vmajor",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        "paged_state": frozenset({True}),
        "fused_output_norm": frozenset({False, True}),
        "recurrent_layout": frozenset({"v_major"}),
    },
    tags={"nvidia", "paged_cache", "cuda_graph", "fusion"},
)
def triton_nvidia_kda_fused_paged_decode_vmajor(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None,
    output_gate: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float | None = None,
) -> torch.Tensor:
    """Decode against a physically V-major ``[pages, HV, V, K]`` state slab."""
    return _nvidia_fused_decode(
        mixed_qkv,
        conv_weights,
        conv_states,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias,
        state_pool=state_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
        output_gate=output_gate,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        recurrent_layout="v_major",
    )


def _nvidia_fused_verify(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    conv_scratch: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    state_scratch: torch.Tensor | None,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None,
    store_states: bool,
    recurrent_layout: str,
    split_producers: bool = False,
) -> torch.Tensor:
    """Shared body for the NVIDIA target-verify registrations."""
    from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
        fused_kda_verify_conv_update,
        fused_recurrent_kda_verify_megafuse,
    )

    conv_qkv = None
    g_raw = None
    if split_producers:
        conv_qkv = fused_kda_verify_conv_update(
            mixed_qkv,
            conv_weights,
            conv_states,
            read_indices,
            num_heads=num_heads,
            head_dim=head_dim,
            draft_token_num=draft_token_num,
        )
        g_raw = torch.mm(f_a_out, f_b_weight.t()).contiguous()

    return fused_recurrent_kda_verify_megafuse(
        mixed_qkv,
        conv_weights,
        conv_states,
        conv_scratch,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias,
        state_pool,
        state_pool if state_scratch is None else state_scratch,
        read_indices,
        write_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
        store_states=store_states,
        recurrent_layout=recurrent_layout,
        g_raw=g_raw,
        conv_qkv=conv_qkv,
    ).view(1, -1, num_heads, head_dim)


@register_kernel(
    "attention",
    "kda_fused_paged_verify",
    name="cutedsl_kda_mtp_verify",
    solution="cute_dsl",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}), min_arch_version=ArchVersion(9, 0)
    ),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        "paged_state": frozenset({True}),
        "recurrent_layout": frozenset({"v_major"}),
        "split_producers": frozenset({True}),
        "replay_ring": frozenset({True}),
    },
    tags={"nvidia", "cute_dsl", "paged_cache", "speculative", "replay_ring"},
)
def cutedsl_kda_mtp_verify(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    conv_scratch: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    state_scratch: torch.Tensor | None,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None,
    cu_seqlens: torch.Tensor,
    replay_rawv: torch.Tensor,
    replay_rawk: torch.Tensor,
    replay_g: torch.Tensor,
    replay_beta: torch.Tensor,
    replay_conv_q: torch.Tensor,
    replay_conv_k: torch.Tensor,
    replay_conv_v: torch.Tensor,
) -> torch.Tensor:
    """Run CuTe MTP verify with caller-owned replay rings and conv tapes.

    ``replay_rawv/rawk/g/beta`` are slot-indexed replay rings. The three
    ``replay_conv_*`` tensors are ``[slots, T, H*D, conv_width-1]`` tapes.
    ``conv_weights`` must contain the three FP32 Q/K/V convolution weights,
    and ``state_pool`` must be contiguous ``[slots, H, V, K]`` FP32 storage.
    """
    del conv_scratch, state_scratch, write_indices
    if head_dim != 128:
        raise ValueError("the vendored KDA MTP kernel requires head_dim=128")
    if conv_weights.dtype is not torch.float32:
        raise ValueError("cutedsl KDA MTP requires FP32 convolution weights")
    from tokenspeed_kernel.thirdparty.cute_dsl.kda_mtp import (
        fused_kda_decode_mtp_dspark,
    )

    rows = mixed_qkv.shape[0]
    requests = cu_seqlens.numel() - 1
    if requests <= 0 or rows != requests * draft_token_num:
        raise ValueError("rows must equal requests * draft_token_num")
    if lower_bound is None:
        raise ValueError("the vendored KDA MTP kernel requires lower_bound")
    projection = num_heads * head_dim
    x_q, x_k, x_v = (
        value.reshape(1, rows, num_heads, head_dim)
        for value in mixed_qkv.split((projection, projection, projection), dim=-1)
    )
    w_q, w_k, w_v = conv_weights.split(
        (projection, projection, projection), dim=0
    )
    conv_state = conv_states.transpose(-1, -2)
    cs_q, cs_k, cs_v = conv_state.split(
        (projection, projection, projection), dim=-1
    )
    gate = torch.mm(f_a_out, f_b_weight.t()).reshape(
        1, rows, num_heads, head_dim
    )
    out = fused_kda_decode_mtp_dspark(
        x_q=x_q,
        x_k=x_k,
        x_v=x_v,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        cs_q=cs_q.transpose(-1, -2),
        cs_k=cs_k.transpose(-1, -2),
        cs_v=cs_v.transpose(-1, -2),
        g=gate,
        beta=beta_logits.reshape(1, rows, num_heads),
        A_log=A_log,
        dt_bias=dt_bias,
        recurrent_state=state_pool,
        intermediate_ssm=None,
        intermediate_state_indices=read_indices,
        intermediate_conv_q=replay_conv_q,
        intermediate_conv_k=replay_conv_k,
        intermediate_conv_v=replay_conv_v,
        ssm_state_indices=read_indices,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
        replayssm_rawv=replay_rawv,
        replayssm_rawk=replay_rawk,
        replayssm_g=replay_g,
        replayssm_beta=replay_beta,
    )
    return out.view(1, -1, num_heads, head_dim)


@register_kernel(
    "attention",
    "kda_fused_paged_verify",
    name="triton_nvidia_kda_fused_paged_verify_no_store",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        "paged_state": frozenset({True}),
        "store_states": frozenset({False}),
        "recurrent_layout": frozenset({"k_major"}),
        "split_producers": frozenset({False}),
    },
    tags={"nvidia", "paged_cache", "cuda_graph", "fusion", "speculative"},
)
def triton_nvidia_kda_fused_paged_verify_no_store(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    conv_scratch: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    state_scratch: torch.Tensor | None,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None,
) -> torch.Tensor:
    """Run fused target verify without materializing rollback states."""
    return _nvidia_fused_verify(
        mixed_qkv,
        conv_weights,
        conv_states,
        conv_scratch,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias,
        state_pool=state_pool,
        state_scratch=state_scratch,
        read_indices=read_indices,
        write_indices=write_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
        store_states=False,
        recurrent_layout="k_major",
    )


@register_kernel(
    "attention",
    "kda_fused_paged_verify",
    name="triton_nvidia_kda_fused_paged_verify_no_store_vmajor",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        "paged_state": frozenset({True}),
        "store_states": frozenset({False}),
        "recurrent_layout": frozenset({"v_major"}),
        "split_producers": frozenset({False}),
    },
    tags={"nvidia", "paged_cache", "cuda_graph", "fusion", "speculative"},
)
def triton_nvidia_kda_fused_paged_verify_no_store_vmajor(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    conv_scratch: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    state_scratch: torch.Tensor | None,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None,
) -> torch.Tensor:
    """V-major target verify without materializing rollback states."""
    return _nvidia_fused_verify(
        mixed_qkv,
        conv_weights,
        conv_states,
        conv_scratch,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias,
        state_pool=state_pool,
        state_scratch=state_scratch,
        read_indices=read_indices,
        write_indices=write_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
        store_states=False,
        recurrent_layout="v_major",
    )


@register_kernel(
    "attention",
    "kda_fused_paged_verify",
    name="triton_nvidia_kda_fused_paged_verify_split",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        "paged_state": frozenset({True}),
        "store_states": frozenset({False}),
        "split_producers": frozenset({True}),
        "recurrent_layout": frozenset({"k_major"}),
    },
    tags={"nvidia", "paged_cache", "cuda_graph", "fusion", "speculative"},
)
def triton_nvidia_kda_fused_paged_verify_split(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    conv_scratch: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    state_scratch: torch.Tensor | None,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None,
) -> torch.Tensor:
    """Run K-major target verify with split convolution and gate producers."""
    return _nvidia_fused_verify(
        mixed_qkv,
        conv_weights,
        conv_states,
        conv_scratch,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias,
        state_pool=state_pool,
        state_scratch=state_scratch,
        read_indices=read_indices,
        write_indices=write_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
        store_states=False,
        recurrent_layout="k_major",
        split_producers=True,
    )


@register_kernel(
    "attention",
    "kda_fused_paged_verify",
    name="triton_nvidia_kda_fused_paged_verify_split_vmajor",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        "paged_state": frozenset({True}),
        "store_states": frozenset({False}),
        "split_producers": frozenset({True}),
        "recurrent_layout": frozenset({"v_major"}),
    },
    tags={"nvidia", "paged_cache", "cuda_graph", "fusion", "speculative"},
)
def triton_nvidia_kda_fused_paged_verify_split_vmajor(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    conv_scratch: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    state_scratch: torch.Tensor | None,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None,
) -> torch.Tensor:
    """Run V-major target verify with split convolution and gate producers."""
    return _nvidia_fused_verify(
        mixed_qkv,
        conv_weights,
        conv_states,
        conv_scratch,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias,
        state_pool=state_pool,
        state_scratch=state_scratch,
        read_indices=read_indices,
        write_indices=write_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
        store_states=False,
        recurrent_layout="v_major",
        split_producers=True,
    )


@register_kernel(
    "attention",
    "kda_paged_decode",
    name="triton_nvidia_kda_paged_decode",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.PERFORMANT,
    traits={
        "indexed_state": frozenset({True}),
        "recurrent_layout": frozenset({"k_major"}),
    },
    tags={"nvidia", "paged_cache", "cuda_graph"},
)
def triton_nvidia_kda_paged_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None,
    recurrent_layout: str = "k_major",
) -> torch.Tensor:
    """Adapt dev's NVIDIA indexed recurrent decode kernel."""
    return _nvidia_kda_paged_decode(
        q,
        k,
        v,
        g_raw,
        beta_logits,
        A_log,
        dt_bias,
        state_pool=state_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
        recurrent_layout="k_major",
    )


def _nvidia_kda_paged_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None,
    recurrent_layout: str,
) -> torch.Tensor:
    from tokenspeed_kernel.ops.attention.triton.linear.kda import (
        kda_recurrent_decode_pool,
    )

    return kda_recurrent_decode_pool(
        q,
        k,
        v,
        g_raw,
        beta_logits,
        A_log,
        dt_bias,
        h_pool=state_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
        recurrent_layout=recurrent_layout,
    )


@register_kernel(
    "attention",
    "kda_paged_decode",
    name="triton_nvidia_kda_paged_decode_vmajor",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.PERFORMANT,
    traits={
        "indexed_state": frozenset({True}),
        "recurrent_layout": frozenset({"v_major"}),
    },
    tags={"nvidia", "paged_cache", "cuda_graph"},
)
def triton_nvidia_kda_paged_decode_vmajor(*args, **kwargs) -> torch.Tensor:
    """Run indexed recurrent decode against a V-major state slab."""
    kwargs.pop("recurrent_layout", None)
    return _nvidia_kda_paged_decode(*args, **kwargs, recurrent_layout="v_major")


def _nvidia_kda_prefill(
    implementation: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None,
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> KdaPrefillResult:
    # Forward the host hint only to implementations that accept it.
    hint = {} if cu_seqlens_cpu is None else {"cu_seqlens_cpu": cu_seqlens_cpu}
    out, final_state = implementation(
        q,
        k,
        v,
        g_raw,
        beta_logits,
        A_log,
        dt_bias,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
        beta_is_logit=True,
        **hint,
    )
    return KdaPrefillResult(out, final_state)


@register_kernel(
    "attention",
    "kda_replay_commit",
    name="triton_sgl_replayssm_fold",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        "flat_state": frozenset({True}),
        "recurrent_layout": frozenset({"v_major"}),
        "replay_ring": frozenset({True}),
    },
    tags={"nvidia", "flat_kv", "speculative", "replay_ring"},
)
def triton_sgl_replayssm_fold(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    conv_out: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    state_out: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None,
    gate_scratch: torch.Tensor | None = None,
    recurrent_layout: str = "v_major",
    replay_rawv: torch.Tensor,
    replay_rawk: torch.Tensor,
    replay_g: torch.Tensor,
    replay_beta: torch.Tensor,
) -> None:
    """Fold accepted replay-ring prefixes into V-major state in place."""
    del (
        mixed_qkv,
        conv_weights,
        conv_states,
        conv_out,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias,
        state_out,
        write_indices,
        lower_bound,
        gate_scratch,
    )
    if recurrent_layout != "v_major":
        raise ValueError("ReplaySSM fold requires V-major recurrent state")
    if head_dim != state_pool.shape[-1]:
        raise ValueError("head_dim must match the V-major state's K dimension")
    if draft_token_num > replay_rawv.shape[2]:
        raise ValueError("replay ring is shorter than draft_token_num")
    from tokenspeed_kernel.thirdparty.triton.kda_replayssm_fold import (
        commit_kda_replayssm_spec,
    )

    commit_kda_replayssm_spec(
        checkpoint_state=state_pool,
        rawv_cache=replay_rawv,
        rawk_cache=replay_rawk,
        gk_cache=replay_g,
        beta_cache=replay_beta,
        ssm_state_indices=read_indices,
        accept_lens=accepted_length,
        max_cache_len=replay_rawv.shape[2],
        num_k_heads=num_heads,
        null_block_id=-1,
    )


@register_kernel(
    "attention",
    "kda_replay_commit",
    name="triton_nvidia_kda_replay_commit",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        "flat_state": frozenset({True}),
        "recurrent_layout": frozenset({"k_major"}),
    },
    tags={"nvidia", "flat_kv", "fusion", "speculative"},
)
def triton_nvidia_kda_replay_commit(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    conv_out: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    state_out: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None,
    gate_scratch: torch.Tensor | None = None,
    recurrent_layout: str = "k_major",
) -> None:
    """Replay the accepted prefix of a verified window into the state pool."""
    _nvidia_kda_replay_commit(
        mixed_qkv,
        conv_weights,
        conv_states,
        conv_out,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias,
        state_pool=state_pool,
        state_out=state_out,
        read_indices=read_indices,
        write_indices=write_indices,
        accepted_length=accepted_length,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
        gate_scratch=gate_scratch,
        recurrent_layout="k_major",
    )


def _nvidia_kda_replay_commit(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    conv_out: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    state_out: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None,
    gate_scratch: torch.Tensor | None,
    recurrent_layout: str,
) -> None:
    from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
        fused_recurrent_kda_replay_commit,
    )

    fused_recurrent_kda_replay_commit(
        mixed_qkv,
        conv_weights,
        conv_states,
        conv_out,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias,
        state_pool,
        state_out,
        read_indices,
        write_indices,
        accepted_length,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
        gate_scratch=gate_scratch,
        recurrent_layout=recurrent_layout,
    )


@register_kernel(
    "attention",
    "kda_replay_commit",
    name="triton_nvidia_kda_replay_commit_vmajor",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        "flat_state": frozenset({True}),
        "recurrent_layout": frozenset({"v_major"}),
    },
    tags={"nvidia", "flat_kv", "fusion", "speculative"},
)
def triton_nvidia_kda_replay_commit_vmajor(*args, **kwargs) -> None:
    """Replay and commit into a V-major recurrent slab."""
    kwargs.pop("recurrent_layout", None)
    _nvidia_kda_replay_commit(*args, **kwargs, recurrent_layout="v_major")


@register_kernel(
    "attention",
    "kda_replay_commit",
    name="triton_nvidia_kda_batched_replay_commit",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_BF16_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        "flat_state": frozenset({True}),
        "batched_layers": frozenset({True}),
        "recurrent_layout": frozenset({"k_major"}),
    },
    tags={"nvidia", "flat_kv", "fusion", "speculative", "batched_layers"},
)
def triton_nvidia_kda_batched_replay_commit(
    descriptors: torch.Tensor,
    *,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    draft_token_num: int,
    num_heads: int,
    head_dim: int,
    f_a_dim: int,
    qkv_stride: int,
    conv_stride: int,
    f_a_stride: int,
    beta_stride: int,
    state_stride: int,
    gate_stride: int,
    conv_width: int,
    layers_per_group: int,
    lower_bound: float,
) -> None:
    """Replay every KDA layer described by stable device pointer tables."""
    _nvidia_kda_batched_replay_commit(
        descriptors,
        read_indices=read_indices,
        write_indices=write_indices,
        accepted_length=accepted_length,
        draft_token_num=draft_token_num,
        num_heads=num_heads,
        head_dim=head_dim,
        f_a_dim=f_a_dim,
        qkv_stride=qkv_stride,
        conv_stride=conv_stride,
        f_a_stride=f_a_stride,
        beta_stride=beta_stride,
        state_stride=state_stride,
        gate_stride=gate_stride,
        conv_width=conv_width,
        layers_per_group=layers_per_group,
        lower_bound=lower_bound,
        recurrent_layout="k_major",
    )


def _nvidia_kda_batched_replay_commit(
    descriptors: torch.Tensor,
    *,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    draft_token_num: int,
    num_heads: int,
    head_dim: int,
    f_a_dim: int,
    qkv_stride: int,
    conv_stride: int,
    f_a_stride: int,
    beta_stride: int,
    state_stride: int,
    gate_stride: int,
    conv_width: int,
    layers_per_group: int,
    lower_bound: float,
    recurrent_layout: str,
) -> None:
    from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
        batched_recurrent_kda_replay_commit,
    )

    batched_recurrent_kda_replay_commit(
        descriptors,
        read_indices,
        write_indices,
        accepted_length,
        draft_token_num=draft_token_num,
        num_heads=num_heads,
        head_dim=head_dim,
        f_a_dim=f_a_dim,
        qkv_stride=qkv_stride,
        conv_stride=conv_stride,
        f_a_stride=f_a_stride,
        beta_stride=beta_stride,
        state_stride=state_stride,
        gate_stride=gate_stride,
        conv_width=conv_width,
        layers_per_group=layers_per_group,
        lower_bound=lower_bound,
        recurrent_layout=recurrent_layout,
    )


@register_kernel(
    "attention",
    "kda_replay_commit",
    name="triton_nvidia_kda_batched_replay_commit_vmajor",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_BF16_SIGNATURES,
    priority=Priority.SPECIALIZED,
    traits={
        "flat_state": frozenset({True}),
        "batched_layers": frozenset({True}),
        "recurrent_layout": frozenset({"v_major"}),
    },
    tags={"nvidia", "flat_kv", "fusion", "speculative", "batched_layers"},
)
def triton_nvidia_kda_batched_replay_commit_vmajor(*args, **kwargs) -> None:
    """Replay every descriptor-table layer against V-major state."""
    kwargs.pop("recurrent_layout", None)
    _nvidia_kda_batched_replay_commit(*args, **kwargs, recurrent_layout="v_major")


@register_kernel(
    "attention",
    "kda_paged_prefill",
    name="triton_nvidia_kda_paged_prefill",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.PERFORMANT,
    tags={"nvidia", "paged_cache"},
)
def triton_nvidia_kda_paged_prefill(**kwargs) -> KdaPrefillResult:
    from tokenspeed_kernel.ops.attention.triton.linear.kda import (
        kda_chunk_prefill,
    )

    # Host-boundary hint is consumed only by the CuteDSL wrapper.
    kwargs.pop("cu_seqlens_cpu", None)
    return _nvidia_kda_prefill(kda_chunk_prefill, **kwargs)


@register_kernel(
    "attention",
    "kda_paged_prefill",
    name="flashkda_nvidia_kda_paged_prefill",
    solution="flashkda",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    tags={"nvidia", "paged_cache"},
)
def flashkda_nvidia_kda_paged_prefill(**kwargs) -> KdaPrefillResult:
    from tokenspeed_kernel.ops.attention.flash_kda import flash_kda_chunk_prefill

    # Host-boundary hint is consumed only by the CuteDSL wrapper.
    kwargs.pop("cu_seqlens_cpu", None)
    return _nvidia_kda_prefill(flash_kda_chunk_prefill, **kwargs)


@register_kernel(
    "attention",
    "kda_paged_prefill",
    name="cutedsl_kda_nvidia_paged_prefill",
    solution="cutedsl_kda",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.SPECIALIZED,
    tags={"nvidia", "paged_cache"},
)
def cutedsl_kda_nvidia_paged_prefill(**kwargs) -> KdaPrefillResult:
    from tokenspeed_kernel.ops.attention.cutedsl_kda import cutedsl_kda_chunk_prefill

    return _nvidia_kda_prefill(cutedsl_kda_chunk_prefill, **kwargs)
