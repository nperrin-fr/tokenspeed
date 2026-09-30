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

"""
MHA attention leaf for TokenSpeed scheduling.
Uses fused kernels optimized for SM100 (Blackwell).
Supports sliding window, attention sinks, and FP8 KV cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.attention.mha.flashinfer import (
    trtllm_batch_context_with_kv_cache,
    trtllm_batch_decode_with_kv_cache,
)
from tokenspeed_kernel.ops.attention.tree import tree_attention, tree_decode_attention
from tokenspeed_kernel.ops.kvcache.triton import (
    fused_fp8_set_kv_buffer,
)

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    is_breakable_capture_active,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.workspace import workspace_pool
from tokenspeed.runtime.layers.attention.backends.paged.base import (
    PagedAttentionBackend,
)
from tokenspeed.runtime.layers.attention.backends.paged.mha import trim_kv_to_locs
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kernel_page_sizes import (
    TRTLLM_MHA_PAGE_SIZE,
)
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.layers.common import fp8_cast_contiguous
from tokenspeed.runtime.utils.common import ceil_div
from tokenspeed.runtime.utils.env import envs

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


def canonicalize_stride(tensor: torch.Tensor) -> torch.Tensor:
    """Adjust degenerate strides for a tensor, make it canonical.

    When a dimension has size=1, PyTorch may use the same stride as the next dim.
    This causes TMA desc validation failures in the TRT-LLM MHA kernels.
    See: https://github.com/flashinfer-ai/flashinfer/issues/2232
    """
    sizes = tensor.size()
    strides = tensor.stride()
    ndim = tensor.dim()

    need_fix = any(
        sizes[i] == 1 and strides[i] == strides[i + 1] for i in range(ndim - 1)
    )

    if not need_fix:
        return tensor

    new_strides = [0] * ndim
    new_strides[-1] = 1
    for i in range(ndim - 2, -1, -1):
        new_strides[i] = new_strides[i + 1] * sizes[i + 1]

    return tensor.as_strided(sizes, new_strides)


@dataclass
class TRTLLMMHAMetadata:
    cache_seqlens_int32: torch.Tensor = None
    max_seq_len_q: int = 1
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    page_table: torch.Tensor = None


class TRTLLMMHAAttnBackend(PagedAttentionBackend):
    """The ``trtllm`` MHA leaf: TRT-LLM fused kernels for SM100 (Blackwell)."""

    default_kernel_page_size = TRTLLM_MHA_PAGE_SIZE
    supports_tree_verify = True
    supports_tree_draft = True
    # Both kernel call sites forward layer.sliding_window_size.
    supports_layer_sliding_window: bool = True

    def __init__(self, config: AttnConfig, spec: MHAConfig, *, kernel_page_size: int):
        super().__init__(config, spec, kernel_page_size=kernel_page_size)

        self.tp_q_head_num = max(spec.num_attention_heads // spec.attn_tp_size, 1)
        self.tp_kv_head_num = max(spec.num_kv_heads // spec.attn_tp_size, 1)
        self.kv_cache_dtype = config.kv_cache_dtype

        self._workspace_pool = workspace_pool(config.device)
        self._workspace_nbytes = envs.TOKENSPEED_WORKSPACE_TRTLLM_MHA_MB.get() * (
            1 << 20
        )
        # Warm the shared block to this backend's peak now: graph capture runs
        # the forward with the pool frozen, and under --disable-autotune no
        # earlier forward will have grown the block by then.
        self._workspace_pool.allocate(((self._workspace_nbytes,), torch.uint8))
        self.num_sms = torch.cuda.get_device_properties(
            config.device
        ).multi_processor_count

        # DFLASH draft: the drafter predicts a whole block of spec_num_tokens
        # per decode forward and needs non-causal (block-diffusion) attention.
        # Instead of a non-causal mask, materialize one single-query decode
        # metadata entry per block position (block_decode_expansion), all
        # sharing the SAME block-end seq_len, so each block position attends
        # over the whole block; target verify and plain decode are untouched.
        self.draft_block_decode = bool(config.draft_block_decode)

        # Separate slots for prefill-kernel vs decode-kernel forward paths:
        # forward_extend reads prefill; forward_decode picks by q_len (target
        # verify is DECODE mode but rides the prefill slot's uniform stride).
        self.forward_prefill_metadata: TRTLLMMHAMetadata | None = None
        self.forward_decode_metadata: TRTLLMMHAMetadata | None = None

        # KV seqlens clamped to >= spec_num_tokens for the MTP verify path.
        # Padded decode requests have seq_len=1; with q_len=spec_num_tokens
        # they'd hit an empty causal span and the kernel returns NaN.
        self.spec_cache_seqlens_buf: torch.Tensor | None = None
        # Pure aranges per bs: pool-independent, so a rebind keeps them.
        self._cu_seqlens_by_bs: dict[int, torch.Tensor] = {}
        self._verify_views_by_bs: dict[int, TRTLLMMHAMetadata] = {}

    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        super()._publish_cache_pool(cache_pool)
        self.forward_prefill_metadata = None
        self.forward_decode_metadata = None
        self.spec_cache_seqlens_buf = None
        self._verify_views_by_bs = {}

    def support_kv_cache_prewrite(
        self, forward_mode: ForwardMode | None = None
    ) -> bool:
        # Under a breakable prefill-graph capture the prewrite would bake this
        # forward's write locations into the graph (stale on every replay) --
        # bake the non-prewrite branch instead: the eager attention break
        # writes KV from fresh metadata.
        if is_breakable_capture_active():
            return False
        # Draft-tree lanes keep their K/V out of the paged cache (side buffer).
        return not self.tree_lane_step_active

    # ------------------------------------------------------------------
    # Metadata initialisation
    # ------------------------------------------------------------------

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_with_prefix: bool,
        **kwargs,
    ):
        if not forward_mode.is_extend_or_mixed():
            raise RuntimeError(
                "trtllm decode metadata goes through refresh_decode_metadata; "
                f"init_forward_metadata only serves extend ({forward_mode})"
            )
        assert (
            seq_lens.dtype == torch.int32
        ), f"seq_lens must be int32, got {seq_lens.dtype}"
        cache_seqlens_int32 = seq_lens[:bs]
        cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(seq_lens[:bs], dim=0, dtype=torch.int32), (1, 0)
        )

        # Read the max from the pinned-CPU mirror — avoids a per-iter
        # GPU->CPU sync that would block the host on the previous step's
        # forward and erase prefill/decode overlap. Both branches want
        # max(new tokens per request); for a no-prefix extend that's
        # seq_lens, for a prefix-cached extend it's seq_lens-prefix_lens —
        # extend_seq_lens_cpu holds those new-token counts in either case.
        max_seq_len_q = int(extend_seq_lens_cpu[:bs].max().item())

        if extend_with_prefix and bool(extend_prefix_lens_cpu[:bs].any()):
            extend_lens = seq_lens[:bs] - extend_prefix_lens[:bs]
            cu_seqlens_q = torch.nn.functional.pad(
                torch.cumsum(extend_lens, dim=0, dtype=torch.int32), (1, 0)
            )
        else:
            cu_seqlens_q = cu_seqlens_k

        self.forward_prefill_metadata = TRTLLMMHAMetadata(
            cache_seqlens_int32=cache_seqlens_int32,
            max_seq_len_q=max_seq_len_q,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_table=page_table[:bs],
        )

    # ------------------------------------------------------------------
    # CUDA graph / decode refresh
    # ------------------------------------------------------------------

    def init_cuda_graph_state(self, max_bs: int) -> None:
        super().init_cuda_graph_state(max_bs)
        self.spec_cache_seqlens_buf = torch.zeros(
            (max_bs,), dtype=torch.int32, device=self.device
        )
        self._cu_seqlens_by_bs = {}
        self._verify_views_by_bs = {}

    def _uniform_cu_seqlens(self, bs: int, stride: int) -> torch.Tensor:
        """Cached ``arange(0, bs*stride+1, stride)`` (pointer-stable per bs)."""
        key = bs * 100000 + stride
        cu = self._cu_seqlens_by_bs.get(key)
        if cu is None:
            cu = torch.arange(
                0, bs * stride + 1, stride, dtype=torch.int32, device=self.device
            )
            self._cu_seqlens_by_bs[key] = cu
        return cu

    def _decode_views(self, bs: int) -> TRTLLMMHAMetadata:
        """Single-token decode slot views (block decode: one per block
        position)."""
        metadata = self._decode_views_by_bs.get(bs)
        if metadata is None:
            if self.block_decode_active:
                expanded_bs = bs * self.block_decode_expansion
                metadata = TRTLLMMHAMetadata(
                    cache_seqlens_int32=self.seq_lens_buf[:expanded_bs],
                    max_seq_len_q=1,
                    cu_seqlens_q=self._uniform_cu_seqlens(expanded_bs, 1),
                    page_table=self.page_table_buf[:expanded_bs],
                )
            else:
                metadata = TRTLLMMHAMetadata(
                    cache_seqlens_int32=self.seq_lens_buf[:bs],
                    max_seq_len_q=1,
                    cu_seqlens_q=self._uniform_cu_seqlens(bs, 1),
                    page_table=self.page_table_buf[:bs],
                )
            self._decode_views_by_bs[bs] = metadata
        return metadata

    def _verify_views(self, bs: int) -> TRTLLMMHAMetadata:
        """Target-verify slot views: uniform ``spec_num_tokens`` stride over
        the clamped seqlens buffer."""
        metadata = self._verify_views_by_bs.get(bs)
        if metadata is None:
            spec = self.spec_num_tokens
            metadata = TRTLLMMHAMetadata(
                cache_seqlens_int32=self.spec_cache_seqlens_buf[:bs],
                max_seq_len_q=spec,
                cu_seqlens_q=self._uniform_cu_seqlens(bs, spec),
                page_table=self.page_table_buf[:bs],
            )
            self._verify_views_by_bs[bs] = metadata
        return metadata

    def refresh_decode_metadata(
        self,
        bs: int,
        actual_bs: int,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        *,
        num_extends: int = 0,
        for_graph_replay: bool = False,
    ) -> None:
        if self.block_decode_active:
            # DFLASH draft block: replicate the page table to each request's
            # block positions. Under replay the seq_lens are re-derived
            # in-graph; eager (and the capture seeding) fills them here.
            spec = self.spec_num_tokens
            self.page_table_buf[: bs * spec].view(bs, spec, self.max_num_pages).copy_(
                page_table[:bs, None, :]
            )
            if not for_graph_replay or actual_bs == 0:
                self.fill_block_decode_seq_lens(bs, seq_lens)
            self.forward_decode_metadata = self._decode_views(bs)
            return

        self.seq_lens_buf[:bs].copy_(seq_lens[:bs])
        self.page_table_buf[:bs].copy_(page_table[:bs])
        self.forward_decode_metadata = self._decode_views(bs)
        # Pure verify (and the draft's multi-token step 1) reads the prefill
        # slot. An extend/mixed draft refresh only seeds later decode steps and
        # must preserve the ragged prefill metadata built immediately before it.
        if self.spec_num_tokens > 1 and num_extends == 0:
            torch.clamp_min(
                seq_lens[:bs],
                self.spec_num_tokens,
                out=self.spec_cache_seqlens_buf[:bs],
            )
            self.forward_prefill_metadata = self._verify_views(bs)

    # ------------------------------------------------------------------
    # KV cache helpers
    # ------------------------------------------------------------------

    @property
    def workspace_buffer(self) -> torch.Tensor:
        """Per-use view of the shared scratch block (contract: workspace.py).

        Fetched on every use rather than held: the content (softmax stats plus
        multi-CTA KV scratch) is consumed within each attention op, so sharing
        the block is safe, and a held view would go stale if the block grew.
        """
        (buf,) = self._workspace_pool.allocate(((self._workspace_nbytes,), torch.uint8))
        return buf

    def _get_kv_cache_permuted(self, layer: PagedAttention, token_to_kv_pool):
        """Get KV cache in [num_pages, num_kv_heads, page_size, head_dim] layout."""
        k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
        k_cache = k_cache.view(
            -1, self.kernel_page_size, layer.tp_k_head_num, layer.head_dim
        ).permute(0, 2, 1, 3)
        v_cache = v_cache.view(
            -1, self.kernel_page_size, layer.tp_v_head_num, layer.head_dim
        ).permute(0, 2, 1, 3)

        if layer.tp_k_head_num == 1:
            k_cache = canonicalize_stride(k_cache)
        if layer.tp_v_head_num == 1:
            v_cache = canonicalize_stride(v_cache)

        return k_cache, v_cache

    def _compute_scales(self, layer: PagedAttention):
        """Compute bmm1/bmm2 scales for the fused kernel."""
        q_scale = 1.0
        k_scale = (
            layer.k_scale_float
            if getattr(layer, "k_scale_float", None) is not None
            else 1.0
        )
        bmm1_scale = q_scale * k_scale * layer.scaling
        bmm2_scale = 1.0
        return bmm1_scale, bmm2_scale

    def _should_use_fused_fp8_path(self, save_kv_cache: bool, k) -> bool:
        return (
            save_kv_cache
            and k is not None
            and self.kv_cache_dtype == torch.float8_e4m3fn
        )

    def _save_kv_and_prepare_q(
        self, q, k, v, layer, out_cache_loc, token_to_kv_pool, save_kv_cache
    ):
        k, v = trim_kv_to_locs(out_cache_loc, k, v)
        if self._should_use_fused_fp8_path(save_kv_cache, k):
            k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
            fused_fp8_set_kv_buffer(
                k=k.view(-1, layer.tp_k_head_num, layer.head_dim),
                v=v.view(-1, layer.tp_k_head_num, layer.head_dim),
                k_cache=k_cache,
                v_cache=v_cache,
                cache_loc=out_cache_loc,
                k_scale=layer.k_scale,
                v_scale=layer.v_scale,
                page_size=self.kernel_page_size,
            )
        elif save_kv_cache and k is not None:
            token_to_kv_pool.set_kv_buffer(
                layer, out_cache_loc, k, v, layer.k_scale, layer.v_scale
            )

        if self.kv_cache_dtype == torch.float8_e4m3fn:
            q = fp8_cast_contiguous(q)
        else:
            q = q.contiguous()

        return q.view(-1, layer.tp_q_head_num, layer.head_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if self.tree_lane_step_active:
            return self._forward_tree_lanes(q, k, v, layer, token_to_kv_pool, bs)
        if self.block_decode_active:
            # DFLASH draft block: metadata is expanded to bs*spec_num_tokens
            # single-query entries, so use the decode slot directly. Inferring
            # q_len_per_req from q.shape[0]//bs would be spec_num_tokens and
            # wrongly pick the verify slot.
            metadata = self.forward_decode_metadata
        else:
            # Multi-token decode (q_len > 1) reads the verify slot's
            # uniform-stride metadata; plain decode reads the single-token slot.
            q_len_per_req = q.shape[0] // bs if bs > 0 else 1
            metadata = (
                self.forward_prefill_metadata
                if q_len_per_req > 1
                else self.forward_decode_metadata
            )

        q = self._save_kv_and_prepare_q(
            q, k, v, layer, out_cache_loc, token_to_kv_pool, save_kv_cache
        )
        if self.tree_verify is not None and metadata.max_seq_len_q > 1:
            if layer.sliding_window_size >= 0 or kwargs.get("sinks") is not None:
                raise NotImplementedError(
                    "draft-tree verify has no sliding-window or attention-sink path yet"
                )
            return self._forward_tree_verify(q, layer, token_to_kv_pool, metadata, bs)
        k_cache, v_cache = self._get_kv_cache_permuted(layer, token_to_kv_pool)
        bmm1_scale, bmm2_scale = self._compute_scales(layer)

        attention_sink = kwargs.get("sinks", None)
        if attention_sink is not None:
            attention_sink = attention_sink.float()

        o = trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=self.workspace_buffer,
            block_tables=metadata.page_table,
            seq_lens=metadata.cache_seqlens_int32,
            max_seq_len=self.max_context_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            window_left=layer.sliding_window_size,
            sinks=attention_sink,
            out_dtype=self.dtype,
            q_len_per_req=metadata.max_seq_len_q,
        )
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def bind_tree_verify(self, inputs) -> None:
        if self.kv_cache_dtype != self.dtype:
            raise NotImplementedError(
                f"draft-tree verify reads the window K/V back unquantized; "
                f"kv_cache_dtype {self.kv_cache_dtype} is not supported yet"
            )
        super().bind_tree_verify(inputs)

    def bind_tree_draft(self, inputs) -> None:
        if self.kv_cache_dtype != self.dtype:
            raise NotImplementedError(
                f"draft-tree lanes keep bf16 side K/V; kv_cache_dtype {self.kv_cache_dtype} "
                "is not supported yet"
            )
        super().bind_tree_draft(inputs)

    def _forward_tree_verify(
        self,
        q: torch.Tensor,
        layer: PagedAttention,
        token_to_kv_pool,
        metadata: TRTLLMMHAMetadata,
        bs: int,
    ) -> torch.Tensor:
        """Verify a draft tree: every window row sees the prefix and its masked ancestors."""
        kv_heads, dim = layer.tp_k_head_num, layer.head_dim
        k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
        bmm1_scale, _ = self._compute_scales(layer)
        out = tree_decode_attention(
            q,
            k_cache.view(-1, kv_heads, dim),
            v_cache.view(-1, kv_heads, dim),
            metadata.page_table,
            metadata.cache_seqlens_int32,
            self.tree_verify.mask[: bs * self.tree_verify.num_nodes],
            num_nodes=self.tree_verify.num_nodes,
            page_size=self.kernel_page_size,
            sm_scale=bmm1_scale,
            num_splits=self._tree_verify_splits(bs, kv_heads),
        )
        return out.view(-1, layer.tp_q_head_num * dim)

    def _tree_verify_splits(self, bs: int, kv_heads: int) -> int:
        """About two (request, KV head, split) programs per SM, at most 16 splits."""
        return max(1, min(16, ceil_div(2 * self.num_sms, bs * kv_heads)))

    def _forward_tree_lanes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        token_to_kv_pool,
        bs: int,
    ) -> torch.Tensor:
        """One drafting step of a draft tree: K lane rows per request over the
        accepted prefix (paged, non-causal) and their ancestors (side buffer)."""
        lanes = self.tree_draft
        topk, slots = lanes.topk, lanes.num_slots
        heads, kv_heads, dim = layer.tp_q_head_num, layer.tp_k_head_num, layer.head_dim
        q = q.contiguous().view(-1, heads, dim)
        side_k, side_v = lanes.side_k[layer.layer_id], lanes.side_v[layer.layer_id]
        rows = lanes.slot_rows[lanes.step - 1][: bs * topk]
        side_k.index_copy_(0, rows, k.view(-1, kv_heads, dim))
        side_v.index_copy_(0, rows, v.view(-1, kv_heads, dim))

        k_cache, v_cache = self._get_kv_cache_permuted(layer, token_to_kv_pool)
        bmm1_scale, bmm2_scale = self._compute_scales(layer)
        prefix_out, prefix_lse = trtllm_batch_context_with_kv_cache(
            query=q,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=self.workspace_buffer,
            block_tables=self.page_table_buf[:bs],
            seq_lens=lanes.prefix_lens[:bs],
            max_q_len=topk,
            max_kv_len=self.max_context_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            batch_size=bs,
            cum_seq_lens_q=lanes.cu_query_lens[: bs + 1],
            cum_seq_lens_kv=lanes.cu_prefix_lens[: bs + 1],
            out_dtype=self.dtype,
            causal=False,
            return_lse=True,
        )
        out, _ = tree_attention(
            q,
            side_k[: bs * slots],
            side_v[: bs * slots],
            lanes.lane_mask[: bs * topk],
            rows_per_req=topk,
            slots_per_req=slots,
            sm_scale=bmm1_scale,
            lse_base2=True,
            prefix=(prefix_out, prefix_lse),
            out=prefix_out,
        )
        return out.view(-1, heads * dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        metadata = self.forward_prefill_metadata
        q = self._save_kv_and_prepare_q(
            q, k, v, layer, out_cache_loc, token_to_kv_pool, save_kv_cache
        )
        k_cache, v_cache = self._get_kv_cache_permuted(layer, token_to_kv_pool)
        bmm1_scale, bmm2_scale = self._compute_scales(layer)

        attention_sink = kwargs.get("sinks", None)
        if attention_sink is not None:
            attention_sink = attention_sink.float()

        o = trtllm_batch_context_with_kv_cache(
            query=q,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=self.workspace_buffer,
            block_tables=metadata.page_table,
            seq_lens=metadata.cache_seqlens_int32,
            max_q_len=metadata.max_seq_len_q,
            max_kv_len=self.max_context_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            batch_size=metadata.cu_seqlens_q.shape[0] - 1,
            cum_seq_lens_q=metadata.cu_seqlens_q,
            cum_seq_lens_kv=metadata.cu_seqlens_k,
            window_left=layer.sliding_window_size,
            sinks=attention_sink,
            out_dtype=self.dtype,
        )
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)


register_backend("trtllm", {AttentionArch.MHA}, TRTLLMMHAAttnBackend)
