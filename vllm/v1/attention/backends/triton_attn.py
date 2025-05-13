# SPDX-License-Identifier: Apache-2.0
"""Attention layer with PagedAttention and Triton prefix prefill."""
from dataclasses import dataclass
from typing import Any, Optional

import torch

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.attention.ops.triton_unified_attention import unified_attention
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.block_table import BlockTable

logger = init_logger(__name__)

@dataclass
class TritonAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: BlockTable
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: Optional[torch.Tensor]
    prefix_kv_lens: Optional[torch.Tensor]
    suffix_kv_lens: Optional[torch.Tensor]

    # Optional aot scheduling
    scheduler_metadata: Optional[torch.Tensor] = None
    prefix_scheduler_metadata: Optional[torch.Tensor] = None

    # TODO 
    # # for local attention
    # @dataclass
    # class LocalAttentionMetadata:
    #     local_query_start_loc: torch.Tensor
    #     local_seqused_k: torch.Tensor
    #     local_block_table: torch.Tensor
    #     local_max_query_len: int
    #     local_max_seq_len: int
    #     local_scheduler_metadata: Optional[torch.Tensor]

    # local_attn_metadata: Optional[LocalAttentionMetadata] = None


class TritonAttentionMetadataBuilder:

    def __init__(self, runner: "GPUModelRunner", kv_cache_spec: AttentionSpec,
                 block_table: BlockTable):
        model_config = runner.model_config

        self.runner = runner
        self.num_heads_q = model_config.get_num_attention_heads(
            runner.parallel_config)
        self.num_heads_kv = model_config.get_num_kv_heads(
            runner.parallel_config)
        self.headdim = model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec
        self.block_table = block_table

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(self, num_reqs: int, num_actual_tokens: int, max_query_len: int,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata):
        max_seq_len = self.runner.seq_lens_np[:num_reqs].max()
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = self.block_table
        block_table_tensor = block_table.get_device_tensor()[:num_reqs]

        block_table.slot_mapping[:num_actual_tokens].copy_(
            block_table.slot_mapping_cpu[:num_actual_tokens],
            non_blocking=True)
        # Fill unused with -1. Needed for reshape_and_cache in full cuda graph
        # mode.
        block_table.slot_mapping[num_actual_tokens:].fill_(-1)

        slot_mapping = block_table.slot_mapping[:num_actual_tokens]

        def schedule(batch_size, cu_query_lens, max_query_len, seqlens,
                     max_seq_len, causal):
            return None

        use_cascade = False
        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        prefix_scheduler_metadata = None
        scheduler_metadata = schedule(batch_size=num_reqs,
                                      cu_query_lens=query_start_loc,
                                      max_query_len=max_query_len,
                                      seqlens=seq_lens,
                                      max_seq_len=max_seq_len,
                                      causal=True)

        attn_metadata = TritonAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
        )
        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


class TritonAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["TritonAttentionImpl"]:
        return TritonAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return TritonAttentionMetadata 

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionMetadataBuilder"]:
        return TritonAttentionMetadataBuilder


class TritonAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        use_irope: bool = False,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "TritonAttention does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        self.use_irope = use_irope

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        support_head_sizes = TritonAttentionBackend.get_supported_head_sizes()
        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by TritonAttention. "
                f"Supported head sizes are: {support_head_sizes}.")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "TritonAttentionImpl")

        self.fp8_dtype = current_platform.fp8_dtype()

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            # Profiling run.
            return output

        assert attn_metadata.use_cascade is False

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        key_cache, value_cache = kv_cache.unbind(0)
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            attn_metadata.slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

        if self.kv_cache_dtype.startswith("fp8"):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
            num_tokens, num_heads, head_size = query.shape
            assert layer._q_scale == 1.0, \
                "A non 1.0 q_scale is not currently supported."
            if not current_platform.is_rocm():
                # Skip Q quantization on ROCm, since dequantizing back to
                # f32 in the attention kernel is not supported.
                query, _ = ops.scaled_fp8_quant(
                    query.reshape(
                        (num_tokens, num_heads * head_size)).contiguous(),
                    layer._q_scale)
            query = query.reshape((num_tokens, num_heads, head_size))

        use_local_attn = \
            (self.use_irope and attn_metadata.local_attn_metadata is not None)

        if use_local_attn:
            assert attn_metadata.local_attn_metadata is not None
            local_metadata = attn_metadata.local_attn_metadata
            cu_seqlens_q = local_metadata.local_query_start_loc
            seqused_k = local_metadata.local_seqused_k
            max_seqlen_q = local_metadata.local_max_query_len
            max_seqlen_k = local_metadata.local_max_seq_len
            block_table = local_metadata.local_block_table.get_device_tensor()
        else:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table.get_device_tensor()

        descale_shape = (cu_seqlens_q.shape[0] - 1, key.shape[1])

        unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=None,  # Not supported
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
        )

        return output
