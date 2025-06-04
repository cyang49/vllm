# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union

import torch
from torch import nn

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce)
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.mamba.mamba2_metadata import Mamba2Metadata
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_state_update)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import (
    LoaderFunction, composed_weight_loader, sharded_weight_loader)
from vllm.model_executor.models.mamba_cache import MambaCacheParams
from vllm.model_executor.utils import set_weight_attrs

# debugging

# Added by the IBM Team, 2024


# assume the input tensor is 2D and has shape (seqlen, hidden_dim)
def varlen_batch_to_padded_blocks(input_tensor,
                                  block_cu_seqlens,
                                  block_size,
                                  pad_value=0.0):
    nblocks = len(block_cu_seqlens) - 1
    hidden_dim = input_tensor.size(-1)
    output_tensor = torch.full((nblocks, block_size, hidden_dim),
                               fill_value=pad_value,
                               dtype=input_tensor.dtype,
                               device=input_tensor.device)
    for i in range(nblocks):
        bstart, bend = block_cu_seqlens[i], block_cu_seqlens[i + 1]
        output_tensor[i, :(bend - bstart)] = input_tensor[bstart:bend]
    return output_tensor


# Adapted from transformers.models.mamba2.modeling_mamba2.MambaRMSNormGated
@CustomOp.register("mixer2_gated_rms_norm")
class Mixer2RMSNormGated(CustomOp):

    def __init__(self,
                 full_hidden_size: int,
                 full_n_groups: int,
                 use_rms_norm: bool = True,
                 eps: float = 1e-6):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.full_hidden_size = full_hidden_size
        self.group_size = full_hidden_size // full_n_groups
        self.per_rank_hidden_size = full_hidden_size // self.tp_size
        self.n_groups = full_hidden_size // self.group_size

        self.variance_epsilon = eps
        self.use_rms_norm = use_rms_norm
        if self.use_rms_norm:
            # Register norm weight only if we're actually applying RMSNorm
            self.weight = nn.Parameter(torch.ones(self.per_rank_hidden_size))
            set_weight_attrs(self.weight,
                             {"weight_loader": sharded_weight_loader(0)})
        else:
            # Avoid checkpoint mismatch by skipping unused parameter
            self.register_parameter("weight", None)
        assert (self.full_hidden_size % self.tp_size == 0
                ), "Tensor parallel world size must divide hidden size."

    def forward_native(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ):
        # Three tensor-parallel cases:
        #   1. n_groups is 1
        #      In this case we parallelize along the reduction dim.
        #      Each rank computes a local sum of squares followed by AllReduce
        #   2. tp_size divides n_groups
        #      Each rank only reduces within its local group(s).
        #      No collective ops necessary.
        #   3. The general case can be pretty complicated so we AllGather
        #      the input and then redundantly compute the RMSNorm.
        input_dtype = x.dtype
        x = x * nn.functional.silu(gate.to(torch.float32))
        if not self.use_rms_norm:
            return x.to(input_dtype)

        if self.n_groups == 1:
            if self.tp_size > 1:
                # Compute local sum and then reduce to obtain global sum
                local_sums = x.pow(2).sum(dim=-1, keepdim=True)
                global_sums = tensor_model_parallel_all_reduce(local_sums)
                # Calculate the variance
                count = self.tp_size * x.shape[-1]
                variance = global_sums / count

            else:
                variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.variance_epsilon)
        else:
            redundant_tp: bool = self.n_groups % self.tp_size != 0
            if redundant_tp:
                # To handle the general case, redundantly apply the variance
                x = tensor_model_parallel_all_gather(x, -1)

            *prefix_dims, hidden_dim = x.shape
            group_count = hidden_dim // self.group_size
            x_grouped = x.view(*prefix_dims, group_count, self.group_size)
            variance = x_grouped.pow(2).mean(-1, keepdim=True)
            x_grouped = x_grouped * torch.rsqrt(variance +
                                                self.variance_epsilon)
            x = x_grouped.view(*prefix_dims, hidden_dim)

            if redundant_tp:
                start = self.per_rank_hidden_size * self.tp_rank
                end = start + self.per_rank_hidden_size
                x = x[..., start:end]

        return self.weight * x.to(input_dtype)

    def forward_cuda(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        input_dtype = x.dtype
        if not self.use_rms_norm:
            # Keep gate in float32 for numerical stability during silu
            return x * nn.functional.silu(gate.to(
                torch.float32)).to(input_dtype)

        if self.tp_size > 1 or self.n_groups != 1:
            return self.forward_native(x, gate)

        from vllm import _custom_ops as ops

        # cast x and gate to float32 before silu
        out = torch.empty_like(x)
        y = x * nn.functional.silu(gate.to(torch.float32))
        ops.rms_norm(
            out,
            y.to(x.dtype),
            self.weight.data,
            self.variance_epsilon,
        )
        return out


def extra_groups_for_head_shards(ngroups: int, tp_size: int):
    """Compute the increase in group numbers to account for
    replication in order to accompany the head shards."""

    # in the case ngoups % tp_size == 0, this will be zero
    if ngroups % tp_size == 0:
        return 0

    # for n_groups == 1, this is exactly tp_size - n_groups
    return tp_size - ngroups


def mamba_v2_sharded_weight_loader(
    shard_spec: list[tuple[int, int, float]],
    tp_size: int,
    tp_rank: int,
) -> LoaderFunction:
    """Create a weight loader for mamba v2. This ensures that the projections 
    are correctly sharded so that they can be split into x, B, C. It also 
    ensures that all the groups corresponding to a head shard is placed 
    together with it.
    """

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:

        # - track boundary of (sharded) param, and loaded_weight, respectively
        boundary, loaded_boundary = 0, 0

        # - iterate over the shard specs
        for full_dim, extra, duplicate_groups in shard_spec:
            # - full dim is the model dim (before TP).
            # - extra > 0, means there is expected overall increase
            #   of dimensions. This is so because of replication.
            # - ratio is used map the tp_rank to the actual shard
            #   rank. This is useful when there is replication of
            #   groups to accompany head shards.

            # - size of the loaded shard
            shard_size = full_dim // tp_size

            # - compute the rank into the loaded shard.
            # - if there is replication, different TP shards will
            #   take from the same rank.
            # NOTE: currently we only support duplication
            # in the case where num_groups == 1
            rank = 0 if duplicate_groups else tp_rank

            # - leftmost boundary index into loaded weight.
            loaded_skip = rank * shard_size
            loaded_start_idx = loaded_boundary + loaded_skip

            # - take these many dims from the loaded weight.
            take = min(shard_size, full_dim - extra - loaded_skip)

            # - always shard on dim 0
            # - the ignore is for a mundane mypy error as it does not
            #   seem to handle slices well.
            # https://github.com/python/mypy/issues/2410
            param.data[
                boundary:(boundary + take),
                ...  # type: ignore[misc]
            ] = loaded_weight[loaded_start_idx:(loaded_start_idx +
                                                take)  # type: ignore[misc]
                              ]  # type: ignore[misc]

            # move indexing boundaries
            boundary += shard_size
            loaded_boundary += full_dim - extra

    return loader


# Adapted from transformers.models.mamba.modeling_mamba.MambaMixer
@CustomOp.register("mamba_mixer2")
class MambaMixer2(CustomOp):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute
    the `contextualized_states`. A, D are input independent
    (see Mamba paper [1] Section 3.5.2 "Interpretation of A"
    for why A isn't selective) ∆, B, C are input-dependent
    (this is a key difference between Mamba and the linear time
    invariant S4, and is why Mamba is called
    **selective** state spaces)
    """

    def __init__(
        self,
        hidden_size: int,
        ssm_state_size: int,
        conv_kernel_size: int,
        intermediate_size: int,
        use_conv_bias: bool,
        use_bias: bool,
        n_groups: int = 1,
        num_heads: int = 128,
        head_dim: int = 64,
        rms_norm_eps: float = 1e-5,
        activation: str = "silu",
        use_rms_norm: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        # For TP, the sharding plan is as follows:
        # - for the conv modules, since
        #   conv_dim = intermediate_size * 2 * n_groups * ssm_state_size,
        #   we shard intermediate_size and n_groups
        # - since intermediate_size = n_heads * head_dim, sharding on
        #   intermediate_size is achieved by sharding on n_heads.
        # - IF, world_size divides groups, then sharding
        #   (n_groups / world_size, n_heads / world_size)
        #   also maintains the invariant n_heads % n_groups == 0
        # - HOWEVER IF, world_size DOES NOT divide groups, then we need
        #   to allocate extra space in the shard, such that groups
        #   may be replicated to follow the head shard.
        # - NOTE: currently for the world size DOES NOT divide groups
        #   case, we only support the case when n_groups == 1
        self.tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        assert (num_heads % self.tp_size == 0
                ), "Tensor parallel world size must divide num heads."

        assert (n_groups % self.tp_size) == 0 or n_groups == 1, (
            "If tensor parallel world size does not divide num_heads, "
            "then num_groups must equal 1.")

        assert (
            self.tp_size == 1 or quant_config is None
        ), "Tensor parallel currently not supported for quantized models."

        self.ssm_state_size = ssm_state_size
        self.activation = activation

        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_heads = num_heads

        self.n_groups = n_groups
        if n_groups % self.tp_size != 0:
            # - for TP we shard conv_dim by sharding on n_groups,
            # - but if n_groups cannot divide tp_size, we need to
            #   extend some extra groups
            self.n_groups = n_groups + extra_groups_for_head_shards(
                n_groups, self.tp_size)

        self.conv_dim = intermediate_size + 2 * self.n_groups * ssm_state_size
        self.conv1d = ColumnParallelLinear(
            input_size=conv_kernel_size,
            output_size=self.conv_dim,
            bias=use_conv_bias,
            quant_config=None,
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.in_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size + self.conv_dim + self.num_heads,
            bias=use_bias,
            quant_config=quant_config,
        )

        # - because in_proj is a concatenation of 3 weights, we
        #   need to interleave them before sharding
        # - use the custom weight loader mamba_v2_sharded_weight_loader
        #   for conv1d.bias, covn1d.weight and in_proj.weight
        # - need to set these settings, to assign the groups to the head shards
        group_shard_settings = (
            self.n_groups * self.ssm_state_size,  # expected model size
            (self.n_groups - n_groups) *
            self.ssm_state_size,  # extra dims assigned
            n_groups == 1,  # if there was only one group
        )
        intermediate_settings = (intermediate_size, 0, False)
        head_setings = (self.num_heads, 0, False)

        # - the weight already has a "weight_loader" attribute
        #   which set_weight_attrs will raise if we do not
        #   delete before trying to override it
        # - ditto for the otther two weights below
        delattr(self.conv1d.bias, "weight_loader")
        set_weight_attrs(
            self.conv1d.bias,
            {
                "weight_loader":
                mamba_v2_sharded_weight_loader(
                    [
                        intermediate_settings,
                        group_shard_settings,
                        group_shard_settings,
                    ],
                    self.tp_size,
                    tp_rank,
                )
            },
        )

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader":
                mamba_v2_sharded_weight_loader(
                    [
                        intermediate_settings,
                        group_shard_settings,
                        group_shard_settings,
                    ],
                    self.tp_size,
                    tp_rank,
                )
            },
        )

        if quant_config is None:
            # - quant layers do not have a weight loader
            delattr(self.in_proj.weight, "weight_loader")
            set_weight_attrs(
                self.in_proj.weight,
                {
                    "weight_loader":
                    mamba_v2_sharded_weight_loader(
                        [
                            intermediate_settings,  # for gate
                            intermediate_settings,
                            group_shard_settings,
                            group_shard_settings,
                            head_setings,  # for dt
                        ],
                        self.tp_size,
                        tp_rank,
                    )
                },
            )

        # - these are TPed by heads to reduce the size of the
        #   temporal shape
        self.A = nn.Parameter(
            torch.empty(
                divide(num_heads, self.tp_size),
                dtype=torch.float32,
            ))
        self.D = nn.Parameter(torch.ones(num_heads // self.tp_size))
        self.dt_bias = nn.Parameter(torch.ones(num_heads // self.tp_size))
        self.use_rms_norm = use_rms_norm

        set_weight_attrs(self.D, {"weight_loader": sharded_weight_loader(0)})
        a_weight_loader = composed_weight_loader(
            sharded_weight_loader(0), lambda x: -torch.exp(x.float()))
        set_weight_attrs(self.A, {"weight_loader": a_weight_loader})
        set_weight_attrs(self.dt_bias,
                         {"weight_loader": sharded_weight_loader(0)})

        self.out_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=use_bias,
            input_is_parallel=True,
            quant_config=quant_config,
        )

        self.norm = Mixer2RMSNormGated(intermediate_size,
                                       n_groups,
                                       self.use_rms_norm,
                                       eps=rms_norm_eps)

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ):
        pass

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        mamba_cache_params: MambaCacheParams,
        mamba2_metadata: Mamba2Metadata,
        mup_vector: Optional[torch.Tensor] = None,
    ):
        # mamba2_metadata contains metadata necessary for the mamba2 triton
        # kernels to operate in continuous batching and in chunked prefill
        # modes; they are computed at top-level model forward since they
        # stay the same and reused for all mamba layers in the same iteration
        attn_metadata: AttentionMetadata = get_forward_context().attn_metadata

        num_prefills = attn_metadata.num_prefills  # request count
        num_decodes = attn_metadata.num_decode_tokens  # token count (=request)
        num_prefill_tokens = attn_metadata.num_prefill_tokens  # token count
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0

        groups_time_state_size = self.n_groups * self.ssm_state_size

        # 1. Gated MLP's linear projection
        projected_states, _ = self.in_proj(hidden_states)

        if mup_vector is not None:
            projected_states = projected_states * mup_vector

        gate, hidden_states_B_C, dt = torch.split(
            projected_states,
            [
                self.intermediate_size // self.tp_size,
                self.conv_dim // self.tp_size,
                self.num_heads // self.tp_size,
            ],
            dim=-1,
        )

        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0),
                                               self.conv1d.weight.size(2))

        # Separate prefill and decode by splitting varlen input
        # Split along token dimension
        hidden_states_B_C_p, hidden_states_B_C_d = torch.split(
            hidden_states_B_C,
            [num_prefill_tokens, num_decodes],
            dim=0,
        )
        dt_p, dt_d = torch.split(
            dt,
            [num_prefill_tokens, num_decodes],
            dim=0,
        )
        # Split along batch dimension
        state_indices_tensor_p, state_indices_tensor_d = torch.split(
            mamba_cache_params.state_indices_tensor,
            [num_prefills, num_decodes],
            dim=0,
        )
        query_start_loc_p = (attn_metadata.query_start_loc[:num_prefills + 1]
                             if has_prefill else None)

        # - get hidden_states, B and C after depthwise convolution.
        split_hidden_states_B_C_fn = lambda hidden_states_B_C: torch.split(
            hidden_states_B_C,
            [
                self.intermediate_size // self.tp_size,
                groups_time_state_size // self.tp_size,
                groups_time_state_size // self.tp_size,
            ],
            dim=-1,
        )

        ssd_output_list = []

        # Process prefill requests
        if has_prefill:
            # 2. Convolution sequence transformation
            # - "cache_indices" updates the conv_state cache in positions
            #   pointed to by "mamba_cache_params.state_indices_tensor"
            hidden_states_B_C_p = causal_conv1d_fn(
                hidden_states_B_C_p.transpose(0, 1),
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=mamba_cache_params.conv_state,
                has_initial_state=mamba2_metadata.has_initial_states,
                cache_indices=state_indices_tensor_p,
                query_start_loc=query_start_loc_p).transpose(
                    0, 1)[:num_prefill_tokens]

            # TODO: Why is this needed?
            hidden_states_B_C_p = hidden_states_B_C_p.contiguous()
            hidden_states_p, B_p, C_p = split_hidden_states_B_C_fn(
                hidden_states_B_C_p)

            # 3. State Space Model sequence transformation
            initial_states = None
            if (mamba2_metadata.has_initial_states is not None
                    and mamba2_metadata.prep_initial_states):
                # making a copy of the states
                initial_states = torch.where(
                    mamba2_metadata.has_initial_states[:, None, None, None],
                    mamba_cache_params.ssm_state[state_indices_tensor_p], 0)

            # Padded blocks implementation
            # reshaping 2d (seqlen, hdim) to 3d (nblocks, block_size, hdim)
            nblocks = len(mamba2_metadata.block_cu_seqlens) - 1
            block_size = mamba2_metadata.chunk_size
            batch = len(mamba2_metadata.req_nblocks)
            assert batch == num_prefills
            x_x, dt_x, B_x, C_x = [
                varlen_batch_to_padded_blocks(
                    x,
                    mamba2_metadata.block_cu_seqlens,
                    block_size=block_size,
                ) for x in (hidden_states_p, dt_p, B_p, C_p)
            ]
            x_x = x_x.view(nblocks, block_size, self.num_heads // self.tp_size,
                           self.head_dim)
            B_x, C_x = [
                x.view(nblocks, block_size, self.n_groups // self.tp_size, -1)
                for x in (B_x, C_x)
            ]
            # Repeat heads of B and C
            B_x = B_x.repeat_interleave(self.num_heads // self.n_groups,
                                        dim=2,
                                        output_size=self.num_heads)
            C_x = C_x.repeat_interleave(self.num_heads // self.n_groups,
                                        dim=2,
                                        output_size=self.num_heads)

            # 1. block_cumsum_fwd
            # dt_x: (nblocks, block_size, nheads)
            # A: (nheads)
            dt_x = nn.functional.softplus(dt_x + self.dt_bias)
            dt_x = torch.clamp(dt_x, 0.0, float("inf"))
            dA = self.A.to(dt_x.dtype) * dt_x
            # need to deal with out of bound values in dA
            for i, n in enumerate(mamba2_metadata.block_ntokens):
                if n < block_size:
                    dA[i, n:] = 0.0
            dA_cumsum = torch.cumsum(dA, dim=1)

            # 2. block_state_fwd
            # B_x: (nblocks, block_size, nheads, dstate)
            # x_x: (nblocks, block_size, nheads, headdim)
            # dt_x: (nblocks, block_size, nheads)
            # dA_cumsum: (nblocks, block_size, nheads)
            # x_x = x_x.to(torch.float32)
            decay_states = (torch.exp(dA_cumsum[:, -1:, :] - dA_cumsum) * dt_x
                            )  # (nblocks, block_size, nheads)

            B_decay = (B_x * decay_states[..., None]
                       )  # (nblocks, block_size, nheads, dstate)
            block_states = torch.einsum(
                "nkhd,nkhs->nhds", x_x,
                B_decay)  # (nblocks, nheads, headdim, dstate)

            # 3. state_passing_fwd
            # block_states: (nblocks, nheads, headdim, dstate)
            # dA_cumsum: (nblocks, block_size, nheads)
            # initial_states: (num_prefills, nheads, headdim, dstate)
            # Sequential implementation of state passing
            block_start = 0  # block index
            prev_states = []
            final_states = []
            for i in range(num_prefills):
                block_end = block_start + mamba2_metadata.req_nblocks[i]
                prev_state = (initial_states[i] if initial_states is not None
                              else torch.zeros_like(block_states[0]))
                prev_states.append(prev_state)

                for j in range(block_start, block_end):
                    block_decay = torch.exp(dA_cumsum[j, -1, :])  # (nheads,)
                    block_states[j] += block_decay * prev_state
                    if j < block_end - 1:
                        prev_states.append(block_states[j])
                    prev_state = block_states[j]

                final_states.append(prev_state)
                block_start = block_end
            prev_states = torch.stack(prev_states)
            final_states = torch.stack(final_states)  # --> FINAL

            # 4. block_bmm_fwd (for diagonal blocks)
            # B_x: (nblocks, block_size, nheads, dstate)
            # C_x: (nblocks, block_size, nheads, dstate)
            # Using c and b to distinguish block_size dimension
            # in C and B accordingly
            # CB: (nblocks, block_size, block_size, nheads)
            CB = torch.einsum("nchs,nbhs->ncbh", C_x, B_x)

            # 5. block_scan_fwd
            # Compute both diagonal output and off diagonal output
            # x_x: (nblocks, block_size, nheads, headdim)
            # dA_cumsum: (nblocks, block_size, nheads)
            # refer to https://tinyurl.com/yc2n32u9
            dt_segment_sum = dA_cumsum[:, :, None] - dA_cumsum[:, None, :]
            # (nblocks, block_size, block_size, nheads)
            decay = torch.exp(dt_segment_sum)
            scores_decay = CB * decay  # elementwise
            causal_mask = torch.tril(torch.ones(block_size,
                                                block_size,
                                                device=x_x.device,
                                                dtype=bool),
                                     diagonal=0)
            scores_decay = scores_decay.masked_fill(
                ~causal_mask[None, ..., None], 0)
            # Diagonal block contributions to the output
            out = torch.einsum('nlkh,nkh,nkhd->nlhd', scores_decay, dt_x,
                               x_x)  # (nblocks, block_size, nheads, headdim)

            # C_x: (nblocks, block_size, nheads, dstate)
            # dA_cumsum: (nblocks, block_size, nheads)
            # prev_states: (nblocks, nheads, headdim, dstate)
            state_decay_out = torch.exp(
                dA_cumsum[..., None])  # (nblocks, block_size, nheads, 1)
            # Off-diagonal block contributions to the output
            out_prev = torch.einsum(
                'nlhs,nhds->nlhd', C_x, prev_states
            ) * state_decay_out  # (nblocks, block_size, nheads, headdim)
            # Sum up contributions
            out = out + out_prev
            if self.D is not None:
                D = self.D
                if D.dim() == 1:
                    D = D[..., None]
                out = out + x_x * D
            # Remove padding
            out_blocks = [
                x[:ntokens]
                for (x, ntokens) in zip(out, mamba2_metadata.block_ntokens)
            ]
            scan_output = torch.vstack(out_blocks)  # --> FINAL

            # scan_output, varlen_states = mamba_chunk_scan_combined_varlen(
            #     hidden_states_p.view(num_prefill_tokens,
            #                          self.num_heads // self.tp_size,
            #                          self.head_dim),
            #     dt_p,
            #     self.A,
            #     B_p.view(num_prefill_tokens, self.n_groups // self.tp_size,
            #              -1),
            #     C_p.view(num_prefill_tokens, self.n_groups // self.tp_size,
            #              -1),
            #     chunk_size=mamba2_metadata.chunk_size,
            #     D=self.D,
            #     z=None,
            #     dt_bias=self.dt_bias,
            #     seq_idx=mamba2_metadata.seq_idx.squeeze(0),
            #     chunk_indices=mamba2_metadata.chunk_indices,
            #     chunk_offsets=mamba2_metadata.chunk_offsets,
            #     cu_seqlens=attn_metadata.query_start_loc[:num_prefills + 1],
            #     initial_states=initial_states,
            #     dt_softplus=True,
            #     dt_limit=(0.0, float("inf")),
            # )

            # update ssm states
            # - varlen state is a (num_prefills, nheads, headdim, dstate) tensor
            # mamba_cache_params.ssm_state[
            #     state_indices_tensor_p] = varlen_states
            mamba_cache_params.ssm_state[state_indices_tensor_p] = final_states

            # - reshape
            ssd_output_list.append(scan_output.view(num_prefill_tokens, -1))

        # Process decode requests
        if has_decode:
            # 2. Convolution sequence transformation
            hidden_states_B_C_d = causal_conv1d_update(
                hidden_states_B_C_d,
                mamba_cache_params.conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=state_indices_tensor_d)

            hidden_states_d, B_d, C_d = split_hidden_states_B_C_fn(
                hidden_states_B_C_d)

            # 3. State Space Model sequence transformation
            n_groups = self.n_groups // self.tp_size
            A_d = self.A[:, None, ...][:, :, None].expand(
                -1, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            dt_d = dt_d[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D_d = self.D[:, None, ...].expand(-1, self.head_dim)
            B_d = B_d.view(-1, n_groups, B_d.shape[1] // n_groups)
            C_d = C_d.view(-1, n_groups, C_d.shape[1] // n_groups)
            hidden_states_d = hidden_states_d.view(
                -1, self.num_heads // self.tp_size, self.head_dim)

            # - the hidden is reshaped into (bs, num_heads, head_dim)
            # - mamba_cache_params.ssm_state's slots will be selected
            #   using state_indices_tensor_d

            hidden_states_d = selective_state_update(
                mamba_cache_params.ssm_state,
                hidden_states_d,
                dt_d,
                A_d,
                B_d,
                C_d,
                D_d,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
                state_batch_indices=state_indices_tensor_d,
            )
            ssd_output_list.append(
                hidden_states_d.view(-1, (self.num_heads // self.tp_size) *
                                     self.head_dim))

        # Merge prefill and decode outputs before passing to gated MLP
        hidden_states = torch.vstack(ssd_output_list)

        # 4. gated MLP
        # GatedRMSNorm internally applying SiLU to the gate
        # SiLU is applied internally before normalization, unlike standard
        # norm usage
        hidden_states = self.norm(hidden_states, gate)

        # 5. Final linear projection
        out, _ = self.out_proj(hidden_states)
        return out
