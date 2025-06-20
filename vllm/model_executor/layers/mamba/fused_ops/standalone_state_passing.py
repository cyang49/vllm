# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E501,SIM102,SIM113

import torch

from vllm.triton_utils import tl, triton

# try:
#     from triton.language.extra.cuda.libdevice import fast_expf
# except ImportError:
#     from triton.language.math import fast_expf

# Notations for readability
#   - b: batch
#   - h: nheads
#   - g: ngroups
#   - n: nblocks
#   - t: seqlen
#   - k: block_size
#   - d: headdim
#   - s: dstate


# from .utils import generate_autotune_combinations
# @triton.autotune(
#     configs=generate_autotune_combinations(
#         spec={'BLOCK_SIZE_D': [16, 32, 64, 128],
#               'BLOCK_SIZE_S': [16, 32, 64],
#               'num_warps': [2, 4],
#               'num_stages': [3, 4, 5],
#              },
#         ),
#     key=[],
# )
@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_D': 16,
            'BLOCK_SIZE_S': 64,
        },
                      num_warps=4,
                      num_stages=5),
    ],
    key=[],
)
@triton.jit
def state_passing_kernel(
    # Inputs
    dA_cumsum_ptr,
    block_states_ptr,
    init_states_ptr,
    block_cu_seqlens_ptr,
    block_packed_cu_seqlens_ptr,
    block_req_idx_ptr,
    req_cu_nblocks_ptr,
    # Outputs
    prev_states_ptr,
    final_states_ptr,
    # Matrix dimensions
    nblocks,
    headdim: tl.constexpr,
    dstate: tl.constexpr,
    # Strides
    stride_dA_cumsum_h,
    stride_dA_cumsum_t: tl.constexpr,
    stride_block_states_n: tl.constexpr,
    stride_block_states_h: tl.constexpr,
    stride_block_states_d: tl.constexpr,
    stride_block_states_s: tl.constexpr,
    stride_init_states_b: tl.constexpr,
    stride_init_states_h: tl.constexpr,
    stride_init_states_d: tl.constexpr,
    stride_init_states_s: tl.constexpr,
    stride_block_cu_seqlens_n: tl.constexpr,
    stride_block_packed_cu_seqlens_n: tl.constexpr,
    stride_block_req_idx_n: tl.constexpr,
    stride_req_cu_nblocks_b: tl.constexpr,
    stride_prev_states_n: tl.constexpr,
    stride_prev_states_h: tl.constexpr,
    stride_prev_states_d: tl.constexpr,
    stride_prev_states_s: tl.constexpr,
    stride_final_states_b: tl.constexpr,
    stride_final_states_h: tl.constexpr,
    stride_final_states_d: tl.constexpr,
    stride_final_states_s: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    IS_STORE_PREV_STATES: tl.constexpr,
    ALIGN_BLOCKS: tl.constexpr,
):
    pid_h = tl.program_id(2)  # head idx
    pid_d = tl.program_id(0)
    pid_s = tl.program_id(1)

    offs_d = (pid_d * BLOCK_SIZE_D) + tl.arange(0, BLOCK_SIZE_D)
    offs_s = (pid_s * BLOCK_SIZE_S) + tl.arange(0, BLOCK_SIZE_S)

    # Mask out-of-bound tokens
    mask_d = offs_d < headdim
    mask_s = offs_s < dstate

    # Set base pointers
    dA_cumsum_ptr_base = dA_cumsum_ptr + pid_h * stride_dA_cumsum_h
    init_states_ptrs = init_states_ptr + (
        pid_h * stride_init_states_h + offs_d[:, None] * stride_init_states_d +
        offs_s[None, :] * stride_init_states_s)
    block_states_ptrs = block_states_ptr + (
        pid_h * stride_block_states_h + offs_d[:, None] * stride_block_states_d
        + offs_s[None, :] * stride_block_states_s)
    final_states_ptrs = final_states_ptr + (
        pid_h * stride_final_states_h + offs_d[:, None] * stride_final_states_d
        + offs_s[None, :] * stride_final_states_s)
    prev_states_ptrs = prev_states_ptr + (
        pid_h * stride_prev_states_h + offs_d[:, None] * stride_prev_states_d +
        offs_s[None, :] * stride_prev_states_s)

    b = tl.full([], -1, dtype=tl.int64)
    block_end = b
    prev_state = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_S), dtype=tl.float32)
    for n in range(nblocks):
        req_idx = tl.load(block_req_idx_ptr + n * stride_block_req_idx_n)
        if req_idx != b:  # new request
            b = req_idx
            prev_state = tl.load(init_states_ptrs + b * stride_init_states_b,
                                 mask=(mask_d[:, None] & mask_s[None, :]),
                                 other=0.0).to(tl.float32)

            block_end = tl.load(req_cu_nblocks_ptr +
                                (b + 1) * stride_req_cu_nblocks_b)

        if IS_STORE_PREV_STATES:
            tl.store(prev_states_ptrs + n * stride_prev_states_n,
                     prev_state,
                     mask=(mask_d[:, None] & mask_s[None, :]))

        # update state
        t_start = tl.load(block_cu_seqlens_ptr + n * stride_block_cu_seqlens_n)
        t_end = tl.load(block_cu_seqlens_ptr +
                        (n + 1) * stride_block_cu_seqlens_n)
        ntokens = t_end - t_start
        align_t_start = t_start
        if ALIGN_BLOCKS:
            align_t_start = tl.load(block_packed_cu_seqlens_ptr +
                                    n * stride_block_packed_cu_seqlens_n)
            align_t_start = tl.multiple_of(
                align_t_start, 4)  # not sure if the hint works in if block

        dA_cumsum_last = tl.load(
            dA_cumsum_ptr_base +
            (align_t_start + ntokens - 1) * stride_dA_cumsum_t)  # scalar
        block_decay = tl.exp(dA_cumsum_last)

        state = tl.load(block_states_ptrs + n * stride_block_states_n,
                        mask=(mask_d[:, None] & mask_s[None, :]),
                        other=0.0)
        state += block_decay * prev_state

        # store back
        if n == (block_end - 1):
            tl.store(final_states_ptrs + b * stride_final_states_b,
                     state,
                     mask=(mask_d[:, None] & mask_s[None, :]))
        else:
            # update prev_state only if it's not the last
            prev_state = state


# Perform inter-block state scan
# output:
# final_states (batch, nheads, headdim, dstate)
# prev_states (nblocks, nheads, headdim, dstate)
def state_passing(
    dA_cumsum,  # (nheads, seqlen)
    block_states,  # (nblocks, block_size, headdim, dstate)
    initial_states,  # (batch, nheads, headdim, dstate)
    # metadata
    block_cu_seqlens,  # (nblocks+1,)
    block_req_idx,  # (nblocks, )
    req_cu_nblocks,  # (batch+1, )
    block_packed_cu_seqlens=None,  # (nblocks+1,)
    return_prev_states=True,
    align_blocks=False,
    out_dtype=None,
):

    nheads, _ = dA_cumsum.shape
    nblocks, _, headdim, dstate = block_states.shape
    batch = initial_states.shape[0]

    assert block_states.shape == (nblocks, nheads, headdim, dstate)
    assert initial_states.shape == (batch, nheads, headdim, dstate)
    assert block_cu_seqlens.shape == (nblocks + 1, )
    assert block_req_idx.shape == (nblocks, )
    assert req_cu_nblocks.shape == (batch + 1, )
    if align_blocks:
        assert block_packed_cu_seqlens.shape == block_cu_seqlens.shape

    device = dA_cumsum.device

    # Allocate outputs
    out_dtype = block_states.dtype if out_dtype is None else out_dtype
    prev_states = (torch.empty_like(block_states, dtype=out_dtype)
                   if return_prev_states else None)
    final_states = torch.empty((batch, nheads, headdim, dstate),
                               dtype=out_dtype,
                               device=device)
    prev_state_strides = ((0, 0, 0, 0) if prev_states is None else
                          (prev_states.stride(0), prev_states.stride(1),
                           prev_states.stride(2), prev_states.stride(3)))
    # Launch grid
    # TODO: need to find a good decomposition strategy for max parallelism
    grid = lambda META: (triton.cdiv(headdim, META['BLOCK_SIZE_D']),
                         triton.cdiv(dstate, META['BLOCK_SIZE_S']), nheads)

    with torch.cuda.device(dA_cumsum.device.index):
        state_passing_kernel[grid](
            dA_cumsum_ptr=dA_cumsum,
            block_states_ptr=block_states,
            init_states_ptr=initial_states,
            block_cu_seqlens_ptr=block_cu_seqlens,
            block_packed_cu_seqlens_ptr=block_packed_cu_seqlens,
            block_req_idx_ptr=block_req_idx,
            req_cu_nblocks_ptr=req_cu_nblocks,
            prev_states_ptr=prev_states,
            final_states_ptr=final_states,
            nblocks=nblocks,
            headdim=headdim,
            dstate=dstate,
            stride_dA_cumsum_h=dA_cumsum.stride(0),
            stride_dA_cumsum_t=dA_cumsum.stride(1),
            stride_block_states_n=block_states.stride(0),
            stride_block_states_h=block_states.stride(1),
            stride_block_states_d=block_states.stride(2),
            stride_block_states_s=block_states.stride(3),
            stride_init_states_b=initial_states.stride(0),
            stride_init_states_h=initial_states.stride(1),
            stride_init_states_d=initial_states.stride(2),
            stride_init_states_s=initial_states.stride(3),
            stride_block_cu_seqlens_n=block_cu_seqlens.stride(0),
            stride_block_packed_cu_seqlens_n=(block_packed_cu_seqlens.stride(0)
                                              if align_blocks else 0),
            stride_block_req_idx_n=block_req_idx.stride(0),
            stride_req_cu_nblocks_b=req_cu_nblocks.stride(0),
            stride_prev_states_n=prev_state_strides[0],
            stride_prev_states_h=prev_state_strides[1],
            stride_prev_states_d=prev_state_strides[2],
            stride_prev_states_s=prev_state_strides[3],
            stride_final_states_b=final_states.stride(0),
            stride_final_states_h=final_states.stride(1),
            stride_final_states_d=final_states.stride(2),
            stride_final_states_s=final_states.stride(3),
            IS_STORE_PREV_STATES=return_prev_states,
            ALIGN_BLOCKS=align_blocks,
        )

    return final_states, prev_states
