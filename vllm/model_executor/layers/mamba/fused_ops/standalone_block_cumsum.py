# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E501,SIM102,SIM113

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import softplus
from vllm.triton_utils import tl, triton


@triton.jit
def align(start, pack_size=4):
    aligned_start = ((start + (pack_size - 1)) // pack_size) * pack_size
    aligned_start = tl.multiple_of(aligned_start, pack_size)
    return aligned_start


# from .utils import generate_autotune_combinations
# @triton.autotune(
#     configs=generate_autotune_combinations(
#         spec={'BLOCK_SIZE_H': [1, 2, 4, 8, 16, 32, 64, 128],
#               'num_warps': [2, 4, 8],
#               'num_stages': [3], # no software pipeline
#              },
#         ),
#     key=[],
# )
# @triton.autotune( # H100, align_blocks=False
#     configs=[
#         triton.Config({
#             'BLOCK_SIZE_H': 8,
#         }, num_warps=8, num_stages=3),
#     ],
#     key=[],
# )
@triton.autotune(  # H100, align_blocks=True
    configs=[
        triton.Config({
            'BLOCK_SIZE_H': 8,
        }, num_warps=4, num_stages=3),
    ],
    key=[],
)
@triton.jit
def block_cumsum_kernel(
    # Inputs
    dt_ptr,
    A_ptr,
    block_cu_seqlens_ptr,
    block_packed_cu_seqlens_ptr,
    dt_bias_ptr,
    # Outputs
    dA_cumsum_ptr,
    dt_out_ptr,
    # Matrix dimensions
    nheads: tl.constexpr,
    block_size: tl.constexpr,
    # Strides
    stride_dt_t: tl.constexpr,
    stride_dt_h: tl.constexpr,
    stride_A_h: tl.constexpr,
    stride_block_cu_seqlens_n: tl.constexpr,
    stride_block_packed_cu_seqlens_n: tl.constexpr,
    stride_dt_bias_h: tl.constexpr,
    stride_dA_cumsum_h,
    stride_dA_cumsum_t: tl.constexpr,
    stride_dt_out_h,
    stride_dt_out_t: tl.constexpr,
    # Meta-parameters
    HAS_DT_BIAS: tl.constexpr,
    USE_DT_SOFTPLUS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    ALLIGN_OUTPUT_BLOCKS: tl.constexpr,
):
    pid_n = tl.program_id(0)  # block idx
    pid_h = tl.program_id(1)  # head idx

    # Load block start and end offset
    t_start = tl.load(block_cu_seqlens_ptr + pid_n * stride_block_cu_seqlens_n)
    t_end = tl.load(block_cu_seqlens_ptr +
                    (pid_n + 1) * stride_block_cu_seqlens_n)
    ntokens = t_end - t_start

    offs_t = t_start + tl.arange(0, block_size)
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    # Mask out-of-bound tokens
    mask_t = offs_t < t_end
    mask_h = offs_h < nheads

    # Compute pointer arrays for blocks
    dt_ptrs = dt_ptr + (
        offs_t[:, None] * stride_dt_t + offs_h[None, :] * stride_dt_h
    )  #(block_size, BLOCK_SIZE_H)

    # dt and dA_cumsum computations
    dt = tl.load(dt_ptrs, mask=mask_t[:, None] & mask_h[None, :],
                 other=0.0).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_h,
                          mask=mask_h,
                          other=0.0)
        dt += dt_bias[None, :]
    if USE_DT_SOFTPLUS:
        dt = softplus(dt)
    dt = tl.clamp(dt, min=0.0, max=float('inf'))
    # reset out of bound values
    dt = tl.where(mask_t[:, None] & mask_h[None, :], dt, 0.0)

    A = tl.load(A_ptr + offs_h * stride_A_h, mask=mask_h, other=0.0)
    dA = dt * A[None, :]
    dA_cs = tl.cumsum(dA, axis=0)

    # Store back
    if ALLIGN_OUTPUT_BLOCKS:
        tl.static_assert(block_size % 4 == 0)
        # NOTE: the indices in block_packed_cu_seqlens can be padded
        #       to the pack_size. As such, while t_start points to
        #       actual starting position containing real data,
        #       (t_end-1) may point to a padded position.
        #       Also, (t_end-t_start) would be the count including
        #       padded elements. To get the real token count, use
        #       either block_cu_seqlen or block_ntokens.
        t_start = tl.load(block_packed_cu_seqlens_ptr +
                          pid_n * stride_block_packed_cu_seqlens_n)
        t_end = tl.load(block_packed_cu_seqlens_ptr +
                        (pid_n + 1) * stride_block_packed_cu_seqlens_n)
        offs_t = t_start + tl.arange(0, block_size)
        mask_t = offs_t < (t_start + ntokens)

    # FIXME: stores are not coalesced?
    dA_cumsum_ptrs = (dA_cumsum_ptr + offs_t[:, None] * stride_dA_cumsum_t +
                      offs_h[None, :] * stride_dA_cumsum_h)
    tl.store(dA_cumsum_ptrs, dA_cs, mask=mask_t[:, None] & mask_h[None, :])

    dt_out_ptrs = (dt_out_ptr + offs_h[None, :] * stride_dt_out_h +
                   offs_t[:, None] * stride_dt_out_t)
    tl.store(dt_out_ptrs, dt, mask=mask_t[:, None] & mask_h[None, :])


def block_cumsum(
    dt,  # (seqlen, nheads)
    A,  # (nheads,)
    block_size,
    # metadata
    block_cu_seqlens,  # (nblocks+1,)
    block_packed_cu_seqlens=None,  # (nblocks+1,)
    packed_seqlen=-1,
    dt_bias=None,  # (nheads, )
    dt_softplus=False,
    align_blocks=False,
):
    seqlen, nheads = dt.shape
    nblocks = block_cu_seqlens.shape[0] - 1

    assert dt.shape == (seqlen, nheads)
    assert A.shape == (nheads, )

    device = dt.device
    # dtype = dt.dtype

    # Allocate outputs
    if align_blocks:
        assert block_packed_cu_seqlens.shape == block_cu_seqlens.shape
        seqlen = packed_seqlen

    dA_cumsum = torch.empty((nheads, seqlen),
                            device=device,
                            dtype=torch.float32)
    dt_out = torch.empty_like(dA_cumsum)

    # Launch grid
    grid = lambda META: (nblocks, triton.cdiv(nheads, META["BLOCK_SIZE_H"]))
    with torch.cuda.device(dt.device.index):
        block_cumsum_kernel[grid](
            dt_ptr=dt,
            A_ptr=A,
            block_cu_seqlens_ptr=block_cu_seqlens,
            block_packed_cu_seqlens_ptr=block_packed_cu_seqlens,
            dt_bias_ptr=dt_bias,
            dA_cumsum_ptr=dA_cumsum,
            dt_out_ptr=dt_out,
            nheads=nheads,
            block_size=block_size,
            stride_dt_t=dt.stride(0),
            stride_dt_h=dt.stride(1),
            stride_A_h=A.stride(0),
            stride_block_cu_seqlens_n=block_cu_seqlens.stride(0),
            stride_block_packed_cu_seqlens_n=(block_packed_cu_seqlens.stride(0)
                                              if align_blocks else 0),
            stride_dt_bias_h=0 if dt_bias is None else dt_bias.stride(0),
            stride_dA_cumsum_h=dA_cumsum.stride(0),
            stride_dA_cumsum_t=dA_cumsum.stride(1),
            stride_dt_out_h=dt_out.stride(0),
            stride_dt_out_t=dt_out.stride(1),
            HAS_DT_BIAS=(dt_bias is not None),
            USE_DT_SOFTPLUS=dt_softplus,
            ALLIGN_OUTPUT_BLOCKS=align_blocks,
        )

    return dA_cumsum, dt_out
