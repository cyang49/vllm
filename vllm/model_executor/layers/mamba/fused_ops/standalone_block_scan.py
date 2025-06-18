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
#     configs=generate_autotune_combinations(spec={
#         'BLOCK_SIZE_D': [16, 32, 64],
#         'BLOCK_SIZE_T0': [16, 32, 64],
#         'BLOCK_SIZE_K': [16, 32, 64],
#         'num_warps': [2, 4],
#         'num_stages': [3, 4, 5],
#     }, ),
#     key=[],
# )
# @triton.autotune(  # best for tiled dstate
#     configs=[
#         triton.Config(
#             {
#                 'BLOCK_SIZE_D': 64,
#                 'BLOCK_SIZE_T0': 64,
#                 'BLOCK_SIZE_K': 64,
#             },
#             num_warps=4,
#             num_stages=3),
#     ],
#     key=[],
# )
@triton.autotune(  # best for non-tiled dstate
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_D': 64,
                'BLOCK_SIZE_T0': 64,
                'BLOCK_SIZE_K': 32,
            },
            num_warps=4,
            num_stages=3),
    ],
    key=[],
)
@triton.jit
def block_scan_kernel(
    # Inputs
    x_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    C_ptr,
    D_ptr,
    CB_ptr,
    block_cu_seqlens_ptr,
    prev_states_ptr,
    # Outputs
    output_ptr,
    # Matrix dimensions
    block_size: tl.constexpr,
    headdim: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # Strides
    stride_x_t: tl.constexpr,
    stride_x_h: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_dt_h,
    stride_dt_t: tl.constexpr,
    stride_dA_cumsum_h,
    stride_dA_cumsum_t: tl.constexpr,
    stride_C_t: tl.constexpr,
    stride_C_g: tl.constexpr,
    stride_C_s: tl.constexpr,
    stride_D_h: tl.constexpr,
    stride_CB_n: tl.constexpr,
    stride_CB_g: tl.constexpr,
    stride_CB_t0: tl.constexpr,
    stride_CB_t1: tl.constexpr,
    stride_block_cu_seqlens_n: tl.constexpr,
    stride_prev_states_n: tl.constexpr,
    stride_prev_states_h: tl.constexpr,
    stride_prev_states_d: tl.constexpr,
    stride_prev_states_s: tl.constexpr,
    stride_output_t: tl.constexpr,
    stride_output_h: tl.constexpr,
    stride_output_d: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_T0: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_n = tl.program_id(1)  # block idx
    pid_h = tl.program_id(2)  # head idx
    ndblocks = tl.cdiv(headdim, BLOCK_SIZE_D)
    pid_t0 = tl.program_id(0) // ndblocks
    pid_d = tl.program_id(0) % ndblocks

    offs_d = (pid_d * BLOCK_SIZE_D) + tl.arange(0, BLOCK_SIZE_D)

    # Load metadata
    # start and end token index in seqlen dimension
    t_start = tl.load(block_cu_seqlens_ptr + pid_n * stride_block_cu_seqlens_n)
    t_end = tl.load(block_cu_seqlens_ptr +
                    (pid_n + 1) * stride_block_cu_seqlens_n)
    ntokens = t_end - t_start

    # Mask out-of-bound tokens
    mask_d = offs_d < headdim

    # Set base pointers
    dA_cumsum_ptr_h = dA_cumsum_ptr + pid_h * stride_dA_cumsum_h

    pid_g = pid_h // nheads_ngroups_ratio  # group idx
    t0_range = tl.arange(0, BLOCK_SIZE_T0)
    t0 = pid_t0 * BLOCK_SIZE_T0
    offs_t0_local = t0 + t0_range
    offs_t0_global = t_start + offs_t0_local
    mask_t0 = offs_t0_local < ntokens

    # compute base pointers
    x_ptr_base = x_ptr + pid_h * stride_x_h
    dt_ptr_base = dt_ptr + pid_h * stride_dt_h
    CB_ptr_base = CB_ptr + pid_n * stride_CB_n + pid_g * stride_CB_g
    C_ptr_base = C_ptr + pid_g * stride_C_g
    output_ptr_base = output_ptr + pid_h * stride_output_h
    prev_states_ptr_base = prev_states_ptr + pid_n * stride_prev_states_n + pid_h * stride_prev_states_h

    # 2. compute output
    x_t0_ptrs = x_ptr_base + (
        offs_t0_global[:, None] * stride_x_t + offs_d[None, :] * stride_x_d
    )  # (BLOCK_SIZE_T0, BLOCK_SIZE_D)
    x_t0 = tl.load(x_t0_ptrs,
                   mask=(mask_t0[:, None] & mask_d[None, :]),
                   other=0.0)
    D = tl.load(D_ptr + pid_h * stride_D_h)
    acc = (x_t0 * D).to(tl.float32)

    dA_cumsum_t0_ptrs = dA_cumsum_ptr_h + (offs_t0_global * stride_dA_cumsum_t
                                           )  # (BLOCK_SIZE_T0,)
    dA_cumsum_t0 = tl.load(dA_cumsum_t0_ptrs, mask=mask_t0, other=0.0)

    k_range = tl.arange(0, BLOCK_SIZE_K)

    # 2.2 compute off-diagonal block contributions to the output
    if (BLOCK_SIZE_S <= 128):
        offs_s = tl.arange(0, BLOCK_SIZE_S)
        mask_s = offs_s < dstate
        prev_states_ptrs = prev_states_ptr_base + (
            offs_d[:, None] * stride_prev_states_d +
            offs_s[None, :] * stride_prev_states_s)
        prev_state = tl.load(prev_states_ptrs,
                             mask=(mask_d[:, None] & mask_s[None, :]),
                             other=0.0)  # (BLOCK_SIZE_D, BLOCK_SIZE_S)
        prev_state = prev_state.to(C_ptr.dtype.element_ty)

        C_ptrs = C_ptr_base + (
            offs_t0_global[:, None] * stride_C_t + offs_s[None, :] * stride_C_s
        )  # (BLOCK_SIZE_T0, BLOCK_SIZE_S)
        C = tl.load(C_ptrs,
                    mask=(mask_t0[:, None] & mask_s[None, :]),
                    other=0.0)
        acc += (tl.dot(C, prev_state.T) * tl.exp(dA_cumsum_t0[:, None])
                )  #(BLOCK_SIZE_T0, BLOCK_SIZE_D)
    else:
        for ss in range(0, dstate, BLOCK_SIZE_K):
            offs_ss = ss + k_range
            mask_ss = offs_ss < dstate

            prev_states_ptrs = prev_states_ptr_base + (
                offs_d[:, None] * stride_prev_states_d +
                offs_ss[None, :] * stride_prev_states_s)
            prev_state = tl.load(prev_states_ptrs,
                                 mask=(mask_d[:, None] & mask_ss[None, :]),
                                 other=0.0)  # (BLOCK_SIZE_D, BLOCK_SIZE_K)
            prev_state = prev_state.to(C_ptr.dtype.element_ty)

            C_ptrs = C_ptr_base + (offs_t0_global[:, None] * stride_C_t +
                                   offs_ss[None, :] * stride_C_s
                                   )  # (BLOCK_SIZE_T0, BLOCK_SIZE_K)
            C = tl.load(C_ptrs,
                        mask=(mask_t0[:, None] & mask_ss[None, :]),
                        other=0.0)
            acc += (tl.dot(C, prev_state.T) * tl.exp(dA_cumsum_t0[:, None])
                    )  #(BLOCK_SIZE_T0, BLOCK_SIZE_D)

    for t1 in range(0, ntokens, BLOCK_SIZE_K):
        mask_t1 = k_range < (ntokens - t1)  # (BLOCK_SIZE_K,)
        offs_t1_local = t1 + k_range
        offs_t1_global = t_start + offs_t1_local
        mask_t1 = offs_t1_local < ntokens

        dA_cumsum_t1_ptrs = dA_cumsum_ptr_h + (
            offs_t1_global * stride_dA_cumsum_t)  # (BLOCK_SIZE_K,)
        dt_t1_ptrs = dt_ptr_base + (offs_t1_global * stride_dt_t
                                    )  # (BLOCK_SIZE_K,)
        dt_t1 = tl.load(dt_t1_ptrs, mask=mask_t1, other=0.0)
        dA_cumsum_t1 = tl.load(dA_cumsum_t1_ptrs, mask=mask_t1, other=0.0)
        x_t1_ptrs = x_ptr_base + (
            offs_t1_global[:, None] * stride_x_t + offs_d[None, :] * stride_x_d
        )  # (BLOCK_SIZE_K, BLOCK_SIZE_D)
        x_t1 = tl.load(x_t1_ptrs,
                       mask=(mask_t1[:, None] & mask_d[None, :]),
                       other=0.0)
        CB_ptrs = CB_ptr_base + (offs_t0_local[:, None] * stride_CB_t0 +
                                 offs_t1_local[None, :] * stride_CB_t1
                                 )  # (BLOCK_SIZE_T0, BLOCK_SIZE_K)

        # 2.1 compute diagonal output contribution
        # ok to not mask - causal mask applied later
        CB = tl.load(CB_ptrs)

        #(BLOCK_SIZE_T0, BLOCK_SIZE_K)
        seg_sum = dA_cumsum_t0[:, None] - dA_cumsum_t1[None, :]
        decay = tl.exp(seg_sum)
        # decay = fast_expf(seg_sum)

        scores = tl.where((offs_t0_local[:, None] >= offs_t1_local[None, :]),
                          CB * decay * dt_t1[None, :], 0.0)
        # lower precision (prevents out of resource compile error on H100)
        scores = scores.to(x_ptr.dtype.element_ty)
        acc += tl.dot(scores, x_t1)

    out_ptrs = output_ptr_base + (offs_t0_global[:, None] * stride_output_t +
                                  offs_d[None, :] * stride_output_d)
    tl.store(out_ptrs, acc, mask=(mask_t0[:, None] & mask_d[None, :]))


# output:
# output (seqlen, nheads, headdim)
def block_scan(
        x,  # (seqlen, nheads, headdim)
        dt,  # (nheads, seqlen)
        dA_cumsum,  # (nheads, seqlen)
        prev_states,  # (nblocks, block_size, headdim, dstate)
        C,  # (seqlen, ngroups, dstate)
        D,
        CB,  # (nblocks, ngroups, block_size, block_size)
        # metadata
    block_cu_seqlens,  # (nblocks+1,)
):
    seqlen, nheads, headdim = x.shape
    ngroups = C.shape[1]
    dstate = C.shape[-1]
    block_size = CB.shape[-1]
    nblocks = prev_states.shape[0]

    assert dt.shape == (nheads, seqlen)
    assert dA_cumsum.shape == (nheads, seqlen)
    assert C.shape == (seqlen, ngroups, dstate)
    assert D.shape == (nheads, )
    assert CB.shape == (nblocks, ngroups, block_size, block_size)
    assert prev_states.shape == (nblocks, nheads, headdim, dstate)
    assert block_cu_seqlens.shape == (nblocks + 1, )

    # Allocate outputs
    output = torch.empty_like(x)

    # Launch grid
    grid = lambda META: (triton.cdiv(
        block_size, META['BLOCK_SIZE_T0']) * triton.cdiv(
            headdim, META['BLOCK_SIZE_D']), nblocks, nheads)

    with torch.cuda.device(x.device.index):
        block_scan_kernel[grid](
            x_ptr=x,
            dt_ptr=dt,
            dA_cumsum_ptr=dA_cumsum,
            C_ptr=C,
            D_ptr=D,
            CB_ptr=CB,
            block_cu_seqlens_ptr=block_cu_seqlens,
            prev_states_ptr=prev_states,
            output_ptr=output,
            block_size=block_size,
            headdim=headdim,
            dstate=dstate,
            nheads_ngroups_ratio=nheads // ngroups,
            stride_x_t=x.stride(0),
            stride_x_h=x.stride(1),
            stride_x_d=x.stride(2),
            stride_dt_h=dt.stride(0),
            stride_dt_t=dt.stride(1),
            stride_dA_cumsum_h=dA_cumsum.stride(0),
            stride_dA_cumsum_t=dA_cumsum.stride(1),
            stride_C_t=C.stride(0),
            stride_C_g=C.stride(1),
            stride_C_s=C.stride(2),
            stride_D_h=D.stride(0),
            stride_CB_n=CB.stride(0),
            stride_CB_g=CB.stride(1),
            stride_CB_t0=CB.stride(2),
            stride_CB_t1=CB.stride(3),
            stride_block_cu_seqlens_n=block_cu_seqlens.stride(0),
            stride_prev_states_n=prev_states.stride(0),
            stride_prev_states_h=prev_states.stride(1),
            stride_prev_states_d=prev_states.stride(2),
            stride_prev_states_s=prev_states.stride(3),
            stride_output_t=output.stride(0),
            stride_output_h=output.stride(1),
            stride_output_d=output.stride(2),
            BLOCK_SIZE_S=max(dstate, 16),
        )

    return output
