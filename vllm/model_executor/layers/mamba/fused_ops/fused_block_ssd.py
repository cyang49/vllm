# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E501,SIM102,SIM113

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import softplus
from vllm.triton_utils import tl, triton


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_TT': 32,
            'BLOCK_SIZE_D': 32,
            'BLOCK_SIZE_SS': 32,
        }),  # best on H100
    ],
    key=[],
)
@triton.jit
def fused_block_ssd_v2_kernel(  # 0.112 mseconds for 8 full blocks on H100
        # Inputs
        x_ptr,
        dt_ptr,
        A_ptr,
        B_ptr,
        C_ptr,
        block_cu_seqlens_ptr,
        dt_bias_ptr,
        # Outputs
        dA_cumsum_ptr,
        block_states_ptr,
        CB_ptr,
        dt_out_ptr,
        # Matrix dimensions
        block_size: tl.constexpr,
        headdim: tl.constexpr,
        dstate: tl.constexpr,
        nheads_ngroups_ratio: tl.constexpr,
        # Strides
        stride_x_t: tl.constexpr,
        stride_x_h: tl.constexpr,
        stride_x_d: tl.constexpr,
        stride_dt_t: tl.constexpr,
        stride_dt_h: tl.constexpr,
        stride_A_h: tl.constexpr,
        stride_B_t: tl.constexpr,
        stride_B_g: tl.constexpr,
        stride_B_s: tl.constexpr,
        stride_C_t: tl.constexpr,
        stride_C_g: tl.constexpr,
        stride_C_s: tl.constexpr,
        stride_block_cu_seqlens_n: tl.constexpr,
        stride_dt_bias_h: tl.constexpr,
        stride_dA_cumsum_h,
        stride_dA_cumsum_t: tl.constexpr,
        stride_block_states_n: tl.constexpr,
        stride_block_states_h: tl.constexpr,
        stride_block_states_d: tl.constexpr,
        stride_block_states_s: tl.constexpr,
        stride_CB_n: tl.constexpr,
        stride_CB_g: tl.constexpr,
        stride_CB_t0: tl.constexpr,
        stride_CB_t1: tl.constexpr,
        stride_dt_out_h,
        stride_dt_out_t: tl.constexpr,
        # Meta-parameters
        HAS_DT_BIAS: tl.constexpr,
        USE_DT_SOFTPLUS: tl.constexpr,
        FUSED_COMPUTE_CB: tl.constexpr,
        # finer grain decomposition of block size dimension
        BLOCK_SIZE_TT: tl.constexpr,
        BLOCK_SIZE_D: tl.constexpr,
        # full dstate for xB loop, in case dstate < MIN_BLOCK_SIZE
        BLOCK_SIZE_S: tl.constexpr,
        BLOCK_SIZE_SS: tl.constexpr,  # for CB loop
):
    tl.static_assert(dstate >= BLOCK_SIZE_SS)
    pid_n = tl.program_id(0)  # block idx
    pid_h = tl.program_id(1)  # head idx
    pid_d = tl.program_id(2)
    pid_g = pid_h // nheads_ngroups_ratio  # group idx

    offs_t = tl.arange(0, block_size)
    offs_tt = tl.arange(0, BLOCK_SIZE_TT)
    offs_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    offs_s = tl.arange(0, BLOCK_SIZE_S)

    # Load block start and end offset
    t_start = tl.load(block_cu_seqlens_ptr + pid_n * stride_block_cu_seqlens_n)
    t_end = tl.load(block_cu_seqlens_ptr +
                    (pid_n + 1) * stride_block_cu_seqlens_n)
    ntokens = t_end - t_start

    # Mask out-of-bound tokens
    mask_t = offs_t < ntokens
    mask_d = offs_d < headdim
    mask_s = offs_s < dstate

    # Compute base pointer addresses
    x_ptr += t_start * stride_x_t + pid_h * stride_x_h
    dt_ptr += t_start * stride_dt_t + pid_h * stride_dt_h
    A_ptr += pid_h * stride_A_h
    B_ptr += t_start * stride_B_t + pid_g * stride_B_g

    block_states_ptr += pid_n * stride_block_states_n + pid_h * stride_block_states_h

    # Compute pointer arrays for blocks
    dt_ptrs = dt_ptr + offs_t * stride_dt_t

    # dt and dA_cumsum computations
    dt = tl.load(dt_ptrs, mask=mask_t,
                 other=0.0).to(tl.float32)  # (block_size,)
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_h
        dt_bias = tl.load(dt_bias_ptr)
        dt += dt_bias
    if USE_DT_SOFTPLUS:
        dt = softplus(dt)
    dt = tl.clamp(dt, min=0.0, max=float('inf'))

    # reset out of bound values
    dt = tl.where(mask_t, dt, 0.0)

    A = tl.load(A_ptr)
    dA = dt * A
    dA_cs = tl.cumsum(dA, axis=0)
    dA_cs_last = tl.sum(dA, axis=0)

    dA_cumsum_ptrs = (dA_cumsum_ptr + pid_h * stride_dA_cumsum_h +
                      (t_start + offs_t) * stride_dA_cumsum_t)
    # NOTE: redundantly computed, but no need to redundantly store
    if (pid_d == 0):
        tl.store(dA_cumsum_ptrs, dA_cs, mask=mask_t)

    dt_out_ptrs = (dt_out_ptr + pid_h * stride_dt_out_h +
                   (t_start + offs_t) * stride_dt_out_t)
    # NOTE: redundantly computed, but no need to redundantly store
    if (pid_d == 0):
        tl.store(dt_out_ptrs, dt, mask=mask_t)

    # 2. Compute block states

    # Compute decay from dt and dA_cs
    decay_states = tl.exp(dA_cs_last - dA_cs) * dt
    decay_states = decay_states.reshape(block_size // BLOCK_SIZE_TT,
                                        BLOCK_SIZE_TT)
    offs_row = tl.arange(0, decay_states.shape[0])

    acc = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_S), dtype=tl.float32)
    i = 0
    # In a for loop, process tokens within the length bound (ntokens)
    for tt in range(0, ntokens, BLOCK_SIZE_TT):
        mask_tt = offs_tt < (ntokens - tt)
        x_ptr_tt = x_ptr + tt * stride_x_t
        B_ptr_tt = B_ptr + tt * stride_B_t

        x_ptrs = x_ptr_tt + offs_d[:, None] * stride_x_d + offs_tt[
            None, :] * stride_x_t  # (headdim, BLOCK_SIZE_TT)
        B_ptrs = B_ptr_tt + offs_tt[:, None] * stride_B_t + offs_s[
            None, :] * stride_B_s  # (BLOCK_SIZE_TT, dstate)

        x_tt = tl.load(x_ptrs,
                       mask=(mask_d[:, None] & mask_tt[None, :]),
                       other=0.0).to(tl.float32)
        B_tt = tl.load(B_ptrs,
                       mask=(mask_tt[:, None] & mask_s[None, :]),
                       other=0.0).to(tl.float32)

        # Row selection work-around with masking
        decay_states_tt = tl.where(
            (offs_row == i)[:, None], decay_states,
            0.0)  #((block_size//BLOCK_SIZE_TT), BLOCK_SIZE_TT,)
        decay_states_tt = tl.sum(decay_states_tt, axis=0)  # (BLOCK_SIZE_TT,)

        B_decay = B_tt * decay_states_tt[:, None]

        acc += tl.dot(x_tt, B_decay)

        # Increment row index
        i += 1

    # Store back
    block_states_ptrs = block_states_ptr + \
        offs_d[:,None] * stride_block_states_d + \
        offs_s[None, :] * stride_block_states_s
    tl.store(block_states_ptrs, acc, mask=(mask_d[:, None] & mask_s[None, :]))

    if FUSED_COMPUTE_CB:
        # Compute CB matrix per group
        # TODO: The if condition make sure only a few triton programs will
        # compute the submatrices. But it is possible to distribute the
        # work and utilize more triton programs.
        # C @ B.T is independent of prior computations and can be
        # parallelized. It's effectively a batched matmul over
        # (nblocks × ngroups) of square matrices (block_size × block_size).
        # We can apply 2D output tiling to better utilize the grid and
        # balance the workload. As a first step, we can try to distribute
        # work of 1 group among multiple heads of that group instead of
        # making only 1 head active
        if (pid_h % nheads_ngroups_ratio == 0) and (pid_d == 0):
            offs_ss = tl.arange(0, BLOCK_SIZE_SS)
            C_ptr += t_start * stride_C_t + pid_g * stride_C_g
            cb = tl.zeros((block_size, block_size), dtype=tl.float32)
            for ss in tl.range(0, dstate, BLOCK_SIZE_SS):
                offs_ss += ss
                mask_ss = offs_ss < dstate
                # # (BLOCK_SIZE_SS, block_size, )
                # B_ptrs = B_ptr + (offs_ss[:, None] * stride_B_s +
                #                   offs_t[None, :] * stride_B_t)
                # # (block_size, BLOCK_SIZE_SS)
                # C_ptrs = C_ptr + (offs_t[:, None] * stride_C_t +
                #                     offs_ss[None, :] * stride_C_s)

                # C = tl.load(C_ptrs,
                #             mask=(mask_t[:, None] & mask_ss[None, :]),
                #             other=0.0)
                # B = tl.load(B_ptrs,
                #             mask=(mask_t[None, :] & mask_ss[:, None]),
                #             other=0.0)
                # cb += tl.dot(C, B)  # (block_size, block_size)

                # (block_size, BLOCK_SIZE_SS)
                B_ptrs = B_ptr + (offs_t[:, None] * stride_B_t +
                                  offs_ss[None, :] * stride_B_s)
                # (block_size, BLOCK_SIZE_SS)
                C_ptrs = C_ptr + (offs_t[:, None] * stride_C_t +
                                  offs_ss[None, :] * stride_C_s)

                C = tl.load(C_ptrs,
                            mask=(mask_t[:, None] & mask_ss[None, :]),
                            other=0.0)
                B = tl.load(B_ptrs,
                            mask=(mask_t[:, None] & mask_ss[None, :]),
                            other=0.0)
                cb += tl.dot(C, B.T)  # (block_size, block_size)

            CB_ptr += pid_n * stride_CB_n + pid_g * stride_CB_g
            CB_ptrs = CB_ptr + offs_t[:, None] * stride_CB_t0 + offs_t[
                None, :] * stride_CB_t1  # (block_size, block_size)
            tl.store(CB_ptrs, cb, mask=(mask_t[:, None] & mask_t[None, :]))


# Fused block SSD performs intra-block computations on varlen input batch
# The implementation uses block metadata to determine memory access ranges
# Padding of sequences are not needed
# NOTE: this function updates dt in-place
def fused_block_ssd(
    x,  # (seqlen, nheads, headdim)
    dt,  # (seqlen, nheads)
    A,  # (nheads,)
    B,  # (seqlen, ngroups, dstate)
    C,  # (seqlen, ngroups, dstate)
    block_size,
    # metadata
    block_ntokens,  # (nblocks,)
    block_cu_seqlens,  # (nblocks+1,)
    dt_bias=None,  # (nheads, )
    dt_softplus=False,
    states_in_fp32=True,
    FUSED_COMPUTE_CB=True,
):
    seqlen, nheads, headdim = x.shape
    ngroups = B.shape[1]
    dstate = B.shape[-1]
    nblocks = block_ntokens.shape[0]

    assert dt.shape == (seqlen, nheads)
    assert A.shape == (nheads, )
    assert C.shape == (seqlen, ngroups, dstate)
    assert C.shape == B.shape
    assert block_cu_seqlens.shape == (nblocks + 1, )

    device = x.device
    dtype = x.dtype

    # Allocate outputs
    dA_cumsum = torch.empty((nheads, seqlen),
                            device=device,
                            dtype=torch.float32)
    dt_out = torch.empty_like(dA_cumsum)
    # NOTE: block_states has nheads (and not ngroups), because each head in x
    #       is paired with a group head in B (many-to-few) resulting in nheads
    #       in the output states. This is different from GQA in regular
    #       transformer KV cache where grouped KV is cached
    block_states = torch.empty(
        (nblocks, nheads, headdim, dstate),
        dtype=torch.float32 if states_in_fp32 else dtype,
        device=device)

    CB = (torch.empty((nblocks, ngroups, block_size, block_size),
                      dtype=torch.float32,
                      device=device) if FUSED_COMPUTE_CB else None)
    CB_strides = (0, 0, 0, 0) if CB is None else (CB.stride(0), CB.stride(1),
                                                  CB.stride(2), CB.stride(3))
    MIN_BLOCK_SIZE = 16  # for tl.dot limitation
    # Launch grid
    # NOTE: parallelizing along headdim result in redundant computations of
    #       dt and dA_cumsum in thread blocks
    grid = lambda META: (nblocks, nheads,
                         triton.cdiv(headdim, META["BLOCK_SIZE_D"]))
    with torch.cuda.device(x.device.index):
        # using v2 as default for now as it's slightly faster
        # still seems to underperform compared with original unfused kernels
        fused_block_ssd_v2_kernel[grid](
            x_ptr=x,
            dt_ptr=dt,
            A_ptr=A,
            B_ptr=B,
            C_ptr=C,
            dA_cumsum_ptr=dA_cumsum,
            block_cu_seqlens_ptr=block_cu_seqlens,
            block_states_ptr=block_states,
            CB_ptr=CB,
            dt_out_ptr=dt_out,
            dt_bias_ptr=dt_bias,
            block_size=block_size,
            headdim=headdim,
            dstate=dstate,
            nheads_ngroups_ratio=nheads // ngroups,
            stride_x_t=x.stride(0),
            stride_x_h=x.stride(1),
            stride_x_d=x.stride(2),
            stride_dt_t=dt.stride(0),
            stride_dt_h=dt.stride(1),
            stride_A_h=A.stride(0),
            stride_B_t=B.stride(0),
            stride_B_g=B.stride(1),
            stride_B_s=B.stride(2),
            stride_C_t=C.stride(0),
            stride_C_g=C.stride(1),
            stride_C_s=C.stride(2),
            stride_block_cu_seqlens_n=block_cu_seqlens.stride(0),
            stride_dt_bias_h=0 if dt_bias is None else dt_bias.stride(0),
            stride_dA_cumsum_h=dA_cumsum.stride(0),
            stride_dA_cumsum_t=dA_cumsum.stride(1),
            stride_dt_out_h=dt_out.stride(0),
            stride_dt_out_t=dt_out.stride(1),
            stride_block_states_n=block_states.stride(0),
            stride_block_states_h=block_states.stride(1),
            stride_block_states_d=block_states.stride(2),
            stride_block_states_s=block_states.stride(3),
            stride_CB_n=CB_strides[0],
            stride_CB_g=CB_strides[1],
            stride_CB_t0=CB_strides[2],
            stride_CB_t1=CB_strides[3],
            HAS_DT_BIAS=(dt_bias is not None),
            USE_DT_SOFTPLUS=dt_softplus,
            FUSED_COMPUTE_CB=FUSED_COMPUTE_CB,
            BLOCK_SIZE_S=max(dstate, MIN_BLOCK_SIZE),
        )

    return dA_cumsum, dt_out, block_states, CB
