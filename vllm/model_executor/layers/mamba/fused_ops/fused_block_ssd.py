# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E501,SIM102

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import softplus
from vllm.triton_utils import tl, triton


# Notations for readability
#   - h: nheads
#   - g: ngroups
#   - n: nblocks
#   - t: seqlen
#   - k: block_size
#   - d: headdim
#   - s: dstate
@triton.jit
def fused_block_ssd_kernel(
    # Inputs
    x_ptr,
    dt_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    block_cu_seqlens_ptr,
    # block_ntokens_ptr,
    dt_bias_ptr,
    # Outputs
    dA_cumsum_ptr,
    block_states_ptr,
    CB_ptr,
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
    # stride_block_ntokens_n: tl.constexpr,
    stride_dt_bias_h: tl.constexpr,
    stride_dA_cumsum_t: tl.constexpr,
    stride_dA_cumsum_h: tl.constexpr,
    stride_block_states_n: tl.constexpr,
    stride_block_states_h: tl.constexpr,
    stride_block_states_d: tl.constexpr,
    stride_block_states_s: tl.constexpr,
    stride_CB_n: tl.constexpr,
    stride_CB_g: tl.constexpr,
    stride_CB_k0: tl.constexpr,
    stride_CB_k1: tl.constexpr,
    # Meta-parameters
    HAS_DT_BIAS: tl.constexpr,
    USE_DT_SOFTPLUS: tl.constexpr,
    # states_in_fp32: tl.constexpr,
    # finer grain decomposition of block size dimension
    # BLOCK_SIZE_KK: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
):
    pid_n = tl.program_id(0)  # block idx
    pid_h = tl.program_id(1)  # head idx
    pid_g = pid_h % nheads_ngroups_ratio  # group idx

    offs_k = tl.arange(0, block_size)
    # offs_kk = tl.arange(0, BLOCK_SIZE_KK)
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    offs_s = tl.arange(0, BLOCK_SIZE_S)

    # Load block start and end offset
    t_start = tl.load(block_cu_seqlens_ptr + pid_n * stride_block_cu_seqlens_n)
    t_end = tl.load(block_cu_seqlens_ptr +
                    (pid_n + 1) * stride_block_cu_seqlens_n)
    ntokens = t_end - t_start
    # Load number of tokens for this block for range checks
    # ntokens <= block_size
    # ntokens = tl.load(block_ntokens_ptr + pid_n * stride_block_ntokens_n)

    # Mask out-of-bound tokens
    mask_k = offs_k < ntokens
    mask_d = offs_d < headdim
    mask_s = offs_s < dstate

    # Compute base pointer addresses
    x_ptr += t_start * stride_x_t + pid_h * stride_x_h
    dt_ptr += t_start * stride_dt_t + pid_h * stride_dt_h
    A_ptr += pid_h * stride_A_h
    B_ptr += t_start * stride_B_t + pid_g * stride_B_g
    block_states_ptr += pid_n * stride_block_states_n + pid_h * stride_block_states_h

    # Compute pointer arrays for blocks
    dt_ptrs = dt_ptr + offs_k

    # dt and dA_cumsum computations
    dt = tl.load(dt_ptrs, mask=mask_k,
                 other=0.0).to(tl.float32)  # (block_size,)
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_h
        dt_bias = tl.load(dt_bias_ptr).to(tl.float32)
        dt += dt_bias
    if USE_DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, softplus(dt), dt)
    dt = tl.clamp(dt, 0.0, float('inf'))

    # reset out of bound values
    dt = tl.where(mask_k, dt, 0.0)

    A = tl.load(A_ptr).to(tl.float32)
    dA = dt * A
    dA_cs = tl.cumsum(dA, axis=0)
    dA_cs_last = tl.sum(dA, axis=0)

    dA_cumsum_ptr += t_start * stride_dA_cumsum_t + pid_h * stride_dA_cumsum_h
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k
    tl.store(dA_cumsum_ptrs, dA_cs, mask=mask_k)

    # Compute block states
    x_ptrs = x_ptr + offs_d[:, None] * stride_x_d + offs_k[
        None, :] * stride_x_t  # (headdim, block_size)
    B_ptrs = B_ptr + offs_k[:, None] * stride_B_t + offs_s[
        None, :] * stride_B_s  # (block_size, dstate)
    x = tl.load(x_ptrs, mask=(mask_d[:, None] & mask_k[None, :]),
                other=0.0).to(tl.float32)
    B = tl.load(B_ptrs, mask=(mask_k[:, None] & mask_s[None, :]), other=0.0)

    decay_states = tl.exp(dA_cs_last - dA_cs) * dt  # (block_size, )
    B_decay = B.to(tl.float32) * decay_states[:, None]

    block_state = tl.dot(x, B_decay)
    block_states_ptrs = block_states_ptr + offs_d[:,
                                                  None] * stride_block_states_d + offs_s[
                                                      None, :] * stride_block_states_s
    tl.store(block_states_ptrs,
             block_state,
             mask=(mask_d[:, None] & mask_s[None, :]))

    # # Accumulate block_states
    # acc = tl.zeros((headdim, dstate), dtype=tl.float32)

    # for t in range(0, block_size):
    #     if t >= ntokens:
    #         continue

    #     decay = decay_states[t]

    #     for i in range(headdim):
    #         x_val = tl.load(x_x_ptr + block_id * stride_x_b + t * stride_x_t + head_id * stride_x_h + i * stride_x_d)
    #         for j in range(dstate):
    #             B_val = tl.load(B_x_ptr + block_id * stride_B_b + t * stride_B_t + head_id * stride_B_h + j * stride_B_d)
    #             acc[i, j] += x_val * B_val * decay

    # # Store block_states
    # for i in range(headdim):
    #     for j in range(dstate):
    #         out_ptr = block_states_ptr + block_id * stride_out_b + head_id * stride_out_h + i * stride_out_hd + j * stride_out_ds
    #         val = acc[i, j]
    #         if dtype_out == tl.float16:
    #             val = val.to(tl.float16)
    #         tl.store(out_ptr, val)

    # Compute CB matrix per group
    if pid_h % nheads_ngroups_ratio == 0:
        C_ptr += t_start * stride_C_t + pid_g * stride_C_g
        CB_ptr += pid_n * stride_CB_n + pid_g * stride_CB_g
        C_ptrs = C_ptr + offs_k[:, None] * stride_C_t + offs_s[
            None, :] * stride_C_s  # (block_size, dstate)
        CB_ptrs = CB_ptr + offs_k[:, None] * stride_CB_k0 + offs_k[
            None, :] * stride_CB_k1  # (block_size, block_size)
        C = tl.load(C_ptrs,
                    mask=(mask_k[:, None] & mask_s[None, :]),
                    other=0.0)

        CB = tl.dot(C, B.T)  # (block_size, block_size)

        tl.store(CB_ptrs, CB, mask=(mask_k[:, None] & mask_k[None, :]))


# Fused block SSD performs intra-block computations on varlen input batch
# The implementation uses block metadata to determine memory access ranges
# Padding of sequences are not needed
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
    dA_cumsum = torch.empty_like(dt)  # (seqlen, nheads)
    block_states = torch.empty(
        (nblocks, nheads, headdim, dstate),
        dtype=torch.float32 if states_in_fp32 else dtype,
        device=device)
    CB = torch.empty((nblocks, ngroups, block_size, block_size),
                     dtype=torch.float32,
                     device=device)

    # Launch grid
    grid = (nblocks, nheads)

    with torch.cuda.device(x.device.index):
        fused_block_ssd_kernel[grid](
            x_ptr=x,
            dt_ptr=dt,
            A_ptr=A,
            B_ptr=B,
            C_ptr=C,
            dA_cumsum_ptr=dA_cumsum,
            block_cu_seqlens_ptr=block_cu_seqlens,
            block_states_ptr=block_states,
            CB_ptr=CB,
            # block_ntokens_ptr=block_ntokens,
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
            # stride_block_ntokens_n=block_ntokens.stride(0),
            stride_dt_bias_h=0 if dt_bias is None else dt_bias.stride(0),
            stride_dA_cumsum_t=dA_cumsum.stride(0),
            stride_dA_cumsum_h=dA_cumsum.stride(1),
            stride_block_states_n=block_states.stride(0),
            stride_block_states_h=block_states.stride(1),
            stride_block_states_d=block_states.stride(2),
            stride_block_states_s=block_states.stride(3),
            stride_CB_n=CB.stride(0),
            stride_CB_g=CB.stride(1),
            stride_CB_k0=CB.stride(2),
            stride_CB_k1=CB.stride(3),
            HAS_DT_BIAS=(dt_bias is not None),
            USE_DT_SOFTPLUS=dt_softplus,
            # states_in_fp32=states_in_fp32,
            BLOCK_SIZE_D=max(headdim, 16),
            BLOCK_SIZE_S=max(dstate, 16),
        )

    return dA_cumsum, block_states, CB
