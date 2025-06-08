# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E501,SIM102,SIM113

import torch

from vllm.triton_utils import tl, triton


# Notations for readability
#   - b: batch
#   - h: nheads
#   - g: ngroups
#   - n: nblocks
#   - t: seqlen
#   - k: block_size
#   - d: headdim
#   - s: dstate
@triton.jit
def fused_block_scan_v0_kernel(
    # Inputs
    x_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    block_states_ptr,
    init_states_ptr,
    C_ptr,
    CB_ptr,
    block_cu_seqlens_ptr,
    block_req_idx_ptr,
    block_ntokens_ptr,
    req_cu_nblocks_ptr,
    # Outputs
    prev_states_ptr,
    final_states_ptr,
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
    stride_dA_cumsum_t: tl.constexpr,
    stride_dA_cumsum_h: tl.constexpr,
    stride_block_states_n: tl.constexpr,
    stride_block_states_h: tl.constexpr,
    stride_block_states_d: tl.constexpr,
    stride_block_states_s: tl.constexpr,
    stride_init_states_b: tl.constexpr,
    stride_init_states_h: tl.constexpr,
    stride_init_states_d: tl.constexpr,
    stride_init_states_s: tl.constexpr,
    stride_C_t: tl.constexpr,
    stride_C_g: tl.constexpr,
    stride_C_s: tl.constexpr,
    stride_CB_n: tl.constexpr,
    stride_CB_g: tl.constexpr,
    stride_CB_k0: tl.constexpr,
    stride_CB_k1: tl.constexpr,
    stride_block_cu_seqlens_n: tl.constexpr,
    stride_block_req_idx_n: tl.constexpr,
    stride_block_ntokens_n: tl.constexpr,
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
    BLOCK_SIZE_KK: tl.constexpr,
):
    # pid_b = tl.program_id(0)
    pid_n = tl.program_id(0)  # block idx
    pid_h = tl.program_id(1)  # head idx
    # pid_g = pid_h // nheads_ngroups_ratio  # group idx

    # offs_k = tl.arange(0, block_size)
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    offs_s = tl.arange(0, BLOCK_SIZE_S)

    # Mask out-of-bound tokens
    # mask_k = offs_k < ntokens
    mask_d = offs_d < headdim
    mask_s = offs_s < dstate

    pid_b = tl.load(block_req_idx_ptr + pid_n * stride_block_req_idx_n)
    block_start = tl.load(req_cu_nblocks_ptr + pid_b * stride_req_cu_nblocks_b)
    block_end = tl.load(req_cu_nblocks_ptr +
                        (pid_b + 1) * stride_req_cu_nblocks_b)

    # Set base pointers
    block_states_ptrs = block_states_ptr + (
        pid_h * stride_block_states_h + offs_d[:, None] * stride_block_states_d
        + offs_s[None, :] * stride_block_states_s)
    prev_states_ptrs = prev_states_ptr + (
        pid_h * stride_prev_states_h + offs_d[:, None] * stride_prev_states_d +
        offs_s[None, :] * stride_prev_states_s)
    final_states_ptrs = final_states_ptr + (
        pid_b * stride_final_states_b + pid_h * stride_final_states_h +
        offs_d[:, None] * stride_final_states_d +
        offs_s[None, :] * stride_final_states_s)
    dA_cumsum_ptr += pid_h * stride_dA_cumsum_h

    init_states_ptrs = init_states_ptr + (
        pid_b * stride_init_states_b + pid_h * stride_init_states_h +
        offs_d[:, None] * stride_init_states_d +
        offs_s[None, :] * stride_init_states_s)
    prev_state = tl.load(init_states_ptrs,
                         mask=(mask_d[:, None] & mask_s[None, :]),
                         other=0.0)
    prev_state = prev_state.to(tl.float32)
    # scan states along nblocks dimension; redundant computed in thread blocks
    # of the same sequence
    for i in range(block_start, pid_n + 1):
        t_end = tl.load(block_cu_seqlens_ptr +
                        (i + 1) * stride_block_cu_seqlens_n)
        dA_cumsum_last = tl.load(dA_cumsum_ptr +
                                 (t_end - 1) * stride_dA_cumsum_t)  # scalar
        block_decay = tl.exp(dA_cumsum_last)

        block_states_ptrs_i = block_states_ptrs + i * stride_block_states_n
        state = tl.load(block_states_ptrs_i,
                        mask=(mask_d[:, None] & mask_s[None, :]),
                        other=0.0)
        state += block_decay * prev_state

        # NOTE: store backs are distributed among blocks of the same sequence
        if pid_n == i:  # last block
            if i == (block_end - 1):
                tl.store(final_states_ptrs,
                         state,
                         mask=(mask_d[:, None] & mask_s[None, :]))
            else:
                prev_states_ptrs_i = prev_states_ptrs + i * stride_prev_states_n
                tl.store(prev_states_ptrs_i,
                         state,
                         mask=(mask_d[:, None] & mask_s[None, :]))
        prev_state = state


# Fused block SSD performs inter-block scan and final output activation
# for now compute state passing results
# output:
# prev_states (nblocks, nheads, headdim, dstate)
# final_states (batch, nheads, headdim, dstate)
def fused_block_scan(
    x,  # (seqlen, nheads, headdim)
    dt,  # (seqlen, nheads)
    dA_cumsum,  # (seqlen, nheads)
    block_states,  # (nblocks, block_size, headdim, dstate)
    initial_states,  # (batch, nheads, headdim, dstate)
    C,  # (seqlen, ngroups, dstate)
    CB,  # (nblocks, ngroups, block_size, block_size)
    # metadata
    block_ntokens,  # (nblocks,)
    block_cu_seqlens,  # (nblocks+1,)
    block_req_idx,  # (nblocks, )
    req_cu_nblocks,  # (batch+1, )
    block_size_kk=None,
):
    seqlen, nheads, headdim = x.shape
    ngroups = C.shape[1]
    dstate = C.shape[-1]
    block_size = CB.shape[-1]
    nblocks = block_states.shape[0]
    batch = initial_states.shape[0]

    assert dt.shape == (seqlen, nheads)
    assert dA_cumsum.shape == (seqlen, nheads)
    assert C.shape == (seqlen, ngroups, dstate)
    assert CB.shape == (nblocks, ngroups, block_size, block_size)
    assert block_states.shape == (nblocks, nheads, headdim, dstate)
    assert initial_states.shape == (batch, nheads, headdim, dstate)
    assert block_cu_seqlens.shape == (nblocks + 1, )
    assert block_req_idx.shape == (nblocks, )
    assert block_ntokens.shape == (nblocks, )
    assert req_cu_nblocks.shape == (batch + 1, )

    device = x.device
    # dtype = x.dtype

    # Allocate outputs
    prev_states = torch.empty_like(block_states)
    final_states = torch.ones((batch, nheads, headdim, dstate),
                              dtype=block_states.dtype,
                              device=device)

    # Launch grid
    # NOTE: parallelizing along nblocks requires that state passing scan
    #       computed redundantly among blocks of the same sequence.
    #       Alternatively, we can implement parallel scan if it brings
    #       significant performance difference
    grid = (nblocks, nheads)

    with torch.cuda.device(x.device.index):
        fused_block_scan_v0_kernel[grid](
            x_ptr=x,
            dt_ptr=dt,
            dA_cumsum_ptr=dA_cumsum,
            block_states_ptr=block_states,
            init_states_ptr=initial_states,
            C_ptr=C,
            CB_ptr=CB,
            block_cu_seqlens_ptr=block_cu_seqlens,
            block_req_idx_ptr=block_req_idx,
            block_ntokens_ptr=block_ntokens,
            req_cu_nblocks_ptr=req_cu_nblocks,
            prev_states_ptr=prev_states,
            final_states_ptr=final_states,
            block_size=block_size,
            headdim=headdim,
            dstate=dstate,
            nheads_ngroups_ratio=nheads // ngroups,
            stride_x_t=x.stride(0),
            stride_x_h=x.stride(1),
            stride_x_d=x.stride(2),
            stride_dt_t=dt.stride(0),
            stride_dt_h=dt.stride(1),
            stride_dA_cumsum_t=dA_cumsum.stride(0),
            stride_dA_cumsum_h=dA_cumsum.stride(1),
            stride_block_states_n=block_states.stride(0),
            stride_block_states_h=block_states.stride(1),
            stride_block_states_d=block_states.stride(2),
            stride_block_states_s=block_states.stride(3),
            stride_init_states_b=initial_states.stride(0),
            stride_init_states_h=initial_states.stride(1),
            stride_init_states_d=initial_states.stride(2),
            stride_init_states_s=initial_states.stride(3),
            stride_C_t=C.stride(0),
            stride_C_g=C.stride(1),
            stride_C_s=C.stride(2),
            stride_CB_n=CB.stride(0),
            stride_CB_g=CB.stride(1),
            stride_CB_k0=CB.stride(2),
            stride_CB_k1=CB.stride(3),
            stride_block_cu_seqlens_n=block_cu_seqlens.stride(0),
            stride_block_req_idx_n=block_req_idx.stride(0),
            stride_block_ntokens_n=block_ntokens.stride(0),
            stride_req_cu_nblocks_b=req_cu_nblocks.stride(0),
            stride_prev_states_n=prev_states.stride(0),
            stride_prev_states_h=prev_states.stride(1),
            stride_prev_states_d=prev_states.stride(2),
            stride_prev_states_s=prev_states.stride(3),
            stride_final_states_b=final_states.stride(0),
            stride_final_states_h=final_states.stride(1),
            stride_final_states_d=final_states.stride(2),
            stride_final_states_s=final_states.stride(3),
            BLOCK_SIZE_D=max(headdim, 16),
            BLOCK_SIZE_S=max(dstate, 16),
            BLOCK_SIZE_KK=block_size_kk,
        )

    return prev_states, final_states
