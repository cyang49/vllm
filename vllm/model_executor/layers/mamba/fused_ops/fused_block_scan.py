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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_D': 16}),
        triton.Config({'BLOCK_SIZE_D': 32}),
        triton.Config({'BLOCK_SIZE_D': 64}),
    ],
    key=[],
)
@triton.jit
def fused_block_scan_v1_kernel(
    # Inputs
    x_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    block_states_ptr,
    init_states_ptr,
    C_ptr,
    D_ptr,
    CB_ptr,
    block_cu_seqlens_ptr,
    block_req_idx_ptr,
    req_cu_nblocks_ptr,
    # Outputs
    prev_states_ptr,
    final_states_ptr,
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
    stride_D_h: tl.constexpr,
    stride_CB_n: tl.constexpr,
    stride_CB_g: tl.constexpr,
    stride_CB_t0: tl.constexpr,
    stride_CB_t1: tl.constexpr,
    stride_block_cu_seqlens_n: tl.constexpr,
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
    stride_output_t: tl.constexpr,
    stride_output_h: tl.constexpr,
    stride_output_d: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_TT: tl.constexpr,
    IS_STORE_PREV_STATES: tl.constexpr,
):
    pid_n = tl.program_id(0)  # block idx
    pid_h = tl.program_id(1)  # head idx
    pid_d = tl.program_id(2)
    pid_g = pid_h // nheads_ngroups_ratio  # group idx

    offs_t = tl.arange(0, block_size)
    offs_tt = tl.arange(0, BLOCK_SIZE_TT)
    offs_d = (pid_d * BLOCK_SIZE_D) + tl.arange(0, BLOCK_SIZE_D)
    offs_s = tl.arange(0, BLOCK_SIZE_S)

    # Load metadata
    pid_b = tl.load(block_req_idx_ptr + pid_n * stride_block_req_idx_n)
    block_start = tl.load(req_cu_nblocks_ptr + pid_b * stride_req_cu_nblocks_b)
    block_end = tl.load(req_cu_nblocks_ptr +
                        (pid_b + 1) * stride_req_cu_nblocks_b)
    # start and end token index in seqlen dimension
    t_start = tl.load(block_cu_seqlens_ptr + pid_n * stride_block_cu_seqlens_n)
    t_end = tl.load(block_cu_seqlens_ptr +
                    (pid_n + 1) * stride_block_cu_seqlens_n)
    ntokens = t_end - t_start

    # Mask out-of-bound tokens
    mask_t = offs_t < ntokens
    mask_d = offs_d < headdim
    mask_s = offs_s < dstate

    # Set base pointers
    block_states_ptrs = block_states_ptr + (
        pid_h * stride_block_states_h + offs_d[:, None] * stride_block_states_d
        + offs_s[None, :] * stride_block_states_s)
    dA_cumsum_ptr_h = dA_cumsum_ptr + pid_h * stride_dA_cumsum_h
    init_states_ptrs = init_states_ptr + (
        pid_b * stride_init_states_b + pid_h * stride_init_states_h +
        offs_d[:, None] * stride_init_states_d +
        offs_s[None, :] * stride_init_states_s)
    prev_state = tl.load(init_states_ptrs,
                         mask=(mask_d[:, None] & mask_s[None, :]),
                         other=0.0)
    prev_state = prev_state.to(tl.float32)

    # 1. scan states along nblocks dimension
    #    redundantly computed in thread blocks of the same sequence
    #    NOTE: this introduces load imbalance among thread blocks
    for t in range(block_start, pid_n + 1):
        # NOTE: store backs are distributed among blocks of the same sequence
        if (t == pid_n):  # my turn to write prev_state
            if IS_STORE_PREV_STATES:
                prev_states_ptrs = prev_states_ptr + (
                    pid_h * stride_prev_states_h +
                    offs_d[:, None] * stride_prev_states_d +
                    offs_s[None, :] * stride_prev_states_s)
                prev_states_ptrs_t = prev_states_ptrs + t * stride_prev_states_n
                tl.store(prev_states_ptrs_t,
                         prev_state,
                         mask=(mask_d[:, None] & mask_s[None, :]))
        # update state
        if (t < pid_n) or (t == (block_end - 1)):
            t_end = tl.load(block_cu_seqlens_ptr +
                            (t + 1) * stride_block_cu_seqlens_n)
            dA_cumsum_last = tl.load(
                dA_cumsum_ptr_h + (t_end - 1) * stride_dA_cumsum_t)  # scalar
            block_decay = tl.exp(dA_cumsum_last)

            block_states_ptrs_t = block_states_ptrs + t * stride_block_states_n
            state = tl.load(block_states_ptrs_t,
                            mask=(mask_d[:, None] & mask_s[None, :]),
                            other=0.0)
            state += block_decay * prev_state

            # store to final state if last block in sequence
            if t == (block_end - 1):
                final_states_ptrs = final_states_ptr + (
                    pid_b * stride_final_states_b +
                    pid_h * stride_final_states_h +
                    offs_d[:, None] * stride_final_states_d +
                    offs_s[None, :] * stride_final_states_s)
                tl.store(final_states_ptrs,
                         state,
                         mask=(mask_d[:, None] & mask_s[None, :]))
            elif (t < pid_n):
                # update prev_state only if it's not the last
                prev_state = state

    # compute base pointers
    x_ptr += pid_h * stride_x_h
    dt_ptr += pid_h * stride_dt_h
    CB_ptr += pid_n * stride_CB_n + pid_g * stride_CB_g
    C_ptr += pid_g * stride_C_g
    output_ptr += pid_h * stride_output_h

    # full block loads
    x_ptrs = x_ptr + (
        (t_start + offs_t)[:, None] * stride_x_t + offs_d[None, :] * stride_x_d
    )  # (block_size, headdim)
    dt_ptrs = dt_ptr + ((t_start + offs_t) * stride_dt_t)  # (block_size,)
    dA_cumsum_ptrs = dA_cumsum_ptr_h + (
        (t_start + offs_t) * stride_dA_cumsum_t)  # (block_size,)

    x = tl.load(x_ptrs, mask=(mask_t[:, None] & mask_d[None, :]), other=0.0)
    dt = tl.load(dt_ptrs, mask=mask_t, other=0.0)
    dA_cumsum = tl.load(dA_cumsum_ptrs, mask=mask_t, other=0.0)
    D = tl.load(D_ptr + pid_h * stride_D_h)

    # NOTE: tt tiles along rows of CB, but keeping the columns full
    #       block_size. This is important and provide significant
    #       performance boost over v0 for full sized blocks
    for tt in range(0, block_size, BLOCK_SIZE_TT):
        mask_tt = offs_tt < (ntokens - tt)

        # 2. compute output
        # NOTE: prev_state tensor from the previous step is reused
        dA_cumsum_tt_ptrs = dA_cumsum_ptr_h + (
            (t_start + tt + offs_tt) * stride_dA_cumsum_t)  # (BLOCK_SIZE_TT,)

        # NOTE: need to reload slices of x and dA_cumsum even though they were
        #       loaded already due to lacking ways to slice tensors in triton
        x_tt_ptrs = x_ptr + (
            (t_start + tt + offs_tt)[:, None] * stride_x_t +
            offs_d[None, :] * stride_x_d)  # (BLOCK_SIZE_TT, headdim)
        dA_cumsum_tt = tl.load(dA_cumsum_tt_ptrs, mask=mask_tt, other=0.0)
        x_tt = tl.load(x_tt_ptrs,
                       mask=(mask_tt[:, None] & mask_d[None, :]),
                       other=0.0)
        CB_ptrs = CB_ptr + (
            (tt + offs_tt)[:, None] * stride_CB_t0 +
            offs_t[None, :] * stride_CB_t1)  # (BLOCK_SIZE_TT, block_size)

        # 2.1 compute diagonal output contribution
        CB = tl.load(
            CB_ptrs)  # ok to not have mask - causal mask applied later

        seg_sum = dA_cumsum_tt[:, None] - dA_cumsum[
            None, :]  #(BLOCK_SIZE_TT, block_size)
        decay = tl.exp(seg_sum)
        # decay = fast_expf(seg_sum)

        scores = tl.where(((tt + offs_tt)[:, None] >= offs_t[None, :]),
                          CB * decay * dt[None, :], 0.0)
        # lower precision (prevents out of resource compile error on H100)
        scores = scores.to(x_ptr.dtype.element_ty)
        out_diag = tl.dot(scores, x)  # (BLOCK_SIZE_TT, headdim)

        # 2.2 compute off-diagonal block contributions to the output
        C_ptrs = C_ptr + (
            (t_start + tt + offs_tt)[:, None] * stride_C_t +
            offs_s[None, :] * stride_C_s)  # (BLOCK_SIZE_TT, dstate)
        C = tl.load(C_ptrs,
                    mask=(mask_tt[:, None] & mask_s[None, :]),
                    other=0.0).to(tl.float32)
        out_off = (tl.dot(C, prev_state.T) * tl.exp(dA_cumsum_tt[:, None])
                   )  #(BLOCK_SIZE_TT, headdim)

        # 2.3 sum up contributions
        out = out_diag + out_off  # (BLOCK_SIZE_TT, headdim)

        # 2.4 optionally for D
        out += x_tt * D

        out_ptrs = output_ptr + (
            (t_start + tt + offs_tt)[:, None] * stride_output_t +
            offs_d[None, :] * stride_output_d)
        tl.store(out_ptrs, out, mask=(mask_tt[:, None] & mask_d[None, :]))


# Fused block SSD performs inter-block scan and final output activation
# for now compute state passing results
# output:
# final_states (batch, nheads, headdim, dstate)
# output (seqlen, nheads, headdim)
# prev_states (nblocks, nheads, headdim, dstate) # Optional for debug
def fused_block_scan(
    x,  # (seqlen, nheads, headdim)
    dt,  # (nheads, seqlen)
    dA_cumsum,  # (nheads, seqlen)
    block_states,  # (nblocks, block_size, headdim, dstate)
    initial_states,  # (batch, nheads, headdim, dstate)
    C,  # (seqlen, ngroups, dstate)
    D,
    CB,  # (nblocks, ngroups, block_size, block_size)
    # metadata
    block_cu_seqlens,  # (nblocks+1,)
    block_req_idx,  # (nblocks, )
    req_cu_nblocks,  # (batch+1, )
    block_size_tt=16,
    return_prev_states=False,
):
    seqlen, nheads, headdim = x.shape
    ngroups = C.shape[1]
    dstate = C.shape[-1]
    block_size = CB.shape[-1]
    nblocks = block_states.shape[0]
    batch = initial_states.shape[0]

    assert dt.shape == (nheads, seqlen)
    assert dA_cumsum.shape == (nheads, seqlen)
    assert C.shape == (seqlen, ngroups, dstate)
    assert D.shape == (nheads, )
    assert CB.shape == (nblocks, ngroups, block_size, block_size)
    assert block_states.shape == (nblocks, nheads, headdim, dstate)
    assert initial_states.shape == (batch, nheads, headdim, dstate)
    assert block_cu_seqlens.shape == (nblocks + 1, )
    assert block_req_idx.shape == (nblocks, )
    assert req_cu_nblocks.shape == (batch + 1, )

    device = x.device

    # Allocate outputs
    prev_states = (torch.empty_like(block_states)
                   if return_prev_states else None)
    final_states = torch.empty((batch, nheads, headdim, dstate),
                               dtype=block_states.dtype,
                               device=device)
    output = torch.empty_like(x)
    prev_state_strides = ((0, 0, 0, 0) if prev_states is None else
                          (prev_states.stride(0), prev_states.stride(1),
                           prev_states.stride(2), prev_states.stride(3)))
    # Launch grid
    # NOTE: parallelizing along nblocks requires that state passing scan
    #       computed redundantly among blocks of the same sequence.
    #       Alternatively, we can implement parallel scan if it brings
    #       significant performance difference
    # TODO: need to find a good decomposition strategy for max parallelism
    grid = lambda META: (nblocks, nheads,
                         triton.cdiv(headdim, META['BLOCK_SIZE_D']))

    with torch.cuda.device(x.device.index):
        fused_block_scan_v1_kernel[grid](
            x_ptr=x,
            dt_ptr=dt,
            dA_cumsum_ptr=dA_cumsum,
            block_states_ptr=block_states,
            init_states_ptr=initial_states,
            C_ptr=C,
            D_ptr=D,
            CB_ptr=CB,
            block_cu_seqlens_ptr=block_cu_seqlens,
            block_req_idx_ptr=block_req_idx,
            req_cu_nblocks_ptr=req_cu_nblocks,
            prev_states_ptr=prev_states,
            final_states_ptr=final_states,
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
            stride_D_h=D.stride(0),
            stride_CB_n=CB.stride(0),
            stride_CB_g=CB.stride(1),
            stride_CB_t0=CB.stride(2),
            stride_CB_t1=CB.stride(3),
            stride_block_cu_seqlens_n=block_cu_seqlens.stride(0),
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
            stride_output_t=output.stride(0),
            stride_output_h=output.stride(1),
            stride_output_d=output.stride(2),
            BLOCK_SIZE_S=max(dstate, 16),
            BLOCK_SIZE_TT=block_size_tt,
            IS_STORE_PREV_STATES=return_prev_states,
        )

    return final_states, output, prev_states
