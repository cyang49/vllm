# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E501,SIM102,SIM113

import torch

from vllm.triton_utils import tl, triton


# from .utils import generate_autotune_combinations
# @triton.autotune(
#     configs=generate_autotune_combinations(
#         spec={'BLOCK_SIZE_TT': [32, 64, 128],
#               'BLOCK_SIZE_D': [32, 64, 128],
#               'BLOCK_SIZE_S': [32, 64],
#               'BLOCK_SIZE_T0': [16, 32, 64],
#               'BLOCK_SIZE_T1': [16, 32, 64],
#               'num_warps': [2, 4],
#               'num_stages': [1, 2, 3,],
#              },
#         ),
#     key=[],
# )
#Triton autotuning for function fused_block_state_bmm_kernel finished after 126.16s; best config selected: BLOCK_SIZE_TT: 64, BLOCK_SIZE_D: 64, BLOCK_SIZE_S: 64, BLOCK_SIZE_T0: 16, BLOCK_SIZE_T1: 16, num_warps: 2, num_ctas: 1, num_stages: 3, num_buffers_warp_spec: 0, num_consumer_groups: 0, reg_dec_producer: 0, reg_inc_consumer: 0, maxnreg: None;
# Best found on H100 # fused
@triton.autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_TT': 64,
                'BLOCK_SIZE_D': 64,
                'BLOCK_SIZE_S': 64,
                'BLOCK_SIZE_T0': 16,
                'BLOCK_SIZE_T1': 16,
            },
            num_warps=2,
            num_stages=3),
    ],
    key=[],
)
@triton.jit
def fused_block_state_bmm_kernel(
    # Inputs
    x_ptr,
    dt_ptr,
    B_ptr,
    C_ptr,
    block_cu_seqlens_ptr,
    block_packed_cu_seqlens_ptr,
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
    stride_dt_h,
    stride_dt_t: tl.constexpr,
    stride_dA_cumsum_h,
    stride_dA_cumsum_t: tl.constexpr,
    stride_B_t: tl.constexpr,
    stride_B_g: tl.constexpr,
    stride_B_s: tl.constexpr,
    stride_C_t: tl.constexpr,
    stride_C_g: tl.constexpr,
    stride_C_s: tl.constexpr,
    stride_block_cu_seqlens_n: tl.constexpr,
    stride_block_packed_cu_seqlens_n: tl.constexpr,
    stride_block_states_n: tl.constexpr,
    stride_block_states_h: tl.constexpr,
    stride_block_states_d: tl.constexpr,
    stride_block_states_s: tl.constexpr,
    stride_CB_n: tl.constexpr,
    stride_CB_g: tl.constexpr,
    stride_CB_t0: tl.constexpr,
    stride_CB_t1: tl.constexpr,
    # Meta-parameters
    FUSED_COMPUTE_CB: tl.constexpr,
    ALIGN_BLOCKS: tl.constexpr,
    # finer grain decomposition of block size dimension
    BLOCK_SIZE_TT: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
    BLOCK_SIZE_T0: tl.constexpr,
    BLOCK_SIZE_T1: tl.constexpr,
):
    tl.static_assert(dstate >= BLOCK_SIZE_S)
    pid_ds = tl.program_id(0)
    nsblocks = tl.cdiv(dstate, BLOCK_SIZE_S)
    pid_d = pid_ds // nsblocks
    pid_s = pid_ds % nsblocks

    pid_n = tl.program_id(1)  # block idx
    pid_h = tl.program_id(2)  # head idx
    pid_g = pid_h // nheads_ngroups_ratio  # group idx

    offs_tt = tl.arange(0, BLOCK_SIZE_TT)

    offs_d = pid_d * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    offs_s = pid_s * BLOCK_SIZE_S + tl.arange(0, BLOCK_SIZE_S)

    # Load block start and end offset
    t_start = tl.load(block_cu_seqlens_ptr + pid_n * stride_block_cu_seqlens_n)
    t_end = tl.load(block_cu_seqlens_ptr +
                    (pid_n + 1) * stride_block_cu_seqlens_n)
    ntokens = t_end - t_start  # number of tokens in this block
    align_t_start, align_t_end = t_start, t_end
    if ALIGN_BLOCKS:
        align_t_start = tl.load(block_packed_cu_seqlens_ptr +
                                pid_n * stride_block_packed_cu_seqlens_n)
        align_t_end = tl.load(block_packed_cu_seqlens_ptr +
                              (pid_n + 1) * stride_block_packed_cu_seqlens_n)
        tl.multiple_of(align_t_start, 4)
        tl.multiple_of(align_t_end, 4)
    align_ntokens = align_t_end - align_t_start

    # Mask out-of-bound tokens
    mask_d = offs_d < headdim
    mask_s = offs_s < dstate

    # Compute base pointer addresses
    x_ptr_base = x_ptr + t_start * stride_x_t + pid_h * stride_x_h
    dt_ptr_base = dt_ptr + align_t_start * stride_dt_t + pid_h * stride_dt_h
    dA_cumsum_ptr_base = dA_cumsum_ptr + align_t_start * stride_dA_cumsum_t + pid_h * stride_dA_cumsum_h
    B_ptr_base = B_ptr + t_start * stride_B_t + pid_g * stride_B_g

    # Load last element value from dA_cumsum block
    dA_cs_last = tl.load(dA_cumsum_ptr_base +
                         (ntokens - 1) * stride_dA_cumsum_t)

    # 2. Compute block states
    acc = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_S), dtype=tl.float32)
    # In a for loop, process tokens within the length bound (ntokens)
    for tt in range(0, ntokens, BLOCK_SIZE_TT):
        mask_tt = offs_tt < (ntokens - tt)
        aligned_mask_tt = offs_tt < (align_ntokens - tt)
        x_ptr_tt = x_ptr_base + tt * stride_x_t
        dt_ptr_tt = dt_ptr_base + tt * stride_dt_t
        dA_cumsum_ptr_tt = dA_cumsum_ptr_base + tt * stride_dA_cumsum_t
        B_ptr_tt = B_ptr_base + tt * stride_B_t

        x_ptrs = x_ptr_tt + offs_d[:, None] * stride_x_d + offs_tt[
            None, :] * stride_x_t  # (headdim, BLOCK_SIZE_TT)
        B_ptrs = B_ptr_tt + offs_tt[:, None] * stride_B_t + offs_s[
            None, :] * stride_B_s  # (BLOCK_SIZE_TT, dstate)
        dt_ptrs = dt_ptr_tt + offs_tt * stride_dt_t  # (BLOCK_SIZE_TT,)
        dA_cumsum_ptrs = dA_cumsum_ptr_tt + offs_tt * stride_dA_cumsum_t  # (BLOCK_SIZE_TT,)
        dt = tl.where(mask_tt, tl.load(dt_ptrs,
                                       mask=aligned_mask_tt,
                                       other=0.0), 0.0)
        dA_cs = tl.where(
            mask_tt, tl.load(dA_cumsum_ptrs, mask=aligned_mask_tt, other=0.0),
            0.0)
        x_tt = tl.load(x_ptrs,
                       mask=(mask_d[:, None] & mask_tt[None, :]),
                       other=0.0)
        B_tt = tl.load(B_ptrs,
                       mask=(mask_tt[:, None] & mask_s[None, :]),
                       other=0.0)

        # Compute decay from dt and dA_cs
        decay_states = tl.exp(dA_cs_last - dA_cs) * dt
        B_decay = B_tt * decay_states[:, None]

        # it helps speed to perform dot at lower precision
        acc += tl.dot(x_tt, B_decay.to(x_ptr.dtype.element_ty))

    # Store back
    block_states_ptr_base = block_states_ptr + (pid_n * stride_block_states_n +
                                                pid_h * stride_block_states_h)
    block_states_ptrs = block_states_ptr_base + \
        offs_d[:,None] * stride_block_states_d + \
        offs_s[None, :] * stride_block_states_s
    tl.store(block_states_ptrs, acc, mask=(mask_d[:, None] & mask_s[None, :]))

    if FUSED_COMPUTE_CB:
        # Compute CB matrix per group
        # C @ B.T is independent of prior computations and can be
        # parallelized differently. It's effectively a batched matmul over
        # (nblocks × ngroups) of square matrices (block_size × block_size).
        # Here we use 2d output decomposition on CB submatrix and distribute
        # work of 1 group among (nheads_ngroups_ratio x ndsblocks) programs
        # TODO: add assertions for sanity check
        ndsblocks = tl.num_programs(0)
        ntcblocks = tl.cdiv(block_size, BLOCK_SIZE_T0)
        ntbblocks = tl.cdiv(block_size, BLOCK_SIZE_T1)
        # assert (ndsblocks *
        #         nheads_ngroups_ratio) == (ntcblocks *
        #                                   ntbblocks), "parallelism check"

        # map from (nheads, ndsblocks) to index space of (ngroups, ntcblocks, ntbblocks)
        nheads = tl.num_programs(2)
        ngroups = nheads // nheads_ngroups_ratio
        num_total_programs = (nheads * ndsblocks)
        ratio = num_total_programs // (ngroups * ntcblocks * ntbblocks)

        # skip no task assigned
        if num_total_programs % ratio != 0:
            return

        flat_idx = (pid_h * ndsblocks + pid_ds) // ratio
        t0_bidx = flat_idx // ntbblocks
        t1_bidx = flat_idx % ntbblocks

        t0_off = t0_bidx * BLOCK_SIZE_T0  # Row
        t1_off = t1_bidx * BLOCK_SIZE_T1  # Column

        offs_t0 = t0_off + tl.arange(0, BLOCK_SIZE_T0)
        offs_t1 = t1_off + tl.arange(0, BLOCK_SIZE_T1)

        mask_t0 = offs_t0 < ntokens
        mask_t1 = offs_t1 < ntokens

        C_ptr_base = C_ptr + t_start * stride_C_t + pid_g * stride_C_g

        cb = tl.zeros((BLOCK_SIZE_T0, BLOCK_SIZE_T1), dtype=tl.float32)

        for ss in tl.range(0, dstate, BLOCK_SIZE_S):
            offs_ss = ss + tl.arange(0, BLOCK_SIZE_S)
            mask_ss = offs_ss < dstate

            # (BLOCK_SIZE_T1, BLOCK_SIZE_S)
            B_ptrs = B_ptr_base + (offs_t1[:, None] * stride_B_t +
                                   offs_ss[None, :] * stride_B_s)
            # (BLOCK_SIZE_T0, BLOCK_SIZE_S)
            C_ptrs = C_ptr_base + (offs_t0[:, None] * stride_C_t +
                                   offs_ss[None, :] * stride_C_s)

            C = tl.load(C_ptrs,
                        mask=(mask_t0[:, None] & mask_ss[None, :]),
                        other=0.0)
            B = tl.load(B_ptrs,
                        mask=(mask_t1[:, None] & mask_ss[None, :]),
                        other=0.0)
            cb += tl.dot(C, B.T)  # (BLOCK_SIZE_T0, BLOCK_SIZE_T1)

        # (BLOCK_SIZE_T0, BLOCK_SIZE_T1)
        CB_ptrs = CB_ptr + (pid_n * stride_CB_n + pid_g * stride_CB_g +
                            offs_t0[:, None] * stride_CB_t0 +
                            offs_t1[None, :] * stride_CB_t1)
        tl.store(CB_ptrs, cb, mask=(mask_t0[:, None] & mask_t1[None, :]))


def fused_block_state_bmm(
    x,  # (seqlen, nheads, headdim)
    dt,  # (nheads, seqlen)
    dA_cumsum,  #(nheads, seqlen)
    B,  # (seqlen, ngroups, dstate)
    C,  # (seqlen, ngroups, dstate)
    block_size,
    # metadata
    block_cu_seqlens,  # (nblocks+1,)
    block_packed_cu_seqlens=None,  # (nblocks+1,)
    states_in_fp32=True,
    FUSED_COMPUTE_CB=True,
    align_blocks=False,
):
    seqlen, nheads, headdim = x.shape
    ngroups = B.shape[1]
    dstate = B.shape[-1]
    nblocks = block_cu_seqlens.shape[0] - 1

    aligned_seqlen = dt.shape[-1] if align_blocks else seqlen

    assert dt.shape == (nheads, aligned_seqlen)
    assert dA_cumsum.shape == (nheads, aligned_seqlen)
    assert C.shape == (seqlen, ngroups, dstate)
    assert C.shape == B.shape
    if align_blocks:
        assert block_packed_cu_seqlens.shape == block_cu_seqlens.shape

    device = x.device
    dtype = x.dtype

    # Allocate outputs
    # NOTE: block_states has nheads (and not ngroups), because each head in x
    #       is paired with a group head in B (many-to-few) resulting in nheads
    #       in the output states. This is different from GQA in regular
    #       transformer KV cache where grouped KV is cached
    block_states = torch.empty(
        (nblocks, nheads, headdim, dstate),
        dtype=torch.float32 if states_in_fp32 else dtype,
        device=device)

    CB = (torch.full((nblocks, ngroups, block_size, block_size),
                     float('-inf'),
                     dtype=torch.float32,
                     device=device) if FUSED_COMPUTE_CB else None)
    CB_strides = (0, 0, 0, 0) if CB is None else (CB.stride(0), CB.stride(1),
                                                  CB.stride(2), CB.stride(3))
    # Launch grid
    grid = lambda META: (triton.cdiv(headdim, META["BLOCK_SIZE_D"]) * triton.
                         cdiv(dstate, META["BLOCK_SIZE_S"]), nblocks, nheads)
    with torch.cuda.device(x.device.index):
        fused_block_state_bmm_kernel[grid](
            x_ptr=x,
            dt_ptr=dt,
            dA_cumsum_ptr=dA_cumsum,
            B_ptr=B,
            C_ptr=C,
            block_cu_seqlens_ptr=block_cu_seqlens,
            block_packed_cu_seqlens_ptr=block_packed_cu_seqlens,
            block_states_ptr=block_states,
            CB_ptr=CB,
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
            stride_B_t=B.stride(0),
            stride_B_g=B.stride(1),
            stride_B_s=B.stride(2),
            stride_C_t=C.stride(0),
            stride_C_g=C.stride(1),
            stride_C_s=C.stride(2),
            stride_block_cu_seqlens_n=block_cu_seqlens.stride(0),
            stride_block_packed_cu_seqlens_n=(block_packed_cu_seqlens.stride(0)
                                              if align_blocks else 0),
            stride_block_states_n=block_states.stride(0),
            stride_block_states_h=block_states.stride(1),
            stride_block_states_d=block_states.stride(2),
            stride_block_states_s=block_states.stride(3),
            stride_CB_n=CB_strides[0],
            stride_CB_g=CB_strides[1],
            stride_CB_t0=CB_strides[2],
            stride_CB_t1=CB_strides[3],
            FUSED_COMPUTE_CB=FUSED_COMPUTE_CB,
            ALIGN_BLOCKS=align_blocks,
        )

    return block_states, CB
