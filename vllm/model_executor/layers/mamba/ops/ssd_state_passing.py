# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_state_passing.py

# ruff: noqa: E501

import torch

from vllm.triton_utils import tl, triton

from .quantization import (dequant_symmetric_per_group,
                           dequant_symmetric_per_tensor)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['dim'],
)
@triton.jit
def _state_passing_fwd_kernel(
    # Pointers to matrices
    states_ptr,
    out_ptr,
    dA_cs_ptr,
    initstates_ptr,
    seq_idx_ptr,
    scales_ptr,
    # Matrix dimensions
    dim: tl.constexpr,
    nchunks,
    seqlen,
    chunk_size: tl.constexpr,
    quant_group_size: tl.constexpr,
    # Strides
    stride_states_chunk: tl.int64,
    stride_states_head: tl.int64,
    stride_states_dim: tl.int64,
    stride_out_chunk: tl.int64,
    stride_out_head: tl.int64,
    stride_out_dim: tl.int64,
    stride_dA_cs_head,
    stride_dA_cs_chunk: tl.int64,
    stride_initstates_batch: tl.int64,
    stride_initstates_head: tl.int64,
    stride_initstates_dim: tl.int64,
    stride_seq_idx_seqlen: tl.int64,
    stride_scales_batch: tl.int64,
    stride_scales_head: tl.int64,
    stride_scales_group: tl.int64,
    # Meta-parameters
    HAS_INITSTATES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_STATIC_SCALE: tl.constexpr,
):
    pid_h = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=0)
    states_ptr += pid_h * stride_states_head
    dA_cs_ptr += pid_h * stride_dA_cs_head
    out_ptr += pid_h * stride_out_head
    if HAS_INITSTATES:
        initstates_ptr += pid_h * stride_initstates_head

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    states_ptrs = states_ptr + offs_m * stride_states_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim

    # - states will be the past state of the sequence that continues on the current check
    if not HAS_INITSTATES:
        states = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    else:
        initstates_ptr += offs_m * stride_initstates_dim
        initstates_ptrs = initstates_ptr
        # - for cont batches, for the first chunk mean it will be the first batch's
        #   init state
        states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0)

        # Dequantization if state is fp8
        if tl.constexpr(initstates_ptr.dtype.element_ty == tl.float8e4nv):
            if quant_group_size == -1:  # per-token or per-head
                fp8_scale = tl.load(scales_ptr + pid_h * stride_scales_head)
                states = dequant_symmetric_per_tensor(states, fp8_scale)
            else:
                tl.static_assert(BLOCK_SIZE % quant_group_size == 0)
                offs_ngroups = tl.arange(0, (BLOCK_SIZE // quant_group_size))
                fp8_scale = tl.load((scales_ptr + pid_h * stride_scales_head +
                                     offs_ngroups * stride_scales_group),
                                    mask=(offs_ngroups
                                          < (BLOCK_SIZE // quant_group_size)),
                                    other=0.0)  # [ngroups_per_row,]
                states = dequant_symmetric_per_group(states, fp8_scale,
                                                     quant_group_size)

        states = states.to(tl.float32)

    tl.store(out_ptrs, states, mask=offs_m < dim)
    out_ptrs += stride_out_chunk
    seq_idx = 0
    for c in range(nchunks - 1):
        new_states = tl.load(states_ptrs, mask=offs_m < dim,
                             other=0.0).to(tl.float32)
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        scale = tl.exp(dA_cs)

        # - the seq to pass forward is the one that is flushed to the right
        #   boundary.
        # - that is given by seq_idx_new below.
        seq_idx_new = tl.load(seq_idx_ptr +
                              (min((c + 1) * chunk_size, seqlen) - 1) *
                              stride_seq_idx_seqlen)
        if HAS_INITSTATES:
            if seq_idx != seq_idx_new:
                # this means in the current chunk the rightmost flushed seq
                # has changed.
                # - so we do not propagate the state from previous chunk
                # - but rather we load that sequence's init state
                initstates_ptrs = initstates_ptr + seq_idx_new * stride_initstates_batch

                # - update state with seq_idx_new's init state
                states = tl.load(initstates_ptrs, mask=offs_m < dim, other=0.0)
                # Dequantization if state is fp8
                if tl.constexpr(
                        initstates_ptr.dtype.element_ty == tl.float8e4nv):
                    if quant_group_size == -1:  # per-token or per-head
                        fp8_scale = tl.load(scales_ptr +
                                            seq_idx_new * stride_scales_batch +
                                            pid_h * stride_scales_head)
                        states = dequant_symmetric_per_tensor(
                            states, fp8_scale)
                    else:
                        tl.static_assert(BLOCK_SIZE % quant_group_size == 0)
                        offs_ngroups = tl.arange(
                            0, (BLOCK_SIZE // quant_group_size))
                        fp8_scale = tl.load(
                            (scales_ptr + seq_idx_new * stride_scales_batch +
                             pid_h * stride_scales_head +
                             offs_ngroups * stride_scales_group),
                            mask=(offs_ngroups
                                  < (BLOCK_SIZE // quant_group_size)),
                            other=0.0)  # [ngroups_per_row,]
                        states = dequant_symmetric_per_group(
                            states, fp8_scale, quant_group_size)
                states = states.to(tl.float32)
        else:
            scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)
        seq_idx = seq_idx_new

        states = scale * states + new_states
        tl.store(out_ptrs, states, mask=offs_m < dim)

        states_ptrs += stride_states_chunk
        dA_cs_ptr += stride_dA_cs_chunk
        out_ptrs += stride_out_chunk


def _state_passing_fwd(
        states,  # local states, read-only
        dA_chunk_cumsum,
        seq_idx,
        chunk_size,
        dstate,
        initial_states=None,  # cached mamba state
        is_fp8_static_scale: bool = None,  # fp8 initial_states
        fp8_scales=None,  # fp8 initial_states
        out_dtype=None,  # output state dtype
):
    nchunks, nheads, dim = states.shape
    assert dA_chunk_cumsum.shape == (nheads, nchunks)
    seqlen = seq_idx.shape[-1]

    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.empty((nchunks, nheads, dim),
                      device=states.device,
                      dtype=out_dtype)

    initial_states_strides = ((initial_states.stride(0),
                               initial_states.stride(1),
                               initial_states.stride(2))
                              if initial_states is not None else (0, 0, 0))
    scales_strides = [0, 0, 0]
    quant_group_size = -1
    if initial_states is not None:
        batch = initial_states.shape[0]
        initial_states_strides = (initial_states.stride(0),
                                  initial_states.stride(1),
                                  initial_states.stride(2))
        if initial_states.dtype == torch.float8_e4m3fn:
            assert is_fp8_static_scale is not None
            assert (fp8_scales is not None) and (fp8_scales.ndim > 0)

            for i in range(fp8_scales.ndim):
                scales_strides[i] = fp8_scales.stride(i)

            if fp8_scales.ndim == initial_states.ndim:  # per group
                assert fp8_scales.ndim == 3
                assert ((fp8_scales.shape[0], fp8_scales.shape[1],
                         fp8_scales.shape[2]) == (batch, nheads, dim))
                quant_ngroups = fp8_scales.shape[-1]
                quant_group_size = dstate // quant_ngroups
                # Simplifying assumption for quant_group_size
                assert quant_group_size <= dstate
                assert dim % quant_group_size == 0
            else:
                assert fp8_scales.ndim == 1 or fp8_scales.ndim == 2
                # No support for dynamic quantization across thread blocks
                assert is_fp8_static_scale, \
                    "must be static scale for per-tensor or per-head quantization"

                if fp8_scales.ndim == 1:  # per batch
                    assert fp8_scales.shape == (batch, )
                elif fp8_scales.ndim == 2:  # per head
                    assert fp8_scales.shape == (batch, nheads)
    else:
        initial_states_strides = (0, 0, 0)

    grid = lambda META: (triton.cdiv(dim, META['BLOCK_SIZE']), nheads)
    with torch.cuda.device(states.device.index):
        _state_passing_fwd_kernel[grid](
            states_ptr=states,
            out_ptr=out,
            dA_cs_ptr=dA_chunk_cumsum,
            initstates_ptr=initial_states,
            scales_ptr=fp8_scales,
            quant_group_size=quant_group_size,
            seq_idx_ptr=seq_idx,
            dim=dim,
            nchunks=nchunks,
            seqlen=seqlen if seq_idx is not None else 0,
            chunk_size=chunk_size if seq_idx is not None else 0,
            stride_states_chunk=states.stride(0),
            stride_states_head=states.stride(1),
            stride_states_dim=states.stride(2),
            stride_out_chunk=out.stride(0),
            stride_out_head=out.stride(1),
            stride_out_dim=out.stride(2),
            stride_dA_cs_head=dA_chunk_cumsum.stride(0),
            stride_dA_cs_chunk=dA_chunk_cumsum.stride(1),
            stride_initstates_batch=initial_states_strides[0],
            stride_initstates_head=initial_states_strides[1],
            stride_initstates_dim=initial_states_strides[2],
            stride_scales_batch=scales_strides[0],
            stride_scales_head=scales_strides[1],
            stride_scales_group=scales_strides[2],
            stride_seq_idx_seqlen=seq_idx.stride(0),
            IS_STATIC_SCALE=is_fp8_static_scale,
            HAS_INITSTATES=initial_states is not None,
        )
    return out
