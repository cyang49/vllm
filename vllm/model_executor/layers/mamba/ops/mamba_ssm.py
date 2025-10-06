# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/selective_state_update.py

import torch
from packaging import version

from vllm import _custom_ops as ops
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.triton_utils import HAS_TRITON, tl, triton

from .quantization import (
    dequant_symmetric_per_group,
    dequant_symmetric_per_tensor,
    quant_symmetric_per_group_fp8e4nv,
    quant_symmetric_per_tensor_fp8e4nv,
)

TRITON3 = HAS_TRITON and (version.parse(triton.__version__) >= version.parse("3.0.0"))

if TRITON3:

    @triton.jit
    def softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)
        return dt
else:

    @triton.jit
    def softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
        return dt


@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics(
    {
        "HAS_STATE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"]
        is not None
    }
)
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
@triton.jit
def _selective_scan_update_kernel(
    # Pointers to matrices
    state_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    state_batch_indices_ptr,
    dst_state_batch_indices_ptr,
    scales_ptr,
    # shared_quant_ratio: tl.constexpr,
    quant_group_size: tl.constexpr,
    pad_slot_id: tl.constexpr,
    # Matrix dimensions
    dim: tl.constexpr,
    dstate: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # Strides
    stride_state_batch: tl.int64,
    stride_state_head: tl.int64,
    stride_state_dim: tl.int64,
    stride_state_dstate: tl.constexpr,
    stride_x_batch: tl.int64,
    stride_x_head: tl.int64,
    stride_x_dim: tl.constexpr,
    stride_dt_batch: tl.int64,
    stride_dt_head: tl.int64,
    stride_dt_dim: tl.constexpr,
    stride_dt_bias_head: tl.int64,
    stride_dt_bias_dim: tl.constexpr,
    stride_A_head: tl.int64,
    stride_A_dim: tl.constexpr,
    stride_A_dstate: tl.constexpr,
    stride_B_batch: tl.int64,
    stride_B_group: tl.int64,
    stride_B_dstate: tl.constexpr,
    stride_C_batch: tl.int64,
    stride_C_group: tl.int64,
    stride_C_dstate: tl.constexpr,
    stride_D_head: tl.int64,
    stride_D_dim: tl.constexpr,
    stride_z_batch: tl.int64,
    stride_z_head: tl.int64,
    stride_z_dim: tl.constexpr,
    stride_out_batch: tl.int64,
    stride_out_head: tl.int64,
    stride_out_dim: tl.constexpr,
    stride_scales_batch: tl.int64,
    stride_scales_head: tl.int64,
    stride_scales_dim: tl.int64,
    stride_scales_group: tl.constexpr,
    # Meta-parameters
    IS_STATIC_SCALE: tl.constexpr,  # scaling factors are read-only
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_STATE_BATCH_INDICES: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    # If HAS_STATE_BATCH_INDICES is true, then the ssm state's batch coordinate
    # is taken from the state_batch_indices_ptr Otherwise, the state coordinate
    # is the same as the batch id.
    if HAS_STATE_BATCH_INDICES:
        dst_state_batch_indices_ptr += pid_b
        dst_state_batch_idx = tl.load(dst_state_batch_indices_ptr).to(tl.int64)
        dst_state_ptr = state_ptr + (
            dst_state_batch_idx * stride_state_batch + pid_h * stride_state_head
        )
        state_batch_indices_ptr += pid_b
        state_batch_idx = tl.load(state_batch_indices_ptr).to(tl.int64)
        state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
    else:
        dst_state_ptr = (
            state_ptr + pid_b * stride_state_batch + pid_h * stride_state_head
        )
        state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head

    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    B_ptr += pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch + pid_h * stride_z_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (
        offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    )
    dst_state_ptrs = dst_state_ptr + (
        offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    )
    x_ptrs = x_ptr + offs_m * stride_x_dim
    dt_ptrs = dt_ptr + offs_m * stride_dt_dim
    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        D_ptr += pid_h * stride_D_head
    A_ptrs = A_ptr + (
        offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate
    )
    B_ptrs = B_ptr + offs_n * stride_B_dstate
    C_ptrs = C_ptr + offs_n * stride_C_dstate
    if HAS_D:
        D_ptrs = D_ptr + offs_m * stride_D_dim
    if HAS_Z:
        z_ptrs = z_ptr + offs_m * stride_z_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)
    if HAS_STATE_BATCH_INDICES:
        mask &= state_batch_idx != pad_slot_id
    state = tl.load(state_ptrs, mask=mask, other=0.0)

    # Dequantization if state is fp8

    if tl.constexpr(state_ptr.dtype.element_ty == tl.float8e4nv):
        if quant_group_size == -1:  # per-token or per-head
            fp8_scale = tl.load(
                scales_ptr + pid_b * stride_scales_batch + pid_h * stride_scales_head
            )
            state = dequant_symmetric_per_tensor(state, fp8_scale)
        else:
            # BLOCK_SIZE_DSTATE can be a power of 2 value >= dstate
            tl.static_assert(BLOCK_SIZE_DSTATE % quant_group_size == 0)
            # ngroups_per_row = tl.cdiv(BLOCK_SIZE_DSTATE, quant_group_size)
            offs_ngroups = tl.arange(0, (BLOCK_SIZE_DSTATE // quant_group_size))
            # fp8_scale: [BLOCK_SIZE_M, ngroups_per_row]
            fp8_scale = tl.load(
                scales_ptr
                + pid_b * stride_scales_batch
                + pid_h * stride_scales_head
                + offs_m[:, None] * stride_scales_dim
                + offs_ngroups[None, :] * stride_scales_group,
                mask=(
                    (offs_m[:, None] < dim)
                    & (offs_ngroups[None, :] < (dstate // quant_group_size))
                ),
                other=0.0,
            )
            state = dequant_symmetric_per_group(state, fp8_scale, quant_group_size)

    x = tl.load(x_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    if not TIE_HDIM:
        dt = tl.load(dt_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if HAS_DT_BIAS:
            dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if DT_SOFTPLUS:
            dt = softplus(dt)
        A = tl.load(
            A_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0
        ).to(tl.float32)
        dA = tl.exp(A * dt[:, None])
    else:
        dt = tl.load(dt_ptr).to(tl.float32)
        if HAS_DT_BIAS:
            dt += tl.load(dt_bias_ptr).to(tl.float32)
        if DT_SOFTPLUS:
            dt = softplus(dt)
        A = tl.load(A_ptr).to(tl.float32)
        dA = tl.exp(A * dt)  # scalar, not a matrix

    B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
    if HAS_D:
        D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
    if HAS_Z:
        z = tl.load(z_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

    dB = B[None, :] * dt[:, None] if not TIE_HDIM else B * dt
    state = state * dA + dB * x[:, None]

    mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)
    if HAS_STATE_BATCH_INDICES:
        mask &= state_batch_idx != pad_slot_id

    out = tl.sum(state * C[None, :], axis=1)
    if HAS_D:
        out += x * D
    if HAS_Z:
        out *= z * tl.sigmoid(z)

    tl.store(out_ptrs, out, mask=offs_m < dim)

    # Quantization before store back
    if tl.constexpr(state_ptr.dtype.element_ty == tl.float8e4nv):
        if IS_STATIC_SCALE:
            if quant_group_size == -1:
                state, _ = quant_symmetric_per_tensor_fp8e4nv(state, fp8_scale)
            else:
                state, _ = quant_symmetric_per_group_fp8e4nv(
                    state, quant_group_size, fp8_scale
                )
        else:  # dynamic
            tl.static_assert(quant_group_size != -1)
            state, scale = quant_symmetric_per_group_fp8e4nv(state, quant_group_size)
            tl.store(
                scales_ptr
                + pid_b * stride_scales_batch
                + pid_h * stride_scales_head
                + offs_m[:, None] * stride_scales_dim
                + offs_ngroups[None, :] * stride_scales_group,
                scale,
                mask=(
                    (offs_m[:, None] < dim)
                    & (offs_ngroups[None, :] < (dstate // quant_group_size))
                ),
            )

    tl.store(dst_state_ptrs, state, mask=mask)


def selective_state_update(
    state,
    x,
    dt,
    A,
    B,
    C,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    state_batch_indices=None,
    dst_state_batch_indices=None,
    pad_slot_id=PAD_SLOT_ID,
    out=None,
    is_fp8_static_scale=None,
    fp8_scales=None,
):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, dim) or (batch, nheads, dim)
        dt: (batch, dim) or (batch, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, dstate) or (batch, ngroups, dstate)
        C: (batch, dstate) or (batch, ngroups, dstate)
        D: (dim,) or (nheads, dim)
        z: (batch, dim) or (batch, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
        pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
        is_fp8_static_scale: determine whether fp8 scales are static
        fp8_scales: fp8 scaling factors. In-place updated if dynamic.
                    For dynamic quantization, we assume that the
                    quantization granularity matches the parallelization
                    granularity exactly. Runtime error is produced if it
                    doesn't hold
        out: Preallocated ssm output tensor. Assume same shape as x.
             In-place updated.
    """
    NO_HEADS = x.dim() == 2

    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    if out.dim() == 2:
        out = out.unsqueeze(1)

    _, nheads, dim, dstate = state.shape
    batch = x.shape[0]

    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
    if state_batch_indices is not None:
        assert state_batch_indices.shape == (batch,)
    if dst_state_batch_indices is not None:
        assert dst_state_batch_indices.shape == (batch,)
    else:
        # revert to the default behavior of in-place state updates
        dst_state_batch_indices = state_batch_indices
    assert out.shape == x.shape

    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE_M"]), batch, nheads)
    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)
    dt_bias_strides = (
        (dt_bias.stride(0), dt_bias.stride(1)) if dt_bias is not None else (0, 0)
    )
    D_strides = (D.stride(0), D.stride(1)) if D is not None else (0, 0)

    # We don't want autotune since it will overwrite the state
    # We instead tune by hand.
    # BLOCK_SIZE_M, num_warps = ((32, 4) if dstate <= 16 else
    #                            ((16, 4) if dstate <= 32 else
    #                             ((8, 4) if dstate <= 64 else
    #                              ((4, 4) if dstate <= 128 else ((4, 8))))))
    BLOCK_SIZE_M, num_warps, num_stages = (
        ((16, 2, 2) if batch > 64 else (16, 4, 2))
        if state.dtype != torch.float8_e4m3fn
        else ((16, 2, 2) if batch > 64 else (32, 2, 1))
    )

    # FP8 quantization metadata
    scales_strides = [0 for _ in range(state.ndim)]
    quant_group_size = -1
    if state.dtype == torch.float8_e4m3fn:
        assert is_fp8_static_scale is not None
        assert (fp8_scales is not None) and (fp8_scales.ndim > 0)

        if NO_HEADS:
            fp8_scales.unsqueeze_(1)

        for i in range(fp8_scales.ndim):
            scales_strides[i] = fp8_scales.stride(i)

        if fp8_scales.ndim == state.ndim:  # per group
            assert fp8_scales.ndim == 4
            assert (fp8_scales.shape[0], fp8_scales.shape[1], fp8_scales.shape[2]) == (
                batch,
                nheads,
                dim,
            )
            quant_ngroups = fp8_scales.shape[-1]
            quant_group_size = dstate // quant_ngroups
            # Simplifying assumption for quant_group_size
            assert quant_group_size <= dstate
            assert dstate % quant_group_size == 0
        else:
            assert fp8_scales.ndim == 1 or fp8_scales.ndim == 2
            # No support for dynamic quantization across thread blocks
            assert is_fp8_static_scale, (
                "must be static scale for per-tensor or per-head quantization"
            )

            if fp8_scales.ndim == 1:  # per batch
                assert fp8_scales.shape == (batch,)
            elif fp8_scales.ndim == 2:  # per head
                assert fp8_scales.shape == (batch, nheads)

    tie_hdim = (
        A.stride(-1) == 0
        and A.stride(-2) == 0
        and dt.stride(-1) == 0
        and dt_bias.stride(-1) == 0
    )
    with torch.cuda.device(x.device.index):
        _selective_scan_update_kernel[grid](
            state_ptr=state,
            x_ptr=x,
            dt_ptr=dt,
            dt_bias_ptr=dt_bias,
            A_ptr=A,
            B_ptr=B,
            C_ptr=C,
            D_ptr=D,
            z_ptr=z,
            out_ptr=out,
            state_batch_indices_ptr=state_batch_indices,
            dst_state_batch_indices_ptr=dst_state_batch_indices,
            pad_slot_id=pad_slot_id,
            dim=dim,
            dstate=dstate,
            nheads_ngroups_ratio=nheads // ngroups,
            scales_ptr=fp8_scales,
            # shared_quant_ratio=shared_quant_ratio,
            quant_group_size=quant_group_size,
            stride_state_batch=state.stride(0),
            stride_state_head=state.stride(1),
            stride_state_dim=state.stride(2),
            stride_state_dstate=state.stride(3),
            stride_x_batch=x.stride(0),
            stride_x_head=x.stride(1),
            stride_x_dim=x.stride(2),
            stride_dt_batch=dt.stride(0),
            stride_dt_head=dt.stride(1),
            stride_dt_dim=dt.stride(2),
            stride_dt_bias_head=dt_bias_strides[0],
            stride_dt_bias_dim=dt_bias_strides[1],
            stride_A_head=A.stride(0),
            stride_A_dim=A.stride(1),
            stride_A_dstate=A.stride(2),
            stride_B_batch=B.stride(0),
            stride_B_group=B.stride(1),
            stride_B_dstate=B.stride(2),
            stride_C_batch=C.stride(0),
            stride_C_group=C.stride(1),
            stride_C_dstate=C.stride(2),
            stride_D_head=D_strides[0],
            stride_D_dim=D_strides[1],
            stride_z_batch=z_strides[0],
            stride_z_head=z_strides[1],
            stride_z_dim=z_strides[2],
            stride_out_batch=out.stride(0),
            stride_out_head=out.stride(1),
            stride_out_dim=out.stride(2),
            stride_scales_batch=scales_strides[0],
            stride_scales_head=scales_strides[1],
            stride_scales_dim=scales_strides[2],
            stride_scales_group=scales_strides[3],
            IS_STATIC_SCALE=is_fp8_static_scale,
            DT_SOFTPLUS=dt_softplus,
            TIE_HDIM=tie_hdim,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            num_warps=num_warps,
            num_stages=num_stages,
        )


def selective_scan_fn(
    u,
    ssm_states,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    query_start_loc=None,
    cache_indices=None,
    has_initial_state=None,
    pad_slot_id=PAD_SLOT_ID,
    block_size=1024,
    block_idx_first_scheduled_token=None,
    block_idx_last_scheduled_token=None,
    initial_state_idx=None,
) -> torch.Tensor:
    """
    u: (dim, total_length) for varlen or (batch, dim, seqlen)
        applies changes in place.
    ssm_states: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        applies changes in place.
    delta: (dim, total_length) for varlen or (batch, dim, seqlen)
    A: (dim, dstate)
    B: (ngroups, dstate, total_length) for varlen or
                                        (batch,ngroups,dstate,seqlen)
    C: (ngroups, dstate, total_length) for varlen or
                                        (batch,ngroups,dstate,seqlen)
    D: (dim,)
    z: (dim, total_length) for varlen or (batch, dim, seqlen)
    dt_bias: (dim,) or (dim)
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended with 0.
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch) int32
        A tensor with each cell is a correspondent
        input and output ssm_state indices
      - Without APC: (batch,) - single state index per batch item
      - With APC: (batch, max_positions) - cache block indices for read/write
        Each non-zero value indicates a cache block to load from and/or write to.
    has_initial_state: (batch) bool
        A tensor populated with ones and zeros,
        indicate if the ssm_state at the corresponding index should be
        used as initial state. Not providing argument assumes
        there's no initial state
    pad_slot_id: int
        if cache_indices is passed, lets the kernel identify padding entries
        that will not be processed,
        for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
        in this case, the kernel will not process entries at indices 0 and 3
    block_size: int
        The block size to align the cached states to
    block_idx_first_scheduled_token: (batch,), dtype int32
        The pointer into cache_indices, where the first
        cache block to be filled is located.
    block_idx_last_scheduled_token: (batch,), dtype int32
        The pointer into cache_indices, where the last cache block
        to be filled is located.
    initial_state_idx: (batch,), dtype int32
        The pointer into cache_indices, where the cache block
        containing the initial state is located.
    returns
        output: (dim, total_length) for varlen or (batch, dim, seqlen)
                supports inplace replacement
    """
    if u.stride(-1) != 1:
        u = u.contiguous()
    if delta.stride(-1) != 1:
        delta = delta.contiguous()
    if D is not None:
        D = D.contiguous()
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if z is not None and z.stride(-1) != 1:
        z = z.contiguous()
    if B.dim() == 3 and query_start_loc is None:
        B = B.unsqueeze(1)
    if B.dim() == 2 and query_start_loc is not None:
        B = B.unsqueeze(0)
    if C.dim() == 3 and query_start_loc is None:
        C = C.unsqueeze(1)
    if C.dim() == 2 and query_start_loc is not None:
        C = C.unsqueeze(0)

    ops.selective_scan_fwd(
        u,
        delta,
        A,
        B,
        C,
        D,
        z,
        delta_bias,
        delta_softplus,
        query_start_loc,
        cache_indices,
        has_initial_state,
        ssm_states,
        pad_slot_id,
        block_size,
        block_idx_first_scheduled_token,
        block_idx_last_scheduled_token,
        initial_state_idx,
    )

    if z is None:
        return delta  # output written inplace to delta
    else:
        return z  # output written inplace to z
