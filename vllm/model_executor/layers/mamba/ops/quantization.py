# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.triton_utils import tl, triton


# x:        input quantized tensor
# scale:    fp32 scalar
# return:   dequantized values in out_dtype
@triton.jit
def dequant_symmetric_per_tensor(x, scale, out_dtype=tl.float32):
    return (x.to(tl.float32) * scale).to(out_dtype)


# x:        input 2d quantized tensor [BLOCK_SIZE_M, BLOCK_SIZE_DSTATE]
# scale:    fp32 tensor of shape [BLOCK_SIZE_M, ngroups_per_row]
# return:   dequantized values in out_dtype
@triton.jit
def dequant_symmetric_per_group(x, scales, group_size, out_dtype=tl.float32):
    tl.static_assert(x.shape[0] == scales.shape[0])

    # reshape to [total_ngroups, group_size]
    x_reshaped = x.reshape((x.shape[0] * x.shape[1]) // group_size, group_size)
    scales_reshaped = scales.reshape(scales.shape[0] * scales.shape[1], 1)
    return (x_reshaped.to(tl.float32) * scales_reshaped).reshape(
        x.shape).to(out_dtype)


@triton.jit
def quant_symmetric_per_tensor_fp8e4nv(x, scale=None):
    if scale is None:
        # Compute scale
        max_val = tl.max(tl.abs(x))
        scale = max_val / 448.0
        scale = tl.where(scale == 0.0, 1.0, scale)  # Avoid div-by-zero

    # Quantize to float8e4nv
    x_scaled = x / scale
    x_clipped = tl.clamp(x_scaled, -448.0, 448.0)
    return x_clipped.to(tl.float8e4nv), scale


# x:        input 2d quantized tensor [BLOCK_SIZE_M, BLOCK_SIZE_DSTATE]
# scale:    fp32 tensor of shape [BLOCK_SIZE_M, ngroups_per_row]
@triton.jit
def quant_symmetric_per_group_fp8e4nv(x, group_size, scale=None):
    # [total_ngroups, group_size]
    x_reshaped = x.reshape((x.shape[0] * x.shape[1]) // group_size, group_size)
    if scale is None:
        # Compute scale
        max_val = tl.max(tl.abs(x_reshaped), axis=1, keep_dims=True)
        scale = max_val / 448.0
        scale = tl.where(scale == 0.0, 1.0, scale)  # Avoid div-by-zero
    scale = scale.reshape(scale.shape[0] * scale.shape[1], 1)

    # Quantize to float8e4nv
    x_scaled = x_reshaped / scale
    x_clipped = tl.clamp(x_scaled, -448.0, 448.0)
    return (x_clipped.to(tl.float8e4nv).reshape(x.shape),
            scale.reshape(x.shape[0], x.shape[1] // group_size))


# Python helper func
def mamba_quant_helper(states_shape,
                       fp8_scales,
                       is_fp8_static_scale=True,
                       NO_HEADS=False):
    scales_strides = [0 for _ in range(len(states_shape))]
    quant_group_size = -1
    batch, nheads, headdim, _ = states_shape

    if fp8_scales is not None:
        assert fp8_scales.ndim > 0

        if NO_HEADS:
            fp8_scales.unsqueeze_(1)

        for i in range(fp8_scales.ndim):
            scales_strides[i] = fp8_scales.stride(i)

        if ((fp8_scales.ndim == 3 or fp8_scales.ndim == 4)
                and fp8_scales.ndim == len(states_shape)):  # per group
            assert (fp8_scales.shape[0], fp8_scales.shape[1]) == (batch,
                                                                  nheads)
            if fp8_scales.ndim == 4:
                assert fp8_scales.shape[2] == headdim

            quant_ngroups = fp8_scales.shape[-1]
            quant_group_size = states_shape[-1] // quant_ngroups
            # # Simplifying assumption for quant_group_size
            # assert quant_group_size <= dstate
            # assert dstate % quant_group_size == 0
        else:
            assert fp8_scales.ndim < 3

            assert is_fp8_static_scale, \
                "must be static scale for per-tensor or per-head quantization"

            if fp8_scales.ndim == 1:  # per batch
                assert fp8_scales.shape == (batch, )
            elif fp8_scales.ndim == 2:  # per head
                assert fp8_scales.shape == (batch, nheads)
    return scales_strides, quant_group_size
