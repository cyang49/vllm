# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.triton_utils import tl, triton


# x:        input quantized tensor
# scale:    fp32 scalar
# return:   dequantized values in fp32
@triton.jit
def dequant_symmetric_per_tensor(x, scale):
    return x.to(tl.float32) * scale


# x:        input 2d quantized tensor [BLOCK_SIZE_M, BLOCK_SIZE_DSTATE]
# scale:    fp32 tensor of shape [BLOCK_SIZE_M, ngroups_per_row]
# return:   dequantized values in fp32
@triton.jit
def dequant_symmetric_per_group(x, scales, group_size):
    tl.static_assert(x.shape[0] == scales.shape[0])

    # reshape to [total_ngroups, group_size]
    x_reshaped = x.reshape((x.shape[0] * x.shape[1]) // group_size, group_size)
    scales_reshaped = scales.reshape(scales.shape[0] * scales.shape[1], 1)
    return (x_reshaped.to(tl.float32) * scales_reshaped).reshape(x.shape)


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
# return:   dequantized values in fp32
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
    return (
        x_clipped.to(tl.float8e4nv).reshape(x.shape),
        scale.reshape(x.shape[0], x.shape[1] // group_size),
    )
