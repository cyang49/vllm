# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.triton_utils import tl, triton


# x:        input quantized tensor
# scale:    fp32 scalar
# return:   dequantized values in fp32
@triton.jit
def dequant_symmetric_per_tensor(x, scale):
    return (x.to(tl.float32) * scale)


# x:        input 2d quantized tensor
# scale:    fp32 tensor of shape [ngroups]
# return:   dequantized values in fp32
@triton.jit
def dequant_symmetric_per_group(x, scales, group_size):
    tl.static_assert(len(x.shape) == 2)
    x_reshaped = x.reshape((x.shape[0] * x.shape[1]) // group_size, group_size)
    return (x_reshaped.to(tl.float32) * scales[:, None]).view(x.shape)


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


@triton.jit
def quant_symmetric_per_group_fp8e4nv(x, group_size, scale=None):
    # [ngroups, group_size]
    x_reshaped = x.reshape((x.shape[0] * x.shape[1]) // group_size, group_size)
    if scale is None:
        # Compute scale
        # [ngroups, ]
        max_val = tl.max(tl.abs(x_reshaped), axis=1)
        scale = max_val / 448.0
        scale = tl.where(scale == 0.0, 1.0, scale)  # Avoid div-by-zero

    # Quantize to float8e4nv
    x_scaled = x_reshaped / scale[:, None]
    x_clipped = tl.clamp(x_scaled, -448.0, 448.0)
    return x_clipped.to(tl.float8e4nv).reshape(x.shape), scale
