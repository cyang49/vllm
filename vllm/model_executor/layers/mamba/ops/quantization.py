# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.triton_utils import tl, triton


# x:        input quantized tensor
# scale:    fp32 scalar
# return:   dequantized values in fp32
@triton.jit
def dequant_symmetric_per_tensor(x, scale):
    return (x.to(tl.float32) * scale)


@triton.jit
def dynamic_quant_symmetric_per_tensor_fp8e4nv(x):
    # Compute scale
    max_val = tl.max(tl.abs(x))
    scale = max_val / 448.0
    scale = tl.where(scale == 0.0, 1.0, scale)  # Avoid div-by-zero

    # Quantize to float8e4nv
    x_scaled = x / scale
    x_clipped = tl.clamp(x_scaled, -448.0, 448.0)
    return x_clipped.to(tl.float8e4nv), scale


@triton.jit
def static_quant_symmetric_per_tensor_fp8e4nv(x, scale):
    # Quantize to float8e4nv
    x_scaled = x / scale
    x_clipped = tl.clamp(x_scaled, -448.0, 448.0)
    return x_clipped.to(tl.float8e4nv)
