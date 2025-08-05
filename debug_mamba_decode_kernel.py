# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import partial
from time import time

import torch
from torch.profiler import ProfilerActivity, profile

from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_state_update)

# generate dummy data for the test


def main(
        n_groups=1,
        dstate=128,
        nheads=128,
        headdim=64,
        dtype='bfloat16',
        state_dtype='bfloat16',
        device='cuda',
        # change this to see different times
        batch_size=256,
        has_initial_state: bool = True,
        repeats: int = 10,
        profile_on=False,
        is_fp8_static_scale: bool = True,
        quant_granularity: str = "request",  # ["request","head","group"]
        quant_ngroups: int = 4,  # number of "rows" in [hdim,dstate] as a group
):
    dtype = getattr(torch, dtype)
    state_dtype = getattr(torch, state_dtype)

    def generate_dummy_data(batch_size):

        #derived parameters
        # seq_len = chunk_prefill_size + batch_size - 1

        hidden_states = torch.randn(batch_size,
                                    nheads,
                                    headdim,
                                    dtype=dtype,
                                    device=device)
        A = torch.rand(nheads, dtype=dtype, device=device)
        B = torch.randn(batch_size, dstate, dtype=dtype, device=device)
        C = torch.randn(batch_size, dstate, dtype=dtype, device=device)
        D = torch.randn(nheads, dtype=dtype, device=device)
        dt = torch.randn(batch_size, nheads, dtype=dtype, device=device)
        dt_bias = torch.randn(nheads, dtype=dtype, device=device)
        state_indices_tensor = torch.arange(batch_size,
                                            dtype=torch.int32,
                                            device=device)

        if quant_granularity == "request":
            fp8_scales = torch.randn(batch_size, device=device, dtype=dtype)
        elif quant_granularity == "head":
            fp8_scales = torch.randn(batch_size,
                                     nheads,
                                     device=device,
                                     dtype=dtype)
        elif quant_granularity == "group":
            fp8_scales = torch.randn(batch_size,
                                     nheads,
                                     quant_ngroups,
                                     device=device,
                                     dtype=dtype)

        A = A[:, None, ...][:, :, None].expand(-1, headdim,
                                               dstate).to(dtype=torch.float32)
        dt = dt[:, :, None].expand(-1, -1, headdim)
        dt_bias = dt_bias[:, None, ...].expand(-1, headdim)
        D = D[:, None, ...].expand(-1, headdim)
        B = B.view(-1, n_groups, B.shape[1] // n_groups)
        C = C.view(-1, n_groups, C.shape[1] // n_groups)

        initial_states = (torch.randn(
            batch_size, nheads, headdim, dstate, dtype=dtype,
            device=device).to(state_dtype) if has_initial_state else None)

        return (
            hidden_states,
            initial_states,
            A,
            B,
            C,
            D,
            dt,
            dt_bias,
            state_indices_tensor,
            is_fp8_static_scale,
            fp8_scales,
        )

    (
        hidden_states,
        initial_states,
        A,
        B,
        C,
        D,
        dt,
        dt_bias,
        state_indices_tensor,
        is_fp8_static_scale,
        fp8_scales,
    ) = generate_dummy_data(batch_size)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    bench_fn = partial(
        selective_state_update,
        initial_states,
        hidden_states,
        dt,
        A,
        B,
        C,
        D,
        z=None,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_batch_indices=state_indices_tensor,
        is_fp8_static_scale=is_fp8_static_scale,
        fp8_scales=fp8_scales,
        out=torch.empty_like(hidden_states),
    )

    # warm up
    warmup_start = time()
    for _ in range(3):
        _ = bench_fn()
    print(f"warmup time {time()-warmup_start:.3f} seconds")

    start_time = time()

    if profile_on:
        with profile(activities=activities, with_stack=True) as prof:
            for _ in range(repeats):
                _ = bench_fn()

        filename = f"traces/mamba2_decode_trace_b{batch_size}.json"
        prof.export_chrome_trace(filename)
    else:

        for _ in range(repeats):
            _ = bench_fn()

    torch.cuda.synchronize()
    end_time = time()
    elapsed_time = (end_time - start_time) * 1000  # ms
    print(f"{elapsed_time=:.3f} mseconds")
    iter_time = elapsed_time / repeats  # ms
    print(f"{iter_time=:.3f} mseconds")


if __name__ == '__main__':
    import fire
    fire.Fire(main)
