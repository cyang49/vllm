# SPDX-License-Identifier: Apache-2.0
from itertools import product

from vllm.triton_utils import tl, triton


def generate_autotune_combinations(spec):
    """
    Generate all possible combinations of key-value pairs 
    given keys and their value ranges.

    Parameters:
    - keys: List of keys.
    - ranges: List of range objects or iterables corresponding to each key.

    Returns:
    - List of dictionaries, each representing a unique combination.
    """
    ranges = list(spec.values())
    keys = list(spec.keys())
    if len(keys) != len(ranges):
        raise ValueError("The number of keys must match the number of ranges.")

    value_combinations = product(*ranges)
    return [
        triton.Config(dict(zip(keys[:-2], values[:-2])), **{
            keys[-2]: values[-2],
            keys[-1]: values[-1],
        }) for values in value_combinations
    ]


@triton.jit
def load_t_offsets(n,
                   cu_seqlens_ptr,
                   stride_cu_seqlens_n,
                   ALIGNED=False,
                   PACK_SIZE=4):
    t_start = tl.load(cu_seqlens_ptr + n * stride_cu_seqlens_n)
    t_end = tl.load(cu_seqlens_ptr + (n + 1) * stride_cu_seqlens_n)

    if ALIGNED:
        tl.multiple_of(t_start, PACK_SIZE)
        tl.multiple_of(t_end, PACK_SIZE)

    ntokens = t_end - t_start

    return t_start, t_end, ntokens


@triton.jit
def load_with_aligned_mask(ptrs,
                           mask,
                           aligned_mask,
                           ALIGN_MASK=False,
                           other=0.0):
    return tl.load(ptrs, mask=mask, other=other) if not ALIGN_MASK else (
        tl.where(mask, tl.load(ptrs, mask=aligned_mask, other=other), other))
