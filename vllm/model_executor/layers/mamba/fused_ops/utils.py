# SPDX-License-Identifier: Apache-2.0
from itertools import product

from vllm.triton_utils import triton


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
