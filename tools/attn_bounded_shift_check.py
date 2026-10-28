#!/usr/bin/env python3
"""CPU/float64 algebra check for the OKC backlog's bounded-shift softmax.

Requires Python 3.10+ and NumPy. No CUDA or internet access is used.
This is NOT a GPU benchmark, an FP32/FP16 error bound, or a contest correctness test.
Logits are already in log2 units. Negative infinity denotes a masked entry.
"""
from __future__ import annotations

import json
import math
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def bounded_shift(x: Array, v: Array, tau: float, tile: int = 64) -> tuple[Array, int]:
    """Return normalized output and rescale count; empty partitions return zero."""
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if x.ndim != 1 or v.ndim != 2 or v.shape[0] != x.size:
        raise ValueError("Expected logits [N] and values [N, D].")
    if tile <= 0 or not math.isfinite(tau) or tau < 0:
        raise ValueError("tile must be positive; tau must be finite and nonnegative.")
    if np.isnan(x).any() or np.isposinf(x).any() or not np.isfinite(v).all():
        raise ValueError("Only finite values/logits or masked -inf logits are supported.")
    shift = -np.inf
    denominator = 0.0
    numerator = np.zeros(v.shape[1], dtype=np.float64)
    rescales = 0
    for start in range(0, x.size, tile):
        logits = x[start:start + tile]
        values = v[start:start + tile]
        valid = np.isfinite(logits)
        if not valid.any():
            continue
        tile_max = float(logits[valid].max())
        if not math.isfinite(shift):
            shift = tile_max
        elif tile_max > shift + tau:
            factor = math.exp2(shift - tile_max)
            denominator *= factor
            numerator *= factor
            shift = tile_max
            rescales += 1
        probabilities = np.exp2(logits - shift)
        denominator += probabilities.sum()
        numerator += probabilities @ values
    output = numerator / denominator if denominator else np.zeros_like(numerator)
    return output, rescales


def reference(x: Array, v: Array) -> Array:
    valid = np.isfinite(x)
    if not valid.any():
        return np.zeros(v.shape[1], dtype=np.float64)
    p = np.exp2(x - x[valid].max())
    return p @ v / p.sum()


def run() -> dict[str, object]:
    rng = np.random.default_rng(71)
    errors: list[float] = []
    for n in [1, 17, 64, 65, 129, 2048, 80000]:
        for kind in ["normal", "jumps", "masked", "coherent_low_values"]:
            x = rng.normal(0, 4, size=n)
            v = rng.normal(size=(n, 4))
            if kind == "jumps":
                x += (np.arange(n) // 64) * (500 / max(1, n // 64))
            elif kind == "masked":
                x[:min(64, n // 2)] = -np.inf
            elif kind == "coherent_low_values":
                x[:] = -12
                x[0] = 0
                v[:] = np.array([0.5, 1.0, -0.75, 2.0])
                v[0] = np.array([-0.5, 0.25, 1.0, -1.0])
            expected = reference(x, v)
            for tau in [0, 2, 4, 8]:
                actual, _ = bounded_shift(x, v, tau)
                error = float(np.linalg.norm(actual - expected) /
                              max(np.linalg.norm(expected), 1e-20))
                errors.append(error)
    n = 1_048_576
    x = rng.normal(0, 4, size=n)
    v = rng.normal(size=(n, 4))
    expected = reference(x, v)
    for tau in [0, 2, 4, 8]:
        actual, _ = bounded_shift(x, v, tau)
        errors.append(float(np.linalg.norm(actual - expected) /
                            max(np.linalg.norm(expected), 1e-20)))
    empty, _ = bounded_shift(np.full(64, -np.inf), np.zeros((64, 4)), 8)
    if not np.all(empty == 0):
        raise AssertionError("Empty-partition behavior failed.")
    if max(errors) >= 1e-11:
        raise AssertionError(f"Float64 algebra check failed: max error {max(errors):.3e}")
    return {
        "passed": True,
        "nonempty_configurations": len(errors),
        "empty_partition_test": "passed",
        "maximum_relative_l2": max(errors),
        "largest_context": n,
        "scope": "CPU float64 algebra only; not GPU speed, GPU precision, or judge compliance",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
