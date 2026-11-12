"""Fallback for the `lap` package, used only if the real one cannot load.

ByteTrack (inside Ultralytics) does `import lap` and calls `lap.lapjv(...)` to
match old tracks to new boxes. On this Windows PC, Smart App Control blocks
lap's compiled DLL. Ultralytics already ships a pure-NumPy solver for the same
"assignment problem", so this tiny stand-in module just forwards to it.
On machines where `lap` loads fine (e.g. Hugging Face Spaces), nothing changes.
"""
import sys
import types

import numpy as np


def _lapjv(cost, extend_cost=True, cost_limit=np.inf, return_cost=True):
    from ultralytics.utils.ops import linear_sum_assignment

    n_rows, n_cols = cost.shape
    x = np.full(n_rows, -1, dtype=int)  # x[row] = matched column, -1 = unmatched
    y = np.full(n_cols, -1, dtype=int)  # y[col] = matched row,    -1 = unmatched
    # Same approach as Ultralytics' own non-lap branch: clip costs at the limit,
    # solve, then drop any pairs that are too expensive (too different) to match.
    rows, cols = linear_sum_assignment(np.minimum(cost, cost_limit))
    for r, c in zip(rows, cols):
        if cost[r, c] <= cost_limit:
            x[r], y[c] = c, r
    return (None, x, y) if return_cost else (x, y)


def install():
    """Register the stand-in as `lap` if the real package can't be imported."""
    try:
        import lap  # noqa: F401
    except ImportError:
        shim = types.ModuleType("lap")
        shim.__version__ = "fallback"
        shim.lapjv = _lapjv
        sys.modules["lap"] = shim
