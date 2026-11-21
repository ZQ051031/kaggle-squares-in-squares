"""Exact decimal, axis-aligned grid baseline."""
import math
import numpy as np


def grid(n):
    if not isinstance(n, (int, np.integer)) or not 1 <= n <= 1000:
        raise ValueError('n must be an integer in 1..1000')
    k = math.isqrt(n - 1) + 1
    i = np.arange(n)
    return np.column_stack((i % k + 0.5, i // k + 0.5, np.zeros(n)))


def grid_total():
    return sum((math.isqrt(n - 1) + 1) / math.sqrt(n) - 1 for n in range(1, 1001))
