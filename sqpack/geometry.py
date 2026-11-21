"""Floating-point geometry; touching is allowed, positive separation is safer."""
import numpy as np


def corners(squares):
    """Return corner coordinates with shape (n, 4, 2)."""
    a = np.asarray(squares, dtype=float)
    c, s = np.cos(a[:, 2]), np.sin(a[:, 2])
    u = np.stack((c, s), axis=1)
    v = np.stack((-s, c), axis=1)
    signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) / 2
    return a[:, None, :2] + signs[None, :, :1] * u[:, None] + signs[None, :, 1:] * v[:, None]


def bounding_side(squares):
    """Translation-invariant bounding-square side (zero for an empty input)."""
    pts = corners(squares).reshape(-1, 2)
    return float(np.ptp(pts, axis=0).max()) if len(pts) else 0.0


def check_overlaps(squares):
    """Return (minimum signed SAT separation, overlapping index pairs).

    Negative separation means overlap; zero means contact. Empty/singleton
    packings have infinite separation. Sweep broadphase excludes pairs whose
    center distance proves they cannot improve the current separation bound.
    """
    a = np.asarray(squares, dtype=float)
    n = len(a)
    if n < 2:
        return float('inf'), []
    c, s = np.cos(a[:, 2]), np.sin(a[:, 2])
    axes = np.stack((np.stack((c, s), -1), np.stack((-s, c), -1)), 1)

    def gaps(i, j):
        ax = np.concatenate((axes[i], axes[j]), axis=1)
        delta = a[j, :2] - a[i, :2]
        distance = np.abs(np.einsum('bi,bki->bk', delta, ax))
        ri = np.abs(np.einsum('bki,bli->bkl', ax, axes[i])).sum(-1) / 2
        rj = np.abs(np.einsum('bki,bli->bkl', ax, axes[j])).sum(-1) / 2
        return (distance - ri - rj).max(-1)

    best = float(gaps(np.array([0]), np.array([1]))[0])
    order = np.argsort(a[:, 0])
    xs = a[order, 0]
    left, right = [], []
    # For any pair, SAT separation >= |dx|/sqrt(2) - sqrt(2).
    cutoff = np.sqrt(2) * (max(best, 0) + np.sqrt(2)) + 1e-12
    for pos, i in enumerate(order):
        end = np.searchsorted(xs, xs[pos] + cutoff, side='right')
        js = order[pos + 1:end]
        left.extend([i] * len(js))
        right.extend(js)
    overlapping = []
    for start in range(0, len(left), 65536):
        ii = np.asarray(left[start:start + 65536])
        jj = np.asarray(right[start:start + 65536])
        sep = gaps(ii, jj)
        best = min(best, float(sep.min()))
        overlapping.extend((int(i), int(j)) for i, j in zip(ii[sep < 0], jj[sep < 0]))
    return best, overlapping
