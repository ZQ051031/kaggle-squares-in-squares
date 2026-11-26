import numpy as np
import pytest

from sqpack.constructions.corners45 import build_corners45, corners45_count
from sqpack.constructions.lstrip import build_lstrip, lstrip_count
from sqpack.geometry import bounding_side
from sqpack.metric_official import evaluate_single_puzzle
from sqpack.optimize import certify_float_packing, energy_and_grad_smooth, shrink_packing


def _certify(packing):
    n = len(packing)
    rows = [dict(n=str(n), square_id=str(i), x=x, y=y, angle=a)
            for i, (x, y, a) in enumerate(packing.normalize().inflate('1e-20').to_rows(45))]
    _, score, side, reason = evaluate_single_puzzle(n, rows)
    assert reason == 'ok' and side is not None
    return float(side), float(score)


def test_corners45_build_and_certify():
    L = '15.75'
    cnt = corners45_count(L, 5, 5, 5, 5, 0.5)
    assert cnt > 0
    pk = build_corners45(L, 5, 5, 5, 5, 0.5)
    assert len(pk) == cnt
    side, _ = _certify(pk)
    assert side <= 15.75 + 1e-9


def test_corners45_invalid_params():
    with pytest.raises(ValueError):
        corners45_count('0.5', 1, 1, 1, 1, 0.5)
    with pytest.raises(ValueError):
        corners45_count('10.0', 1, 1, 1, 1, 1.5)
    assert corners45_count('10.0', 6, 6, 6, 6, 0.5) == 0


def test_lstrip_build_and_certify():
    L = '17.985'
    cnt = lstrip_count(L, 12, 12, 6, 6, 0, 0)
    assert cnt > 0
    pk = build_lstrip(L, 12, 12, 6, 6, 0, 0)
    assert len(pk) == cnt
    side, _ = _certify(pk)
    assert side <= 17.985 + 1e-9


def test_lstrip_invalid_params():
    assert lstrip_count('1.5', 0, 0, 2, 2) == 0
    assert lstrip_count('10.0', 5, 5, 4, 4) == 0


def test_optimize_gradient_and_shrink():
    rng = np.random.default_rng(7)
    n = 8
    p = rng.uniform(0.5, 2.5, 3 * n)
    _, g = energy_and_grad_smooth(p, n, 2.8, 1e-10)
    eps = 1e-7
    for k in range(3 * n):
        p_hi = p.copy()
        p_hi[k] += eps
        p_lo = p.copy()
        p_lo[k] -= eps
        fd = (energy_and_grad_smooth(p_hi, n, 2.8, 1e-10)[0] -
              energy_and_grad_smooth(p_lo, n, 2.8, 1e-10)[0]) / (2 * eps)
        assert abs(fd - g[k]) < 1e-6

    pk = build_lstrip('17.985', 12, 12, 6, 6, 0, 0)
    arr = np.array([[float(x), float(y), float(a)] for x, y, a in pk.squares])
    L0 = bounding_side(arr)
    out, L1 = shrink_packing(arr, max_rounds=6, step_init=0.015, seed=0)
    assert L1 < L0 - 1e-3
    cert = certify_float_packing(out, len(arr))
    assert cert is not None
    _, off_L, _ = cert
    assert off_L < L0 - 1e-3
