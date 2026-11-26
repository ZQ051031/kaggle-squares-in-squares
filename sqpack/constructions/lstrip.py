"""L-shaped two-strip tilted-row construction around an axis-aligned block."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

from numba import njit
import numpy as np

from ..hp import Packing, mp

TOL = mp.mpf('1e-48')


def _setup(L, A, B, m1, m2, d1=0, d2=0):
    L = mp.mpf(L)
    d1, d2 = mp.mpf(d1), mp.mpf(d2)
    if not mp.isfinite(L) or L < 2:
        return None
    if any(not isinstance(v, int) or v < 0 for v in (A, B, m1, m2)):
        raise ValueError('A, B, m1, m2 must be nonnegative integers')
    if d1 < -TOL or d2 < -TOL:
        return None
    W = L - A
    H = L - B
    if W <= 1 + mp.mpf('1e-12') or H <= 1 + mp.mpf('1e-12'):
        return None
    if m1 <= W + mp.mpf('1e-12') or m2 <= H + mp.mpf('1e-12'):
        return None
    R1 = mp.sqrt(m1 * m1 + 1)
    R2 = mp.sqrt(m2 * m2 + 1)
    if W >= R1 - mp.mpf('1e-12') or H >= R2 - mp.mpf('1e-12'):
        return None
    th1 = mp.atan2(1, m1) + mp.acos(W / R1)
    th2 = mp.atan2(1, m2) + mp.acos(H / R2)
    if th1 + th2 >= mp.pi / 2 - mp.mpf('1e-12'):
        return None
    return L, W, H, th1, th2, d1, d2


def lstrip_count(L, A, B, m1, m2, d1=0, d2=0):
    """Count squares in an L-shaped two-strip tilted-row packing, or 0 if infeasible."""
    cfg = _setup(L, A, B, m1, m2, d1, d2)
    if cfg is None:
        return 0
    L, W, H, th1, th2, d1, d2 = cfg
    c1, s1 = mp.cos(th1), mp.sin(th1)
    c2, s2 = mp.cos(th2), mp.sin(th2)
    half = mp.mpf('.5')
    C1 = -s1 * W + c1 * H + d1
    C2 = c2 * W - s2 * H + d2

    y_start = (C1 + half + s1 * (c1 + s1) / 2) / c1
    left_cnt = 0
    r = 0
    while True:
        yc = y_start + r / c1
        if yc - (c1 + s1) / 2 > L + TOL:
            break
        j_lo = max(0, int(mp.ceil(((c1 + s1) / 2 - yc - TOL) / s1)))
        j_hi = min(m1 - 1, int(mp.floor((L - (c1 + s1) / 2 - yc + TOL) / s1)))
        if j_hi >= j_lo:
            left_cnt += j_hi - j_lo + 1
        r += 1

    x_start = (C2 + half + s2 * (c2 + s2) / 2) / c2
    bot_cnt = 0
    r = 0
    while True:
        xc = x_start + r / c2
        if xc - (c2 + s2) / 2 > L + TOL:
            break
        j_lo = max(0, int(mp.ceil(((c2 + s2) / 2 - xc - TOL) / s2)))
        j_hi = min(m2 - 1, int(mp.floor((L - (c2 + s2) / 2 - xc + TOL) / s2)))
        if j_hi >= j_lo:
            bot_cnt += j_hi - j_lo + 1
        r += 1

    bl_cnt = 0
    w_int = int(mp.floor(W + TOL))
    h_int = int(mp.floor(H + TOL))
    for i in range(w_int):
        for j in range(h_int):
            if -s1 * i + c1 * (j + 1) <= C1 + TOL and c2 * (i + 1) - s2 * j <= C2 + TOL:
                bl_cnt += 1

    return A * B + left_cnt + bot_cnt + bl_cnt


def build_lstrip(L, A, B, m1, m2, d1=0, d2=0):
    """Construct a high-precision L-shaped two-strip tilted-row packing in [0, L]^2."""
    cfg = _setup(L, A, B, m1, m2, d1, d2)
    if cfg is None:
        raise ValueError('lstrip configuration does not fit')
    L, W, H, th1, th2, d1, d2 = cfg
    c1, s1 = mp.cos(th1), mp.sin(th1)
    c2, s2 = mp.cos(th2), mp.sin(th2)
    half = mp.mpf('.5')
    C1 = -s1 * W + c1 * H + d1
    C2 = c2 * W - s2 * H + d2
    squares = []

    for i in range(A):
        for j in range(B):
            squares.append((L - A + i + half, L - B + j + half, mp.mpf(0)))

    y_start = (C1 + half + s1 * (c1 + s1) / 2) / c1
    r = 0
    while True:
        yc = y_start + r / c1
        if yc - (c1 + s1) / 2 > L + TOL:
            break
        j_lo = max(0, int(mp.ceil(((c1 + s1) / 2 - yc - TOL) / s1)))
        j_hi = min(m1 - 1, int(mp.floor((L - (c1 + s1) / 2 - yc + TOL) / s1)))
        for j in range(j_lo, j_hi + 1):
            squares.append(((c1 + s1) / 2 + j * c1, yc + j * s1, th1))
        r += 1

    x_start = (C2 + half + s2 * (c2 + s2) / 2) / c2
    ang2 = mp.pi / 2 - th2
    r = 0
    while True:
        xc = x_start + r / c2
        if xc - (c2 + s2) / 2 > L + TOL:
            break
        j_lo = max(0, int(mp.ceil(((c2 + s2) / 2 - xc - TOL) / s2)))
        j_hi = min(m2 - 1, int(mp.floor((L - (c2 + s2) / 2 - xc + TOL) / s2)))
        for j in range(j_lo, j_hi + 1):
            squares.append((xc + j * s2, (c2 + s2) / 2 + j * c2, ang2))
        r += 1

    w_int = int(mp.floor(W + TOL))
    h_int = int(mp.floor(H + TOL))
    for i in range(w_int):
        for j in range(h_int):
            if -s1 * i + c1 * (j + 1) <= C1 + TOL and c2 * (i + 1) - s2 * j <= C2 + TOL:
                squares.append((i + half, j + half, mp.mpf(0)))

    return Packing(squares)


@njit
def _eval_float(L, A, B, m1, m2, d1, d2):
    W = L - A
    H = L - B
    if W <= 1.0 + 1e-9 or H <= 1.0 + 1e-9:
        return 0
    if m1 <= W + 1e-9 or m2 <= H + 1e-9:
        return 0
    R1 = math.sqrt(m1 * m1 + 1.0)
    R2 = math.sqrt(m2 * m2 + 1.0)
    if W >= R1 - 1e-12 or H >= R2 - 1e-12:
        return 0
    th1 = math.atan2(1.0, float(m1)) + math.acos(W / R1)
    th2 = math.atan2(1.0, float(m2)) + math.acos(H / R2)
    if th1 + th2 >= math.pi / 2.0 - 1e-9:
        return 0
    c1, s1 = math.cos(th1), math.sin(th1)
    c2, s2 = math.cos(th2), math.sin(th2)
    C1 = -s1 * W + c1 * H + d1
    C2 = c2 * W - s2 * H + d2

    y_start = (C1 + 0.5 + s1 * (c1 + s1) / 2.0) / c1
    left_cnt = 0
    r = 0
    while True:
        yc = y_start + r / c1
        if yc - (c1 + s1) / 2.0 > L:
            break
        j_lo = int(math.ceil(((c1 + s1) / 2.0 - yc - 1e-12) / s1))
        j_hi = int(math.floor((L - (c1 + s1) / 2.0 - yc + 1e-12) / s1))
        if j_lo < 0:
            j_lo = 0
        if j_hi > m1 - 1:
            j_hi = m1 - 1
        if j_hi >= j_lo:
            left_cnt += j_hi - j_lo + 1
        r += 1

    x_start = (C2 + 0.5 + s2 * (c2 + s2) / 2.0) / c2
    bot_cnt = 0
    r = 0
    while True:
        xc = x_start + r / c2
        if xc - (c2 + s2) / 2.0 > L:
            break
        j_lo = int(math.ceil(((c2 + s2) / 2.0 - xc - 1e-12) / s2))
        j_hi = int(math.floor((L - (c2 + s2) / 2.0 - xc + 1e-12) / s2))
        if j_lo < 0:
            j_lo = 0
        if j_hi > m2 - 1:
            j_hi = m2 - 1
        if j_hi >= j_lo:
            bot_cnt += j_hi - j_lo + 1
        r += 1

    bl_cnt = 0
    w_int = int(math.floor(W + 1e-12))
    h_int = int(math.floor(H + 1e-12))
    for i in range(w_int):
        for j in range(h_int):
            if -s1 * i + c1 * (j + 1) <= C1 + 1e-12 and c2 * (i + 1) - s2 * j <= C2 + 1e-12:
                bl_cnt += 1

    return A * B + left_cnt + bot_cnt + bl_cnt


@njit
def _search_float(nmax):
    best_L = np.full(nmax + 1, np.inf)
    best_cfg = np.zeros((nmax + 1, 6), dtype=np.int64)
    shifts = np.array([0.0, 0.25, 0.5])
    for k in range(4, min(31, int(math.sqrt(nmax))) + 1):
        for li in range(1, 200):
            L = k + li / 200.0
            for A in range(0, k - 1):
                W = L - A
                m1_lo = int(math.ceil(W))
                for B in range(A, k - 1):
                    H = L - B
                    m2_lo = int(math.ceil(H))
                    for m1 in range(m1_lo, min(m1_lo + 4, int(W * 1.42) + 2)):
                        for m2 in range(m2_lo, min(m2_lo + 4, int(H * 1.42) + 2)):
                            for d1i in range(3):
                                d1 = shifts[d1i]
                                for d2i in range(3):
                                    d2 = shifts[d2i]
                                    cnt = _eval_float(L, A, B, m1, m2, d1, d2)
                                    if k * k < cnt < (k + 1) ** 2 and cnt <= nmax and L < best_L[cnt]:
                                        best_L[cnt] = L
                                        best_cfg[cnt, 0] = A
                                        best_cfg[cnt, 1] = B
                                        best_cfg[cnt, 2] = m1
                                        best_cfg[cnt, 3] = m2
                                        best_cfg[cnt, 4] = d1i
                                        best_cfg[cnt, 5] = d2i
    for n in range(1, nmax + 1):
        if math.isfinite(best_L[n]):
            A, B, m1, m2, d1i, d2i = (best_cfg[n, 0], best_cfg[n, 1], best_cfg[n, 2],
                                      best_cfg[n, 3], best_cfg[n, 4], best_cfg[n, 5])
            d1, d2 = shifts[d1i], shifts[d2i]
            hi = best_L[n]
            lo = float(int(math.floor(hi)))
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if _eval_float(mid, A, B, m1, m2, d1, d2) >= n:
                    hi = mid
                else:
                    lo = mid
            best_L[n] = hi
    return best_L, best_cfg


@dataclass(frozen=True)
class LStripConfig:
    count: int
    L: str
    A: int
    B: int
    m1: int
    m2: int
    d1: float
    d2: float


SHIFTS = (0.0, 0.25, 0.5)


def _refine(count, estimate, A, B, m1, m2, d1, d2):
    k = math.isqrt(count)
    low = mp.mpf(k)
    high = mp.mpf(estimate) + mp.mpf('1e-7')
    if lstrip_count(high, A, B, m1, m2, d1, d2) < count:
        raise ArithmeticError(f'lstrip float search did not bracket count={count}')
    while high - low > mp.mpf('1e-45'):
        mid = (low + high) / 2
        if lstrip_count(mid, A, B, m1, m2, d1, d2) >= count:
            high = mid
        else:
            low = mid
    side = high + mp.mpf('1e-44')
    actual = lstrip_count(side, A, B, m1, m2, d1, d2)
    if actual < count:
        raise ArithmeticError(f'lstrip count {count} not reached at threshold')
    return actual, mp.nstr(side, 60)


def best_lstrip_configs(Nmax=1000, cache_path='cache/lstrip_configs.json', refresh=False):
    """Return count -> minimal-side LStripConfig (counts strictly between grids)."""
    if not isinstance(Nmax, int) or not 1 <= Nmax <= 1000:
        raise ValueError('Nmax must be an integer in 1..1000')
    path = Path(cache_path) if cache_path is not None else None
    if path is not None and path.exists() and not refresh:
        payload = json.loads(path.read_text())
        if payload.get('version') == 1 and payload.get('Nmax', 0) >= Nmax:
            return {int(n): LStripConfig(**value)
                    for n, value in payload['configs'].items() if int(n) <= Nmax}
    sides, configs = _search_float(Nmax)
    result = {}
    for count in range(1, Nmax + 1):
        if not math.isfinite(sides[count]):
            continue
        A, B, m1, m2, d1i, d2i = map(int, configs[count])
        d1, d2 = SHIFTS[d1i], SHIFTS[d2i]
        actual, side_str = _refine(count, sides[count], A, B, m1, m2, d1, d2)
        if actual <= Nmax and (actual not in result or mp.mpf(side_str) < mp.mpf(result[actual].L)):
            result[actual] = LStripConfig(actual, side_str, A, B, m1, m2, d1, d2)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'version': 1, 'Nmax': Nmax,
                   'configs': {str(n): vars(cfg) for n, cfg in result.items()}}
        path.write_text(json.dumps(payload, indent=2) + '\n')
    return result
