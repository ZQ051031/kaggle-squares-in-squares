"""Four axis-aligned corner staircases surrounding a 45-degree central grid."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

from numba import njit
import numpy as np

from ..hp import Packing, mp

TOL = mp.mpf('1e-48')
SHIFTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _parameters(L, p1, p2, p3, p4, u_shift):
    L = mp.mpf(L)
    if not mp.isfinite(L) or L < 1:
        raise ValueError('L must be finite and >= 1')
    if any(not isinstance(v, int) or v < 0 for v in (p1, p2, p3, p4)):
        raise ValueError('p1,p2,p3,p4 must be nonnegative integers')
    u_shift = mp.mpf(u_shift)
    if not mp.isfinite(u_shift) or u_shift < 0 or u_shift > 1:
        raise ValueError('u_shift must be in [0, 1]')
    k = int(mp.floor(L + TOL))
    return L, k, u_shift


def _stair(p, k):
    return sum(max(0, min(k, p - i)) for i in range(k))


def _layout(L, p1, p2, p3, p4, u_shift):
    L, k, u_shift = _parameters(L, p1, p2, p3, p4, u_shift)
    if (p1 + p3 > L + TOL or p1 + p4 > L + TOL or
            p2 + p3 > L + TOL or p2 + p4 > L + TOL):
        return None
    root = mp.sqrt(2)
    half = mp.mpf('.5')
    u_lo = (p1 + 1) / root if p1 > 0 else mp.mpf(0)
    u_hi = (2 * L - p2 - 1) / root if p2 > 0 else root * L
    v_lo = -(L - p4 - 1) / root if p4 > 0 else -L / root
    v_hi = (L - p3 - 1) / root if p3 > 0 else L / root
    if u_hi - u_lo < 1 - TOL or v_hi - v_lo < 1 - TOL:
        return None
    nu = int(mp.floor(u_hi - u_lo + TOL))
    slack_u = (u_hi - u_lo) - nu
    u0 = u_lo + slack_u * u_shift + half
    hi_box = root * L - 1
    rows = []
    for r in range(nu):
        uc = u0 + r
        vmin = max(v_lo + half, uc - hi_box, 1 - uc)
        vmax = min(v_hi - half, uc - 1, hi_box - uc)
        if vmax >= vmin - TOL:
            cnt = int(mp.floor(vmax - vmin + 1 + TOL))
            if cnt > 0:
                rows.append((uc, vmin, vmax, cnt))
    return L, k, rows


def corners45_count(L, p1, p2, p3, p4, u_shift=0.5):
    """Count squares in a 4-corner + 45-degree central grid, or 0 if infeasible."""
    layout = _layout(L, p1, p2, p3, p4, u_shift)
    if layout is None:
        return 0
    _, k, rows = layout
    stair = _stair(p1, k) + _stair(p2, k) + _stair(p3, k) + _stair(p4, k)
    return stair + sum(cnt for _, _, _, cnt in rows)


def build_corners45(L, p1, p2, p3, p4, u_shift=0.5):
    """Construct a high-precision 4-corner + 45-degree packing in [0, L]^2."""
    layout = _layout(L, p1, p2, p3, p4, u_shift)
    if layout is None:
        raise ValueError('corners45 configuration does not fit')
    L, k, rows = layout
    half, root = mp.mpf('.5'), mp.sqrt(2)
    squares = []
    for i in range(k):
        for j in range(k):
            if i + j <= p1 - 1:
                squares.append((i + half, j + half, mp.mpf(0)))
            if i + j <= p2 - 1:
                squares.append((L - i - half, L - j - half, mp.mpf(0)))
            if i + j <= p3 - 1:
                squares.append((i + half, L - j - half, mp.mpf(0)))
            if i + j <= p4 - 1:
                squares.append((L - i - half, j + half, mp.mpf(0)))
    ang = mp.pi / 4
    for uc, vmin, vmax, cnt in rows:
        v_start = (vmin + vmax - (cnt - 1)) / 2
        for j in range(cnt):
            vc = v_start + j
            squares.append(((uc - vc) / root, (uc + vc) / root, ang))
    return Packing(squares)


@njit
def _stair_float(p, k):
    total = 0
    for i in range(k):
        v = p - i
        if v > k:
            v = k
        if v > 0:
            total += v
    return total


@njit
def _eval_float(L, p1, p2, p3, p4, u_shift):
    if (p1 + p3 > L + 1e-12 or p1 + p4 > L + 1e-12 or
            p2 + p3 > L + 1e-12 or p2 + p4 > L + 1e-12):
        return 0
    root = math.sqrt(2.0)
    u_lo = (p1 + 1.0) / root if p1 > 0 else 0.0
    u_hi = (2.0 * L - p2 - 1.0) / root if p2 > 0 else root * L
    v_lo = -(L - p4 - 1.0) / root if p4 > 0 else -L / root
    v_hi = (L - p3 - 1.0) / root if p3 > 0 else L / root
    if u_hi - u_lo < 1.0 - 1e-12 or v_hi - v_lo < 1.0 - 1e-12:
        return 0
    nu = int(math.floor(u_hi - u_lo + 1e-12))
    slack_u = (u_hi - u_lo) - nu
    u0 = u_lo + slack_u * u_shift + 0.5
    hi_box = root * L - 1.0
    tilted = 0
    for r in range(nu):
        uc = u0 + r
        vmin = max(v_lo + 0.5, max(uc - hi_box, 1.0 - uc))
        vmax = min(v_hi - 0.5, min(uc - 1.0, hi_box - uc))
        if vmax >= vmin - 1e-12:
            tilted += int(math.floor(vmax - vmin + 1.0 + 1e-12))
    k = int(math.floor(L + 1e-12))
    return (_stair_float(p1, k) + _stair_float(p2, k) +
            _stair_float(p3, k) + _stair_float(p4, k) + tilted)


@njit
def _search_float(nmax):
    best_L = np.full(nmax + 1, np.inf)
    best_cfg = np.zeros((nmax + 1, 5), dtype=np.int64)
    shifts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    root = math.sqrt(2.0)
    for k in range(3, min(31, int(math.sqrt(nmax))) + 1):
        for p1 in range(0, k + 1):
            for p2 in range(p1, k + 1):
                max_p3 = k - p2
                for p3 in range(0, max_p3 + 1):
                    for p4 in range(p3, max_p3 + 1):
                        if p3 == 0 and p4 == 0:
                            # Covered by band.py
                            continue
                        L_min = float(k)
                        if p1 > 0 or p2 > 0:
                            req = ((p1 + 1.0 if p1 > 0 else 0.0) +
                                   (p2 + 1.0 if p2 > 0 else 0.0) + root) / 2.0
                            if req > L_min:
                                L_min = req
                        req_v = ((p3 + 1.0 if p3 > 0 else 0.0) +
                                 (p4 + 1.0 if p4 > 0 else 0.0) + root) / 2.0
                        if req_v > L_min:
                            L_min = req_v
                        if L_min >= k + 1.0 - 1e-9:
                            continue
                        steps = 120
                        span = (k + 1.0) - L_min
                        for li in range(steps):
                            L = L_min + span * li / steps
                            for si in range(5):
                                cnt = _eval_float(L, p1, p2, p3, p4, shifts[si])
                                if k * k < cnt < (k + 1) ** 2 and cnt <= nmax and L < best_L[cnt]:
                                    best_L[cnt] = L
                                    best_cfg[cnt, 0] = p1
                                    best_cfg[cnt, 1] = p2
                                    best_cfg[cnt, 2] = p3
                                    best_cfg[cnt, 3] = p4
                                    best_cfg[cnt, 4] = si
    for n in range(1, nmax + 1):
        if math.isfinite(best_L[n]):
            p1, p2, p3, p4, si = (best_cfg[n, 0], best_cfg[n, 1],
                                  best_cfg[n, 2], best_cfg[n, 3], best_cfg[n, 4])
            sh = shifts[si]
            hi = best_L[n]
            lo = float(int(math.floor(hi)))
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if _eval_float(mid, p1, p2, p3, p4, sh) >= n:
                    hi = mid
                else:
                    lo = mid
            best_L[n] = hi
    return best_L, best_cfg


@dataclass(frozen=True)
class Corners45Config:
    count: int
    L: str
    p1: int
    p2: int
    p3: int
    p4: int
    u_shift: float


def _refine(count, estimate, p1, p2, p3, p4, u_shift):
    k = math.isqrt(count)
    low = mp.mpf(k)
    high = mp.mpf(estimate) + mp.mpf('1e-7')
    if corners45_count(high, p1, p2, p3, p4, u_shift) < count:
        raise ArithmeticError(f'corners45 float search did not bracket count={count}')
    while high - low > mp.mpf('1e-45'):
        mid = (low + high) / 2
        if corners45_count(mid, p1, p2, p3, p4, u_shift) >= count:
            high = mid
        else:
            low = mid
    side = high + mp.mpf('1e-44')
    actual = corners45_count(side, p1, p2, p3, p4, u_shift)
    if actual < count:
        raise ArithmeticError(f'corners45 count {count} not reached at threshold')
    return actual, mp.nstr(side, 60)


def best_corners45_configs(Nmax=1000, cache_path='cache/corners45_configs.json', refresh=False):
    """Return count -> minimal-side Corners45Config (counts strictly between grids)."""
    if not isinstance(Nmax, int) or not 1 <= Nmax <= 1000:
        raise ValueError('Nmax must be an integer in 1..1000')
    path = Path(cache_path) if cache_path is not None else None
    if path is not None and path.exists() and not refresh:
        payload = json.loads(path.read_text())
        if payload.get('version') == 1 and payload.get('Nmax', 0) >= Nmax:
            return {int(n): Corners45Config(**value)
                    for n, value in payload['configs'].items() if int(n) <= Nmax}
    sides, configs = _search_float(Nmax)
    result = {}
    for count in range(1, Nmax + 1):
        if not math.isfinite(sides[count]):
            continue
        p1, p2, p3, p4, si = map(int, configs[count])
        u_shift = SHIFTS[si]
        actual, side_str = _refine(count, sides[count], p1, p2, p3, p4, u_shift)
        if actual <= Nmax and (actual not in result or mp.mpf(side_str) < mp.mpf(result[actual].L)):
            result[actual] = Corners45Config(actual, side_str, p1, p2, p3, p4, u_shift)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'version': 1, 'Nmax': Nmax,
                   'configs': {str(n): vars(cfg) for n, cfg in result.items()}}
        path.write_text(json.dumps(payload, indent=2) + '\n')
    return result
