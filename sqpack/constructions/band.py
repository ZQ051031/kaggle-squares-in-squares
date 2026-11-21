"""Two axis-aligned staircases separated by a diagonal band of unit squares."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

from numba import njit
import numpy as np

from ..hp import Packing, mp

TOL = mp.mpf('1e-48')


def _parameters(L, p, q, t):
    L = mp.mpf(L)
    if not mp.isfinite(L) or L < 1:
        raise ValueError('L must be finite and >= 1')
    if any(not isinstance(v, int) or v < 0 for v in (p, q, t)) or t == 0:
        raise ValueError('p,q must be nonnegative integers and t a positive integer')
    k = int(mp.floor(L + TOL))
    if p >= 2 * k or q >= 2 * k:
        raise ValueError('p,q must be less than 2*floor(L)')
    return L, k


def _stair(p, k):
    return sum(max(0, min(k, p - i)) for i in range(k))


def _row_counts(L, p, q, t):
    root = mp.sqrt(2)
    if 2 * L - p - q - 2 - t * root < -TOL:
        return None
    return [max(0, int(mp.floor(root * L - 1 - abs((p - q) / root + 2 * r + 1 - t) + TOL)))
            for r in range(t)]


def band_count(L, p, q, t):
    """Count a fixed configuration, or zero if its separating band won't fit.

    Floors allow a 1e-48 tolerance to stabilize exact-contact thresholds.
    """
    L, k = _parameters(L, p, q, t)
    counts = _row_counts(L, p, q, t)
    return 0 if counts is None else _stair(p, k) + _stair(q, k) + sum(counts)


def build_band(L, p, q, t):
    """Construct a high-precision packing in [0,L]^2, possibly touching."""
    L, k = _parameters(L, p, q, t)
    counts = _row_counts(L, p, q, t)
    if counts is None:
        raise ValueError('band does not fit')
    half, root = mp.mpf('.5'), mp.sqrt(2)
    squares = []
    for i in range(k):
        for j in range(k):
            if i + j <= p - 1:
                squares.append((i + half, j + half, 0))
            if i + j <= q - 1:
                squares.append((L - i - half, L - j - half, 0))
    slack = 2 * L - p - q - 2 - t * root
    d0 = p + 1 + slack / 2
    for r, count in enumerate(counts):
        u = (d0 + (mp.mpf(r) + half) * root) / root
        for j in range(count):
            v = j + half - mp.mpf(count) / 2
            squares.append(((u - v) / root, (u + v) / root, mp.pi / 4))
    return Packing(squares)


@njit
def _stair_float(p, k):
    result = 0
    for i in range(k):
        result += max(0, min(k, p - i))
    return result


@njit
def _event_search(nmax):
    """Exhaust all floor-change events, rather than sampling and missing steps."""
    root = math.sqrt(2)
    sides = np.full(nmax + 1, np.inf)
    configs = np.zeros((nmax + 1, 3), dtype=np.int64)
    for k in range(1, min(31, int(math.sqrt(nmax))) + 1):
        for p in range(2 * k):
            for q in range(p, 2 * k):
                # p <= q loses nothing: swapping staircases mirrors the packing.
                for t in range(1, int(root * (k + 1)) + 1):
                    lower = max(float(k), (p + q + 2 + t * root) / 2)
                    if lower >= k + 1 - 1e-10:
                        continue
                    count = _stair_float(p, k) + _stair_float(q, k)
                    events = np.empty(3 * t)
                    size = 0
                    for r in range(t):
                        offset = abs((p - q) / root + 2 * r + 1 - t)
                        cells = max(0, int(math.floor(root * lower - 1 - offset + 1e-10)))
                        count += cells
                        event = (cells + 2 + offset) / root
                        while event < k + 1 - 1e-10:
                            events[size] = event
                            size += 1
                            event += 1 / root
                    if k * k < count < (k + 1) ** 2 and count <= nmax and lower < sides[count]:
                        sides[count] = lower
                        configs[count] = (p, q, t)
                    ordered = np.sort(events[:size])
                    pos = 0
                    while pos < size:
                        event = ordered[pos]
                        end = pos + 1
                        while end < size and ordered[end] - event < 1e-10:
                            end += 1
                        count += end - pos
                        if k * k < count < (k + 1) ** 2 and count <= nmax and event < sides[count]:
                            sides[count] = event
                            configs[count] = (p, q, t)
                        pos = end
    return sides, configs


@dataclass(frozen=True)
class BandConfig:
    count: int
    L: str
    p: int
    q: int
    t: int


def _refine(count, estimate, p, q, t):
    root = mp.sqrt(2)
    k = math.isqrt(count)
    base = _stair(p, k) + _stair(q, k)
    offsets = [abs((p - q) / root + 2 * r + 1 - t) for r in range(t)]
    required = (p + q + 2 + t * root) / 2

    def reaches(L):
        if L < required - TOL / 2:
            return False
        value = root * L - 1
        return base + sum(max(0, int(mp.floor(value - offset + TOL))) for offset in offsets) >= count

    low, high = mp.mpf(k), mp.mpf(estimate) + mp.mpf('1e-9')
    if not reaches(high):
        raise ArithmeticError('float search did not bracket count threshold')
    while high - low > mp.mpf('1e-45'):
        mid = (low + high) / 2
        if reaches(mid):
            high = mid
        else:
            low = mid
    # Stay above the exact threshold despite tolerated floors and serialization.
    side = high + mp.mpf('1e-44')
    if band_count(side, p, q, t) != count:
        raise ArithmeticError(f'count {count} skipped at its selected threshold')
    return mp.nstr(side, 60)


def best_band_configs(Nmax=1000, cache_path='cache/band_configs.json', refresh=False):
    """Return count -> minimal-side BandConfig (counts strictly between grids).

    All configs and their count-change events are enumerated in compiled float
    arithmetic, then winners are bisected at 60 dps. Event sides are of the
    form a/2+b/sqrt(2), with small integer a,b: distinct thresholds here are
    separated by far more than float roundoff. This avoids coarse-scan misses.
    Set cache_path=None to disable disk caching, or refresh=True to recompute.
    """
    if not isinstance(Nmax, int) or not 1 <= Nmax <= 1000:
        raise ValueError('Nmax must be an integer in 1..1000')
    path = Path(cache_path) if cache_path is not None else None
    if path is not None and path.exists() and not refresh:
        payload = json.loads(path.read_text())
        if payload.get('version') == 1 and payload.get('Nmax') == Nmax:
            return {int(n): BandConfig(**value) for n, value in payload['configs'].items()}
    sides, configs = _event_search(Nmax)
    result = {}
    for count in range(1, Nmax + 1):
        if not math.isfinite(sides[count]):
            continue
        p, q, t = map(int, configs[count])
        result[count] = BandConfig(count, _refine(count, sides[count], p, q, t), p, q, t)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'version': 1, 'Nmax': Nmax,
                   'configs': {str(n): vars(cfg) for n, cfg in result.items()}}
        path.write_text(json.dumps(payload, indent=2) + '\n')
    return result


def envelope_score(configs, Nmax=1000):
    """Float grid+band envelope score; dropping squares can only improve L."""
    sides = [math.inf] + [math.ceil(math.sqrt(n)) for n in range(1, Nmax + 1)]
    for count, cfg in configs.items():
        for n in range(1, min(count, Nmax) + 1):
            sides[n] = min(sides[n], float(cfg.L))
    return sum(sides[n] / math.sqrt(n) - 1 for n in range(1, Nmax + 1))
