import math

import pytest

from sqpack.constructions.band import band_count, best_band_configs, build_band, envelope_score
from sqpack.hp import mp
from sqpack.metric_official import evaluate_single_puzzle


@pytest.fixture(scope='module')
def small_configs():
    return best_band_configs(35, cache_path=None)


def test_known_ten_and_exact_threshold(small_configs):
    cfg = small_configs[10]
    expected = 3 + 1 / mp.sqrt(2)
    assert abs(mp.mpf(cfg.L) - expected) < mp.mpf('1e-40')
    assert band_count(expected, cfg.p, cfg.q, cfg.t) == 10
    assert band_count(expected - mp.mpf('1e-40'), cfg.p, cfg.q, cfg.t) < 10
    assert band_count(expected, cfg.q, cfg.p, cfg.t) == 10


def test_build_and_certify(small_configs):
    for n, cfg in small_configs.items():
        packing = build_band(cfg.L, cfg.p, cfg.q, cfg.t)
        assert len(packing) == n
        assert abs(packing.side() - mp.mpf(cfg.L)) < mp.mpf('1e-40')
        assert all(-mp.mpf('1e-40') <= packing.bbox()[axis] for axis in (0, 1))
        assert all(packing.bbox()[axis] <= mp.mpf(cfg.L) + mp.mpf('1e-40') for axis in (2, 3))
        rows = [dict(n=str(n), square_id=str(i), x=x, y=y, angle=a)
                for i, (x, y, a) in enumerate(packing.normalize().inflate().to_rows())]
        assert evaluate_single_puzzle(n, rows)[3] == 'ok'
        assert band_count(mp.mpf(cfg.L) - mp.mpf('1e-39'), cfg.p, cfg.q, cfg.t) < n


def test_large_known_band():
    L = 28 + 3 * mp.sqrt(2) / 2 + mp.mpf('1.5')
    assert band_count('31.6214', 26, 31, 3) == 967
    assert band_count(L, 26, 31, 3) == 967
    packing = build_band(L, 26, 31, 3)
    assert len(packing) == 967
    assert abs(packing.side() - L) < mp.mpf('1e-40')


def test_complete_search_against_small_scan(small_configs):
    root = math.sqrt(2)
    for k in range(1, 6):
        for step in range(1, 20):
            L = k + step / 20
            for p in range(2 * k):
                for q in range(p, 2 * k):
                    for t in range(1, int(root * L) + 1):
                        if p + q + 2 + t * root > 2 * L:
                            continue
                        stair = sum(max(0, min(k, p - i)) + max(0, min(k, q - i)) for i in range(k))
                        slack = 2 * L - q - p - 2 - t * root
                        count = stair
                        for r in range(t):
                            d1 = p + 1 + slack / 2 + r * root
                            d2 = d1 + root
                            count += max(0, math.floor(root * min(d1, 2 * L - d1, d2, 2 * L - d2) + 1e-10))
                        if k * k < count < min((k + 1) ** 2, 36):
                            assert count in small_configs
                            assert float(small_configs[count].L) <= L + 1e-12


def test_cache_and_empty_result(tmp_path, small_configs):
    path = tmp_path / 'band.json'
    first = best_band_configs(12, path)
    assert best_band_configs(12, path) == first
    assert best_band_configs(12, path, refresh=True) == first
    assert best_band_configs(1, cache_path=None) == {}
    assert envelope_score({}, 12) == pytest.approx(sum(math.ceil(math.sqrt(n)) / math.sqrt(n) - 1 for n in range(1, 13)))
    assert envelope_score(small_configs, 35) < envelope_score({}, 35)


@pytest.mark.parametrize('args', [('nan', 0, 0, 1), ('0', 0, 0, 1),
                                  ('3', -1, 0, 1), ('3', 0, 0, 0),
                                  ('3', 6, 0, 1), ('3', 0, 0, 1.5)])
def test_bad_parameters(args):
    with pytest.raises(ValueError):
        band_count(*args)


def test_infeasible_band_and_bad_search():
    assert band_count('2', 1, 1, 2) == 0
    with pytest.raises(ValueError, match='does not fit'):
        build_band('2', 1, 1, 2)
    for n in [0, 1001, 1.5]:
        with pytest.raises(ValueError):
            best_band_configs(n, cache_path=None)
