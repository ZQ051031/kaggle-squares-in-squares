import csv
import math
import time

import numpy as np
import pytest

from sqpack.baseline import grid
from sqpack.geometry import bounding_side, check_overlaps, corners
from sqpack.io import DECIMAL, HEADER, read_submission, read_submission_strings, write_submission
from sqpack.store import BestStore
from sqpack.validate import official_score, validate_submission


def test_geometry_touch_overlap_and_gap():
    a = grid(2)
    assert check_overlaps(a) == (0, [])
    a[1, 0] -= 0.01
    gap, pairs = check_overlaps(a)
    assert gap == pytest.approx(-0.01)
    assert len(pairs) == 1
    a[1, 0] += 0.02
    assert check_overlaps(a)[0] == pytest.approx(0.01)
    assert bounding_side(a) == pytest.approx(2.01)
    assert corners(a).shape == (2, 4, 2)
    assert bounding_side(a + [100, -50, 0]) == pytest.approx(2.01)


def test_empty_singleton_and_rotated():
    assert bounding_side(np.empty((0, 3))) == 0
    assert check_overlaps(np.empty((0, 3))) == (math.inf, [])
    assert check_overlaps(grid(1)) == (math.inf, [])
    assert bounding_side(np.array([[0, 0, math.pi / 4]])) == pytest.approx(math.sqrt(2))
    assert check_overlaps(np.array([[0, 0, 0], [0, 0, 1]]))[1]


def test_sweep_matches_brute_force():
    rng = np.random.default_rng(42)
    a = rng.uniform(-10, 10, (50, 3))
    pts = corners(a)
    best = math.inf
    pairs = set()
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            angles = a[[i, j], 2]
            axes = np.vstack((np.column_stack((np.cos(angles), np.sin(angles))),
                              np.column_stack((-np.sin(angles), np.cos(angles)))))
            p, q = pts[i] @ axes.T, pts[j] @ axes.T
            gap = np.maximum(q.min(0) - p.max(0), p.min(0) - q.max(0)).max()
            best = min(best, gap)
            if gap < 0:
                pairs.add((i, j))
    actual, overlaps = check_overlaps(a)
    assert actual == pytest.approx(best)
    assert {tuple(sorted(pair)) for pair in overlaps} == pairs


def test_thousand_squares_fast():
    start = time.perf_counter()
    gap, pairs = check_overlaps(grid(1000))
    assert not pairs and gap == 0
    assert time.perf_counter() - start < 1


@pytest.mark.parametrize('value', ['0', '-.5', '+1.', '1e-15', '1.234E+2'])
def test_decimal_accept(value):
    assert DECIMAL.fullmatch(value)


@pytest.mark.parametrize('value', ['nan', 'inf', '1/2', 'sqrt(2)', ' 1', '1\n', '', '١'])
def test_decimal_reject(value):
    assert not DECIMAL.fullmatch(value)


def test_io_roundtrip_and_strings(tmp_path):
    path = tmp_path / 'submission.csv'
    solutions = {3: grid(3), 1: grid(1)}
    write_submission(solutions, path)
    loaded = read_submission(path)
    assert list(loaded) == [1, 3]
    for n in solutions:
        np.testing.assert_array_equal(loaded[n], solutions[n])
    assert read_submission_strings(path)[1][0, 0] == '0.5'
    with path.open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == HEADER
    assert [int(row[0]) for row in rows[1:]] == list(range(4))


@pytest.mark.parametrize('column,value', [(0, '1'), (1, '1001'), (2, '1'),
                                        (3, 'nan'), (3, '10001'), (5, '101'),
                                        (3, '0' * 101)])
def test_bad_format(tmp_path, column, value):
    row = ['0', '1', '0', '0.5', '0.5', '0']
    row[column] = value
    path = tmp_path / 'bad.csv'
    with path.open('w', newline='') as f:
        csv.writer(f).writerows([HEADER, row])
    with pytest.raises(ValueError):
        read_submission(path)


def test_missing_duplicate_header_and_empty(tmp_path):
    path = tmp_path / 's.csv'
    for rows in [[], [['n', 'x']], [HEADER, ['0', '2', '0', '0', '0', '0']],
                 [HEADER, ['0', '2', '0', '0', '0', '0'], ['1', '2', '0', '1', '0', '0']]]:
        with path.open('w', newline='') as f:
            csv.writer(f).writerows(rows)
        with pytest.raises(ValueError):
            read_submission(path)
    write_submission({}, path)
    assert read_submission(path) == {}
    assert validate_submission(path).invalid == 1000
    with pytest.raises(ValueError):
        write_submission({1: np.array([[np.nan, 0, 0]])}, path)


def test_store_merge_and_persistence(tmp_path):
    store = BestStore(tmp_path / 'best.csv')
    assert not store.merge(2, grid(2))
    a = grid(2)
    a[1, 0] += 1e-7
    assert store.merge(2, a)
    assert not store.merge(2, a)
    b = a.copy()
    b[1, 0] -= 5e-8
    assert store.merge(2, b)
    assert not store.merge(2, np.zeros((2, 3)))
    assert not store.merge(2, np.zeros((1, 3)))
    store.save()
    np.testing.assert_array_equal(BestStore(store.path).solutions[2], b)
    assert BestStore(tmp_path / 'touch.csv', min_gap=0).merge(2, grid(2))


def test_grid_official_first_thirty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / 'grid.csv'
    write_submission({n: grid(n) for n in range(1, 31)}, path)
    expected = sum(bounding_side(grid(n)) / math.sqrt(n) - 1 for n in range(1, 31))
    assert official_score(path, jobs=2, puzzle_ns=range(1, 31)) == pytest.approx(expected, abs=1e-12)
    with (tmp_path / 'submission_details.csv').open() as f:
        details = list(csv.DictReader(f))
    assert all(float(row['L']) > 0 for row in details[:30])
    assert validate_submission(path).invalid == 970


def test_invalid_geometry_penalty(tmp_path):
    path = tmp_path / 'overlap.csv'
    write_submission({2: np.zeros((2, 3))}, path)
    result = validate_submission(path)
    assert result.invalid == 1000
    assert result.scores[2] == pytest.approx(2 / math.sqrt(2) - 1)
