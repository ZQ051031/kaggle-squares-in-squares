"""Strict submission CSV serialization, preserving decimal strings on read."""
import csv
from pathlib import Path
import re

import numpy as np

HEADER = ['id', 'n', 'square_id', 'x', 'y', 'angle']
DECIMAL = re.compile(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$', re.ASCII)


def validate_array(n, arr):
    a = np.asarray(arr, dtype=float)
    if not isinstance(n, (int, np.integer)) or not 1 <= n <= 1000:
        raise ValueError('n must be in 1..1000')
    if a.shape != (n, 3) or not np.isfinite(a).all():
        raise ValueError(f'n={n}: expected finite array ({n}, 3)')
    if (np.abs(a) > [1e4, 1e4, 100]).any():
        raise ValueError(f'n={n}: coordinate or angle out of bounds')
    return a


def write_submission(solutions, path):
    arrays = [(n, validate_array(n, arr)) for n, arr in sorted(solutions.items())]
    with Path(path).open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        row_id = 0
        for n, a in arrays:
            for square_id, row in enumerate(a):
                values = [repr(float(v)) for v in row]
                if not all(len(v) <= 100 and DECIMAL.fullmatch(v) for v in values):
                    raise ValueError('invalid decimal literal')
                writer.writerow([row_id, n, square_id, *values])
                row_id += 1


def read_submission_strings(path):
    """Return n -> (n,3) string array, ordered by square_id; validate format."""
    groups = {}
    with Path(path).open(newline='') as f:
        reader = csv.reader(f)
        if next(reader, None) != HEADER:
            raise ValueError('header must be ' + ','.join(HEADER))
        for expected_id, row in enumerate(reader):
            if len(row) != 6:
                raise ValueError(f'row {expected_id}: expected six fields')
            if any(not re.fullmatch(r'[0-9]+', v) for v in row[:3]):
                raise ValueError('ids and n must be unsigned integers')
            row_id, n, sid = map(int, row[:3])
            if row_id != expected_id or not 1 <= n <= 1000 or not 0 <= sid < n:
                raise ValueError(f'row {expected_id}: invalid id, n or square_id')
            if any(len(v) > 100 or not DECIMAL.fullmatch(v) for v in row[3:]):
                raise ValueError(f'row {expected_id}: invalid decimal')
            if sid in groups.setdefault(n, {}):
                raise ValueError(f'n={n}: duplicate square_id')
            groups[n][sid] = row[3:]
    result = {}
    for n, rows in groups.items():
        if set(rows) != set(range(n)):
            raise ValueError(f'n={n}: incorrect count or square_id set')
        result[n] = np.array([rows[i] for i in range(n)])
        validate_array(n, result[n])
    return result


def read_submission(path):
    return {n: a.astype(float) for n, a in read_submission_strings(path).items()}
