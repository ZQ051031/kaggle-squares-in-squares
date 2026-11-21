"""Validate submission format/geometry and optionally certify with the metric."""
import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import time

import pandas as pd

from . import metric_official
from .baseline import grid_total
from .geometry import bounding_side, check_overlaps
from .io import read_submission_strings


@dataclass
class ValidationResult:
    total: float
    improved: int
    invalid: int
    sides: dict
    scores: dict


def validate_submission(path, defer_uncertain=False):
    """Missing/overlapping puzzles get grid penalties and count as invalid.

    Malformed CSV raises ValueError. Returned sides include present puzzles;
    scores include all 1000 puzzles. Tiny negative float SAT gaps are certified
    using the original decimals, unless deferred to a subsequent official run.
    """
    puzzles = read_submission_strings(path)
    sides, scores = {}, {}
    invalid = improved = 0
    for n in range(1, 1001):
        baseline = (math.isqrt(n - 1) + 1) / math.sqrt(n) - 1
        scores[n] = baseline
        if n not in puzzles:
            invalid += 1
            continue
        arr = puzzles[n].astype(float)
        sides[n] = bounding_side(arr)
        gap, pairs = check_overlaps(arr)
        if pairs:
            if gap < -1e-8:
                invalid += 1
                continue
            if not defer_uncertain:
                rows = [dict(n=str(n), square_id=str(i), x=x, y=y, angle=a)
                        for i, (x, y, a) in enumerate(puzzles[n])]
                _, _, side, reason = metric_official.evaluate_single_puzzle(n, rows)
                if reason != 'ok':
                    invalid += 1
                    continue
                sides[n] = float(side)
        scores[n] = sides[n] / math.sqrt(n) - 1
        improved += scores[n] < baseline - 1e-12
    return ValidationResult(sum(scores.values()), improved, invalid, sides, scores)


def official_score(path, jobs=8, puzzle_ns=range(1, 1001)):
    """Run the unmodified official entry point (writes submission_details.csv)."""
    if jobs < 1:
        raise ValueError('jobs must be positive')
    metric_official.KAGGLE_CORES = jobs
    ns = list(puzzle_ns)
    solution = pd.DataFrame({'id': range(len(ns)), 'n': ns})
    submission = pd.read_csv(path, dtype=str, keep_default_na=False)
    return metric_official.score(solution, submission, 'id')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('submission')
    parser.add_argument('--official', action='store_true')
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    start = time.perf_counter()
    try:
        if args.jobs < 1:
            raise ValueError('jobs must be positive')
        result = validate_submission(args.submission, defer_uncertain=args.official)
        if args.official:
            certified = official_score(args.submission, args.jobs)
            if not math.isclose(result.total, certified, rel_tol=0, abs_tol=1e-9):
                raise ValueError(f'official total {certified:.12f} != fast total {result.total:.12f}')
            # Matching a grid penalty cannot distinguish an invalid baseline.
            output_dir = Path('/kaggle/working')
            if not output_dir.is_dir():
                output_dir = Path('.')
            details = output_dir / 'submission_details.csv'
            with details.open(newline='') as f:
                rejected = sum(float(row['L']) == 0 for row in csv.DictReader(f))
            result.invalid = max(result.invalid, rejected)
            print(f'Official total: {certified:.12f} (agrees within 1e-9)')
        print(f'Total score: {result.total:.12f}; grid: {grid_total():.12f}; '
              f'improved: {result.improved}; invalid: {result.invalid}; '
              f'runtime: {time.perf_counter() - start:.3f}s')
        return int(result.invalid > 0)
    except (ValueError, OSError, AssertionError) as exc:
        print(f'Validation failed: {exc}; runtime: {time.perf_counter() - start:.3f}s')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
