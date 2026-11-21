"""Best feasible packings, persisted in submission format."""
from pathlib import Path
import math

from .geometry import bounding_side, check_overlaps
from .io import read_submission, validate_array, write_submission


class BestStore:
    def __init__(self, path='solutions/best.csv', min_gap=1e-9):
        if not math.isfinite(min_gap) or min_gap < 0:
            raise ValueError('min_gap must be finite and nonnegative')
        self.path = Path(path)
        self.min_gap = min_gap
        self.solutions = {}
        if self.path.exists():
            self.load()

    def load(self):
        self.solutions = read_submission(self.path)
        for n, a in self.solutions.items():
            if check_overlaps(a)[0] < self.min_gap:
                raise ValueError(f'n={n}: stored packing violates min_gap')
        return self.solutions

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_submission(self.solutions, self.path)

    def merge(self, n, arr):
        """Accept a feasible strict improvement; return whether accepted."""
        try:
            a = validate_array(n, arr)
        except (ValueError, TypeError):
            return False
        if check_overlaps(a)[0] < self.min_gap:
            return False
        if n in self.solutions and bounding_side(a) >= bounding_side(self.solutions[n]):
            return False
        self.solutions[n] = a.copy()
        return True
