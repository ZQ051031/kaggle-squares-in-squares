"""Reproduce the public submission format with exact grid packings."""
from pathlib import Path

from sqpack.baseline import grid
from sqpack.io import write_submission


def main():
    path = Path(__file__).parent / 'data' / 'sample_submission_grid.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    write_submission({n: grid(n) for n in range(1, 1001)}, path)
    print(f'Wrote {path} (500500 squares)')


if __name__ == '__main__':
    main()
