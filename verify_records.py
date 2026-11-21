"""Certify every local parsed record and selected derived packings.

Writes a durable per-record CSV table; no network access is required.
"""
import argparse
import csv
from multiprocessing import Pool
from pathlib import Path
import time

from sqpack.hp import mp
from sqpack.metric_official import evaluate_single_puzzle
from sqpack.records import read_parsed


def check(task):
    filename, transform = task
    record = read_parsed(filename)
    p = record.packing
    expected = mp.mpf(record.s)
    if transform == 'extend_twice':
        p = p.extend().extend()
        expected += 2
    elif transform == 'remove_to_1000':
        p = p.remove_to(1000)
        expected = None
    p = p.normalize().inflate('1e-20')
    n = len(p)
    rows = [dict(n=str(n), square_id=str(i), x=x, y=y, angle=a)
            for i, (x, y, a) in enumerate(p.to_rows(45))]
    start = time.perf_counter()
    _, _, side, reason = evaluate_single_puzzle(n, rows)
    runtime = time.perf_counter() - start
    if reason == 'ok' and expected is not None and abs(mp.mpf(side) - expected) > mp.mpf('1e-15'):
        reason = 'L differs from expected side by more than 1e-15'
    return dict(source=record.source, transform=transform, n=n, s=record.s,
                expected_s='' if expected is None else str(expected), official_L=side or '',
                status=reason, runtime_seconds=f'{runtime:.6f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parsed-dir', required=True)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--output', default='reports/record_verification.csv')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    paths = sorted(Path(args.parsed_dir).glob('square-*.txt'))
    if not paths:
        parser.error('no parsed records found')
    tasks = [(str(p), 'record') for p in paths]
    for name, transform in [('square-11.txt', 'extend_twice'),
                            ('square-1097_r1.txt', 'remove_to_1000')]:
        path = Path(args.parsed_dir) / name
        if not path.exists():
            parser.error(f'missing transform input {path}')
        tasks.append((str(path), transform))
    start = time.perf_counter()
    with Pool(args.jobs) as pool:
        results = pool.map(check, tasks)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    failures = [row for row in results if row['status'] != 'ok']
    print(f'Records: {sum(r["status"] == "ok" for r in results[:len(paths)])} ok / '
          f'{sum(r["status"] != "ok" for r in results[:len(paths)])} failed ({len(paths)} total)')
    for row in results[len(paths):] + failures:
        print(row)
    print(f'Table: {output}; total runtime: {time.perf_counter() - start:.3f}s')
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
