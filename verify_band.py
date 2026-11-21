"""Search and certify every member of the independent band construction table."""
import argparse
import csv
from multiprocessing import Pool
from pathlib import Path
import time

from sqpack.constructions.band import best_band_configs, build_band, envelope_score
from sqpack.hp import mp
from sqpack.metric_official import evaluate_single_puzzle


def check(cfg):
    start = time.perf_counter()
    packing = build_band(cfg.L, cfg.p, cfg.q, cfg.t).normalize().inflate('1e-20')
    rows = [dict(n=str(cfg.count), square_id=str(i), x=x, y=y, angle=a)
            for i, (x, y, a) in enumerate(packing.to_rows(45))]
    _, _, side, status = evaluate_single_puzzle(cfg.count, rows)
    if status == 'ok' and abs(mp.mpf(side) - mp.mpf(cfg.L)) > mp.mpf('1e-15'):
        status = 'L differs from config by more than 1e-15'
    return dict(n=cfg.count, L=cfg.L, p=cfg.p, q=cfg.q, t=cfg.t,
                official_L=side or '', status=status,
                runtime_seconds=f'{time.perf_counter() - start:.6f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--output', default='reports/band_verification.csv')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    start = time.perf_counter()
    configs = best_band_configs(refresh=args.refresh)
    search_runtime = time.perf_counter() - start
    print(f'Band configs: {len(configs)}; search runtime: {search_runtime:.3f}s', flush=True)
    with Pool(args.jobs) as pool:
        results = pool.map(check, configs.values())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    failures = [r for r in results if r['status'] != 'ok']
    print(f'Official verification: {len(results) - len(failures)} ok / {len(failures)} failed')
    for row in failures:
        print(row)
    print(f'Envelope(grid + band): {envelope_score(configs):.12f}')
    print(f'Table: {output}; total runtime: {time.perf_counter() - start:.3f}s')
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
