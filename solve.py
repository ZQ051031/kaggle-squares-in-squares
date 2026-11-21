"""Build and certify the grid/band/record/store removal-and-extension envelope."""
import argparse
from collections import Counter
import csv
from dataclasses import dataclass, replace
from functools import lru_cache
import math
from multiprocessing import Pool
from pathlib import Path
import tempfile
import time

from sqpack.constructions.band import best_band_configs, build_band
from sqpack.hp import Packing, mp
from sqpack.io import HEADER, read_submission_strings
from sqpack.metric_official import evaluate_single_puzzle
from sqpack.records import load_records


@dataclass(frozen=True)
class Source:
    kind: str
    count: int
    L: str
    name: str
    payload: object


@dataclass(frozen=True)
class Candidate:
    base: int
    count: int
    L: str
    extensions: int = 0


def load_sources(nmax=1000, records=None, records_path='cache/records.json',
                 band_path='cache/band_configs.json', store_path='solutions/best.csv'):
    """Load candidate metadata, not constructed grid/band packings.

    records=None enables cached records automatically; True requires the cache.
    """
    sources = []
    for k in range(1, math.isqrt(nmax - 1) + 2):
        sources.append(Source('grid', k * k, str(k), 'grid', k))
    for c, cfg in best_band_configs(nmax, band_path).items():
        sources.append(Source('band', c, cfg.L, f'band({cfg.p},{cfg.q},{cfg.t})',
                              (cfg.p, cfg.q, cfg.t)))
    path = Path(records_path)
    if records is True or (records is None and path.exists()):
        for c, record in load_records(path).items():
            sources.append(Source('record', c, record.s,
                                  f'record:{Path(record.source).stem}', record.packing.to_strings()))
    if store_path is not None and Path(store_path).exists():
        for c, strings in read_submission_strings(store_path).items():
            packing = Packing.from_strings(strings.tolist())
            sources.append(Source('store', c, mp.nstr(packing.side(), 60),
                                  f'store:{Path(store_path).name}:n={c}', strings.tolist()))
    return sources


def extension_closure(sources, nmax=1000, margin=None):
    """Expand metadata only, retaining one-step overshoot beyond nmax by default."""
    if margin is None:
        margin = 2 * (math.isqrt(nmax - 1) + 1) + 1
    if margin < 0:
        raise ValueError('margin must be nonnegative')
    candidates = []
    limit = nmax + margin
    for index, source in enumerate(sources):
        candidate = Candidate(index, source.count, source.L)
        candidates.append(candidate)
        while True:
            side = mp.mpf(candidate.L)
            count = candidate.count + int(mp.floor(side + 1)) + int(mp.floor(side))
            if count > limit:
                break
            candidate = replace(candidate, count=count, L=mp.nstr(side + 1, 60),
                                extensions=candidate.extensions + 1)
            candidates.append(candidate)
    return sorted(candidates, key=lambda c: (mp.mpf(c.L), c.count, c.extensions, c.base))


_sources = []
_candidates = []
_eps = '1e-20'


def _initialize(sources, candidates, eps):
    global _sources, _candidates, _eps
    _sources, _candidates, _eps = sources, candidates, eps
    _build.cache_clear()


@lru_cache(maxsize=32)
def _build(base, extensions):
    source = _sources[base]
    if extensions:
        return _build(base, extensions - 1).extend()
    if source.kind == 'grid':
        k = source.payload
        half = mp.mpf('.5')
        return Packing((i + half, j + half, 0) for i in range(k) for j in range(k))
    if source.kind == 'band':
        return build_band(source.L, *source.payload)
    return Packing.from_strings(source.payload)


def _solve_one(n):
    failures = []
    for candidate in _candidates:
        if candidate.count < n:
            continue
        source = _sources[candidate.base]
        try:
            packing = _build(candidate.base, candidate.extensions)
            if len(packing) != candidate.count:
                raise ValueError('constructed count differs from metadata')
            packing = packing.remove_to(n).normalize().inflate(_eps)
            triples = packing.to_rows(45)
            rows = [dict(n=str(n), square_id=str(i), x=x, y=y, angle=a)
                    for i, (x, y, a) in enumerate(triples)]
            _, score, side, reason = evaluate_single_puzzle(n, rows)
            if reason != 'ok' or side is None:
                raise ValueError(reason)
        except Exception as exc:
            failures.append(f'{source.name}:extend^{candidate.extensions}: {exc}')
            continue
        chain = []
        if candidate.extensions:
            chain.append(f'extend^{candidate.extensions}')
        if candidate.count != n:
            chain.append(f'remove_to({n})')
        chain.extend(('normalize', f'inflate({_eps})'))
        summary = dict(n=n, L=side, score=score, source=source.name,
                       transforms=' -> '.join(chain), source_count=source.count,
                       extended_count=candidate.count, fallbacks=len(failures))
        return triples, summary, source.kind, failures
    raise RuntimeError(f'n={n}: all candidates including grid failed: {failures}')


def _write_results(results, out, summary_path):
    """Serialize the already certified decimal strings, never round via floats."""
    out, summary_path = Path(out), Path(summary_path)
    if out.resolve() == summary_path.resolve():
        raise ValueError('submission and summary paths must differ')
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', newline='', dir=out.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            row_id = 0
            for triples, summary, _, _ in results:
                for sid, row in enumerate(triples):
                    writer.writerow([row_id, summary['n'], sid, *row])
                    row_id += 1
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(out)
    with summary_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0][1]))
        writer.writeheader()
        writer.writerows(result[1] for result in results)


def solve(nmax=1000, out='submission.csv', summary_path='reports/solution_summary.csv',
          jobs=8, eps='1e-20', records=None, records_path='cache/records.json',
          band_path='cache/band_configs.json', store_path='solutions/best.csv', sources=None):
    """Certify 1..nmax and write nothing until every puzzle has a valid packing."""
    if not isinstance(nmax, int) or not 1 <= nmax <= 1000:
        raise ValueError('nmax must be an integer in 1..1000')
    if not isinstance(jobs, int) or jobs < 1:
        raise ValueError('jobs must be a positive integer')
    eps = mp.mpf(eps)
    if not mp.isfinite(eps) or eps < 0:
        raise ValueError('eps must be finite and nonnegative')
    start = time.perf_counter()
    if sources is None:
        sources = load_sources(nmax, records, records_path, band_path, store_path)
    candidates = extension_closure(sources, nmax)
    print(f'Sources: {len(sources)}; candidates including extensions: {len(candidates)}', flush=True)
    initargs = (sources, candidates, mp.nstr(eps, 60))
    if jobs == 1:
        _initialize(*initargs)
        results = [_solve_one(n) for n in range(1, nmax + 1)]
    else:
        with Pool(jobs, initializer=_initialize, initargs=initargs) as pool:
            results = list(pool.imap(_solve_one, range(1, nmax + 1), chunksize=8))
    _write_results(results, out, summary_path)
    total = sum(result[1]['score'] for result in results)
    counts = dict(sorted(Counter(result[2] for result in results).items()))
    fallbacks = sum(result[1]['fallbacks'] for result in results)
    print(f'Total score: {total:.12f}; per-source counts: {counts}; fallbacks: {fallbacks}')
    for _, summary, _, failures in results:
        for failure in failures:
            print(f'Fallback n={summary["n"]}: {failure}')
    print(f'Wrote {out}; summary: {summary_path}; runtime: {time.perf_counter() - start:.3f}s')
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nmax', type=int, default=1000)
    parser.add_argument('--out', default='submission.csv')
    parser.add_argument('--summary', default='reports/solution_summary.csv')
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--eps', default='1e-20')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--records', dest='records', action='store_true')
    group.add_argument('--no-records', dest='records', action='store_false')
    parser.set_defaults(records=None)
    parser.add_argument('--store', default='solutions/best.csv')
    args = parser.parse_args()
    try:
        solve(nmax=args.nmax, out=args.out, summary_path=args.summary, jobs=args.jobs,
              eps=args.eps, records=args.records, store_path=args.store)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'Solve failed: {exc}\n')


if __name__ == '__main__':
    main()
