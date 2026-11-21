# Kaggle "Squares in Squares" solver

Entry tooling for <https://www.kaggle.com/competitions/squares-in-squares>:
pack n = 1..1000 non-overlapping unit squares (any rotation) into the smallest
axis-aligned square. Submission CSV columns: `id,n,square_id,x,y,angle`
(square centers, angle in radians; 500,500 rows). Per-puzzle score is
`L/sqrt(n) - 1` where L is the measured bounding-square side; total is the sum
(lower is better). Invalid/missing puzzles get `ceil(sqrt(n))/sqrt(n) - 1`.

## Competition timeline and prizes (from the competition page, Nov 2026)

| Item | Date / amount |
|---|---|
| Start | Oct 21, 2026 |
| Milestone 1 cut-off ($5,000) | Jan 17, 2027 (23:59 UTC; = Jan 18 07:59 CST) |
| Milestone 2 cut-off ($5,000) | Apr 18, 2027 |
| Entry / team-merger deadline | Jul 11, 2027 |
| Final submission deadline | Jul 18, 2027 (23:59 UTC; = Jul 19 07:59 CST) |
| Final prizes | 1st $12,000, 2nd $10,000, 3rd $8,000, 4th $5,000, 5th $5,000 |
| Total pool | $50,000 |

Check the Kaggle page for authoritative times. Nothing here joins or submits.

## Approach

1. **Grid baseline**: ceil(sqrt(n))^2 axis-aligned grid (score 28.9377).
2. **Diagonal-band family** (own construction, `sqpack/constructions/band.py`):
   axis-aligned staircases in two opposite corners with t rows of 45-degree
   squares along the diagonal; (p,q,t) and side searched exhaustively and
   bisected in 60-digit precision.
3. **Public best-known packings** (optional, `cache/records.json`): parsed from
   David Ellsworth's public page (Erich Friedman's catalogue); used only when
   enabled.
4. **Extension/removal closure**: any packing of c squares in side L yields
   packings for fewer squares (greedy removal, can shrink L) and, by adding an
   axis-aligned row+column, c+2*ceil(L)+1 squares in side L+1.
5. **Selection + certification**: for each n choose the smallest-L candidate,
   inflate gaps by 1e-20 in high precision, emit exact decimal strings, and
   certify with the vendored official metric; fall back to the next candidate
   (ultimately grid) if certification fails.

## Quick start

```sh
python -m pip install -r requirements.txt
python solve.py --no-records --out submissions/submission_own.csv   # own constructions only
python solve.py --out submissions/submission_records.csv            # + cached public records
python -m sqpack.validate submissions/submission_records.csv --official --jobs 8
```

Expected official totals: **24.1118** (own) and **23.0790** (with records);
see "Measured full runs" below.

## Tools

Install with `python -m pip install -r requirements.txt`.

```sh
python make_sample.py
pytest -q
python -m sqpack.validate data/sample_submission_grid.csv --official --jobs 8
```

The sample is a locally generated grid baseline, not a downloaded Kaggle file.
No Kaggle credentials or network access are used. The official metric writes
`submission_details.csv` in the current directory (or `/kaggle/working` when
that directory exists). Its source is vendored under Apache-2.0 from
<https://www.kaggle.com/code/metric/squares-in-squares>.

## API

- `sqpack.baseline.grid(n)`: `(n, 3)` center-x, center-y, radian-angle array.
- `sqpack.geometry.corners(arr)`, `bounding_side(arr)`: corners and measured L.
- `sqpack.geometry.check_overlaps(arr)`: `(min_signed_sat_gap, pairs)`.
  Negative gaps indicate overlap, zero indicates touching, positive indicates
  separation. This is a floating-point SAT gap, not Euclidean distance.
  Empty/singleton inputs have infinite gap. Geometry assumes finite `(n,3)`
  inputs; use `sqpack.io.validate_array` to validate external arrays first.
- `sqpack.io.write_submission(solutions, path)`, `read_submission(path)`:
  strict CSV serialization/deserialization. Partial puzzle sets are supported.
- `sqpack.io.read_submission_strings(path)`: same per-n arrays retaining exact
  submitted decimal strings and square_id ordering.
- `sqpack.validate.validate_submission(path)`: per-n sides/scores, total,
  improved count, invalid count. Missing/overlapping puzzles receive grid
  penalties; malformed CSV raises `ValueError`. CLI exits nonzero for either.
- `sqpack.validate.official_score(path, jobs=8, puzzle_ns=range(1,1001))`:
  executes the official metric's `score()` with decimal strings intact.
- `sqpack.store.BestStore(path='solutions/best.csv', min_gap=1e-9)`:
  `merge(n, arr)` returns acceptance, `save()` persists, `load()` reloads.
  Merge requires strict L improvement and the configured minimum SAT gap.
  The default intentionally rejects touching grid packings with n > 1;
  choose `min_gap=0` explicitly if contact is desired. Changes stay in memory
  until `save()`; an empty store does not create a file automatically.

The fast validator is not an interval certificate. Use positive gaps for
rotated layouts and `--official` before trusting a candidate. L is measured
translation-invariantly; no separate container constraints are imposed.

## High-precision public records

`sqpack.hp.Packing` uses an isolated 60-dps mpmath context, independent of the
official metric's precision changes. `from_strings`/`to_strings` retain high
precision; `to_rows(digits=45)` emits fixed decimal string triples (45 decimal
places by default, exact zero angles emitted as `"0"`). `side`, `bbox`,
`normalize`, `inflate`, `remove_to`, and `extend` operate without binary floats.
Transforms return new packings. Inflation scales only centers about the bbox
center, not the unit squares. `remove_to` greedily prefers bbox shrinkage.

```sh
python -m sqpack.records fetch
# Offline reuse of already downloaded / parsed source files:
python -m sqpack.records fetch --svg-dir /tmp/svgs --parsed-dir /tmp/parsed
python verify_records.py --parsed-dir /tmp/parsed --jobs 8
```

The fetcher downloads David Ellsworth's public page and listed square SVGs,
then invokes `parse_svg_packing.py` from a runtime clone of
<https://github.com/Davidebyzero/packing_squares_in_squares__tools>. That
repository has no license file; its code is **not** included in this project.
All downloads, parser outputs, and the clone live in gitignored `cache/` by
default. Supplying only `--parsed-dir` skips SVGs and network entirely.

`sqpack.records.load_records()` returns `dict[int, Record]` from
`cache/records.json`; each record has `n`, original `s` string, `packing`, and
source filename. Multiple records for the same count are reduced to the
smallest published side. Raw records can be exactly touching: normalize,
inflate by `1e-20`, and certify the formatted output before using them.
Credit goes to David Ellsworth, Erich Friedman, and the individual authors
identified on the source page. These records are external data, not parser code.

`verify_records.py` checks **all** parsed records, including alternate records
for duplicate counts, and writes `reports/record_verification.csv` with
published s, official L, status, and official-check runtime. It also checks
n=11 extended twice and n=1097 reduced to 1000. For records and extensions,
L must agree with the expected side to `1e-15` after `1e-20` inflation.

## Independent diagonal-band family

`sqpack.constructions.band` builds two wall-anchored staircases with centered
45-degree rows in between; it uses no external data. APIs:

- `band_count(L, p, q, t)`: high-precision count, zero for a band that cannot fit.
- `build_band(L, p, q, t)`: high-precision `Packing`, initially touching.
- `best_band_configs(Nmax=1000)`: count -> `BandConfig(count,L,p,q,t)`; L is a
  high-precision decimal string. Results are cached in `cache/band_configs.json`.
- `envelope_score(configs)`: float grid+band envelope total. A c-square packing
  is also a candidate for every n <= c by dropping squares.

The search exhaustively enumerates configs and **all floor-change events**
using numba, rather than coarse L samples that can miss short-lived counts.
Each row count simplifies to
`floor(sqrt(2)*L - 1 - abs((p-q)/sqrt(2) + 2*r + 1 - t))`.
Event thresholds and the feasibility threshold are integer combinations of
`1/2` and `1/sqrt(2)`; their spacing in the searched range is much larger than
float roundoff. The winning config for each count is then bisected at 60 dps
to within `1e-45` and nudged upward by `1e-44`. Floors use a `1e-48` tolerance.
Only counts <= Nmax strictly between consecutive grid counts are retained.

```sh
python verify_band.py --refresh --jobs 8
```

This times the uncached search (including compilation), certifies every table
entry after normalization and `1e-20` inflation, and writes
`reports/band_verification.csv`. `--refresh` bypasses the result cache.

## End-to-end certified solver

```sh
python solve.py --no-records --out submissions/submission_own.csv
python solve.py --out submissions/submission_records.csv
python -m sqpack.validate submissions/submission_records.csv --official --jobs 8
# Quick subset, alternate report location, configurable microscopic gaps:
python solve.py --nmax 100 --eps 1e-20 --jobs 8 --summary reports/subset.csv
```

The runner combines grid, band, optional records, and an optional
`solutions/best.csv` store (`--store PATH`). Records default to enabled when
`cache/records.json` exists; `--records` requires that cache, and `--no-records`
disables it. The runner performs no downloads. Band metadata is computed if
its cache is absent or incompatible with nmax.

Extension closure is computed on counts/sides without building packings. Its
default cap is `nmax + 2*ceil(sqrt(nmax)) + 1`, allowing a one-layer overshoot;
original sources exceeding the cap are still eligible for removal. Candidates
are ordered by side, count, then extension count. Workers lazily build winners,
extend, remove squares, normalize and inflate, and certify the exact emitted
decimal strings. If certification fails, they try the next candidate, with grid
as a safe fallback. Submission serialization never passes coordinates through
binary floats. Nothing is written until every puzzle has been certified.

Output defaults to `submission.csv`; `--out` can place it in gitignored
`submissions/`. The default `reports/solution_summary.csv` includes n, official
L and score, detailed source, transform chain, and fallback count. Use
`--summary` to retain separate reports for multiple runs. Printed per-source
counts group sources into grid/band/record/store.

The validator resolves tiny negative float SAT gaps using original-decimal
official certification. With `--official`, that resolution is deferred to the
full multiprocessing metric run; tiny projection roundoff alone is not treated
as proof of overlap. Clearly overlapping float configurations are still rejected.

### Measured full runs (n=1..1000, 8 jobs, eps=1e-20)

The metric is **lower-is-better**; the grid total is **28.937669916551**.
Both runs certified all puzzles without any candidate fallbacks, and both
submissions subsequently passed the independent official CLI validation.

| Sources | Official total | Solve runtime | Validation runtime | Improved over grid | Invalid |
|---|---:|---:|---:|---:|---:|
| Own (grid + band + extensions/removals) | 24.111836619133 | 473.666s | 292.542s | 388 | 0 |
| Own + cached public records | 23.078964214894 | 190.015s | 224.924s | 435 | 0 |

Selected source counts were `band=388, grid=612` for the own run, and
`band=161, grid=32, record=807` for the records run. A selected record need not
improve the grid score: ties prefer the candidate with fewer squares to remove.
The actual own score slightly improves the count/side envelope estimate because
greedy removal can reduce the measured bounding side.

Submissions are `submissions/submission_own.csv` and
`submissions/submission_records.csv` (gitignored). Separate per-n summaries are
retained as `reports/solution_summary_own.csv` and
`reports/solution_summary_records.csv`; the default `solution_summary.csv`
contains the latest records-enabled run. Logs are `reports/solve_own.log`,
`reports/solve_records.log`, `reports/validate_own.log`, and
`reports/validate_records.log`. Tests: **63 passed in 31.37s**
(`reports/pytest_solve.txt`). Runtimes are measured wall times, not guarantees.
