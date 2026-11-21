import csv
import json

import pytest

import solve as runner
from sqpack.baseline import grid
from sqpack.constructions.band import build_band
from sqpack.hp import mp
from sqpack.io import read_submission_strings, write_submission
from sqpack.validate import validate_submission


def grids(nmax):
    return [runner.Source('grid', k * k, str(k), 'grid', k)
            for k in range(1, int(nmax ** .5) + 2)]


def test_extension_closure_and_order():
    source = runner.Source('grid', 1, '1', 'grid', 1)
    candidates = runner.extension_closure([source], nmax=10)
    assert [(c.count, c.L, c.extensions) for c in candidates] == [
        (1, '1', 0), (4, '2.0', 1), (9, '3.0', 2), (16, '4.0', 3)]
    # Smaller count wins a side tie, then fewer extensions for the same count.
    candidates = runner.extension_closure(grids(10), nmax=10)
    fours = [c for c in candidates if c.L in ('2', '2.0')]
    assert fours[0].count == 4 and fours[0].extensions == 0


def test_sources_records_flags_and_store(tmp_path):
    band = tmp_path / 'band.json'
    cache = tmp_path / 'records.json'
    store = tmp_path / 'best.csv'
    cache.write_text(json.dumps({'1': dict(s='1', source='square-1.txt', squares=[['0', '0', '0']])}))
    write_submission({2: grid(2)}, store)
    sources = runner.load_sources(3, records_path=cache, band_path=band, store_path=store)
    assert {s.kind for s in sources} == {'grid', 'record', 'store'}
    assert all(s.kind != 'record' for s in runner.load_sources(3, records=False,
               records_path=cache, band_path=band, store_path=None))
    cache.unlink()
    assert all(s.kind != 'record' for s in runner.load_sources(3, records=None,
               records_path=cache, band_path=band, store_path=None))
    with pytest.raises(FileNotFoundError):
        runner.load_sources(3, records=True, records_path=cache, band_path=band, store_path=None)


def test_solver_certified_fallback_and_roundtrip(tmp_path):
    invalid = runner.Source('store', 4, '1', 'bad', [['0', '0', '101']] * 4)
    out = tmp_path / 'submission.csv'
    summary = tmp_path / 'summary.csv'
    total = runner.solve(nmax=4, sources=[invalid, *grids(4)], out=out,
                         summary_path=summary, jobs=1)
    assert total == pytest.approx(2 / 2 ** .5 + 2 / 3 ** .5 - 2)
    result = read_submission_strings(out)
    assert list(result) == [1, 2, 3, 4]
    assert validate_submission(out).invalid == 996
    with summary.open() as f:
        rows = list(csv.DictReader(f))
    assert all(r['source'] == 'grid' for r in rows)
    assert int(rows[0]['fallbacks']) == 0
    assert all(int(r['fallbacks']) >= 1 for r in rows[1:])
    assert 'remove_to(2)' in rows[1]['transforms']


def test_solver_multiprocessing_and_high_precision_strings(tmp_path):
    L = 3 + 1 / mp.sqrt(2)
    source = runner.Source('band', 10, mp.nstr(L, 60), 'band(2,2,1)', (2, 2, 1))
    out = tmp_path / 'sub.csv'
    summary = tmp_path / 'summary.csv'
    runner.solve(nmax=10, sources=[*grids(10), source], out=out, summary_path=summary, jobs=2)
    strings = read_submission_strings(out)
    assert any(len(value) > 30 for row in strings[10] for value in row)
    with summary.open() as f:
        summaries = list(csv.DictReader(f))
    assert summaries[-1]['source'] == 'band(2,2,1)'
    assert float(summaries[-1]['L']) == pytest.approx(float(L))


def test_uncertain_fast_contacts_use_original_decimals(tmp_path, monkeypatch):
    path = tmp_path / 'tiny.csv'
    write_submission({2: [[.5, .5, 0], [1.5 - 1e-10, .5, 0]]}, path)
    assert validate_submission(path).invalid == 1000
    assert validate_submission(path, defer_uncertain=True).invalid == 999
    # A rotated packing with microscopic certified gaps must not be rejected
    # merely because its binary-float projection has a tiny negative gap.
    triples = build_band(3 + 1 / mp.sqrt(2), 2, 2, 1).inflate().to_rows()
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'n', 'square_id', 'x', 'y', 'angle'])
        writer.writerows([i, 10, i, *row] for i, row in enumerate(triples))
    monkeypatch.setattr('sqpack.validate.check_overlaps', lambda a: (-1e-15, [(0, 1)]))
    assert validate_submission(path).invalid == 999


def test_no_certified_candidate_does_not_write(tmp_path):
    out = tmp_path / 'absent.csv'
    with pytest.raises(RuntimeError, match='all candidates'):
        runner.solve(nmax=1, sources=[], out=out, summary_path=tmp_path / 's.csv', jobs=1)
    assert not out.exists()


@pytest.mark.parametrize('kwargs', [dict(nmax=0), dict(nmax=1001), dict(jobs=0),
                                   dict(eps='-1'), dict(eps='nan')])
def test_bad_options(kwargs):
    with pytest.raises(ValueError):
        runner.solve(**kwargs)
