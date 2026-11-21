import json
from pathlib import Path

import mpmath
import pytest

from sqpack.hp import Packing, fixed, mp
from sqpack.io import DECIMAL
from sqpack.metric_official import evaluate_single_puzzle
from sqpack.records import _Links, fetch, load_records, read_parsed
from sqpack import records as records_module


def certify(packing):
    n = len(packing)
    rows = [dict(n=str(n), square_id=str(i), x=x, y=y, angle=a)
            for i, (x, y, a) in enumerate(packing.to_rows())]
    return evaluate_single_puzzle(n, rows)


def test_hp_normalize_inflate_and_precision():
    p = Packing.from_strings([['-1.23456789012345678901234567890123456789', '2', '0'],
                              ['-0.23456789012345678901234567890123456789', '2', '0']])
    assert p.side() == 2
    assert p.normalize().bbox() == (0, 0, 2, 1)
    inflated = p.normalize().inflate('1e-20')
    assert abs(inflated.side() - mp.mpf('2.00000000000000000001')) < mp.mpf('1e-58')
    assert p.side() == 2
    assert certify(inflated)[3] == 'ok'
    restored = Packing.from_strings(p.to_strings())
    assert abs(restored.squares[0][0] - p.squares[0][0]) < mp.mpf('1e-59')
    with mpmath.workdps(10):
        assert p.side() == 2


def test_hp_rotated_side_and_format():
    p = Packing([(0, 0, mp.pi / 4)])
    assert abs(p.side() - mp.sqrt(2)) < mp.mpf('1e-59')
    for row in p.to_rows():
        assert all(DECIMAL.fullmatch(v) and 'e' not in v.lower() for v in row)
    assert fixed('1e-44') == '0.' + '0' * 43 + '1'
    assert fixed('-1e-50') == '0'
    assert fixed('100') == '100'
    assert fixed('12.4', 0) == '12'
    assert Packing([(0, 0, 0)]).to_rows()[0][2] == '0'


def test_extend_remove_and_certify():
    p = Packing([('0.5', '0.5', '0')])
    extended = p.extend().extend()
    assert len(extended) == 9
    assert extended.side() == 3
    assert certify(extended.inflate())[3] == 'ok'
    removed = extended.remove_to(4)
    assert len(removed) == 4
    assert removed.side() <= extended.side()
    assert certify(removed.inflate())[3] == 'ok'
    assert len(extended.remove_to(1)) == 1
    assert extended.remove_to(9).to_rows() == extended.to_rows()
    tilted = Packing([('0.5', '0.5', mp.pi / 4)])
    q = tilted.extend()
    assert len(q) == 4
    assert abs(q.side() - (tilted.side() + 1)) < mp.mpf('1e-58')
    assert certify(q.inflate())[3] == 'ok'


@pytest.mark.parametrize('squares', [[], [(0, 0)], [(0, 0, 'nan')]])
def test_bad_packing(squares):
    with pytest.raises(ValueError):
        Packing(squares)


def test_bad_transforms_and_format():
    p = Packing([(0, 0, 0)])
    for n in [0, 2, 0.5]:
        with pytest.raises(ValueError):
            p.remove_to(n)
    for eps in ['-1', 'nan', 'inf']:
        with pytest.raises(ValueError):
            p.inflate(eps)
    for value, digits in [('nan', 45), ('0', -1), ('0', 61)]:
        with pytest.raises(ValueError):
            fixed(value, digits)


def test_parsed_fetch_best_and_cache(tmp_path):
    parsed = tmp_path / 'parsed'
    parsed.mkdir()
    (parsed / 'square-2.txt').write_text('Final s: 2.1\nSquare 1: x=0, y=0, deg=0\n'
                                        'Square 2: x=1.1, y=0, deg=90\n')
    (parsed / 'square-2_r1.txt').write_text('Final s: 2\nSquare 1: x=0, y=0, deg=0\n'
                                           'Square 2: x=1, y=0, deg=90\n')
    record = read_parsed(parsed / 'square-2.txt')
    assert record.packing.squares[1][2] == mp.pi / 2
    cache = tmp_path / 'cache'
    records = fetch(cache, parsed_dir=parsed)
    assert records[2].s == '2'
    assert records[2].source == 'square-2_r1.txt'
    loaded = load_records(cache / 'records.json')
    assert loaded[2].packing.to_rows() == records[2].packing.to_rows()
    svgs = tmp_path / 'svgs'
    svgs.mkdir()
    (svgs / 'square-2.svg').write_text('not used: parsed output already exists')
    assert fetch(cache, svg_dir=svgs, parsed_dir=parsed)[2].s == '2.1'


def test_bad_parsed_and_empty_cache(tmp_path):
    path = tmp_path / 'bad.txt'
    for text in ['', 'Final s: 1\n', 'Final s: 1\nSquare 2: x=0, y=0, deg=0\n']:
        path.write_text(text)
        with pytest.raises(ValueError):
            read_parsed(path)
    with pytest.raises(ValueError, match='no parsed records'):
        fetch(tmp_path / 'cache', parsed_dir=tmp_path)
    path.write_text(json.dumps({'2': {'s': '1', 'source': 'bad', 'squares': [['0', '0', '0']]}}))
    with pytest.raises(ValueError):
        load_records(path)


def test_page_links():
    parser = _Links()
    parser.feed('<a href="square-11.svg">11</a><img src="square-199_r1.svg">'
                '<a href="else.svg">skip</a>')
    assert parser.urls == {'https://kingbird.myphotos.cc/packing/square-11.svg',
                           'https://kingbird.myphotos.cc/packing/square-199_r1.svg'}


def test_download_and_runtime_parser_invocation(tmp_path, monkeypatch):
    calls = []

    def download(url, path):
        if url == records_module.PAGE:
            path.write_text('<a href="square-1.svg">one</a>')
        else:
            path.write_text('<svg/>')

    def run(args, check):
        assert check is True
        calls.append(args)
        if args[:2] == ['git', 'clone']:
            (tmp_path / 'ellsworth_tools').mkdir()
        else:
            assert args[-2] == '0'
            assert args[1].endswith('ellsworth_tools/parse_svg_packing.py')
            Path(args[-1]).write_text('Final s: 1\nSquare 1: x=0, y=0, deg=0\n')

    monkeypatch.setattr(records_module, '_download', download)
    monkeypatch.setattr(records_module.subprocess, 'run', run)
    assert fetch(tmp_path)[1].s == '1'
    assert len(calls) == 2
    fetch(tmp_path)
    assert len(calls) == 2
