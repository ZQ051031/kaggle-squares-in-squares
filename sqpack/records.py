"""Public square-packing records from David Ellsworth's page (credit also
Erich Friedman and the individual record authors credited there).

The unlicensed Ellsworth parser is invoked from a runtime cache clone, never
vendored. Cached coordinates are raw records; inflate before certification.
"""
import argparse
from dataclasses import dataclass
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

from .hp import Packing, mp

PAGE = 'https://kingbird.myphotos.cc/packing/squares_in_squares.html'
TOOLS = 'https://github.com/Davidebyzero/packing_squares_in_squares__tools'
NUMBER = r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?'
SQUARE = re.compile(rf'^Square (\d+): x=({NUMBER}), y=({NUMBER}), deg=({NUMBER})\s*$', re.M)


@dataclass(frozen=True)
class Record:
    n: int
    s: str
    packing: Packing
    source: str


def read_parsed(path):
    """Read an external parser output, converting degrees at 60 dps."""
    path = Path(path)
    text = path.read_text()
    side = re.search(rf'^Final s:\s*({NUMBER})\s*$', text, re.M)
    rows = SQUARE.findall(text)
    if not side or not rows or [int(row[0]) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f'{path}: missing side or nonsequential square records')
    s = mp.mpf(side[1])
    if not mp.isfinite(s) or s < 1:
        raise ValueError(f'{path}: invalid side')
    packing = Packing((x, y, mp.mpf(deg) * mp.pi / 180) for _, x, y, deg in rows)
    return Record(len(packing), side[1], packing, path.name)


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = set()

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('href', 'src', 'data') and value:
                url = urljoin(PAGE, value)
                if re.fullmatch(r'square-[\w.-]+\.svg', Path(urlparse(url).path).name):
                    self.urls.add(url)


def _download(url, path):
    with urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())


def fetch(cache_dir='cache', svg_dir=None, parsed_dir=None):
    """Fetch/parse records, or use local SVGs and/or parsed texts offline."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    parsed = Path(parsed_dir) if parsed_dir is not None else cache / 'parsed'
    parsed.mkdir(parents=True, exist_ok=True)
    if parsed_dir is not None and svg_dir is None:
        paths = sorted(parsed.glob('square-*.txt'))
    else:
        svgs = Path(svg_dir) if svg_dir is not None else cache / 'svgs'
        svgs.mkdir(parents=True, exist_ok=True)
        if svg_dir is None:
            page = cache / 'squares_in_squares.html'
            _download(PAGE, page)
            links = _Links()
            links.feed(page.read_text())
            if not links.urls:
                raise ValueError('no square SVG links on records page')
            for url in sorted(links.urls):
                target = svgs / Path(urlparse(url).path).name
                if not target.exists():
                    _download(url, target)
        paths = []
        for svg in sorted(svgs.glob('square-*.svg')):
            output = parsed / (svg.stem + '.txt')
            if not output.exists():
                tools = cache / 'ellsworth_tools'
                if not tools.exists():
                    subprocess.run(['git', 'clone', TOOLS, str(tools)], check=True)
                subprocess.run([sys.executable, str(tools / 'parse_svg_packing.py'),
                                str(svg), '0', str(output)], check=True)
            paths.append(output)
    if not paths:
        raise ValueError('no parsed records found')
    records = {}
    for path in paths:
        record = read_parsed(path)
        if record.n not in records or mp.mpf(record.s) < mp.mpf(records[record.n].s):
            records[record.n] = record
    payload = {str(n): {'s': r.s, 'source': r.source, 'squares': r.packing.to_strings()}
               for n, r in sorted(records.items())}
    (cache / 'records.json').write_text(json.dumps(payload, indent=2) + '\n')
    return records


def load_records(path='cache/records.json'):
    """Return n -> Record, preserving decimal coordinates without floats."""
    records = {}
    for key, value in json.loads(Path(path).read_text()).items():
        n = int(key)
        packing = Packing.from_strings(value['squares'])
        if len(packing) != n or not mp.isfinite(mp.mpf(value['s'])) or mp.mpf(value['s']) < 1:
            raise ValueError(f'invalid cached record n={n}')
        records[n] = Record(n, value['s'], packing, value['source'])
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('fetch')
    command.add_argument('--cache-dir', default='cache')
    command.add_argument('--svg-dir')
    command.add_argument('--parsed-dir')
    args = parser.parse_args()
    records = fetch(args.cache_dir, args.svg_dir, args.parsed_dir)
    print(f'Cached {len(records)} best-per-n records in {args.cache_dir}/records.json')


if __name__ == '__main__':
    main()
