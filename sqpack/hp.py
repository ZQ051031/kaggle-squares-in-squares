"""Immutable 60-decimal-digit packings; transforms never resize unit squares."""
import mpmath

mp = mpmath.mp.clone()
mp.dps = 60


def fixed(value, digits=45):
    """Round to fixed decimal places, without exponent notation or signed zero."""
    if not isinstance(digits, int) or not 0 <= digits <= 60:
        raise ValueError('digits must be an integer in 0..60')
    value = mp.mpf(value)
    if not mp.isfinite(value):
        raise ValueError('nonfinite coordinate')
    scale = 10 ** digits
    units = int(mp.nint(value * scale))
    sign = '-' if units < 0 else ''
    whole, fraction = divmod(abs(units), scale)
    text = f'{sign}{whole}.{fraction:0{digits}d}' if digits else f'{sign}{whole}'
    return text.rstrip('0').rstrip('.') if '.' in text else text


class Packing:
    def __init__(self, squares):
        self.squares = tuple(tuple(mp.mpf(v) for v in row) for row in squares)
        if not self.squares or any(len(row) != 3 or not all(mp.isfinite(v) for v in row)
                                   for row in self.squares):
            raise ValueError('expected nonempty finite (n,3) packing')

    def __len__(self):
        return len(self.squares)

    @classmethod
    def from_strings(cls, triples):
        return cls(triples)

    def to_strings(self):
        """Roundtrip strings retaining all 60 significant digits."""
        return [[mp.nstr(v, 60) for v in row] for row in self.squares]

    def to_rows(self, digits=45):
        """String triples with fixed decimals (digits decimal places)."""
        return [[fixed(v, digits) for v in row] for row in self.squares]

    def _bounds(self):
        bounds = []
        for x, y, angle in self.squares:
            # The extrema of the metric's four rotated corners.
            radius = (abs(mp.cos(angle)) + abs(mp.sin(angle))) / 2
            bounds.append((x - radius, y - radius, x + radius, y + radius))
        return bounds

    def bbox(self):
        b = self._bounds()
        return (min(row[0] for row in b), min(row[1] for row in b),
                max(row[2] for row in b), max(row[3] for row in b))

    def side(self):
        x0, y0, x1, y1 = self.bbox()
        return max(x1 - x0, y1 - y0)

    def normalize(self):
        x0, y0, _, _ = self.bbox()
        return Packing((x - x0, y - y0, a) for x, y, a in self.squares)

    def inflate(self, eps='1e-20'):
        eps = mp.mpf(eps)
        if not mp.isfinite(eps) or eps < 0:
            raise ValueError('eps must be finite and nonnegative')
        x0, y0, x1, y1 = self.bbox()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return Packing((cx + (x - cx) * (1 + eps), cy + (y - cy) * (1 + eps), a)
                       for x, y, a in self.squares)

    def remove_to(self, n):
        if not isinstance(n, int) or not 1 <= n <= len(self):
            raise ValueError('target must be an integer in 1..len(packing)')
        bounds = self._bounds()
        orders = [sorted(range(len(self)), key=lambda i: bounds[i][axis], reverse=axis >= 2)
                  for axis in range(4)]
        active = set(range(len(self)))
        pointers = [0] * 4
        while len(active) > n:
            extremes = []
            for axis, order in enumerate(orders):
                p = pointers[axis]
                while order[p] not in active:
                    p += 1
                pointers[axis] = p
                q = p + 1
                while order[q] not in active:
                    q += 1
                extremes.append((order[p], order[q]))

            def remaining_side(candidate):
                values = [bounds[second if first == candidate else first][axis]
                          for axis, (first, second) in enumerate(extremes)]
                return max(values[2] - values[0], values[3] - values[1])

            candidates = sorted({first for first, _ in extremes})
            active.remove(min(candidates, key=remaining_side))
        return Packing(self.squares[i] for i in sorted(active))

    def extend(self):
        base = self.normalize()
        s = base.side()
        right = [(s + mp.mpf('.5'), mp.mpf(i) + mp.mpf('.5'), 0)
                 for i in range(int(mp.floor(s + 1)))]
        top = [(mp.mpf(i) + mp.mpf('.5'), s + mp.mpf('.5'), 0)
               for i in range(int(mp.floor(s)))]
        return Packing((*base.squares, *right, *top))
