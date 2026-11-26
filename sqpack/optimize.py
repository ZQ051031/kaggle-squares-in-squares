"""Continuous local search (Numba smooth-SAT L-BFGS-B) to shrink bounding side L."""
import argparse
import csv
import math
from multiprocessing import Pool
from pathlib import Path
import time

from numba import njit
import numpy as np
from scipy.optimize import minimize

from .geometry import bounding_side, check_overlaps, corners
from .hp import Packing, mp
from .io import HEADER, read_submission_strings
from .metric_official import evaluate_single_puzzle


@njit
def energy_and_grad_smooth(params, n, L, eps_abs=1e-12):
    """Smooth wall + pairwise SAT overlap energy and exact analytical gradient."""
    xs = params[0:n]
    ys = params[n:2 * n]
    th = params[2 * n:3 * n]
    gx = np.zeros(n)
    gy = np.zeros(n)
    gt = np.zeros(n)
    cs = np.cos(th)
    sn = np.sin(th)
    E = 0.0

    for i in range(n):
        c = cs[i]
        s = sn[i]
        ac = math.sqrt(c * c + eps_abs)
        as_ = math.sqrt(s * s + eps_abs)
        r = 0.5 * (ac + as_)
        dr = 0.5 * (-c * s / ac + s * c / as_)

        v = r - xs[i]
        if v > 0.0:
            E += 0.5 * v * v
            gx[i] -= v
            gt[i] += v * dr
        v = xs[i] + r - L
        if v > 0.0:
            E += 0.5 * v * v
            gx[i] += v
            gt[i] += v * dr
        v = r - ys[i]
        if v > 0.0:
            E += 0.5 * v * v
            gy[i] -= v
            gt[i] += v * dr
        v = ys[i] + r - L
        if v > 0.0:
            E += 0.5 * v * v
            gy[i] += v
            gt[i] += v * dr

    cell = 1.42
    nx = max(1, int(L / cell) + 2)
    head = np.full(nx * nx, -1, dtype=np.int32)
    nxt = np.full(n, -1, dtype=np.int32)
    cix = np.empty(n, dtype=np.int32)
    ciy = np.empty(n, dtype=np.int32)
    for i in range(n):
        ix = int(xs[i] / cell)
        iy = int(ys[i] / cell)
        if ix < 0:
            ix = 0
        elif ix >= nx:
            ix = nx - 1
        if iy < 0:
            iy = 0
        elif iy >= nx:
            iy = nx - 1
        cix[i] = ix
        ciy[i] = iy
        idx = iy * nx + ix
        nxt[i] = head[idx]
        head[idx] = i

    for i in range(n):
        ix0 = cix[i]
        iy0 = ciy[i]
        xi = xs[i]
        yi = ys[i]
        ci = cs[i]
        si = sn[i]
        for dyc in range(-1, 2):
            iy = iy0 + dyc
            if iy < 0 or iy >= nx:
                continue
            for dxc in range(-1, 2):
                ix = ix0 + dxc
                if ix < 0 or ix >= nx:
                    continue
                j = head[iy * nx + ix]
                while j != -1:
                    if j > i:
                        dx = xs[j] - xi
                        dy = ys[j] - yi
                        if dx * dx + dy * dy < 2.0001:
                            cj = cs[j]
                            sj = sn[j]
                            cd = ci * cj + si * sj
                            sd = sj * ci - cj * si
                            acd = math.sqrt(cd * cd + eps_abs)
                            asd = math.sqrt(sd * sd + eps_abs)
                            R = 0.5 * (1.0 + acd + asd)

                            p0 = dx * ci + dy * si
                            p1 = -dx * si + dy * ci
                            p2 = dx * cj + dy * sj
                            p3 = -dx * sj + dy * cj

                            a0 = abs(p0)
                            a1 = abs(p1)
                            a2 = abs(p2)
                            a3 = abs(p3)
                            best_k = 0
                            best_a = a0
                            if a1 > best_a:
                                best_a = a1
                                best_k = 1
                            if a2 > best_a:
                                best_a = a2
                                best_k = 2
                            if a3 > best_a:
                                best_a = a3
                                best_k = 3

                            gap = best_a - R
                            if gap < 0.0:
                                E += 0.5 * gap * gap
                                dR_dd = 0.5 * (-cd * sd / acd + sd * cd / asd)
                                if best_k == 0:
                                    sp = 1.0 if p0 >= 0.0 else -1.0
                                    ax = ci
                                    ay = si
                                    dp_dti = p1
                                    dp_dtj = 0.0
                                elif best_k == 1:
                                    sp = 1.0 if p1 >= 0.0 else -1.0
                                    ax = -si
                                    ay = ci
                                    dp_dti = -p0
                                    dp_dtj = 0.0
                                elif best_k == 2:
                                    sp = 1.0 if p2 >= 0.0 else -1.0
                                    ax = cj
                                    ay = sj
                                    dp_dti = 0.0
                                    dp_dtj = p3
                                else:
                                    sp = 1.0 if p3 >= 0.0 else -1.0
                                    ax = -sj
                                    ay = cj
                                    dp_dti = 0.0
                                    dp_dtj = -p2

                                w = gap * sp
                                gx[j] += w * ax
                                gy[j] += w * ay
                                gx[i] -= w * ax
                                gy[i] -= w * ay
                                gt[i] += gap * (sp * dp_dti + dR_dd)
                                gt[j] += gap * (sp * dp_dtj - dR_dd)
                    j = nxt[j]

    grad = np.empty(3 * n)
    grad[0:n] = gx
    grad[n:2 * n] = gy
    grad[2 * n:3 * n] = gt
    return E, grad


def _relax_to_box(arr, target_L, maxiter_coarse=300, maxiter_fine=400, jitter=0.0, rng=None):
    n = len(arr)
    cur_L = bounding_side(arr)
    pts = corners(arr).reshape(-1, 2)
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    sc = target_L / cur_L if cur_L > 0 else 1.0
    p0 = np.empty(3 * n)
    p0[0:n] = (arr[:, 0] - x0) * sc
    p0[n:2 * n] = (arr[:, 1] - y0) * sc
    p0[2 * n:3 * n] = arr[:, 2]
    if jitter > 0.0 and rng is not None:
        p0[0:n] += rng.normal(0.0, jitter, n)
        p0[n:2 * n] += rng.normal(0.0, jitter, n)
        p0[2 * n:3 * n] += rng.normal(0.0, jitter * 1.5, n)

    res = minimize(energy_and_grad_smooth, p0, args=(n, target_L, 1e-6),
                   method='L-BFGS-B', jac=True,
                   options={'maxiter': maxiter_coarse, 'ftol': 1e-22, 'gtol': 1e-12})
    res = minimize(energy_and_grad_smooth, res.x, args=(n, target_L, 1e-13),
                   method='L-BFGS-B', jac=True,
                   options={'maxiter': maxiter_fine, 'ftol': 1e-26, 'gtol': 1e-14})
    out = np.column_stack([res.x[0:n], res.x[n:2 * n], res.x[2 * n:3 * n]])
    gap, _ = check_overlaps(out)
    side = bounding_side(out)
    return out, float(res.fun), float(gap), float(side)


def shrink_packing(arr, max_rounds=12, step_init=0.015, step_min=5e-4,
                   maxiter_coarse=150, maxiter_fine=200, seed=0):
    """Iteratively shrink bounding side L via smooth-SAT L-BFGS-B."""
    cur = np.asarray(arr, dtype=float).copy()
    cur_L = bounding_side(cur)
    step = step_init
    rng = np.random.default_rng(seed)

    for _ in range(max_rounds):
        L_try = cur_L - step
        if L_try <= math.sqrt(len(cur)):
            step *= 0.5
            if step < step_min:
                break
            continue
        out, E1, gap, side = _relax_to_box(cur, L_try, maxiter_coarse, maxiter_fine)
        if gap >= -1e-10 and side < cur_L - 1e-6:
            cur = out
            cur_L = side
            step = min(step * 1.35, 0.04)
            continue
        # Only try a jittered basin-hop when close to feasible
        if E1 < 1e-3 and step >= 0.002:
            out2, _, gap2, side2 = _relax_to_box(cur, L_try, maxiter_coarse, maxiter_fine,
                                                 jitter=min(0.01, step), rng=rng)
            if gap2 >= -1e-10 and side2 < cur_L - 1e-6:
                cur = out2
                cur_L = side2
                step = min(step * 1.2, 0.04)
                continue
        step *= 0.5
        if step < step_min:
            break
    return cur, cur_L


def certify_float_packing(arr, n):
    """Convert float packing to 60-dps Packing, inflate safely, and certify officially."""
    gap, _ = check_overlaps(arr)
    side_f = bounding_side(arr)
    if gap < -1e-8:
        return None
    # Inflate enough to guarantee strictly positive separation in 60-dps arithmetic
    eps = max(1e-11, -4.0 * min(gap, 0.0) / max(side_f, 1.0) + 1e-11)
    pk = Packing((repr(float(x)), repr(float(y)), repr(float(a))) for x, y, a in arr)
    pk = pk.normalize().inflate(repr(eps))
    triples = pk.to_rows(45)
    rows = [dict(n=str(n), square_id=str(i), x=x, y=y, angle=a)
            for i, (x, y, a) in enumerate(triples)]
    _, score, side, reason = evaluate_single_puzzle(n, rows)
    if reason != 'ok' or side is None:
        # Retry with slightly larger inflation if near contact
        pk = Packing((repr(float(x)), repr(float(y)), repr(float(a))) for x, y, a in arr)
        pk = pk.normalize().inflate(repr(eps * 10.0))
        triples = pk.to_rows(45)
        rows = [dict(n=str(n), square_id=str(i), x=x, y=y, angle=a)
                for i, (x, y, a) in enumerate(triples)]
        _, score, side, reason = evaluate_single_puzzle(n, rows)
        if reason != 'ok' or side is None:
            return None
    return triples, float(side), float(score)


def _worker_optimize(task):
    n, arr, base_L, max_rounds, step_init = task
    out, new_L = shrink_packing(arr, max_rounds=max_rounds, step_init=step_init, seed=n)
    if new_L >= base_L - 1e-5:
        return n, None, base_L, base_L
    cert = certify_float_packing(out, n)
    if cert is None:
        return n, None, base_L, base_L
    triples, off_L, _ = cert
    if off_L < base_L - 1e-5:
        return n, triples, base_L, off_L
    return n, None, base_L, base_L


def write_store(store_dict, path):
    """Write certified decimal string packings to submission CSV format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        row_id = 0
        for n in sorted(store_dict):
            for sid, row in enumerate(store_dict[n]):
                writer.writerow([row_id, n, sid, *row])
                row_id += 1


def optimize_from_submission(submission_path, summary_path, store_path='solutions/best.csv',
                             jobs=8, max_rounds=16, step_init=0.015, sources_filter=None):
    """Run parallel local search on selected puzzles from a submission and save to store."""
    t0 = time.perf_counter()
    puzzles = read_submission_strings(submission_path)
    with open(summary_path) as f:
        summary = {int(r['n']): r for r in csv.DictReader(f)}

    # Load existing store if present
    store = {}
    if Path(store_path).exists():
        for n, strings in read_submission_strings(store_path).items():
            store[n] = strings.tolist()

    tasks = []
    for n, r in sorted(summary.items()):
        src = r['source']
        base_L = float(r['L'])
        k = math.isqrt(n)
        if k * k == n:
            continue
        if sources_filter is not None and not any(src.startswith(prefix) for prefix in sources_filter):
            continue
        arr = puzzles[n].astype(float)
        tasks.append((n, arr, base_L, max_rounds, step_init))

    print(f'Optimizing {len(tasks)} puzzles across {jobs} workers...', flush=True)
    # Warm up numba in main process before fork
    _dummy = np.zeros(9)
    energy_and_grad_smooth(_dummy, 3, 2.0, 1e-12)

    improved_count = 0
    total_score_gain = 0.0
    with Pool(jobs) as pool:
        for n, triples, base_L, off_L in pool.imap_unordered(_worker_optimize, tasks, chunksize=1):
            if triples is not None:
                gain = (base_L - off_L) / math.sqrt(n)
                total_score_gain += gain
                improved_count += 1
                store[n] = triples
                print(f'  n={n:4d}: {base_L:.6f} -> {off_L:.6f} (dL={off_L-base_L:+.6f}, dScore={-gain:+.6f})',
                      flush=True)
                if improved_count % 10 == 0:
                    write_store(store, store_path)

    if store:
        write_store(store, store_path)
    print(f'Done in {time.perf_counter()-t0:.1f}s: improved {improved_count}/{len(tasks)} puzzles, '
          f'direct score gain = {-total_score_gain:+.6f}; saved {len(store)} entries to {store_path}')
    return improved_count, total_score_gain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', default='submissions/submission_v1_analytical.csv')
    parser.add_argument('--summary', default='submissions/summary_v1_analytical.csv')
    parser.add_argument('--store', default='solutions/best.csv')
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--max-rounds', type=int, default=16)
    parser.add_argument('--step-init', type=float, default=0.015)
    parser.add_argument('--sources', nargs='+', default=['corners45', 'lstrip', 'band'])
    args = parser.parse_args()
    optimize_from_submission(args.submission, args.summary, args.store,
                             args.jobs, args.max_rounds, args.step_init, args.sources)


if __name__ == '__main__':
    main()
