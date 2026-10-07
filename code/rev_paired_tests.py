#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-image paired tests for the four decoder settings and the box-oracle reference.

Source of the per-image data: results/signal_ra_coco_n500_msg256_d15_*.json
(COCO val2017, n = 500, 256-bit payload, Delta = 15, one coupled comparison per image:
every setting is evaluated on the same attacked image, so all tests are paired).

Reported per comparison: mean difference (points of BA), paired bootstrap 95% CI
(20,000 resamples), win/tie/loss fractions, paired t-test p, Wilcoxon signed-rank p,
and Cohen's d.

Usage: cd code && python rev_paired_tests.py [--src ../results/signal_ra_coco_n500_*.json]
Writes ../results/paired_tests_<ts>.json
"""
import argparse
import datetime
import glob
import json
import os

import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KEY = {'undefended': 'uniform_ba', 'binary': 'signal_ra_ba',
       'soft': 'signal_soft_ba', 'ours': 'signal_soft_dilate_ba',
       'oracle': 'bbox_ra_ba'}

COMPARISONS = [('ours', 'undefended'), ('ours', 'oracle'), ('binary', 'undefended'),
               ('soft', 'binary'), ('ours', 'soft')]

NBOOT = 20000
SEED = 20261003


def cohen_d(a, b):
    d = a - b
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float('nan')


def bootstrap_ci(d, nboot=NBOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), (nboot, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=None)
    args = ap.parse_args()
    src = args.src or sorted(glob.glob(
        os.path.join(BASE, 'results', 'signal_ra_coco_n500_msg256_d15_*.json')))[-1]
    data = json.load(open(src))
    pi = data['per_image']
    cols = {k: np.array([r[v] for r in pi], float) for k, v in KEY.items()}
    n = len(pi)

    out = {'meta': {'source': os.path.basename(src), 'n': n,
                    'delta': data.get('meta', {}).get('delta'),
                    'msglen': data.get('meta', {}).get('msglen'),
                    'bootstrap_resamples': NBOOT, 'ts': datetime.datetime.now().isoformat()},
           'means_pct': {k: float(100 * v.mean()) for k, v in cols.items()},
           'comparisons': {}}

    for a, b in COMPARISONS:
        d = cols[a] - cols[b]
        lo, hi = bootstrap_ci(d)
        try:
            t, p_t = stats.ttest_rel(cols[a], cols[b])
        except Exception:
            t, p_t = float('nan'), float('nan')
        try:
            w, p_w = stats.wilcoxon(cols[a], cols[b])
        except Exception:
            w, p_w = float('nan'), float('nan')
        out['comparisons'][f'{a}_vs_{b}'] = {
            'mean_diff_pt': float(100 * d.mean()),
            'boot_ci95_pt': [100 * lo, 100 * hi],
            'win_pct': float(100 * (d > 0).mean()),
            'tie_pct': float(100 * (d == 0).mean()),
            'loss_pct': float(100 * (d < 0).mean()),
            'paired_t': float(t), 'p_ttest': float(p_t),
            'wilcoxon_stat': float(w), 'p_wilcoxon': float(p_w),
            'cohen_d': cohen_d(cols[a], cols[b])}

    dst = os.path.join(BASE, 'results',
                       f'paired_tests_{datetime.datetime.now():%Y%m%d_%H%M%S}.json')
    json.dump(out, open(dst, 'w'), indent=2)
    print(json.dumps(out, indent=2))
    print('saved', dst)


if __name__ == '__main__':
    main()
