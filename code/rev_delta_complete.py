#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Complete the step-size sweep for the supplementary table (R2-2: no empty cells).

Runs every arm of the released decoder at every step size in {10,12,13,15,17,20,25} on the
200-sample attack set, plus the single-coefficient payload-only arm, so that Table S-Delta has
a value in each cell. Uses the scalar reference implementation (embed_signal/decode_signal),
i.e. exactly the pipeline behind the paper's headline numbers.
"""
import datetime
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rev_analysis_r2r4r5 as R                                  # noqa: E402
from run_idea_probe import psnr, bit_accuracy, y_channel         # noqa: E402
from run_signal_ra import (embed_signal, decode_signal, check_bits,   # noqa: E402
                           PAY_COEF, CHECK_COEF)
from run_mllm_benchmark import extract_uniform_bbox_ra           # noqa: E402

AS = f'{BASE}/results/attackset_ip2p_20260908_075547'
MSGS, DELTA = 256, None
DELTAS = [10, 12, 13, 15, 17, 20, 25]

meta = json.load(open(os.path.join(AS, 'meta.json')))
img_dir = os.path.join(AS, 'images')
msg = np.random.RandomState(42).randint(0, 2, MSGS).astype(np.uint8)

rows = []
for k, s in enumerate(meta['samples']):
    orig, att = R.load_pair(img_dir, s['id'])
    if orig is None:
        continue
    orig = orig[:orig.shape[0] - orig.shape[0] % 8, :orig.shape[1] - orig.shape[1] % 8]
    att = att[:orig.shape[0], :orig.shape[1]]
    H, W = orig.shape[:2]
    nh, nw = H // 8, W // 8
    ck = check_bits(H, W, nh, nw)
    paybits = msg[np.arange(nh * nw) % MSGS]
    row = {'id': s['id'], 'deltas': {}}
    for dl in DELTAS:
        wm = embed_signal(orig, msg, dl)
        awm = R.paste_edit(wm, att, s['bbox'])
        rec = {}
        m, _ = decode_signal(awm, MSGS, dl, use_check=False)
        rec['uniform'] = bit_accuracy(msg, m)
        m, _ = decode_signal(awm, MSGS, dl, use_check=True, soft=False)
        rec['sc_hard'] = bit_accuracy(msg, m)
        m, _ = decode_signal(awm, MSGS, dl, use_check=True, soft=True)
        rec['sc_soft'] = bit_accuracy(msg, m)
        m, nf = decode_signal(awm, MSGS, dl, use_check=True, soft=True, dilate=1)
        rec['sc'] = bit_accuracy(msg, m)
        rec['ckfail'] = nf / (nh * nw)
        m = extract_uniform_bbox_ra(awm, MSGS, dl, s['bbox'])
        rec['oracle'] = bit_accuracy(msg, m)
        rec['psnr_dual'] = psnr(orig, wm)
        w1 = R.embed_vec(orig, msg, dl, {PAY_COEF: paybits})
        a1 = R.paste_edit(w1, att, s['bbox'])
        co, _, _ = R.coefs_of(a1)
        p1 = R.demod_bits(co[..., PAY_COEF[0], PAY_COEF[1]], dl)
        rec['ba_single'] = bit_accuracy(
            msg, R.vote(p1, np.ones((nh, nw)), MSGS, nh, nw))
        rec['psnr_single'] = psnr(orig, w1)
        row['deltas'][str(dl)] = rec
    rows.append(row)
    if (k + 1) % 25 == 0:
        print(f'  {k+1}/200', flush=True)

summary = {}
for dl in DELTAS:
    g = lambda key: 100.0 * float(np.mean([r['deltas'][str(dl)][key] for r in rows]))
    summary[str(dl)] = {'psnr_dual': round(float(np.mean([r['deltas'][str(dl)]['psnr_dual'] for r in rows])), 2),
                        'psnr_single': round(float(np.mean([r['deltas'][str(dl)]['psnr_single'] for r in rows])), 2),
                        'uniform': round(g('uniform'), 2), 'sc_hard': round(g('sc_hard'), 2),
                        'sc_soft': round(g('sc_soft'), 2), 'sc': round(g('sc'), 2),
                        'oracle': round(g('oracle'), 2),
                        'single': round(g('ba_single'), 2),
                        'ckfail': round(100 * float(np.mean([r['deltas'][str(dl)]['ckfail'] for r in rows])), 2),
                        'gain': round(g('sc') - g('uniform'), 2)}
out = {'meta': {'attackset': os.path.basename(AS), 'images': len(rows), 'msglen': MSGS,
                'deltas': DELTAS,
                'ts': datetime.datetime.now().isoformat()},
       'summary': summary}
p = f'{BASE}/results/delta_sweep_complete_{datetime.datetime.now():%Y%m%d_%H%M%S}.json'
with open(p, 'w') as f:
    json.dump(out, f, indent=2)
print(json.dumps(summary, indent=2))
print('saved', p)
