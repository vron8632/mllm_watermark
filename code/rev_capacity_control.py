#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capacity-curve control for Reviewer 5, comment (5).

Two questions are separated here:
  (i)  does the payload-length trend depend on the single message that was used?
       -> 5 independent messages (seeds 42..46) per payload length, same cached attack;
  (ii) does it depend on this particular attack realisation?
       -> the same experiment on a second, independently generated attack set (n=80).

Protocol note (what "same cached attacked images" means): the diffusion edit is executed
once, on the Delta=15 / 256-bit watermarked image, and its bounding-box patch is cached.
For every payload length the ORIGINAL is re-watermarked with that length's message and the
cached patch is pasted over the same box. The pasted pixels are therefore bit-identical
across rows (verified below), while all pixels outside the box carry the new watermark.
"""
import datetime
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rev_analysis_r2r4r5 as M                     # noqa: E402
from run_signal_ra import check_bits, embed_signal, decode_signal  # noqa: E402
from run_idea_probe import bit_accuracy, psnr       # noqa: E402

SETS = {'attackset_ip2p_20260908_075547': 50,
        'attackset_20261001_222447': 80}
MSGS = [32, 64, 128, 256]
SEEDS = [42, 43, 44, 45, 46]
DELTA = 15


def run(attackset, n):
    meta = json.load(open(os.path.join(BASE, 'results', attackset, 'meta.json')))
    img_dir = os.path.join(BASE, 'results', attackset, 'images')
    samples = meta['samples'][:n]
    out = {'n': n, 'mllm': meta.get('mllm_model') or meta.get('mimo_model'),
           'generator': meta.get('generator', 'instruct-pix2pix'), 'rows': []}
    patch_identity = []
    for mlen in MSGS:
        for seed in SEEDS:
            msg = np.random.RandomState(seed).randint(0, 2, mlen).astype(np.uint8)
            uni, ours, clean, ps = [], [], [], []
            for s in samples:
                orig, att = M.load_pair(img_dir, s['id'])
                if orig is None:
                    continue
                orig = orig[:orig.shape[0] - orig.shape[0] % 8,
                             :orig.shape[1] - orig.shape[1] % 8]
                att = att[:orig.shape[0], :orig.shape[1]]
                H, W = orig.shape[:2]
                nh, nw = H // 8, W // 8
                ck = check_bits(H, W, nh, nw)
                wm = embed_signal(orig, msg, DELTA)
                awm = M.paste_edit(wm, att, s['bbox'])
                patch_identity.append(int(np.abs(
                    awm[int(round(s['bbox'][1] * H)):int(round(s['bbox'][3] * H)),
                    int(round(s['bbox'][0] * W)):int(round(s['bbox'][2] * W))]
                    - att[int(round(s['bbox'][1] * H)):int(round(s['bbox'][3] * H)),
                    int(round(s['bbox'][0] * W)):int(round(s['bbox'][2] * W))]).max()))
                m, _ = decode_signal(awm, mlen, DELTA, use_check=False)
                uni.append(bit_accuracy(msg, m))
                m, _ = decode_signal(awm, mlen, DELTA, use_check=True, soft=True,
                                     dilate=1)
                ours.append(bit_accuracy(msg, m))
                mc, _ = decode_signal(wm, mlen, DELTA, use_check=True, soft=True,
                                      dilate=1)
                clean.append(bit_accuracy(msg, mc))
                ps.append(psnr(orig, wm))
            out['rows'].append({'bits': mlen, 'seed': seed,
                                'uniform_pct': 100 * float(np.mean(uni)),
                                'ours_pct': 100 * float(np.mean(ours)),
                                'gain_pt': 100 * float(np.mean(ours) - np.mean(uni)),
                                'clean_pct': 100 * float(np.mean(clean)),
                                'psnr_db': float(np.mean(ps))})
    # aggregate per length over seeds
    agg = {}
    for mlen in MSGS:
        rs = [r for r in out['rows'] if r['bits'] == mlen]
        agg[str(mlen)] = {
            'uniform_pct': float(np.mean([r['uniform_pct'] for r in rs])),
            'uniform_std_over_seeds': float(np.std([r['uniform_pct'] for r in rs])),
            'ours_pct': float(np.mean([r['ours_pct'] for r in rs])),
            'ours_std_over_seeds': float(np.std([r['ours_pct'] for r in rs])),
            'gain_pt': float(np.mean([r['gain_pt'] for r in rs])),
            'gain_min_pt': float(np.min([r['gain_pt'] for r in rs])),
            'gain_max_pt': float(np.max([r['gain_pt'] for r in rs])),
            'clean_pct': float(np.mean([r['clean_pct'] for r in rs])),
            'psnr_db': float(np.mean([r['psnr_db'] for r in rs]))}
    out['aggregate'] = agg
    out['patch_bit_identity_maxdiff'] = int(max(patch_identity))
    return out


res = {'meta': {'delta': DELTA, 'seeds': SEEDS, 'ts': datetime.datetime.now().isoformat(),
                'question': 'R5(5): payload-length trend vs cached-attack protocol'},
       'sets': {}}
for name, n in SETS.items():
    print('running', name, flush=True)
    res['sets'][name] = run(name, n)
    print(json.dumps(res['sets'][name]['aggregate'], indent=2), flush=True)

out = f'{BASE}/results/capacity_control_{datetime.datetime.now():%Y%m%d_%H%M%S}.json'
with open(out, 'w') as f:
    json.dump(res, f, indent=2)
print('saved', out)
