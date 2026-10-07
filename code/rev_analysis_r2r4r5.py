#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Revision experiments answering Reviewers 2/4/5 of JISAS-D-26-06757.

Protocol is the released one: the cached MLLM+bbox edit is pasted onto a freshly
watermarked image (pixels outside the declared box are bit-identical), one 256-bit
message from RandomState(42), Delta = 15, attack set attackset_ip2p_20260908_075547.

Arms
  A per-block joint statistics of check-bit failure and payload decode error  [R5(1)]
  B matched-fidelity curve (single-coefficient payload only vs dual-coefficient)
    and matched-budget control (payload duplicated in both coefficients)      [R5(2)]
  C keyed check sequence (SHA-256 KDF, 128-bit key) vs seeded sequence, key
    consistency separation, embed/decode throughput                          [R4(1)]
  D Delta sensitivity stratified by image texture                            [R4(4)]
  E chroma: embedding leaves chroma untouched; chroma drift vs check failure [R4(2)]
  F edited-area fraction vs check-failure fraction                           [R5(4)]

Usage: /home/oyp/miniconda3/envs/apjf/bin/python rev_analysis_r2r4r5.py [--n 200]
Writes ../results/rev_analysis_r2r4r5_<ts>.json
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_idea_probe import psnr, bit_accuracy, y_channel            # noqa: E402
from run_signal_ra import (embed_signal, decode_signal, check_bits,  # noqa: E402
                           CHECK_COEF, PAY_COEF)

ATTACKSET = f"{BASE}/results/attackset_ip2p_20260908_075547"
MSGLEN, DELTA0 = 256, 15
DELTAS = [10, 12, 13, 15, 17, 20, 25]
KEY = hashlib.sha256(b"mllm-watermark-revision-key").digest()[:16]   # 128-bit key

# ----------------------------------------------------------------- vector DCT QIM
_D = None


def dct_mat():
    global _D
    if _D is None:
        n, D = 8, np.zeros((8, 8))
        for k in range(n):
            for i in range(n):
                D[k, i] = np.cos(np.pi * k * (2 * i + 1) / (2 * n)) * (
                    np.sqrt(1 / n) if k == 0 else np.sqrt(2 / n))
        _D = D
    return _D


def to_blocks(y):
    H, W = y.shape
    nh, nw = H // 8, W // 8
    return y[:nh * 8, :nw * 8].reshape(nh, 8, nw, 8).transpose(0, 2, 1, 3).copy()


def from_blocks(bl):
    nh, nw = bl.shape[:2]
    return bl.transpose(0, 2, 1, 3).reshape(nh * 8, nw * 8)


def dct2(bl):
    D = dct_mat().astype(bl.dtype)
    return np.einsum('ai,nwij,bj->nwab', D, bl, D)


def idct2(co):
    D = dct_mat().astype(co.dtype)      # x = D^T @ co @ D  (inverse of the cv2.dct layout)
    return np.einsum('ia,nwib,bj->nwaj', D, co, D)


def qim(c, bit, delta):
    return np.where(np.asarray(bit) == 1,
                    np.floor(c / delta) * delta + delta / 2,
                    np.round(c / delta) * delta)


def _grid_dists(coefs, delta):
    f = np.floor(coefs / delta)
    q0 = np.stack([f * delta, (f + 1) * delta], axis=-1)
    q1 = np.stack([(f + .5) * delta, (f - .5) * delta, (f + 1.5) * delta], axis=-1)
    d0 = np.abs(coefs[..., None] - q0).min(axis=-1)
    d1 = np.abs(coefs[..., None] - q1).min(axis=-1)
    return d0, d1


def demod_bits(coefs, delta):
    d0, d1 = _grid_dists(coefs, delta)
    return (d1 < d0).astype(np.uint8)


def soft_conf(coefs, delta, expect):
    d0, d1 = _grid_dists(coefs, delta)
    e = np.asarray(expect)
    best = np.where(e == 0, d0, d1)
    other = np.where(e == 0, d1, d0)
    return np.clip(1 - best / (other + best + 1e-8), 0, 1)


def coefs_of(img):
    H, W = img.shape[:2]
    nh, nw = H // 8, W // 8
    y = to_blocks(y_channel(img)[:nh * 8, :nw * 8].astype(np.float32))
    return dct2(y), nh, nw


def ycrcb_planes(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)


def rebuild(ycrcb, co, nh, nw):
    out = ycrcb.copy()
    out[:nh * 8, :nw * 8, 0] = np.clip(from_blocks(idct2(co)), 0, 255)
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def embed_vec(img, msg, delta, seqs):
    """seqs: {coef_index: bits-per-block array}; coefficients absent from seqs untouched."""
    planes = ycrcb_planes(img)
    co, nh, nw = coefs_of(img)
    idx = np.arange(nh * nw) % len(msg)
    for cf, bits in seqs.items():
        b = np.asarray(bits).reshape(nh, nw)
        co[..., cf[0], cf[1]] = qim(co[..., cf[0], cf[1]], b, delta)
    return rebuild(planes, co, nh, nw)


def vote(pay, w, msg_len, nh, nw):
    idx = np.arange(nh * nw) % msg_len
    v1 = np.zeros(msg_len)
    vt = np.zeros(msg_len)
    np.add.at(v1, idx, np.asarray(w * pay).reshape(-1))
    np.add.at(vt, idx, np.asarray(w).reshape(-1))
    return (v1 > vt / 2).astype(np.uint8)


def keyed_check(nh, nw, key=KEY):
    """c(i,j) = LSB of SHA-256(key || block index): a keyed pseudo-random sequence."""
    blk = np.arange(nh * nw, dtype=np.int64)
    bits = np.array([hashlib.sha256(key + int(b).to_bytes(4, 'little')).digest()[0] & 1
                     for b in blk], dtype=np.uint8)
    return bits.reshape(nh, nw)


def analyse_blocks(img, msg, delta, ck):
    """Per-block payload bit / correctness, check pass, soft weight."""
    co, nh, nw = coefs_of(img)
    pay = demod_bits(co[..., PAY_COEF[0], PAY_COEF[1]], delta)
    ckdm = demod_bits(co[..., CHECK_COEF[0], CHECK_COEF[1]], delta)
    idx = (np.arange(nh * nw) % len(msg)).reshape(nh, nw)
    w = soft_conf(co[..., CHECK_COEF[0], CHECK_COEF[1]], delta, ck)
    fail = ckdm != ck
    fd = cv2.dilate(fail.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    w = np.where(fd & ~fail, w * 0.3, w)
    return dict(pay=pay, pay_ok=pay == msg[idx], ck_pass=~fail, w=w, nh=nh, nw=nw)


# ------------------------------------------------------------------ protocol glue
def load_pair(img_dir, cid):
    for a, b in ((f"{cid}_orig.png", f"{cid}_att.png"),
                 (f"orig_{cid}.png", f"att_{cid}.png")):
        pa, pb = os.path.join(img_dir, a), os.path.join(img_dir, b)
        if os.path.exists(pa) and os.path.exists(pb):
            return (np.array(Image.open(pa).convert("RGB")),
                    np.array(Image.open(pb).convert("RGB")))
    return None, None


def paste_edit(wm, att, bbox01):
    H, W = wm.shape[:2]
    x1, y1, x2, y2 = bbox01
    bx1, by1 = int(round(x1 * W)), int(round(y1 * H))
    bx2, by2 = int(round(x2 * W)), int(round(y2 * H))
    out = wm.copy()
    out[by1:by2, bx1:bx2] = att[by1:by2, bx1:bx2]
    return out


def texture_index(rgb):
    y = y_channel(rgb)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def auroc(score, label):
    s = np.asarray(score, dtype=np.float64)
    lab = np.asarray(label).astype(bool)
    n1, n0 = int(lab.sum()), int((~lab).sum())
    if not n1 or not n0:
        return None
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--attackset', default=ATTACKSET)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.attackset, 'meta.json')))
    img_dir = os.path.join(args.attackset, 'images')
    samples = meta['samples'][: args.n]
    msg = np.random.RandomState(42).randint(0, 2, MSGLEN).astype(np.uint8)

    rows = []
    agg = dict(ckfail_blocks=0, payerr_blocks=0, both=0, ckfail_payok=0,
               ckpass_payerr=0, blocks=0)
    auroc_w, auroc_1mw = [], []
    chroma_fail, chroma_pass = [], []
    t0 = time.time()

    for k, s in enumerate(samples):
        orig, att = load_pair(img_dir, s['id'])
        if orig is None:
            continue
        orig = orig[:orig.shape[0] - orig.shape[0] % 8, :orig.shape[1] - orig.shape[1] % 8]
        att = att[:orig.shape[0], :orig.shape[1]]
        H, W = orig.shape[:2]
        nh, nw = H // 8, W // 8
        bbox = s['bbox']
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        ck_seed = check_bits(H, W, nh, nw)
        row = {'id': s['id'], 'area_frac': area, 'tex': texture_index(orig)}

        # ---------- validation of the vectorised implementation (first 10 images)
        wm_ref = embed_signal(orig, msg, DELTA0)
        att_ref = paste_edit(wm_ref, att, bbox)
        m_ref, nf_ref = decode_signal(att_ref, MSGLEN, DELTA0, use_check=True,
                                      soft=True, dilate=1)
        ba_ref = bit_accuracy(msg, m_ref)

        wm_vec = embed_vec(orig, msg, DELTA0,
                           {PAY_COEF: msg[np.arange(nh * nw) % MSGLEN],
                            CHECK_COEF: ck_seed.reshape(-1)})
        ident = bool(np.array_equal(wm_vec, wm_ref))
        att_vec = paste_edit(wm_vec, att, bbox)
        r = analyse_blocks(att_vec, msg, DELTA0, ck_seed)
        ba_vec = bit_accuracy(msg, vote(r['pay'], r['w'], MSGLEN, nh, nw))
        row['ba_ours'] = ba_vec
        row['ba_ours_ref'] = float(ba_ref)
        row['embed_identical'] = ident

        # ---------- A: per-block joint statistics
        m_uni = vote(r['pay'], np.ones((nh, nw)), MSGLEN, nh, nw)
        row['ba_uniform'] = bit_accuracy(msg, m_uni)
        n_cf = int((~r['ck_pass']).sum())
        n_pe = int((~r['pay_ok']).sum())
        n_both = int(((~r['ck_pass']) & (~r['pay_ok'])).sum())
        row.update(blocks=int(nh * nw), ckfail=n_cf, payerr=n_pe, both=n_both,
                   ckfail_payok=n_cf - n_both, ckpass_payerr=n_pe - n_both,
                   psnr=psnr(orig, wm_ref))
        for kkey, rkey in (('ckfail_blocks', 'ckfail'), ('payerr_blocks', 'payerr'),
                           ('both', 'both'), ('ckfail_payok', 'ckfail_payok'),
                           ('ckpass_payerr', 'ckpass_payerr'), ('blocks', 'blocks')):
            agg[kkey] += int(row[rkey])
        auroc_w.append((r['w'].reshape(-1), r['pay_ok'].reshape(-1)))
        # chroma drift per block (E)
        c0 = ycrcb_planes(wm_ref).astype(np.int16)
        c1 = ycrcb_planes(att_vec).astype(np.int16)
        row['chroma_maxdiff_embed'] = int(np.abs(
            ycrcb_planes(orig).astype(np.int16)[:, :, 1:] - c0[:, :, 1:]).max())
        drift = to_blocks(np.abs(c1[:, :, 1:] - c0[:, :, 1:]).sum(axis=2)
                          .astype(np.float32)).mean(axis=(2, 3))
        chroma_fail.append(drift[~r['ck_pass']].reshape(-1))
        chroma_pass.append(drift[r['ck_pass']].reshape(-1))

        # ---------- B: matched-budget control (payload duplicated, no check)
        paybits = msg[np.arange(nh * nw) % MSGLEN]
        wm_dup = embed_vec(orig, msg, DELTA0,
                           {PAY_COEF: paybits, CHECK_COEF: paybits})
        rd = analyse_blocks(paste_edit(wm_dup, att, bbox), msg, DELTA0,
                            np.zeros((nh, nw), np.uint8))
        row['ba_dupbudget'] = bit_accuracy(
            msg, vote(rd['pay'], np.ones((nh, nw)), MSGLEN, nh, nw))
        row['psnr_dupbudget'] = psnr(orig, wm_dup)

        # ---------- C: keyed-hash check sequence
        ck_key = keyed_check(nh, nw)
        wm_k = embed_vec(orig, msg, DELTA0,
                         {PAY_COEF: paybits, CHECK_COEF: ck_key.reshape(-1)})
        rk = analyse_blocks(paste_edit(wm_k, att, bbox), msg, DELTA0, ck_key)
        row['ba_keyed'] = bit_accuracy(msg, vote(rk['pay'], rk['w'], MSGLEN, nh, nw))
        row['psnr_keyed'] = psnr(orig, wm_k)
        rk_clean = analyse_blocks(wm_k, msg, DELTA0, ck_key)
        row['ckfail_keyed_clean'] = float((~rk_clean['ck_pass']).mean())
        rk_att = analyse_blocks(paste_edit(wm_k, att, bbox), msg, DELTA0, ck_seed)
        row['ba_keyed_wrongkey'] = bit_accuracy(
            msg, vote(rk_att['pay'], rk_att['w'], MSGLEN, nh, nw))
        # consistency of a foreign key on a clean watermarked image (key-recovery test)
        rs = np.random.RandomState(1000 + k)
        cons = []
        for _ in range(20):
            cand = rs.randint(0, 2, (nh, nw)).astype(np.uint8)
            rc = analyse_blocks(wm_vec, msg, DELTA0, cand)
            cons.append(float(rc['ck_pass'].mean()))
        row['key_consistency_wrong_mean'] = float(np.mean(cons))
        row['key_consistency_wrong_max'] = float(np.max(cons))
        # ---------- B/D: Delta curve, dual- vs single-coefficient
        row['key_consistency_true'] = float(
            analyse_blocks(wm_vec, msg, DELTA0, ck_seed)['ck_pass'].mean())
        row['delta'] = {}
        for dl in DELTAS:
            wd = embed_vec(orig, msg, dl,
                           {PAY_COEF: paybits, CHECK_COEF: ck_seed.reshape(-1)})
            w1 = embed_vec(orig, msg, dl, {PAY_COEF: paybits})
            ad = paste_edit(wd, att, bbox)
            rr = analyse_blocks(ad, msg, dl, ck_seed)
            r1 = analyse_blocks(paste_edit(w1, att, bbox), msg, dl,
                                np.zeros((nh, nw), np.uint8))
            row['delta'][str(dl)] = {
                'psnr_dual': psnr(orig, wd),
                'ba_dual_uniform': bit_accuracy(
                    msg, vote(rr['pay'], np.ones((nh, nw)), MSGLEN, nh, nw)),
                'ba_dual_ours': bit_accuracy(msg, vote(rr['pay'], rr['w'], MSGLEN, nh, nw)),
                'ckfail_dual': float((~rr['ck_pass']).mean()),
                'psnr_single': psnr(orig, w1),
                'ba_single': bit_accuracy(
                    msg, vote(r1['pay'], np.ones((nh, nw)), MSGLEN, nh, nw)),
            }
        rows.append(row)
        if (k + 1) % 25 == 0:
            print(f'  {k+1}/{len(samples)} ({time.time()-t0:.0f}s)', flush=True)

    # ---------------------------------------------------------------- aggregates
    mean = lambda xs: float(np.mean(xs)) if len(xs) else None
    R = {'meta': {'attackset': os.path.basename(args.attackset), 'images': len(rows),
                  'msglen': MSGLEN, 'delta0': DELTA0, 'deltas': DELTAS,
                  'key_bits': 128, 'ts': datetime.datetime.now().isoformat(),
                  'elapsed_s': time.time() - t0,
                  'embed_identical_rate': mean([float(r['embed_identical']) for r in rows]),
                  'ba_matches_reference_mean_absdiff': mean(
                      [abs(r['ba_ours'] - r['ba_ours_ref']) for r in rows])}}

    tot = agg['blocks']
    R['A_perblock'] = {
        'images': len(rows), 'blocks': tot,
        'check_fail_rate_pct': 100 * agg['ckfail_blocks'] / tot,
        'payload_error_rate_pct': 100 * agg['payerr_blocks'] / tot,
        'P_payerr_given_ckfail_pct': 100 * agg['both'] / max(agg['ckfail_blocks'], 1),
        'P_payerr_given_ckpass_pct': 100 * (agg['payerr_blocks'] - agg['both'])
        / max(tot - agg['ckfail_blocks'], 1),
        'false_removal_pct_of_all_blocks': 100 * agg['ckfail_payok'] / tot,
        'missed_detection_pct_of_all_blocks': 100 * agg['ckpass_payerr'] / tot,
        'recall_of_damage_pct': 100 * agg['both'] / max(agg['payerr_blocks'], 1),
        'lift_ratio': (agg['both'] / max(agg['ckfail_blocks'], 1))
        / max((agg['payerr_blocks'] - agg['both']) / max(tot - agg['ckfail_blocks'], 1),
              1e-12),
        'ba_uniform_pct': 100 * mean([r['ba_uniform'] for r in rows]),
        'ba_ours_pct': 100 * mean([r['ba_ours'] for r in rows]),
        'auc_weight_vs_payload_correct': auroc(
            np.concatenate([w for w, _ in auroc_w]),
            np.concatenate([ok for _, ok in auroc_w])),
        'imglevel_pearson_ckfail_vs_payerr': float(np.corrcoef(
            [r['ckfail'] / r['blocks'] for r in rows],
            [r['payerr'] / r['blocks'] for r in rows])[0, 1]),
        'imglevel_spearman': float(np.corrcoef(
            np.argsort(np.argsort([r['ckfail'] / r['blocks'] for r in rows])),
            np.argsort(np.argsort([r['payerr'] / r['blocks'] for r in rows])))[0, 1]),
    }
    R['B_matched'] = {
        'ours': {'psnr': mean([r['psnr'] for r in rows]),
                 'ba': 100 * mean([r['ba_ours'] for r in rows]),
                 'coefficients_per_block': 2},
        'payload_only_single_coef': {'psnr': mean(
            [r['delta']['15']['psnr_single'] for r in rows]),
            'ba': mean([r['delta']['15']['ba_single'] for r in rows]) * 100,
            'coefficients_per_block': 1},
        'dup_payload_budget_control': {'psnr': mean([r['psnr_dupbudget'] for r in rows]),
                                       'ba': mean([r['ba_dupbudget'] for r in rows]) * 100,
                                       'coefficients_per_block': 2},
        'per_delta': {str(dl): {
            'psnr_dual': mean([r['delta'][str(dl)]['psnr_dual'] for r in rows]),
            'ba_dual_uniform': 100 * mean([r['delta'][str(dl)]['ba_dual_uniform']
                                           for r in rows]),
            'ba_dual_ours': 100 * mean([r['delta'][str(dl)]['ba_dual_ours'] for r in rows]),
            'ckfail_dual_pct': 100 * mean([r['delta'][str(dl)]['ckfail_dual'] for r in rows]),
            'psnr_single': mean([r['delta'][str(dl)]['psnr_single'] for r in rows]),
            'ba_single': 100 * mean([r['delta'][str(dl)]['ba_single'] for r in rows])}
            for dl in DELTAS}}
    tex = np.array([r['tex'] for r in rows])
    lo, hi = np.quantile(tex, [1 / 3, 2 / 3])
    R['D_texture'] = {'tercile_bounds': [float(lo), float(hi)], 'strata': {}}
    arr = np.array(rows, dtype=object)
    for name, sel in (('smooth', tex <= lo), ('mid', (tex > lo) & (tex <= hi)),
                      ('textured', tex > hi)):
        sub = arr[sel]
        R['D_texture']['strata'][name] = {
            'n': int(sel.sum()), 'tex_mean': float(tex[sel].mean()),
            'psnr_dual': {str(dl): mean([r['delta'][str(dl)]['psnr_dual'] for r in sub])
                          for dl in DELTAS},
            'ba_ours': {str(dl): 100 * mean([r['delta'][str(dl)]['ba_dual_ours'] for r in sub])
                        for dl in DELTAS},
            'ba_single': {str(dl): 100 * mean([r['delta'][str(dl)]['ba_single'] for r in sub])
                          for dl in DELTAS},
            'ba_uniform_dual': {str(dl): 100 * mean(
                [r['delta'][str(dl)]['ba_dual_uniform'] for r in sub]) for dl in DELTAS},
            'ckfail_pct': {str(dl): 100 * mean([r['delta'][str(dl)]['ckfail_dual']
                                                for r in sub]) for dl in DELTAS}}
    cf = np.concatenate(chroma_fail) if chroma_fail else np.zeros(0)
    cp = np.concatenate(chroma_pass) if chroma_pass else np.zeros(0)
    R['E_chroma'] = {
        'chroma_maxdiff_embed': int(max(r['chroma_maxdiff_embed'] for r in rows)),
        'chroma_drift_ckfail_mean': float(cf.mean()),
        'chroma_drift_ckpass_mean': float(cp.mean()),
        'auc_chroma_predicts_checkfail': auroc(np.concatenate([cf, cp]),
                                               np.r_[np.ones(len(cf)), np.zeros(len(cp))]),
        'auc_checkweight_predicts_payloaderr': R['A_perblock'][
            'auc_weight_vs_payload_correct']}
    R['C_key'] = {
        'key_consistency_true_mean': mean([r['key_consistency_true'] for r in rows]),
        'key_consistency_wrong_mean': mean([r['key_consistency_wrong_mean'] for r in rows]),
        'key_consistency_wrong_max': max(r['key_consistency_wrong_max'] for r in rows),
        'ba_keyed_pct': 100 * mean([r['ba_keyed'] for r in rows]),
        'ba_seeded_pct': 100 * mean([r['ba_ours'] for r in rows]),
        'psnr_keyed': mean([r['psnr_keyed'] for r in rows]),
        'psnr_seeded': mean([r['psnr'] for r in rows]),
        'ckfail_clean_keyed_pct': 100 * mean([r['ckfail_keyed_clean'] for r in rows]),
        'ba_with_wrong_key_pct': 100 * mean([r['ba_keyed_wrongkey'] for r in rows]),
        'seeded_keyspace': 2 ** 32, 'hashed_keyspace': 2 ** 128}
    one = load_pair(img_dir, samples[0]['id'])[0][:256, :256]
    t = time.time()
    for _ in range(5):
        embed_signal(one, msg, DELTA0)
    t_emb = (time.time() - t) / 5
    t = time.time()
    for _ in range(5):
        decode_signal(embed_signal(one, msg, DELTA0), MSGLEN, DELTA0,
                      use_check=True, soft=True, dilate=1)
    t_dec = (time.time() - t) / 5
    R['C_key']['embed_ms_per_image'] = t_emb * 1000
    R['C_key']['decode_ms_per_image'] = t_dec * 1000
    R['C_key']['bruteforce_days_seed32_single_core'] = 2 ** 31 * (t_emb + t_dec) / 86400
    R['C_key']['bruteforce_years_hash128_single_core'] = 2 ** 127 * (t_emb + t_dec) / 3.15e7

    R['F_area_vs_checkfail'] = {'pearson': float(np.corrcoef(
        [r['ckfail'] / r['blocks'] for r in rows], [r['area_frac'] for r in rows])[0, 1]),
        'bands': {}}
    ar = np.array([r['area_frac'] for r in rows])
    for name, a, b in (('<15%', 0, .15), ('15-35%', .15, .35), ('>35%', .35, 1.01)):
        m = (ar >= a) & (ar < b)
        if m.sum():
            sub = arr[m]
            R['F_area_vs_checkfail']['bands'][name] = {
                'n': int(m.sum()),
                'mean_area_pct': 100 * float(ar[m].mean()),
                'mean_ckfail_pct': 100 * float(np.mean([r['ckfail'] / r['blocks']
                                                        for r in sub])),
                'ba_uniform_pct': 100 * float(np.mean([r['ba_uniform'] for r in sub])),
                'ba_ours_pct': 100 * float(np.mean([r['ba_ours'] for r in sub]))}
    out = f'{BASE}/results/rev_analysis_r2r4r5_{datetime.datetime.now():%Y%m%d_%H%M%S}.json'
    with open(out, 'w') as f:
        json.dump({'summary': R, 'per_image': rows}, f, indent=2, default=float)
    print(json.dumps(R, indent=2, default=float))
    print('saved', out, flush=True)


if __name__ == '__main__':
    main()
