#!/usr/bin/env python3
"""Fair head-to-head: content-adaptive vs uniform at matched PSNR.

Both schemes operate at equal distortion budget (PSNR ≈ 48.75 dB):
  - adaptive: Δ=15 with per-block Δ_eff = Δ*(1+0.5*S̄_b)
  - uniform:  Δ=19 (calibrated so clean PSNR matches adaptive)
Reports JPEG / noise / removal robustness. Saved to results/icassp_fair.json
"""
import sys, os, time, json, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm
from dct_watermark import (embed_uniform, extract_uniform,
                           embed_adaptive, extract_adaptive, bit_accuracy)
from run_experiments import get_strength_map, YOLO_MODEL

BASE = '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/watermark-yolo26-icassp'
D = f'{BASE}/data/COCO/val2017'
rng = np.random.RandomState(42)
DELTA_A = 15
DELTA_U = 19  # matched-PSNR uniform


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32))**2)
    return 20*np.log10(255/(np.sqrt(mse)+1e-8))


def jpeg(img, q):
    buf = io.BytesIO(); Image.fromarray(img).save(buf, format='JPEG', quality=q); buf.seek(0)
    return np.array(Image.open(buf).convert('RGB'))


def gblur(img, r):
    return np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=r)))


def brightness(img, d):
    return np.clip(img.astype(np.float32) + d*255, 0, 255).astype(np.uint8)


def remove_instances(img, frac):
    h, w = img.shape[:2]
    res = YOLO_MODEL(img)[0]
    if not res or not res.masks or len(res.masks) == 0:
        return None
    masks = res.masks.data.cpu().numpy()
    m_total = np.zeros((h, w), dtype=bool)
    idxs = np.argsort([m.sum() for m in masks])[::-1]
    removed = 0
    for idx in idxs:
        m = np.array(Image.fromarray((masks[idx]*255).astype(np.uint8)).resize((w, h), Image.NEAREST)) > 128
        m_total = m_total | m
        removed = m_total.sum()
        if removed > frac * h * w:
            break
    if removed < 0.05 * h * w:
        return None
    bg = img[~m_total].mean(axis=0).astype(np.uint8) if (~m_total).sum() > 0 else np.array([128, 128, 128])
    out = img.copy(); out[m_total] = bg
    return out


def load(p):
    img = np.array(Image.open(p).convert('RGB'))
    h, w = img.shape[:2]; h -= h % 8; w -= w % 8
    return img[:h, :w]


def main():
    print('='*65)
    print('  Fair comparison: adaptive vs uniform @ matched PSNR')
    print('='*65)
    t0 = time.time()
    R = {}
    imgs = sorted([os.path.join(D, f) for f in os.listdir(D) if f.endswith('.jpg')])[:120]

    def run(attack_fn, L=64, use_removal=False, rem_frac=0.0):
        ba_a, ba_u, ps_a, ps_u = [], [], [], []
        for p in tqdm(imgs, desc='eval'):
            img = load(p)
            msg = rng.randint(0, 2, L).astype(np.uint8)
            sm = get_strength_map(img)
            wm_a = embed_adaptive(img, msg, sm, delta=DELTA_A)
            wm_u = embed_uniform(img, msg, delta=DELTA_U)
            if use_removal:
                att_a = remove_instances(wm_a, rem_frac)
                att_u = remove_instances(wm_u, rem_frac)
                if att_a is None or att_u is None:
                    continue
            else:
                att_a, att_u = wm_a, wm_u
            if attack_fn is not None:
                att_a, att_u = attack_fn(att_a), attack_fn(att_u)
            me_a = extract_adaptive(att_a, msg_len=L, delta=DELTA_A, strength_map=sm)
            me_u = extract_uniform(att_u, msg_len=L, delta=DELTA_U)
            ba_a.append(bit_accuracy(msg, me_a)); ba_u.append(bit_accuracy(msg, me_u))
            ps_a.append(psnr(img, wm_a)); ps_u.append(psnr(img, wm_u))
        return (np.mean(ba_a), np.std(ba_a), np.mean(ba_u), np.std(ba_u),
                np.mean(ps_a), np.mean(ps_u))

    def rec(key, res):
        ba_a, sd_a, ba_u, sd_u, ps_a, ps_u = res
        R[key] = {'adaptive_ba': float(ba_a), 'adaptive_std': float(sd_a),
                  'uniform_ba': float(ba_u), 'uniform_std': float(sd_u),
                  'adaptive_psnr': float(ps_a), 'uniform_psnr': float(ps_u)}
        print(f'  {key:<28} adaptive={ba_a*100:6.1f}%  uniform={ba_u*100:6.1f}%  '
              f'(PSNR {ps_a:.1f}/{ps_u:.1f})')

    print('\n--- Clean ---');        rec('clean_64', run(None, 64))
    print('\n--- JPEG ---')
    rec('jpeg95_64', run(lambda im: jpeg(im, 95), 64))
    rec('jpeg75_64', run(lambda im: jpeg(im, 75), 64))
    rec('jpeg75_256', run(lambda im: jpeg(im, 75), 256))
    rec('jpeg50_64', run(lambda im: jpeg(im, 50), 64))
    print('\n--- Noise / filter ---')
    rec('gblur1_64', run(lambda im: gblur(im, 1.0), 64))
    rec('bright_pos_64', run(lambda im: brightness(im, 0.2), 64))
    print('\n--- Removal (256-bit) ---')
    rec('rem30_256', run(None, 256, use_removal=True, rem_frac=0.3))
    rec('rem50_256', run(None, 256, use_removal=True, rem_frac=0.5))

    R['delta_adaptive'] = DELTA_A
    R['delta_uniform'] = DELTA_U
    R['n_images'] = len(imgs)
    R['total_time_s'] = round(time.time() - t0, 1)
    out = os.path.join(BASE, 'results', 'icassp_fair.json')
    with open(out, 'w') as f:
        json.dump(R, f, indent=2)
    print(f'\nSaved to {out} ({R["total_time_s"]:.0f}s)')

if __name__ == '__main__':
    main()
