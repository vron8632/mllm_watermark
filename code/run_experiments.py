#!/usr/bin/env python3
"""ICASSP 2027 — Complete experiment suite.
Usage: cd code && python run_experiments.py
Output: ../results/icassp_results.json
"""
import sys, os, time, json, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm
from dct_watermark import embed_watermark, extract_watermark, bit_accuracy

BASE = '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/watermark-yolo26-icassp'
D = f'{BASE}/data'
RESULTS_DIR = f'{BASE}/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
rng = np.random.RandomState(SEED)

def psnr(a, b):
    mse = np.mean((a.astype(np.float32)-b.astype(np.float32))**2)
    return 20*np.log10(255/(np.sqrt(mse)+1e-8))

def collect(dir, max_n=None):
    if not os.path.isdir(dir): return []
    exts = ('.jpg','.jpeg','.png','.bmp','.tif','.tiff')
    files = sorted([os.path.join(dir,f) for f in os.listdir(dir) if f.lower().endswith(exts)])
    if max_n: files = files[:max_n]
    return files

# ── YOLO26-seg & strength map ──────────────────────────────────
from ultralytics import YOLO
YOLO_MODEL = YOLO('yolo26n-seg.pt')

def get_strength_map(img, base_val=0.3, inst_val=0.8, sigma=5):
    h, w = img.shape[:2]
    results = YOLO_MODEL(img)[0]
    S = np.full((h, w), base_val, dtype=np.float32)
    if results and results.masks:
        masks = results.masks.data.cpu().numpy()
        for mask in masks:
            m = np.array(Image.fromarray((mask*255).astype(np.uint8)).resize((w, h), Image.NEAREST))
            S[m > 128] = inst_val
    S = np.array(Image.fromarray((S*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=sigma))) / 255.0
    return np.clip(S, 0.1, 1.0)


# ── Helpers ─────────────────────────────────────────────────────
def _run_batch(img_paths, attack_fn=None, delta=15, use_strength=False, strength_params=None):
    """Run watermark over a batch of images. Returns (ba_array, psnr_array)."""
    accs, pns = [], []
    for p in tqdm(img_paths, desc='Processing'):
        img = np.array(Image.open(p).convert('RGB'))
        h, w = img.shape[:2]; h -= h % 8; w -= w % 8
        if h < 64 or w < 64: continue
        img = img[:h, :w]
        msg = rng.randint(0, 2, 64).astype(np.uint8)

        sm = None
        if use_strength:
            sp = strength_params or {}
            sm = get_strength_map(img, **sp)

        wm = embed_watermark(img, msg, strength_map=sm, delta=delta)
        if attack_fn:
            wm = attack_fn(wm)
        # Blind extraction (no strength map needed for demodulation)
        me = extract_watermark(wm, msg_len=64, delta=delta, strength_map=sm if use_strength else None)
        accs.append(bit_accuracy(msg, me))
        pns.append(psnr(img, wm))
    return np.array(accs), np.array(pns)


# ── Attack factories ────────────────────────────────────────────
def jpeg_attack(q):
    def fn(img):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format='JPEG', quality=q); buf.seek(0)
        return np.array(Image.open(buf).convert('RGB'))
    return fn

def gaussian_noise(sigma):
    def fn(img):
        return np.clip(img + rng.randn(*img.shape)*sigma*255, 0, 255).astype(np.uint8)
    return fn

def center_crop(pct):
    def fn(img):
        h, w = img.shape[:2]
        ch, cw = int(h*pct/100), int(w*pct/100)
        y0, x0 = (h-ch)//2, (w-cw)//2
        out = img.copy(); out[y0:y0+ch, x0:x0+cw] = 128
        return out
    return fn

def median_filter(k=3):
    def fn(img):
        return np.array(Image.fromarray(img).filter(ImageFilter.MedianFilter(k)))
    return fn

def resize_attack(scale):
    def fn(img):
        h, w = img.shape[:2]
        small = np.array(Image.fromarray(img).resize((int(w*scale), int(h*scale))))
        return np.array(Image.fromarray(small).resize((w, h)))
    return fn

def brightness(delta_b=0.2):
    def fn(img):
        return np.clip(img.astype(np.float32) + delta_b*255, 0, 255).astype(np.uint8)
    return fn

def gaussian_blur(radius=1.0):
    def fn(img):
        return np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=radius)))
    return fn


# ═══════════════════════════════════════════════════════════════
def main():
    print('='*60)
    print('  ICASSP 2027 — Full Experiment Suite (Fixed QIM)')
    print('='*60)
    t0 = time.time()
    R = {}

    imgs200 = collect(f'{D}/COCO/val2017', 200)
    imgs1000 = collect(f'{D}/COCO/val2017', 1000)

    # ──────────── E1: COCO 1000 (clean) ────────────
    print('\n--- E1: COCO 1000 ---')
    accs, pnrs = _run_batch(imgs1000, delta=15)
    R['coco_1000'] = {
        'ba_mean': float(np.mean(accs)), 'ba_std': float(np.std(accs)),
        'psnr_mean': float(np.mean(pnrs)), 'psnr_std': float(np.std(pnrs))}
    print(f'  BA={R["coco_1000"]["ba_mean"]*100:.2f}%  PSNR={R["coco_1000"]["psnr_mean"]:.1f}dB')

    # ──────────── E2: Cross-dataset ────────────
    print('\n--- E2: Cross-Dataset ---')
    for name, path, n in [
        ('COCO', f'{D}/COCO/val2017', 200),
        ('CASIA_auth', f'{D}/CASIA/authentic', 200),
        ('CASIA_tamp', f'{D}/CASIA/tampered', 200),
        ('AIGC_auth', f'{D}/AIGC/authentic', 200),
        ('AIGC_forg', f'{D}/AIGC/forgery', 200),
        ('Columbia', f'{D}/Columbia', 180),
    ]:
        paths = collect(path, n)
        if not paths: continue
        accs, _ = _run_batch(paths, delta=15)
        R[f'cross_{name}'] = {'ba_mean': float(np.mean(accs)), 'ba_std': float(np.std(accs))}
        print(f'  {name}: {R[f"cross_{name}"]["ba_mean"]*100:.1f}%')

    # ──────────── E3-E7: Robustness ────────────
    print('\n--- E3: JPEG Compression ---')
    for q in [95, 75, 50, 25]:
        accs, _ = _run_batch(imgs200, attack_fn=jpeg_attack(q), delta=15)
        R[f'jpeg_{q}'] = {'ba_mean': float(np.mean(accs)), 'ba_std': float(np.std(accs))}
        print(f'  JPEG q={q}: {R[f"jpeg_{q}"]["ba_mean"]*100:.1f}%')

    print('\n--- E4: Gaussian Noise ---')
    for s in [0.005, 0.01, 0.02, 0.05]:
        accs, _ = _run_batch(imgs200, attack_fn=gaussian_noise(s), delta=15)
        R[f'noise_{s}'] = {'ba_mean': float(np.mean(accs)), 'ba_std': float(np.std(accs))}
        print(f'  Noise σ={s}: {R[f"noise_{s}"]["ba_mean"]*100:.1f}%')

    print('\n--- E5: Center Cropping ---')
    for pct in [10, 25, 50]:
        accs, _ = _run_batch(imgs200, attack_fn=center_crop(pct), delta=15)
        R[f'crop_{pct}'] = {'ba_mean': float(np.mean(accs)), 'ba_std': float(np.std(accs))}
        print(f'  Crop {pct}%: {R[f"crop_{pct}"]["ba_mean"]*100:.1f}%')

    print('\n--- E6: Additional Attacks ---')
    for name, fn in [
        ('median_3', median_filter(3)),
        ('median_5', median_filter(5)),
        ('scale_75', resize_attack(0.75)),
        ('scale_50', resize_attack(0.5)),
        ('brightness+0.2', brightness(0.2)),
        ('brightness-0.2', brightness(-0.2)),
        ('gblur_r1', gaussian_blur(1.0)),
        ('gblur_r2', gaussian_blur(2.0)),
    ]:
        accs, _ = _run_batch(imgs200, attack_fn=fn, delta=15)
        R[f'attack_{name}'] = {'ba_mean': float(np.mean(accs)), 'ba_std': float(np.std(accs))}
        print(f'  {name}: {R[f"attack_{name}"]["ba_mean"]*100:.1f}%')

    # ──────────── E8: Ablation — Uniform vs Strength ────────────
    print('\n--- E8: Ablation (Uniform vs Strength Map) ---')
    accs_u, _ = _run_batch(imgs200, delta=15, use_strength=False)
    accs_s, _ = _run_batch(imgs200, delta=15, use_strength=True)
    R['ablation_clean'] = {
        'uniform_ba': float(np.mean(accs_u)), 'uniform_std': float(np.std(accs_u)),
        'strength_ba': float(np.mean(accs_s)), 'strength_std': float(np.std(accs_s))}
    print(f'  Uniform:  {R["ablation_clean"]["uniform_ba"]*100:.1f}%')
    print(f'  Strength: {R["ablation_clean"]["strength_ba"]*100:.1f}%')

    # ──────────── E9: Localized Attack (Instance Removal) ────────────
    print('\n--- E9: Localized Attack (Instance Removal) ---')
    accs_u2, accs_s2 = [], []
    for p in tqdm(imgs200, desc='Local Attack'):
        img = np.array(Image.open(p).convert('RGB'))
        h, w = img.shape[:2]; h -= h % 8; w -= w % 8; img = img[:h, :w]
        # Detect instances
        res = YOLO_MODEL(img)[0]
        if not res or not res.masks or len(res.masks) == 0:
            continue
        masks = res.masks.data.cpu().numpy()
        # Pick the largest instance to remove
        idx = np.argmax(masks.sum(axis=(1, 2)))
        m = np.array(Image.fromarray((masks[idx]*255).astype(np.uint8)).resize((w, h), Image.NEAREST)) > 128
        if m.sum() > 0.9 * h * w:  # skip near-full mask
            continue
        # Fill with mean color of surrounding (background) pixels
        bg_color = img[~m].mean(axis=0).astype(np.uint8)
        img_attacked = img.copy(); img_attacked[m] = bg_color

        msg = rng.randint(0, 2, 64).astype(np.uint8)

        # Uniform embedding on attacked image
        wm_u = embed_watermark(img_attacked, msg, strength_map=None, delta=15)
        me_u = extract_watermark(wm_u, msg_len=64, delta=15, strength_map=None)
        accs_u2.append(bit_accuracy(msg, me_u))

        # Strength-map embedding on attacked image
        sm = get_strength_map(img)  # strength from ORIGINAL (pre-attack) image
        wm_s = embed_watermark(img_attacked, msg, strength_map=sm, delta=15)
        me_s = extract_watermark(wm_s, msg_len=64, delta=15, strength_map=sm)
        accs_s2.append(bit_accuracy(msg, me_s))

    if accs_u2:
        R['local_attack'] = {
            'uniform_mean': float(np.mean(accs_u2)), 'uniform_std': float(np.std(accs_u2)),
            'ours_mean': float(np.mean(accs_s2)), 'ours_std': float(np.std(accs_s2))}
        print(f'  Uniform:  {R["local_attack"]["uniform_mean"]*100:.1f}% ± {R["local_attack"]["uniform_std"]*100:.1f}')
        print(f'  Ours:     {R["local_attack"]["ours_mean"]*100:.1f}% ± {R["local_attack"]["ours_std"]*100:.1f}')

    # ──────────── E10: Strength Map Parameter Sweep ────────────
    print('\n--- E10: Strength Map Parameter Sensitivity ---')
    for sigma in [3, 5, 7, 11]:
        for base_val in [0.2, 0.3]:
            accs, _ = _run_batch(imgs200, delta=15, use_strength=True,
                                 strength_params={'sigma': sigma, 'base_val': base_val})
            R[f'param_s{sigma}_b{int(base_val*10)}'] = {
                'ba_mean': float(np.mean(accs)), 'ba_std': float(np.std(accs))}
            print(f'  σ={sigma} base={base_val}: {R[f"param_s{sigma}_b{int(base_val*10)}"]["ba_mean"]*100:.1f}%')

    # ──────────── E11: Δ Sweep (replaces paper Table IV) ────────────
    print('\n--- E11: Delta Parameter Sweep ---')
    for delta in [10, 15, 20, 25, 30]:
        accs, pnrs = _run_batch(imgs200, delta=delta)
        R[f'delta_{delta}'] = {
            'ba_mean': float(np.mean(accs)), 'ba_std': float(np.std(accs)),
            'psnr_mean': float(np.mean(pnrs)), 'psnr_std': float(np.std(pnrs))}
        print(f'  Δ={delta}: BA={R[f"delta_{delta}"]["ba_mean"]*100:.1f}% PSNR={R[f"delta_{delta}"]["psnr_mean"]:.1f}dB')

    # ──────────── E12: Message Length ────────────
    print('\n--- E12: Message Length ---')
    # (Using special inline run)
    from dct_watermark import extract_watermark as ext_wn
    for msg_len in [32, 64, 128, 256]:
        ba_list = []
        for p in tqdm(imgs200[:100], desc=f'Msg len={msg_len}'):
            img = np.array(Image.open(p).convert('RGB'))
            h, w = img.shape[:2]; h -= h % 8; w -= w % 8
            img = img[:h, :w]
            msg = rng.randint(0, 2, msg_len).astype(np.uint8)
            wm = embed_watermark(img, msg, delta=15)
            me = ext_wn(wm, msg_len=msg_len, delta=15)
            ba_list.append(bit_accuracy(msg, me))
        R[f'msglen_{msg_len}'] = {'ba_mean': float(np.mean(ba_list)), 'ba_std': float(np.std(ba_list))}
        print(f'  {msg_len}-bit: {R[f"msglen_{msg_len}"]["ba_mean"]*100:.1f}%')

    # ──────────── E13: SSIM ────────────
    print('\n--- E13: SSIM ---')
    try:
        from skimage.metrics import structural_similarity as ssim
        ssim_vals = []
        for p in tqdm(imgs200, desc='SSIM'):
            img = np.array(Image.open(p).convert('RGB'))
            h, w = img.shape[:2]; h -= h % 8; w -= w % 8; img = img[:h, :w]
            msg = rng.randint(0, 2, 64).astype(np.uint8)
            wm = embed_watermark(img, msg, delta=15)
            ssim_vals.append(ssim(img, wm, channel_axis=2, data_range=255))
        R['ssim'] = {'mean': float(np.mean(ssim_vals)), 'std': float(np.std(ssim_vals))}
        print(f'  SSIM = {R["ssim"]["mean"]:.4f} ± {R["ssim"]["std"]:.4f}')
    except ImportError:
        print('  scikit-image not available, skipping SSIM')

    # ──────────── Summary ────────────
    R['total_time_s'] = round(time.time() - t0, 1)
    out = os.path.join(RESULTS_DIR, 'icassp_results.json')
    with open(out, 'w') as f:
        json.dump(R, f, indent=2)
    print(f'\n✅ Saved to {out}  ⏱ {R["total_time_s"]:.0f}s')

if __name__ == '__main__':
    main()
