#!/usr/bin/env python3
"""嵌入保真度指标 (fidelity): PSNR + SSIM + LPIPS, 6 个水印方法, 600 张图.

与 results/sota_comparison_20260910_174221.json / fidelity_psnr_summary.json 同口径:
  - 数据集顺序: collect_source('coco',500) + collect_source('div2k',100)
  - 消息: rng = np.random.RandomState(42); 每张图 rng.randint(0,2,256).uint8 (逐字复现 main())
  - 嵌入: 直接复用 run_sota_comparison 的加载器与嵌入函数
  - PSNR: run_idea_probe.psnr
  - SSIM: skimage structural_similarity(data_range=256, channel_axis=2)
  - LPIPS: lpips.LPIPS(net='alex'), float/127.5-1, HWC->CHW, 单模型循环复用

用法:
  cd code && conda activate apjf && unset ALL_PROXY all_proxy
  python run_fidelity_metrics.py --n 3 --out <scratch>/smoke.json     # 冒烟
  python run_fidelity_metrics.py --resume                             # 全量 600
"""
import sys, os, time, json, argparse, datetime, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity

import run_sota_comparison as S
from run_idea_probe import load, collect_source, psnr

# 固定键名 / 简写 (与 sota_comparison per_image 字段后缀一致)
METHOD_KEYS = ['dwt_dct', 'signal_check_v2', 'robust_wide', 'trustmark',
               'stable_signature', 'watermark_anything']
SUF = {'dwt_dct': 'dwt', 'signal_check_v2': 'sc', 'robust_wide': 'rw',
       'trustmark': 'tm', 'stable_signature': 'ss', 'watermark_anything': 'wa'}
OURS = 'signal_check_v2'


# ── 指标 ─────────────────────────────────────────────────────────
def ssim_fn(a, b):
    """uint8 HWC -> SSIM. data_range=256, channel_axis=2 (逐字按协议)."""
    try:
        return float(structural_similarity(a, b, data_range=256, channel_axis=2))
    except TypeError:
        # 老版本 skimage 无 channel_axis: 逐通道 win_size=7 取均值
        return float(np.mean([structural_similarity(a[:, :, c], b[:, :, c],
                                                    data_range=256, win_size=7)
                              for c in range(a.shape[2])]))


def make_lpips_fn(net='alex'):
    """LPIPS 模型只初始化一次, 之后循环复用."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = lpips.LPIPS(net=net).to(device).eval()

    def fn(a, b):
        ta = torch.from_numpy(a.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        tb = torch.from_numpy(b.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        with torch.no_grad():
            v = model(ta.to(device), tb.to(device))
        return float(v.reshape(-1)[0].item())

    return fn


# ── 单张图嵌入 (复用 run_sota_comparison) ───────────────────────
def embed_all(models, img, msg_long, delta, msglens):
    """返回 {method: wm(uint8 HWC) 或 None}, 每个方法独立 try/except."""
    rw_model, tm_model, ss_model, wa_model = models
    rw_len, tm_len, ss_len, wa_len = msglens
    wm, err = {}, {}
    jobs = [
        ('dwt_dct', lambda: S.dwt_dct_embed(img, msg_long, delta)),
        ('signal_check_v2', lambda: S.signal_check_embed(img, msg_long, delta)),
        ('robust_wide', (lambda: S.robust_wide_embed(rw_model, img, msg_long[:rw_len]))
         if rw_len > 0 else (lambda: None)),
        ('trustmark', (lambda: S.trustmark_embed(tm_model, img, msg_long[:tm_len]))
         if tm_len > 0 else (lambda: None)),
        ('stable_signature', (lambda: S.stable_signature_embed(ss_model, img, msg_long[:ss_len]))
         if ss_len > 0 else (lambda: None)),
        ('watermark_anything', (lambda: S.watermark_anything_embed(wa_model, img, msg_long[:wa_len]))
         if wa_len > 0 else (lambda: None)),
    ]
    for k, fn in jobs:
        try:
            out = fn()
            if out is None:
                wm[k] = None
                err[k] = 'embed returned None'
            elif out.shape != img.shape:
                wm[k] = None
                err[k] = f'shape {out.shape} != {img.shape}'
            else:
                wm[k] = out
        except Exception as e:
            wm[k] = None
            err[k] = repr(e)[:160]
    return wm, err


# ── 聚合 ─────────────────────────────────────────────────────────
def _stats(v):
    a = np.asarray(v, dtype=np.float64)
    if a.size == 0:
        return {'n': 0, 'mean': None, 'std': None}
    return {'n': int(a.size), 'mean': float(a.mean()), 'std': float(a.std(ddof=1)) if a.size > 1 else 0.0}


def _ci95(diff):
    a = np.asarray(diff, dtype=np.float64)
    if a.size == 0:
        return None, [None, None]
    m = float(a.mean())
    if a.size == 1:
        return m, [m, m]
    se = float(a.std(ddof=1) / math.sqrt(a.size))
    return m, [m - 1.96 * se, m + 1.96 * se]


def aggregate(rows):
    vals = {k: {'psnr': [], 'ssim': [], 'lpips': []} for k in METHOD_KEYS}
    for r in rows:
        for k in METHOD_KEYS:
            s = SUF[k]
            for m in ('psnr', 'ssim', 'lpips'):
                v = r.get(f'{m}_{s}')
                if v is not None:
                    vals[k][m].append(v)

    per_method = {}
    for k in METHOD_KEYS:
        per_method[k] = {
            'n': vals[k]['psnr'].__len__(),
            'psnr_mean': _stats(vals[k]['psnr'])['mean'],
            'psnr_std': _stats(vals[k]['psnr'])['std'],
            'ssim_mean': _stats(vals[k]['ssim'])['mean'],
            'ssim_std': _stats(vals[k]['ssim'])['std'],
            'lpips_mean': _stats(vals[k]['lpips'])['mean'],
            'lpips_std': _stats(vals[k]['lpips'])['std'],
        }

    paired = {}
    for k in METHOD_KEYS:
        if k == OURS:
            continue
        d_ssim, d_lpips, d_psnr = [], [], []
        for r in rows:
            so, sk = r.get(f'ssim_{SUF[OURS]}'), r.get(f'ssim_{SUF[k]}')
            lo, lk = r.get(f'lpips_{SUF[OURS]}'), r.get(f'lpips_{SUF[k]}')
            po, pk = r.get(f'psnr_{SUF[OURS]}'), r.get(f'psnr_{SUF[k]}')
            if so is not None and sk is not None:
                d_ssim.append(so - sk)
            if lo is not None and lk is not None:
                d_lpips.append(lo - lk)
            if po is not None and pk is not None:
                d_psnr.append(po - pk)
        ms, ci_s = _ci95(d_ssim)
        ml, ci_l = _ci95(d_lpips)
        mp, ci_p = _ci95(d_psnr)
        paired[k] = {
            'ssim_diff': ms, 'ssim_ci95': ci_s,
            'lpips_diff': ml, 'lpips_ci95': ci_l,
            'psnr_diff': mp, 'psnr_ci95': ci_p,
            'n_paired': len(d_ssim),
        }
    return per_method, paired


def atomic_dump(obj, path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = os.path.join(os.path.dirname(path), '.tmp_' + os.path.basename(path))
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, default=float)
    os.replace(tmp, path)


def fmt(v, nd=4):
    return '  n/a ' if v is None else f'{v:.{nd}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=600, help='>=600 时取 coco500+div2k100, 否则按数据集各取 n')
    ap.add_argument('--delta', type=int, default=15)
    ap.add_argument('--msglen', type=int, default=256)
    ap.add_argument('--lpips-net', default='alex')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'results', 'fidelity_ssim_lpips_600.json'))
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--save-every', type=int, default=50)
    ap.add_argument('--progress-every', type=int, default=25)
    args = ap.parse_args()

    t_start = time.time()

    # ── 数据 (顺序必须与 sota_comparison 一致) ──
    if args.n >= 600:
        paths = collect_source('coco', 500) + collect_source('div2k', 100)
        datasets = 'coco500+div2k100'
    else:
        paths = collect_source('coco', args.n) + collect_source('div2k', args.n)
        datasets = f'coco{args.n}+div2k{args.n}'
    print(f'paths: {len(paths)} (coco={sum(1 for p in paths if p.endswith(".jpg"))}, '
          f'other={sum(1 for p in paths if not p.endswith(".jpg"))})')
    basenames = [os.path.basename(p) for p in paths]
    assert len(set(basenames)) == len(basenames), 'basename 重复, resume 依赖唯一文件名'

    # ── 模型 (复用 run_sota_comparison 加载器) ──
    print('加载水印模型 ...')
    rw_model, rw_len = S.load_robust_wide_model()
    tm_model, tm_len = S.load_trustmark_model()
    ss_model, ss_len = S.load_stable_signature_model()
    wa_model, wa_len = S.load_watermark_anything_model()
    print(f'  Robust-Wide: {"OK" if rw_model is not None else "FAIL"} (msg_len={rw_len})')
    print(f'  TrustMark: {"OK" if tm_model is not None else "FAIL"} (msg_len={tm_len})')
    print(f'  Stable Signature: {"OK" if ss_model is not None else "FAIL"} (msg_len={ss_len})')
    print(f'  Watermark Anything: {"OK" if wa_model is not None else "FAIL"} (msg_len={wa_len})')
    models = (rw_model, tm_model, ss_model, wa_model)
    msglens = (rw_len, tm_len, ss_len, wa_len)

    lpips_fn = make_lpips_fn(args.lpips_net)
    print(f'LPIPS net={args.lpips_net} ready; torch.cuda={torch.cuda.is_available()}')

    # ── resume ──
    done = set()
    prev_meta = None
    if args.resume and os.path.exists(args.out):
        try:
            with open(args.out) as f:
                old = json.load(f)
            prev_meta = old.get('meta')
            rows = old.get('per_image', [])
            done = {r['img'] for r in rows}
            print(f'resume: 已有 {len(rows)} 条记录, 跳过 {len(done)} 张')
        except Exception as e:
            print(f'[!] resume 读取失败({e}), 重新开始')
            rows = []
    else:
        rows = []

    meta = prev_meta or {
        'n_total': len(paths), 'datasets': datasets, 'imgsize': 256,
        'delta': args.delta, 'msglen': args.msglen, 'seed': 42,
        'lpips_net': args.lpips_net,
        'ssim': 'skimage.metrics.structural_similarity(data_range=256, channel_axis=2)',
        'psnr_fn': 'run_idea_probe.psnr',
        'paired_sign': 'paired_vs_ours[*].*_diff = ours(signal_check_v2) - method; '
                       'ssim/psnr 正值=ours 更好, lpips 负值=ours 更好(越低越好)',
        'ts': datetime.datetime.now().isoformat(),
    }
    meta['n_total'] = len(paths)
    meta['datasets'] = datasets

    # ── 主循环 (消息生成逐字复现 run_sota_comparison.main) ──
    rng = np.random.RandomState(42)
    n_done = 0
    t_loop = time.time()
    for idx, p in enumerate(paths):
        img = load(p, 256)
        msg_long = rng.randint(0, 2, args.msglen).astype(np.uint8)   # 每张图固定消费一次
        name = basenames[idx]
        if name in done:
            continue

        row = {'img': name}
        try:
            wm, err = embed_all(models, img, msg_long, args.delta, msglens)
            for k in METHOD_KEYS:
                s = SUF[k]
                if wm.get(k) is None:
                    row[f'psnr_{s}'] = None
                    row[f'ssim_{s}'] = None
                    row[f'lpips_{s}'] = None
                    row.setdefault('error', {})[k] = err.get(k, 'unknown')
                    continue
                w = wm[k]
                row[f'psnr_{s}'] = float(psnr(img, w))
                row[f'ssim_{s}'] = float(ssim_fn(img, w))
                row[f'lpips_{s}'] = float(lpips_fn(img, w))
        except Exception as e:
            row.setdefault('error', {})['_fatal'] = repr(e)[:200]
            for k in METHOD_KEYS:
                s = SUF[k]
                row.setdefault(f'psnr_{s}', None)
                row.setdefault(f'ssim_{s}', None)
                row.setdefault(f'lpips_{s}', None)
            print(f'  ! 图 {name} 失败: {row["error"]}')

        rows.append(row)
        n_done += 1

        if n_done % args.progress_every == 0:
            el = time.time() - t_loop
            rate = n_done / el
            left = (len(paths) - len(done) - n_done)
            eta = left / rate if rate > 0 else 0
            print(f'  [{len(rows)}/{len(paths)}] {name[:20]} elapsed={el:.0f}s '
                  f'rate={rate:.2f} img/s eta={eta/60:.1f} min', flush=True)

        if n_done % args.save_every == 0:
            per_method, paired = aggregate(rows)
            atomic_dump({'meta': dict(meta, ts=datetime.datetime.now().isoformat(),
                                      updated=True, n_rows=len(rows)),
                         'per_method': per_method, 'paired_vs_ours': paired,
                         'per_image': rows}, args.out)

    # ── 汇总 ──
    per_method, paired = aggregate(rows)
    meta['n_rows'] = len(rows)
    meta['elapsed_sec'] = round(time.time() - t_start, 1)
    out = {'meta': meta, 'per_method': per_method, 'paired_vs_ours': paired, 'per_image': rows}
    atomic_dump(out, args.out)

    print('\n' + '=' * 92)
    print(f'{"method":<20}{"n":>5}{"PSNR":>18}{"SSIM":>18}{"LPIPS":>18}')
    for k in METHOD_KEYS:
        d = per_method[k]
        print(f'{k:<20}{d["n"]:>5}'
              f'{fmt(d["psnr_mean"],2)+"±"+fmt(d["psnr_std"],2):>18}'
              f'{fmt(d["ssim_mean"],4)+"±"+fmt(d["ssim_std"],4):>18}'
              f'{fmt(d["lpips_mean"],4)+"±"+fmt(d["lpips_std"],4):>18}')
    print('-' * 92)
    print(f'paired vs ours ({OURS}): diff = ours - method')
    for k, d in paired.items():
        print(f'  {k:<20} ssim {d["ssim_diff"]:+.4f} {d["ssim_ci95"]} | '
              f'lpips {d["lpips_diff"]:+.4f} {d["lpips_ci95"]} | '
              f'psnr {d["psnr_diff"]:+.3f} {d["psnr_ci95"]} | n={d["n_paired"]}')
    print('=' * 92)
    print(f'保存: {args.out}  (rows={len(rows)}, 耗时 {meta["elapsed_sec"]}s)')

    # ── 口径自检 (仅全量时) ──
    if len(rows) >= 600:
        expect = {'dwt_dct': 48.09, 'signal_check_v2': 47.11,
                  'robust_wide': 33.27, 'trustmark': 41.49}
        print('\n口径核对 (|dev| 必须 <= 0.15 dB):')
        ok = True
        for k, e in expect.items():
            m = per_method[k]['psnr_mean']
            dev = None if m is None else m - e
            flag = 'OK' if (dev is not None and abs(dev) <= 0.15) else 'MISMATCH'
            ok = ok and flag == 'OK'
            print(f'  {k:<18} got={m if m is None else round(m,3)} expect={e} dev={dev} -> {flag}')
        if not ok:
            print('[!] PSNR 口径不一致!')
            sys.exit(2)


if __name__ == '__main__':
    main()
