#!/usr/bin/env python3
"""对 run_hires_demo_samples.py 生成的高分辨率样本打**自然度**分并排序.

自然度: DeepSeek 对比 (orig, att) 打 0-10 分 (复用 run_mllm_benchmark.deepseek_naturalness)
接缝  : 编辑框边界跨边梯度 编辑后/编辑前, ~1 表示无接缝 (越小越自然)

输出: ../results/hires_sample_scores_<ts>.json  (含排序结果, 供选图)
用法: cd code && python evaluate_hires_samples.py
"""
import sys, os, json, time, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image

for k in ['ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)
os.environ.setdefault('http_proxy', 'http://127.0.0.1:7897')
os.environ.setdefault('https_proxy', 'http://127.0.0.1:7897')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f'{BASE}/results/attackset_hires_sample'


def seam_ratio(orig, att, bbox, d=3):
    h, w = orig.shape[:2]
    x1, y1, x2, y2 = [int(v * s) for v, s in zip(bbox, (w, h, w, h))]
    g_o = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY).astype(float)
    g_a = cv2.cvtColor(att, cv2.COLOR_RGB2GRAY).astype(float)

    def border(g):
        H, W = g.shape
        vals = []
        sx, sy = max(1, (x2 - x1) // 60), max(1, (y2 - y1) // 60)
        for x in range(max(x1 + d, 0), min(x2 - d, W - 1), sx):
            vals += [abs(g[max(0, y1 - d), x] - g[min(H - 1, y1 + d), x]),
                     abs(g[max(0, y2 - d), x] - g[min(H - 1, y2 + d), x])]
        for y in range(max(y1 + d, 0), min(y2 - d, H - 1), sy):
            vals += [abs(g[y, max(0, x1 - d)] - g[y, min(W - 1, x1 + d)]),
                     abs(g[y, max(0, x2 - d)] - g[y, min(W - 1, x2 + d)])]
        return float(np.mean(vals)) if vals else 0.0

    return border(g_a) / (border(g_o) + 1e-6)


def parse_nat(raw):
    try:
        o = json.loads(raw[raw.find('{'): raw.rfind('}') + 1])
        return float(o.get('naturalness', -1)), str(o.get('edited_region', ''))[:120]
    except Exception:
        return -1.0, raw[:120]


def main():
    from run_mllm_benchmark import load_env, deepseek_naturalness
    _, _, _, ds_key, ds_url, ds_model = load_env()

    meta = json.load(open(f'{OUT}/meta.json'))
    rows = []
    t0 = time.time()
    for k, s in enumerate(meta['samples']):
        o = np.array(Image.open(f'{OUT}/images/{s["id"]}_orig.png').convert('RGB'))
        a = np.array(Image.open(f'{OUT}/images/{s["id"]}_att.png').convert('RGB'))
        nat, desc = parse_nat(deepseek_naturalness(ds_url, ds_key, ds_model, o, a))
        seam = seam_ratio(o, a, s['bbox'])
        rows.append(dict(id=s['id'], source=s['source'], instruction=s['instruction'],
                         bbox=s['bbox'], wh=[s['w'], s['h']], psnr=s['psnr'],
                         naturalness=nat, seam=seam, nat_desc=desc))
        print(f'  [{k+1}/{len(meta["samples"])}] {s["id"]:>6} nat={nat:4.1f} '
              f'seam={seam:.2f} psnr={s["psnr"]:.1f}  {desc[:60]}  ({time.time()-t0:.0f}s)',
              flush=True)

    def combined(r):
        return max(r['naturalness'], 0) * (1.0 / (1.0 + max(r['seam'] - 1.0, 0)))
    rows.sort(key=combined, reverse=True)
    outp = f'{BASE}/results/hires_sample_scores_{datetime.datetime.now():%Y%m%d_%H%M%S}.json'
    json.dump(dict(generated=datetime.datetime.now().isoformat(), samples=rows),
              open(outp, 'w'), ensure_ascii=False, indent=2)
    print('\n== ranked (naturalness / seam) ==')
    for r in rows:
        print(f"  nat={r['naturalness']:4.1f} seam={r['seam']:.2f} {r['wh']} {r['id']:>6}  {r['instruction'][:40]}")
    print('saved', outp)


if __name__ == '__main__':
    main()
