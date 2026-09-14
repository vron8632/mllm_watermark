#!/usr/bin/env python3
"""动机图候选预览: 从攻击集里挑"错位明显 + 画面干净"的例子, 拼成一张对比图供选择.

判据:
  score  = (框外被破坏比例) x (框内完好比例)   —— 错位越明显越大
  edges  = 原图平均梯度 (越低 = 背景越干净, 橙斑越显眼)
输出: paper/figures/gpt_assets/candidates_teaser.png

用法: cd code && python preview_teaser_candidates.py
"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Noto Sans CJK JP',
                                   'SimHei', 'Droid Sans Fallback', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from PIL import Image
from dct_watermark import _block_grid, _demodulate_bit

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f'{BASE}/paper/figures/gpt_assets'
AS = f'{BASE}/results/attackset_20260817_082819/images'
META = f'{BASE}/results/attackset_20260817_082819/meta.json'
RED = (214, 39, 40)


def load(p, s=256):
    im = np.array(Image.open(p).convert('RGB'))
    if s:
        im = np.array(Image.fromarray(im).resize((s, s), Image.BILINEAR))
    h, w = im.shape[:2]
    return im[:h - h % 8, :w - w % 8]


def y_ch(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)


def fail_map(o, a, delta=15):
    H, W, nh, nw = _block_grid(*o.shape[:2])
    ck = np.random.RandomState(42).randint(0, 2, (nh, nw)).astype(np.uint8)
    yo, ya = y_ch(o)[:H, :W], y_ch(a)[:H, :W]
    fm = np.zeros((nh, nw))
    for i in range(nh):
        for j in range(nw):
            do = cv2.dct(yo[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32))
            da = cv2.dct(ya[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32))
            co, ca = _demodulate_bit(do[3, 2], delta), _demodulate_bit(da[3, 2], delta)
            if co != ca:
                fm[i, j] = 1.0
            elif ca != ck[i, j]:
                fm[i, j] = 0.5
    return fm, nh, nw


def stats(o, a, bbox):
    fm, nh, nw = fail_map(o, a)
    h, w = o.shape[:2]
    x1, y1, x2, y2 = [v * d for v, d in zip(bbox, (w, h, w, h))]
    em = np.zeros((nh, nw))
    for i in range(nh):
        for j in range(nw):
            cy, cx = (i + .5) * 8, (j + .5) * 8
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                em[i, j] = 1
    ins, out = em == 1, em == 0
    if ins.sum() == 0 or out.sum() == 0:
        return None
    holes = (fm[ins] == 0).mean()
    spill = (fm[out] > 0).mean()
    edges = cv2.Sobel(cv2.cvtColor(o, cv2.COLOR_RGB2GRAY), cv2.CV_32F, 1, 0, 3)
    edges = float(np.abs(cv2.Sobel(cv2.cvtColor(o, cv2.COLOR_RGB2GRAY),
                                   cv2.CV_32F, 1, 1, 3)).mean())
    return dict(spill=spill, holes=holes, score=spill * holes,
                n_in=int(ins.sum()), edges=edges, fm=fm, nh=nh, nw=nw)


def main():
    meta = json.load(open(META))
    rows = []
    for s in meta['samples']:
        cid = s['id']
        ins = s.get('instruction', '')
        if isinstance(ins, str) and ins.strip().startswith('{'):
            try:
                ins = json.loads(ins).get('instruction', ins)
            except Exception:
                pass
        try:
            o = load(f'{AS}/orig_{cid}.png')
            a = load(f'{AS}/att_{cid}.png')
        except Exception:
            continue
        st = stats(o, a, s['bbox'])
        if st is None:
            continue
        rows.append((st, cid, s['source'], ins, o, a))
    # 错位明显 + 编辑框别太小(>=80 块)
    rows = [r for r in rows if r[0]['n_in'] >= 80]
    rows.sort(key=lambda r: -r[0]['score'])

    print(f"{'score':>6} {'spill':>6} {'holes':>6} {'n_in':>5} {'edges':>6} {'id':>5}  instruction")
    for st, cid, src, ins, o, a in rows[:16]:
        print(f"{st['score']:6.3f} {st['spill']:6.2f} {st['holes']:6.2f} "
              f"{st['n_in']:5d} {st['edges']:6.1f} {cid:>5}  {ins[:28]}")

    top = rows[:12]
    ncol, nrow = 3, 4
    fig, axes = plt.subplots(nrow, ncol * 2, figsize=(3.1 * ncol * 2, 2.05 * nrow))
    for k, (st, cid, src, ins, o, a) in enumerate(top):
        r, c = divmod(k, ncol)
        fm, nh, nw = st['fm'], st['nh'], st['nw']
        S = o.shape[0]
        # bbox from meta
        bbox = next(s['bbox'] for s in meta['samples'] if s['id'] == cid)
        x1, y1, x2, y2 = [v * S for v in bbox]
        for t, img in enumerate((o, a)):
            ax = axes[r, c * 2 + t]
            ax.imshow(img)
            ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                         ec='#d62728', lw=1.6, ls='--'))
            if t == 1:
                ax.imshow(np.ma.masked_where(fm <= 0, fm), cmap='Oranges', vmin=0,
                          vmax=1.2, alpha=0.6, interpolation='nearest',
                          extent=(0, S, S, 0))
            ax.axis('off')
        axes[r, c * 2].set_title(f"{cid} {src[:10]}  score={st['score']:.2f} "
                                 f"edge={st['edges']:.0f}", fontsize=7.5, loc='left')
        axes[r, c * 2 + 1].set_title(ins[:34], fontsize=7.5, loc='left', color='#555')
    fig.suptitle('Teaser candidates: original+bbox  |  edited+damage   (pick one)',
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(f'{OUT}/candidates_teaser.png', dpi=170, bbox_inches='tight', facecolor='white')
    print('saved', f'{OUT}/candidates_teaser.png')


if __name__ == '__main__':
    main()
