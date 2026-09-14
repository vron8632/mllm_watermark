#!/usr/bin/env python3
"""Graphical Abstract (Elsevier/JISA).

要求 (Elsevier guide for authors):
  - 最小 531 x 1328 px (h x w), 即宽高比约 1:2.5 (横向)
  - 在 5 x 13 cm 尺寸下可读
  - 推荐 TIFF / EPS / PDF / MS Office

设计: 三段式横向叙事, 全部基于论文真实内容:
  (1) MLLM 提议语义编辑指令 + bbox; 扩散在 bbox 内执行 (编辑局部)
  (2) 但水印损伤不局限在 bbox 内 (真实 check-bit 失败图, 橙=受损)
  (3) 我们的防御: 用 check bit 逐块定位损伤, 软权重 + 膨胀后投票 -> BA 88.3%->94.5%

输出: ../paper/figures/graphical_abstract.pdf / .png
用法: cd code && python make_graphical_abstract.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = f'{BASE}/paper/figures'
HIRES = f'{BASE}/results/attackset_hires_sample'


def load(p):
    im = np.array(Image.open(p).convert('RGB'))
    h, w = im.shape[:2]
    return im[:h - h % 8, :w - w % 8]


def y_channel(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)


def check_fail_map(orig, att, seed=42):
    """真实逐块 check-bit 失败图 (与论文 fig_s5 / teaser 同一计算)."""
    from dct_watermark import _block_grid, _demodulate_bit
    h, w = orig.shape[:2]
    H, W, n_h, n_w = _block_grid(h, w)
    ck = np.random.RandomState(seed).randint(0, 2, (n_h, n_w)).astype(np.uint8)
    yo, ya = y_channel(orig)[:H, :W], y_channel(att)[:H, :W]
    fm = np.zeros((n_h, n_w))
    for i in range(n_h):
        for j in range(n_w):
            do = cv2.dct(yo[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32))
            da = cv2.dct(ya[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32))
            co = _demodulate_bit(do[3, 2], 15)
            ca = _demodulate_bit(da[3, 2], 15)
            if co != ca:
                fm[i, j] = 1.0
            elif ca != ck[i, j]:
                fm[i, j] = 0.5
    return fm, n_h, n_w


def main():
    hi = json.load(open(f'{HIRES}/meta.json'))
    hm = {s['id']: s for s in hi['samples']}
    cid = '0002_42'
    s = hm[cid]
    orig = load(f'{HIRES}/images/{cid}_orig.png')
    att = load(f'{HIRES}/images/{cid}_att.png')
    fm, n_h, n_w = check_fail_map(orig, att)
    h, w = orig.shape[:2]
    x1, y1, x2, y2 = [int(v * q) for v, q in zip(s['bbox'], (w, h, w, h))]

    # 横向 3 面板 (宽高比 ~2.5:1)
    fw, fh = 13.0, 5.2
    fig = plt.figure(figsize=(fw, fh))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.30], hspace=0.10, wspace=0.06,
                          left=0.01, right=0.99, top=0.90, bottom=0.02)

    def panel(ax, img, title, tcolor):
        ax.imshow(img, interpolation='lanczos')
        ax.axis('off')
        ax.set_title(title, fontsize=15, fontweight='bold', color=tcolor, pad=6)

    # ---------- (1) 局部编辑 ----------
    ax = fig.add_subplot(gs[0, 0])
    panel(ax, orig, '(1) MLLM proposes a local semantic edit', '#1f4e79')
    ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                 ec='#d62728', lw=3.0, ls=(0, (6, 4))))
    ax.text(0.03, 0.03, f'MLLM: "{s["instruction"][:34]}"', transform=ax.transAxes,
            fontsize=10.5, color='white', va='bottom',
            bbox=dict(boxstyle='round,pad=0.35', fc='#4b3f8f', ec='none', alpha=0.92))

    # ---------- (2) 损伤外溢 ----------
    ax = fig.add_subplot(gs[0, 1])
    panel(ax, att, '(2) But watermark damage spills outside it', '#b34700')
    ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                 ec='#d62728', lw=3.0, ls=(0, (6, 4))))
    ax.imshow(np.ma.masked_where(fm <= 0, fm), cmap='Oranges', vmin=0, vmax=1.2,
              alpha=0.55, interpolation='nearest', extent=(0, w, h, 0))

    # ---------- (3) 防御 ----------
    ax = fig.add_subplot(gs[0, 2])
    panel(ax, att, '(3) Signal-check defense: trust per block', '#2e7d32')
    ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                 ec='#d62728', lw=3.0, ls=(0, (6, 4))))
    ok = np.ma.masked_where(fm > 0, np.ones_like(fm))
    ax.imshow(ok, cmap='Greens', vmin=0, vmax=1, alpha=0.30,
              interpolation='nearest', extent=(0, w, h, 0))

    # ---------- 底部结论条 ----------
    ax = fig.add_subplot(gs[1, :])
    ax.axis('off')
    txt = ('Each 8$\\times$8 DCT block carries a payload bit and a pseudo-random check bit. '
           'Blocks whose check bit fails are soft-weighted and dilated before voting.\n'
           'Bit accuracy under MLLM-guided semantic editing:  '
           '$\\mathbf{88.3\\% \\rightarrow 94.5\\%}$  ($+6.2$pt), training-free, '
           'beating the best trained baseline at 256-bit payload.')
    ax.text(0.5, 0.62, txt, ha='center', va='center', fontsize=13.5, color='#1a1a1a',
            bbox=dict(boxstyle='round,pad=0.6', fc='#eaf3fb', ec='#2c6fbb', lw=1.6))
    ax.text(0.5, 0.06, 'Image Watermarking Robustness under MLLM-Guided Semantic Editing',
            ha='center', va='bottom', fontsize=12, style='italic', color='#555')

    fig.savefig(f'{FIG_DIR}/graphical_abstract.pdf', bbox_inches='tight', facecolor='white')
    # Elsevier 要求: 最小 1328x531 px, >=300 dpi, 比例 2.5:1 (500x200)
    # 不用 bbox_inches='tight', 保持 figsize 的精确 2.5:1 比例
    fig.savefig(f'{FIG_DIR}/graphical_abstract.png', dpi=300, facecolor='white')
    plt.close(fig)
    print(f'graphical_abstract done ({cid}, {w}x{h})')


if __name__ == '__main__':
    main()
