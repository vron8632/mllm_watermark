#!/usr/bin/env python3
"""Discussion 配图 (fig4_discussion): 为什么必须用**信号级**检测, 而不是 MLLM 视觉检测.

三个面板, 全部为**真实数据** (与论文实验同协议, attackset_20260817_082819):
  (A) edited image            : MLLM+扩散编辑后的真实图像 (bbox 外严格保持原图)
  (B) diffusion residual      : |att-orig| 放大 4x 后的残差 —— MLLM 看到的就是这样,
                                无法可靠区分"被改写区域"与"原本就有的纹理/边缘"
  (C) watermark check-bit map : 真实逐块 check-bit 解调结果 (绿=校验一致, 橙=不一致),
                                与水印密钥绑定, 定位精确且确定

协议常量与 eval_attackset.py 完全一致:
  msg = RandomState(42).randint(0,2,256), Δ=15, payload=(4,1), check=(3,2), check seed=42

用法: cd code && python make_discussion_figure.py [--ids 0002_43,0045_42,0032_43]
输出: ../paper/figures/fig4_discussion.pdf / .png
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from run_idea_probe import y_channel, _block_grid, _demodulate_bit
from run_signal_ra import embed_signal, check_bits, CHECK_COEF, PAY_COEF

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = f'{BASE}/paper/figures'
AS = f'{BASE}/results/attackset_hires_sample'


def crop8(a):
    return a[:a.shape[0] - a.shape[0] % 8, :a.shape[1] - a.shape[1] % 8]


def paste_edit(wm, att, bbox01):
    """论文发布协议: 把 att 的 bbox 区域贴回水印图, bbox 外保持水印图."""
    H, W = wm.shape[:2]
    x1, y1, x2, y2 = bbox01
    bx1, by1 = int(round(x1 * W)), int(round(y1 * H))
    bx2, by2 = int(round(x2 * W)), int(round(y2 * H))
    out = wm.copy()
    out[by1:by2, bx1:bx2] = att[by1:by2, bx1:bx2]
    return out


def check_bit_map(att_wm, delta=15):
    """真实逐块校验位解调: 1=与期望一致, 0=不一致 (即被编辑破坏)."""
    H, W, n_h, n_w = _block_grid(*att_wm.shape[:2])
    y = y_channel(att_wm)[:H, :W]
    ck = check_bits(H, W, n_h, n_w)
    m = np.ones((n_h, n_w), dtype=np.uint8)
    for i in range(n_h):
        for j in range(n_w):
            d = cv2.dct(y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8].astype(np.float32))
            if _demodulate_bit(d[CHECK_COEF], delta) != ck[i, j]:
                m[i, j] = 0
    return m


def residual_vis(orig, att, gain=4.0):
    """残差放大可视化: MLLM 实际"看到"的编辑痕迹就是这种量级."""
    r = np.abs(att.astype(np.float32) - orig.astype(np.float32))
    return np.clip(r * gain, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', default='0002_43,0045_42,0032_43')
    ap.add_argument('--msglen', type=int, default=256)
    ap.add_argument('--delta', type=int, default=15)
    args = ap.parse_args()

    meta = json.load(open(f'{AS}/meta.json'))
    by_id = {s['id']: s for s in meta['samples']}
    rng = np.random.RandomState(42)
    msg = rng.randint(0, 2, args.msglen).astype(np.uint8)

    # 若无 EN 表项则回退到 meta 里的英文 instruction (hires 样本均走此分支,
    # 无需再维护硬编码表; 仅保留以兼容 256px 目录的旧 id)
    EN = {}

    ids = args.ids.split(',')
    n = len(ids)
    fig, axes = plt.subplots(n, 3, figsize=(12.4, 3.9 * n))
    axes = np.atleast_2d(axes)
    cmap = ListedColormap(['#D55E00', '#0072B2'])   # 0=fail(orange), 1=ok(blue)
    # 配色依据: Okabe-Ito 无障碍调色板 (对红绿色盲仍可分辨)

    for r, cid in enumerate(ids):
        s = by_id[cid]
        orig = crop8(np.array(Image.open(f'{AS}/images/{cid}_orig.png').convert('RGB')))
        att = crop8(np.array(Image.open(f'{AS}/images/{cid}_att.png').convert('RGB')))
        wm = embed_signal(orig, msg, args.delta)
        att_wm = paste_edit(wm, att, s['bbox'])
        H, W, n_h, n_w = _block_grid(*att_wm.shape[:2])

        x1, y1, x2, y2 = [int(v * q) for v, q in zip(s['bbox'], (W, H, W, H))]

        # (A) edited image
        ax = axes[r, 0]
        ax.imshow(att)
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                     ec='white', lw=2.2, ls='--'))
        ax.set_title(f'(A) edited image — {EN.get(cid, s["instruction"][:26])}',
                     fontsize=11, loc='left', color='#1f4e79')
        ax.axis('off')

        # (B) residual
        ax = axes[r, 1]
        ax.imshow(residual_vis(orig, att))
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                     ec='white', lw=2.2, ls='--'))
        ax.set_title('(B) diffusion residual ($\\times4$) — what vision sees',
                     fontsize=11, loc='left', color='#6a6a6a')
        ax.axis('off')

        # (C) check-bit map
        ax = axes[r, 2]
        ax.imshow(att)
        cm = check_bit_map(att_wm, args.delta)
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=1, alpha=0.45,
                  interpolation='nearest', extent=(0, W, H, 0))
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                     ec='white', lw=2.2, ls='--'))
        n_fail = int((cm == 0).sum())
        ax.set_title(f'(C) check-bit map — {n_fail}/{n_h * n_w} blocks flagged',
                     fontsize=11, loc='left', color='#b34700')
        ax.axis('off')

    handles = [mpatches.Patch(color='#D55E00', label='check bit inconsistent (edited)'),
               mpatches.Patch(color='#0072B2', label='check bit consistent (trusted)'),
               mpatches.Patch(facecolor='none', edgecolor='white', linestyle='--',
                              label='MLLM-declared edit region')]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10.5,
               frameon=False, bbox_to_anchor=(0.5, -0.004))
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    for ext in ('pdf', 'png'):
        fig.savefig(f'{FIG_DIR}/fig4_discussion.{ext}', dpi=220,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'fig4_discussion done ({n} rows), saved to {FIG_DIR}')


if __name__ == '__main__':
    main()
