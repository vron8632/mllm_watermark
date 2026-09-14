#!/usr/bin/env python3
"""用**客观指标**筛选高分辨率编辑样本 (不依赖 LLM 打分).

背景: DeepSeek 的 naturalness 会把"几乎没编辑"的图打成高分 (0031 甜甜圈仍在却得 9.0),
因此必须改用像素级/梯度级指标.

指标 (全部自动计算, 可复现):
  edit_strength  = bbox 内 |att-orig| 的均值      -> 编辑是否真的发生 (太低=没编辑)
  leak           = bbox 外 |att-orig| 的均值      -> 局部性 (应为 ~0, 否则不是局部编辑)
  seam           = bbox 边界跨边梯度比 att/orig   -> ~1 表示无接缝 (越大越像拼贴)
  edge_in        = bbox 内梯度能量比 att/orig     -> ~1 表示纹理自然延续
  score          = 编辑足够 + 无接缝 + 局部性 的加权

用法: cd code && python screen_hires_samples.py --top 12
输出: 终端排名 + ../results/hires_screen_<ts>.json + 对比图 (供人工抽查)
"""
import sys, os, json, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = f'{BASE}/results/attackset_hires_sample'


def px(o, a, bbox):
    h, w = o.shape[:2]
    x1, y1, x2, y2 = [int(v * q) for v, q in zip(bbox, (w, h, w, h))]
    x2, y2 = max(x2, x1 + 8), max(y2, y1 + 8)
    ins = np.abs(a[y1:y2, x1:x2].astype(float) - o[y1:y2, x1:x2].astype(float)).mean()
    out = np.concatenate([(a[:y1] - o[:y1]).ravel(), (a[y2:] - o[y2:]).ravel(),
                          (a[y1:y2, :x1] - o[y1:y2, :x1]).ravel(),
                          (a[y1:y2, x2:] - o[y1:y2, x2:]).ravel()]).astype(float)
    return ins, float(np.abs(out).mean()), (x1, y1, x2, y2)


def seam_and_edge(o, a, box, d=3):
    x1, y1, x2, y2 = box
    go = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY).astype(float)
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(float)
    H, W = go.shape

    def border(g):
        vals = []
        sx, sy = max(1, (x2 - x1) // 60), max(1, (y2 - y1) // 60)
        for x in range(max(x1 + d, 0), min(x2 - d, W - 1), sx):
            vals += [abs(g[max(0, y1 - d), x] - g[min(H - 1, y1 + d), x]),
                     abs(g[max(0, y2 - d), x] - g[min(H - 1, y2 + d), x])]
        for y in range(max(y1 + d, 0), min(y2 - d, H - 1), sy):
            vals += [abs(g[y, max(0, x1 - d)] - g[y, min(W - 1, x1 + d)]),
                     abs(g[y, max(0, x2 - d)] - g[y, min(W - 1, x2 + d)])]
        return float(np.mean(vals)) if vals else 0.0

    seam = border(ga) / (border(go) + 1e-6)

    def sobel(g):
        g = np.ascontiguousarray(g, dtype=np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return gx ** 2 + gy ** 2

    eo = sobel(go[y1:y2, x1:x2]).mean()
    ea = sobel(ga[y1:y2, x1:x2]).mean()
    return float(seam), float(ea / (eo + 1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=12)
    ap.add_argument('--min-strength', type=float, default=8.0,
                    help='bbox 内像素差下限: 低于该值认为"编辑没发生"')
    args = ap.parse_args()

    meta = json.load(open(f'{D}/meta.json'))
    rows = []
    for s in meta['samples']:
        o = np.array(Image.open(f'{D}/images/{s["id"]}_orig.png').convert('RGB'))
        a = np.array(Image.open(f'{D}/images/{s["id"]}_att.png').convert('RGB'))
        ins, leak, box = px(o, a, s['bbox'])
        seam, edge = seam_and_edge(o, a, box)
        edited = ins >= args.min_strength
        # 接缝越小越好, 局部性越强越好; 编辑必须真的发生
        seam_pen = 1.0 / (1.0 + max(seam - 1.0, 0) * 3.0)
        edge_pen = 1.0 / (1.0 + abs(edge - 1.0))
        loc_pen = 1.0 / (1.0 + leak / 3.0)
        score = (seam_pen * edge_pen * loc_pen) if edited else 0.0
        rows.append(dict(id=s['id'], cid=s['cid'], seed=s['seed'], wh=[s['w'], s['h']],
                         instruction=s['instruction'], bbox=s['bbox'],
                         edit_strength=ins, leak=leak, seam=seam, edge_ratio=edge,
                         edited=bool(edited), psnr=s['psnr'], score=score))

    rows.sort(key=lambda r: -r['score'])
    print(f"{'id':>8} {'wh':>10} {'edit':>6} {'leak':>6} {'seam':>5} {'edge':>5} "
          f"{'psnr':>5} {'score':>6}  instruction")
    for r in rows[:args.top]:
        print(f"{r['id']:>8} {r['wh'][0]:>4}x{r['wh'][1]:<4} {r['edit_strength']:6.1f} "
              f"{r['leak']:6.3f} {r['seam']:5.2f} {r['edge_ratio']:5.2f} "
              f"{r['psnr']:5.1f} {r['score']:6.3f}  {r['instruction'][:34]} "
              f"{'' if r['edited'] else '  [NO EDIT]'}")

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    outp = f'{BASE}/results/hires_screen_{ts}.json'
    json.dump(dict(generated=datetime.datetime.now().isoformat(),
                   criteria=dict(min_strength=args.min_strength), samples=rows),
              open(outp, 'w'), ensure_ascii=False, indent=2)
    print('saved', outp)

    # 对比图供人工抽查
    top = rows[:args.top]
    ncol = 3
    nrow = int(np.ceil(len(top) / ncol))
    fig, axes = plt.subplots(nrow, ncol * 2, figsize=(2.9 * ncol * 2, 2.0 * nrow))
    axes = np.atleast_2d(axes)
    for k, r in enumerate(top):
        rr, cc = divmod(k, ncol)
        sid = r['id']
        o = np.array(Image.open(f'{D}/images/{sid}_orig.png').convert('RGB'))
        a = np.array(Image.open(f'{D}/images/{sid}_att.png').convert('RGB'))
        h, w = o.shape[:2]
        x1, y1, x2, y2 = [int(v * q) for v, q in zip(r['bbox'], (w, h, w, h))]
        for t, im in enumerate((o, a)):
            ax = axes[rr, cc * 2 + t]
            ax.imshow(im)
            ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                         ec='#d62728', lw=1.5, ls='--'))
            ax.axis('off')
        axes[rr, cc * 2].set_title(
            f"#{k+1} {sid}  {w}x{h}  score={r['score']:.2f}", fontsize=8,
            loc='left', color='#1f4e79')
        axes[rr, cc * 2 + 1].set_title(
            f"seam={r['seam']:.2f} edge={r['edge_ratio']:.2f} "
            f"edit={r['edit_strength']:.0f}", fontsize=8, loc='left', color='#b03030')
    fig.tight_layout()
    outp2 = f'{BASE}/paper/figures/gpt_assets/screened_candidates.png'
    fig.savefig(outp2, dpi=150, bbox_inches='tight', facecolor='white')
    print('saved', outp2)


if __name__ == '__main__':
    main()
