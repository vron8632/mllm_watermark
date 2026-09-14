#!/usr/bin/env python3
"""按"错位明显" + "inpaint 自然度"联合给动机图候选排序.

动机图要求编辑**看不出痕迹**, 所以除了 mismatch score 还要评自然度:
  naturalness = deepseek 对比编辑前后图打 0-10 分 (复用 run_mllm_benchmark.deepseek_naturalness)
  seam        = 编辑后 bbox 边界梯度 / 编辑前, 越大越像有接缝

输出:
  results/teaser_candidates_<ts>.json
  paper/figures/gpt_assets/candidates_teaser_ranked.png  (按自然度排序的对比图)

用法: cd code && python rank_teaser_candidates.py --top 12
"""
import sys, os, json, time, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from preview_teaser_candidates import load, stats, META, AS, OUT

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Noto Sans CJK JP',
                                   'SimHei', 'Droid Sans Fallback', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def seam_ratio(orig, att, bbox, d=3):
    """bbox 边界处的跨边梯度: 编辑后 / 编辑前. ~1 表示无接缝."""
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
        s = raw[raw.find('{'): raw.rfind('}') + 1]
        o = json.loads(s)
        return float(o.get('naturalness', -1)), str(o.get('edited_region', ''))[:60]
    except Exception:
        return -1.0, raw[:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=12)
    ap.add_argument('--minin', type=int, default=60, help='编辑框最小块数')
    args = ap.parse_args()

    from run_mllm_benchmark import load_env, deepseek_naturalness
    _, _, _, ds_key, ds_url, ds_model = load_env()

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
        if st is None or st['n_in'] < args.minin:
            continue
        st['seam'] = seam_ratio(o, a, s['bbox'])
        rows.append(dict(st=st, cid=cid, src=s['source'], ins=ins, o=o, a=a, bbox=s['bbox']))

    rows.sort(key=lambda r: -r['st']['score'])
    cand = rows[:args.top]
    print(f'候选 {len(cand)} 个, 开始评自然度...')
    t0 = time.time()
    for k, r in enumerate(cand):
        raw = deepseek_naturalness(ds_url, ds_key, ds_model, r['o'], r['a'])
        r['nat'], r['nat_desc'] = parse_nat(raw)
        print(f"  [{k+1}/{len(cand)}] {r['cid']} nat={r['nat']:.0f} "
              f"seam={r['st']['seam']:.2f} score={r['st']['score']:.3f} "
              f"({time.time()-t0:.0f}s) {r['ins'][:22]}")

    # 综合: 自然度优先(看不到痕迹), 再看错位
    def combined(r):
        nat = max(r['nat'], 0)
        seam_pen = 1.0 / (1.0 + max(r['st']['seam'] - 1.0, 0))
        return nat * seam_pen * (0.5 + r['st']['score'])
    cand.sort(key=combined, reverse=True)

    print('\n== 最终排序 (自然度 / 接缝 / 错位) ==')
    print(f"{'nat':>4} {'seam':>5} {'score':>6} {'id':>5}  instruction")
    for r in cand:
        print(f"{r['nat']:4.0f} {r['st']['seam']:5.2f} {r['st']['score']:6.3f} "
              f"{r['cid']:>5}  {r['ins'][:26]}")

    # 结果 JSON
    out = dict(generated=datetime.datetime.now().isoformat(), top=args.top,
               candidates=[dict(cid=r['cid'], source=r['src'], instruction=r['ins'],
                                bbox=r['bbox'], naturalness=r['nat'], seam=r['st']['seam'],
                                score=r['st']['score'], spill=r['st']['spill'],
                                holes=r['st']['holes'], n_in=r['st']['n_in'],
                                nat_desc=r['nat_desc']) for r in cand])
    jp = f'{BASE}/results/teaser_candidates_{datetime.datetime.now():%Y%m%d_%H%M%S}.json'
    json.dump(out, open(jp, 'w'), ensure_ascii=False, indent=1)
    print('saved', jp)

    # 排序后的对比图
    ncol = 3
    nrow = int(np.ceil(len(cand) / ncol))
    fig, axes = plt.subplots(nrow, ncol * 2, figsize=(3.1 * ncol * 2, 2.15 * nrow))
    axes = np.atleast_2d(axes)
    for k, r in enumerate(cand):
        rr, cc = divmod(k, ncol)
        S = r['o'].shape[0]
        fm, nh, nw = r['st']['fm'], r['st']['nh'], r['st']['nw']
        x1, y1, x2, y2 = [v * S for v in r['bbox']]
        for t, img in enumerate((r['o'], r['a'])):
            ax = axes[rr, cc * 2 + t]
            ax.imshow(img)
            ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                         ec='#d62728', lw=1.6, ls='--'))
            if t == 1:
                ax.imshow(np.ma.masked_where(fm <= 0, fm), cmap='Oranges', vmin=0,
                          vmax=1.2, alpha=0.6, interpolation='nearest',
                          extent=(0, S, S, 0))
            ax.axis('off')
        axes[rr, cc * 2].set_title(f"#{k+1} {r['cid']}  nat={r['nat']:.0f} "
                                   f"seam={r['st']['seam']:.2f}", fontsize=8, loc='left',
                                   color='#1f4e79')
        axes[rr, cc * 2 + 1].set_title(r['ins'][:26], fontsize=8, loc='left', color='#555')
    fig.suptitle('Teaser candidates ranked by NATURALNESS (original+bbox | edited+damage)',
                 fontsize=12, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(f'{OUT}/candidates_teaser_ranked.png', dpi=170,
                bbox_inches='tight', facecolor='white')
    print('saved', f'{OUT}/candidates_teaser_ranked.png')


if __name__ == '__main__':
    main()
