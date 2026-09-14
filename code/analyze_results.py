#!/usr/bin/env python3
"""ICASSP 2027 — 统计检验: 对 idea_probe 结果做 paired 检验.

用法:
  cd code && python analyze_results.py                 # 分析最新 idea_probe_*.json
  python analyze_results.py results/idea_probe_xxx.json

对 removal 的每个 (数据集, 攻击) 场景做:
  - paired t-test (ttest_rel) 与 Wilcoxon signed-rank
  - 对比: semantic_ra vs uniform, full vs uniform, semantic vs uniform
  - 效应量 Cohen's d (有符号)
输出: 终端表格 + results/analysis_<时间戳>.json
"""
import sys, os, json, glob, datetime
import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = f'{BASE}/results'


def cohen_d(a, b):
    """有符号 Cohen's d: (mean(a)-mean(b)) / pooled_std. 正 = a 优于 b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    sd = np.sqrt(((len(a) - 1) * a.std(ddof=1) ** 2 +
                  (len(b) - 1) * b.std(ddof=1) ** 2) / (2 * n - 2))
    return float((a.mean() - b.mean()) / (sd + 1e-12))


def fmt_p(p):
    if p < 1e-4:
        return f'{p:.2e}'
    return f'{p:.4f}'


def main():
    if len(sys.argv) > 1:
        src = sys.argv[1]
    else:
        fs = sorted(glob.glob(os.path.join(RESULTS_DIR, 'idea_probe_*.json')))
        if not fs:
            sys.exit('没有找到 idea_probe_*.json, 请先运行 run_idea_probe.py')
        src = fs[-1]
    R = json.load(open(src))
    rem = R.get('removal', {})
    if not rem:
        sys.exit(f'{src} 中没有 removal 表 (可能用了 --quick 但结果未含 removal?)')
    first = next(iter(rem.values()))
    if 'ba_per_image' not in next(iter(next(iter(first.values())).values())):
        sys.exit(f'{src} 是旧版结果 (无 per-image 数据). '
                 f'请重新运行: cd code && python run_idea_probe.py')

    print('=' * 78)
    print(f'  统计检验 — {os.path.basename(src)}')
    print('  (paired: 同一张图各方案共用同消息同移除掩膜, n 严格一致)')
    print('=' * 78)

    out = {'source': os.path.basename(src),
           'ts': datetime.datetime.now().isoformat(), 'tests': {}}
    for dname in rem:
        for key in ['rem30', 'rem50']:
            row = rem[dname][key]
            n = row['uniform']['n']
            print(f'\n[{dname} / {key}]  n = {n}')
            print(f"  {'对比':<28}{'mean差(pt)':>12}{'t-stat':>10}{'p(t)':>12}"
                  f"{'p(Wilc)':>12}{'Cohen d':>10}")
            base = np.array(row['uniform']['ba_per_image'])
            for target, label in [('semantic', 'semantic vs uniform'),
                                  ('semantic_ra', 'semantic_ra vs uniform'),
                                  ('full', 'full vs uniform')]:
                a = np.array(row[target]['ba_per_image'])
                if len(a) != len(base):
                    print(f'  !! {label}: 长度不一致 ({len(a)} vs {len(base)}), 跳过')
                    continue
                t, p_t = stats.ttest_rel(a, base)
                try:
                    w, p_w = stats.wilcoxon(a, base)
                except ValueError:
                    w, p_w = float('nan'), float('nan')  # 差全为 0 时无定义
                d = cohen_d(a, base)
                sig = ' ***' if p_t < 0.001 else (' **' if p_t < 0.01 else
                      (' *' if p_t < 0.05 else ''))
                print(f'  {label:<28}{(a.mean()-base.mean())*100:>+11.1f}%'
                      f'{t:>10.2f}{fmt_p(p_t):>12}{fmt_p(p_w):>12}{d:>+10.2f}{sig}')
                out['tests'][f'{dname}_{key}_{target}'] = {
                    'mean_diff_pt': float((a.mean() - base.mean()) * 100),
                    't': float(t), 'p_t': float(p_t),
                    'p_wilcoxon': float(p_w) if p_w == p_w else None,
                    'cohen_d': d, 'n': int(n)}

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    dst = os.path.join(RESULTS_DIR, f'analysis_{ts}.json')
    with open(dst, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f'\n结果已保存: {dst}')
    print('解读: p<0.05 差异显著; Cohen d≈0.2 小 / 0.5 中 / 0.8 大.')


if __name__ == '__main__':
    main()
