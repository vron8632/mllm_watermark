#!/usr/bin/env python3
"""用 n=500 结果重算 Table 2 / Table 3 / Table 5, 并输出 LaTeX 行与核对报告.

背景: 论文 Table 2/3/5 声称 n=500, 但归档数据曾是 n=200, 且 Table 3 的分层计数
      122+58+20=200 与表头 n=500 自相矛盾. 本脚本从 n=500 原始 per_image 重算全部
      数值, 使表头与分层计数一致.

分层阈值沿用论文定义 (不可改动, 否则与正文描述不符):
  Small: att_fail_frac < 0.15
  Mid  : 0.15 <= att_fail_frac <= 0.35
  Large: att_fail_frac > 0.35

用法:
  cd code && python recompute_n500.py <结果文件.json>          # 打印报告
  cd code && python recompute_n500.py <结果文件.json> --latex  # 额外输出 LaTeX 行
"""
import json, sys, argparse
import numpy as np

TH_SMALL, TH_LARGE = 0.15, 0.35


def mean(xs):
    return 100.0 * float(np.mean(xs)) if xs else float('nan')


def sd(xs):
    return 100.0 * float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('result_json')
    ap.add_argument('--latex', action='store_true')
    args = ap.parse_args()

    d = json.load(open(args.result_json))
    pi = d['per_image']
    n = len(pi)
    m = d.get('meta', {})

    print('=' * 72)
    print(f'从 n={n} 结果重算 (源: {args.result_json.split("/")[-1]})')
    print(f'  meta: {json.dumps({k: v for k, v in m.items() if k != "ts"}, default=str)}')
    print('=' * 72)

    # ---------- Table 2 / Table 5: 主指标 ----------
    def get(k):
        return [r[k] for r in pi if k in r and r[k] is not None]

    uni, v1 = get('uniform_ba'), get('signal_ra_ba')
    soft, v2 = get('signal_soft_ba'), get('signal_soft_dilate_ba')
    orc = get('bbox_ra_ba')

    print('\n### Table 2 (主实验) / Table 5 (消融) 的绝对 BA')
    print(f'{"配置":<28}{"BA (%)":>9}{"±SD":>8}{"增益(pt)":>10}')
    for name, xs in [('Uniform (no defense)', uni), ('v1 binary removal', v1),
                     ('soft confidence', soft), ('v2 soft+dilate', v2),
                     ('Oracle (edited region known)', orc)]:
        g = mean(xs) - mean(uni)
        print(f'  {name:<26}{mean(xs):>9.1f}{sd(xs):>8.1f}'
              + (f'{g:>+10.1f}' if name != 'Uniform (no defense)' else f'{"---":>10}'))

    # 逐级增益 (消融表用)
    print('\n  逐级增益:')
    for a, b, nm in [(uni, v1, 'uniform -> v1'), (v1, soft, 'v1 -> soft'),
                     (soft, v2, 'soft -> v2')]:
        print(f'    {nm:<18} {(mean(b)-mean(a)):+.1f}pt')

    # ---------- Table 3: 分层 ----------
    print('\n### Table 3 (分层) — 计数三档合计必须 = n')
    strata = {'Small': [], 'Mid': [], 'Large': []}
    for r in pi:
        f = r.get('att_fail_frac')
        if f is None:
            continue
        key = 'Small' if f < TH_SMALL else ('Mid' if f <= TH_LARGE else 'Large')
        strata[key].append(r)

    total = 0
    print(f'{"档位":<34}{"n":>6}{"Uniform":>10}{"v2":>9}{"Oracle":>9}')
    for name, label in [('Small', f'Small (<15% blocks failed)'),
                        ('Mid', f'Mid (15-35%)'),
                        ('Large', f'Large (>35%)')]:
        rs = strata[name]
        total += len(rs)
        print(f'  {label:<32}{len(rs):>6}'
              f'{mean([r["uniform_ba"] for r in rs]):>10.1f}'
              f'{mean([r["signal_soft_dilate_ba"] for r in rs]):>9.1f}'
              f'{mean([r["bbox_ra_ba"] for r in rs]):>9.1f}')
    print(f'  {"合计":<32}{total:>6}')
    ok = (total == n)
    print(f'\n  ✓ 计数一致 (合计 {total} = n {n})' if ok
          else f'\n  ✗ 计数不一致 (合计 {total} != n {n}) — 有 {n-total} 张缺 att_fail_frac')

    # ---------- 校验失败率 ----------
    cf = get('clean_fail_frac')
    af = get('att_fail_frac')
    if cf and af:
        print(f'\n### 校验失败率')
        print(f'  clean {mean(cf):.1f}% (误报)   攻击后 {mean(af):.1f}%')

    # ---------- LaTeX ----------
    if args.latex:
        print('\n' + '=' * 72)
        print('LaTeX 行 (可直接替换)')
        print('=' * 72)
        sh, md, lg = strata['Small'], strata['Mid'], strata['Large']
        print('\n% --- Table 2 主体 ---')
        print(f'Uniform (no defense) & {mean(uni):.1f} & --- & {mean(cf):.1f}\\% \\\\')
        print(f'v1 (binary removal) & {mean(v1):.1f} & ${mean(v1)-mean(uni):+.1f}$ & --- \\\\')
        print(f'v2 soft confidence & {mean(soft):.1f} & ${mean(soft)-mean(uni):+.1f}$ & --- \\\\')
        print(f'\\textbf{{Signal-check v2 (soft+dilate)}} & \\textbf{{{mean(v2):.1f}}} & '
              f'\\textbf{{${mean(v2)-mean(uni):+.1f}$}} & --- \\\\')
        print(f'Oracle (edited region known) & {mean(orc):.1f} & ${mean(orc)-mean(uni):+.1f}$ & --- \\\\')

        print('\n% --- Table 3 分层 ---')
        print(f'Small ($<15$\\% blocks failed) & {len(sh)} & {mean([r["uniform_ba"] for r in sh]):.1f} & '
              f'\\textbf{{{mean([r["signal_soft_dilate_ba"] for r in sh]):.1f}}} & '
              f'{mean([r["bbox_ra_ba"] for r in sh]):.1f} \\\\')
        print(f'Mid ($15$--$35$\\%) & {len(md)} & {mean([r["uniform_ba"] for r in md]):.1f} & '
              f'\\textbf{{{mean([r["signal_soft_dilate_ba"] for r in md]):.1f}}} & '
              f'{mean([r["bbox_ra_ba"] for r in md]):.1f} \\\\')
        print(f'Large ($>35$\\%) & {len(lg)} & {mean([r["uniform_ba"] for r in lg]):.1f} & '
              f'\\textbf{{{mean([r["signal_soft_dilate_ba"] for r in lg]):.1f}}} & '
              f'{mean([r["bbox_ra_ba"] for r in lg]):.1f} \\\\')


if __name__ == '__main__':
    main()
