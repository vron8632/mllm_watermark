#!/usr/bin/env python3
"""补充材料图表生成.

Fig S1: 逐图 BA 增益分布 (v2 - uniform, box + scatter)
Fig S2: Per-image uniform vs v2 散点 (对角线 = 无改进)
Fig S3: 编辑强度 (check-failure rate) vs BA 增益散点
Fig S4: 逐图各方案 BA 对比 (分组散点)
Table S1: Kodak 消融
Table S2: Bootstrap 95% CI for main results
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = f'{BASE}/paper/figures'
os.makedirs(FIG_DIR, exist_ok=True)


def load_coco():
    d = json.load(open(f'{BASE}/results/signal_ra_20260817_071246.json'))
    pi = d['per_image']
    return {
        'uniform': np.array([im['uniform_ba'] for im in pi]),
        'v1': np.array([im['signal_ra_ba'] for im in pi]),
        'soft': np.array([im['signal_soft_ba'] for im in pi]),
        'v2': np.array([im['signal_soft_dilate_ba'] for im in pi]),
        'oracle': np.array([im['bbox_ra_ba'] for im in pi]),
        'fail': np.array([im['att_fail_frac'] for im in pi]),
        'n': len(pi),
    }


def bootstrap_ci(data, n_boot=10000, ci=0.95):
    """Bootstrap 95% CI for the mean."""
    rng = np.random.RandomState(42)
    means = np.array([np.mean(rng.choice(data, size=len(data), replace=True))
                      for _ in range(n_boot)])
    lo = np.percentile(means, (1-ci)/2 * 100)
    hi = np.percentile(means, (1+ci)/2 * 100)
    return np.mean(data), lo, hi


# ── Fig S1: 逐图 BA 增益分布 ──
def fig_s1_improvement_distribution(data):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # (a) v2 - uniform 增益分布
    gain = (data['v2'] - data['uniform']) * 100
    ax = axes[0]
    bp = ax.boxplot([gain], widths=0.5, patch_artist=True,
                    boxprops=dict(facecolor='#eaf3fb', edgecolor='#2c6fbb'),
                    medianprops=dict(color='#b03030', lw=2))
    jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(gain))
    ax.scatter(np.ones(len(gain)) + jitter, gain, alpha=0.5, s=20, c='#2c6fbb', zorder=3)
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_ylabel('BA improvement (v2 $-$ uniform, pt)')
    ax.set_xticks([1])
    ax.set_xticklabels(['COCO ($n=50$)'])
    ax.set_title('(a) Per-image improvement distribution', fontsize=10)
    ax.text(1.2, gain.max()*0.9, f'median={np.median(gain):+.1f}pt\n'
            f'mean={np.mean(gain):+.1f}pt\n'
            f'positive: {(gain>0).mean()*100:.0f}%',
            fontsize=8.5, va='top')
    
    # (b) 各方案 per-image BA
    ax = axes[1]
    labels = ['uniform', 'v1', 'soft', 'v2', 'oracle']
    keys = ['uniform', 'v1', 'soft', 'v2', 'oracle']
    colors = ['#999', '#6baed6', '#4292c6', '#2171b5', '#c77d1e']
    for i, (k, c) in enumerate(zip(keys, colors)):
        vals = data[k] * 100
        jitter = np.random.RandomState(42).uniform(-0.2, 0.2, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, alpha=0.4, s=15, c=c, zorder=3)
        ax.errorbar(i, np.mean(vals), yerr=np.std(vals), fmt='D', color=c,
                    markersize=6, zorder=4, capsize=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Bit accuracy (%)')
    ax.set_title('(b) Per-image BA by defense variant', fontsize=10)
    ax.axhline(95, color='gray', ls=':', lw=0.8, label='95% threshold')
    ax.legend(fontsize=8)
    
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/fig_s1_distribution.pdf', bbox_inches='tight')
    fig.savefig(f'{FIG_DIR}/fig_s1_distribution.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('fig_s1 done')


# ── Fig S2: uniform vs v2 散点 ──
def fig_s2_scatter(data):
    fig, ax = plt.subplots(figsize=(5, 5))
    u, v = data['uniform'] * 100, data['v2'] * 100
    ax.scatter(u, v, alpha=0.6, s=30, c='#2171b5', edgecolors='white', linewidth=0.5)
    # 对角线
    lim = [min(u.min(), v.min()) - 2, 102]
    ax.plot(lim, lim, 'k--', lw=0.8, alpha=0.5, label='y = x (no improvement)')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Uniform BA (%)')
    ax.set_ylabel('Signal-check v2 BA (%)')
    ax.set_title('Per-image: uniform vs v2', fontsize=10)
    ax.legend(fontsize=8)
    # 统计
    above = (v > u).sum()
    ax.text(0.05, 0.92, f'v2 > uniform: {above}/{len(u)} ({above/len(u)*100:.0f}%)\n'
            f'mean gain: {np.mean(v-u):+.1f}pt',
            transform=ax.transAxes, fontsize=8.5, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/fig_s2_scatter.pdf', bbox_inches='tight')
    fig.savefig(f'{FIG_DIR}/fig_s2_scatter.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('fig_s2 done')


# ── Fig S3: 编辑强度 vs 增益 ──
def fig_s3_intensity(data):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    fail = data['fail'] * 100
    gain = (data['v2'] - data['uniform']) * 100
    
    # (a) 散点
    ax = axes[0]
    ax.scatter(fail, gain, alpha=0.6, s=25, c='#2171b5', edgecolors='white', linewidth=0.5)
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('Check-failure rate (%)')
    ax.set_ylabel('BA improvement: v2 $-$ uniform (pt)')
    ax.set_title('(a) Editing intensity vs. defense gain', fontsize=10)
    
    # 分箱趋势线
    bins = [(0, 15, 'Small'), (15, 35, 'Mid'), (35, 100, 'Large')]
    for lo, hi, lbl in bins:
        mask = (fail >= lo) & (fail < hi)
        if mask.sum() > 0:
            ax.axvspan(lo, min(hi, fail.max()+1), alpha=0.08, color='gray')
            ax.text(np.clip((lo+min(hi, fail.max()))/2, lo+1, hi-1), gain.max()*0.95,
                    f'{lbl}\nn={mask.sum()}', ha='center', fontsize=7.5, color='#666')
    
    # (b) 分箱柱状图
    ax = axes[1]
    labels, u_means, v2_means, orc_means = [], [], [], []
    for lo, hi, lbl in bins:
        mask = (fail >= lo) & (fail < hi)
        if mask.sum() > 0:
            labels.append(f'{lbl}\n($n={mask.sum()}$)')
            u_means.append(data['uniform'][mask].mean() * 100)
            v2_means.append(data['v2'][mask].mean() * 100)
            orc_means.append(data['oracle'][mask].mean() * 100)
    
    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, u_means, w, label='Uniform', color='#ccc', edgecolor='white')
    ax.bar(x, v2_means, w, label='v2', color='#2171b5', edgecolor='white')
    ax.bar(x + w, orc_means, w, label='Oracle', color='#c77d1e', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Mean BA (%)')
    ax.set_title('(b) Stratified mean BA', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(50, 105)
    
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/fig_s3_intensity.pdf', bbox_inches='tight')
    fig.savefig(f'{FIG_DIR}/fig_s3_intensity.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('fig_s3 done')


# ── Table S1: Kodak 消融 ──
def table_s1_kodak_ablation():
    d = json.load(open(f'{BASE}/results/signal_ra_20260817_073707.json'))
    pi = d['per_image']
    u = np.array([im['uniform_ba'] for im in pi]) * 100
    v1 = np.array([im['signal_ra_ba'] for im in pi]) * 100
    soft = np.array([im['signal_soft_ba'] for im in pi]) * 100
    v2 = np.array([im['signal_soft_dilate_ba'] for im in pi]) * 100
    orc = np.array([im['bbox_ra_ba'] for im in pi]) * 100
    
    thresh = lambda x: (x >= 95).mean() * 100
    print("\nTable S1: Kodak ablation (n=24)")
    print(f"{'Configuration':<30s} {'BA':>6s} {'ΔBA':>6s} {'≥95%':>6s}")
    print("-" * 50)
    base = u.mean()
    for name, vals in [('Uniform (baseline)', u), ('+ check bit, binary (v1)', v1),
                       ('+ soft confidence', soft), ('+ spatial dilation (v2)', v2),
                       ('Oracle (bbox known)', orc)]:
        mean = vals.mean()
        delta = mean - base
        thr = thresh(vals)
        print(f"{name:<30s} {mean:5.1f}% {delta:+5.1f} {thr:5.0f}%")


# ── Table S2: Bootstrap CIs ──
def table_s2_confidence_intervals():
    coco = json.load(open(f'{BASE}/results/signal_ra_20260817_071246.json'))
    kodak = json.load(open(f'{BASE}/results/signal_ra_20260817_073707.json'))
    
    print("\nTable S2: Bootstrap 95% CI for mean BA")
    print(f"{'Dataset':<12s} {'Method':<12s} {'Mean':>6s} {'95% CI':>16s}")
    print("-" * 50)
    
    for name, res in [('COCO', coco), ('Kodak', kodak)]:
        pi = res['per_image']
        for label, key in [('Uniform', 'uniform_ba'), ('v1', 'signal_ra_ba'),
                           ('soft', 'signal_soft_ba'), ('v2', 'signal_soft_dilate_ba'),
                           ('Oracle', 'bbox_ra_ba')]:
            vals = np.array([im[key] for im in pi]) * 100
            mean, lo, hi = bootstrap_ci(vals)
            print(f"{name:<12s} {label:<12s} {mean:5.1f}% [{lo:.1f}, {hi:.1f}]")


if __name__ == '__main__':
    data = load_coco()
    fig_s1_improvement_distribution(data)
    fig_s2_scatter(data)
    fig_s3_intensity(data)
    table_s1_kodak_ablation()
    table_s2_confidence_intervals()
    print('\n所有补充材料图表已生成')


# ---------------------------------------------------------------- S4
def fig_s4_delta_defense():
    """补充图 S4: (a) 不同嵌入步长 Δ 下 uniform vs v2 的 BA;
    (b) 保真度 (PSNR) 随 Δ 变化.

    依据 supplementary.tex 的 caption 重建 (原文件只有 png, 生成脚本已丢失).
    数据: results/delta_sensitivity_20260817_182952.json
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = json.load(open(f'{BASE}/results/delta_sensitivity_20260817_182952.json'))
    rows = d['per_image']
    deltas = sorted({r['delta'] for r in rows})

    def mean_at(key, dl):
        v = [r[key] for r in rows if r['delta'] == dl]
        return 100.0 * float(np.mean(v)), 100.0 * float(np.std(v))

    u_m = [mean_at('uniform', dl)[0] for dl in deltas]
    u_s = [mean_at('uniform', dl)[1] for dl in deltas]
    v_m = [mean_at('v2', dl)[0] for dl in deltas]
    v_s = [mean_at('v2', dl)[1] for dl in deltas]
    p_m = [float(np.mean([r['psnr'] for r in rows if r['delta'] == dl])) for dl in deltas]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

    ax = axes[0]
    ax.errorbar(deltas, u_m, yerr=u_s, marker='o', lw=2, capsize=4,
                color='#7f7f7f', label='uniform (no defense)')
    ax.errorbar(deltas, v_m, yerr=v_s, marker='s', lw=2, capsize=4,
                color='#c0392b', label='v2 (soft + dilate)')
    ax.set_xlabel(r'Embedding step size $\Delta$', fontsize=11)
    ax.set_ylabel('Bit accuracy (%)', fontsize=11)
    ax.set_xticks(deltas)
    ax.grid(alpha=0.3, ls='--')
    ax.legend(fontsize=9.5, frameon=False)
    ax.set_title('(a) defense performance vs $\\Delta$', fontsize=11.5, loc='left')

    ax = axes[1]
    ax.plot(deltas, p_m, marker='D', lw=2, color='#2c6fbb')
    for x, y in zip(deltas, p_m):
        ax.annotate(f'{y:.1f}', (x, y), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=9)
    ax.set_xlabel(r'Embedding step size $\Delta$', fontsize=11)
    ax.set_ylabel('PSNR (dB)', fontsize=11)
    ax.set_xticks(deltas)
    ax.grid(alpha=0.3, ls='--')
    ax.set_title('(b) fidelity vs $\\Delta$', fontsize=11.5, loc='left')

    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/fig_s4_delta.pdf', bbox_inches='tight')
    fig.savefig(f'{FIG_DIR}/fig_s4_delta.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  fig_s4_delta done (rebuilt from delta_sensitivity json)')


if __name__ == '__main__':
    fig_s4_delta_defense()
