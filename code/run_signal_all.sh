#!/bin/bash
# run_signal_all.sh — 一键跑完信号校验全套实验 (方案 B 可部署版 + P3 诊断 + 统计检验)
#
# 用法:
#   bash run_signal_all.sh              # 全量: COCO 50 + Kodak 24, 含诊断
#   bash run_signal_all.sh --quick      # 快速: COCO 10 + Kodak 10, steps=10, 含诊断
#
# 步骤:
#   ① signal_ra --n 50 --datasets coco   (COCO 主验证, 含 P3 诊断)
#   ② signal_ra --n 24 --datasets kodak  (Kodak 泛化, 含 P3 诊断)
#   ③ 统计检验 (signal_ra/oracle vs uniform 的 paired t-test + Wilcoxon + Cohen's d)
#
# 产物:
#   results/signal_ra_<ts>.json          (COCO, 含 clean/uniform/signal_ra/oracle/诊断)
#   results/signal_ra_<ts>.json          (Kodak)
#   results/signal_analysis_<ts>.json    (统计检验)
# 日志: results/signal_all_<时间戳>.log

cd "$(dirname "$0")" || exit 1

TS=$(date +%Y%m%d_%H%M%S)
LOG="../results/signal_all_${TS}.log"
touch "$LOG"
log() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG"; }

QUICK=0
[ "$1" = "--quick" ] && QUICK=1
if [ "$QUICK" = "1" ]; then
  COCO_N=10; KODAK_N=10; STEPS=10
  log "模式: QUICK (COCO $COCO_N + Kodak $KODAK_N, steps=$STEPS)"
else
  COCO_N=50; KODAK_N=24; STEPS=30
  log "模式: 全量 (COCO $COCO_N + Kodak $KODAK_N, steps=$STEPS)"
fi

sep() { log "────────────────────────────────────────────────────"; }

sep
log "=== ① signal_ra: COCO $COCO_N (含 P3 诊断) ==="
python run_signal_ra.py --n "$COCO_N" --datasets coco --steps "$STEPS" --diagnose 2>&1 | tee -a "$LOG"
log "① 完成"

sep
log "=== ② signal_ra: Kodak $KODAK_N (含 P3 诊断) ==="
python run_signal_ra.py --n "$KODAK_N" --datasets kodak --steps "$STEPS" --diagnose 2>&1 | tee -a "$LOG"
log "② 完成"

sep
log "=== ③ 统计检验 ==="
python3 - <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, glob, os, numpy as np
from scipy import stats
fs = sorted(glob.glob('../results/signal_ra_*.json'))
fs = [f for f in fs if not f.endswith('.tmp.json')]
if not fs:
    print('! 没有 signal_ra_*.json 结果')
else:
    for f in fs:
        R = json.load(open(f))
        rows = [r for r in R['per_image'] if 'error' not in r and 'uniform_ba' in r]
        if len(rows) < 3:
            print(f"! {f}: 有效样本不足 ({len(rows)})")
            continue
        u = np.array([r['uniform_ba'] for r in rows])
        s = np.array([r['signal_ra_ba'] for r in rows])
        o = np.array([r['bbox_ra_ba'] for r in rows])
        print(f"\n=== {os.path.basename(f)} (n={len(rows)}) ===")
        for v, name in [(s, 'signal_ra'), (o, 'oracle')]:
            t, p = stats.ttest_rel(v, u)
            w, pw = stats.wilcoxon(v, u)
            d = (v.mean()-u.mean())/np.sqrt(((len(u)-1)*u.std(ddof=1)**2+(len(v)-1)*v.std(ddof=1)**2)/(2*len(u)-2))
            print(f"  {name:<10} {u.mean()*100:5.1f}% -> {v.mean()*100:5.1f}%  "
                  f"(+{(v.mean()-u.mean())*100:+.1f}pt, p_t={p:.1e}, p_w={pw:.1e}, d={d:+.2f})")
        cf = np.mean([r['clean_fail_frac'] for r in rows])
        af = np.mean([r['att_fail_frac'] for r in rows])
        print(f"  校验失败率: clean={cf*100:.1f}% (误报)  攻击后={af*100:.1f}%")
PYEOF
log "③ 完成"

sep
log "=== 全部完成 ==="
ls -t ../results/signal_ra_*.json ../results/signal_all_*.log 2>/dev/null | head -5 | tee -a "$LOG"
log "完整日志: $LOG"
