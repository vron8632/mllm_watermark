#!/bin/bash
# run_all.sh — 一键跑完三个补充实验
#   ① A2 消融 (uniform+RA, run_idea_probe 已内置 uniform_ra 方案)
#   ② OmniGuard 基线: 局部移除 + SD inpainting 两种攻击
#   ③ 64-bit 消息下的 SD inpainting 补充实验
# 用法:
#   bash run_all.sh            # 全量 (COCO 100 + Kodak 24; OmniGuard 100; SD inpaint 100)
#   bash run_all.sh --quick    # 小规模验证 (~10 分钟, 每实验 n≈20)
# 日志: results/run_all_<时间戳>.log (终端同步显示)

# 进入脚本所在目录 (code/), 兼容从项目根或 code 目录任意位置调用
cd "$(dirname "$0")" || exit 1

TS=$(date +%Y%m%d_%H%M%S)
LOG="../results/run_all_${TS}.log"
touch "$LOG"

log() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG"; }
sep() { log "────────────────────────────────────────────────────────"; }

QUICK=0
[ "$1" = "--quick" ] && QUICK=1
if [ "$QUICK" = "1" ]; then
  log "模式: QUICK (小规模, 每实验 n≈20)"
else
  log "模式: 全量"
fi

sep
log "=== ① A2 消融: run_idea_probe (含 uniform_ra, 5 方案 × removal 30/50%) ==="
if [ "$QUICK" = "1" ]; then
  python run_idea_probe.py --quick 2>&1 | tee -a "$LOG"
else
  python run_idea_probe.py 2>&1 | tee -a "$LOG"
fi
log "① 完成"

sep
log "=== ②a OmniGuard 基线: 局部实例移除 ==="
if [ "$QUICK" = "1" ]; then
  python run_baseline_omniguard.py --n 20 2>&1 | tee -a "$LOG"
else
  python run_baseline_omniguard.py --n 100 2>&1 | tee -a "$LOG"
fi
log "②a 完成"

sep
log "=== ②b OmniGuard 基线: SD inpainting 攻击 ==="
if [ "$QUICK" = "1" ]; then
  python run_baseline_omniguard.py --n 20 --inpaint --steps 10 2>&1 | tee -a "$LOG"
else
  python run_baseline_omniguard.py --n 100 --inpaint 2>&1 | tee -a "$LOG"
fi
log "②b 完成"

sep
log "=== ③ 64-bit 消息 × SD inpainting 补充实验 ==="
if [ "$QUICK" = "1" ]; then
  python run_inpaint_attack.py --n 20 --msglen 64 --steps 10 2>&1 | tee -a "$LOG"
else
  python run_inpaint_attack.py --n 100 --msglen 64 2>&1 | tee -a "$LOG"
fi
log "③ 完成"

sep
log "=== 全部完成, 本次新结果文件: ==="
ls -t ../results/idea_probe_*.json ../results/baseline_omniguard_*.json \
      ../results/inpaint_attack_*.json 2>/dev/null | head -8 | tee -a "$LOG"
log "完整日志: $LOG"
