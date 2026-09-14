#!/bin/bash
# run_mllm.sh — 一键运行 MLLM 驱动的语义级水印鲁棒性基准 (方案 A)
# 用法:
#   bash run_mllm.sh               # 全量 (COCO 100 张, instruct 模式)
#   bash run_mllm.sh --n 20        # 最小验证 (推荐先跑这个)
#   bash run_mllm.sh --n 20 --quick-eval   # 更快的 SD 推理 (steps=10, 质量略降)
# API 配置: 项目根 .mllm_env (mimo + deepseek), 已内置你的 key
cd "$(dirname "$0")" || exit 1

N="${2:-100}"
STEPS=30
[ "$1" = "--quick-eval" ] || [ "$3" = "--quick-eval" ] && STEPS=10

echo "[$(date +%H:%M:%S)] MLLM 水印鲁棒性基准 | n=${N} | SD steps=${STEPS}"
echo "[$(date +%H:%M:%S)] ① mimo 生成语义编辑指令 → ② SD inpainting 执行攻击 → ③ uniform/RA 评估"

python run_mllm_benchmark.py --n "$N" --mode instruct --steps "$STEPS" 2>&1

echo "[$(date +%H:%M:%S)] 完成. 结果: ../results/mllm_benchmark_*.json"
