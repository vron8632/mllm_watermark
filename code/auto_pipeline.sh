#!/usr/bin/env bash
# 自动串联流水线: 等 n=500 完成 -> 归档 -> 重算表格 -> 回编译 -> 跑第二 MLLM
#
# 用途: 无人值守时自动完成 n=500 之后的所有步骤.
# 用法: bash code/auto_pipeline.sh
# 日志: /tmp/auto_pipeline.log
#
# 安全设计:
#   - 等待阶段不占用 GPU (只轮询进程)
#   - 每步失败即停止, 不继续后续步骤
#   - 原始结果先备份到 /tmp
#   - 不自动改论文正文; LaTeX 行生成到 /tmp 供人工确认
#   - 回编译用 /tmp 副本, 不触碰工作区交付物

set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
LOG=/tmp/auto_pipeline.log
PY=/home/oyp/miniconda3/envs/apjf/bin/python
export PATH=/tmp/.TinyTeX/bin/x86_64-linux:$PATH
unset ALL_PROXY all_proxy http_proxy https_proxy 2>/dev/null || true

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail() { log "失败: $*"; exit 1; }

echo "=== auto_pipeline 启动 $(date '+%Y-%m-%d %H:%M:%S') ===" > "$LOG"

# ---------- 阶段 0: 等待 n=500 完成 ----------
log "阶段 0: 等待 n=500 主实验完成..."
while pgrep -f "run_signal_ra.py --n 500" > /dev/null; do
    cur=$(grep -o '\[[0-9]*/500\]' /tmp/n500_run.log 2>/dev/null | tail -1)
    log "  仍在运行 $cur"
    sleep 300
done
log "n=500 进程已结束"
sleep 10

FINAL=$(ls -t "$ROOT"/results/signal_ra_2026091[0-9]_*.json 2>/dev/null \
        | grep -v '\.tmp\.json' | head -1)
[ -n "$FINAL" ] || fail "找不到 n=500 最终结果文件"
log "最终结果: $(basename "$FINAL")"

N=$($PY -c "import json;print(len(json.load(open('$FINAL'))['per_image']))" 2>/dev/null)
log "样本数 = $N"
[ "$N" -ge 450 ] || fail "样本数仅 $N, 远少于 500, 请人工检查"
log "阶段 0 完成: n=$N"

# ---------- 阶段 1: 归档 ----------
log "阶段 1: 归档 (自描述命名)..."
cp "$FINAL" /tmp/backup_n500_raw.json
$PY "$ROOT/code/archive_results.py" --pattern 'signal_ra_2026091*.json' >> "$LOG" 2>&1 \
    || log "  (归档脚本返回非零, 见日志)"
ARCHIVED=$(ls -t "$ROOT"/results/signal_ra_coco_n*.json 2>/dev/null | head -1)
[ -n "$ARCHIVED" ] || ARCHIVED="$FINAL"
log "阶段 1 完成: $(basename "$ARCHIVED")"

# ---------- 阶段 2: 重算 Table 2/3/5 ----------
log "阶段 2: 从 n=$N 重算 Table 2/3/5..."
$PY "$ROOT/code/recompute_n500.py" "$ARCHIVED" --latex > /tmp/recompute_n500.txt 2>&1 \
    || fail "重算脚本出错, 见 /tmp/recompute_n500.txt"
grep -q "计数一致" /tmp/recompute_n500.txt && log "  分层计数一致" \
    || log "  警告: 分层计数不一致, 需人工检查"
log "阶段 2 完成 (LaTeX 行在 /tmp/recompute_n500.txt)"

# ---------- 阶段 3: 回编译验证 ----------
log "阶段 3: 回编译验证 (在 /tmp 副本中进行)..."
rm -rf /tmp/vfy_auto && mkdir -p /tmp/vfy_auto
cp -r "$ROOT"/paper/. /tmp/vfy_auto/ 2>/dev/null
cd /tmp/vfy_auto || fail "无法进入编译目录"
latex -interaction=nonstopmode elsarticle.ins > /dev/null 2>&1
for i in 1 2; do timeout 300 pdflatex -interaction=nonstopmode main.tex > "m$i.txt" 2>&1; done
bibtex main > mb.txt 2>&1
for i in 3 4; do timeout 300 pdflatex -interaction=nonstopmode main.tex > "m$i.txt" 2>&1; done
# 注意: grep -c 无匹配时返回码为 1, 直接 || echo 0 会拼出多行值导致整数比较失败.
# 用 `grep -c ... || true` 保留其输出 (0), 不用 echo 兜底.
ERR=$(grep -c '^! ' m4.txt 2>/dev/null || true)
UND=$(grep -c undefined m4.txt 2>/dev/null || true)
OVF=$(grep -c Overfull m4.txt 2>/dev/null || true)
ERR=${ERR:-0}; UND=${UND:-0}; OVF=${OVF:-0}
PGS=$(pdfinfo main.pdf 2>/dev/null | grep -oP 'Pages:\s+\K\d+' || true)
PGS=${PGS:-?}
log "  main: 报错=$ERR undefined=$UND Overfull=$OVF 页数=$PGS"
[ "$ERR" -eq 0 ] || fail "编译有致命错误, 见 /tmp/vfy_auto/m4.txt"
log "阶段 3 完成"

# ---------- 阶段 4: 第二 MLLM (DeepSeek, 100 张) ----------
# 等待 GPU 显存释放, 避免与前一个进程的清理竞争
log "等待 GPU 显存释放..."
for _ in $(seq 1 30); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$USED" ] && [ "$USED" -lt 3000 ] && break
    log "  显存仍占用 ${USED:-?} MiB, 等待..."
    sleep 20
done
log "阶段 4: 第二 MLLM 攻击者 (dsvision, 100 张)..."
cd "$ROOT/code" || fail "无法进入 code/"
$PY -u run_second_mllm_attacker.py --provider dsvision --n 100 > /tmp/second_mllm_run.log 2>&1
RC=$?
if [ $RC -eq 0 ]; then
    log "阶段 4 完成"
    grep -A4 "汇总" /tmp/second_mllm_run.log | tail -5 | tee -a "$LOG"
else
    log "阶段 4 返回 $RC, 见 /tmp/second_mllm_run.log"
fi

log "================ 全部阶段结束 ================"
log "待人工确认:"
log "  1. /tmp/recompute_n500.txt 的 LaTeX 行 -> 替换 paper/main.tex 的 Table 2/3/5"
log "  2. dsvision 结果 -> 写入论文的攻击者泛化性小节"
