#!/usr/bin/env python3
"""鲁棒性-不可见性平面图 (PSNR vs attacked BA, marker 面积 = 载荷容量).

数据源（均为已有结果，不跑任何模型）:
  results/fidelity_ssim_lpips_600.json   -> 每方法 PSNR (n=600)
  results/fidelity_psnr_summary.json     -> 每方法 pooled clean/attacked BA 与 bits
输出:
  paper/figures/fig_fidelity_plane.pdf / .png
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

WS = "/media/oyp/\u5de5\u4f5c/Projects/042_image_forensic/watermark-mllm-robustness"
FID = os.path.join(WS, "results/fidelity_ssim_lpips_600.json")
SUM = os.path.join(WS, "results/fidelity_psnr_summary.json")
OUT_PDF = os.path.join(WS, "paper/figures/fig_fidelity_plane.pdf")
OUT_PNG = os.path.join(WS, "paper/figures/fig_fidelity_plane.png")

def _load_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"failed to load {path}: {exc}") from exc


fid = _load_json(FID)
summ = _load_json(SUM)

# 方法键 -> 显示名 / 载荷位数 / 是否本文方法 / 是否训练型
META = {
    "dwt_dct":             ("DWT-DCT",           256, False, False),
    "signal_check_v2":     ("Ours (Signal-Check)", 256, True,  False),
    "trustmark":           ("TrustMark",         100, False, True),
    "watermark_anything":  ("Watermark Anything",  32, False, True),
    "robust_wide":         ("Robust-Wide",        64, False, True),
    "stable_signature":    ("Stable Signature",   48, False, True),
}

rows = []
for key in META:
    pm = fid["per_method"][key]
    sm = summ["per_method"][key]
    name, bits, is_ours, trained = META[key]
    if sm["capacity_bits"] != bits:
        raise ValueError(f"{key}: capacity {sm['capacity_bits']} != metadata {bits}")
    rows.append({
        "key": key, "name": name, "bits": bits, "ours": is_ours,
        "trained": trained, "psnr": pm["psnr_mean"], "ssim": pm["ssim_mean"],
        "lpips": pm["lpips_mean"], "ba": sm["attacked_ba_mean_pct"],
    })

for r in sorted(rows, key=lambda x: -x["psnr"]):
    print(f"{r['name']:22s} PSNR {r['psnr']:6.2f}  SSIM {r['ssim']:.4f}  "
          f"LPIPS {r['lpips']:.4f}  BA {r['ba']:5.2f}  bits {r['bits']:3d}  "
          f"trained={r['trained']}")

fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=200)

# marker 面积随载荷位数单调增大，但**不**做完全正比：用幂律 (bits/256)**0.85，
# 把 256 / 100 / 64 / 48 / 32 位几档明显拉开，同时让 256 位保持适中（直径 ~14pt）。
SIZE_REF_BITS = 256.0
SIZE_MAX = 200.0
SIZE_POW = 0.85


def marker_area(bits):
    """载荷位数 -> scatter 面积 (points^2)；单射但非严格正比。"""
    return SIZE_MAX * (bits / SIZE_REF_BITS) ** SIZE_POW


for r in rows:
    # 相同载荷（Ours 与 DWT-DCT，均 256 bit）面积严格一致，靠红实心 + 加粗标签区分。
    # 注意：Stable Signature 的方块只是被缩小，其坐标仍是实测值，不做移动。
    size = marker_area(r["bits"])
    if r["ours"]:
        ax.scatter(r["psnr"], r["ba"], s=size, marker="o", zorder=5,
                   facecolor="#C62828", edgecolor="black", linewidth=1.0, alpha=0.95)
    elif r["trained"]:
        # 浅灰实心（而非空心）：面积差才看得清
        ax.scatter(r["psnr"], r["ba"], s=size, marker="s", zorder=4,
                   facecolor="#E0E0E0", edgecolor="#424242", linewidth=1.1)
    else:
        ax.scatter(r["psnr"], r["ba"], s=size, marker="^", zorder=4,
                   facecolor="#BBDEFB", edgecolor="#1565C0", linewidth=1.1)

# 标注（手工微调偏移，避免重叠）
# 标注偏移随标记缩小同步收紧，避免标签离点太远
OFF = {
    "signal_check_v2":    (0,  2.6, "center"),
    "dwt_dct":            (0, -2.5, "center"),
    "trustmark":          (0, -2.3, "center"),
    "watermark_anything": (0,  2.0, "center"),
    "robust_wide":        (0, -2.7, "center"),
    "stable_signature":   (0,  2.0, "center"),
}
for r in rows:
    dx, dy, ha = OFF[r["key"]]
    weight = "bold" if r["ours"] else "normal"
    ax.annotate(r["name"], (r["psnr"] + dx, r["ba"] + dy), ha=ha, va="center",
                fontsize=8.2, fontweight=weight,
                color="#C62828" if r["ours"] else "#212121")

ax.set_xlabel("Embedding fidelity: PSNR vs. original (dB)  $\\rightarrow$  less distortion",
              fontsize=9.5)
ax.set_ylabel("Robustness: attacked bit accuracy (%)", fontsize=9.5)
ax.set_xlim(14, 53)
ax.set_ylim(70, 102)
ax.grid(True, linestyle=":", linewidth=0.6, color="#BDBDBD", zorder=0)
ax.tick_params(labelsize=8.5)

# 图例（用散点代理）：不再单列 "marker area" 黑圈，面积随载荷递增由 caption + 尺寸图例说明
handles = [
    Line2D([], [], marker="o", color="none", markerfacecolor="#C62828",
           markeredgecolor="black", markersize=7.0, label="Ours (train-free)"),
    Line2D([], [], marker="^", color="none", markerfacecolor="#BBDEFB",
           markeredgecolor="#1565C0", markersize=6.5, label="Train-free baseline"),
    Line2D([], [], marker="s", color="none", markerfacecolor="#E0E0E0",
           markeredgecolor="#424242", markersize=6.0, label="Trained baseline"),
]
# 图例压缩成单行 (ncol=3) 并置于左下角最下方：整条图例高度大幅降低，
# 其顶边远低于 Stable Signature 方块的下沿，彻底避免左下角图例遮挡数据点。
leg_main = ax.legend(handles=handles, loc="lower left", fontsize=6.5, frameon=True,
                     framealpha=0.95, edgecolor="#BDBDBD", ncol=3,
                     labelspacing=0.2, columnspacing=0.9, borderpad=0.3,
                     handletextpad=0.4, handlelength=1.3, borderaxespad=0.2)
ax.add_artist(leg_main)

# 尺寸参考：在左上角空白区直接画出 "载荷位数 -> 标记大小" 的对照（手绘，避免图例
# handle 被大标记撑破）。round markers 的 s 以 points^2 计，与真实数据点同一标度。
REF_X = 16.0
REF_Y = {256: 97.6, 100: 93.0, 64: 88.8, 32: 85.0}
ax.annotate("Payload capacity", (REF_X - 1.0, 101.0), ha="left", va="center",
            fontsize=6.2, color="#616161")
for _bits, _y in REF_Y.items():
    ax.scatter([REF_X], [_y], s=marker_area(_bits), marker="o", zorder=3,
               facecolor="#E0E0E0", edgecolor="#424242", linewidth=0.9)
    ax.annotate(f"{_bits}-bit", (REF_X + 2.8, _y), ha="left", va="center",
                fontsize=6.2, color="#616161")

ax.set_title("Robustness vs. imperceptibility of the six compared methods (600 images)",
             fontsize=10, pad=9)

fig.tight_layout()
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG)
print("written:", OUT_PDF)
print("written:", OUT_PNG)
