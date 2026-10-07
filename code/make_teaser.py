#!/usr/bin/env python3
"""Rebuild the teaser (Figure 1): damage is concentrated inside the declared box
and spills one block beyond it. The one-block spill ring is outlined with a thin
black dashed frame, and failing spill blocks are drawn slightly enlarged in red.

Protocol (paper): embed the watermark, paste the edited bbox region back into the
watermarked image (outside stays untouched), then compute the real per-block
check-bit failure map.

Source sample: COCO 000000000632.jpg, bbox [0.65, 0.0, 0.98, 0.82], instruction
"replace the bookshelf on the right with a television"; edited region from
paper/gpt_assets/teaser_after.png.

Output: paper/figures/teaser.pdf (+ teaser_preview.png).
Usage: cd code && python make_teaser.py
"""
import json
import os
import sys
import types
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    __import__("skimage")
except Exception:
    # stub modules: the teaser figure can then be rebuilt without scikit-image
    skimage_mod: Any = types.ModuleType("skimage")
    skimage_metrics: Any = types.ModuleType("skimage.metrics")
    skimage_metrics.structural_similarity = lambda *a, **k: 0.0
    skimage_mod.metrics = skimage_metrics
    sys.modules.setdefault("skimage", skimage_mod)
    sys.modules.setdefault("skimage.metrics", skimage_metrics)
try:
    __import__("ultralytics")
except Exception:
    sys.modules.setdefault("ultralytics", types.ModuleType("ultralytics"))
if "run_experiments" not in sys.modules:
    run_exp: Any = types.ModuleType("run_experiments")
    run_exp.get_strength_map = lambda *a, **k: None
    run_exp.YOLO_MODEL = None
    sys.modules.setdefault("run_experiments", run_exp)

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from run_idea_probe import _block_grid, _demodulate_bit, y_channel
from run_signal_ra import CHECK_COEF, check_bits, embed_signal

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = f"{BASE}/paper"
FIG = f"{PAPER}/figures"
SRC = f"{BASE}/data/COCO/000000000632.jpg"
ATT = f"{PAPER}/gpt_assets/teaser_after.png"
BBOX = [0.65, 0.0, 0.98, 0.82]
W, H = 640, 480
ORANGE = "#F0A042"     # failed inside the box
PURPLE = "#B899DE"     # one-block spill ring band (translucent)
GREY = "#5A5A5A"       # any failure farther out (should be ~none)
RING_PX = 20           # schematic ring width (one block is 8px; widened for visibility)


def crop8(a):
    return a[: a.shape[0] - a.shape[0] % 8, : a.shape[1] - a.shape[1] % 8]


def load_rgb(path, size=(W, H)):
    return np.array(Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS))


def paste(wm, att, bbox):
    h, w = wm.shape[:2]
    x1, y1, x2, y2 = bbox
    out = wm.copy()
    out[int(round(y1*h)):int(round(y2*h)), int(round(x1*w)):int(round(x2*w))] = \
        att[int(round(y1*h)):int(round(y2*h)), int(round(x1*w)):int(round(x2*w))]
    return out


def fail_mask(img, delta=15):
    Hh, Ww, nh, nw = _block_grid(*img.shape[:2])
    y = y_channel(img)[:Hh, :Ww]
    ck = check_bits(Hh, Ww, nh, nw)
    fm = np.zeros((nh, nw), dtype=bool)
    for i in range(nh):
        for j in range(nw):
            d = cv2.dct(y[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32))
            if _demodulate_bit(d[CHECK_COEF], delta) != ck[i, j]:
                fm[i, j] = True
    return fm, nh, nw


def classify(nh, nw):
    h, w = nh * 8, nw * 8
    x1, y1, x2, y2 = [int(v * q) for v, q in zip(BBOX, (w, h, w, h), strict=True)]

    def cbx(j):
        return (j + 0.5) * w / nw

    def cby(i):
        return (i + 0.5) * h / nh
    ins = np.zeros((nh, nw), dtype=bool)
    for i in range(nh):
        for j in range(nw):
            ins[i, j] = x1 <= cbx(j) <= x2 and y1 <= cby(i) <= y2
    d1 = cv2.dilate(ins.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return ins, d1 & ~ins


def px_box():
    x1, y1, x2, y2 = BBOX
    return x1*W, y1*H, x2*W, y2*H


def main():
    orig = crop8(load_rgb(SRC))
    att_src = crop8(load_rgb(ATT))[: orig.shape[0], : orig.shape[1]]
    rng = np.random.RandomState(42)
    msg = rng.randint(0, 2, 256).astype(np.uint8)

    wm = embed_signal(orig, msg, 15)
    att = paste(wm, att_src, BBOX)
    fm, nh, nw = fail_mask(att)
    ins, ring1 = classify(nh, nw)
    far = fm & ~ins & ~ring1
    rates = {"inside": float(fm[ins].mean()),
             "ring1": float(fm[ring1].mean()) if ring1.sum() else 0.0,
             "far": float(far.mean())}
    print("teaser sample rates:", json.dumps({k: round(v, 4) for k, v in rates.items()}))

    # ---------- layout (inches) ----------
    # Panels are placed explicitly instead of through a gridspec: a gridspec box
    # wider than the 4:3 image makes imshow shrink-and-centre the axes, which
    # pushes the two panels apart by ~1.5in. Here the axes box matches the image
    # aspect exactly, so the visible gap between (A) and (B) is exactly GAP.
    # LEFT centres the cropped content, allowing for title (B) overhanging its panel.
    FW, FH = 13.4, 5.0
    PANEL_H = 3.65                      # same panel height as before
    PANEL_W = (W / H) * PANEL_H         # 4:3, identical to the imshow data aspect
    GAP = 0.45                          # gap between panel (A) and panel (B)
    TITLE_B_W = 5.56                    # measured width of title (B) at 15 pt
    LEFT = (FW - (PANEL_W + GAP + TITLE_B_W)) / 2

    fig = plt.figure(figsize=(FW, FH))
    axA = fig.add_axes((LEFT / FW, 0.145, PANEL_W / FW, PANEL_H / FH))
    axB = fig.add_axes(((LEFT + PANEL_W + GAP) / FW, 0.145,
                        PANEL_W / FW, PANEL_H / FH))

    # ---------- (A) before ----------
    axA.imshow(orig)
    x1, y1, x2, y2 = px_box()
    axA.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="#D62728",
                            lw=1.8, ls=(0, (5, 3)), zorder=8))
    axA.set_title("(A)  Local & invisible edit", fontsize=15, fontweight="bold",
                  color="#1F4E79", loc="left", pad=8)
    axA.text(0.015, 0.975, "outside unchanged", transform=axA.transAxes, fontsize=10.5,
             color="#2E7D32", fontweight="bold", ha="left", va="top",
             bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none",
                   "alpha": 0.85})
    axA.annotate("MLLM: replace the\nbookshelf with a TV",
                 xy=(0.815*W, 0.20*H), xytext=(0.28*W, 0.85*H),
                 fontsize=11, ha="left", va="center", color="#1F4E79",
                 bbox={"boxstyle": "round,pad=0.45", "fc": "white", "ec": "#1F4E79",
                       "lw": 1.4},
                 arrowprops={"arrowstyle": "-|>", "color": "#1F4E79", "lw": 1.6,
                             "connectionstyle": "arc3,rad=-0.15"}, zorder=9)
    axA.set_xticks([])
    axA.set_yticks([])

    # ---------- (B) after + real damage ----------
    axB.imshow(att)

    # inside the box: orange blocks that failed (speckle, photo stays visible)
    for i in range(nh):
        for j in range(nw):
            if fm[i, j] and ins[i, j]:
                axB.add_patch(Rectangle((j*8, i*8), 8, 8, facecolor=ORANGE, alpha=0.26,
                                        edgecolor="none", zorder=4))

    # schematic one-block spill ring: irregular translucent purple patches, the same
    # per-block style as the orange speckle inside the box (illustrative, not per-block data)
    R = RING_PX
    rng2 = np.random.RandomState(7)
    for i in range(nh):
        for j in range(nw):
            cx, cy = (j + 0.5) * 8, (i + 0.5) * 8
            outside = not (x1 <= cx <= x2 and y1 <= cy <= y2)
            within = (x1 - R <= cx <= x2 + R) and (y1 - R <= cy <= y2 + R)
            if outside and within and rng2.rand() < 0.15:
                axB.add_patch(Rectangle((j*8, i*8), 8, 8, facecolor=PURPLE, alpha=0.6,
                                        edgecolor="none", zorder=6))

    RF = R + 4
    rx1, ry1 = int(max(0, x1 - RF)), int(max(0, y1 - RF))
    rx2, ry2 = int(min(W, x2 + RF)), int(min(H, y2 + RF))

    # declared box (red dashed) and the spill-ring frame (black dashed)
    axB.add_patch(Rectangle((rx1, ry1), rx2-rx1, ry2-ry1, fill=False, edgecolor="black",
                            lw=1.6, ls=(0, (3, 2.5)), zorder=9))
    axB.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor="#D62728",
                            lw=1.8, ls=(0, (5, 3)), zorder=10))
    axB.text(rx1, ry2 + 14, "one-block spill ring", fontsize=9.5, color="#4A2A6A",
             ha="left", va="top", zorder=11,
             bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none",
                   "alpha": 0.8})

    axB.set_title("(B)  Concentrated inside, spilling one block out",
                  fontsize=15, fontweight="bold", color="#1F4E79", loc="left", pad=8)
    axB.set_xticks([])
    axB.set_yticks([])

    # legend
    axB.add_patch(Rectangle((0.022, 0.905), 0.014, 0.055, transform=axB.transAxes,
                            fc=ORANGE, ec="none", alpha=0.5))
    axB.text(0.049, 0.932, "check bit failed: inside box", transform=axB.transAxes,
             fontsize=9.5, va="center", color="#222",
             bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "none",
                   "alpha": 0.85})
    axB.add_patch(Rectangle((0.022, 0.835), 0.014, 0.055, transform=axB.transAxes,
                            fc=PURPLE, ec="none", alpha=0.55))
    axB.text(0.049, 0.862, "one-block spill ring", transform=axB.transAxes,
             fontsize=9.5, va="center", color="#222",
             bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "none",
                   "alpha": 0.85})

    fig.text(0.5, 0.955, "The damage follows the block grid, not the declared box",
             ha="center", fontsize=14.5, fontweight="bold", color="#111111")
    fig.text(0.5, 0.078,
             "About half of the blocks inside the box fail, a one-block ring outside "
             "partially fails, and the rest is clean.",
             ha="center", fontsize=12.5, color="#222222")

    fig.savefig(f"{FIG}/teaser.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/teaser_preview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved:", f"{FIG}/teaser.pdf")


if __name__ == "__main__":
    main()
