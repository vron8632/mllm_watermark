#!/usr/bin/env python3
"""导出给 GPT-image-2 当底图的**真实素材** (GPT 自己拿不到的那些).

GPT 联网只能找到"编辑前"的 COCO 原图; "编辑后"图是本项目 SD inpainting 的输出,
"真实 check-bit 失败斑"需要水印密钥 + 逐块 DCT 解调 —— 这两样 GPT 无论如何画不出来,
所以这里导出成文件, 直接丢给 GPT 当底图/参考.

输出目录: paper/figures/gpt_assets/ (示例样本由 demo_example.py 配置)
  teaser_after.png              编辑后 (本项目 SD inpainting 输出, GPT 无法获得)
  teaser_after_damage.png       编辑后 + 真实 check-bit 失败斑(橙), 右面板可直接用
  teaser_damage_only.png        真实失败斑, 透明背景 RGBA (让 GPT 自己叠)
  teaser_before_bbox.png        编辑前 + 红虚线编辑框
  teaser_after_damage_bbox.png  编辑后 + 红框 + 橙斑 (最接近论文右面板)
  README.md                     每张图说明 + 推荐给 GPT 的附加指令

用法:
  cd code && python export_gpt_assets.py
"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image, ImageDraw
from dct_watermark import _block_grid, _demodulate_bit

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f'{BASE}/paper/figures/gpt_assets'
os.makedirs(OUT, exist_ok=True)

RED = (214, 39, 40)
ORANGE = (255, 140, 0)


def load(p, imgsize=None):
    """默认**保持原始分辨率** (旧版强制缩到 256, 导致导出素材分辨率过低).

    高分辨率示例样本由 run_hires_demo_samples.py 生成 (原始 COCO 分辨率),
    只需裁剪到 8 的倍数; 不再做任何缩放.
    """
    img = np.array(Image.open(p).convert('RGB'))
    if imgsize:
        img = np.array(Image.fromarray(img).resize((imgsize, imgsize), Image.BILINEAR))
    h, w = img.shape[:2]
    return img[:h - h % 8, :w - w % 8]

def y_channel(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)


def check_bits(n_h, n_w, seed=42):
    return np.random.RandomState(seed).randint(0, 2, (n_h, n_w)).astype(np.uint8)


def check_failure_map(orig, att, delta=15):
    """与论文/fig_s5 完全一致的真实失败图: 1.0=check 被改写, 0.5=与期望不符."""
    H, W, n_h, n_w = _block_grid(*orig.shape[:2])
    ck = check_bits(n_h, n_w)
    yo, ya = y_channel(orig)[:H, :W], y_channel(att)[:H, :W]
    fm = np.zeros((n_h, n_w))
    for i in range(n_h):
        for j in range(n_w):
            do = cv2.dct(yo[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32))
            da = cv2.dct(ya[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32))
            co, ca = _demodulate_bit(do[3, 2], delta), _demodulate_bit(da[3, 2], delta)
            if co != ca:
                fm[i, j] = 1.0
            elif ca != ck[i, j]:
                fm[i, j] = 0.5
    return fm


def dashed_rect(draw, box, color=RED, width=None, dash=None, gap=None):
    """红虚线框. 线宽/虚线段长按图幅自适应 (旧版硬编码 5/24 只适合 ~1024px 图)."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    w_img = draw.im.size[0]
    scale = max(w_img / 1024.0, 0.5)
    width = width or max(2, int(round(5 * scale)))
    dash = dash or max(8, int(round(24 * scale)))
    gap = gap or max(5, int(round(16 * scale)))

    def seg(a, b):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L == 0:
            return
        ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        d = 0.0
        while d < L:
            dd = min(dash, L - d)
            draw.line([(a[0] + ux * d, a[1] + uy * d),
                       (a[0] + ux * (d + dd), a[1] + uy * (d + dd))], fill=color, width=width)
            d += dash + gap

    seg((x1, y1), (x2, y1)); seg((x2, y1), (x2, y2))
    seg((x2, y2), (x1, y2)); seg((x1, y2), (x1, y1))


def square_crop(im, xy, margin=0.1):
    """以 bbox 为中心的方形裁剪 (给框架图当缩略图).

    注意 im 现在可能是非正方形 (原分辨率样本), 裁剪需按各自边长处理.
    """
    iw, ih = im.size
    x1, y1, x2, y2 = xy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1, (x2 - x1) * ih / iw) * (1 + margin)
    l = max(0, int(cx - side / 2)); t = max(0, int(cy - side / 2))
    r = min(iw, l + int(side)); b = min(ih, t + int(side))
    l = max(0, r - int(side)); t = max(0, b - int(side))
    return im.crop((l, t, r, b))


def damage_rgba(fm, size):
    """真实失败斑 -> 透明背景橙色 RGBA (块分辨率 nearest 放大).

    size = (W, H), 与底图逐像素对齐 (不再假设方形).
    """
    W, H = size
    nh, nw = fm.shape
    if (W, H) == (nw, nh):
        fm_up = fm
    else:
        # 块 -> 像素: 用最近邻, 保证每块整块同色
        fm_up = cv2.resize(fm, (W, H), interpolation=cv2.INTER_NEAREST)
    a = np.where(fm_up >= 1.0, 100, np.where(fm_up > 0, 65, 0)).astype(np.uint8)
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = ORANGE
    rgba[..., 3] = a
    return rgba


def main():
    from demo_example import sample as demo_sample
    cid, src, bbox, zh, en, url, img_dir = demo_sample()

    # 保持原始分辨率 (不再缩到 256); teaser 与框架图共用同一样本
    orig = load(f'{img_dir}/{cid}_orig.png')
    att = load(f'{img_dir}/{cid}_att.png')
    fm = check_failure_map(orig, att)
    H, W = orig.shape[:2]
    # 保持原分辨率导出 (旧版强行把 256 上采样到 1024, 属于虚高分辨率)
    o, a = Image.fromarray(orig), Image.fromarray(att)
    xy = [v * W if i % 2 == 0 else v * H for i, v in enumerate(bbox)]

    # 1) 原始 / 编辑后
    o.save(f'{OUT}/teaser_before.png')
    a.save(f'{OUT}/teaser_after.png')

    # 1b) Fig.1 框架图缩略图素材 (Image I / Edited image Ia 两个节点用)
    o.save(f'{OUT}/framework_image_I.png')
    a.save(f'{OUT}/framework_edited_Ia.png')
    oc = square_crop(o, xy); ac = square_crop(a, xy)
    oc.save(f'{OUT}/framework_image_I_crop.png')
    ac.save(f'{OUT}/framework_edited_Ia_crop.png')

    # 2) 真实失败斑 (透明背景) + 叠加到编辑后
    dmg = Image.fromarray(damage_rgba(fm, (W, H)), 'RGBA')
    dmg.save(f'{OUT}/teaser_damage_only.png')
    Image.alpha_composite(a.convert('RGBA'), dmg).convert('RGB').save(
        f'{OUT}/teaser_after_damage.png')

    # 3) 带红虚线框的版本
    ob = o.copy(); dashed_rect(ImageDraw.Draw(ob), xy); ob.save(f'{OUT}/teaser_before_bbox.png')
    ab = Image.alpha_composite(a.convert('RGBA'), dmg).convert('RGB')
    dashed_rect(ImageDraw.Draw(ab), xy); ab.save(f'{OUT}/teaser_after_damage_bbox.png')

    write_readme(cid, src, bbox, zh, en, url, (W, H))
    print(f'exported -> {OUT}')
    for f in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, f)
        if f.endswith('.png'):
            print(f'  {f:34s} {Image.open(p).size}')


def write_readme(cid, src, bbox, zh, en, url, wh):
    txt = f"""# GPT-image-2 底图素材（真实数据，GPT 自己拿不到）

示例样本：COCO val2017 `{src}`（高分辨率样本 `{cid}`，{wh[0]}x{wh[1]}）

> 素材由 `code/export_gpt_assets.py` 从 `results/attackset_hires_sample/` 导出，
> **保持原始分辨率，不做任何缩放**（旧版曾强制缩到 256px 再放大回 1024，两处都是虚的）。

这些是喂给 GPT-image-2 当底图/参考的**真实图片**。它联网只能找到“编辑前”的 COCO 原图，
“编辑后”是本项目 InstructPix2Pix 的输出、“真实失败斑”需要水印密钥，GPT 都画不出来。

---

## A. 动机图（teaser）用

| 文件 | 内容 | 用途 |
|---|---|---|
| `teaser_before.png` | 编辑前真实图（COCO {src}） | 左面板底图 |
| `teaser_before_bbox.png` | 编辑前 + 红虚线编辑框 | 左面板可直接用 |
| `teaser_after.png` | **编辑后（本项目 SD inpainting 输出）** | 右面板底图 |
| `teaser_after_damage.png` | 编辑后 + **真实 check-bit 失败斑（橙）** | 右面板 |
| `teaser_after_damage_bbox.png` | 编辑后 + 红框 + 橙斑 | **右面板最终形态，直接能用** |
| `teaser_damage_only.png` | 真实失败斑，透明背景 RGBA | 让 GPT 自己叠到任意底图 |

---

## B. 方法框架图（Fig.1）用

| 文件 | 内容 | 用途 |
|---|---|---|
| `framework_image_I.png` | 编辑前真实图 | `Image I` 节点缩略图 |
| `framework_edited_Ia.png` | 编辑后（本项目 SD 输出） | `Edited image Ia` 节点缩略图 |
| `framework_image_I_crop.png` | 围绕编辑区域的方形裁剪 | 缩略图（更清楚） |
| `framework_edited_Ia_crop.png` | 编辑后同一区域方形裁剪 | 缩略图（更清楚） |

> 框架图其余部分（圆角框、箭头、标签）是示意图，GPT 能画，但英文标签/公式容易被画错，
> 所以最终仍建议用 matplotlib 版（`code/make_figures.py::fig1_framework()`）。

---

## 公共信息

- 编辑指令：`{zh}`（英文："{en}"）
- 编辑框 bbox（归一化 x1,y1,x2,y2）：`{bbox}`
- 原图下载地址（GPT 可自行核对）：{url}

---

## 给 GPT 的附加指令（务必连同参考图一起上传）

### 动机图
```
这是真实数据，请严格使用这些参考图，不要重画场景、不要替换物体：
- 左面板用 teaser_before_bbox.png（已含红虚线编辑框）。
- 右面板用 teaser_after_damage_bbox.png（编辑后 + 真实橙色水印损伤斑 + 红框）。
- 你只负责版面：加深色标题横幅、加 "MLLM" 气泡、加 "≠ mismatch" 标注、
  底部加一行结论横幅；不要改动照片内容和橙斑形状。
- 图上一律不要出现数字、百分比、公式。
```

### 框架图
```
缩略图请用参考图：Image I 节点用 framework_image_I_crop.png，
Edited image Ia 节点用 framework_edited_Ia_crop.png，不要用 AI 虚构的图。
英文标签必须逐字准确。
```
"""
    open(f'{OUT}/README.md', 'w').write(txt)


if __name__ == '__main__':
    main()
