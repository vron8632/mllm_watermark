#!/usr/bin/env python3
"""论文图表生成 (paper/figures/).

Fig1  framework: 双系数嵌入 + 校验位 + v2 软权重/膨胀 + 投票流水线
Fig2  qualitative: 攻击示例 (orig/att/mask + 每图 BA)
Fig3  diagnosis: 可解释诊断示例
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = f'{BASE}/paper/figures'
os.makedirs(FIG_DIR, exist_ok=True)
ATTACKSET = f'{BASE}/results/attackset_20260817_082819/images'
# 图例 BA 数字必须来自 attackset 自身 (eval_attackset.py 按论文协议重算),
# 不能引用 signal_ra 主实验 (两批编辑指令不同, 图与数字会不对应).
ATTACKSET_EVAL = f'{BASE}/results/attackset_eval_20260817_124219.json'


# ---------------------------------------------------------------- Fig 0 (teaser)
def fig0_teaser():
    """动机图: 语义编辑区 != 水印损伤区.

    用**真实数据**绘制: 高分辨率示例样本 (原始 COCO 分辨率, InstructPix2Pix 重跑,
    DeepSeek 自然度挑选) + 真实 check-bit 失败图 (与 fig_s5 同一计算),
    以 matplotlib 叠加红虚线框与橙色损伤斑. 不包含任何实验数字.

    注意: 不再把图像缩到 256 — 256px 印刷过小且编辑痕迹明显.
    """
    from PIL import Image
    import cv2
    from dct_watermark import _block_grid, _demodulate_bit

    def load(p):
        """原分辨率加载, 只裁剪到 8 的倍数 (不做任何缩放)."""
        img = np.array(Image.open(p).convert('RGB'))
        h, w = img.shape[:2]
        return img[:h - h % 8, :w - w % 8]

    y_channel = lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)

    # 与 run_signal_ra.check_bits 等价 (seed=42), 避免引入 YOLO/重依赖
    def check_bits(hh, ww, nh, nw, seed=42):
        rng = np.random.RandomState(seed)
        return rng.randint(0, 2, (nh, nw)).astype(np.uint8)

    from demo_example import sample as demo_sample
    cid, src, bbox, _zh, instruction, _url, img_dir = demo_sample()

    orig = load(f'{img_dir}/{cid}_orig.png')
    att = load(f'{img_dir}/{cid}_att.png')
    h, w = orig.shape[:2]
    H, W, n_h, n_w = _block_grid(h, w)

    # --- 真实 check-bit 失败图 (与 fig_s5 完全一致) ---
    ck = check_bits(h, w, n_h, n_w, seed=42)
    yo, ya = y_channel(orig)[:H, :W], y_channel(att)[:H, :W]
    fm = np.zeros((n_h, n_w))
    for i in range(n_h):
        for j in range(n_w):
            do = cv2.dct(yo[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32))
            da = cv2.dct(ya[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32))
            co, ca = _demodulate_bit(do[3, 2], 15), _demodulate_bit(da[3, 2], 15)
            if co != ca:
                fm[i, j] = 1.0
            elif ca != ck[i, j]:
                fm[i, j] = 0.5

    # 显示尺寸: 保持原分辨率 (仅当像素过多时轻微下采样, 不做放大)
    MAXS = 640
    scale = min(1.0, MAXS / max(h, w))
    ds = lambda im: np.array(Image.fromarray(im).resize(
        (max(8, int(im.shape[1] * scale)), max(8, int(im.shape[0] * scale))), Image.LANCZOS))
    o_d, a_d = ds(orig), ds(att)
    S_h, S_w = o_d.shape[:2]
    x1, y1, x2, y2 = [int(v * q) for v, q in zip(bbox, (S_w, S_h, S_w, S_h))]
    cbx = lambda j: (j + 0.5) * S_w / n_w
    cby = lambda i: (i + 0.5) * S_h / n_h

    fw = 9.6                                # 更紧凑, 左右两图间距更小
    fh = fw * S_h / S_w * 0.52
    fig, axes = plt.subplots(1, 2, figsize=(fw, fh))
    fig.subplots_adjust(top=0.80, bottom=0.17, left=0.005, right=0.995, wspace=0.02)
    box = dict(boxstyle='round,pad=0.32', fc='white', ec='none', alpha=0.82)

    # ---------- (A) 局部、隐形的编辑 ----------
    ax = axes[0]
    ax.imshow(o_d, interpolation='lanczos')
    ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                 ec='#111111', lw=2.4, ls=(0, (6, 4))))
    ax.text(x1 + 3, max(y1 - 6, 14), 'edit region (semantic)', color='#111111',
            fontsize=10.5, fontweight='bold', va='bottom', bbox=box)
    ax.text(0.03, 0.03, f'MLLM: "{instruction}"', transform=ax.transAxes,
            fontsize=10, color='white', va='bottom',
            bbox=dict(boxstyle='round,pad=0.35', fc='#4b3f8f', ec='none', alpha=0.92))
    # 注: 不在图内标注源文件名 (COCO 编号), 该信息放入 caption
    ax.set_title('(A) local, invisible edit', fontsize=13, fontweight='bold',
                 color='#1f4e79', loc='left', pad=8)
    ax.axis('off')

    # ---------- (B) 隐藏、错位的损伤 ----------
    ax = axes[1]
    ax.imshow(a_d, interpolation='lanczos')
    ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                 ec='#111111', lw=2.4, ls=(0, (6, 4))))
    # 两类损伤分开显示, 避免半透明色块糊满整图:
    #   区外外溢 = 橙色主色 (论文动机); 区内受损 = 淡灰 (对比参考)
    in_fm = np.zeros_like(fm)
    out_fm = np.zeros_like(fm)
    for i in range(n_h):
        for j in range(n_w):
            if fm[i, j] > 0:
                if x1 <= cbx(j) <= x2 and y1 <= cby(i) <= y2:
                    in_fm[i, j] = fm[i, j]
                else:
                    out_fm[i, j] = fm[i, j]
    ax.imshow(np.ma.masked_where(in_fm <= 0, in_fm), cmap='Greys', vmin=0, vmax=1,
              alpha=0.30, interpolation='nearest', extent=(0, S_w, S_h, 0))
    ax.imshow(np.ma.masked_where(out_fm <= 0, out_fm), cmap=ListedColormap(['#D55E00']), vmin=0.0, vmax=1.2,
              alpha=0.55, interpolation='nearest', extent=(0, S_w, S_h, 0))

    n_in, n_out = int((in_fm > 0).sum()), int((out_fm > 0).sum())
    tot_in = sum(1 for i in range(n_h) for j in range(n_w)
                 if x1 <= cbx(j) <= x2 and y1 <= cby(i) <= y2)
    tot_out = n_h * n_w - tot_in
    # 标注框外被破坏的块 -> 外溢
    outside = [(i, j) for i in range(n_h) for j in range(n_w) if out_fm[i, j] > 0]
    if outside:
        cxm, cym = (x1 + x2) / 2, (y1 + y2) / 2
        i0, j0 = min(outside, key=lambda t: (cbx(t[1]) - cxm) ** 2 + (cby(t[0]) - cym) ** 2)
        ax.annotate(f'damage outside the edit region\n{n_out}/{tot_out} blocks '
                    f'({100*n_out/tot_out:.0f}\\%)',
                    xy=(cbx(j0), cby(i0)),
                    xytext=(S_w * 0.03, S_h * 0.80), fontsize=10, fontweight='bold',
                    color='#b34700', bbox=box,
                    arrowprops=dict(arrowstyle='-|>', color='#b34700', lw=1.8))
    ax.text(x1 + 3, max(y2 + 8, 16), f'damage inside: {n_in}/{tot_in} '
            f'({100*n_in/tot_in:.0f}\\%)', color='#555', fontsize=9.5,
            va='top', bbox=box)
    ax.text(x1 + 3, max(y1 - 6, 14), 'edit region (semantic)', color='#111111',
            fontsize=10.5, fontweight='bold', va='bottom', bbox=box)
    ax.set_title('(B) damage is not confined to the edit region', fontsize=13,
                 fontweight='bold', color='#b34700', loc='left', pad=8)
    ax.axis('off')

    # ---------- 版面调整 ----------
    # 注: 依 JISA 规定 "a caption should comprise a brief title (not on the figure
    # itself)" —— 图内**不放**整图标题, 也不放重复 caption 的说明句;
    # 仅保留面板编号 (A)/(B)、区域标注与箭头, 其余信息全部写入 caption.
    fig.subplots_adjust(top=0.90, bottom=0.06)

    fig.savefig(f'{FIG_DIR}/teaser.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{FIG_DIR}/teaser.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'fig0 teaser done (COCO {src}, {w}x{h}, real check-bit failure map)')


# ---------------------------------------------------------------- Fig 1
def fig1_framework():
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    ax.set_xlim(0, 125); ax.set_ylim(0, 64)
    ax.axis('off')

    def box(x, y, w, h, text, fc='#eaf3fb', ec='#2c6fbb', fs=10.5, bold=False,
            sub=None):
        b = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.35',
                           fc=fc, ec=ec, lw=1.4)
        ax.add_patch(b)
        if sub:
            ax.text(x + w/2, y + h*0.62, text, ha='center', va='center',
                    fontsize=fs, fontweight='bold' if bold else 'normal')
            ax.text(x + w/2, y + h*0.25, sub, ha='center', va='center',
                    fontsize=fs-1.5, color='#555')
        else:
            ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                    fontsize=fs, fontweight='bold' if bold else 'normal')

    def arrow(x1, y1, x2, y2, text=None, color='#333', style='-|>'):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                            mutation_scale=14, color=color, lw=1.5)
        ax.add_patch(a)
        if text:
            mx, my = (x1+x2)/2, (y1+y2)/2 + 1.2
            ax.text(mx, my, text, ha='center', va='bottom', fontsize=8.5,
                    color='#444')

    # --- 嵌入分支 (上)
    ax.text(2, 60.5, 'Embedding (shared key $\\mathcal{K}$)', fontsize=12,
            fontweight='bold', color='#2c6fbb')
    box(2, 52, 17, 6.5, 'Image', sub='$I$')
    box(22, 52, 19, 6.5, '8$\\times$8 blocks', sub='DCT')
    box(44, 52, 27, 6.5, 'Dual-coef. QIM', sub='payload $(4,1)$ + check $(3,2)$', bold=True)
    box(74, 52, 17, 6.5, 'IDCT', sub='merge Y')
    box(94, 52, 17, 6.5, 'Watermarked', sub='$I_w$')
    arrow(19.5, 55.2, 21.5, 55.2)
    arrow(41.5, 55.2, 43.5, 55.2)
    arrow(71.5, 55.2, 73.5, 55.2)
    arrow(91.5, 55.2, 93.5, 55.2)
    box(50, 43, 24, 5, 'check bits $c_k(i,j)$', sub='PRNG($\\mathcal{K}$, coords)',
        fc='#fdf3e3', ec='#c77d1e', fs=9)

    # --- 攻击
    ax.text(2, 35, 'MLLM-driven semantic editing attack', fontsize=12,
            fontweight='bold', color='#b03030')
    box(2, 26, 21, 6.5, 'MLLM', sub='edit instr. + bbox')
    box(26, 26, 23, 6.5, 'SD inpainting', sub='local rewrite')
    box(52, 26, 21, 6.5, 'Attacked image', sub='$I_a$')
    arrow(4, 52, 4, 33, 'edit region')
    arrow(10, 26.5, 10, 30.5)
    arrow(23.5, 29.2, 25.5, 29.2)
    arrow(49.5, 29.2, 51.5, 29.2)

    # --- 提取分支 (下)
    ax.text(2, 17.5, 'Extraction (signal-check v2)', fontsize=12,
            fontweight='bold', color='#2c6fbb')
    box(2, 8, 15, 6.5, '$I_a$', sub='8$\\times$8 DCT')
    box(20, 8, 19, 6.5, 'Check-bit verify', sub='$c_k$ mismatch?')
    box(42, 8, 18, 6.5, 'Soft conf.', sub='$w_b = 1-\\frac{d_b}{d_b+d_o}$')
    box(63, 8, 16, 6.5, 'Dilation', sub='$\\times0.3$ neighbors')
    box(82, 8, 19, 6.5, 'Weighted vote', sub='per payload bit')
    box(104, 8, 17, 6.5, 'Message', sub='$\\hat{m}$')
    arrow(17.5, 11.2, 19.5, 11.2)
    arrow(39.5, 11.2, 41.5, 11.2)
    arrow(60.5, 11.2, 62.5, 11.2)
    arrow(79.5, 11.2, 81.5, 11.2)
    arrow(101.5, 11.2, 103.5, 11.2)
    arrow(31, 8, 31, 25, text='extract', color='#555', style='-|>')
    arrow(29, 26, 29, 24.5, color='#999', style='-')
    ax.plot([29, 29], [24.5, 16], color='#999', lw=1.0, ls='--')
    ax.plot([29, 25.5], [16, 16], color='#999', lw=1.0, ls='--')

    # --- 诊断 (右侧)
    box(104, 26, 19, 6.5, 'MLLM diagnosis', sub='natural language',
        fc='#eafaf0', ec='#2e9e5b')
    arrow(73.5, 29.2, 103.5, 29.2, text='crop region pair')
    arrow(63, 16, 63, 25.5, color='#999')
    ax.plot([63, 63], [25.5, 26], color='#999', lw=1.0, ls='--')

    ax.text(62, 3.2, 'per-block votes with soft weights; check-failing / adjacent blocks down-weighted',
            ha='center', fontsize=8.5, color='#666')
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/fig1_framework.pdf', bbox_inches='tight')
    fig.savefig(f'{FIG_DIR}/fig1_framework.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('fig1 done')


# ---------------------------------------------------------------- Fig 2
def fig2_qualitative():
    """代表性 MLLM 引导语义编辑 (Figure 3).

    样本来自 attackset_hires_sample: **原始 COCO 分辨率** + InstructPix2Pix 重跑.
    挑选依据是**客观指标 + 多模态核查**, 不是 LLM 自然度分 (后者不可靠):
      - bbox 外像素差 ≈ 0   -> 编辑确实是局部的
      - bbox 内像素差 >= 8  -> 编辑确实发生了 (避免"看不出改了哪里")
      - 接缝比 seam ≈ 1     -> 没有拼贴痕迹

    三行覆盖两种编辑类型 (object replacement / attribute editing),
    每行都是**不同图像**, 避免出现"上下两行看起来一样"的观感问题.
    早期版本中间行用过同一指令的第二个 seed (0002_43), 与首行过于相似, 已替换.

    注: 早期版本还用过 0030_42 (雪景删人, 像素差仅 11.9) 编辑过弱, 读者看不出差别.
    """
    from PIL import Image
    import json as _json

    HIRES = f'{BASE}/results/attackset_hires_sample'
    hs_meta = _json.load(open(f'{HIRES}/meta.json'))
    hm = {x['id']: x for x in hs_meta['samples']}
    rows = [
        ('0002_42', 'replace bookshelf $\\rightarrow$ TV', 'object replacement'),
        ('0032_43', 'replace skull logo $\\rightarrow$ smiley', 'object replacement (2nd type)'),
        ('0045_42', 'jacket: yellow $\\rightarrow$ red', 'attribute editing'),
    ]
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(10.2, 3.0 * n),
                             gridspec_kw={'hspace': 0.12, 'wspace': 0.04})
    for r, (cid, inst, lbl) in enumerate(rows):
        o = Image.open(f'{HIRES}/images/{cid}_orig.png').convert('RGB')
        a = Image.open(f'{HIRES}/images/{cid}_att.png').convert('RGB')
        m = Image.open(f'{HIRES}/images/{cid}_mask.png').convert('L')
        for c, (im, ttl) in enumerate(((o, 'original'), (a, 'edited'),
                                       (m, 'edit mask'))):
            ax = axes[r, c]
            if c == 2:
                ax.imshow(im, cmap='gray', vmin=0, vmax=255)
            else:
                ax.imshow(im)
            ax.axis('off')
            # 列标题**左对齐**、行标签**放图上方独立一行** -> 不再互相重叠
            ax.set_title(ttl, fontsize=10, loc='left')
        h, w = hm[cid]['h'], hm[cid]['w']
        # 行标签移到列标题之上 (y=1.20), 与 set_title(y≈1.0) 分离
        axes[r, 0].annotate(f'{lbl}   ({w}$\\times${h})', xy=(0, 1.185),
                            xycoords='axes fraction', fontsize=9.5,
                            fontweight='bold', color='#2c6fbb', va='bottom')
    fig.savefig(f'{FIG_DIR}/fig2_qualitative.pdf', bbox_inches='tight')
    fig.savefig(f'{FIG_DIR}/fig2_qualitative.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('fig2 done (3 rows, labels de-overlapped; rows: 0002_42 / 0032_43 / 0045_42)')


# ---------------------------------------------------------------- Fig 3
def fig3_diagnosis():
    """可解释诊断示例 (Figure 4).

    图像内容与 caption 严格对应: 左=编辑前区域, 中=编辑后区域, 右=**该样本真实调用
    MLLM 得到的诊断 JSON** (由 code/run_diagnosis_demo.py 生成并缓存).
    若缓存不存在则明确报错, 不再使用硬编码文本 —— 之前硬编码的 "bus 被删除"
    配的却是其它样本的图, 图文不符.
    """
    from PIL import Image
    import json as _json

    HIRES = f'{BASE}/results/attackset_hires_sample'
    cache = f'{BASE}/results/diagnosis_demo.json'
    if not os.path.exists(cache):
        raise SystemExit(
            f'缺少 {cache}: 请先运行 `python run_diagnosis_demo.py` 生成真实诊断文本')
    d = _json.load(open(cache))
    cid = d['cid']
    diag = d['diagnosis']
    conf = d.get('confidence')

    o = Image.open(f'{HIRES}/images/{cid}_orig.png').convert('RGB')
    a = Image.open(f'{HIRES}/images/{cid}_att.png').convert('RGB')
    m = Image.open(f'{HIRES}/images/{cid}_mask.png').convert('L')
    bbox = m.getbbox()          # 用 mask 的紧致包围盒裁剪编辑区域
    c_o, c_a = o.crop(bbox), a.crop(bbox)
    side = max(c_o.size)
    def pad(im):
        out = Image.new('RGB', (side, side), (255, 255, 255))
        out.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
        return out
    c_o, c_a = pad(c_o), pad(c_a)

    txt = f'"{diag}"'
    if conf is not None:
        txt += f'\n\nconfidence: {conf}'
    # 右栏加宽 (图像:文字 = 1:1.15) 并把字号提到 12 -> 诊断文字清晰可读
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.1),
                             gridspec_kw={'width_ratios': [1, 1, 1.15]})
    axes[0].imshow(c_o); axes[0].set_title('edited region (original)', fontsize=10)
    axes[1].imshow(c_a); axes[1].set_title('edited region (attacked)', fontsize=10)
    axes[2].axis('off')
    axes[2].text(0.02, 0.95, txt, fontsize=12, va='top', wrap=True,
                 linespacing=1.45,
                 bbox=dict(boxstyle='round,pad=0.6', fc='#eafaf0', ec='#2e9e5b'))
    for i in range(2):
        axes[i].axis('off')
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/fig3_diagnosis.pdf', bbox_inches='tight')
    fig.savefig(f'{FIG_DIR}/fig3_diagnosis.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'fig3 done (sample {cid}, real MLLM diagnosis)')


if __name__ == '__main__':
    fig0_teaser()
    fig1_framework()
    fig2_qualitative()
    fig3_diagnosis()
