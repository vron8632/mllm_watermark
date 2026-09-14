#!/usr/bin/env python3
"""ICASSP 2027 — Idea Probe: 最小实验验证改进方向的新颖性与可行性.

背景: 现有论文《Training-Free Semantic-Guided DCT-QIM Watermarking》的负面结果为
  1) 等 PSNR 失真预算下, YOLO 语义强度调制 vs 均匀嵌入在 JPEG/噪声下无差异;
  2) 局部实例移除攻击下语义方案更差 (移除 50% 时 71.9% vs 84.7%).
本脚本用最小实验验证三个改进方向:

  A. removal-aware voting (移除感知投票): 提取时重新跑 YOLO, 识别"嵌入时存在但
     攻击后消失"的实例区域, 将其块的投票权重置零, 修复"移除摧毁最高权重块"的失败机制.
  B. perceptual budget (感知预算强度图): 用简化的 Watson 对比度/亮度掩蔽 (块标准差/
     均值偏离 128 代理) 生成感知强度图替代拍脑袋的 0.3/0.8, 在"等 SSIM"协议下对比.
  C. full = A + B 组合.

公平协议: 对每个方案做 Δ 扫描, 输出 (PSNR, SSIM, BA), 在"等 PSNR"与"等 SSIM"
邻域分别比较, 避免只在一个失真点比较的偏差.

用法:
  cd code && python run_idea_probe.py                    # 默认: COCO 100 张 + Kodak 24 张
  python run_idea_probe.py --quick                       # 快速模式: 只跑方向 A, 每集 20 张
  python run_idea_probe.py --n 60 --datasets coco        # 自定义规模/数据集
  python run_idea_probe.py --datasets columbia-sp        # Columbia 拼接全集 (需 --n 912)
  python run_idea_probe.py --n 500 --datasets casia-tp   # CASIA 篡改全集抽样
  python run_idea_probe.py --msglen 256 --scan-attacks clean,jpeg50,noise005  # 自定义攻击
  python analyze_results.py                              # 统计检验 (paired t-test/Wilcoxon/d)

输出: ../results/idea_probe_<时间戳>.json (全部扫描表 + 判据判定 + per-image BA)
"""
import sys, os, time, json, io, argparse, datetime, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image, ImageFilter
from skimage.metrics import structural_similarity
from dct_watermark import (embed_uniform, extract_uniform,
                           embed_adaptive, extract_adaptive, bit_accuracy,
                           _block_grid, _delta_eff_map, _demodulate_bit,
                           _majority_vote)
from run_experiments import get_strength_map, YOLO_MODEL

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = f'{BASE}/data'
PARENT_DATA = os.path.join(os.path.dirname(BASE), 'data')  # DuetGuard/data (完整数据集)
RESULTS_DIR = f'{BASE}/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# 数据源映射: 键 -> 路径 (支持 glob)
#   coco/kodak 在本项目 data/ 下; casia/columbia 全集在父目录 DuetGuard/data/ 下
#   casia-au 7491 张真实, casia-tp 5123 张篡改 (官方修订版 CASIA2.0_revised)
#   columbia-au 935 张真实, columbia-sp 912 张拼接 (ImSpliceDataset 全集)
DATA_SOURCES = {
    'coco':        f'{D}/COCO',
    'kodak':       f'{D}/Kodak',
    'div2k':       f'{D}/DIV2K_valid_HR',
    'casia':       f'{PARENT_DATA}/casia2/CASIA2.0_revised',
    'casia-au':    f'{PARENT_DATA}/casia2/CASIA2.0_revised/Au',
    'casia-tp':    f'{PARENT_DATA}/casia2/CASIA2.0_revised/Tp',
    'columbia':    f'{PARENT_DATA}/columbia/ImSpliceDataset',
    'columbia-au': f'{PARENT_DATA}/columbia/ImSpliceDataset/Au-*',
    'columbia-sp': f'{PARENT_DATA}/columbia/ImSpliceDataset/Sp-*',
}


# ── 进度显示 (剩余时分秒 / ETA) ─────────────────────────────────
class Progress:
    """轻量进度条: 打印 [时刻] 完成数/总数 已用 剩余(时分秒) ETA."""

    def __init__(self, total, desc=''):
        self.total, self.desc, self.n = total, desc, 0
        self.t0 = time.time()
        self.t_last = 0.0
        self._emit()

    @staticmethod
    def _fmt(s):
        s = int(s)
        return f'{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}'

    def update(self, k=1):
        self.n += k
        if time.time() - self.t_last < 2.0 and self.n < self.total:
            return  # 节流: 每 2 秒打印一行
        self._emit()

    def _emit(self):
        self.t_last = time.time()
        el = self.t_last - self.t0
        if self.n > 0:
            rem = el / self.n * (self.total - self.n)
            eta = datetime.datetime.now() + datetime.timedelta(seconds=rem)
            info = (f'已用 {self._fmt(el)}  剩余约 {self._fmt(rem)}  '
                    f'(ETA {eta.strftime("%H:%M:%S")})')
        else:
            info = '已用 00:00:00'
        print(f'  [{datetime.datetime.now().strftime("%H:%M:%S")}] '
              f'{self.desc} {self.n}/{self.total}  {info}', flush=True)

    def done(self):
        if self.n < self.total:
            self.n = self.total
        self._emit()


# ── 通用工具 ────────────────────────────────────────────────────
def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 20 * np.log10(255 / (np.sqrt(mse) + 1e-8))


def ssim(a, b):
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


def load(p, imgsize=256):
    """读取并裁剪到 8 的倍数; 默认 resize 到 256x256 (与论文协议一致, 控制运行时间)."""
    img = np.array(Image.open(p).convert('RGB'))
    if imgsize:
        img = np.array(Image.fromarray(img).resize((imgsize, imgsize), Image.BILINEAR))
    h, w = img.shape[:2]
    return img[:h - h % 8, :w - w % 8]


def jpeg(img, q):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='JPEG', quality=q)
    buf.seek(0)
    return np.array(Image.open(buf).convert('RGB'))


def gnoise(img, sigma):
    """高斯噪声 (sigma 为像素值标准差, 与论文协议一致: σ=0.01 等按 [0,1] 归一化)."""
    rng_n = np.random.RandomState(0)
    return np.clip(img.astype(np.float32) + rng_n.randn(*img.shape) * sigma * 255,
                   0, 255).astype(np.uint8)


def gblur(img, r):
    return np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=r)))


# scan 阶段可选攻击: 键 -> 攻击函数 (None = 无攻击)
SCAN_ATTACKS = {
    'clean': None,
    'jpeg95': lambda im: jpeg(im, 95),
    'jpeg75': lambda im: jpeg(im, 75),
    'jpeg50': lambda im: jpeg(im, 50),
    'jpeg25': lambda im: jpeg(im, 25),
    'noise001': lambda im: gnoise(im, 0.01),
    'noise002': lambda im: gnoise(im, 0.02),
    'noise005': lambda im: gnoise(im, 0.05),
    'gblur1': lambda im: gblur(im, 1.0),
    'gblur2': lambda im: gblur(im, 2.0),
}


def y_channel(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)


def detect_removal_mask(img, frac):
    """攻击者视角: 基于原图检测要移除的实例掩膜 (按面积从大到小, 直到达到 frac 比例).

    返回 m_total (h,w) bool 掩膜; 若检测不到实例或面积过小返回 None.
    """
    h, w = img.shape[:2]
    res = YOLO_MODEL(img)[0]
    if not res or not res.masks or len(res.masks) == 0:
        return None
    masks = res.masks.data.cpu().numpy()
    m_total = np.zeros((h, w), dtype=bool)
    idxs = np.argsort([m.sum() for m in masks])[::-1]
    removed = 0
    for idx in idxs:
        m = np.array(Image.fromarray((masks[idx] * 255).astype(np.uint8))
                     .resize((w, h), Image.NEAREST)) > 128
        m_total = m_total | m
        removed = m_total.sum()
        if removed > frac * h * w:
            break
    if removed < 0.05 * h * w:
        return None
    return m_total


def apply_removal(img, m_total, bg_img=None):
    """用背景均值填充移除区域 (bg_img 默认取 img 自身的背景均值)."""
    h, w = img.shape[:2]
    src = img if bg_img is None else bg_img
    bg = (src[~m_total].mean(axis=0).astype(np.uint8)
          if (~m_total).sum() > 0 else np.array([128, 128, 128]))
    out = img.copy()
    out[m_total] = bg
    return out


# ── 强度图生成器 ────────────────────────────────────────────────
def jnd_strength_map(img, sigma=5):
    """方向 B: 感知预算强度图 (简化 Watson JND).

    对 Y 通道逐 8x8 块计算感知掩蔽: 对比度掩蔽用块标准差 (纹理复杂→人眼不敏感→
    可嵌更强), 亮度掩蔽用块均值偏离 128 的程度. 两者取大者归一化后映射到
    [0.3, 0.8] (与语义图同尺度, 保证 Δ_eff 公式一致、比较公平), 再经与语义图
    相同的高斯模糊 + clamp 后处理.
    """
    h, w = img.shape[:2]
    y = y_channel(img)
    n_h, n_w = h // 8, w // 8
    tex = np.zeros((n_h, n_w), dtype=np.float32)
    lum = np.zeros((n_h, n_w), dtype=np.float32)
    for i in range(n_h):
        for j in range(n_w):
            b = y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            tex[i, j] = b.std() / 128.0
            lum[i, j] = abs(b.mean() - 128.0) / 128.0
    j = np.maximum(tex, lum)
    j = (j - j.min()) / (j.max() - j.min() + 1e-8)   # [0,1]
    S = 0.3 + 0.5 * j                                 # [0.3, 0.8]
    S = Image.fromarray((S * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
    S = np.array(S.filter(ImageFilter.GaussianBlur(radius=sigma))) / 255.0
    return np.clip(S, 0.1, 1.0).astype(np.float32)


def full_strength_map(img, sigma=5):
    """方向 C: 感知 × 语义 组合强度图 (两者平均, 保持同尺度)."""
    s_sem = get_strength_map(img, sigma=sigma)
    s_jnd = jnd_strength_map(img, sigma=sigma)
    return np.clip(0.5 * s_sem + 0.5 * s_jnd, 0.1, 1.0).astype(np.float32)


# ── 移除感知投票 (方向 A) ───────────────────────────────────────
def removed_blocks(sm_embed, sm_new, H, W):
    """返回 (n_h, n_w) bool 掩膜: 嵌入时是实例 (sm>0.5) 但攻击后不再是实例."""
    sm_e = cv2.resize(sm_embed.astype(np.float32), (W, H))
    sm_n = cv2.resize(sm_new.astype(np.float32), (W, H))
    n_h, n_w = H // 8, W // 8
    out = np.zeros((n_h, n_w), dtype=bool)
    for i in range(n_h):
        for j in range(n_w):
            se = sm_e[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8].mean()
            sn = sm_n[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8].mean()
            out[i, j] = (se > 0.5) and (sn <= 0.5)
    return out


def extract_adaptive_ra(watermarked, msg_len, delta, sm_embed, sm_new):
    """方向 A 解码器: 被移除块的投票权重置 0, 其余同 extract_adaptive."""
    H, W, n_h, n_w = _block_grid(*watermarked.shape[:2])
    y = y_channel(watermarked)[:H, :W]
    deff = _delta_eff_map(sm_embed, H, W, delta)
    removed = removed_blocks(sm_embed, sm_new, H, W)
    votes = [[] for _ in range(msg_len)]
    sm_r = cv2.resize(sm_embed.astype(np.float32), (W, H))
    for i in range(n_h):
        for j in range(n_w):
            bi = (i * n_w + j) % msg_len
            d = cv2.dct(y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8].astype(np.float32))
            bit = _demodulate_bit(d[4, 1], deff[i, j])
            w = 0.0 if removed[i, j] else sm_r[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8].mean()
            votes[bi].append((bit, w))
    return _majority_vote(votes, msg_len)


def extract_uniform_ra(watermarked, msg_len, delta, sm_embed, sm_new):
    """A2 消融解码器: uniform 嵌入 + 移除感知剔除 + 均匀权重.

    嵌入端不使用语义强度图 (固定 Δ); 提取端仅用强度图判定"被移除的实例块"
    (sm_embed>0.5 且 sm_new<=0.5), 这些块的投票权重置 0, 其余块权重均为 1
    (uniform 嵌入的块信噪比相同, 不应加权).
    用于验证: 语义强度图(嵌入端调制)是否有贡献, 还是 RA(解码端)独立起效.
    """
    H, W, n_h, n_w = _block_grid(*watermarked.shape[:2])
    y = y_channel(watermarked)[:H, :W]
    removed = removed_blocks(sm_embed, sm_new, H, W)
    votes = [[] for _ in range(msg_len)]
    for i in range(n_h):
        for j in range(n_w):
            bi = (i * n_w + j) % msg_len
            d = cv2.dct(y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8].astype(np.float32))
            bit = _demodulate_bit(d[4, 1], delta)
            w = 0.0 if removed[i, j] else 1.0
            votes[bi].append((bit, w))
    return _majority_vote(votes, msg_len)


# ── 方案注册表 ──────────────────────────────────────────────────
# 方案: (强度图生成器 or None, 解码器名)
#   解码器: 'uniform' | 'adaptive' | 'adaptive_ra'
SCHEMES = {
    'uniform':     (None, 'uniform'),
    'uniform_ra':  (lambda im: get_strength_map(im), 'uniform_ra'),
    'semantic':    (lambda im: get_strength_map(im), 'adaptive'),
    'semantic_ra': (lambda im: get_strength_map(im), 'adaptive_ra'),
    'jnd':         (lambda im: jnd_strength_map(im), 'adaptive'),
    'full':        (lambda im: full_strength_map(im), 'adaptive_ra'),
}


def embed_scheme(img, msg, delta, sm, dec):
    """按解码器决定嵌入方式: uniform_ra 嵌入端不用强度图 (A2 消融)."""
    if dec in ('uniform', 'uniform_ra'):
        return embed_uniform(img, msg, delta=delta)
    return embed_adaptive(img, msg, sm, delta=delta)


def decode_scheme(att, msg, delta, sm, dec, sm_new=None):
    if dec == 'uniform':
        return extract_uniform(att, msg_len=len(msg), delta=delta)
    if dec == 'uniform_ra':
        assert sm_new is not None
        return extract_uniform_ra(att, len(msg), delta, sm, sm_new)
    if dec == 'adaptive':
        return extract_adaptive(att, msg_len=len(msg), delta=delta, strength_map=sm)
    if dec == 'adaptive_ra':
        assert sm_new is not None
        return extract_adaptive_ra(att, len(msg), delta, sm, sm_new)
    raise ValueError(dec)


# ── 数据收集 ────────────────────────────────────────────────────
def collect(dirp, max_n=None):
    """递归收集目录下所有图片 (支持子目录结构, 如 CASIA Au/Tp)."""
    if not os.path.isdir(dirp):
        return []
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    fs = []
    for root, _, files in os.walk(dirp):
        for f in files:
            if f.lower().endswith(exts):
                fs.append(os.path.join(root, f))
    fs.sort()
    return fs[:max_n] if max_n else fs


def collect_source(key, max_n=None):
    """按 DATA_SOURCES 键收集 (支持 glob 通配符路径, 如 columbia-sp 的 Au-*)."""
    pat = DATA_SOURCES[key]
    if '*' in pat:
        fs = []
        for d in sorted(glob.glob(pat)):
            fs.extend(collect(d))
        fs.sort()
        return fs[:max_n] if max_n else fs
    return collect(pat, max_n)


# ── 实验主流程 ──────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100, help='每个数据集图片数 (默认 100)')
    ap.add_argument('--datasets', default='coco,kodak',
                    help='逗号分隔: coco,kodak,casia,casia-au,casia-tp,'
                         'columbia,columbia-au,columbia-sp')
    ap.add_argument('--deltas', default='13,15,17,19', help='Δ 扫描值')
    ap.add_argument('--msglen', type=int, default=64, help='扫描阶段消息长度')
    ap.add_argument('--scan-attacks', default='clean,jpeg75',
                    help='扫描阶段攻击 (逗号分隔): clean,jpeg95,jpeg75,jpeg50,'
                         'jpeg25,noise001,noise002,noise005,gblur1,gblur2')
    ap.add_argument('--rem-msglen', type=int, default=256, help='局部移除攻击的消息长度')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--imgsize', type=int, default=256,
                    help='嵌入/提取分辨率 (默认 256, 与论文协议一致)')
    ap.add_argument('--quick', action='store_true',
                    help='快速模式: 只跑方向 A (semantic_ra), 每集 20 张, 只测 removal')
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    t0 = time.time()
    deltas = [int(x) for x in args.deltas.split(',')]

    ds_paths = {}
    for name in args.datasets.split(','):
        name = name.strip()
        if name not in DATA_SOURCES:
            print(f'未知数据集: {name} (支持: {",".join(DATA_SOURCES)})')
            continue
        n = 20 if args.quick else args.n
        ds_paths[name] = collect_source(name, n)
        if not ds_paths[name]:
            print(f'警告: 数据集 {name} 无可用图片, 已跳过')
            del ds_paths[name]
    if not ds_paths:
        sys.exit('没有可用数据集, 请检查 --datasets 参数与 data 目录')
    for name, paths in ds_paths.items():
        print(f'数据集 {name}: {len(paths)} 张')

    print('=' * 72)
    print('  Idea Probe: 改进方向最小验证')
    print('  方向 A=移除感知投票  B=感知预算(JND)  C=A+B 组合')
    print('=' * 72)

    R = {'meta': {'n': args.n, 'quick': args.quick, 'datasets': args.datasets,
                  'deltas': deltas, 'msglen': args.msglen,
                  'rem_msglen': args.rem_msglen, 'seed': args.seed,
                  'ts': datetime.datetime.now().isoformat()}}

    # ── Step 0: 嵌入前后 YOLO 掩膜一致性 (方向 A 前提校验) ──
    print('\n[Step 0] 嵌入前后 YOLO 掩膜一致性 (方向 A 前提校验)')
    check_paths = (ds_paths['coco'][:20] if 'coco' in ds_paths
                   else list(ds_paths.values())[0][:20])
    ious = []
    p0 = Progress(len(check_paths), 'mask-IoU')
    for p in check_paths:
        img = load(p, args.imgsize)
        sm0 = get_strength_map(img)
        wm = embed_adaptive(img, rng.randint(0, 2, 64).astype(np.uint8), sm0, delta=15)
        sm1 = get_strength_map(wm)
        a = sm0 > 0.5
        b = sm1 > 0.5
        inter = (a & b).sum()
        union = (a | b).sum()
        ious.append(inter / (union + 1e-8) if union > 0 else 1.0)
        p0.update()
    p0.done()
    iou_mean = float(np.mean(ious))
    iou_std = float(np.std(ious))
    R['mask_consistency'] = {'iou_mean': iou_mean, 'iou_std': iou_std, 'n': len(ious)}
    verdict = 'PASS (前提成立, 方向 A 可行)' if iou_mean > 0.85 else 'WARN (掩膜受水印影响较大)'
    print(f'  嵌入前后掩膜 IoU = {iou_mean:.3f} ± {iou_std:.3f}  →  {verdict}')

    # ── Step 1: Δ 扫描 (clean + JPEG q75) ──
    if not args.quick:
        print('\n[Step 1] Δ 扫描: ' + ' / '.join(args.scan_attacks.split(','))
              + ' (等 PSNR 与等 SSIM 双协议)')
        atk_keys = [a.strip() for a in args.scan_attacks.split(',')]
        bad = [a for a in atk_keys if a not in SCAN_ATTACKS]
        if bad:
            sys.exit(f'未知攻击: {bad} (支持: {",".join(SCAN_ATTACKS)})')
        attacks1 = {a: SCAN_ATTACKS[a] for a in atk_keys}
        schemes1 = SCHEMES
        total1 = sum(len(ps) * len(schemes1) * len(deltas) * len(attacks1)
                     for ps in ds_paths.values())
        p1 = Progress(total1, 'Δ-scan')
        scan = {}
        for dname, paths in ds_paths.items():
            scan[dname] = {}
            for atk_name, atk_fn in attacks1.items():
                # 累积器: sname -> Δ -> 列表; 消息按图固定 (同图同消息跨方案, paired 比较)
                acc = {sname: {str(d): {'ba': [], 'ps': [], 'ss': []}
                               for d in deltas} for sname in schemes1}
                for p in paths:
                    img = load(p, args.imgsize)
                    msg = rng.randint(0, 2, args.msglen).astype(np.uint8)
                    for sname, (sm_fn, dec) in schemes1.items():
                        sm = None if sm_fn is None else sm_fn(img)
                        for d in deltas:
                            wm = embed_scheme(img, msg, d, sm, dec)
                            att = wm if atk_fn is None else atk_fn(wm)
                            if att is None:
                                p1.update()
                                continue
                            if dec in ('adaptive_ra', 'uniform_ra'):
                                me = decode_scheme(att, msg, d, sm, dec,
                                                   sm_new=sm_fn(att))
                            else:
                                me = decode_scheme(att, msg, d, sm, dec)
                            acc[sname][str(d)]['ba'].append(bit_accuracy(msg, me))
                            acc[sname][str(d)]['ps'].append(psnr(img, wm))
                            acc[sname][str(d)]['ss'].append(ssim(img, wm))
                            p1.update()
                scan[dname][atk_name] = {
                    sname: {d: {'ba': float(np.mean(v['ba'])),
                                'ba_std': float(np.std(v['ba'])),
                                'psnr': float(np.mean(v['ps'])),
                                'ssim': float(np.mean(v['ss'])),
                                'n': len(v['ba'])}
                            for d, v in cells.items()}
                    for sname, cells in acc.items()}
        p1.done()
        R['scan'] = scan

    # ── Step 2: 局部实例移除 (rem-msglen bit, 与论文协议一致) ──
    print('\n[Step 2] 局部实例移除攻击 30%/50%')
    rem_schemes = {k: SCHEMES[k] for k in
                   ['uniform', 'uniform_ra', 'semantic', 'semantic_ra', 'full']}
    total2 = sum(len(ps) * len(rem_schemes) * 2 for ps in ds_paths.values())
    p2 = Progress(total2, 'removal')
    rem = {}
    for dname, paths in ds_paths.items():
        rem[dname] = {}
        for frac, key in [(0.3, 'rem30'), (0.5, 'rem50')]:
            # 图级累积 + 对齐: 每张图记录 4 方案的 (ba, psnr, ssim);
            # 最后只保留 4 方案都成功的图 → n 严格一致, 可直接 paired 检验
            per_img = []
            for p in paths:
                img = load(p, args.imgsize)
                msg = rng.randint(0, 2, args.rem_msglen).astype(np.uint8)
                m_total = detect_removal_mask(img, frac)
                row = {}
                for sname, (sm_fn, dec) in rem_schemes.items():
                    if m_total is None:
                        row[sname] = None
                        p2.update()
                        continue
                    sm = None if sm_fn is None else sm_fn(img)
                    d = 15 if sname not in ('uniform', 'uniform_ra') else 19  # 等 PSNR 匹配
                    wm = embed_scheme(img, msg, d, sm, dec)
                    att = apply_removal(wm, m_total, bg_img=img)
                    if dec in ('adaptive_ra', 'uniform_ra'):
                        me = decode_scheme(att, msg, d, sm, dec, sm_new=sm_fn(att))
                    else:
                        me = decode_scheme(att, msg, d, sm, dec)
                    row[sname] = (bit_accuracy(msg, me), psnr(img, wm), ssim(img, wm))
                    p2.update()
                if m_total is not None:
                    per_img.append(row)
            # 对齐: 丢弃任一方案失败的图
            aligned = [r for r in per_img if all(v is not None for v in r.values())]
            rem[dname][key] = {
                sname: {'ba': float(np.mean([r[sname][0] for r in aligned])),
                        'ba_std': float(np.std([r[sname][0] for r in aligned])),
                        'psnr': float(np.mean([r[sname][1] for r in aligned])),
                        'ssim': float(np.mean([r[sname][2] for r in aligned])),
                        'n': len(aligned),
                        # per-image BA 数组 (aligned, 可直接做 paired t-test / Wilcoxon)
                        'ba_per_image': [float(r[sname][0]) for r in aligned]}
                for sname in rem_schemes}
    p2.done()
    R['removal'] = rem

    # ── Step 3: 判据汇总 ──
    print('\n' + '=' * 72)
    print('  判据汇总 (方向 A 成立: semantic_ra/full/uniform_ra ≥ uniform +3pt)')
    print('=' * 72)
    summary = {}
    for dname in rem:
        for key in ['rem30', 'rem50']:
            u = rem[dname][key]['uniform']['ba']
            ura = rem[dname][key]['uniform_ra']['ba']
            se = rem[dname][key]['semantic']['ba']
            ra = rem[dname][key]['semantic_ra']['ba']
            f = rem[dname][key]['full']['ba']
            ok = (ra - u) >= 0.03 or (f - u) >= 0.03 or (ura - u) >= 0.03
            print(f'  [{dname}/{key}] uniform={u*100:5.1f}%  uniform_ra={ura*100:5.1f}%  '
                  f'semantic={se*100:5.1f}%  semantic_ra={ra*100:5.1f}%  full={f*100:5.1f}%'
                  + ('   ← A 成立' if ok else ''))
            summary[f'{dname}_{key}'] = {'uniform': u, 'uniform_ra': ura,
                                         'semantic': se,
                                         'semantic_ra': ra, 'full': f,
                                         'A_ok': bool(ok),
                                         # A2 消融判定: uniform_ra 提升说明 RA 独立起效
                                         'RA_alone_ok': bool((ura - u) >= 0.03)}
    if not args.quick:
        for dname in scan:
            for atk in scan[dname]:  # 遍历实际运行的攻击 (支持自定义 --scan-attacks)
                row = {}
                for sname in SCHEMES:
                    best = None
                    for d, v in scan[dname][atk][sname].items():
                        if v['n'] == 0:
                            continue
                        score = abs(v['psnr'] - 48.8) + abs(v['ssim'] - 0.994)
                        if best is None or score < best[0]:
                            best = (score, v)
                    if best:
                        row[sname] = best[1]
                summary[f'{dname}_{atk}_best_point'] = row
    R['summary'] = summary
    R['total_time_s'] = round(time.time() - t0, 1)

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(RESULTS_DIR, f'idea_probe_{ts}.json')
    with open(out, 'w') as f:
        json.dump(R, f, indent=2, default=float)
    tt = R['total_time_s']
    print(f'\n结果已保存: {out}  (总耗时 {tt:.0f}s = '
          f'{int(tt // 3600)}h{int(tt % 3600 // 60)}m{int(tt % 60)}s)')
    print('下一步: 打开 JSON 查看 mask_consistency / scan / removal / summary,')
    print('对照 IDEA_PROBE.md 中的判据判断方向 A/B/C 是否成立.')


if __name__ == '__main__':
    main()
