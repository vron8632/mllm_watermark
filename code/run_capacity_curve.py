#!/usr/bin/env python3
"""容量-鲁棒性曲线 (capacity vs robustness) 实验.

动机: 审稿意见指出 Table 4 里我们以 256-bit 对比 Watermark Anything 的 32-bit,
"不是同条件". 本实验回答: 固定同一批攻击图, 只改变消息长度, 我们的方法在不同
容量下表现如何 —— 从而把 "capacity 不可比" 变成一条可读的曲线.

设计要点 (保证可比性):
  1. **固定攻击**: 复用 results/attackset_20260817_082819 已缓存的 (orig, att) 配对,
     不重新调用 MLLM/扩散. 因此所有 msglen 档位面对**完全相同的攻击**.
  2. **各自嵌入**: 每个档位用各自长度的随机消息嵌入原图, 再把已缓存的攻击区域
     贴回水印图 (与论文发布协议一致: bbox 外保持水印图, bbox 内用攻击图).
  3. 指标: 攻击后 BA (uniform / v2), 以及 clean BA 与 PSNR.

输出: ../results/capacity_curve_<ts>.json
用法: cd code && python run_capacity_curve.py [--msglens 32,64,128,256] [--delta 15]
"""
import sys, os, json, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image

from run_signal_ra import embed_signal, decode_signal, CHECK_COEF, PAY_COEF
from dct_watermark import bit_accuracy

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AS = f'{BASE}/results/attackset_20260817_082819'


def crop8(a):
    h, w = a.shape[:2]
    return a[:h - h % 8, :w - w % 8]


def psnr(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    return float('inf') if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def paste_edit(wm, att, bbox01):
    """论文发布协议: bbox 内用攻击图, bbox 外保持水印图."""
    H, W = wm.shape[:2]
    x1, y1, x2, y2 = bbox01
    bx1, by1 = int(round(x1 * W)), int(round(y1 * H))
    bx2, by2 = int(round(x2 * W)), int(round(y2 * H))
    out = wm.copy()
    out[by1:by2, bx1:bx2] = att[by1:by2, bx1:bx2]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--msglens', default='32,64,128,256')
    ap.add_argument('--delta', type=int, default=15)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    meta = json.load(open(f'{AS}/meta.json'))
    samples = meta['samples']
    msglens = [int(x) for x in args.msglens.split(',')]

    print(f'容量曲线: n={len(samples)} 图, msglen={msglens}, Δ={args.delta}')
    print(f'  数据源: {AS} (imgsize={meta.get("imgsize")}, 复用已缓存攻击)')

    R = {'meta': {'n': len(samples), 'msglens': msglens, 'delta': args.delta,
                  'attackset': AS, 'imgsize': meta.get('imgsize'),
                  'ts': datetime.datetime.now().isoformat()},
         'per_image': [], 'summary': {}}

    # 预加载 (orig, att), 保证各档位用同一批图
    pairs = []
    for s in samples:
        try:
            o = crop8(np.array(Image.open(f'{AS}/images/orig_{s["id"]}.png').convert('RGB')))
            a = crop8(np.array(Image.open(f'{AS}/images/att_{s["id"]}.png').convert('RGB')))
        except FileNotFoundError:
            continue
        pairs.append((s, o, a))
    print(f'  实际可用配对: {len(pairs)}')

    for mlen in msglens:
        rng = np.random.RandomState(args.seed)
        rows = []
        for s, o, a in pairs:
            msg = rng.randint(0, 2, mlen).astype(np.uint8)
            wm = embed_signal(o, msg, args.delta)
            row = {'img': s['id'], 'msglen': mlen, 'psnr': psnr(o, wm)}
            # clean
            me, _ = decode_signal(wm, mlen, args.delta, use_check=False)
            row['clean_uniform_ba'] = float(bit_accuracy(msg, me))
            # 攻击后 (同一张缓存攻击图)
            att_wm = paste_edit(wm, a, s['bbox'])
            me, _ = decode_signal(att_wm, mlen, args.delta, use_check=False)
            row['att_uniform_ba'] = float(bit_accuracy(msg, me))
            me, _ = decode_signal(att_wm, mlen, args.delta, use_check=True,
                                  soft=True, dilate=1)
            row['att_v2_ba'] = float(bit_accuracy(msg, me))
            rows.append(row)
        R['per_image'] += rows
        u = float(np.mean([r['att_uniform_ba'] for r in rows]))
        v = float(np.mean([r['att_v2_ba'] for r in rows]))
        c = float(np.mean([r['clean_uniform_ba'] for r in rows]))
        p = float(np.mean([r['psnr'] for r in rows]))
        R['summary'][str(mlen)] = {
            'n': len(rows), 'att_uniform_ba': u, 'att_v2_ba': v,
            'gain_pt': (v - u) * 100, 'clean_ba': c, 'psnr': p}
        print(f'  msglen={mlen:>3}: clean={c*100:.1f}%  攻击后 uniform={u*100:.1f}%  '
              f'v2={v*100:.1f}%  gain=+{(v-u)*100:.1f}pt  PSNR={p:.1f}dB')

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = f'{BASE}/results/capacity_curve_{ts}.json'

    def to_py(o):
        """递归把 numpy 标量/数组转成 python 原生类型, 保证 JSON 可序列化."""
        if isinstance(o, dict):
            return {k: to_py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [to_py(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    # 先完整序列化到内存, 成功后再落盘 —— 避免序列化失败留下半截文件
    payload = json.dumps(to_py(R), indent=1)
    with open(out, 'w') as f:
        f.write(payload)
    print(f'\n已保存: {out}')


if __name__ == '__main__':
    main()
