#!/usr/bin/env python3
"""JPEG 后压缩级联实验 (回应审稿意见: "编辑后再存一次 JPEG 更真实").

动机: 真实平台流程通常是 "语义编辑 -> 再压缩存储/传输". 我们的主实验只测了
      编辑本身, 没有叠加后处理压缩. 本实验检验: 在编辑后再加一级 JPEG 压缩,
      防御是否仍然有效 (以及经典鲁棒性是否仍成立).

设计 (沿用固定攻击集, 保证可比):
  对每个样本: 嵌入 -> 贴回编辑区域 -> JPEG 压缩(q) -> 解码
  q ∈ {95, 85, 75, 65}  (高质量到中等质量)
  同时报告 uniform 与 v2, 以及相对"无 JPEG"的衰减.

输出: ../results/jpeg_cascade_<ts>.json
用法: cd code && python run_jpeg_cascade.py
"""
import sys, os, json, argparse, datetime, io as _io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image

from run_signal_ra import embed_signal, decode_signal
from dct_watermark import bit_accuracy

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AS = f'{BASE}/results/attackset_20260817_082819'


def crop8(a):
    h, w = a.shape[:2]
    return a[:h - h % 8, :w - w % 8]


def paste_edit(wm, att, bbox01):
    H, W = wm.shape[:2]
    x1, y1, x2, y2 = bbox01
    bx1, by1 = int(round(x1 * W)), int(round(y1 * H))
    bx2, by2 = int(round(x2 * W)), int(round(y2 * H))
    out = wm.copy()
    out[by1:by2, bx1:bx2] = att[by1:by2, bx1:bx2]
    return out


def jpeg(img, q):
    """用 PIL 做一次 JPEG 编解码 (模拟平台压缩)."""
    buf = _io.BytesIO()
    Image.fromarray(img).save(buf, format='JPEG', quality=q, subsampling=0)
    buf.seek(0)
    return np.array(Image.open(buf).convert('RGB'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--msglen', type=int, default=256)
    ap.add_argument('--delta', type=int, default=15)
    ap.add_argument('--quality', default='95,85,75,65')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    qs = [int(x) for x in args.quality.split(',')]
    meta = json.load(open(f'{AS}/meta.json'))
    pairs = []
    for s in meta['samples']:
        try:
            o = crop8(np.array(Image.open(f'{AS}/images/orig_{s["id"]}.png').convert('RGB')))
            a = crop8(np.array(Image.open(f'{AS}/images/att_{s["id"]}.png').convert('RGB')))
        except FileNotFoundError:
            continue
        pairs.append((s, o, a))
    rng = np.random.RandomState(args.seed)
    msg = rng.randint(0, 2, args.msglen).astype(np.uint8)
    print(f'JPEG 级联: n={len(pairs)}, msglen={args.msglen}, quality={qs}')

    out = {'meta': {'n': len(pairs), 'msglen': args.msglen, 'delta': args.delta,
                    'quality': qs, 'ts': datetime.datetime.now().isoformat()},
           'summary': {}}

    # 基准: 无 JPEG
    base_u, base_v = [], []
    for s, o, a in pairs:
        wm = embed_signal(o, msg, args.delta)
        att_wm = paste_edit(wm, a, s['bbox'])
        me, _ = decode_signal(att_wm, args.msglen, args.delta, use_check=False)
        base_u.append(float(bit_accuracy(msg, me)))
        me, _ = decode_signal(att_wm, args.msglen, args.delta, use_check=True,
                              soft=True, dilate=1)
        base_v.append(float(bit_accuracy(msg, me)))
    out['summary']['no_jpeg'] = {'uniform': float(np.mean(base_u)),
                                 'v2': float(np.mean(base_v))}
    print(f"\n  无 JPEG         : uniform {np.mean(base_u)*100:5.1f}%  "
          f"v2 {np.mean(base_v)*100:5.1f}%")

    for q in qs:
        us, vs = [], []
        for s, o, a in pairs:
            wm = embed_signal(o, msg, args.delta)
            att_wm = paste_edit(wm, a, s['bbox'])
            comp = jpeg(att_wm, q)
            me, _ = decode_signal(comp, args.msglen, args.delta, use_check=False)
            us.append(float(bit_accuracy(msg, me)))
            me, _ = decode_signal(comp, args.msglen, args.delta, use_check=True,
                                  soft=True, dilate=1)
            vs.append(float(bit_accuracy(msg, me)))
        out['summary'][f'jpeg{q}'] = {'uniform': float(np.mean(us)), 'v2': float(np.mean(vs))}
        print(f"  attack + JPEG q={q:<3}: uniform {np.mean(us)*100:5.1f}%  "
              f"v2 {np.mean(vs)*100:5.1f}%")

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/jpeg_cascade_{ts}.json'

    def to_py(o):
        if isinstance(o, dict): return {k: to_py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [to_py(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open(fp, 'w') as f:
        f.write(json.dumps(to_py(out), indent=1))
    print(f'\n已保存: {fp}')


if __name__ == '__main__':
    main()
