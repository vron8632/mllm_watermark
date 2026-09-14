#!/usr/bin/env python3
"""ECC 基线对照 (回应审稿意见最狠的一条: "check bit 是否只是另一种冗余?").

质疑: 训练-free 路线里, 用 check bit 做软加权 是否比 "重复嵌入 + 纠错码(ECC)" 更好?
      如果 BCH/Hamming 也能追平, 那 check bit 就不是新机制.

本实验在同一嵌入/攻击条件下对比三种冗余策略 (256-bit, Δ=15, 固定攻击集):
  (A) uniform            : 循环重复 + 多数投票 (无任何额外冗余) —— 下界
  (B) ECC (BCH-like)     : 同样循环重复, 但对 payload 加纠错编码 (Hamming(7,4) 简化版)
                           解码时按码字纠错后再投票
  (C) ours (v2)          : 每块额外嵌 1 个 check bit, 按校验置信度软加权 + 膨胀

关键判据: 在**相同冗余开销**下比较. 注意检查位本身占用了嵌入容量, 因此
          (B) 的 ECC 需要把 code rate 调到与 (C) 的可比位置.

输出: ../results/ecc_vs_checkbit_<ts>.json
用法: cd code && python run_ecc_vs_checkbit.py
"""
import sys, os, json, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image

from run_signal_ra import embed_signal, decode_signal, check_bits, PAY_COEF, CHECK_COEF
from dct_watermark import bit_accuracy, _block_grid, _demodulate_bit

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


# ---------- Hamming(7,4): 经典 (7,4) 码, 1-bit 纠错 ----------
H74_G = np.array([[1,0,0,0,1,1,1],
                  [0,1,0,0,1,1,0],
                  [0,0,1,0,1,0,1],
                  [0,0,0,1,0,1,1]], dtype=np.uint8)   # 生成矩阵
H74_H = np.array([[1,1,1,0,1,0,0],
                  [1,1,0,1,0,1,0],
                  [1,0,1,1,0,0,1]], dtype=np.uint8)   # 校验矩阵


def hamming_encode(bits):
    """每 4 bit -> 7 bit 码字 (系统码)."""
    out = []
    for i in range(0, len(bits) - len(bits) % 4, 4):
        blk = np.array(bits[i:i+4], dtype=np.uint8)
        out.extend((blk @ H74_G) % 2)
    return np.array(out, dtype=np.uint8)


def hamming_decode(recv):
    """每 7 bit 做 syndrome 纠错 (最多纠 1 位)."""
    out = []
    for i in range(0, len(recv) - len(recv) % 7, 7):
        blk = np.array(recv[i:i+7], dtype=np.uint8)
        syn = (H74_H @ blk) % 2
        idx = syn[0]*4 + syn[1]*2 + syn[2]     # 综合位 -> 出错位置
        if idx > 0 and idx <= 7:
            blk[idx-1] ^= 1
        out.extend(blk[:4])                    # 系统位在前
    return np.array(out, dtype=np.uint8)


def embed_with_ecc(img, msg, delta):
    """把 msg 用 Hamming(7,4) 编码后, 再用与论文相同的循环 QIM 嵌入."""
    coded = hamming_encode(msg)
    return embed_signal(img, coded, delta), coded


def decode_with_ecc(wm_or_att, coded_len, msg_len, delta):
    """解码 -> 纠错 -> 截断回 msg_len (与 uniform 同协议)."""
    from run_signal_ra import _majority_vote
    y = cv2.cvtColor(wm_or_att, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)
    H, W, n_h, n_w = _block_grid(*wm_or_att.shape[:2])
    y = y[:H, :W]
    votes = [[] for _ in range(coded_len)]
    for i in range(n_h):
        for j in range(n_w):
            d = cv2.dct(y[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32))
            bi = (i * n_w + j) % coded_len
            votes[bi].append(_demodulate_bit(d[PAY_COEF], delta))
    recv = np.array([int(np.mean(v) > 0.5) if v else 0 for v in votes], dtype=np.uint8)
    fixed = hamming_decode(recv)
    return fixed[:msg_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--msglen', type=int, default=256)
    ap.add_argument('--delta', type=int, default=15)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

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
    coded_len = (args.msglen // 4) * 7
    print(f'ECC 对照: n={len(pairs)}, msglen={args.msglen} -> hamming coded={coded_len}')

    res = {'meta': {'n': len(pairs), 'msglen': args.msglen, 'delta': args.delta,
                    'coded_len': coded_len, 'ts': datetime.datetime.now().isoformat()},
           'A_uniform': [], 'B_ecc': [], 'C_ours': []}

    for s, o, a in pairs:
        # (A) uniform: 循环重复 + 投票
        wm = embed_signal(o, msg, args.delta)
        att_wm = paste_edit(wm, a, s['bbox'])
        me, _ = decode_signal(att_wm, args.msglen, args.delta, use_check=False)
        res['A_uniform'].append(float(bit_accuracy(msg, me)))

        # (B) ECC: Hamming(7,4) + 同样循环重复
        wm_e, coded = embed_with_ecc(o, msg, args.delta)
        att_e = paste_edit(wm_e, a, s['bbox'])
        fixed = decode_with_ecc(att_e, coded_len, args.msglen, args.delta)
        res['B_ecc'].append(float(bit_accuracy(msg, fixed)))

        # (C) ours: check bit 软加权 + 膨胀
        me, _ = decode_signal(att_wm, args.msglen, args.delta, use_check=True,
                              soft=True, dilate=1)
        res['C_ours'].append(float(bit_accuracy(msg, me)))

    sA = float(np.mean(res['A_uniform']))
    sB = float(np.mean(res['B_ecc']))
    sC = float(np.mean(res['C_ours']))
    print(f"\n=== 结果 (同一攻击集, msglen={args.msglen}) ===")
    print(f"  (A) uniform (重复+投票, 无冗余)   {sA*100:5.1f}%")
    print(f"  (B) ECC (Hamming(7,4) + 重复)      {sB*100:5.1f}%   "
          f"(vs A: {(sB-sA)*100:+.1f}pt, 码率 4/7)")
    print(f"  (C) ours (check bit 软加权+膨胀)   {sC*100:5.1f}%   "
          f"(vs A: {(sC-sA)*100:+.1f}pt)")
    print()
    print(f"  结论: check bit {'优于' if sC > sB else '不如' if sC < sB else '等于'} ECC "
          f"({(sC-sB)*100:+.1f}pt)")

    res['summary'] = {'A_uniform': sA, 'B_ecc': sB, 'C_ours': sC,
                      'ecc_gain_pt': (sB - sA) * 100, 'ours_gain_pt': (sC - sA) * 100,
                      'ours_minus_ecc_pt': (sC - sB) * 100}

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/ecc_vs_checkbit_{ts}.json'

    def to_py(o):
        if isinstance(o, dict): return {k: to_py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [to_py(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open(fp, 'w') as f:
        f.write(json.dumps(to_py(res), indent=1))
    print(f'\n已保存: {fp}')


if __name__ == '__main__':
    main()
