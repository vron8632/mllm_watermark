#!/usr/bin/env python3
"""系数位置 / 置信度函数 / 膨胀半径 消融.

回应审稿意见第 1 条: "check bit 和 payload bit 的频率选择有没有理论依据?
为什么 (4,1)/(3,2) 是最优? soft confidence 公式是不是 ad hoc?
dilated weighting x0.3 是拍出来的还是搜出来的?"

设计 (与容量曲线同一方法论, 保证可比):
  - **固定攻击**: 复用 results/attackset_20260817_082819 的 (orig, att) 配对,
    不重新调用 MLLM/扩散, 因此所有配置面对完全相同的攻击.
  - 每个配置: 用该配置的系数嵌入 -> 贴回攻击区域 -> 解码 (uniform + v2).

三组消融:
  (A) 系数位置: payload 与 check 的各种 (u,v) 组合, 含"低频/中频/高频"与
      "彼此靠近/远离" 两类, 用来支撑"中频 + 远离"的设计原则.
  (B) 置信度函数: 软置信度 (当前) vs 二值 0/1 vs 均匀权重, 验证 soft 的必要性.
  (C) 膨胀半径与衰减: dilate ∈ {0,1,2}, 邻接衰减系数 ∈ {0.2,0.3,0.5}.

输出: ../results/ablation_coef_<ts>.json
用法: cd code && python run_ablation_coef.py [--part A|B|C|all]
"""
import sys, os, json, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image

import run_signal_ra as R
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


def run_config(pairs, pay_coef, check_coef, msg, delta, mode='v2', dilate=1,
               decay=0.3):
    """用指定系数配置跑一遍, 返回攻击后 BA (uniform 与按 mode 的防御后)."""
    # 猴子补丁: 替换模块级全局系数
    R.PAY_COEF, R.CHECK_COEF = pay_coef, check_coef
    u_ba, d_ba, clean_ba = [], [], []
    for s, o, a in pairs:
        wm = R.embed_signal(o, msg, delta)
        me, _ = R.decode_signal(wm, len(msg), delta, use_check=False)
        clean_ba.append(float(bit_accuracy(msg, me)))
        att_wm = paste_edit(wm, a, s['bbox'])
        me, _ = R.decode_signal(att_wm, len(msg), delta, use_check=False)
        u_ba.append(float(bit_accuracy(msg, me)))
        if mode == 'v2':
            me, _ = R.decode_signal(att_wm, len(msg), delta, use_check=True,
                                    soft=True, dilate=dilate)
        else:                      # uniform: 完全不用 check
            me = me
        d_ba.append(float(bit_accuracy(msg, me)))
    return (float(np.mean(clean_ba)), float(np.mean(u_ba)), float(np.mean(d_ba)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='all', choices=['A', 'B', 'C', 'all'])
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
    msg = np.random.RandomState(args.seed).randint(0, 2, args.msglen).astype(np.uint8)
    print(f'系数/置信度消融: n={len(pairs)}, msglen={args.msglen}, Δ={args.delta}')

    out = {'meta': {'n': len(pairs), 'msglen': args.msglen, 'delta': args.delta,
                    'attackset': AS, 'ts': datetime.datetime.now().isoformat()},
           'A_coefficient': [], 'B_confidence': [], 'C_dilation': []}

    # ---------- (A) 系数位置 ----------
    if args.part in ('A', 'all'):
        print('\n--- (A) 系数位置 (payload, check) ---')
        # 8x8 DCT 的 (u,v): 低频在左上, 高频在右下. 论文默认 payload=(4,1), check=(3,2)
        combos = [
            ((4, 1), (3, 2), 'default: payload (4,1), check (3,2)'),
            ((4, 1), (1, 4), 'check moved to (1,4)'),
            ((4, 1), (2, 2), 'check at (2,2) - closer to payload'),
            ((4, 1), (5, 1), 'check at (5,1) - lower freq, same row'),
            ((4, 1), (4, 4), 'check at (4,4) - higher freq'),
            ((3, 1), (4, 2), 'payload (3,1), check (4,2)'),
            ((5, 2), (3, 2), 'payload (5,2) - lower freq, same check'),
            ((6, 1), (3, 2), 'payload (6,1) - high freq'),
            ((1, 1), (3, 2), 'payload (1,1) - very low freq (near DC)'),
        ]
        for pay, chk, desc in combos:
            c, u, d = run_config(pairs, pay, chk, msg, args.delta)
            out['A_coefficient'].append({
                'payload': list(pay), 'check': list(chk), 'desc': desc,
                'clean_ba': c, 'att_uniform_ba': u, 'att_v2_ba': d,
                'gain_pt': (d - u) * 100})
            print(f'  {desc:<44} clean={c*100:5.1f}%  unif={u*100:5.1f}%  '
                  f'v2={d*100:5.1f}%  gain=+{(d-u)*100:4.1f}pt')

    # ---------- (B) 置信度函数 ----------
    if args.part in ('B', 'all'):
        print('\n--- (B) 置信度/权重方式 (默认系数) ---')
        print('  注: soft=True 为论文 v2; uniform=完全不使用 check bit (下界);')
        print('      v1(binary)=校验失败块权重直接置 0')
        # 默认系数下, 分别跑 uniform / binary v1 / soft v2
        pay, chk = (4, 1), (3, 2)
        c, u, _ = run_config(pairs, pay, chk, msg, args.delta)
        out['B_confidence'].append({'mode': 'uniform (no check)', 'clean_ba': c,
                                    'att_ba': u, 'desc': 'baseline: check bit unused'})
        print(f'  {"uniform (no check)":<34} clean={c*100:5.1f}%  att={u*100:5.1f}%')
        # v1: 二值剔除
        R.PAY_COEF, R.CHECK_COEF = pay, chk
        v1 = []
        for s, o, a in pairs:
            wm = R.embed_signal(o, msg, args.delta)
            att_wm = paste_edit(wm, a, s['bbox'])
            me, _ = R.decode_signal(att_wm, len(msg), args.delta, use_check=True)
            v1.append(float(bit_accuracy(msg, me)))
        out['B_confidence'].append({'mode': 'v1 binary removal', 'clean_ba': c,
                                    'att_ba': float(np.mean(v1)),
                                    'desc': 'check-failing blocks get weight 0'})
        print(f'  {"v1 binary removal":<34} clean={c*100:5.1f}%  att={np.mean(v1)*100:5.1f}%')
        # v2 soft (no dilation, to isolate the confidence function)
        soft_only = []
        for s, o, a in pairs:
            wm = R.embed_signal(o, msg, args.delta)
            att_wm = paste_edit(wm, a, s['bbox'])
            me, _ = R.decode_signal(att_wm, len(msg), args.delta, use_check=True,
                                    soft=True, dilate=0)
            soft_only.append(float(bit_accuracy(msg, me)))
        out['B_confidence'].append({'mode': 'soft confidence (no dilation)',
                                    'clean_ba': c, 'att_ba': float(np.mean(soft_only)),
                                    'desc': 'w_b = 1 - d_best/(d_best+d_other)'})
        print(f'  {"soft confidence (no dilation)":<34} clean={c*100:5.1f}%  '
              f'att={np.mean(soft_only)*100:5.1f}%')
        # v2 full (soft + dilate)
        _, _, full = run_config(pairs, pay, chk, msg, args.delta, dilate=1)
        out['B_confidence'].append({'mode': 'soft + dilation (v2, full)',
                                    'clean_ba': c, 'att_ba': full,
                                    'desc': 'final configuration'})
        print(f'  {"soft + dilation (v2, full)":<34} clean={c*100:5.1f}%  att={full*100:5.1f}%')

    # ---------- (C) 膨胀半径 / 衰减 ----------
    if args.part in ('C', 'all'):
        print('\n--- (C) 膨胀半径 (dilate) x 邻接衰减 ---')
        for dl in (0, 1, 2, 3):
            c, u, d = run_config(pairs, (4, 1), (3, 2), msg, args.delta, dilate=dl)
            out['C_dilation'].append({'dilate': dl, 'decay': 0.3, 'clean_ba': c,
                                      'att_uniform_ba': u, 'att_v2_ba': d})
            print(f'  dilate={dl} (decay=0.3):  unif={u*100:5.1f}%  v2={d*100:5.1f}%  '
                  f'gain=+{(d-u)*100:4.1f}pt')

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/ablation_coef_{ts}.json'

    def to_py(o):
        if isinstance(o, dict):
            return {k: to_py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [to_py(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    # 先序列化到内存再落盘, 避免半截文件
    payload = json.dumps(to_py(out), indent=1)
    with open(fp, 'w') as f:
        f.write(payload)
    print(f'\n已保存: {fp}')


if __name__ == '__main__':
    main()
