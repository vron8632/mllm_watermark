#!/usr/bin/env python3
"""在已发布的 MLLM 语义编辑攻击集上重算 BA (论文协议, 无需 MLLM/SD API).

背景: 图例 (fig2/fig3) 展示的是 attackset_20260817_082819 的图像, 但此前
make_figures.py 引用的 BA 数字来自 signal_ra_20260817_071246 (另一批编辑指令),
两批实验编辑不同 → 图与数字不对应. 本脚本在 attackset 本身上按论文协议重算
uniform / v1 / soft / v2 / oracle 的逐图 BA, 保证"图例显示的图像 = 数字来源".

协议 (与 run_build_attackset.py 发布说明一致):
  嵌入水印 → 用 att 的 bbox 区域替换水印图对应区域 (bbox 外保持水印图) → 提取.
- msg = RandomState(42).randint(0,2,256), Δ=15, payload=(4,1), check=(3,2), key seed 42
- oracle = bbox_ra (块中心落在 mimo bbox 内 → 剔除), 与主实验一致

用法:
  cd code && python eval_attackset.py [--attackset results/attackset_20260817_082819]
输出: ../results/attackset_eval_<ts>.json  (含逐图 + summary, 供 make_figures.py 使用)
"""
import sys, os, json, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image
from run_idea_probe import load, psnr, bit_accuracy, y_channel, _block_grid, _demodulate_bit
from run_signal_ra import (embed_signal, decode_signal, CHECK_SEED, CHECK_COEF, PAY_COEF)
from run_mllm_benchmark import extract_uniform_bbox_ra

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def paste_edit(wm, att, bbox01):
    """把攻击后图的 bbox 区域贴回水印图 (bbox 外保持水印图) — 论文发布协议."""
    H, W = wm.shape[:2]
    x1, y1, x2, y2 = bbox01
    bx1, by1 = int(round(x1 * W)), int(round(y1 * H))
    bx2, by2 = int(round(x2 * W)), int(round(y2 * H))
    out = wm.copy()
    out[by1:by2, bx1:bx2] = att[by1:by2, bx1:bx2]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attackset', default=f'{BASE}/results/attackset_20260817_082819')
    ap.add_argument('--msglen', type=int, default=256)
    ap.add_argument('--delta', type=int, default=15)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.attackset, 'meta.json')))
    img_dir = os.path.join(args.attackset, 'images')
    rng = np.random.RandomState(42)
    msg = rng.randint(0, 2, args.msglen).astype(np.uint8)

    R = {'meta': {'attackset': os.path.basename(args.attackset),
                  'msglen': args.msglen, 'delta': args.delta,
                  'check_coef': CHECK_COEF, 'pay_coef': PAY_COEF,
                  'ts': datetime.datetime.now().isoformat()},
         'per_image': [], 'summary': {}}

    for s in meta['samples']:
        cid = s['id']
        orig = np.array(Image.open(os.path.join(img_dir, f'orig_{cid}.png')).convert('RGB'))
        att = np.array(Image.open(os.path.join(img_dir, f'att_{cid}.png')).convert('RGB'))
        # 保证 8 的倍数 (attackset 已是 256)
        orig = orig[:orig.shape[0]-orig.shape[0]%8, :orig.shape[1]-orig.shape[1]%8]
        att = att[:att.shape[0]-att.shape[0]%8, :att.shape[1]-att.shape[1]%8]

        row = {'id': cid, 'img': s['source'], 'instruction': s['instruction'],
               'bbox': s['bbox']}
        try:
            wm = embed_signal(orig, msg, args.delta)
            row['psnr'] = psnr(orig, wm)
            me, nf = decode_signal(wm, args.msglen, args.delta, use_check=True)
            row['clean_ba'] = bit_accuracy(msg, me)
            row['clean_fail_frac'] = nf / ((orig.shape[0]//8) * (orig.shape[1]//8))

            att_wm = paste_edit(wm, att, s['bbox'])
            me, _ = decode_signal(att_wm, args.msglen, args.delta, use_check=False)
            row['uniform_ba'] = bit_accuracy(msg, me)
            me, nf = decode_signal(att_wm, args.msglen, args.delta, use_check=True)
            row['signal_ra_ba'] = bit_accuracy(msg, me)
            me, _ = decode_signal(att_wm, args.msglen, args.delta, use_check=True, soft=True)
            row['signal_soft_ba'] = bit_accuracy(msg, me)
            me, _ = decode_signal(att_wm, args.msglen, args.delta, use_check=True,
                                  soft=True, dilate=1)
            row['signal_soft_dilate_ba'] = bit_accuracy(msg, me)
            row['att_fail_frac'] = nf / ((orig.shape[0]//8) * (orig.shape[1]//8))
            me = extract_uniform_bbox_ra(att_wm, args.msglen, args.delta, s['bbox'])
            row['bbox_ra_ba'] = bit_accuracy(msg, me)
        except Exception as e:
            row['error'] = repr(e)[:120]
        R['per_image'].append(row)

    # summary (与主实验相同的键名)
    keys = ['clean_ba', 'uniform_ba', 'signal_ra_ba', 'signal_soft_ba',
            'signal_soft_dilate_ba', 'bbox_ra_ba']
    for k in keys:
        vals = [r[k] for r in R['per_image'] if k in r]
        R['summary'][k] = {'ba': float(np.mean(vals)), 'n': len(vals)}
    R['summary']['clean_fail_frac'] = float(np.mean(
        [r['clean_fail_frac'] for r in R['per_image']]))
    R['summary']['att_fail_frac'] = float(np.mean(
        [r['att_fail_frac'] for r in R['per_image']]))

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(BASE, 'results', f'attackset_eval_{ts}.json')
    with open(out, 'w') as f:
        json.dump(R, f, indent=2, ensure_ascii=False, default=float)

    print(f'attackset 评估: n={len(R["per_image"])} 保存: {out}')
    for k, lbl in [('clean_ba', 'clean'), ('uniform_ba', 'uniform'),
                   ('signal_ra_ba', 'v1(binary)'), ('signal_soft_ba', 'soft'),
                   ('signal_soft_dilate_ba', 'v2'), ('bbox_ra_ba', 'oracle')]:
        print(f'  {lbl:<14} BA={R["summary"][k]["ba"]*100:6.1f}%')
    print(f'  fail_frac: clean={R["summary"]["clean_fail_frac"]*100:.2f}% '
          f'att={R["summary"]["att_fail_frac"]*100:.1f}%')
    # 为 make_figures 打印三示例行
    print('\n示例行 (fig2 候选):')
    for r in R['per_image']:
        if r['id'] in ('0018', '0022', '0048'):
            print(f"  {r['id']} {r['img']} | {r['instruction'][:24]} | "
                  f"uniform={r['uniform_ba']*100:.1f} v2={r['signal_soft_dilate_ba']*100:.1f} "
                  f"oracle={r['bbox_ra_ba']*100:.1f}")


if __name__ == '__main__':
    main()
