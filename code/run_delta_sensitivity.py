#!/usr/bin/env python3
"""Δ 敏感性分析: 不同嵌入强度下各方案的 BA.

在 COCO n=10 子集上, Δ ∈ {10, 15, 20, 25}, 统一 seed=42.
输出 results/delta_sensitivity_<ts>.json + 表格.
"""
import sys, os, time, json, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from run_idea_probe import load, collect_source, psnr, bit_accuracy
from run_signal_ra import embed_signal, decode_signal, CHECK_SEED
from run_mllm_benchmark import load_env, mllm_edit_instruct, sd_execute, extract_uniform_bbox_ra

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--deltas', type=int, nargs='+', default=[10, 15, 20, 25])
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--msglen', type=int, default=256)
    args = ap.parse_args()
    load_env()

    paths = collect_source('coco', args.n)
    rng = np.random.RandomState(42)
    # 固定 message 和 edit prompts (同一组图片, 同一个 msg, 不同 Δ)
    msg = rng.randint(0, 2, args.msglen).astype(np.uint8)

    # 先生成一组固定的编辑指令 (对 n 张图各生成一次)
    print(f'生成 {len(paths)} 张图的编辑指令...')
    edits = []
    for p in paths:
        img = load(p, 256)
        edit = mllm_edit_instruct(os.environ.get('MIMO_BASE_URL', ''),
                                  os.environ.get('MIMO_API_KEY', ''),
                                  os.environ.get('MIMO_MODEL', 'mimo-v2.5'), img)
        edits.append(edit)
    print(f'编辑指令已生成')

    R = {'meta': {'n': args.n, 'deltas': args.deltas, 'msglen': args.msglen,
                  'steps': args.steps, 'ts': datetime.datetime.now().isoformat()},
         'per_image': [], 'summary': {}}

    for delta in args.deltas:
        print(f'\nΔ={delta}:')
        delta_results = []
        for idx, (p, edit) in enumerate(zip(paths, edits)):
            img = load(p, 256)
            wm = embed_signal(img, msg, delta)
            att = sd_execute(edit, wm, steps=args.steps)
            # 提取各方案
            me, _ = decode_signal(att, args.msglen, delta, use_check=False)
            uni = bit_accuracy(msg, me)
            me, _ = decode_signal(att, args.msglen, delta, use_check=True)
            v1 = bit_accuracy(msg, me)
            me, _ = decode_signal(att, args.msglen, delta, use_check=True, soft=True)
            sft = bit_accuracy(msg, me)
            me, _ = decode_signal(att, args.msglen, delta, use_check=True, soft=True, dilate=1)
            v2 = bit_accuracy(msg, me)
            me = extract_uniform_bbox_ra(att, args.msglen, delta, edit['bbox'])
            orc = bit_accuracy(msg, me)
            delta_results.append({
                'img': os.path.basename(p), 'delta': delta,
                'uniform': uni, 'v1': v1, 'soft': sft, 'v2': v2, 'oracle': orc,
                'psnr': psnr(img, wm),
            })
            print(f'  [{idx+1}/{len(paths)}] {os.path.basename(p)[:18]} '
                  f'uni={uni*100:.0f}% v2={v2*100:.0f}% orc={orc*100:.0f}%')
        R['per_image'].extend(delta_results)
        # summary
        for key in ['uniform', 'v1', 'soft', 'v2', 'oracle']:
            vals = [r[key] for r in delta_results]
            k = f'delta_{delta}_{key}'
            R['summary'][k] = {'ba': float(np.mean(vals)), 'n': len(vals)}
        psnr_vals = [r['psnr'] for r in delta_results]
        R['summary'][f'delta_{delta}_psnr'] = {'mean': float(np.mean(psnr_vals)), 'n': len(psnr_vals)}

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = f'{BASE}/results/delta_sensitivity_{ts}.json'
    with open(out, 'w') as f:
        json.dump(R, f, indent=2, default=float)
    print(f'\n保存: {out}')

    # 打印汇总表
    print(f'\n{"Δ":>4s} {"PSNR":>6s} {"uniform":>8s} {"v1":>8s} {"soft":>8s} {"v2":>8s} {"oracle":>8s}')
    for delta in args.deltas:
        psnr_m = R['summary'][f'delta_{delta}_psnr']['mean']
        row = f'{delta:>4d} {psnr_m:>5.1f}dB'
        for key in ['uniform', 'v1', 'soft', 'v2', 'oracle']:
            ba = R['summary'][f'delta_{delta}_{key}']['ba'] * 100
            row += f' {ba:>7.1f}%'
        print(row)


if __name__ == '__main__':
    main()
