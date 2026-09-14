#!/usr/bin/env python3
"""Full-inpainting 变体实验 (回应审稿意见: "paste-back 太 convenient").

问题: 我们的发布协议把扩散输出**只取 bbox 内**贴回原图, 使 bbox 外像素逐位不变.
      这保证攻击严格局部, 但审稿人可质疑: 真实的 IP2P/SD inpainting 输出**不**是
      零外溢的, 所以我们的设定可能"太干净", 且与 baseline 的评测协议不同.

本实验: 对同一批样本, 生成**完整 inpainting 输出**(不粘贴回, 保留扩散的全部改动,
      包括 bbox 外的边缘渗色/全局色调漂移), 然后:
        (1) 量化 bbox 外的实际外溢幅度 (证明"零外溢"是协议造成的, 非自然如此)
        (2) 在 full output 上重跑 uniform 与 v2, 报告 gain 是否仍为正

输出: ../results/full_inpaint_<ts>.json
用法: cd code && python run_full_inpaint_variant.py [--n 50] [--ids 0002_42,...]
"""
import sys, os, json, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image

from run_signal_ra import embed_signal, decode_signal
from dct_watermark import bit_accuracy

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIRES = f'{BASE}/results/attackset_hires_sample'
DEVICE = 'cuda'


def crop8(a):
    h, w = a.shape[:2]
    return a[:h - h % 8, :w - w % 8]


def psnr(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    m = np.mean((a - b) ** 2)
    return float('inf') if m == 0 else float(10 * np.log10(255.0 ** 2 / m))


def bbox_overlap_metrics(orig, full_out, bbox01):
    """量化 bbox 内/外的实际改动幅度."""
    H, W = orig.shape[:2]
    x1, y1, x2, y2 = bbox01
    bx1, by1 = int(round(x1 * W)), int(round(y1 * H))
    bx2, by2 = int(round(x2 * W)), int(round(y2 * H))
    d = np.abs(full_out.astype(np.float32) - orig.astype(np.float32)).mean(axis=2)
    m_in = np.zeros((H, W), bool); m_in[by1:by2, bx1:bx2] = True
    return {
        'inside_mae': float(d[m_in].mean()) if m_in.any() else 0.0,
        'outside_mae': float(d[~m_in].mean()) if (~m_in).any() else 0.0,
        'outside_max': float(d[~m_in].max()) if (~m_in).any() else 0.0,
        # bbox 外有多少像素被改动了哪怕 1 个灰度级
        'outside_changed_frac': float((d[~m_in] > 0).mean()) if (~m_in).any() else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--msglen', type=int, default=256)
    ap.add_argument('--delta', type=int, default=15)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    meta = json.load(open(f'{HIRES}/meta.json'))
    samples = meta['samples'][:args.n]
    print(f'full-inpainting 变体: n={len(samples)} (不做 paste-back)')

    from diffusers import StableDiffusionInstructPix2PixPipeline
    import torch
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        'timbrooks/instruct-pix2pix', torch_dtype=torch.float16, safety_checker=None)
    pipe = pipe.to(DEVICE)
    print(f'  模型已加载 ({DEVICE})')

    rng = np.random.RandomState(args.seed)
    msg = rng.randint(0, 2, args.msglen).astype(np.uint8)
    out = {'meta': {'n': len(samples), 'msglen': args.msglen, 'delta': args.delta,
                    'steps': args.steps, 'protocol': 'full_inpainting_no_paste',
                    'ts': datetime.datetime.now().isoformat()},
           'per_image': [], 'summary': {}}

    for idx, s in enumerate(samples):
        cid = s['id']
        try:
            o = crop8(np.array(Image.open(f'{HIRES}/images/{cid}_orig.png').convert('RGB')))
        except FileNotFoundError:
            continue
        # 用完整扩散输出 (不粘贴回)
        gen = pipe(prompt=s['instruction'], image=Image.fromarray(o),
                   num_inference_steps=args.steps, image_guidance_scale=1.5,
                   guidance_scale=7.5).images[0]
        full = crop8(np.array(gen.convert('RGB')))

        row = {'img': cid}
        row.update(bbox_overlap_metrics(o, full, s['bbox']))
        row['psnr_full'] = psnr(o, full)

        wm = embed_signal(o, msg, args.delta)
        me, _ = decode_signal(wm, args.msglen, args.delta, use_check=False)
        row['clean_ba'] = float(bit_accuracy(msg, me))

        # 关键: 直接把水印图送入扩散, 得到真实完整编辑结果
        gen_wm = pipe(prompt=s['instruction'], image=Image.fromarray(wm),
                      num_inference_steps=args.steps, image_guidance_scale=1.5,
                      guidance_scale=7.5).images[0]
        att_wm = crop8(np.array(gen_wm.convert('RGB')))
        me, _ = decode_signal(att_wm, args.msglen, args.delta, use_check=False)
        row['uni_ba'] = float(bit_accuracy(msg, me))
        me, _ = decode_signal(att_wm, args.msglen, args.delta, use_check=True,
                              soft=True, dilate=1)
        row['v2_ba'] = float(bit_accuracy(msg, me))
        out['per_image'].append(row)
        if (idx + 1) % 5 == 0:
            print(f'  [{idx+1}/{len(samples)}] {cid} 外溢MAE={row["outside_mae"]:.2f} '
                  f'uni={row["uni_ba"]*100:.1f}% v2={row["v2_ba"]*100:.1f}%')

    pi = out['per_image']
    if pi:
        out['summary'] = {
            'n': len(pi),
            'outside_mae_mean': float(np.mean([r['outside_mae'] for r in pi])),
            'outside_changed_frac_mean': float(np.mean([r['outside_changed_frac'] for r in pi])),
            'uni_ba': float(np.mean([r['uni_ba'] for r in pi])),
            'v2_ba': float(np.mean([r['v2_ba'] for r in pi])),
            'gain_pt': float((np.mean([r['v2_ba'] for r in pi])
                              - np.mean([r['uni_ba'] for r in pi])) * 100),
        }
        s = out['summary']
        print(f"\n=== 汇总 ===")
        print(f"  bbox 外平均 MAE: {s['outside_mae_mean']:.2f}  "
              f"(被改动像素占比 {s['outside_changed_frac_mean']*100:.1f}%)")
        print(f"  uniform {s['uni_ba']*100:.1f}%  v2 {s['v2_ba']*100:.1f}%  "
              f"gain +{s['gain_pt']:.1f}pt")

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/full_inpaint_{ts}.json'

    def to_py(o):
        if isinstance(o, dict): return {k: to_py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [to_py(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    payload = json.dumps(to_py(out), indent=1)
    with open(fp, 'w') as f:
        f.write(payload)
    print(f'\n已保存: {fp}')


if __name__ == '__main__':
    main()
