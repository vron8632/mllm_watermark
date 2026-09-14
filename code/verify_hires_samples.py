#!/usr/bin/env python3
"""复核: 高分辨率样本的编辑是否"真的发生了 + 是否看不出痕迹".

目的: DeepSeek 的 naturalness 可能给"几乎没编辑"的图打高分 (如 0031 甜甜圈仍在).
本脚本让 mimo (视觉模型) 直接回答两个问题, 并对 bbox 区域做像素级差异统计:
  Q1 编辑是否真的发生在 bbox 内? (yes/no)
  Q2 编辑痕迹是否肉眼可见? (invisible/slight/obvious)

用法: cd code && python verify_hires_samples.py --ids 0030,0031,0005
"""
import sys, os, json, argparse, io, base64
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image

for k in ['ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)
os.environ.setdefault('http_proxy', 'http://127.0.0.1:7897')
os.environ.setdefault('https_proxy', 'http://127.0.0.1:7897')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = f'{BASE}/results/attackset_hires_sample'


def bbox_stats(o, a, bbox):
    h, w = o.shape[:2]
    x1, y1, x2, y2 = [int(v * q) for v, q in zip(bbox, (w, h, w, h))]
    inside = np.abs(a[y1:y2, x1:x2].astype(float) - o[y1:y2, x1:x2].astype(float)).mean()
    outside = np.r_[a[:y1, :].astype(float).ravel() - o[:y1, :].astype(float).ravel(),
                    a[y2:, :].astype(float).ravel() - o[y2:, :].astype(float).ravel()]
    return inside, float(np.abs(outside).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', default='0030,0031,0005,0004,0002,0049')
    args = ap.parse_args()

    from run_mllm_benchmark import load_env, openai_chat, img_to_data_url
    _, _, mimo_model, _, _, _ = load_env()
    mimo_key = os.environ.get('MIMO_API_KEY')
    mimo_url = os.environ.get('MIMO_BASE_URL', '')

    meta = json.load(open(f'{D}/meta.json'))
    by = {s['id']: s for s in meta['samples']}
    sysp = ('You are an image-forensics expert. You get a PAIR of images: LEFT = before, '
            'RIGHT = after a local AI edit. Answer ONLY JSON: '
            '{"edit_visible": true/false, "edit_region_correct": true/false, '
            '"artifact_level": "invisible|slight|obvious", "note": "<=15 words"}')

    for cid in args.ids.split(','):
        s = by[cid]
        o = np.array(Image.open(f'{D}/images/{cid}_orig.png').convert('RGB'))
        a = np.array(Image.open(f'{D}/images/{cid}_att.png').convert('RGB'))
        di, do = bbox_stats(o, a, s['bbox'])
        h = 384
        oi = Image.fromarray(o).resize((int(o.shape[1] * h / o.shape[0]), h))
        ai = Image.fromarray(a).resize((int(a.shape[1] * h / a.shape[0]), h))
        pair = np.hstack([np.array(oi), np.ones((h, 8, 3), np.uint8) * 255, np.array(ai)])
        raw = openai_chat(mimo_url, mimo_key, mimo_model,
                          [{'role': 'system', 'content': sysp},
                           {'role': 'user', 'content': [
                               {'type': 'text', 'text': f'instruction: {s["instruction"]}'},
                               {'type': 'image_url',
                                'image_url': {'url': img_to_data_url(pair)}}]}],
                          response_format={'type': 'json_object'})
        print(f'{cid:>5} inside_MAE={di:6.2f} outside_MAE={do:5.3f} '
              f'{s["w"]}x{s["h"]}  ->  {raw[:170]}', flush=True)


if __name__ == '__main__':
    main()
