#!/usr/bin/env python3
"""快速对比: SD 1.5 inpainting vs InstructPix2Pix.

只用已下载的模型, 不等 SDXL.
"""
import sys, os, time, json, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
import torch
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = f'{BASE}/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Fix proxy
for k in ['ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)
if 'http_proxy' not in os.environ:
    os.environ.setdefault('http_proxy', 'http://127.0.0.1:7897')
if 'https_proxy' not in os.environ:
    os.environ.setdefault('https_proxy', 'http://127.0.0.1:7897')

from run_idea_probe import load, collect_source
from run_mllm_benchmark import load_env, mllm_edit_instruct


def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def edit_sd15(pipe, img, bbox, instruction, steps=30, seed=0):
    """SD 1.5 inpainting: 用 mask + 通用 prompt."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v * (w if i % 2 == 0 else h)) for i, v in enumerate(bbox)]
    x2, y2 = max(x2, x1 + 8), max(y2, y1 + 8)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))

    gen = torch.Generator(device='cuda').manual_seed(seed)
    out = pipe(
        prompt='a realistic photo, natural lighting, consistent style',
        negative_prompt='blurry, unnatural, artifact, painting, cartoon',
        image=Image.fromarray(img),
        mask_image=Image.fromarray(mask),
        height=h, width=w,
        num_inference_steps=steps,
        guidance_scale=7.5,
        generator=gen,
    ).images[0]
    result = np.array(out.convert('RGB'))
    # bbox 外保持原图
    mask3 = np.stack([mask > 127] * 3, axis=-1)
    result = np.where(mask3, result, img).astype(np.uint8)
    return result


def edit_ip2p(pipe, img, bbox, instruction, steps=30, seed=0, use_bbox=True):
    """InstructPix2Pix: 用文本指令编辑."""
    gen = torch.Generator(device='cuda').manual_seed(seed)
    out = pipe(
        prompt=instruction,
        image=Image.fromarray(img),
        num_inference_steps=steps,
        guidance_scale=7.5,
        image_guidance_scale=1.5,
        generator=gen,
    ).images[0]
    result = np.array(out.convert('RGB'))

    if use_bbox:
        # bbox 外保持原图 (公平对比)
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(v * (w if i % 2 == 0 else h)) for i, v in enumerate(bbox)]
        x2, y2 = max(x2, x1 + 8), max(y2, y1 + 8)
        mask3 = np.zeros((h, w, 3), dtype=bool)
        mask3[y1:y2, x1:x2] = True
        result = np.where(mask3, result, img).astype(np.uint8)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num', type=int, default=6, help='对比图片数')
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--imgsize', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    load_env()
    t0 = time.time()
    paths = collect_source('coco', args.num)
    print(f'对比 {len(paths)} 张图')

    # 加载模型
    print('\n加载模型...')
    from diffusers import StableDiffusionInpaintPipeline, StableDiffusionInstructPix2PixPipeline

    print('  [A] SD 1.5 inpainting...', end=' ', flush=True)
    pipe_a = StableDiffusionInpaintPipeline.from_pretrained(
        'runwayml/stable-diffusion-inpainting',
        torch_dtype=torch.float16, variant='fp16', safety_checker=None,
    ).to('cuda')
    pipe_a.set_progress_bar_config(disable=True)
    print('✅')

    print('  [B] InstructPix2Pix...', end=' ', flush=True)
    pipe_b = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        'timbrooks/instruct-pix2pix',
        torch_dtype=torch.float16, safety_checker=None,
    ).to('cuda')
    pipe_b.set_progress_bar_config(disable=True)
    print('✅')

    # MLLM
    mimo_key = os.environ.get('MIMO_API_KEY')
    mimo_url = os.environ.get('MIMO_BASE_URL', '')
    mimo_model = os.environ.get('MIMO_MODEL', 'mimo-v2.5')

    out_dir = os.path.join(RESULTS_DIR, f'compare_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for idx, p in enumerate(paths):
        img = load(p, args.imgsize)
        print(f'\n[{idx+1}/{len(paths)}] {os.path.basename(p)}')

        edit = mllm_edit_instruct(mimo_url, mimo_key, mimo_model, img)
        instruction = edit['instruction']
        bbox = edit['bbox']
        print(f'  指令: {instruction}')
        print(f'  bbox: {[f"{v:.2f}" for v in bbox]}')

        row = {'img': os.path.basename(p), 'instruction': instruction, 'bbox': bbox}

        # 保存原图
        Image.fromarray(img).save(os.path.join(out_dir, f'{idx:02d}_original.png'))

        # [A] SD 1.5
        print('  [A] SD 1.5...', end=' ', flush=True)
        t1 = time.time()
        att_a = edit_sd15(pipe_a, img, bbox, instruction, args.steps, args.seed)
        dt = time.time() - t1
        p_a = psnr(img, att_a)
        print(f'{dt:.1f}s  PSNR={p_a:.1f}dB')
        Image.fromarray(att_a).save(os.path.join(out_dir, f'{idx:02d}_A_sd15.png'))
        row['A_sd15'] = {'psnr': p_a, 'time': dt}

        # [B] IP2P (bbox 局部)
        print('  [B] IP2P (bbox)...', end=' ', flush=True)
        t1 = time.time()
        att_b = edit_ip2p(pipe_b, img, bbox, instruction, args.steps, args.seed, use_bbox=True)
        dt = time.time() - t1
        p_b = psnr(img, att_b)
        print(f'{dt:.1f}s  PSNR={p_b:.1f}dB')
        Image.fromarray(att_b).save(os.path.join(out_dir, f'{idx:02d}_B_ip2p_bbox.png'))
        row['B_ip2p_bbox'] = {'psnr': p_b, 'time': dt}

        # [B+] IP2P (整图编辑, 不限制 bbox)
        print('  [B+] IP2P (full)...', end=' ', flush=True)
        t1 = time.time()
        att_bf = edit_ip2p(pipe_b, img, bbox, instruction, args.steps, args.seed, use_bbox=False)
        dt = time.time() - t1
        p_bf = psnr(img, att_bf)
        print(f'{dt:.1f}s  PSNR={p_bf:.1f}dB')
        Image.fromarray(att_bf).save(os.path.join(out_dir, f'{idx:02d}_B_ip2p_full.png'))
        row['B_ip2p_full'] = {'psnr': p_bf, 'time': dt}

        results.append(row)

    # 保存
    out_json = os.path.join(out_dir, 'comparison.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 汇总
    print(f'\n{"="*60}')
    print(f'  对比完成! 图片: {out_dir}')
    print(f'  耗时: {time.time()-t0:.0f}s')
    print(f'{"="*60}')

    print(f'\n  平均 PSNR (越低=编辑越强, 越高=越保真):')
    for key in ['A_sd15', 'B_ip2p_bbox', 'B_ip2p_full']:
        vals = [r[key]['psnr'] for r in results if key in r]
        print(f'    {key:<20} {np.mean(vals):.1f} dB')

    print(f'\n  编辑质量主观评价: 请查看 {out_dir}/ 中的图片对比')


if __name__ == '__main__':
    main()
