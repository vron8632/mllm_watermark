#!/usr/bin/env python3
"""下载并对比不同生成模型的编辑效果.

模型:
  A) 当前 SD 1.5 inpainting (runwayml/stable-diffusion-inpainting) — 论文现状
  B) InstructPix2Pix (timbrooks/instruct-pix2pix) — 指令驱动编辑, ~1.7GB
  C) SDXL Inpainting (diffusers/stable-diffusion-xl-1.0-inpainting-0.1) — ~6.5GB

用法:
  cd code
  conda activate apjf
  python download_and_compare.py --num 6          # 下载 + 对比 6 张
  python download_and_compare.py --num 6 --skip-download  # 已下载, 直接对比
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

# Fix proxy: remove SOCKS ALL_PROXY that breaks httpx, keep only HTTP proxy
for k in ['ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)
if 'http_proxy' not in os.environ:
    os.environ.setdefault('http_proxy', 'http://127.0.0.1:7897')
if 'https_proxy' not in os.environ:
    os.environ.setdefault('https_proxy', 'http://127.0.0.1:7897')

from run_idea_probe import load, collect_source, detect_removal_mask
from run_mllm_benchmark import load_env, mllm_edit_instruct, img_to_data_url


def download_models():
    """下载所有候选模型."""
    print('='*60)
    print('  下载模型')
    print('='*60)

    # Model B: InstructPix2Pix
    print('\n[B] 下载 InstructPix2Pix (~1.7GB)...')
    t0 = time.time()
    from diffusers import StableDiffusionInstructPix2PixPipeline
    pipe_b = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        'timbrooks/instruct-pix2pix',
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to('cuda')
    pipe_b.set_progress_bar_config(disable=True)
    print(f'  ✅ InstructPix2Pix 下载完成 ({time.time()-t0:.0f}s)')

    # Model C: SDXL Inpainting
    print('\n[C] 下载 SDXL Inpainting (~6.5GB)...')
    t0 = time.time()
    from diffusers import AutoPipelineForInpainting
    pipe_c = AutoPipelineForInpainting.from_pretrained(
        'diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
        torch_dtype=torch.float16,
        variant='fp16',
    ).to('cuda')
    pipe_c.set_progress_bar_config(disable=True)
    print(f'  ✅ SDXL Inpainting 下载完成 ({time.time()-t0:.0f}s)')

    # Model A: 当前 SD 1.5 inpainting
    print('\n[A] 加载当前 SD 1.5 Inpainting...')
    t0 = time.time()
    from diffusers import StableDiffusionInpaintPipeline
    pipe_a = StableDiffusionInpaintPipeline.from_pretrained(
        'runwayml/stable-diffusion-inpainting',
        torch_dtype=torch.float16,
        variant='fp16',
        safety_checker=None,
    ).to('cuda')
    pipe_a.set_progress_bar_config(disable=True)
    print(f'  ✅ SD 1.5 Inpainting 加载完成 ({time.time()-t0:.0f}s)')

    return {'A_sd15': pipe_a, 'B_instructpix2pix': pipe_b, 'C_sdxl': pipe_c}


def edit_with_sd15(pipe, img, bbox, steps=30, seed=0):
    """Model A: SD 1.5 inpainting (当前论文方法)."""
    from diffusers import StableDiffusionInpaintPipeline
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v * (w if i % 2 == 0 else h)) for i, v in enumerate(bbox)]
    x2 = max(x2, x1 + 8)
    y2 = max(y2, y1 + 8)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    mask_img = Image.fromarray(mask)

    gen = torch.Generator(device='cuda').manual_seed(seed)
    out = pipe(
        prompt='a realistic photo, natural lighting, consistent style',
        negative_prompt='blurry, unnatural, artifact, painting, cartoon',
        image=Image.fromarray(img),
        mask_image=mask_img,
        height=h, width=w,
        num_inference_steps=steps,
        guidance_scale=7.5,
        generator=gen,
    ).images[0]
    result = np.array(out.convert('RGB'))
    # 强制 bbox 外保持原图 (局部编辑)
    mask3 = np.stack([mask > 127] * 3, axis=-1)
    result = np.where(mask3, result, img).astype(np.uint8)
    return result


def edit_with_instructpix2pix(pipe, img, instruction, steps=30, seed=0):
    """Model B: InstructPix2Pix — 直接用文本指令编辑整图."""
    gen = torch.Generator(device='cuda').manual_seed(seed)
    out = pipe(
        prompt=instruction,
        image=Image.fromarray(img),
        num_inference_steps=steps,
        guidance_scale=7.5,
        image_guidance_scale=1.5,
        generator=gen,
    ).images[0]
    return np.array(out.convert('RGB'))


def edit_with_instructpix2pix_bbox(pipe, img, instruction, bbox, steps=30, seed=0):
    """Model B+: InstructPix2Pix — 指令编辑 + bbox 外强制保持原图."""
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
    # bbox 外保持原图
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v * (w if i % 2 == 0 else h)) for i, v in enumerate(bbox)]
    x2 = max(x2, x1 + 8)
    y2 = max(y2, y1 + 8)
    mask3 = np.zeros((h, w, 3), dtype=bool)
    mask3[y1:y2, x1:x2] = True
    result = np.where(mask3, result, img).astype(np.uint8)
    return result


def edit_with_sdxl(pipe, img, bbox, steps=30, seed=0):
    """Model C: SDXL Inpainting."""
    h, w = img.shape[:2]
    # SDXL needs larger resolution
    target_h, target_w = max(h, 1024), max(w, 1024)
    img_resized = cv2.resize(img, (target_w, target_h))

    x1, y1, x2, y2 = [int(v * (target_w if i % 2 == 0 else target_h)) for i, v in enumerate(bbox)]
    x2 = max(x2, x1 + 8)
    y2 = max(y2, y1 + 8)

    mask = np.zeros((target_h, target_w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    mask = cv2.dilate(mask, np.ones((10, 10), np.uint8))
    mask_img = Image.fromarray(mask)

    gen = torch.Generator(device='cuda').manual_seed(seed)
    out = pipe(
        prompt='a realistic photo, natural lighting, consistent style',
        negative_prompt='blurry, unnatural, artifact, painting, cartoon, text',
        image=Image.fromarray(img_resized),
        mask_image=mask_img,
        num_inference_steps=steps,
        guidance_scale=7.5,
        generator=gen,
    ).images[0]
    result = np.array(out.convert('RGB'))
    # 缩放回原尺寸
    result = cv2.resize(result, (w, h))
    # bbox 外保持原图
    x1o, y1o, x2o, y2o = [int(v * (w if i % 2 == 0 else h)) for i, v in enumerate(bbox)]
    x2o = max(x2o, x1o + 8)
    y2o = max(y2o, y1o + 8)
    mask3 = np.zeros((h, w, 3), dtype=bool)
    mask3[y1o:y2o, x1o:x2o] = True
    result = np.where(mask3, result, img).astype(np.uint8)
    return result


def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num', type=int, default=6, help='对比图片数')
    ap.add_argument('--steps', type=int, default=30, help='推理步数')
    ap.add_argument('--imgsize', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--skip-download', action='store_true', help='跳过下载(已下载)')
    args = ap.parse_args()

    load_env()
    t0 = time.time()
    paths = collect_source('coco', args.num)
    print(f'对比 {len(paths)} 张图, steps={args.steps}, imgsize={args.imgsize}')

    # 下载模型
    if not args.skip_download:
        pipes = download_models()
    else:
        print('跳过下载, 从缓存加载...')
        pipes = download_models()  # 会自动用缓存

    # 生成 MLLM 编辑指令
    mimo_key = os.environ.get('MIMO_API_KEY')
    mimo_url = os.environ.get('MIMO_BASE_URL', '')
    mimo_model = os.environ.get('MIMO_MODEL', 'mimo-v2.5')

    out_dir = os.path.join(RESULTS_DIR, f'model_comparison_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for idx, p in enumerate(paths):
        img = load(p, args.imgsize)
        print(f'\n[{idx+1}/{len(paths)}] {os.path.basename(p)}')

        # 获取编辑指令
        edit = mllm_edit_instruct(mimo_url, mimo_key, mimo_model, img)
        instruction = edit['instruction']
        bbox = edit['bbox']
        print(f'  指令: {instruction}')
        print(f'  bbox: {bbox}')

        row = {'img': os.path.basename(p), 'instruction': instruction, 'bbox': bbox}

        # 保存原图
        Image.fromarray(img).save(os.path.join(out_dir, f'{idx:02d}_original.png'))

        # Model A: SD 1.5 (当前论文)
        print('  [A] SD 1.5 inpainting...', end=' ', flush=True)
        t1 = time.time()
        att_a = edit_with_sd15(pipes['A_sd15'], img, bbox, args.steps, args.seed)
        print(f'{time.time()-t1:.1f}s, PSNR={psnr(img, att_a):.1f}dB')
        Image.fromarray(att_a).save(os.path.join(out_dir, f'{idx:02d}_A_sd15.png'))
        row['A_sd15_psnr'] = psnr(img, att_a)

        # Model B: InstructPix2Pix (整图编辑)
        print('  [B] InstructPix2Pix (full)...', end=' ', flush=True)
        t1 = time.time()
        att_b_full = edit_with_instructpix2pix(pipes['B_instructpix2pix'], img, instruction, args.steps, args.seed)
        print(f'{time.time()-t1:.1f}s, PSNR={psnr(img, att_b_full):.1f}dB')
        Image.fromarray(att_b_full).save(os.path.join(out_dir, f'{idx:02d}_B_ip2p_full.png'))
        row['B_ip2p_full_psnr'] = psnr(img, att_b_full)

        # Model B+: InstructPix2Pix (bbox 局部)
        print('  [B+] InstructPix2Pix (bbox)...', end=' ', flush=True)
        t1 = time.time()
        att_b_bbox = edit_with_instructpix2pix_bbox(pipes['B_instructpix2pix'], img, instruction, bbox, args.steps, args.seed)
        print(f'{time.time()-t1:.1f}s, PSNR={psnr(img, att_b_bbox):.1f}dB')
        Image.fromarray(att_b_bbox).save(os.path.join(out_dir, f'{idx:02d}_B_ip2p_bbox.png'))
        row['B_ip2p_bbox_psnr'] = psnr(img, att_b_bbox)

        # Model C: SDXL Inpainting
        print('  [C] SDXL Inpainting...', end=' ', flush=True)
        t1 = time.time()
        att_c = edit_with_sdxl(pipes['C_sdxl'], img, bbox, args.steps, args.seed)
        print(f'{time.time()-t1:.1f}s, PSNR={psnr(img, att_c):.1f}dB')
        Image.fromarray(att_c).save(os.path.join(out_dir, f'{idx:02d}_C_sdxl.png'))
        row['C_sdxl_psnr'] = psnr(img, att_c)

        results.append(row)

    # 保存结果
    out_json = os.path.join(out_dir, 'comparison.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f'\n{"="*60}')
    print(f'  对比完成! 图片保存在: {out_dir}')
    print(f'  结果: {out_json}')
    print(f'  总耗时: {time.time()-t0:.0f}s')
    print(f'{"="*60}')

    # 汇总
    print(f'\n  平均 PSNR (越低=编辑越强):')
    for key in ['A_sd15_psnr', 'B_ip2p_full_psnr', 'B_ip2p_bbox_psnr', 'C_sdxl_psnr']:
        vals = [r[key] for r in results if key in r]
        print(f'    {key:<25} {np.mean(vals):.1f} dB')


if __name__ == '__main__':
    main()
