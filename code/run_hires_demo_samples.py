#!/usr/bin/env python3
"""为论文**定性示例图**生成高分辨率的 MLLM 引导语义编辑样本.

与 run_attack_ip2p.py 的区别:
  run_attack_ip2p.py  : 全量攻击集, imgsize=256 (实验用)
  本脚本              : 少量入选样本, **原始 COCO 分辨率** (论文插图用, 256px 印刷过小)

要点:
  - 指令改用**英文** (InstructPix2Pix 对英文指令更稳, 此前中文指令是编辑不自然的主因之一)
  - bbox 外严格保持原图 (真正的局部编辑)
  - 每张候选多 seed 生成, 由 evaluate_hires_samples.py 打自然度分后挑选

用法:
  conda activate apjf && cd code && python run_hires_demo_samples.py [--steps 40] [--seeds 1]
输出: ../results/attackset_hires_sample/{images/,meta.json}
"""
import sys, os, json, time, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image

for k in ['ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)
os.environ.setdefault('http_proxy', 'http://127.0.0.1:7897')
os.environ.setdefault('https_proxy', 'http://127.0.0.1:7897')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COCO = f'{BASE}/data/COCO'
OUT = f'{BASE}/results/attackset_hires_sample'
IMG_DIR = f'{OUT}/images'

# (cid, source, instruction_en, bbox)  bbox/instruction 沿用攻击集 meta, 指令译为英文
CANDIDATES = [
    ('0002', '000000000632.jpg', 'replace the bookshelf on the right with a television', [0.65, 0.0, 0.98, 0.82]),
    ('0000', '000000000139.jpg', 'replace the television on the left with a bookshelf', [0.05, 0.3, 0.4, 0.6]),
    ('0005', '000000000785.jpg', 'remove the skier in the center of the image', [0.35, 0.15, 0.65, 0.85]),
    ('0006', '000000000802.jpg', 'replace the refrigerator on the right with a tall potted plant', [0.5, 0.1, 1.0, 0.9]),
    ('0028', '000000002431.jpg', 'remove the wine glass', [0.7, 0.1, 0.9, 0.4]),
    ('0030', '000000002532.jpg', 'remove the person in the center of the image', [0.4, 0.3, 0.6, 0.7]),
    ('0031', '000000002587.jpg', 'remove the donut', [0.3, 0.1, 0.7, 0.6]),
    ('0043', '000000004395.jpg', 'replace the transparent cup in the hand with a red balloon', [0.7, 0.35, 0.95, 0.65]),
    # --- 第二批: 替换类编辑 (LLM+像素双重核查表明替换比删除更容易做到无痕) ---
    ('0023', '000000002149.jpg', 'replace the large apple on the right with a red apple', [0.55, 0.15, 0.95, 0.85]),
    ('0035', '000000003156.jpg', 'replace the toilet with a bathtub', [0.42, 0.55, 0.92, 0.92]),
    ('0045', '000000004765.jpg', 'change the surfer jacket from yellow to red', [0.35, 0.15, 0.65, 0.45]),
    ('0048', '000000005037.jpg', 'replace the bus with a truck', [0.15, 0.2, 0.95, 0.95]),
    ('0017', '000000001584.jpg', 'replace the red double-decker bus with a black taxi', [0.11, 0.12, 0.78, 0.95]),
    ('0032', '000000002592.jpg', 'replace the skull logo on the cup with a smiley face', [0.32, 0.33, 0.54, 0.59]),
]

PIPE = None


def load_ip2p():
    global PIPE
    if PIPE is not None:
        return PIPE
    from diffusers import StableDiffusionInstructPix2PixPipeline
    import torch
    print('[IP2P] loading instruct-pix2pix (local cache)...', flush=True)
    t0 = time.time()
    PIPE = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        'timbrooks/instruct-pix2pix', torch_dtype=torch.float16,
        safety_checker=None, local_files_only=True).to('cuda')
    PIPE.set_progress_bar_config(disable=True)
    print(f'[IP2P] loaded in {time.time()-t0:.0f}s', flush=True)
    return PIPE


def crop8(img):
    h, w = img.shape[:2]
    return img[:h - h % 8, :w - w % 8]


def edit_ip2p(img, instruction, steps, seed,
              guidance_scale=7.5, image_guidance_scale=1.5):
    import torch
    pipe = load_ip2p()
    gen = torch.Generator(device='cuda').manual_seed(seed)
    out = pipe(prompt=instruction, image=Image.fromarray(img),
               num_inference_steps=steps, guidance_scale=guidance_scale,
               image_guidance_scale=image_guidance_scale,
               generator=gen).images[0]
    return np.array(out.convert('RGB'))


def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return float('inf') if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--seeds', type=int, default=1, help='每样本生成的 seed 数 (seed=42,43,...)')
    ap.add_argument('--guidance', type=float, default=7.5)
    ap.add_argument('--img-guidance', type=float, default=1.5)
    ap.add_argument('--only', default='', help='逗号分隔的 cid, 只跑这些候选')
    ap.add_argument('--skip-existing', action='store_true', default=True,
                    help='已存在同名输出的 (cid,seed) 组合跳过 (增量补跑)')
    args = ap.parse_args()

    os.makedirs(IMG_DIR, exist_ok=True)
    t0 = time.time()
    meta_p = f'{OUT}/meta.json'
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {
        'generated': datetime.datetime.now().isoformat(), 'steps': args.steps,
        'guidance': args.guidance, 'img_guidance': args.img_guidance,
        'generator': 'instruct-pix2pix', 'instruction_lang': 'en', 'samples': []}
    meta.update(steps=args.steps, guidance=args.guidance,
                img_guidance=args.img_guidance)
    have = {s['id'] for s in meta['samples']}
    only = [c for c in args.only.split(',') if c] or None

    todo = [(cid, src, inst, bbox, k)
            for cid, src, inst, bbox in CANDIDATES if (only is None or cid in only)
            for k in range(args.seeds)
            if f'{cid}_{42 + k}' not in have]
    print(f'待生成 {len(todo)} 张 (已有 {len(have)} 张)', flush=True)

    for cid, src, inst, bbox, k in todo:
        seed = 42 + k
        img = crop8(np.array(Image.open(f'{COCO}/{src}').convert('RGB')))
        h, w = img.shape[:2]
        att = edit_ip2p(img, inst, args.steps, seed,
                        args.guidance, args.img_guidance)
        x1, y1, x2, y2 = [int(v * (w if i % 2 == 0 else h)) for i, v in enumerate(bbox)]
        x2, y2 = max(x2, x1 + 8), max(y2, y1 + 8)
        mask = np.zeros((h, w), dtype=bool)
        mask[y1:y2, x1:x2] = True
        att = np.where(mask[:, :, None], att, img).astype(np.uint8)

        tag = f'{cid}_{seed}' if args.seeds > 1 else cid
        Image.fromarray(img).save(f'{IMG_DIR}/{tag}_orig.png')
        Image.fromarray(att).save(f'{IMG_DIR}/{tag}_att.png')
        Image.fromarray((mask * 255).astype(np.uint8)).save(f'{IMG_DIR}/{tag}_mask.png')
        p = psnr(img, att)
        meta['samples'].append({'id': tag, 'cid': cid, 'source': src, 'seed': seed,
                                'instruction': inst, 'bbox': bbox,
                                'w': w, 'h': h, 'psnr': p})
        print(f'  {tag:>10}  {w}x{h}  PSNR={p:5.1f}dB  {inst}', flush=True)
        with open(meta_p, 'w') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    meta['total_time_s'] = round(time.time() - t0, 1)
    json.dump(meta, open(meta_p, 'w'), ensure_ascii=False, indent=2)
    print(f'\ndone: {len(meta["samples"])} samples total -> {OUT} '
          f'(本次 {meta["total_time_s"]:.0f}s)')


if __name__ == '__main__':
    main()
