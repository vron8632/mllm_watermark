#!/usr/bin/env python3
"""用 InstructPix2Pix 生成 MLLM 引导的语义编辑攻击集.

流程:
  1) mimo 看原图 → 编辑指令 + bbox
  2) InstructPix2Pix 用文本指令编辑图片 (bbox 外保持原图)
  3) 保存 orig/att/mask/meta.json 供后续实验使用

用法:
  cd code
  conda activate apjf
  python run_attack_ip2p.py --num 200              # 生成 200 张攻击图
  python run_attack_ip2p.py --num 50 --quick       # 快速测试

输出: ../results/attackset_ip2p_<timestamp>/
"""
import sys, os, time, json, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
import torch
from PIL import Image

# Fix proxy
for k in ['ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)
if 'http_proxy' not in os.environ:
    os.environ.setdefault('http_proxy', 'http://127.0.0.1:7897')
if 'https_proxy' not in os.environ:
    os.environ.setdefault('https_proxy', 'http://127.0.0.1:7897')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = f'{BASE}/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

from run_idea_probe import load, collect_source
from run_mllm_benchmark import load_env, mllm_edit_instruct


class Progress:
    def __init__(self, total, desc=''):
        self.total, self.desc, self.n = total, desc, 0
        self.t0 = time.time()
        self.t_last = 0.0

    @staticmethod
    def _fmt(s):
        s = int(s)
        return f'{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}'

    def update(self, k=1):
        self.n += k
        if time.time() - self.t_last < 3.0 and self.n < self.total:
            return
        self.t_last = time.time()
        el = self.t_last - self.t0
        rem = el / self.n * (self.total - self.n) if self.n > 0 else 0
        eta = datetime.datetime.now() + datetime.timedelta(seconds=rem)
        print(f'  [{datetime.datetime.now().strftime("%H:%M:%S")}] {self.desc} '
              f'{self.n}/{self.total}  已用 {self._fmt(el)}  剩余约 {self._fmt(rem)}  '
              f'(ETA {eta.strftime("%H:%M:%S")})', flush=True)

    def done(self):
        self.n = self.total
        self.update()


def load_ip2p():
    """加载 InstructPix2Pix 模型."""
    from diffusers import StableDiffusionInstructPix2PixPipeline
    print('[IP2P] 加载 InstructPix2Pix...')
    t0 = time.time()
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        'timbrooks/instruct-pix2pix',
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    print(f'[IP2P] 加载完成 ({time.time()-t0:.0f}s)')
    return pipe


def edit_ip2p(pipe, img, instruction, bbox, steps=30, seed=0,
              guidance_scale=7.5, image_guidance_scale=1.5):
    """用 InstructPix2Pix 执行编辑, bbox 外保持原图."""
    gen = torch.Generator(device='cuda').manual_seed(seed)
    out = pipe(
        prompt=instruction,
        image=Image.fromarray(img),
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        generator=gen,
    ).images[0]
    result = np.array(out.convert('RGB'))

    # bbox 外保持原图 (局部编辑)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v * (w if i % 2 == 0 else h)) for i, v in enumerate(bbox)]
    x2, y2 = max(x2, x1 + 8), max(y2, y1 + 8)
    mask = np.zeros((h, w, 3), dtype=bool)
    mask[y1:y2, x1:x2] = True
    result = np.where(mask, result, img).astype(np.uint8)
    return result, mask[:, :, 0]


def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num', type=int, default=200, help='生成攻击图数量')
    ap.add_argument('--steps', type=int, default=30, help='IP2P 推理步数')
    ap.add_argument('--imgsize', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--guidance', type=float, default=7.5, help='guidance_scale')
    ap.add_argument('--img-guidance', type=float, default=1.5, help='image_guidance_scale')
    args = ap.parse_args()

    load_env()
    t0 = time.time()

    # 收集图片
    paths = collect_source('coco', args.num)
    if len(paths) < args.num:
        print(f'⚠️ 只找到 {len(paths)} 张图 (请求 {args.num})')
    print(f'生成 {len(paths)} 张 InstructPix2Pix 攻击图')

    # 加载模型
    pipe = load_ip2p()

    # MLLM
    mimo_key = os.environ.get('MIMO_API_KEY')
    mimo_url = os.environ.get('MIMO_BASE_URL', '')
    mimo_model = os.environ.get('MIMO_MODEL', 'mimo-v2.5')

    # 输出目录
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(RESULTS_DIR, f'attackset_ip2p_{ts}')
    img_dir = os.path.join(out_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    meta = {
        'n': len(paths),
        'imgsize': args.imgsize,
        'steps': args.steps,
        'seed': args.seed,
        'guidance_scale': args.guidance,
        'image_guidance_scale': args.img_guidance,
        'generator': 'instruct-pix2pix',
        'mllm_model': mimo_model,
        'generated': ts,
        'samples': [],
    }

    rng = np.random.RandomState(args.seed)
    prog = Progress(len(paths), 'ip2p-attack')

    success = 0
    fail = 0
    psnrs = []

    for idx, p in enumerate(paths):
        img = load(p, args.imgsize)
        img_name = os.path.basename(p)

        # 1) MLLM 生成编辑指令
        try:
            edit = mllm_edit_instruct(mimo_url, mimo_key, mimo_model, img)
            instruction = edit['instruction']
            bbox = edit['bbox']
        except Exception as e:
            print(f'  ! MLLM 失败: {img_name}: {repr(e)[:60]}')
            fail += 1
            prog.update()
            continue

        # 2) IP2P 执行编辑
        try:
            att, mask = edit_ip2p(
                pipe, img, instruction, bbox,
                steps=args.steps, seed=idx,
                guidance_scale=args.guidance,
                image_guidance_scale=args.img_guidance,
            )
        except Exception as e:
            print(f'  ! IP2P 失败: {img_name}: {repr(e)[:60]}')
            fail += 1
            prog.update()
            continue

        # 3) 保存
        prefix = f'{idx:04d}'
        Image.fromarray(img).save(os.path.join(img_dir, f'{prefix}_orig.png'))
        Image.fromarray(att).save(os.path.join(img_dir, f'{prefix}_att.png'))
        cv2.imwrite(os.path.join(img_dir, f'{prefix}_mask.png'),
                    (mask * 255).astype(np.uint8))

        p_val = psnr(img, att)
        psnrs.append(p_val)

        meta['samples'].append({
            'id': prefix,
            'source': img_name,
            'instruction': instruction,
            'bbox': bbox,
            'psnr': p_val,
        })

        success += 1
        if success % 10 == 0:
            print(f'  [{success}/{len(paths)}] {img_name[:20]} '
                  f'inst="{instruction[:30]}" PSNR={p_val:.1f}dB')

        # 增量保存 meta
        with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        prog.update()

    prog.done()

    # 汇总
    print(f'\n{"="*60}')
    print(f'  InstructPix2Pix 攻击集生成完成')
    print(f'{"="*60}')
    print(f'  成功: {success}/{len(paths)}  失败: {fail}')
    print(f'  平均 PSNR: {np.mean(psnrs):.1f} dB')
    print(f'  输出目录: {out_dir}')
    print(f'  耗时: {time.time()-t0:.0f}s')

    # 保存最终 meta
    meta['summary'] = {
        'success': success,
        'fail': fail,
        'mean_psnr': float(np.mean(psnrs)),
        'total_time_s': round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f'\n  下一步: 用新攻击集跑信号校验实验')
    print(f'  python run_signal_ra.py --attack-dir {out_dir}')


if __name__ == '__main__':
    main()
