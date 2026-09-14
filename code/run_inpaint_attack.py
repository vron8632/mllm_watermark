#!/usr/bin/env python3
"""ICASSP 2027 — SD Inpainting 攻击测试（方向 A 的最大风险验证）.

背景: removal-aware voting 依赖"提取时重检测, 识别被移除的实例区域并剔除其投票".
此前测试的局部移除攻击用背景均值填充; 更真实/更强的攻击是用 Stable Diffusion
Inpainting 把实例"语义级抹除"（填充自然背景）. 若 inpaint 填充后 YOLO 检测不到原实例,
RA 判定依然有效; 若填充出"新物体", RA 可能失效.

本脚本:
  1) 对水印图做 SD inpaint 攻击（掩膜 = YOLO 实例掩膜, 复用 detect_removal_mask）
  2) 对比 uniform / semantic / semantic_ra 在 inpaint 攻击下的 BA (256-bit, 等 PSNR)
  3) 报告 RA 判定质量: 被 inpaint 的实例块中被正确剔除投票的比例

用法 (模型在 HF 本地缓存, 离线可用):
  cd code
  python run_inpaint_attack.py --n 20            # 快速验证 (默认 20 张)
  python run_inpaint_attack.py --n 100           # 全量
  python run_inpaint_attack.py --n 20 --steps 10 # 更快的 SD 推理 (质量略降)

输出: ../results/inpaint_attack_<时间戳>.json
"""
import sys, os, time, json, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from run_idea_probe import (load, detect_removal_mask, embed_scheme, decode_scheme,
                            SCHEMES, psnr, ssim, bit_accuracy, collect_source)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = f'{BASE}/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_ID = 'runwayml/stable-diffusion-inpainting'
PROMPT = 'a realistic photo, natural lighting, consistent style'
NEGATIVE = 'blurry, unnatural, artifact, painting, cartoon, text, watermark'


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


_CACHED_PIPE = None


def load_pipe():
    """离线加载 SD inpainting (HF 缓存 7.7GB). 使用全局缓存避免重复加载."""
    global _CACHED_PIPE
    if _CACHED_PIPE is not None:
        return _CACHED_PIPE
    print(f'[SD] 加载 {MODEL_ID} (本地缓存)...')
    t0 = time.time()
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    print(f'[SD] 模型加载完成 ({time.time()-t0:.0f}s)')
    _CACHED_PIPE = pipe
    return pipe


def inpaint_attack(pipe, img, m_total, steps, seed=0):
    """SD inpaint: 用掩膜 m_total 覆盖的区域语义级填充. img: uint8 RGB (H,W,3)."""
    h, w = img.shape[:2]
    # 膨胀掩膜 5px, 避免实例边缘残留
    m = (m_total.astype(np.uint8) * 255)
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    mask_img = Image.fromarray(m).resize((w, h), Image.NEAREST)
    gen = torch.Generator(device='cuda').manual_seed(seed)
    out = pipe(prompt=PROMPT, negative_prompt=NEGATIVE, image=Image.fromarray(img),
               mask_image=mask_img, height=h, width=w,
               num_inference_steps=steps, guidance_scale=7.5,
               generator=gen).images[0]
    return np.array(out.convert('RGB'))


def block_diff_diag(att, wm, m_total, thresh=6.0):
    """破坏范围诊断: 掩膜内/外 8x8 块被明显改写 (逐块 MAE > thresh) 的比例.

    返回 (frac_changed_inside, frac_changed_outside). 若掩膜外也被大量改写,
    说明 SD inpaint 是全局扰动 (RA 无法通过剔除局部块修复).
    """
    a = att.astype(np.float32)
    b = wm.astype(np.float32)
    H, W = a.shape[:2]
    n_h, n_w = H // 8, W // 8
    m8 = cv2.resize(m_total.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    in_chg = out_chg = in_tot = out_tot = 0
    for i in range(n_h):
        for j in range(n_w):
            blk = m8[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            inside = blk.mean() > 0.5
            mae = np.abs(a[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
                         - b[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]).mean()
            if inside:
                in_tot += 1
                in_chg += mae > thresh
            else:
                out_tot += 1
                out_chg += mae > thresh
    return (in_chg / max(in_tot, 1), out_chg / max(out_tot, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20, help='COCO 图片数 (默认 20)')
    ap.add_argument('--steps', type=int, default=30, help='SD 推理步数 (默认 30)')
    ap.add_argument('--msglen', type=int, default=256, help='消息长度 (默认 256)')
    ap.add_argument('--imgsize', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--rem-frac', type=float, default=0.5,
                    help='移除面积比例 (默认 0.5; 0 = 移除全部检测实例)')
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    t0 = time.time()
    paths = collect_source('coco', args.n)
    print(f'数据集 coco: {len(paths)} 张, 消息 {args.msglen}-bit, SD steps={args.steps}')

    # 方案: uniform Δ19 / semantic Δ15 / semantic_ra Δ15 (等 PSNR, 与 removal 协议一致)
    schemes = {k: SCHEMES[k] for k in ['uniform', 'semantic', 'semantic_ra']}
    delta = {'uniform': 19, 'semantic': 15, 'semantic_ra': 15}

    pipe = load_pipe()
    R = {'meta': {'n': args.n, 'steps': args.steps, 'msglen': args.msglen,
                  'imgsize': args.imgsize, 'seed': args.seed,
                  'rem_frac': args.rem_frac,
                  'ts': datetime.datetime.now().isoformat()},
         'per_image': [], 'summary': {}}

    prog = Progress(len(paths), 'inpaint-attack')
    for idx, p in enumerate(paths):
        img = load(p, args.imgsize)
        msg = rng.randint(0, 2, args.msglen).astype(np.uint8)
        # 实例掩膜 (攻击目标); rem_frac=0 时移除全部检测实例
        frac = args.rem_frac if args.rem_frac > 0 else 1.0
        m_total = detect_removal_mask(img, frac)
        if m_total is None:
            prog.update()
            continue
        row = {'img': os.path.basename(p)}
        wm_by_scheme, sm_by_scheme = {}, {}
        for sname, (sm_fn, dec) in schemes.items():
            sm = None if sm_fn is None else sm_fn(img)
            sm_by_scheme[sname] = sm
            wm_by_scheme[sname] = embed_scheme(img, msg, delta[sname], sm, dec)
        # SD inpaint 攻击 (基于 uniform 方案的 wm, 与方案无关的全局扰动诊断)
        att = inpaint_attack(pipe, wm_by_scheme['uniform'], m_total, args.steps, seed=idx)
        in_chg, out_chg = block_diff_diag(att, wm_by_scheme['uniform'], m_total)
        row['inpaint_area_frac'] = float(m_total.mean())
        row['changed_inside_frac'] = float(in_chg)
        row['changed_outside_frac'] = float(out_chg)
        for sname, (sm_fn, dec) in schemes.items():
            wm = wm_by_scheme[sname]
            if dec == 'adaptive_ra':
                me = decode_scheme(att, msg, delta[sname], sm_by_scheme[sname], dec,
                                   sm_new=sm_fn(att))
            else:
                me = decode_scheme(att, msg, delta[sname], sm_by_scheme[sname], dec)
            row[sname] = {'ba': bit_accuracy(msg, me), 'psnr': psnr(img, wm),
                          'ssim': ssim(img, wm)}
        R['per_image'].append(row)
        prog.update()
    prog.done()

    # 汇总 + 判据
    print('\n' + '=' * 70)
    print('  SD Inpainting 攻击结果 (BA%, 等 PSNR: uniform Δ19 / 语义族 Δ15)')
    print('=' * 70)
    summary = {}
    for sname in schemes:
        bas = [r[sname]['ba'] for r in R['per_image'] if sname in r]
        summary[sname] = {'ba': float(np.mean(bas)), 'ba_std': float(np.std(bas)),
                          'n': len(bas)}
        print(f'  {sname:<12} BA={np.mean(bas)*100:6.1f}% ± {np.std(bas)*100:4.1f}  (n={len(bas)})')
    u = summary['uniform']['ba']
    ra = summary['semantic_ra']['ba']
    in_chg = np.mean([r['changed_inside_frac'] for r in R['per_image']])
    out_chg = np.mean([r['changed_outside_frac'] for r in R['per_image']])
    ok = (ra - u) >= 0.03
    print(f'\n  破坏范围诊断: 掩膜内被改块 {in_chg*100:.0f}% / 掩膜外被改块 {out_chg*100:.0f}%')
    if out_chg > 0.5:
        print('  ⚠️ 掩膜外也被大量改写 → SD inpaint 近似全局扰动, RA 的"局部剔除"策略无解;')
        print('     建议: 提高 SD 分辨率(512) / 更精准掩膜 / 或接受该攻击为方法上限并写入 Limitations')
    print(f'\n  判定: semantic_ra vs uniform = {(ra-u)*100:+.1f}pt  '
          f'→ {"✅ RA 在 SD inpainting 下仍成立" if ok else "❌ RA 未显示增益 (见破坏范围诊断)"}')
    summary['changed_inside_frac'] = float(in_chg)
    summary['changed_outside_frac'] = float(out_chg)
    summary['verdict_A_holds_under_inpaint'] = bool(ok)
    R['summary'] = summary
    R['total_time_s'] = round(time.time() - t0, 1)

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(RESULTS_DIR, f'inpaint_attack_{ts}.json')
    with open(out, 'w') as f:
        json.dump(R, f, indent=2, default=float)
    print(f'\n结果已保存: {out}  (总耗时 {R["total_time_s"]:.0f}s)')


if __name__ == '__main__':
    main()
