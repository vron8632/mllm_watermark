#!/usr/bin/env python3
"""消融实验合集: 在同一子集上跑3组消融.

1. Delta sensitivity (Δ=10/15/20/25, n=50)
2. IP2P vs SD1.5 攻击对比 (n=50)
3. Guidance scale 影响 (7.5/10/15, n=50)

统一 seed=42, 同一组图片, 同一组编辑指令.
输出 results/ablation_<ts>.json
"""
import sys, os, time, json, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from PIL import Image
from run_idea_probe import (load, collect_source, psnr, bit_accuracy, y_channel,
                            _block_grid, _demodulate_bit)
from run_signal_ra import embed_signal, decode_signal
from run_mllm_benchmark import load_env, mllm_edit_instruct, sd_execute
from dct_watermark import _majority_vote

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dwt_dct_embed(img, msg, delta=15):
    H, W, n_h, n_w = _block_grid(*img.shape[:2])
    y = y_channel(img)[:H, :W]
    for i in range(n_h):
        for j in range(n_w):
            d = cv2.dct(y[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32))
            bi = (i * n_w + j) % len(msg)
            c = d[4, 1]
            if msg[bi] == 1:
                d[4, 1] = (np.floor(c/delta)*delta + delta/2)
            else:
                d[4, 1] = np.round(c/delta)*delta
            y[i*8:(i+1)*8, j*8:(j+1)*8] = cv2.idct(d)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    out = ycrcb.copy()
    out[:H, :W, 0] = np.clip(y, 0, 255)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB).astype(np.uint8)


def dwt_dct_extract(watermarked, msg_len, delta=15):
    H, W, n_h, n_w = _block_grid(*watermarked.shape[:2])
    y = y_channel(watermarked)[:H, :W]
    votes = [[] for _ in range(msg_len)]
    for i in range(n_h):
        for j in range(n_w):
            d = cv2.dct(y[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32))
            bi = (i * n_w + j) % msg_len
            pay = _demodulate_bit(d[4, 1], delta)
            votes[bi].append((pay, 1.0))
    return _majority_vote(votes, msg_len)


import cv2


def run_ablation_delta(n, deltas, msglen, steps, edits, paths, rng):
    """消融1: 不同 Δ 的影响"""
    print(f'\n{"="*60}')
    print(f'消融1: Delta sensitivity (n={n}, deltas={deltas})')
    print(f'{"="*60}')
    results = {}
    for delta in deltas:
        print(f'\n  Δ={delta}:')
        delta_data = {'uniform_ba': [], 'sc_v2_ba': []}
        for idx, (p, edit) in enumerate(zip(paths, edits)):
            img = load(p, 256)
            msg = rng.randint(0, 2, msglen).astype(np.uint8)
            wm = dwt_dct_embed(img, msg, delta)
            att = sd_execute(edit, wm, steps=steps)
            # uniform
            me_u = dwt_dct_extract(att, msglen, delta)
            delta_data['uniform_ba'].append(bit_accuracy(msg, me_u))
            # sc_v2
            me_v2, _ = decode_signal(att, msglen, delta, use_check=True, soft=True, dilate=1)
            delta_data['sc_v2_ba'].append(bit_accuracy(msg, me_v2))
            if (idx+1) % 10 == 0:
                u = np.mean(delta_data['uniform_ba'])*100
                v = np.mean(delta_data['sc_v2_ba'])*100
                print(f'    [{idx+1}/{n}] uniform={u:.1f}%  sc_v2={v:.1f}%')
        u_mean = np.mean(delta_data['uniform_ba'])*100
        v_mean = np.mean(delta_data['sc_v2_ba'])*100
        print(f'  Δ={delta} 完成: uniform={u_mean:.1f}%  sc_v2={v_mean:.1f}%')
        results[f'delta_{delta}'] = delta_data
    return results


def run_ablation_ip2p_vs_sd15(n, msglen, delta, steps, edits, paths, rng):
    """消融2: IP2P vs SD1.5 攻击"""
    print(f'\n{"="*60}')
    print(f'消融2: IP2P vs SD1.5 攻击 (n={n})')
    print(f'{"="*60}')
    import cv2

    # 加载 SD1.5 inpainting
    from diffusers import StableDiffusionInpaintPipeline
    sd15_model = 'runwayml/stable-diffusion-inpainting'
    print(f'  加载 SD1.5 Inpainting...')
    pipe15 = StableDiffusionInpaintPipeline.from_pretrained(
        sd15_model, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False,
        local_files_only=True).to('cuda')
    pipe15.set_progress_bar_config(disable=True)
    print(f'  SD1.5 加载完成')

    # IP2P 使用相同的 sd_execute
    results = {'ip2p_ba': [], 'sd15_ba': [], 'ip2p_psnr': [], 'sd15_psnr': []}
    for idx, (p, edit) in enumerate(zip(paths, edits)):
        img = load(p, 256)
        msg = rng.randint(0, 2, msglen).astype(np.uint8)
        wm = dwt_dct_embed(img, msg, delta)
        
        # IP2P 攻击
        att_ip2p = sd_execute(edit, wm, steps=steps)
        me_ip2p = dwt_dct_extract(att_ip2p, msglen, delta)
        results['ip2p_ba'].append(bit_accuracy(msg, me_ip2p))
        results['ip2p_psnr'].append(psnr(img, att_ip2p))
        
        # SD1.5 攻击 (用相同 edit prompt + bbox)
        h, w = img.shape[:2]
        x1, y1, x2, y2 = edit['bbox']
        bx1, by1 = int(x1 * w), int(y1 * h)
        bx2, by2 = max(int(x2 * w), bx1 + 8), max(int(y2 * h), by1 + 8)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[by1:by2, bx1:bx2] = 255
        mask_pil = Image.fromarray(mask)
        gen = torch.Generator(device='cuda').manual_seed(0)
        out15 = pipe15(
            prompt='a realistic photo, natural lighting',
            negative_prompt='blurry, unnatural, artifact',
            image=Image.fromarray(wm),
            mask_image=mask_pil,
            height=h, width=w,
            num_inference_steps=steps,
            guidance_scale=7.5,
            generator=gen
        ).images[0]
        att_sd15 = np.array(out15.convert('RGB'))
        # 掩膜外保持原图
        mask3 = np.stack([mask > 127] * 3, axis=-1)
        att_sd15 = np.where(mask3, att_sd15, wm).astype(np.uint8)
        
        me_sd15 = dwt_dct_extract(att_sd15, msglen, delta)
        results['sd15_ba'].append(bit_accuracy(msg, me_sd15))
        results['sd15_psnr'].append(psnr(img, att_sd15))
        
        if (idx+1) % 10 == 0:
            ip2p = np.mean(results['ip2p_ba'])*100
            sd15 = np.mean(results['sd15_ba'])*100
            print(f'    [{idx+1}/{n}] IP2P={ip2p:.1f}%  SD1.5={sd15:.1f}%')
    
    ip2p_mean = np.mean(results['ip2p_ba'])*100
    sd15_mean = np.mean(results['sd15_ba'])*100
    print(f'  完成: IP2P={ip2p_mean:.1f}%  SD1.5={sd15_mean:.1f}%')
    del pipe15
    torch.cuda.empty_cache()
    return results


def run_ablation_guidance(n, guidance_scales, msglen, delta, steps, edits, paths, rng):
    """消融3: 不同 guidance_scale 的影响"""
    print(f'\n{"="*60}')
    print(f'消融3: Guidance scale sensitivity (n={n}, scales={guidance_scales})')
    print(f'{"="*60}')

    results = {}
    for gs in guidance_scales:
        print(f'\n  guidance_scale={gs}:')
        gs_data = {'ba': [], 'psnr': []}
        for idx, (p, edit) in enumerate(zip(paths, edits)):
            img = load(p, 256)
            msg = rng.randint(0, 2, msglen).astype(np.uint8)
            wm = dwt_dct_embed(img, msg, delta)
            # 自定义 guidance_scale 的 sd_execute
            from run_inpaint_attack import inpaint_attack, load_pipe, PROMPT, NEGATIVE
            from run_mllm_benchmark import detect_removal_mask
            pipe = load_pipe()
            h, w = wm.shape[:2]
            x1, y1, x2, y2 = edit['bbox']
            bx1, by1 = int(x1 * w), int(y1 * h)
            bx2, by2 = max(int(x2 * w), bx1 + 8), max(int(y2 * h), by1 + 8)
            m_total = np.zeros((h, w), dtype=bool)
            m_total[by1:by2, bx1:bx2] = True
            if m_total.sum() < 0.05 * h * w:
                m_inst = detect_removal_mask(wm, 0.5)
                if m_inst is not None:
                    m_total |= m_inst
            
            m_uint8 = (m_total.astype(np.uint8) * 255)
            m_uint8 = cv2.dilate(m_uint8, np.ones((5, 5), np.uint8))
            mask_img = Image.fromarray(m_uint8).resize((w, h), Image.NEAREST)
            gen = torch.Generator(device='cuda').manual_seed(0)
            out = pipe(prompt=PROMPT, negative_prompt=NEGATIVE,
                       image=Image.fromarray(wm),
                       mask_image=mask_img, height=h, width=w,
                       num_inference_steps=steps, guidance_scale=gs,
                       generator=gen).images[0]
            att = np.array(out.convert('RGB'))
            mask3 = np.stack([m_total] * 3, axis=-1)
            att = np.where(mask3, att, wm).astype(np.uint8)
            
            me = dwt_dct_extract(att, msglen, delta)
            gs_data['ba'].append(bit_accuracy(msg, me))
            gs_data['psnr'].append(psnr(img, att))
            
            if (idx+1) % 10 == 0:
                ba = np.mean(gs_data['ba'])*100
                print(f'    [{idx+1}/{n}] BA={ba:.1f}%')
        
        ba_mean = np.mean(gs_data['ba'])*100
        print(f'  guidance={gs} 完成: BA={ba_mean:.1f}%')
        results[f'gs_{gs}'] = gs_data
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--msglen', type=int, default=256)
    ap.add_argument('--delta', type=int, default=15)
    args = ap.parse_args()
    load_env()

    t0 = time.time()
    rng = np.random.RandomState(42)
    paths = collect_source('coco', args.n)
    print(f'数据集: {len(paths)} 张 COCO 图片')

    # 1) 生成统一编辑指令 (n 张图各一次)
    print(f'\n生成 {len(paths)} 张图的编辑指令...')
    edits = []
    for p in paths:
        img = load(p, 256)
        edit = mllm_edit_instruct(os.environ.get('MIMO_BASE_URL', ''),
                                  os.environ.get('MIMO_API_KEY', ''),
                                  os.environ.get('MIMO_MODEL', 'mimo-v2.5'), img)
        edits.append(edit)
    print(f'编辑指令已生成')

    R = {
        'meta': {'n': args.n, 'msglen': args.msglen, 'delta': args.delta,
                 'steps': args.steps, 'ts': datetime.datetime.now().isoformat()},
        'delta_sensitivity': {},
        'ip2p_vs_sd15': {},
        'guidance_scale': {},
    }

    # 消融1: Delta sensitivity
    R['delta_sensitivity'] = run_ablation_delta(
        args.n, [10, 15, 20, 25], args.msglen, args.steps, edits, paths, rng)

    # 消融2: IP2P vs SD1.5
    R['ip2p_vs_sd15'] = run_ablation_ip2p_vs_sd15(
        args.n, args.msglen, args.delta, args.steps, edits, paths, rng)

    # 消融3: Guidance scale
    R['guidance_scale'] = run_ablation_guidance(
        args.n, [7.5, 10.0, 15.0], args.msglen, args.delta, args.steps, edits, paths, rng)

    # 保存
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = f'{BASE}/results/ablation_{ts}.json'
    with open(out, 'w') as f:
        json.dump(R, f, indent=2, default=float)

    # 汇总
    print(f'\n{"="*60}')
    print(f'消融实验汇总')
    print(f'{"="*60}')
    print(f'\n1. Delta sensitivity:')
    for k, v in R['delta_sensitivity'].items():
        u = np.mean(v['uniform_ba'])*100
        s = np.mean(v['sc_v2_ba'])*100
        print(f'  {k}: uniform={u:.1f}%  sc_v2={s:.1f}%')
    print(f'\n2. IP2P vs SD1.5:')
    ip2p = np.mean(R['ip2p_vs_sd15']['ip2p_ba'])*100
    sd15 = np.mean(R['ip2p_vs_sd15']['sd15_ba'])*100
    print(f'  IP2P:  {ip2p:.1f}%')
    print(f'  SD1.5: {sd15:.1f}%')
    print(f'\n3. Guidance scale:')
    for k, v in R['guidance_scale'].items():
        ba = np.mean(v['ba'])*100
        print(f'  {k}: BA={ba:.1f}%')

    print(f'\n保存: {out} (耗时 {time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
