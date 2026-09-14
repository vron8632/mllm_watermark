#!/usr/bin/env python3
"""SOTA 方法对比实验: 在相同数据集和配置下测试多种水印方法

测试方法:
1. DWT-DCT (传统基线)
2. Signal-check v2 (我们的方法)
3. Robust-Wide (ECCV 2024)

数据集: COCO val2017 (n=200)
攻击: MLLM + IP2P
指标: Bit Accuracy (BA)

用法:
  cd code
  conda activate apjf
  python run_benchmark_comparison.py --n 200 --steps 30
"""
import sys, os, time, json, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
from run_idea_probe import (load, collect_source, psnr, bit_accuracy, y_channel,
                            _block_grid, _demodulate_bit)
from run_mllm_benchmark import (load_env, mllm_edit_instruct, sd_execute)
from dct_watermark import _majority_vote
from omegaconf import OmegaConf

# 添加 baselines 目录
BASELINES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'baselines')
ROBUST_WIDE_DIR = os.path.join(BASELINES_DIR, 'Robust-Wide')
sys.path.insert(0, ROBUST_WIDE_DIR)


def load_robust_wide_model():
    """加载 Robust-Wide 模型"""
    try:
        from model import WatermarkModel
        
        ckpt_dir = os.path.join(ROBUST_WIDE_DIR, 'checkpoints')
        config_path = os.path.join(ckpt_dir, 'wm_model_config.yaml')
        model_path = os.path.join(ckpt_dir, 'wm_model.ckpt')
        
        if not os.path.exists(model_path):
            print(f"  [!] Robust-Wide 模型不存在: {model_path}")
            return None, 0
        
        # 加载配置和模型
        wm_model_config = OmegaConf.load(config_path)
        message_length = wm_model_config["wm_enc_config"]["message_length"]
        model = WatermarkModel(**wm_model_config)
        model_ckpt = torch.load(model_path, map_location='cpu')
        model.load_state_dict(model_ckpt)
        model.eval()
        model = model.to('cuda')
        
        return model, message_length
    except Exception as e:
        print(f"  [!] 加载 Robust-Wide 失败: {e}")
        return None, 0


def dwt_dct_embed(img, msg, delta=15):
    """DWT-DCT 水印嵌入 (传统方法)"""
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
    """DWT-DCT 水印提取"""
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


def signal_check_embed(img, msg, delta=15):
    """Signal-check v2 水印嵌入"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_signal_ra import embed_signal
    return embed_signal(img, msg, delta)


def signal_check_extract(watermarked, msg_len, delta=15):
    """Signal-check v2 水印提取"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_signal_ra import decode_signal
    me, _ = decode_signal(watermarked, msg_len, delta, use_check=True, soft=True, dilate=1)
    return me


def robust_wide_embed(model, img, msg):
    """Robust-Wide 水印嵌入"""
    if model is None:
        return None
    
    try:
        size = 512
        transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        # 转换为 PIL Image
        img_pil = Image.fromarray(img)
        img_tensor = transform(img_pil).unsqueeze(0).to('cuda')
        
        # 转换消息为 tensor
        msg_tensor = torch.tensor(msg, dtype=torch.float32).unsqueeze(0).to('cuda')
        
        # 嵌入水印
        with torch.no_grad():
            watermarked = model.encoder(img_tensor, msg_tensor)
        
        # 转换回 numpy
        wm_np = watermarked.squeeze(0).cpu().numpy()
        wm_np = (wm_np.transpose(1, 2, 0) + 1) / 2  # denormalize
        wm_np = (wm_np * 255).clip(0, 255).astype(np.uint8)
        
        # 调整回原始尺寸
        wm_pil = Image.fromarray(wm_np).resize((img.shape[1], img.shape[0]), Image.BILINEAR)
        return np.array(wm_pil)
    except Exception as e:
        print(f"    [!] Robust-Wide 嵌入失败: {e}")
        return None


def robust_wide_extract(model, watermarked, msg_len):
    """Robust-Wide 水印提取"""
    if model is None:
        return None
    
    try:
        size = 512
        transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        img_pil = Image.fromarray(watermarked)
        img_tensor = transform(img_pil).unsqueeze(0).to('cuda')
        
        with torch.no_grad():
            decoded = model.decoder(img_tensor)
        
        # 转换为二进制消息
        extracted = (decoded.squeeze(0) > 0.5).long().cpu().numpy()
        return extracted[:msg_len]
    except Exception as e:
        print(f"    [!] Robust-Wide 提取失败: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--msglen', type=int, default=256)
    ap.add_argument('--delta', type=int, default=15)
    ap.add_argument('--robust-wide-msglen', type=int, default=64,
                    help='Robust-Wide 使用 64-bit 水印')
    args = ap.parse_args()
    load_env()
    
    t0 = time.time()
    
    # 加载数据
    paths = collect_source('coco', args.n)
    rng = np.random.RandomState(42)
    
    # 加载 Robust-Wide 模型
    print("加载 Robust-Wide 模型...")
    robust_wide_model, rw_msg_len = load_robust_wide_model()
    if robust_wide_model:
        print(f"  Robust-Wide 加载成功 (msg_len={rw_msg_len})")
    else:
        print("  Robust-Wide 加载失败，跳过该方法")
    
    # 结果存储
    results = {
        'meta': {
            'n': args.n,
            'msglen': args.msglen,
            'robust_wide_msglen': rw_msg_len,
            'delta': args.delta,
            'steps': args.steps,
            'ts': datetime.datetime.now().isoformat()
        },
        'methods': {
            'dwt_dct': {'clean_ba': [], 'attacked_ba': []},
            'signal_check_v2': {'clean_ba': [], 'attacked_ba': []},
            'robust_wide': {'clean_ba': [], 'attacked_ba': []},
        },
        'per_image': []
    }
    
    print(f"\n开始对比实验: n={args.n}, msglen={args.msglen}")
    print("=" * 60)
    
    for idx, p in enumerate(paths):
        img = load(p, 256)
        msg_long = rng.randint(0, 2, args.msglen).astype(np.uint8)
        msg_short = rng.randint(0, 2, rw_msg_len).astype(np.uint8) if rw_msg_len > 0 else None
        
        row = {'img': os.path.basename(p)}
        
        try:
            # 1. 嵌入水印
            wm_dwt = dwt_dct_embed(img, msg_long, args.delta)
            wm_sc = signal_check_embed(img, msg_long, args.delta)
            wm_rw = robust_wide_embed(robust_wide_model, img, msg_short) if msg_short is not None else None
            
            row['psnr_dwt'] = psnr(img, wm_dwt)
            row['psnr_sc'] = psnr(img, wm_sc)
            if wm_rw is not None:
                row['psnr_rw'] = psnr(img, wm_rw)
            
            # 2. Clean 提取
            me_dwt = dwt_dct_extract(wm_dwt, args.msglen, args.delta)
            row['clean_ba_dwt'] = bit_accuracy(msg_long, me_dwt)
            results['methods']['dwt_dct']['clean_ba'].append(row['clean_ba_dwt'])
            
            me_sc = signal_check_extract(wm_sc, args.msglen, args.delta)
            row['clean_ba_sc'] = bit_accuracy(msg_long, me_sc)
            results['methods']['signal_check_v2']['clean_ba'].append(row['clean_ba_sc'])
            
            if wm_rw is not None:
                me_rw = robust_wide_extract(robust_wide_model, wm_rw, rw_msg_len)
                if me_rw is not None:
                    row['clean_ba_rw'] = bit_accuracy(msg_short, me_rw)
                    results['methods']['robust_wide']['clean_ba'].append(row['clean_ba_rw'])
            
            # 3. MLLM 攻击
            mimo_key = os.environ.get('MIMO_API_KEY')
            mimo_url = os.environ.get('MIMO_BASE_URL', '')
            mimo_model = os.environ.get('MIMO_MODEL', 'mimo-v2.5')
            edit = mllm_edit_instruct(mimo_url, mimo_key, mimo_model, img)
            row['edit_prompt'] = edit['instruction']
            
            att_dwt = sd_execute(edit, wm_dwt, steps=args.steps)
            att_sc = sd_execute(edit, wm_sc, steps=args.steps)
            if wm_rw is not None:
                att_rw = sd_execute(edit, wm_rw, steps=args.steps)
            
            # 4. Attacked 提取
            me_dwt = dwt_dct_extract(att_dwt, args.msglen, args.delta)
            row['attacked_ba_dwt'] = bit_accuracy(msg_long, me_dwt)
            results['methods']['dwt_dct']['attacked_ba'].append(row['attacked_ba_dwt'])
            
            me_sc = signal_check_extract(att_sc, args.msglen, args.delta)
            row['attacked_ba_sc'] = bit_accuracy(msg_long, me_sc)
            results['methods']['signal_check_v2']['attacked_ba'].append(row['attacked_ba_sc'])
            
            if wm_rw is not None:
                me_rw = robust_wide_extract(robust_wide_model, att_rw, rw_msg_len)
                if me_rw is not None:
                    row['attacked_ba_rw'] = bit_accuracy(msg_short, me_rw)
                    results['methods']['robust_wide']['attacked_ba'].append(row['attacked_ba_rw'])
            
        except Exception as e:
            row['error'] = repr(e)[:120]
            print(f"  ! 图 {os.path.basename(p)} 失败: {row['error']}")
        
        results['per_image'].append(row)
        
        # 打印进度
        if 'error' not in row:
            ba_dwt = row.get('attacked_ba_dwt', 0) * 100
            ba_sc = row.get('attacked_ba_sc', 0) * 100
            ba_rw = row.get('attacked_ba_rw', 0) * 100 if 'attacked_ba_rw' in row else None
            
            rw_str = f"RW={ba_rw:.0f}%" if ba_rw is not None else "RW=N/A"
            print(f"  [{idx+1}/{args.n}] {os.path.basename(p)[:18]} "
                  f"DWT={ba_dwt:.0f}% SC={ba_sc:.0f}% {rw_str}")
        
        # 增量保存
        with open('../results/benchmark_comparison.tmp.json', 'w') as f:
            json.dump(results, f, indent=2, default=float)
    
    # 汇总
    print('\n' + '=' * 60)
    print("对比实验结果")
    print('=' * 60)
    
    for method, data in results['methods'].items():
        if data['clean_ba']:
            clean = np.mean(data['clean_ba']) * 100
            attacked = np.mean(data['attacked_ba']) * 100 if data['attacked_ba'] else 0
            print(f"  {method:<20} Clean={clean:6.1f}%  Attacked={attacked:6.1f}%")
    
    # 保存结果
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = f'../results/benchmark_comparison_{ts}.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\n保存: {out} (耗时 {time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
