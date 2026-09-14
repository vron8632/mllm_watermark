#!/usr/bin/env python3
"""SOTA 方法对比实验: 在相同数据集和配置下测试多种水印方法

测试方法:
1. DWT-DCT (传统基线)
2. Signal-check v2 (我们的方法)
3. Robust-Wide (ECCV 2024)
4. Stable Signature (ICCV 2023)
5. TrustMark (ICCV 2025)
6. Watermark Anything (ICLR 2025)

数据集: COCO val2017 (n=200)
攻击: MLLM + IP2P
指标: Bit Accuracy (BA)

用法:
  cd code
  conda activate apjf
  python run_sota_comparison.py --n 200 --steps 30
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


def load_trustmark_model():
    """加载 TrustMark 模型"""
    try:
        import sys
        sys.path.insert(0, os.path.join(BASELINES_DIR, 'trustmark/python'))
        from trustmark import TrustMark
        tm = TrustMark(model_type='Q', device='cpu', loadRemover=False, loadBBoxDetector=False)
        return tm, 100  # TrustMark 默认使用 100-bit 水印
    except Exception as e:
        print(f"  [!] 加载 TrustMark 失败: {e}")
        return None, 0


def load_stable_signature_model():
    """加载 Stable Signature 模型 (encoder-decoder)"""
    try:
        # 添加 hidden 模块路径
        hidden_dir = os.path.join(BASELINES_DIR, 'stable_signature/hidden')
        if hidden_dir not in sys.path:
            sys.path.insert(0, hidden_dir)
        
        from models import HiddenEncoder, HiddenDecoder
        
        # 加载 hidden_replicate.pth (包含 encoder 和 decoder)
        ckpt_path = os.path.join(BASELINES_DIR, 'stable_signature/hidden/ckpts/hidden_replicate.pth')
        if not os.path.exists(ckpt_path):
            print(f"  [!] Stable Signature 模型不存在: {ckpt_path}")
            return None, 0
        
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        params = checkpoint['params']
        
        # 创建 encoder 和 decoder
        encoder = HiddenEncoder(
            num_blocks=params.encoder_depth,
            num_bits=params.num_bits,
            channels=params.encoder_channels,
            last_tanh=params.use_tanh
        )
        decoder = HiddenDecoder(
            num_blocks=params.decoder_depth,
            num_bits=params.num_bits,
            channels=params.decoder_channels
        )
        
        # 加载权重
        state_dict = checkpoint['encoder_decoder']
        # 移除 'module.' 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '')
            new_state_dict[new_key] = v
        
        # 分离 encoder 和 decoder 权重
        enc_keys = {k: v for k, v in new_state_dict.items() if k.startswith('encoder.')}
        dec_keys = {k: v for k, v in new_state_dict.items() if k.startswith('decoder.')}
        
        # 移除前缀
        enc_keys = {k.replace('encoder.', ''): v for k, v in enc_keys.items()}
        dec_keys = {k.replace('decoder.', ''): v for k, v in dec_keys.items()}
        
        encoder.load_state_dict(enc_keys)
        decoder.load_state_dict(dec_keys)
        
        encoder.eval()
        decoder.eval()
        encoder = encoder.to('cuda')
        decoder = decoder.to('cuda')
        
        # 返回一个简单的容器
        class StableSignatureModel:
            def __init__(self, encoder, decoder, num_bits, scaling_w):
                self.encoder = encoder
                self.decoder = decoder
                self.num_bits = num_bits
                self.scaling_w = scaling_w
        
        model = StableSignatureModel(encoder, decoder, params.num_bits, params.scaling_w)
        return model, params.num_bits
    except Exception as e:
        print(f"  [!] 加载 Stable Signature 失败: {e}")
        import traceback
        traceback.print_exc()
        return None, 0


def load_watermark_anything_model():
    """加载 Watermark Anything 模型"""
    try:
        import os
        original_dir = os.getcwd()
        os.chdir(os.path.join(BASELINES_DIR, 'watermark-anything'))
        
        sys.path.insert(0, os.path.join(BASELINES_DIR, 'watermark-anything'))
        from notebooks.inference_utils import load_model_from_checkpoint
        
        json_path = 'checkpoints/params.json'
        checkpoint_path = 'checkpoints/wam_mit.pth'
        
        if not os.path.exists(json_path) or not os.path.exists(checkpoint_path):
            print(f"  [!] Watermark Anything 配置或模型不存在")
            os.chdir(original_dir)
            return None, 0
        
        model = load_model_from_checkpoint(json_path, checkpoint_path)
        model.eval()
        model = model.to('cuda')
        
        os.chdir(original_dir)
        return model, 32  # Watermark Anything 使用 32-bit 水印
    except Exception as e:
        print(f"  [!] 加载 Watermark Anything 失败: {e}")
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


def trustmark_embed(model, img, msg):
    """TrustMark 水印嵌入"""
    if model is None:
        return None
    
    try:
        # 转换为 PIL Image
        img_pil = Image.fromarray(img)
        
        # 嵌入水印 - TrustMark.encode() 参数是 (in_cover_image, string_secret)
        msg_str = ''.join([str(b) for b in msg])
        wm_pil = model.encode(img_pil, msg_str)
        
        return np.array(wm_pil)
    except Exception as e:
        print(f"    [!] TrustMark 嵌入失败: {e}")
        return None


def trustmark_extract(model, watermarked, msg_len):
    """TrustMark 水印提取"""
    if model is None:
        return None
    
    try:
        img_pil = Image.fromarray(watermarked)
        # TrustMark.decode() 返回 (secret_pred, detected, version)
        secret_pred, detected, version = model.decode(img_pil)
        
        if not detected or not secret_pred:
            return None
        
        # secret_pred 是字符串，需要转换为二进制数组
        # TrustMark 使用 ECC，实际输出长度可能与输入不同
        extracted = np.array([int(b) for b in secret_pred])
        return extracted
    except Exception as e:
        print(f"    [!] TrustMark 提取失败: {e}")
        return None


def stable_signature_embed(model, img, msg):
    """Stable Signature 水印嵌入"""
    if model is None:
        return None
    
    try:
        # 转换为 tensor: (1, 3, H, W) normalized to [-1, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        img_tensor = img_tensor.to('cuda')
        
        # 转换消息为 tensor: (1, num_bits)
        # Stable Signature 使用 48-bit 消息
        msg_tensor = torch.tensor([msg[:model.num_bits]], dtype=torch.float32).to('cuda')
        
        # 嵌入水印
        with torch.no_grad():
            # encoder 输出是残差，需要加到原图上
            residual = model.encoder(img_tensor, msg_tensor)
            watermarked = img_tensor + model.scaling_w * residual
            watermarked = torch.clamp(watermarked, -1, 1)
        
        # 转换回 numpy: (H, W, 3) uint8
        wm_np = (watermarked.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
        wm_np = np.clip(wm_np, 0, 255).astype(np.uint8)
        
        return wm_np
    except Exception as e:
        print(f"    [!] Stable Signature 嵌入失败: {e}")
        return None


def stable_signature_extract(model, watermarked, msg_len):
    """Stable Signature 水印提取"""
    if model is None:
        return None
    
    try:
        # 转换为 tensor: (1, 3, H, W) normalized to [-1, 1]
        img_tensor = torch.from_numpy(watermarked).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        img_tensor = img_tensor.to('cuda')
        
        with torch.no_grad():
            # decoder 输出是 logits
            logits = model.decoder(img_tensor)
            # 阈值化得到二进制消息
            bits = (logits > 0).squeeze().cpu().numpy().astype(int)
        
        return bits[:msg_len]
    except Exception as e:
        print(f"    [!] Stable Signature 提取失败: {e}")
        return None


def watermark_anything_embed(model, img, msg):
    """Watermark Anything 水印嵌入"""
    if model is None:
        return None
    
    try:
        # 转换为 tensor: (1, 3, H, W) normalized to [-1, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        img_tensor = img_tensor.to('cuda')
        
        # 转换消息为 tensor: (1, nbits)
        msg_tensor = torch.tensor([msg], dtype=torch.float32).to('cuda')
        
        # 嵌入水印
        with torch.no_grad():
            outputs = model.embed(img_tensor, msg_tensor)
            watermarked = outputs['imgs_w']
        
        # 转换回 numpy: (H, W, 3) uint8
        wm_np = (watermarked.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
        wm_np = np.clip(wm_np, 0, 255).astype(np.uint8)
        
        return wm_np
    except Exception as e:
        print(f"    [!] Watermark Anything 嵌入失败: {e}")
        return None


def watermark_anything_extract(model, watermarked, msg_len):
    """Watermark Anything 水印提取"""
    if model is None:
        return None
    
    try:
        # 转换为 tensor: (1, 3, H, W) normalized to [-1, 1]
        img_tensor = torch.from_numpy(watermarked).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        img_tensor = img_tensor.to('cuda')
        
        with torch.no_grad():
            outputs = model.detect(img_tensor)
            preds = outputs['preds']  # (1, 1+nbits, H, W)
        
        # preds[:, 0] 是 mask, preds[:, 1:] 是 bits
        # 对每个 bit 通道取平均并阈值化
        bits = preds[0, 1:, :, :].mean(dim=(1, 2)) > 0  # (nbits,)
        return bits.cpu().numpy().astype(int)[:msg_len]
    except Exception as e:
        print(f"    [!] Watermark Anything 提取失败: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--msglen', type=int, default=256)
    ap.add_argument('--delta', type=int, default=15)
    ap.add_argument('--datasets', default='coco,div2k', help='逗号分隔: coco,div2k,kodak')
    args = ap.parse_args()
    load_env()
    
    t0 = time.time()
    
    # 加载数据
    paths = []
    for ds in args.datasets.split(','):
        ds = ds.strip()
        ds_paths = collect_source(ds, args.n)
        paths.extend(ds_paths)
        print(f'{ds}: {len(ds_paths)} 张')
    print(f'总计: {len(paths)} 张图片')
    rng = np.random.RandomState(42)
    
    # 加载所有模型
    print("加载模型...")
    robust_wide_model, rw_msg_len = load_robust_wide_model()
    trustmark_model, tm_msg_len = load_trustmark_model()
    stable_sig_model, ss_msg_len = load_stable_signature_model()
    watermark_anything_model, wa_msg_len = load_watermark_anything_model()
    
    print(f"  Robust-Wide: {'OK' if robust_wide_model else 'FAIL'} (msg_len={rw_msg_len})")
    print(f"  TrustMark: {'OK' if trustmark_model else 'FAIL'} (msg_len={tm_msg_len})")
    print(f"  Stable Signature: {'OK' if stable_sig_model else 'FAIL'} (msg_len={ss_msg_len})")
    print(f"  Watermark Anything: {'OK' if watermark_anything_model else 'FAIL'} (msg_len={wa_msg_len})")
    
    # 结果存储
    results = {
        'meta': {
            'n': args.n,
            'msglen': args.msglen,
            'delta': args.delta,
            'steps': args.steps,
            'ts': datetime.datetime.now().isoformat()
        },
        'methods': {
            'dwt_dct': {'clean_ba': [], 'attacked_ba': []},
            'signal_check_v2': {'clean_ba': [], 'attacked_ba': []},
            'robust_wide': {'clean_ba': [], 'attacked_ba': []},
            'trustmark': {'clean_ba': [], 'attacked_ba': []},
            'stable_signature': {'clean_ba': [], 'attacked_ba': []},
            'watermark_anything': {'clean_ba': [], 'attacked_ba': []},
        },
        'per_image': []
    }
    
    print(f"\n开始对比实验: n={args.n}, msglen={args.msglen}")
    print("=" * 60)
    
    for idx, p in enumerate(paths):
        img = load(p, 256)
        msg_long = rng.randint(0, 2, args.msglen).astype(np.uint8)
        
        row = {'img': os.path.basename(p)}
        
        try:
            # 1. 嵌入水印
            wm_dwt = dwt_dct_embed(img, msg_long, args.delta)
            wm_sc = signal_check_embed(img, msg_long, args.delta)
            wm_rw = robust_wide_embed(robust_wide_model, img, msg_long[:rw_msg_len]) if rw_msg_len > 0 else None
            wm_tm = trustmark_embed(trustmark_model, img, msg_long[:tm_msg_len]) if tm_msg_len > 0 else None
            wm_ss = stable_signature_embed(stable_sig_model, img, msg_long[:ss_msg_len]) if ss_msg_len > 0 else None
            wm_wa = watermark_anything_embed(watermark_anything_model, img, msg_long[:wa_msg_len]) if wa_msg_len > 0 else None
            
            row['psnr_dwt'] = psnr(img, wm_dwt)
            row['psnr_sc'] = psnr(img, wm_sc)
            if wm_rw is not None:
                row['psnr_rw'] = psnr(img, wm_rw)
            if wm_tm is not None:
                row['psnr_tm'] = psnr(img, wm_tm)
            
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
                    row['clean_ba_rw'] = bit_accuracy(msg_long[:rw_msg_len], me_rw)
                    results['methods']['robust_wide']['clean_ba'].append(row['clean_ba_rw'])
            
            if wm_tm is not None:
                me_tm = trustmark_extract(trustmark_model, wm_tm, tm_msg_len)
                if me_tm is not None and len(me_tm) > 0:
                    # TrustMark 使用 ECC，实际输出长度可能与输入不同
                    # 取前 min(tm_msg_len, len(me_tm)) 位进行比较
                    compare_len = min(len(msg_long), len(me_tm))
                    row['clean_ba_tm'] = bit_accuracy(msg_long[:compare_len], me_tm[:compare_len])
                    results['methods']['trustmark']['clean_ba'].append(row['clean_ba_tm'])
            
            if wm_ss is not None:
                me_ss = stable_signature_extract(stable_sig_model, wm_ss, ss_msg_len)
                if me_ss is not None:
                    row['clean_ba_ss'] = bit_accuracy(msg_long[:ss_msg_len], me_ss)
                    results['methods']['stable_signature']['clean_ba'].append(row['clean_ba_ss'])
            
            if wm_wa is not None:
                me_wa = watermark_anything_extract(watermark_anything_model, wm_wa, wa_msg_len)
                if me_wa is not None:
                    row['clean_ba_wa'] = bit_accuracy(msg_long[:wa_msg_len], me_wa)
                    results['methods']['watermark_anything']['clean_ba'].append(row['clean_ba_wa'])
            
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
            if wm_tm is not None:
                att_tm = sd_execute(edit, wm_tm, steps=args.steps)
            if wm_ss is not None:
                att_ss = sd_execute(edit, wm_ss, steps=args.steps)
            if wm_wa is not None:
                att_wa = sd_execute(edit, wm_wa, steps=args.steps)
            
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
                    row['attacked_ba_rw'] = bit_accuracy(msg_long[:rw_msg_len], me_rw)
                    results['methods']['robust_wide']['attacked_ba'].append(row['attacked_ba_rw'])
            
            if wm_tm is not None:
                me_tm = trustmark_extract(trustmark_model, att_tm, tm_msg_len)
                if me_tm is not None and len(me_tm) > 0:
                    # TrustMark 使用 ECC，实际输出长度可能与输入不同
                    compare_len = min(len(msg_long), len(me_tm))
                    row['attacked_ba_tm'] = bit_accuracy(msg_long[:compare_len], me_tm[:compare_len])
                    results['methods']['trustmark']['attacked_ba'].append(row['attacked_ba_tm'])
            
            if wm_ss is not None:
                me_ss = stable_signature_extract(stable_sig_model, att_ss, ss_msg_len)
                if me_ss is not None:
                    row['attacked_ba_ss'] = bit_accuracy(msg_long[:ss_msg_len], me_ss)
                    results['methods']['stable_signature']['attacked_ba'].append(row['attacked_ba_ss'])
            
            if wm_wa is not None:
                me_wa = watermark_anything_extract(watermark_anything_model, att_wa, wa_msg_len)
                if me_wa is not None:
                    row['attacked_ba_wa'] = bit_accuracy(msg_long[:wa_msg_len], me_wa)
                    results['methods']['watermark_anything']['attacked_ba'].append(row['attacked_ba_wa'])
            
        except Exception as e:
            row['error'] = repr(e)[:120]
            print(f"  ! 图 {os.path.basename(p)} 失败: {row['error']}")
        
        results['per_image'].append(row)
        
        # 打印进度
        if 'error' not in row:
            ba_dwt = row.get('attacked_ba_dwt', 0) * 100
            ba_sc = row.get('attacked_ba_sc', 0) * 100
            ba_rw = row.get('attacked_ba_rw', 0) * 100 if 'attacked_ba_rw' in row else None
            ba_tm = row.get('attacked_ba_tm', 0) * 100 if 'attacked_ba_tm' in row else None
            
            rw_str = f"RW={ba_rw:.0f}%" if ba_rw is not None else "RW=N/A"
            tm_str = f"TM={ba_tm:.0f}%" if ba_tm is not None else "TM=N/A"
            print(f"  [{idx+1}/{args.n}] {os.path.basename(p)[:18]} "
                  f"DWT={ba_dwt:.0f}% SC={ba_sc:.0f}% {rw_str} {tm_str}")
        
        # 增量保存
        with open('../results/sota_comparison.tmp.json', 'w') as f:
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
    out = f'../results/sota_comparison_{ts}.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\n保存: {out} (耗时 {time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
