#!/usr/bin/env python3
"""ICASSP 2027 — OmniGuard (CVPR 2025) 基线对比: 局部实例移除攻击下的版权水印提取.

对比协议: 与 run_idea_probe.py 的 removal 一致 (掩膜基于原图, 等 PSNR 由各自方案决定).
OmniGuard 用其原生 64-bit 消息协议 (训练式, HiNet 可逆网络 + encoder_Q/decoder_Q).

用法:
  cd code && python run_baseline_omniguard.py --n 20     # COCO 前 20 张
  python run_baseline_omniguard.py --n 100 --msglen 64

输出: ../results/baseline_omniguard_<时间戳>.json
"""
import sys, os, time, json, argparse, datetime
import numpy as np
import torch
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = f'{BASE}/results'
os.makedirs(RESULTS_DIR, exist_ok=True)
OMNIGUARD = '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/baselines/OmniGuard'

# 复用本项目的嵌入/攻击工具
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_idea_probe import load, detect_removal_mask, apply_removal, psnr, collect_source


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


def load_model(device):
    """加载 OmniGuard 版权水印模型 (Model 自动加载 encoder_Q/decoder_Q + 主权重)."""
    cwd = os.getcwd()
    os.chdir(OMNIGUARD)  # Model() 内硬编码相对路径 checkpoint/encoder_Q.ckpt
    sys.path.insert(0, OMNIGUARD)
    import config as c
    from model_invert import Model, init_model
    from modules import Unet_common as common

    net = Model()
    init_model(net)
    net = torch.nn.DataParallel(net, device_ids=c.device_ids)
    net = net.to(device)
    # 主权重 (与官方 test_invert 一致): state_dicts['net'] 过滤 tmp_var/bm
    sd = torch.load(os.path.join(OMNIGUARD, c.MODEL_PATH, c.suffix), map_location='cpu')
    ck = sd.get('net', sd)
    network_state_dict = {k: v for k, v in ck.items()
                          if 'tmp_var' not in k and 'bm' not in k}
    # DataParallel 包装: 尝试带/不带 module. 前缀加载
    try:
        net.module.load_state_dict(network_state_dict, strict=False)
    except Exception:
        net.load_state_dict(network_state_dict, strict=False)
    net.eval()
    os.chdir(cwd)
    return net, common


def bits_acc(msg, bits):
    """OmniGuard 消息为连续 logits, >0 阈值化为 bit; 返回 bit accuracy."""
    pred = (bits > 0).int().flatten().cpu().numpy()
    gt = (msg > 0).int().flatten().cpu().numpy()
    return float(np.mean(pred == gt))


def embed_msg(net, cover01, msg_t):
    """消息嵌入: 官方 encode() 方法 (bm.encoder, secret_len=100). cover01: [0,1]."""
    return net.module.encode(net.module.bm, cover01, msg_t)


def extract_msg(net, att, device):
    """消息提取: 直接用 bm.decoder 对带消息的图像解码 (TrustMark Q 路径, 不走 rev)."""
    att_t = torch.from_numpy(att.transpose(2, 0, 1)[None]).float().to(device) / 255.0
    bits = net.module.bm.decoder(att_t)
    return bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--msglen', type=int, default=100,
                    help='OmniGuard 消息长度 (encoder secret_len=100)')
    ap.add_argument('--imgsize', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--inpaint', action='store_true',
                    help='用 SD inpainting 攻击替代局部移除 (全局 latent 扰动, 对照我们的方法上限)')
    ap.add_argument('--steps', type=int, default=30, help='SD 推理步数 (--inpaint 时)')
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    t0 = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    paths = collect_source('coco', args.n)
    print(f'数据集 coco: {len(paths)} 张, OmniGuard 消息 {args.msglen}-bit, 分辨率 {args.imgsize}')

    net, common = load_model(device)
    dwt, iwt = common.DWT(), common.IWT()

    # secret = 固定信任标记图 (与官方一致)
    R = {'meta': {'n': args.n, 'msglen': args.msglen, 'imgsize': args.imgsize,
                  'seed': args.seed, 'model': 'OmniGuard checkpoint-175 + model_checkpoint_01500',
                  'ts': datetime.datetime.now().isoformat()},
         'per_image': [], 'summary': {}}
    atts = {'clean': None, 'rem30': 0.3, 'rem50': 0.5}
    pipe = None
    if args.inpaint:
        from run_inpaint_attack import inpaint_attack, load_pipe
        pipe = load_pipe()
        atts = {'clean': None, 'inpaint': 'sd'}
        print('[OmniGuard] 攻击模式: SD inpainting (steps=%d)' % args.steps)

    prog = Progress(len(paths), 'omniguard')
    for idx, p in enumerate(paths):
        img = load(p, args.imgsize)          # uint8 RGB (256,256,3)
        h, w = img.shape[:2]
        msg = rng.randint(0, 2, (1, args.msglen)).astype(np.float32)
        msg_t = torch.from_numpy(msg).to(device)

        # cover 张量 [0,1]
        cover = torch.from_numpy(img.transpose(2, 0, 1)[None]).float().to(device) / 255.0

        row = {'img': os.path.basename(p)}
        with torch.no_grad():
            steg = embed_msg(net, cover, msg_t)   # (1,3,256,256) [0,1] 附近
            steg_np = np.clip(steg.permute(0, 2, 3, 1).cpu().numpy().squeeze(), 0, 1) * 255
            steg_u8 = steg_np.astype(np.uint8)
            row['clean_psnr'] = psnr(img, steg_u8)

            for atk_name, frac in atts.items():
                if frac is None:
                    att = steg_u8
                elif frac == 'sd':
                    m_total = detect_removal_mask(img, 0.5)
                    if m_total is None:
                        row[atk_name] = None
                        continue
                    att = inpaint_attack(pipe, steg_u8, m_total, args.steps, seed=idx)
                else:
                    m_total = detect_removal_mask(img, frac)
                    if m_total is None:
                        row[atk_name] = None
                        continue
                    att = apply_removal(steg_u8, m_total, bg_img=img)
                bits = extract_msg(net, att, device)
                row[atk_name] = {'ba': bits_acc(msg_t, bits)}
        R['per_image'].append(row)
        prog.update()
    prog.done()

    # 汇总
    mode = 'SD inpainting' if args.inpaint else '局部实例移除'
    print('\n' + '=' * 60)
    print(f'  OmniGuard 基线: {mode} 下的版权水印 BA')
    print('=' * 60)
    for atk_name in atts:
        bas = [r[atk_name]['ba'] for r in R['per_image'] if r.get(atk_name)]
        if bas:
            R['summary'][atk_name] = {'ba': float(np.mean(bas)),
                                      'ba_std': float(np.std(bas)), 'n': len(bas)}
            print(f'  {atk_name:<8} BA={np.mean(bas)*100:6.1f}% ± {np.std(bas)*100:4.1f}  (n={len(bas)})')
    R['summary']['clean_psnr'] = float(np.mean([r['clean_psnr'] for r in R['per_image']]))
    R['total_time_s'] = round(time.time() - t0, 1)

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(RESULTS_DIR, f'baseline_omniguard_{ts}.json')
    with open(out, 'w') as f:
        json.dump(R, f, indent=2, default=float)
    print(f'\n结果已保存: {out}  (总耗时 {R["total_time_s"]:.0f}s)')
    print('对比: 与 run_idea_probe 的 uniform/semantic_ra 相比 (注意 OmniGuard 为 64-bit 训练式协议)')


if __name__ == '__main__':
    main()
