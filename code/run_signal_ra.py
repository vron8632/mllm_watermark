#!/usr/bin/env python3
"""信号校验检测最小验证 (方案 B 可部署版): 双系数 DCT-QIM 水印.

嵌入: 每块两个中频系数 —
  (4,1) 存 payload bit (循环分配, 与之前协议一致)
  (3,2) 存校验 bit   (按块坐标的伪随机序列, seed=42, 嵌入/提取共享 = key)
提取: 校验 bit 与期望不符的块 = 被篡改 → payload 投票权重 0 (signal_ra).
对比: uniform (无剔除) / signal_ra (校验剔除) / bbox_ra (oracle 上界).
附加: clean 与 JPEG q75 下 signal_ra 的误报率 (校验剔除不应误伤未篡改块).

用法:
  cd code && python run_signal_ra.py --n 4 --steps 10   # 快速验证
  python run_signal_ra.py --n 20                        # 最小验证
输出: ../results/signal_ra_<时间戳>.json
"""
import sys, os, time, json, io, base64, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image
from run_idea_probe import (load, collect_source, psnr, bit_accuracy, y_channel,
                            _block_grid, _demodulate_bit)
from run_mllm_benchmark import (load_env, mllm_edit_instruct, sd_execute)
from dct_watermark import _majority_vote

CHECK_SEED = 42
CHECK_COEF = (3, 2)     # 校验系数
PAY_COEF = (4, 1)       # payload 系数


def check_bits(h, w, n_h, n_w, seed=CHECK_SEED):
    """逐块校验期望序列 (嵌入/提取共享)."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, (n_h, n_w)).astype(np.uint8)


def embed_signal(img, msg, delta=15):
    """双系数 QIM 嵌入: payload + 校验."""
    H, W, n_h, n_w = _block_grid(*img.shape[:2])
    y = y_channel(img)[:H, :W]
    ck = check_bits(H, W, n_h, n_w)
    for i in range(n_h):
        for j in range(n_w):
            d = cv2.dct(y[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32))
            bi = (i * n_w + j) % len(msg)
            # payload
            c = d[PAY_COEF]
            d[PAY_COEF] = (np.floor(c/delta)*delta + delta/2) if msg[bi] == 1 \
                          else np.round(c/delta)*delta
            # 校验
            c = d[CHECK_COEF]
            d[CHECK_COEF] = (np.floor(c/delta)*delta + delta/2) if ck[i, j] == 1 \
                            else np.round(c/delta)*delta
            y[i*8:(i+1)*8, j*8:(j+1)*8] = cv2.idct(d)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    out = ycrcb.copy()
    out[:H, :W, 0] = np.clip(y, 0, 255)
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB).astype(np.uint8)


def _check_soft_confidence(c, delta, expect):
    """校验位的软置信度: 0-1, 越接近期望网格点越接近 1 (连续权重, 减少误剔损失)."""
    q0 = np.round(c / delta) * delta          # bit=0 网格
    q1 = np.floor(c / delta) * delta + delta / 2   # bit=1 网格
    d0 = abs(c - q0)
    d1 = abs(c - q1)
    d_best = d0 if expect == 0 else d1
    d_other = d1 if expect == 0 else d0
    return float(np.clip(1.0 - d_best / (d_other + d_best + 1e-8), 0.0, 1.0))


def decode_signal(watermarked, msg_len, delta, use_check=True, soft=False,
                  dilate=0):
    """提取: 校验剔除 (use_check) / 软置信度加权 (soft) / 空间膨胀邻接降权 (dilate).

    v2 改进 (A): soft=True 时权重 = 校验位软置信度 (而非 0/1), 减少"校验被破坏但
    payload 完好"块的误剔损失; dilate>0 时把校验失败区域膨胀 dilate 圈,
    邻接块权重乘 0.3 (减少"payload 已被破坏但校验完好"块的漏检).
    """
    H, W, n_h, n_w = _block_grid(*watermarked.shape[:2])
    y = y_channel(watermarked)[:H, :W]
    ck = check_bits(H, W, n_h, n_w)
    votes = [[] for _ in range(msg_len)]
    n_fail = 0
    # 先算校验失败掩膜 (用于膨胀)
    fail_mask = np.zeros((n_h, n_w), dtype=bool)
    if use_check and dilate > 0:
        for i in range(n_h):
            for j in range(n_w):
                d = cv2.dct(y[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32))
                if _demodulate_bit(d[CHECK_COEF], delta) != ck[i, j]:
                    fail_mask[i, j] = True
        fail_mask = cv2.dilate(fail_mask.astype(np.uint8),
                               np.ones((2*dilate+1, 2*dilate+1), np.uint8)).astype(bool)
    for i in range(n_h):
        for j in range(n_w):
            d = cv2.dct(y[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32))
            bi = (i * n_w + j) % msg_len
            pay = _demodulate_bit(d[PAY_COEF], delta)
            if not use_check:
                votes[bi].append((pay, 1.0))
                continue
            ck_bit = _demodulate_bit(d[CHECK_COEF], delta)
            if soft:
                w = _check_soft_confidence(d[CHECK_COEF], delta, ck[i, j])
            else:
                w = 0.0 if ck_bit != ck[i, j] else 1.0
            if ck_bit != ck[i, j]:
                n_fail += 1
            if w > 0 and dilate > 0 and fail_mask[i, j] and ck_bit == ck[i, j]:
                w *= 0.3   # 校验通过但邻接编辑区: 降权
            votes[bi].append((pay, w))
    return _majority_vote(votes, msg_len), n_fail


def mllm_diagnose(base_url, api_key, model, img_orig, img_att, bbox01):
    """P3: 让 mimo 对比"原图 vs 攻击后图"的编辑区域, 输出一句可解释诊断."""
    from run_mllm_benchmark import openai_chat, img_to_data_url
    h, w = img_orig.shape[:2]
    x1, y1, x2, y2 = bbox01
    bx1, by1 = int(x1 * w), int(y1 * h)
    bx2, by2 = max(int(x2 * w), bx1 + 8), max(int(y2 * h), by1 + 8)
    # 裁剪编辑区域并排拼接
    c_orig = img_orig[by1:by2, bx1:bx2]
    c_att = img_att[by1:by2, bx1:bx2]
    side = max(c_orig.shape[0], c_orig.shape[1], 8)
    def pad_crop(c):
        out = np.zeros((side, side, 3), dtype=np.uint8)
        out[:c.shape[0], :c.shape[1]] = c
        return Image.fromarray(out).resize((128, 128), Image.BILINEAR)
    pair = np.hstack([np.array(pad_crop(c_orig)), np.array(pad_crop(c_att))])
    buf = io.BytesIO()
    Image.fromarray(pair).save(buf, format='PNG')
    data_url = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    sys_p = ('你是图像取证专家。左图是编辑前区域，右图是编辑后区域。'
             '诊断该区域发生了什么编辑。严格输出 JSON，字段为：'
             '{"operation": "删除|替换|属性修改|无变化", '
             '"diagnosis": "一句话英文描述对该区域的判断", '
             '"confidence": 0.0-1.0 的数字, '
             '"evidence": "英文，判断依据（如残留阴影/边缘/纹理不连续）"}。只输出 JSON。')
    raw = openai_chat(base_url, api_key, model,
                      [{'role': 'system', 'content': sys_p},
                       {'role': 'user', 'content': [
                           {'type': 'text', 'text': '对比左右两张图，诊断编辑方式。'},
                           {'type': 'image_url',
                            'image_url': {'url': data_url}}]}],
                      response_format={'type': 'json_object'})
    try:
        return json.loads(raw)
    except Exception:
        return {'diagnosis': raw[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--msglen', type=int, default=256)
    ap.add_argument('--delta', type=int, default=15)
    ap.add_argument('--datasets', default='coco',
                    help='逗号分隔: coco,kodak (默认 coco)')
    ap.add_argument('--diagnose', action='store_true',
                    help='P3: 让 mimo 对编辑区域输出可解释诊断文本 (每图多一次 API 调用)')
    args = ap.parse_args()
    load_env()
    t0 = time.time()
    paths = []
    for name in args.datasets.split(','):
        paths += collect_source(name.strip(), args.n)
    rng = np.random.RandomState(42)
    R = {'meta': {'n': args.n, 'datasets': args.datasets, 'msglen': args.msglen,
                  'delta': args.delta, 'steps': args.steps,
                  'check_coef': CHECK_COEF, 'diagnose': args.diagnose,
                  'ts': datetime.datetime.now().isoformat()},
         'per_image': [], 'summary': {}}
    ts0 = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    R['tmp_out'] = f'../results/signal_ra_{ts0}.tmp.json'   # 增量保存 (崩溃可恢复)
    print(f'信号校验验证: n={len(paths)} ({args.datasets}), Δ={args.delta}, '
          f'payload=({PAY_COEF}) 校验=({CHECK_COEF})' + (', 含诊断' if args.diagnose else ''))
    for idx, p in enumerate(paths):
        img = load(p, 256)
        msg = rng.randint(0, 2, args.msglen).astype(np.uint8)
        row = {'img': os.path.basename(p)}
        try:
            # 1) 嵌入 (双系数)
            wm = embed_signal(img, msg, args.delta)
            row['psnr'] = psnr(img, wm)
            # 2) clean: 校验误报率 + BA
            me, n_fail = decode_signal(wm, args.msglen, args.delta, use_check=True)
            row['clean_ba'] = bit_accuracy(msg, me)
            row['clean_fail_frac'] = n_fail / ((img.shape[0]//8) * (img.shape[1]//8))
            # 3) MLLM 编辑攻击
            mimo_key = os.environ.get('MIMO_API_KEY')
            mimo_url = os.environ.get('MIMO_BASE_URL', '')
            mimo_model = os.environ.get('MIMO_MODEL', 'mimo-v2.5')
            edit = mllm_edit_instruct(mimo_url, mimo_key, mimo_model, img)
            row['edit_prompt'] = edit['instruction']
            att = sd_execute(edit, wm, steps=args.steps)
            # 4) 攻击后提取: uniform vs v1(二值剔除) vs v2(软权重) vs v2+膨胀 vs oracle
            me, _ = decode_signal(att, args.msglen, args.delta, use_check=False)
            row['uniform_ba'] = bit_accuracy(msg, me)
            me, n_fail = decode_signal(att, args.msglen, args.delta, use_check=True)
            row['signal_ra_ba'] = bit_accuracy(msg, me)
            me, _ = decode_signal(att, args.msglen, args.delta, use_check=True, soft=True)
            row['signal_soft_ba'] = bit_accuracy(msg, me)
            me, _ = decode_signal(att, args.msglen, args.delta, use_check=True, soft=True,
                                  dilate=1)
            row['signal_soft_dilate_ba'] = bit_accuracy(msg, me)
            row['att_fail_frac'] = n_fail / ((img.shape[0]//8) * (img.shape[1]//8))
            from run_mllm_benchmark import extract_uniform_bbox_ra
            me = extract_uniform_bbox_ra(att, args.msglen, args.delta, edit['bbox'])
            row['bbox_ra_ba'] = bit_accuracy(msg, me)
            # P3: 可解释诊断 (可选)
            if args.diagnose:
                diag = mllm_diagnose(mimo_url, mimo_key, mimo_model, img, att,
                                     edit['bbox'])
                row['diagnosis'] = diag
        except Exception as e:
            row['error'] = repr(e)[:120]
            print(f'  ! 图 {os.path.basename(p)} 失败: {row["error"]}')
        R['per_image'].append(row)
        # 增量保存 (崩溃/超时后已跑结果不丢)
        with open(R['tmp_out'], 'w') as f:
            json.dump(R, f, indent=2, default=float)
        if 'error' not in row:
            print(f'  [{idx+1}/{len(paths)}] {os.path.basename(p)[:18]} '
                  f'uniform={row["uniform_ba"]*100:.0f}% signal_ra={row["signal_ra_ba"]*100:.0f}% '
                  f'oracle={row["bbox_ra_ba"]*100:.0f}% 校验失败率={row["att_fail_frac"]*100:.0f}%')
    # 汇总
    print('\n' + '=' * 60)
    for k, lbl in [('clean_ba', 'clean'), ('uniform_ba', 'MLLM攻击-uniform'),
                   ('signal_ra_ba', 'MLLM攻击-signal_ra(v1二值)'),
                   ('signal_soft_ba', 'MLLM攻击-signal_soft(v2软权重)'),
                   ('signal_soft_dilate_ba', 'MLLM攻击-signal_soft+dilate(v2)'),
                   ('bbox_ra_ba', 'MLLM攻击-oracle(上界)')]:
        vals = [r[k] for r in R['per_image']]
        R['summary'][k] = {'ba': float(np.mean(vals)), 'n': len(vals)}
        print(f'  {lbl:<32} BA={np.mean(vals)*100:6.1f}%')
    cf = np.mean([r['clean_fail_frac'] for r in R['per_image']])
    af = np.mean([r['att_fail_frac'] for r in R['per_image']])
    R['summary']['clean_fail_frac'] = float(cf)
    R['summary']['att_fail_frac'] = float(af)
    print(f'  校验失败率: clean={cf*100:.1f}% (误报)  攻击后={af*100:.1f}%')
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = f'../results/signal_ra_{ts}.json'
    with open(out, 'w') as f:
        json.dump(R, f, indent=2, default=float)
    print(f'\n保存: {out} (耗时 {time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
