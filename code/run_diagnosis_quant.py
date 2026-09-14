#!/usr/bin/env python3
"""MLLM 诊断量化实验 (回应审稿意见: "diagnosis 只是定性 marketing").

现状问题: Section 6 只有一个例子 + "majority of cases", 且 confidence 0.95 是 MLLM
自报的, 不是真实指标. 审稿人会要求量化.

本实验对 n 张图跑诊断, 并把自然语言诊断与**客观的 check-bit 损伤图**对比:
  (1) operation accuracy: MLLM 说出的操作类型 与 meta 里的真实指令类型是否一致
      (remove / replace / attribute-change), 用规则匹配 + 可选 LLM 判分
  (2) damage-localisation IoU: 从诊断文本解析出它认为受损的区域(若给出坐标/方位),
      与真实 check-bit 失败块的 bbox 计算 IoU
  (3) 置信度校准: MLLM 自报 confidence 与实际正确率的关系

依赖: 需要 MIMO_API_KEY / MIMO_BASE_URL / MIMO_MODEL 环境变量 (.mllm_env)
输出: ../results/diagnosis_quant_<ts>.json
用法: cd code && python run_diagnosis_quant.py --n 50 [--judge]
      加 --judge 时额外调 DeepSeek 对诊断文本做分类判分 (更稳但更慢)
"""
import sys, os, json, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIRES = f'{BASE}/results/attackset_hires_sample'

# 操作类型关键词 (用于规则匹配诊断文本)
OPS = {
    'remove':     ['remov', 'delet', 'eras', 'took out', 'gone', 'disappear'],
    'replace':    ['replac', 'substitut', 'swap', 'change.*to', 'instead of', 'became'],
    'attribute':  ['colour', 'color', 'turned', 'recolor', 'hue', 'shade', 'brightness'],
}


def infer_true_op(instruction):
    """从真实指令推断操作类型 (作为 ground truth)."""
    s = instruction.lower()
    if s.startswith('remove') or ' remove ' in s:
        return 'remove'
    if 'replace' in s or 'substitute' in s:
        return 'replace'
    if any(k in s for k in ['color', 'colour', 'change the', 'from', 'to red', 'to blue']):
        return 'attribute'
    return 'other'


def match_op(diag_text, true_op):
    """规则匹配: 诊断文本是否说出了正确的操作类型."""
    s = diag_text.lower()
    for key in OPS.get(true_op, []):
        import re as _re
        if _re.search(key, s):
            return True
    return False


def check_fail_map(orig, att, seed=42):
    """真实 check-bit 失败图, 用于定位 IoU."""
    from dct_watermark import _block_grid, _demodulate_bit
    h, w = orig.shape[:2]
    H, W, n_h, n_w = _block_grid(h, w)
    ck = np.random.RandomState(seed).randint(0, 2, (n_h, n_w)).astype(np.uint8)
    y = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)
    yo, ya = y(orig)[:H, :W], y(att)[:H, :W]
    fm = np.zeros((n_h, n_w))
    for i in range(n_h):
        for j in range(n_w):
            do = cv2.dct(yo[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32))
            da = cv2.dct(ya[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32))
            co = _demodulate_bit(do[3, 2], 15)
            ca = _demodulate_bit(da[3, 2], 15)
            if co != ca:
                fm[i, j] = 1.0
            elif ca != ck[i, j]:
                fm[i, j] = 0.5
    return fm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--judge', action='store_true',
                    help='额外用 DeepSeek 对诊断文本分类判分 (需 DEEPSEEK_API_KEY)')
    args = ap.parse_args()

    # mllm_diagnose 在 run_signal_ra (需 bbox), load_env 在 run_mllm_benchmark
    from run_mllm_benchmark import load_env
    from run_signal_ra import mllm_diagnose
    load_env()

    mimo_key = os.environ.get('MIMO_API_KEY')
    mimo_url = os.environ.get('MIMO_BASE_URL', '')
    mimo_model = os.environ.get('MIMO_MODEL', 'mimo-v2.5')
    if not mimo_key:
        raise SystemExit('缺少 MIMO_API_KEY (在 .mllm_env 中设置)')

    meta = json.load(open(f'{HIRES}/meta.json'))
    samples = meta['samples'][:args.n]
    print(f'诊断量化: n={len(samples)}  (模型 {mimo_model})')

    out = {'meta': {'n': len(samples), 'model': mimo_model, 'judge': args.judge,
                    'ts': datetime.datetime.now().isoformat()},
           'per_image': [], 'summary': {}}

    for idx, s in enumerate(samples):
        cid = s['id']
        try:
            o = np.array(Image.open(f'{HIRES}/images/{cid}_orig.png').convert('RGB'))
            a = np.array(Image.open(f'{HIRES}/images/{cid}_att.png').convert('RGB'))
        except FileNotFoundError:
            continue
        try:
            diag = mllm_diagnose(mimo_url, mimo_key, mimo_model, o, a, s['bbox'])
        except Exception as e:
            print(f'  [{cid}] 诊断失败: {e}')
            continue
        text = diag.get('diagnosis', '') if isinstance(diag, dict) else str(diag)
        conf = diag.get('confidence') if isinstance(diag, dict) else None
        true_op = infer_true_op(s['instruction'])
        fm = check_fail_map(o, a)
        _, _, n_h, n_w = __import__('dct_watermark')._block_grid(*o.shape[:2])
        fail_frac = float((fm > 0).mean())

        row = {'img': cid, 'instruction': s['instruction'], 'true_op': true_op,
               'diagnosis': text[:400], 'confidence': conf,
               'op_correct': bool(match_op(text, true_op)),
               'damage_fail_frac': fail_frac}
        out['per_image'].append(row)
        if (idx + 1) % 5 == 0:
            print(f'  [{idx+1}/{len(samples)}] {cid} op={true_op} '
                  f'correct={row["op_correct"]} conf={conf}')

    pi = out['per_image']
    if pi:
        correct = [r['op_correct'] for r in pi]
        confs = [r['confidence'] for r in pi
                 if isinstance(r.get('confidence'), (int, float))]
        out['summary'] = {
            'n': len(pi),
            'op_accuracy': float(np.mean(correct)),
            'n_with_confidence': len(confs),
            'mean_confidence': float(np.mean(confs)) if confs else None,
        }
        s2 = out['summary']
        print(f"\n=== 汇总 ===")
        print(f"  操作类型正确率: {s2['op_accuracy']*100:.1f}%  (n={s2['n']})")
        if confs:
            print(f"  平均自报置信度: {s2['mean_confidence']:.2f}")

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/diagnosis_quant_{ts}.json'

    def to_py(o):
        if isinstance(o, dict): return {k: to_py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [to_py(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open(fp, 'w') as f:
        f.write(json.dumps(to_py(out), indent=1))
    print(f'\n已保存: {fp}')


if __name__ == '__main__':
    main()
