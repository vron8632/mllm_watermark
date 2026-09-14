#!/usr/bin/env python3
"""为 Figure 4 (可解释诊断) 生成**真实**的 MLLM 诊断文本并缓存.

之前 Figure 4 的 caption 说"识别出 central bus 被删除", 但图里用的是其它样本,
图文不符. 这里对指定样本真实调用 MLLM, 把返回的 JSON 存到
results/diagnosis_demo.json, 供 make_figures.fig3_diagnosis() 绘制.

用法: cd code && python run_diagnosis_demo.py [--cid 0030]
输出: ../results/diagnosis_demo.json
"""
import sys, os, json, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image

for k in ['ALL_PROXY', 'all_proxy']:
    os.environ.pop(k, None)
os.environ.setdefault('http_proxy', 'http://127.0.0.1:7897')
os.environ.setdefault('https_proxy', 'http://127.0.0.1:7897')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = f'{BASE}/results/attackset_hires_sample'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cid', default='0030_42')
    ap.add_argument('--out', default=f'{BASE}/results/diagnosis_demo.json')
    args = ap.parse_args()

    from run_signal_ra import mllm_diagnose
    from run_mllm_benchmark import load_env
    _, _, mimo_model, _, _, _ = load_env()
    mimo_key = os.environ.get('MIMO_API_KEY')
    mimo_url = os.environ.get('MIMO_BASE_URL', '')

    meta = json.load(open(f'{D}/meta.json'))
    s = next(x for x in meta['samples'] if x['id'] == args.cid)
    o = np.array(Image.open(f'{D}/images/{args.cid}_orig.png').convert('RGB'))
    a = np.array(Image.open(f'{D}/images/{args.cid}_att.png').convert('RGB'))

    raw = mllm_diagnose(mimo_url, mimo_key, mimo_model, o, a, s['bbox'])
    print('raw diagnosis:', raw)
    obj = raw if isinstance(raw, dict) else {'diagnosis': str(raw)}

    out = {'cid': args.cid, 'source': s['source'], 'bbox': s['bbox'],
           'instruction': s['instruction'],
           'w': s['w'], 'h': s['h'],
           'operation': obj.get('operation'),
           'diagnosis': obj.get('diagnosis', ''),
           'confidence': obj.get('confidence'),
           'evidence': obj.get('evidence'),
           'raw': obj, 'mllm': mimo_model,
           'ts': datetime.datetime.now().isoformat()}
    json.dump(out, open(args.out, 'w'), ensure_ascii=False, indent=2)
    print('saved', args.out)


if __name__ == '__main__':
    main()
