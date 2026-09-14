#!/usr/bin/env python3
"""物化 MLLM 语义编辑攻击测试集 (论文发布基准, C 事项).

对 COCO 前 N 张: mimo 生成编辑指令+bbox → SD inpaint 生成攻击后图 (对原图, 与嵌入解耦).
保存: images/orig_XXXX.png, images/att_XXXX.png, images/mask_XXXX.png, meta.json

用法:
  cd code && python run_build_attackset.py --n 50
输出: ../results/attackset_<时间戳>/  (orig/att/mask/meta.json)
"""
import sys, os, time, json, io, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from run_idea_probe import load, collect_source
from run_mllm_benchmark import load_env, mllm_edit_instruct, sd_execute

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = f'{BASE}/results'


class Progress:
    def __init__(self, total, desc=''):
        self.total, self.desc, self.n = total, desc, 0
        self.t0 = time.time()
        self.t_last = 0.0

    @staticmethod
    def _fmt(s):
        s = int(s)
        return f'{s//3600:02d}:{s%3600//60:02d}:{s%60:02d}'

    def update(self, k=1):
        self.n += k
        if time.time() - self.t_last < 3.0 and self.n < self.total:
            return
        self.t_last = time.time()
        el = self.t_last - self.t0
        rem = el / self.n * (self.total - self.n) if self.n > 0 else 0
        eta = datetime.datetime.now() + datetime.timedelta(seconds=rem)
        print(f'  [{datetime.datetime.now().strftime("%H:%M:%S")}] {self.desc} '
              f'{self.n}/{self.total}  已用 {self._fmt(el)}  剩余约 {self._fmt(rem)}', flush=True)

    def done(self):
        self.n = self.total
        self.update()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--imgsize', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    load_env()
    t0 = time.time()
    paths = collect_source('coco', args.n)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(RESULTS_DIR, f'attackset_{ts}')
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    print(f'攻击集物化: n={len(paths)}, steps={args.steps}, 输出 {out_dir}')

    meta = {'n': len(paths), 'imgsize': args.imgsize, 'steps': args.steps,
            'seed': args.seed, 'mimo_model': os.environ.get('MIMO_MODEL', ''),
            'generated': ts, 'samples': []}
    prog = Progress(len(paths), 'attackset')
    for idx, p in enumerate(paths):
        try:
            img = load(p, args.imgsize)
            edit = mllm_edit_instruct(os.environ.get('MIMO_BASE_URL', ''),
                                      os.environ.get('MIMO_API_KEY', ''),
                                      os.environ.get('MIMO_MODEL', 'mimo-v2.5'), img)
            att = sd_execute(edit, img, steps=args.steps)
            # 掩膜 (bbox → 二值图)
            h, w = img.shape[:2]
            x1, y1, x2, y2 = edit['bbox']
            m = np.zeros((h, w), dtype=np.uint8)
            m[int(y1*h):int(y2*h), int(x1*w):int(x2*w)] = 255
            base = f'{idx:04d}'
            Image.fromarray(img).save(os.path.join(out_dir, 'images', f'orig_{base}.png'))
            Image.fromarray(att).save(os.path.join(out_dir, 'images', f'att_{base}.png'))
            Image.fromarray(m).save(os.path.join(out_dir, 'images', f'mask_{base}.png'))
            meta['samples'].append({'id': base, 'source': os.path.basename(p),
                                    'instruction': edit['instruction'],
                                    'bbox': [round(v, 3) for v in edit['bbox']]})
        except Exception as e:
            print(f'  ! {os.path.basename(p)} 失败: {repr(e)[:80]}')
        prog.update()
    prog.done()
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f'\n攻击集已保存: {out_dir}  (n={len(meta["samples"])}, 耗时 {time.time()-t0:.0f}s)')
    print('发布时: 上传 images/ + meta.json; 测试协议: 嵌入水印 → 用 att 的 bbox 区域替换\n'
          '水印图对应区域 (bbox 外保持水印图) → 提取. 与论文实验协议一致.')


if __name__ == '__main__':
    main()
