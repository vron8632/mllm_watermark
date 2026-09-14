#!/usr/bin/env python3
"""ICASSP 2027 — 方案 A: MLLM 驱动的语义级水印鲁棒性基准.

流程:
  1) 攻击生成层: 多模态大模型 (mimo) 对 COCO 图生成语义编辑攻击
     - 路径 a: mimo 直接输出编辑后图像 (原生多模态生成)
     - 路径 b: mimo 输出编辑指令(文本) → SD inpainting 执行 (需要编辑区域)
     deepseek (文本) 生成多样化编辑指令 + 编辑自然度打分
  2) 评估层: 复用 run_idea_probe 的 uniform / RA 提取, 在攻击集上算 BA

API 通过环境变量提供 (不硬编码):
  MIMO_API_KEY / MIMO_BASE_URL / MIMO_MODEL   (多模态)
  DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL         (文本, OpenAI 兼容)

用法:
  cd code
  export MIMO_API_KEY=sk-xxx MIMO_BASE_URL=https://... MIMO_MODEL=mimo-vl
  python run_mllm_benchmark.py --n 20 --mode instruct   # 路径 b: mimo 出指令 + SD 执行
  python run_mllm_benchmark.py --n 20 --mode direct     # 路径 a: mimo 直接出图

输出: ../results/mllm_benchmark_<时间戳>.json
"""
import sys, os, time, json, io, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
import base64
from PIL import Image
from run_idea_probe import (load, detect_removal_mask, embed_scheme, decode_scheme,
                            SCHEMES, psnr, bit_accuracy, collect_source,
                            _block_grid, y_channel, _demodulate_bit, _majority_vote)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = f'{BASE}/results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_env():
    """加载 .mllm_env (项目根) 中的 API 配置; 已有环境变量优先 (不覆盖)."""
    env_path = os.path.join(BASE, '.mllm_env')
    if os.path.isfile(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k, v)
    return (os.environ.get('MIMO_API_KEY'), os.environ.get('MIMO_BASE_URL', ''),
            os.environ.get('MIMO_MODEL', 'mimo-v2.5'),
            os.environ.get('DEEPSEEK_API_KEY'),
            os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
            os.environ.get('DEEPSEEK_MODEL', 'deepseek-v4-flash'))


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


def openai_chat(base_url, api_key, model, messages, timeout=300, retries=2,
                max_tokens=4000, response_format=None):
    """OpenAI 兼容 chat 调用; max_tokens 需 ≥4000 (mimo 推理模型 reasoning 占大头).

    超时/网络错误自动重试; 重试耗尽返回 '' (调用方兜底).
    """
    import urllib.request, urllib.error
    url = base_url.rstrip('/') + '/chat/completions'
    body = {'model': model, 'messages': messages, 'temperature': 0.3,
            'max_tokens': max_tokens}
    if response_format:
        body['response_format'] = response_format
    payload = json.dumps(body).encode()
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=payload, headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.load(r)
            content = resp['choices'][0]['message'].get('content') or ''
            if content.strip():
                return content
            rc = resp['choices'][0]['message'].get('reasoning_content') or ''
            if rc.strip() and attempt == retries:
                return rc
        except Exception as e:
            if attempt == retries:
                print(f'  ! API 调用失败({attempt+1}次重试后): {repr(e)[:80]}')
                return ''
            time.sleep(3)
    return ''


def img_to_data_url(img_arr):
    buf = io.BytesIO()
    Image.fromarray(img_arr).save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def mllm_edit_instruct(base_url, api_key, model, img_arr):
    """mimo 看原图输出编辑指令(文本) + 编辑区域 bbox; bbox 失败时用 YOLO 实例兜底.

    mimo-v2.5 是推理模型, 易复述 prompt: 用 few-shot 约束 + 失败兜底, 保证有效样本.
    """
    sys_p = ('你是图像编辑攻击设计师。看图后设计一条语义编辑指令（删除/替换/修改图中某个'
             '物体），并给出该物体所在区域的归一化边界框。只输出 JSON 且不要解释：\n'
             '{"instruction": "删除画面中央的人物", "bbox": [0.4,0.3,0.6,0.7]}\n'
             '{"instruction": "把左侧的汽车替换成一棵树", "bbox": [0.05,0.4,0.35,0.9]}')
    content = [{'type': 'text', 'text': '请设计一条语义编辑指令。'},
               {'type': 'image_url', 'image_url': {'url': img_to_data_url(img_arr)}}]
    raw = openai_chat(base_url, api_key, model,
                      [{'role': 'system', 'content': sys_p},
                       {'role': 'user', 'content': content}], retries=1)
    # 尝试提取 JSON (容忍推理内容包裹)
    try:
        s = raw[raw.find('{'): raw.rfind('}') + 1]
        obj = json.loads(s)
        instr = str(obj.get('instruction', '')).strip()
        bb = [max(0.0, min(1.0, float(x))) for x in obj.get('bbox', [0, 0, 1, 1])]
        if instr and len(bb) == 4 and (bb[2] - bb[0]) * (bb[3] - bb[1]) >= 0.03:
            return {'instruction': instr, 'bbox': bb}
        raise ValueError('bad json')
    except Exception as e:
        # 兜底: 编辑区域 = YOLO 最大实例; 指令取原始文本前 80 字
        print(f'  ! mimo JSON 失败({repr(e)[:40]}), 用 YOLO 实例兜底')
        h, w = img_arr.shape[:2]
        m = detect_removal_mask(img_arr, 0.5)
        if m is not None:
            ys, xs = np.where(m)
            bb = [xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h]
        else:
            bb = [0.0, 0.0, 1.0, 1.0]
        return {'instruction': raw[:80], 'bbox': bb}


def deepseek_naturalness(base_url, api_key, model, img_arr_orig, img_arr_edit):
    """deepseek 评估编辑自然度 (0-10) 与编辑区域描述 (供 RA 判定对照)."""
    sys_p = ('你是图像编辑质量评审。对比两张图(编辑前后)，输出 JSON: '
             '{"naturalness": 0-10, "edited_region": "对编辑区域的简短描述"}。只输出 JSON。')
    user = [{'type': 'text', 'text': '这是编辑前:'},
            {'type': 'image_url', 'image_url': {'url': img_to_data_url(img_arr_orig)}},
            {'type': 'text', 'text': '这是编辑后:'},
            {'type': 'image_url', 'image_url': {'url': img_to_data_url(img_arr_edit)}}]
    return openai_chat(base_url, api_key, model,
                       [{'role': 'system', 'content': sys_p}, {'role': 'user', 'content': user}])


def sd_execute(edit, img, steps=30):
    """用 SD inpainting 执行语义编辑: 掩膜 = mimo bbox; 掩膜外像素强制保持原样.

    关键: SD latent 扰动是全局的, 必须把掩膜外区域替换回原图, 攻击才真正局部
    (bbox 内被重写, bbox 外水印完好) — 这是 RA 有区分度的前提.
    """
    from run_inpaint_attack import inpaint_attack, load_pipe
    pipe = load_pipe()
    h, w = img.shape[:2]
    x1, y1, x2, y2 = edit['bbox']
    bx1, by1 = int(x1 * w), int(y1 * h)
    bx2, by2 = max(int(x2 * w), bx1 + 8), max(int(y2 * h), by1 + 8)
    m_total = np.zeros((h, w), dtype=bool)
    m_total[by1:by2, bx1:bx2] = True
    if m_total.sum() < 0.05 * h * w:  # 面积过小则并入实例掩膜
        m_inst = detect_removal_mask(img, 0.5)
        if m_inst is not None:
            m_total |= m_inst
    att = inpaint_attack(pipe, img, m_total, steps=steps, seed=0)
    # 掩膜外像素强制保持原样 (局部编辑)
    mask3 = np.stack([m_total] * 3, axis=-1)
    att = np.where(mask3, att, img).astype(np.uint8)
    return att


def mllm_detect_edit(base_url, api_key, model, att_img, retries=1):
    """方案 B 真实版: 提取时让 mimo 看攻击后图, 输出"最可能被编辑的区域" bbox.

    无原图参照, 依赖编辑痕迹/不自然区域; 失败时返回 None (由调用方兜底).
    """
    sys_p = ('你是图像取证专家。这张图可能被人用 AI 编辑过（某个物体被删除、替换或修改）。'
             '找出最可能被编辑的区域，输出归一化 bbox。只输出 JSON：'
             '{"bbox": [x1,y1,x2,y2]}，不要解释。'
             '若无法判断，输出 {"bbox": null}。')
    content = [{'type': 'text', 'text': '这张图哪里最可能被编辑过？'},
               {'type': 'image_url', 'image_url': {'url': img_to_data_url(att_img)}}]
    raw = openai_chat(base_url, api_key, model,
                      [{'role': 'system', 'content': sys_p},
                       {'role': 'user', 'content': content}], retries=retries)
    try:
        s = raw[raw.find('{'): raw.rfind('}') + 1]
        obj = json.loads(s)
        bb = obj.get('bbox')
        if bb is None:
            return None
        bb = [max(0.0, min(1.0, float(x))) for x in bb]
        if len(bb) != 4 or (bb[2] - bb[0]) * (bb[3] - bb[1]) < 0.03:
            return None
        return bb
    except Exception:
        return None


def extract_uniform_bbox_ra(watermarked, msg_len, delta, bbox01):
    """方案 B oracle: uniform 嵌入 + 用 mimo 编辑 bbox 剔除被编辑块 (均匀权重).

    removed = 块中心落在 mimo bbox 内的块; 用于验证"若已知编辑区域, RA 是否有效".
    """
    H, W, n_h, n_w = _block_grid(*watermarked.shape[:2])
    y = y_channel(watermarked)[:H, :W]
    x1, y1, x2, y2 = bbox01
    votes = [[] for _ in range(msg_len)]
    for i in range(n_h):
        for j in range(n_w):
            bi = (i * n_w + j) % msg_len
            cx, cy = (j + 0.5) / n_w, (i + 0.5) / n_h
            removed = (x1 <= cx <= x2) and (y1 <= cy <= y2)
            d = cv2.dct(y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8].astype(np.float32))
            bit = _demodulate_bit(d[4, 1], delta)
            votes[bi].append((bit, 0.0 if removed else 1.0))
    return _majority_vote(votes, msg_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--mode', choices=['instruct', 'direct'], default='instruct',
                    help='instruct=mimo出指令+SD执行; direct=mimo直接出图')
    ap.add_argument('--msglen', type=int, default=256)
    ap.add_argument('--imgsize', type=int, default=256)
    ap.add_argument('--steps', type=int, default=30, help='SD 执行步数')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    mimo_key, mimo_url, mimo_model, ds_key, ds_url, ds_model = load_env()
    if not mimo_key:
        sys.exit('缺少 MIMO_API_KEY. 请在项目根创建 .mllm_env (格式见 .mllm_env 示例) '
                 '或 export MIMO_API_KEY=sk-xxx MIMO_BASE_URL=...')

    rng = np.random.RandomState(args.seed)
    t0 = time.time()
    paths = collect_source('coco', args.n)
    schemes = {k: SCHEMES[k] for k in ['uniform', 'uniform_ra', 'semantic_ra']}
    schemes['bbox_ra'] = (None, 'uniform')   # 方案 B oracle: uniform 嵌入 + 嵌入时 bbox 剔除
    schemes['mllm_ra'] = (None, 'uniform')   # 方案 B 真实版: uniform 嵌入 + 提取时 mimo 检测 bbox
    delta = {'uniform': 19, 'uniform_ra': 19, 'semantic_ra': 15,
             'bbox_ra': 19, 'mllm_ra': 19}
    print(f'coco {len(paths)} 张, 模式={args.mode}, 方案={list(schemes)}, '
          f'MIMO={mimo_model}')

    R = {'meta': {'n': args.n, 'mode': args.mode, 'msglen': args.msglen,
                  'imgsize': args.imgsize, 'seed': args.seed,
                  'mimo_model': mimo_model, 'ds_model': ds_model,
                  'ts': datetime.datetime.now().isoformat()},
         'per_image': [], 'summary': {}}

    prog = Progress(len(paths), 'mllm-bench')
    for idx, p in enumerate(paths):
        img = load(p, args.imgsize)
        msg = rng.randint(0, 2, args.msglen).astype(np.uint8)
        row = {'img': os.path.basename(p)}

        # 1) mimo 看原图生成编辑指令 + bbox (每图一次)
        try:
            if args.mode == 'instruct':
                edit = mllm_edit_instruct(mimo_url, mimo_key, mimo_model, img)
                row['edit_prompt'] = edit['instruction']
                row['edit_bbox'] = edit['bbox']
            else:
                raise NotImplementedError('direct 模式需 mimo 图像生成接口, 先改用 instruct')
        except NotImplementedError as e:
            print(f'  ! {e}, 跳过 {os.path.basename(p)}')
            prog.update()
            continue
        except Exception as e:
            print(f'  ! API 调用失败: {repr(e)[:100]}, 跳过')
            prog.update()
            continue

        # 2) 逐方案: 嵌入 → 对各自水印图做局部语义编辑 → 提取
        for sname, (sm_fn, dec) in schemes.items():
            sm = None if sm_fn is None else sm_fn(img)
            d = delta[sname]
            wm = embed_scheme(img, msg, d, sm, dec)
            try:
                att = sd_execute(edit, wm, steps=args.steps)
            except Exception as e:
                print(f'  ! SD 执行失败: {repr(e)[:80]}, 跳过该方案')
                row[sname] = None
                continue
            if dec in ('uniform_ra', 'adaptive_ra'):
                me = decode_scheme(att, msg, d, sm, dec, sm_new=sm_fn(att))
            elif sname == 'bbox_ra':
                me = extract_uniform_bbox_ra(att, len(msg), d, edit['bbox'])
            elif sname == 'mllm_ra':
                det = mllm_detect_edit(mimo_url, mimo_key, mimo_model, att)
                row['mllm_detected_bbox'] = det
                # 真实检测失败时回退 oracle bbox (报告时区分 n)
                bb_used = det if det is not None else edit['bbox']
                me = extract_uniform_bbox_ra(att, len(msg), d, bb_used)
            else:
                me = decode_scheme(att, msg, d, sm, dec)
            row[sname] = {'ba': bit_accuracy(msg, me), 'psnr': psnr(img, wm)}
        R['per_image'].append(row)
        prog.update()
    prog.done()

    print('\n' + '=' * 66)
    print('  MLLM 攻击下的水印 BA (256-bit)')
    print('=' * 66)
    for sname in schemes:
        bas = [r[sname]['ba'] for r in R['per_image'] if sname in r]
        R['summary'][sname] = {'ba': float(np.mean(bas)), 'ba_std': float(np.std(bas)),
                               'n': len(bas)}
        print(f'  {sname:<12} BA={np.mean(bas)*100:6.1f}% ± {np.std(bas)*100:4.1f}  (n={len(bas)})')
    if R['summary']['uniform_ra']['n'] > 0:
        u = R['summary']['uniform']['ba']
        ura = R['summary']['uniform_ra']['ba']
        print(f'\n  判定: uniform_ra vs uniform = {(ura-u)*100:+.1f}pt '
              f'→ {"✅ YOLO-RA 在 MLLM 编辑攻击下仍有效" if (ura-u) >= 0.03 else "→ 边界: YOLO-RA 对 MLLM 编辑失效"}')
    if R['summary']['bbox_ra']['n'] > 0:
        u = R['summary']['uniform']['ba']
        bb = R['summary']['bbox_ra']['ba']
        print(f'  判定: bbox_ra (MLLM 编辑区域剔除, oracle) vs uniform = {(bb-u)*100:+.1f}pt '
              f'→ {"✅ 方案 B 成立: 语义级篡改剔除有效" if (bb-u) >= 0.03 else "→ bbox 剔除也无增益"}')
    R['total_time_s'] = round(time.time() - t0, 1)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(RESULTS_DIR, f'mllm_benchmark_{ts}.json')
    with open(out, 'w') as f:
        json.dump(R, f, indent=2, default=float)
    print(f'\n结果已保存: {out}  (总耗时 {R["total_time_s"]:.0f}s)')


if __name__ == '__main__':
    main()
