#!/usr/bin/env python3
"""第二个 MLLM 攻击者实验 (回应审稿意见: "mimo-v2.5 是黑箱, 换个别 MLLM 还成立吗?").

目的: 证明 "损伤外溢出 bbox" 的现象与 "v2 防御有效" 的结论 **不依赖特定的 MLLM**.
      只换攻击生成用的 MLLM, 防御与评测协议完全不变, 因此是纯粹的"攻击者替换"对照.

== 需要准备的 API key (任选 1-2 个即可, 建议至少 1 个) ==

脚本用标准 OpenAI 兼容接口, 因此任何提供 chat/completions 的多模态模型都能接入。
在 .mllm_env 里设置以下变量之一 (前缀可自定义, 见 --provider):

  # 方案 A: OpenAI GPT-4o / GPT-4V  (最权威, 审稿人最认可)
  OPENAI_API_KEY=sk-...
  OPENAI_BASE_URL=https://api.openai.com/v1
  OPENAI_MODEL=gpt-4o

  # 方案 B: 通义千问 Qwen-VL (国内可直连, 免费额度)
  QWEN_API_KEY=sk-...
  QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
  QWEN_MODEL=qwen-vl-max

  # 方案 C: 智谱 GLM-4V
  GLM_API_KEY=...
  GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
  GLM_MODEL=glm-4v-plus

  # 方案 D: 月之暗面 Kimi (视觉版)
  MOONSHOT_API_KEY=...
  MOONSHOT_BASE_URL=https://api.moonshot.cn/v1
  MOONSHOT_MODEL=moonshot-v1-8k-vision-preview

  # 方案 E: 本地部署 LLaVA / Qwen-VL (无需 key, 用 vLLM 起 OpenAI 兼容服务)
  LOCAL_API_KEY=EMPTY
  LOCAL_BASE_URL=http://localhost:8000/v1
  LOCAL_MODEL=llava-hf/llava-1.5-7b-hf

  # 方案 F: DeepSeek 视觉实验模型 (已配置, 见 .mllm_env)
  #   实测: deepseek-v4-flash-vision-exp 会被服务端路由到 deepseek-flash,
  #   该回退模型确实具备图像输入能力 (已验证能正确描述图像内容);
  #   而 deepseek-v4-pro 明确拒绝图像输入。故本方案实际等价于 deepseek-flash。
  DSVISION_API_KEY=sk-...
  DSVISION_BASE_URL=https://api.deepseek.com
  DSVISION_MODEL=deepseek-v4-flash-vision-exp

用法:
  cd code && python run_second_mllm_attacker.py --provider dsvision --n 100
  cd code && python run_second_mllm_attacker.py --provider qwen     --n 100
  # 跑完两个后再比较: python run_second_mllm_attacker.py --compare

输出: ../results/second_mllm_<provider>_<ts>.json
"""
import sys, os, json, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image

from run_mllm_benchmark import openai_chat, img_to_data_url

# 强制 IPv4: 阿里云百炼 workspace 端点解析出 IPv6, 但本机 IPv6 不通,
# urllib 会依次尝试所有地址并在 IPv6 上抛 gaierror(-3), 导致 ~20% 样本被丢弃.
# 实测 IPv4 直连 205ms 正常, IPv6 完全不通(0.9ms 即失败).
import socket as _socket
_orig_getaddrinfo = _socket.getaddrinfo
def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, _socket.AF_INET, type, proto, flags)
_socket.getaddrinfo = _ipv4_only_getaddrinfo
from run_signal_ra import embed_signal, decode_signal, check_bits
from dct_watermark import bit_accuracy, _block_grid, _demodulate_bit

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COCO = f'{BASE}/data/COCO'
AS = f'{BASE}/results/attackset_20260817_082819'   # bbox/指令来源 (供参考)


def crop8(a):
    h, w = a.shape[:2]
    return a[:h - h % 8, :w - w % 8]


def ask_edit(base_url, key, model, img, strong=False):
    """让目标 MLLM 输出编辑指令 + bbox.

    与论文 prompt 保持一致; strong=True 时额外要求选择**显著较大**的物体,
    用于跨 MLLM 对齐攻击强度 (实测 DeepSeek 默认偏爱小物体, 面积仅 0.10,
    约为 mimo 的 0.25 的 41%, 导致攻击过弱、防御增益无法显现).
    """
    extra = ('\n重要：请选择画面中**较显著、面积较大**的主要物体（而非小物件或局部细节），'
             '使其边界框面积不低于整幅图的 15%。') if strong else ''
    sys_p = ('你是图像编辑攻击设计师。看图后设计一条语义编辑指令（删除/替换/修改图中某个'
             '物体），并给出该物体所在区域的归一化边界框。只输出 JSON 且不要解释：\n'
             '{"instruction": "删除画面中央的人物", "bbox": [0.4,0.3,0.6,0.7]}\n'
             '{"instruction": "把左侧的汽车替换成一棵树", "bbox": [0.05,0.4,0.35,0.9]}' + extra)
    content = [{'type': 'text', 'text': '请设计一条语义编辑指令。'},
               {'type': 'image_url', 'image_url': {'url': img_to_data_url(img)}}]
    # 外层重试: 服务端偶发空返回/截断 JSON, 单次失败率实测 ~10-35%.
    # 注意 openai_chat 自身的 retries 只管网络层; JSON 解析失败需在此重试,
    # 否则会把本可成功的样本直接丢弃 (实测手动重测均可成功).
    for attempt in range(4):
        raw = openai_chat(base_url, key, model,
                          [{'role': 'system', 'content': sys_p},
                           {'role': 'user', 'content': content}],
                          retries=2, timeout=300)
        if not raw:
            continue
        try:
            obj = json.loads(raw[raw.find('{'): raw.rfind('}') + 1])
            instr = str(obj.get('instruction', '')).strip()
            bbox = [float(v) for v in obj.get('bbox', [])]
            if len(bbox) != 4 or not instr:
                continue
            # 容错: 部分模型偶尔返回像素坐标而非归一化坐标
            # (实测 qwen3-vl-plus 偶发返回 [658,80,875,742]).
            # 仅当坐标同时满足"超出[0,1]"且"除以图像尺寸后合法"时才转换;
            # 否则视为无效框 (强行截断会产生面积≈0 的退化框).
            if max(bbox) > 1.5:
                H_, W_ = img.shape[:2]
                conv = [bbox[0] / W_, bbox[1] / H_, bbox[2] / W_, bbox[3] / H_]
                if max(conv) <= 1.0 and conv[2] > conv[0] and conv[3] > conv[1]:
                    bbox = conv
                else:
                    continue   # 坐标本身就异常, 交给外层重试
            # 过滤: 只拒绝真正过大的编辑区域, 用**面积**而非边长.
            # 原按边长 >0.6 拒绝会系统性排除高瘦物体(人/树/建筑):
            # 实测一张人物图连调 5 次高度均 >0.6 而被全数拒绝.
            # 论文现有攻击集 (n=200) 中高度 >0.6 的样本有 79/200, 面积均值 0.25,
            # 说明该边长规则并非论文所用; 改用面积阈值以保持可比性.
            # 与论文 mllm_edit_instruct 完全对齐:
            #   1) 坐标裁剪到 [0,1] (论文用 max(0,min(1,x)), 而非拒绝越界)
            #   2) 面积**下限** 0.03 (拒绝过小的框, 不设上限)
            # 早先我用"面积上限 0.5"方向相反, 系统性拒绝了论文赖以产生强攻击的
            # 大编辑样本 (论文攻击集有 14% 面积 >0.5, 失败率可达 49.9%),
            # 导致平均失败率虚低、uniform BA 虚高至 100%.
            bbox = [max(0.0, min(1.0, v)) for v in bbox]
            w_, h_ = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if w_ * h_ < 0.03:
                continue
            return {'instruction': instr, 'bbox': bbox}
        except Exception:
            continue
    # 兜底: 与论文一致 —— 用 YOLO 最大实例作编辑区域, 指令取原图默认文本.
    # 这样保证不因模型偶发输出异常而丢样本 (论文亦如此).
    try:
        from run_idea_probe import detect_removal_mask
        m = detect_removal_mask(img, 0.5)
        if m is not None:
            ys, xs = np.where(m)
            h, w = img.shape[:2]
            bb = [xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h]
            bb = [max(0.0, min(1.0, v)) for v in bb]
            if (bb[2] - bb[0]) * (bb[3] - bb[1]) >= 0.03:
                return {'instruction': 'remove the main object in the image',
                        'bbox': bb, 'fallback': True}
    except Exception:
        pass
    return None


def damage_leakage(orig, att):
    """核心指标: bbox 内/外的 check-bit 失败率 (论文的核心发现)."""
    h, w = orig.shape[:2]
    H, W, n_h, n_w = _block_grid(h, w)
    ck = check_bits(H, W, n_h, n_w)
    y = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)
    yo, ya = y(orig)[:H, :W], y(att)[:H, :W]
    fm = np.zeros((n_h, n_w))
    for i in range(n_h):
        for j in range(n_w):
            do = cv2.dct(yo[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32))
            da = cv2.dct(ya[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32))
            co, ca = _demodulate_bit(do[3, 2], 15), _demodulate_bit(da[3, 2], 15)
            if co != ca:
                fm[i, j] = 1.0
            elif ca != ck[i, j]:
                fm[i, j] = 0.5
    return fm, n_h, n_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--provider', default='dsvision',
                    help='dsvision / openai / qwen / glm / moonshot / local')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--msglen', type=int, default=256)
    ap.add_argument('--delta', type=int, default=15)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--strong', action='store_true',
                    help='要求 MLLM 选择显著较大物体(面积>=15%%), 用于跨模型对齐攻击强度')
    ap.add_argument('--compare', action='store_true',
                    help='不跑实验, 只汇总已有的 second_mllm_*.json 做对比')
    args = ap.parse_args()

    if args.compare:
        import glob
        print('=== 多 MLLM 攻击者对比 ===')
        print(f"{'provider':<12}{'n':>5}{'内MAE':>9}{'外MAE':>9}{'uniform':>9}{'v2':>8}{'gain':>8}")
        for f in sorted(glob.glob(f'{BASE}/results/second_mllm_*.json')):
            d = json.load(open(f))
            s = d.get('summary', {})
            if not s: continue
            print(f"{d['meta']['provider']:<12}{s.get('n',0):>5}"
                  f"{s.get('inside_mae',0):>9.1f}{s.get('outside_mae',0):>9.1f}"
                  f"{s.get('uni_ba',0)*100:>8.1f}%{s.get('v2_ba',0)*100:>7.1f}%"
                  f"{s.get('gain_pt',0):>+7.1f}pt")
        return

    p = args.provider.upper()
    # 先从 .mllm_env 载入 (已有环境变量优先, 不覆盖)
    from run_mllm_benchmark import load_env
    load_env()
    key = os.environ.get(f'{p}_API_KEY')
    url = os.environ.get(f'{p}_BASE_URL', '')
    model = os.environ.get(f'{p}_MODEL', '')
    if not (key and url and model):
        raise SystemExit(
            f'缺少环境变量: {p}_API_KEY / {p}_BASE_URL / {p}_MODEL\n'
            f'请在 .mllm_env 中设置 (见本文件头部注释的接入方案 A-F)')

    import glob
    imgs = sorted(glob.glob(f'{COCO}/*.jpg'))[:args.n]
    print(f'第二 MLLM 攻击者: provider={args.provider} model={model} n={len(imgs)}')

    # 复用论文的攻击管线 (run_inpaint_attack), 保证两个 MLLM 走完全相同的攻击
    from run_inpaint_attack import load_pipe
    pipe = load_pipe()

    rng = np.random.RandomState(42)
    out = {'meta': {'provider': args.provider, 'model': model, 'n': len(imgs),
                    'strong_prompt': args.strong,
                    'msglen': args.msglen, 'delta': args.delta,
                    'ts': datetime.datetime.now().isoformat()},
           'per_image': [], 'summary': {}}

    for idx, ip in enumerate(imgs):
        o = crop8(np.array(Image.open(ip).convert('RGB')))
        # 与论文一致: 每张图使用独立随机生成的消息
        msg = rng.randint(0, 2, args.msglen).astype(np.uint8)
        edit = ask_edit(url, key, model, o, strong=args.strong)
        if edit is None:
            print(f'  [{idx+1}] {os.path.basename(ip)}: 指令生成失败, 跳过')
            continue
        # --- 与论文完全一致的攻击流程 ---
        # 1) 先在原图上嵌入水印
        wm = embed_signal(o, msg, args.delta)
        me, _ = decode_signal(wm, args.msglen, args.delta, use_check=False)
        clean = float(bit_accuracy(msg, me))
        # 2) 把**已嵌水印的图**送入论文的 sd_execute (SD inpainting + 掩膜;
        #    掩膜外强制保持原样, 小区域自动并入实例掩膜)
        try:
            from run_mllm_benchmark import sd_execute
            att = sd_execute(edit, wm, steps=args.steps)
        except Exception as e:
            print(f'  [{idx+1}] 扩散失败: {e}')
            continue

        # 3) 损伤外溢度量（bbox 内 vs 外）
        fm, n_h, n_w = damage_leakage(wm, att)
        H, W = o.shape[:2]
        x1, y1, x2, y2 = [int(v * q) for v, q in zip(edit['bbox'], (W, H, W, H))]
        cbx = lambda j: (j + 0.5) * W / n_w
        cby = lambda i: (i + 0.5) * H / n_h
        ii = oo = ti = to = 0
        for i in range(n_h):
            for j in range(n_w):
                ins = x1 <= cbx(j) <= x2 and y1 <= cby(i) <= y2
                if ins: ti += 1; ii += fm[i, j] > 0
                else:   to += 1; oo += fm[i, j] > 0

        # 4) 五种解码方案（与论文同一套）
        me, _ = decode_signal(att, args.msglen, args.delta, use_check=False)
        uni = float(bit_accuracy(msg, me))
        me, _ = decode_signal(att, args.msglen, args.delta, use_check=True)
        v1 = float(bit_accuracy(msg, me))
        me, _ = decode_signal(att, args.msglen, args.delta, use_check=True, soft=True)
        soft = float(bit_accuracy(msg, me))
        me, _ = decode_signal(att, args.msglen, args.delta, use_check=True,
                              soft=True, dilate=1)
        v2 = float(bit_accuracy(msg, me))

        out['per_image'].append({
            'img': os.path.basename(ip), 'instruction': edit['instruction'],
            'bbox': edit['bbox'], 'clean_ba': clean,
            'uni_ba': uni, 'v1_ba': v1, 'soft_ba': soft, 'v2_ba': v2,
            'inside_fail': ii / max(ti, 1), 'outside_fail': oo / max(to, 1)})
        if (idx + 1) % 5 == 0:
            print(f'  [{idx+1}/{len(imgs)}] 内{100*ii/max(ti,1):.0f}% 外{100*oo/max(to,1):.0f}% '
                  f'uni={uni*100:.1f}% v2={v2*100:.1f}%')

    pi = out['per_image']
    if pi:
        out['summary'] = {
            'n': len(pi),
            'inside_fail': float(np.mean([r['inside_fail'] for r in pi])),
            'outside_fail': float(np.mean([r['outside_fail'] for r in pi])),
            'uni_ba': float(np.mean([r['uni_ba'] for r in pi])),
            'v1_ba': float(np.mean([r['v1_ba'] for r in pi])),
            'soft_ba': float(np.mean([r['soft_ba'] for r in pi])),
            'v2_ba': float(np.mean([r['v2_ba'] for r in pi])),
            'gain_pt': float((np.mean([r['v2_ba'] for r in pi])
                              - np.mean([r['uni_ba'] for r in pi])) * 100)}
        s = out['summary']
        print(f"\n=== 汇总 ({args.provider}) ===")
        print(f"  区内失败率 {s['inside_fail']*100:.1f}%  区外失败率 {s['outside_fail']*100:.1f}%")
        print(f"  uniform {s['uni_ba']*100:.1f}%  v2 {s['v2_ba']*100:.1f}%  gain +{s['gain_pt']:.1f}pt")

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = f'{BASE}/results/second_mllm_{args.provider}_{ts}.json'

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
