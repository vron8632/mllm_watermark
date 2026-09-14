#!/usr/bin/env python3
"""动机图/框架图用的**示例样本**（集中配置，改一处即可换图）.

当前样本来自 `results/attackset_hires_sample`（run_hires_demo_samples.py 生成）:
在**原始 COCO 分辨率**上用 InstructPix2Pix 重跑, 并由 DeepSeek 自然度评分挑选
—— 256px 版本印刷过小、且编辑痕迹明显, 已弃用.

改例子只改这里的 CID, 然后重跑:
  cd code && python make_figures.py && python export_gpt_assets.py

高分辨率候选（results/hires_sample_scores_*.json，nat=DeepSeek 自然度 0-10）:
  0030  remove the person in the center    nat=9.0  (480x640)  ★ 默认
  0031  remove the donut                   nat=9.0  (496x368)
  0005  remove the skier in the center     nat=9.0  (640x424)
  0004  replace the plush bear with a toy car  nat=3.0
  0002  replace the bookshelf with a TV    nat=2.0
"""
import os, json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIRES = f'{BASE}/results/attackset_hires_sample'
META = f'{HIRES}/meta.json'

# ---- 当前示例 (像素指标 + 多模态模型双重核查通过的样本) ----
# 核查记录 (results/hires_screen_*.json + verify_hires_samples):
#   0002_42  replace bookshelf->TV      slight    seam 1.05  ★ teaser
#   0030_42  remove person (snow)       slight    seam 1.02
#   0045_42  change jacket yellow->red  INVISIBLE seam 1.29
#   0032_42  replace skull->smiley      slight    seam 0.95
DEMO = dict(
    cid='0002_42',
    instr_en='replace the bookshelf on the right with a television',
)


def sample(hires=True):
    """返回 (cid, source, bbox, instruction_en, image_dir).

    hires=True  -> 原始分辨率样本 (attackset_hires_sample, 论文插图用)
    hires=False -> 旧 256px 攻击集 (attackset_20260817_082819, 仅兼容保留)
    """
    if not hires or not os.path.exists(META):
        old = f'{BASE}/results/attackset_20260817_082819'
        meta = json.load(open(f'{old}/meta.json'))
        s = next(x for x in meta['samples'] if x['id'] == DEMO['cid'])
        url = f'http://images.cocodataset.org/val2017/{s["source"]}'
        return DEMO['cid'], s['source'], s['bbox'], s['instruction'], DEMO['instr_en'], url, f'{old}/images'
    meta = json.load(open(META))
    s = next(x for x in meta['samples'] if x['id'] == DEMO['cid'])
    url = f'http://images.cocodataset.org/val2017/{s["source"]}'
    return (DEMO['cid'], s['source'], s['bbox'], s['instruction'],
            DEMO['instr_en'], url, f'{HIRES}/images')


if __name__ == '__main__':
    print(sample())
