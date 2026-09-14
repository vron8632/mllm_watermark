#!/usr/bin/env python3
"""归档实验结果: 把仅有时间戳的文件名改为自描述名 (参数 + 样本量 + 日期).

动机: 现有结果文件名为 signal_ra_20260913_195130.json, 只含时间戳, 事后无法分辨
      是哪次实验 (n 多少? 什么 delta? 哪个数据集?). 本脚本读 meta 生成自描述名,
      并写一份索引清单到 results/INDEX.md, 便于长期检索.

用法:
  cd code && python archive_results.py --dry-run          # 预览将要重命名什么
  cd code && python archive_results.py                    # 执行
  cd code && python archive_results.py --pattern 'signal_ra_2026*.json'
"""
import os, re, json, glob, argparse, datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = f'{BASE}/results'


def describe(fp):
    """从结果文件的 meta 生成自描述标签."""
    try:
        d = json.load(open(fp))
    except Exception:
        return None
    m = d.get('meta') or {}
    if not isinstance(m, dict):
        return None

    stem = os.path.basename(fp)
    kind = re.sub(r'_\d{8}_\d{6}.*$', '', stem)   # 去掉时间戳部分

    parts = [kind]
    ds = m.get('datasets')
    if ds:
        parts.append(str(ds).replace(',', '-'))
    n = m.get('n')
    if n is None:
        # 有些文件用 per_image 长度推断
        pi = d.get('per_image')
        n = len(pi) if isinstance(pi, list) and pi else None
    if n:
        parts.append(f'n{n}')
    if m.get('msglen'):
        parts.append(f'msg{m["msglen"]}')
    if m.get('delta'):
        parts.append(f'd{m["delta"]}')
    if m.get('model'):
        parts.append(re.sub(r'[^A-Za-z0-9.]+', '', str(m['model'])))
    ts = re.search(r'(\d{8}_\d{6})', stem)
    if ts:
        parts.append(ts.group(1))
    return '_'.join(parts) + '.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', default='*.json')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    files = sorted(glob.glob(f'{RES}/{args.pattern}'))
    plan = []
    for fp in files:
        stem = os.path.basename(fp)
        # 只处理含时间戳的规范实验文件 (跳过索引/清单类)
        if not re.search(r'_\d{8}_\d{6}', stem):
            continue
        new = describe(fp)
        if not new or new == stem:
            continue
        plan.append((fp, os.path.join(RES, new)))

    if not plan:
        print('无需重命名的文件')
        return

    print(f'{"原文件名":<52}{"新文件名"}')
    print('-' * 110)
    for a, b in plan:
        print(f'  {os.path.basename(a):<50}{os.path.basename(b)}')

    if args.dry_run:
        print(f'\n(dry-run, 共 {len(plan)} 个待重命名)')
        return

    done = 0
    for a, b in plan:
        if os.path.exists(b):
            print(f'  跳过(目标已存在): {os.path.basename(b)}')
            continue
        if os.path.exists(a):
            os.rename(a, b)
            done += 1
    print(f'\n已重命名 {done} 个文件')

    write_index()


def write_index():
    """生成 results/INDEX.md 索引, 便于长期检索."""
    rows = []
    for fp in sorted(glob.glob(f'{RES}/*.json')):
        stem = os.path.basename(fp)
        if stem.startswith('INDEX'):
            continue
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        m = d.get('meta') or {}
        if not isinstance(m, dict):
            m = {}
        n = m.get('n') or (len(d['per_image']) if isinstance(d.get('per_image'), list) else None)
        s = d.get('summary') or {}
        size_kb = os.path.getsize(fp) / 1024
        rows.append({
            'file': stem,
            'n': n,
            'datasets': m.get('datasets') or '',
            'msglen': m.get('msglen') or '',
            'delta': m.get('delta') or '',
            'model': m.get('model') or m.get('mllm_model') or '',
            'has_summary': bool(s),
            'kb': round(size_kb, 1),
        })

    lines = ['# results/ 索引', '',
             f'自动生成于 {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}'
             '（`code/archive_results.py`）。', '',
             '| 文件 | n | 数据集 | msglen | Δ | MLLM/生成器 | 含 summary | KB |',
             '|---|---|---|---|---|---|---|---|']
    for r in sorted(rows, key=lambda x: (str(x['datasets']), -(x['n'] or 0))):
        lines.append(f"| `{r['file']}` | {r['n'] or ''} | {r['datasets']} | {r['msglen']} | "
                     f"{r['delta']} | {r['model']} | {'✓' if r['has_summary'] else ''} | {r['kb']} |")
    with open(f'{RES}/INDEX.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'已写索引: results/INDEX.md ({len(rows)} 条)')


if __name__ == '__main__':
    main()
