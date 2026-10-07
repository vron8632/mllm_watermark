#!/usr/bin/env python3
"""按编辑面积（edited area / image area）对攻击集分层，统计各解码变体的 BA。

回应审稿 major-7：主文的分层用的是 check-failure rate（防御侧量），审稿建议
改用或补充攻击侧量（edited area / image area）。本脚本在**已发布攻击集**上做
按面积的分层，协议与 code/eval_attackset.py 完全一致：
  嵌入水印 → 把攻击图的 bbox 区域贴回水印图（bbox 外保持水印图）→ 解码
  msg = RandomState(42).randint(0,2,256), Δ=15, payload=(4,1), check=(3,2), key seed 42

输出: results/attackset_area_strata_<ts>.json
"""
import sys, os, json, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from run_idea_probe import load, psnr, bit_accuracy
from run_signal_ra import embed_signal, decode_signal
from run_mllm_benchmark import extract_uniform_bbox_ra

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 与 tab:strata 相同的分层边界（按百分比）
BANDS = [("<15%", 0.0, 0.15), ("15-35%", 0.15, 0.35), (">35%", 0.35, 1.01)]


def load_pair(img_dir, cid):
    """兼容两种命名: {cid}_orig.png / orig_{cid}.png"""
    for a, b in ((f"{cid}_orig.png", f"{cid}_att.png"),
                 (f"orig_{cid}.png", f"att_{cid}.png")):
        pa, pb = os.path.join(img_dir, a), os.path.join(img_dir, b)
        if os.path.exists(pa) and os.path.exists(pb):
            o = np.array(Image.open(pa).convert("RGB"))
            t = np.array(Image.open(pb).convert("RGB"))
            return o, t
    return None, None


def paste_edit(wm, att, bbox01):
    H, W = wm.shape[:2]
    x1, y1, x2, y2 = bbox01
    bx1, by1 = int(round(x1 * W)), int(round(y1 * H))
    bx2, by2 = int(round(x2 * W)), int(round(y2 * H))
    out = wm.copy()
    out[by1:by2, bx1:bx2] = att[by1:by2, bx1:bx2]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attackset",
                    default=f"{BASE}/results/attackset_ip2p_20260908_075547")
    ap.add_argument("--msglen", type=int, default=256)
    ap.add_argument("--delta", type=int, default=15)
    ap.add_argument("--maxn", type=int, default=0)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.attackset, "meta.json")))
    img_dir = os.path.join(args.attackset, "images")
    rng = np.random.RandomState(42)
    msg = rng.randint(0, 2, args.msglen).astype(np.uint8)

    rows = []
    samples = meta["samples"][: args.maxn] if args.maxn else meta["samples"]
    for k, s in enumerate(samples):
        cid = s["id"]
        orig, att = load_pair(img_dir, cid)
        if orig is None:
            print(f"  [skip] {cid} 图像缺失", flush=True)
            continue
        orig = orig[: orig.shape[0] - orig.shape[0] % 8,
                    : orig.shape[1] - orig.shape[1] % 8]
        att = att[: orig.shape[0], : orig.shape[1]]
        bbox = s["bbox"]
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        row = {"id": cid, "img": s["source"], "bbox": bbox, "area_frac": area}
        try:
            wm = embed_signal(orig, msg, args.delta)
            att_wm = paste_edit(wm, att, bbox)
            me, _ = decode_signal(att_wm, args.msglen, args.delta, use_check=False)
            row["ba_undefended"] = bit_accuracy(msg, me)
            me, nf = decode_signal(att_wm, args.msglen, args.delta, use_check=True,
                                   soft=True, dilate=1)
            row["ba_ours"] = bit_accuracy(msg, me)
            row["check_fail_frac"] = nf / ((orig.shape[0] // 8) * (orig.shape[1] // 8))
            me = extract_uniform_bbox_ra(att_wm, args.msglen, args.delta, bbox)
            row["ba_oracle"] = bit_accuracy(msg, me)
        except Exception as e:
            row["error"] = repr(e)[:120]
        rows.append(row)
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(samples)}", flush=True)

    ok = [r for r in rows if "ba_ours" in r]
    print(f"\n有效 {len(ok)}/{len(rows)}")

    out = {"meta": {"attackset": os.path.basename(args.attackset),
                    "n": len(ok), "msglen": args.msglen, "delta": args.delta,
                    "protocol": "eval_attackset: paste bbox region, then decode",
                    "bands": [b[0] for b in BANDS],
                    "ts": datetime.datetime.now().isoformat()},
           "per_image": rows, "strata": {}, "corr": {}}

    print(f"\n{'band':<9}{'n':>5}{'area%':>9}{'undef':>9}{'ours':>9}"
          f"{'oracle':>9}{'gain':>8}")
    for name, lo, hi in BANDS:
        sel = [r for r in ok if lo <= r["area_frac"] < hi]
        if not sel:
            out["strata"][name] = {"n": 0}
            continue
        u = float(np.mean([r["ba_undefended"] for r in sel])) * 100
        o = float(np.mean([r["ba_ours"] for r in sel])) * 100
        g = float(np.mean([r["ba_oracle"] for r in sel])) * 100
        a = float(np.mean([r["area_frac"] for r in sel])) * 100
        f = float(np.mean([r["check_fail_frac"] for r in sel])) * 100
        out["strata"][name] = {"n": len(sel), "mean_area_pct": a,
                               "mean_check_fail_pct": f,
                               "ba_undefended": u, "ba_ours": o,
                               "ba_oracle": g, "gain": o - u}
        print(f"{name:<9}{len(sel):>5}{a:>9.1f}{u:>9.1f}{o:>9.1f}"
              f"{g:>9.1f}{o-u:>+8.1f}")

    # 两个分层轴的关系（审稿关心：check-failure 是防御侧量）
    ar = np.array([r["area_frac"] for r in ok])
    fr = np.array([r["check_fail_frac"] for r in ok])
    if ar.std() > 0 and fr.std() > 0:
        c = float(np.corrcoef(ar, fr)[0, 1])
        out["corr"]["area_vs_checkfail"] = c
        print(f"\npearson r( edited area , check-failure rate ) = {c:.3f}")

    out["summary"] = {k: float(np.mean([r[k] for r in ok])) * 100
                      for k in ("ba_undefended", "ba_ours", "ba_oracle")}
    print("overall:", {k: round(v, 1) for k, v in out["summary"].items()})

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = f"{BASE}/results/attackset_area_strata_{ts}.json"
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("written:", p)


if __name__ == "__main__":
    main()
