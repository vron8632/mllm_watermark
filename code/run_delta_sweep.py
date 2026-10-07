#!/usr/bin/env python3
"""Embedding-step-size (Delta) sensitivity sweep, recomputed from scratch on the
released 200-sample attack set with the paper protocol.

Why: two archived Delta datasets disagree (n=10 says the defense helps at Delta=10,
n=50 says it hurts). This regenerates one authoritative curve.

Protocol: embed (msg = RandomState(42).randint(0,2,256), payload (4,1), check (3,2),
key seed 42) -> paste the edited bbox region back -> decode uniform / v1 / soft / v2 /
oracle, for Delta in {10,15,20,25}.

Usage: cd code && python run_delta_sweep.py --n 200
Output: ../results/delta_sweep_<attackset>_<ts>.json
"""
import sys, os, json, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from run_signal_ra import embed_signal, decode_signal
from run_idea_probe import psnr, bit_accuracy
from run_mllm_benchmark import extract_uniform_bbox_ra

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AS_DEFAULT = f"{BASE}/results/attackset_ip2p_20260908_075547"


def load_pair(img_dir, cid):
    for a, b in ((f"{cid}_orig.png", f"{cid}_att.png"),
                 (f"orig_{cid}.png", f"att_{cid}.png")):
        pa, pb = os.path.join(img_dir, a), os.path.join(img_dir, b)
        if os.path.exists(pa) and os.path.exists(pb):
            return np.array(Image.open(pa).convert("RGB")), np.array(Image.open(pb).convert("RGB"))
    return None, None


def paste(wm, att, bbox):
    H, W = wm.shape[:2]
    x1, y1, x2, y2 = bbox
    out = wm.copy()
    out[int(round(y1*H)):int(round(y2*H)), int(round(x1*W)):int(round(x2*W))] = \
        att[int(round(y1*H)):int(round(y2*H)), int(round(x1*W)):int(round(x2*W))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attackset", default=AS_DEFAULT)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--msglen", type=int, default=256)
    ap.add_argument("--deltas", default="10,15,20,25")
    args = ap.parse_args()
    deltas = [int(x) for x in args.deltas.split(",")]

    meta = json.load(open(os.path.join(args.attackset, "meta.json")))
    img_dir = os.path.join(args.attackset, "images")
    rng = np.random.RandomState(42)
    msg = rng.randint(0, 2, args.msglen).astype(np.uint8)

    rows = []
    samples = meta["samples"][: args.n]
    for k, s in enumerate(samples):
        cid, bbox = s["id"], s["bbox"]
        o, a = load_pair(img_dir, cid)
        if o is None:
            continue
        o = o[: o.shape[0] - o.shape[0] % 8, : o.shape[1] - o.shape[1] % 8]
        a = a[: o.shape[0], : o.shape[1]]
        try:
            for dl in deltas:
                wm = embed_signal(o, msg, dl)
                att_wm = paste(wm, a, bbox)
                row = {"id": cid, "delta": dl, "img": s["source"],
                       "psnr": float(psnr(o, wm))}
                me, _ = decode_signal(att_wm, args.msglen, dl, use_check=False)
                row["uniform"] = float(bit_accuracy(msg, me))
                me, _ = decode_signal(att_wm, args.msglen, dl, use_check=True)
                row["v1"] = float(bit_accuracy(msg, me))
                me, _ = decode_signal(att_wm, args.msglen, dl, use_check=True, soft=True)
                row["soft"] = float(bit_accuracy(msg, me))
                me, _ = decode_signal(att_wm, args.msglen, dl, use_check=True,
                                      soft=True, dilate=1)
                row["v2"] = float(bit_accuracy(msg, me))
                row["oracle"] = float(bit_accuracy(
                    msg, extract_uniform_bbox_ra(att_wm, args.msglen, dl, bbox)))
                rows.append(row)
        except Exception as e:
            rows.append({"id": cid, "delta": -1, "error": repr(e)[:100]})
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(samples)}", flush=True)

    summary = {}
    for dl in deltas:
        rr = [r for r in rows if r.get("delta") == dl and "error" not in r]
        if not rr:
            continue
        summary[str(dl)] = {
            "n": len(rr),
            "psnr": float(np.mean([r["psnr"] for r in rr])),
            **{f"{k}_ba": float(np.mean([r[k] for r in rr]))
               for k in ["uniform", "v1", "soft", "v2", "oracle"]},
        }
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fp = f"{BASE}/results/delta_sweep_attackset200_{ts}.json"
    json.dump({"meta": {"attackset": os.path.basename(args.attackset), "n": len(samples),
                        "msglen": args.msglen, "deltas": deltas, "ts": ts},
               "per_image": rows, "summary": summary},
              open(fp, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("saved:", fp)


if __name__ == "__main__":
    main()
