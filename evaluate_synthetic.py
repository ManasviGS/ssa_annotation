#!/usr/bin/env python3
"""Score detections against make_synthetic.py ground truth. Usage: evaluate_synthetic.py out_dir gt.json"""
import json, sys, numpy as np
out, gtp = sys.argv[1], sys.argv[2]
d = json.load(open(f"{out}/full/synthetic_0_meta.json")); obj = d["objects"]
lbl = np.load(f"{out}/full/synthetic_0_labels.npy"); gt = json.load(open(gtp))
cls = {o["id"]: o["cls"] for o in obj}
H, W = lbl.shape
def hit(x, y, r=3):
    x0, y0 = int(round(x)), int(round(y))
    w = lbl[max(0,y0-r):y0+r+1, max(0,x0-r):x0+r+1]
    ids = np.unique(w[w>0]); return ids
tp = {"star":0,"streak":0}; n = {"star":0,"streak":0}; wrongcls = {"star":0,"streak":0}
miss = []
used = set()
for g in gt:
    pts = g.get("pts", [(g["x"], g["y"])])
    ids = set()
    for x, y in pts: ids |= set(hit(x, y).tolist())
    n[g["cls"]] += 1
    if ids:
        used |= ids
        tp[g["cls"]] += 1
        if not any(cls[i] == g["cls"] for i in ids): wrongcls[g["cls"]] += 1
    else: miss.append(g)
for c in n: print(f"{c}: recall {tp[c]}/{n[c]} = {tp[c]/n[c]:.2%}; detected-but-wrong-class {wrongcls[c]}")
fp = [o for o in obj if o["id"] not in used]
print("unmatched detections (false positives):", len(fp), " of ", len(obj))
if fp: print("  fp peak_snr median", np.median([o["peak_snr"] for o in fp]), "area median", np.median([o["area"] for o in fp]))
ms = [g for g in miss if g["cls"]=="streak"]
print("missed streaks:", len(ms), "fluxpp:", sorted(round(g["flux_per_px"]) for g in ms)[:20], "len:", sorted(round(g["length"]) for g in ms)[:20])
mst = [g for g in miss if g["cls"]=="star"]
print("missed stars:", len(mst), "flux max:", max([g["flux"] for g in mst], default=0))
