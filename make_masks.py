#!/usr/bin/env python3
"""Tile manifest + mask check (and fallback masks) for an `annotate.py run` output folder.

If annotate.py already wrote exact masks to masks/ (current version), they are verified and used. If not (older run),
the masks are rasterised from the YOLO polygons instead (approximation, see below).

Usage:  python make_masks.py results/

Writes
  <out>/masks/<frame>_r<row>_c<col>.png   uint8, 1024x1024, same grid as images/:
                                            0 = background, 1 = star/blob (class 0), 2 = streak (class 1)
  <out>/manifest.csv                      one row per tile: source frame, row/col, valid pixel box in the source
                                            frame, padding, object counts, relative file paths

How the masks are made: each YOLO polygon in labels/ is rasterised (cv2.fillPoly, polygon vertices are
pixel-centre coordinates, boundary pixels included). So the masks are exactly what the YOLO labels describe.
They match the detection mask to within the polygon simplification (<= ~0.4 px, a few pixels more where the
exporter had to fall back to a convex hull / box); `--check` measures this when `full/*_labels.npy` exists
(run annotate.py with --save-labels).
Overlap policy: where a star polygon and a streak polygon overlap, the pixel is labelled STREAK (a star sitting on a
streak is far more often part of the track than a separate object); the number of such pixels is reported.
Padding: the padded band of edge tiles is always 0 (padding never carries labels).
"""
import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np


def raster_tile(label_path, T):
    """Return (mask uint8 TxT, n_star, n_streak, n_overlap_pixels)."""
    layers = {0: np.zeros((T, T), np.uint8), 1: np.zeros((T, T), np.uint8)}
    n = {0: 0, 1: 0}
    with open(label_path) as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            cls = int(tok[0])
            pts = np.array([float(v) for v in tok[1:]]).reshape(-1, 2) * T - 0.5   # -> pixel indices
            cv2.fillPoly(layers[cls], [np.round(pts).astype(np.int32)], 1)
            n[cls] += 1
    overlap = int((layers[0] & layers[1]).sum())
    mask = layers[0].copy()
    mask[layers[1] > 0] = 2
    return mask, n[0], n[1], overlap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--check", action="store_true", help="compare masks with full/*_labels.npy if present")
    a = ap.parse_args()
    d, T = a.out_dir, a.tile

    metas = sorted(glob.glob(os.path.join(d, "full", "*_meta.json")))
    if not metas:
        sys.exit(f"no *_meta.json in {d}/full - run annotate.py first")
    os.makedirs(os.path.join(d, "masks"), exist_ok=True)
    marker = os.path.join(d, "masks", "FROM_POLYGONS.txt")   # present = masks here are polygon rasterisations, not exact

    rows, problems = [], []
    tot_overlap = 0
    n_exact = 0
    agree = {}
    for mp in metas:
        doc = json.load(open(mp))
        m = doc["meta"]
        stem, nty, ntx, H, W = m["stem"], m["ty"], m["tx"], m["height"], m["width"]
        lab_path = os.path.join(d, "full", stem + "_labels.npy")
        gt = None
        if a.check and os.path.exists(lab_path):
            lbl = np.load(lab_path, mmap_mode="r")
            cls_of = {o["id"]: (1 if o["cls"] == "star" else 2) for o in doc["objects"]}
            lut = np.zeros(int(max(cls_of, default=0)) + 1, np.uint8)
            for k, v in cls_of.items():
                lut[k] = v
            gt = (lbl, lut)
        inter = union = same_cls = fg = 0
        for ty in range(nty):
            for tx in range(ntx):
                name = f"{stem}_r{ty:02d}_c{tx:02d}"
                lp = os.path.join(d, "labels", name + ".txt")
                ip = os.path.join(d, "images", name + ".png")
                if not os.path.exists(lp):
                    problems.append(f"{name}: label file missing")
                    continue
                mp_ = os.path.join(d, "masks", name + ".png")
                mask_r, ns, nk, ov = raster_tile(lp, T)          # polygon counts (always) + fallback raster
                if os.path.exists(mp_) and not os.path.exists(marker):   # exact mask written by annotate.py: use it
                    mask = cv2.imread(mp_, cv2.IMREAD_UNCHANGED)
                    n_exact += 1
                    if mask is None or mask.shape != (T, T) or mask.dtype != np.uint8 or mask.max() > 2:
                        problems.append(f"{name}: mask is not a {T}x{T} uint8 image with values 0/1/2")
                        continue
                else:                                            # older run without masks: rasterise the polygons
                    mask = mask_r
                    tot_overlap += ov
                x0, y0 = tx * T, ty * T
                x1, y1 = min(x0 + T, W), min(y0 + T, H)
                vw, vh = x1 - x0, y1 - y0
                if mask[vh:, :].any() or mask[:, vw:].any():
                    problems.append(f"{name}: mask has foreground inside the padded band")
                im = cv2.imread(ip, cv2.IMREAD_UNCHANGED) if os.path.exists(ip) else None
                if im is None or im.shape != mask.shape:
                    problems.append(f"{name}: mask {mask.shape} does not align with image tile "
                                    f"{None if im is None else im.shape}")
                if mask is mask_r:
                    cv2.imwrite(mp_, mask)
                rows.append(dict(tile=name, source=stem, row=ty, col=tx, x0=x0, y0=y0, x1=x1, y1=y1,
                                 valid_w=vw, valid_h=vh, pad_right=T - vw, pad_bottom=T - vh,
                                 n_star_polygons=ns, n_streak_polygons=nk,
                                 image=f"images/{name}.png", raw16=f"tiles_raw16/{name}.png",
                                 label=f"labels/{name}.txt", mask=f"masks/{name}.png"))
                if gt is not None:
                    lbl, lut = gt
                    tile = np.zeros((T, T), np.int64)
                    sub = np.asarray(lbl[y0:y1, x0:x1])
                    tile[:vh, :vw] = sub
                    g = lut[tile]
                    gf, mf = g > 0, mask > 0
                    inter += int((gf & mf).sum())
                    union += int((gf | mf).sum())
                    both = gf & mf
                    same_cls += int((g[both] == mask[both]).sum())
                    fg += int(both.sum())
        if gt is not None:
            agree[stem] = (inter / max(union, 1), same_cls / max(fg, 1))

    with open(os.path.join(d, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} masks to {os.path.join(d, 'masks')} and {os.path.join(d, 'manifest.csv')}")
    print(f"{n_exact} masks were exact (written by annotate.py); {len(rows) - n_exact} rasterised from the polygons")
    if len(rows) - n_exact:
        open(marker, "w").write("These masks were rasterised from the YOLO polygons by make_masks.py (approximation: holes filled, "
                                "<=~0.4 px simplification, hull fallbacks). Re-run annotate.py (current version) for exact masks.\n")
        print(f"star/streak overlap pixels resolved as streak: {tot_overlap}")
    for stem, (iou, cls_ok) in agree.items():
        print(f"[{stem}] mask vs detection label map: foreground IoU {iou:.4f}; class agreement on shared pixels {cls_ok:.4f}")
    if problems:
        print(f"\nFAILED: {len(problems)} problem(s)")
        for p in problems[:30]:
            print("  -", p)
        sys.exit(1)
    print("OK: every mask is 1024x1024, aligned with its image tile, and empty in the padded band.")


if __name__ == "__main__":
    main()
