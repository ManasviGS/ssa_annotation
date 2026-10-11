#!/usr/bin/env python3
"""Validate the YOLO-Ultralytics segmentation export written by `annotate.py run`.

Usage: python validate_dataset.py results/ [--strict]

Checks, for every label file:
  * a matching image exists in images/ (and vice versa), 1024x1024, 8-bit; tile raw16 matches if present
  * every line is `cls x1 y1 x2 y2 ...` with an integer class in {0,1} and an even number of >= 3 points
  * all coordinates are finite and inside [0, 1]
  * every polygon has non-zero area and no self-intersection
  * no label lies in the reflect-padded band of an edge tile (padding never carries labels)
Exit code 0 = all good, 1 = problems found.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _seg_cross(p1, p2, p3, p4):
    """True if open segments p1p2 and p3p4 properly cross."""
    d1, d2 = _orient(p3, p4, p1), _orient(p3, p4, p2)
    d3, d4 = _orient(p1, p2, p3), _orient(p1, p2, p4)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def self_intersects(pts):
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue          # adjacent edges (closing edge)
            c, d = pts[j], pts[(j + 1) % n]
            if _seg_cross(a, b, c, d):
                return True
    return False


def poly_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def check_label_file(path, T=1024, valid_w=None, valid_h=None, tile_origin=(0, 0)):
    """Return a list of problem strings for one label file."""
    problems, counts = [], {0: 0, 1: 0}
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    for ln, line in enumerate(lines, 1):
        tok = line.split()
        try:
            cls = int(tok[0])
            vals = np.array([float(v) for v in tok[1:]], np.float64)
        except ValueError:
            problems.append(f"{path}:{ln}: non-numeric token")
            continue
        if cls not in (0, 1) or tok[0] != str(cls):
            problems.append(f"{path}:{ln}: class id {tok[0]!r} not in {{0,1}}")
            continue
        if len(vals) % 2 or len(vals) < 6:
            problems.append(f"{path}:{ln}: need an even number of >=3 points, got {len(vals)} values")
            continue
        if not np.isfinite(vals).all() or vals.min() < 0 or vals.max() > 1:
            problems.append(f"{path}:{ln}: coordinate outside [0,1] or not finite")
            continue
        pts = vals.reshape(-1, 2) * T
        if poly_area(pts) <= 1e-6:
            problems.append(f"{path}:{ln}: zero-area polygon")
        elif self_intersects(pts):
            problems.append(f"{path}:{ln}: self-intersecting polygon")
        if valid_w is not None and pts[:, 0].max() > valid_w + 1e-3:
            problems.append(f"{path}:{ln}: label extends into padded columns (valid width {valid_w}px)")
        if valid_h is not None and pts[:, 1].max() > valid_h + 1e-3:
            problems.append(f"{path}:{ln}: label extends into padded rows (valid height {valid_h}px)")
        counts[cls] += 1
    return problems, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--tile", type=int, default=1024)
    a = ap.parse_args()
    T, d = a.tile, a.out_dir
    problems = []

    metas = {}
    for mp in glob.glob(os.path.join(d, "full", "*_meta.json")):
        m = json.load(open(mp))["meta"]
        metas[m["stem"]] = m

    labels = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(d, "labels", "*.txt"))}
    images = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(d, "images", "*.png"))}
    if not labels:
        sys.exit(f"no label files found in {d}/labels")
    for n in sorted(set(labels) ^ set(images)):
        problems.append(f"unpaired file: {n} has {'a label but no image' if n in labels else 'an image but no label'}")

    per_frame = {}
    for name, lp in sorted(labels.items()):
        stem, r, c = name.rsplit("_", 2)
        ty, tx = int(r[1:]), int(c[1:])
        vw = vh = None
        if stem in metas:
            m = metas[stem]
            vw = min(T, m["width"] - tx * T)
            vh = min(T, m["height"] - ty * T)
            if vw <= 0 or vh <= 0 or ty >= m["ty"] or tx >= m["tx"]:
                problems.append(f"{name}: tile index outside the {m['ty']}x{m['tx']} grid")
                continue
        p, cnt = check_label_file(lp, T, vw, vh)
        problems += p
        tot = per_frame.setdefault(stem, {0: 0, 1: 0, "tiles": 0})
        tot[0] += cnt[0]; tot[1] += cnt[1]; tot["tiles"] += 1
        if name in images:
            im = cv2.imread(images[name], cv2.IMREAD_UNCHANGED)
            if im is None or im.shape != (T, T) or im.dtype != np.uint8:
                problems.append(f"{images[name]}: expected {T}x{T} uint8, got {None if im is None else (im.shape, im.dtype)}")
        raw = os.path.join(d, "tiles_raw16", name + ".png")
        if os.path.exists(raw):
            r16 = cv2.imread(raw, cv2.IMREAD_UNCHANGED)
            if r16 is None or r16.shape != (T, T) or r16.dtype != np.uint16:
                problems.append(f"{raw}: expected {T}x{T} uint16")

    yaml = os.path.join(d, "dataset.yaml")
    if not os.path.exists(yaml):
        problems.append("dataset.yaml missing")
    else:
        txt = open(yaml).read()
        if "0: star" not in txt or "1: streak" not in txt:
            problems.append("dataset.yaml class map is not {0: star, 1: streak}")

    print(f"checked {len(labels)} label files, {len(images)} images, {len(per_frame)} frames")
    print(f"{'frame':45s} {'tiles':>5s} {'stars':>8s} {'streaks':>8s}   (polygons incl. tile-clipped pieces)")
    for s, t in sorted(per_frame.items()):
        print(f"{s:45s} {t['tiles']:5d} {t[0]:8d} {t[1]:8d}")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems[:50]:
            print("  -", p)
        if len(problems) > 50:
            print(f"  ... and {len(problems) - 50} more")
        sys.exit(1)
    print("\nOK: every label file passed (class ids, coordinate range, polygon validity, padding, image pairing).")


if __name__ == "__main__":
    main()
