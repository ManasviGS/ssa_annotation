#!/usr/bin/env python3
"""Explain what the detector does on one frame (or a crop of it): where candidate sources are lost.

Usage: python diagnose.py frame.fits [--x0 4000 --y0 2500 --size 2048]
Prints background / noise statistics, noise-correlation check, and how many candidates each rule rejects.
Nothing is written to disk. Paste the output back if a frame gives odd counts.
"""
import argparse

import cv2
import numpy as np
from skimage import measure
from skimage.filters import apply_hysteresis_threshold

import annotate as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fits")
    ap.add_argument("--x0", type=int)
    ap.add_argument("--y0", type=int)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--snr-high", type=float, default=A.CONFIG["snr_high"])
    a = ap.parse_args()

    raw = A.read_fits(a.fits)
    H, W = raw.shape
    x0 = a.x0 if a.x0 is not None else W // 2 - a.size // 2
    y0 = a.y0 if a.y0 is not None else H // 2 - a.size // 2
    img = raw[y0:y0 + a.size, x0:x0 + a.size]
    cfg = dict(A.CONFIG, snr_high=a.snr_high)
    print(f"frame {W}x{H} {raw.dtype}; crop x0={x0} y0={y0} size={img.shape[::-1]}")
    print(f"raw: min={img.min()} p1={np.percentile(img,1):.0f} median={np.median(img):.0f} "
          f"p99={np.percentile(img,99):.0f} max={img.max()}  distinct values={len(np.unique(img))}")

    snr, resid, bg, sig = A.snr_map(img, cfg)
    print(f"noise sigma (mesh): median={np.median(sig):.2f} min={sig.min():.2f} max={sig.max():.2f}")
    # noise correlation: if neighbouring pixels are correlated, lag-1 differences underestimate the noise
    f = img.astype(np.float32)
    for lag in (1, 2, 4):
        d = f[:, lag:] - f[:, :-lag]
        s = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
        print(f"  sigma from lag-{lag} pixel differences: {s:.2f}")
    print(f"smoothed SNR map: p50={np.percentile(snr,50):.2f} p99={np.percentile(snr,99):.2f} "
          f"p99.99={np.percentile(snr,99.99):.2f} max={snr.max():.1f}")

    mask = apply_hysteresis_threshold(snr, cfg["snr_low"], cfg["snr_high"])
    lbl = measure.label(mask, connectivity=2)
    props = measure.regionprops(lbl, intensity_image=snr)
    ring = cv2.boxFilter(resid, -1, (3, 3), normalize=False) - resid
    n_small = n_hot = n_ok = 0
    zs, hot_snr = [], []
    for p in props:
        if p.area < cfg["min_area"]:
            n_small += 1
            continue
        ys, xs = p.coords[:, 0], p.coords[:, 1]
        k = np.argmax(resid[ys, xs])
        z = ring[ys[k], xs[k]] / (sig[ys[k], xs[k]] * np.sqrt(8.0))
        zs.append(z)
        if z < cfg["hot_neigh_snr"]:
            n_hot += 1
            hot_snr.append(p.intensity_max)
        else:
            n_ok += 1
    print(f"\nhysteresis components (seed SNR>={cfg['snr_high']}): {len(props)}")
    print(f"  dropped: area < {cfg['min_area']} px: {n_small}")
    print(f"  dropped: 'hot pixel' rule (neighbours of the peak carry < {cfg['hot_neigh_snr']} sigma): {n_hot}"
          + (f"   (their peak SNR: median {np.median(hot_snr):.1f}, max {np.max(hot_snr):.1f})" if hot_snr else ""))
    print(f"  kept: {n_ok}")
    if zs:
        print("  neighbour significance over components >= min area (p10/p50/p90): "
              + " / ".join(f"{np.percentile(zs, q):.1f}" for q in (10, 50, 90)))
    print("\nHow to read this: a large 'hot pixel' count with high peak SNR means the stars are sharper than the\n"
          "rule assumes. Very large totals with SNR near the seed level mean the noise is under-estimated.")


if __name__ == "__main__":
    main()
