#!/usr/bin/env python3
"""Make a synthetic SSA-like FITS frame (9568x6380, uint16) with ground truth, for pipeline testing.
Usage: python make_synthetic.py out_dir [seed]"""
import json
import os
import sys

import numpy as np
from astropy.io import fits

W, H = 9568, 6380


def stamp(img, x, y, flux, sig, half=None):
    half = half or int(np.ceil(4 * sig)) + 1
    x0, y0 = int(round(x)), int(round(y))
    ys, xs = np.mgrid[y0 - half:y0 + half + 1, x0 - half:x0 + half + 1]
    ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    g = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sig ** 2)) * flux / (2 * np.pi * sig ** 2)
    img[ys[ok], xs[ok]] += g[ok]


def main(out, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    yy, xx = np.mgrid[0:H:8, 0:W:8]
    grad = 800 + 60 * (xx / W) + 40 * (yy / H)
    bgm = np.kron(grad, np.ones((8, 8)))[:H, :W]
    img = bgm + rng.normal(0, 25, (H, W))
    gt = []
    # stars
    for _ in range(900):
        x, y = rng.uniform(5, W - 5), rng.uniform(5, H - 5)
        flux = 10 ** rng.uniform(2.6, 5.2)
        sig = rng.uniform(1.0, 1.6)
        stamp(img, x, y, flux, sig)
        gt.append(dict(cls="star", x=x, y=y, flux=flux))
    # big saturated stars with halos
    for _ in range(25):
        x, y = rng.uniform(30, W - 30), rng.uniform(30, H - 30)
        stamp(img, x, y, 10 ** 6.3, 1.8)
        stamp(img, x, y, 10 ** 5.2, 5.0, half=25)
        gt.append(dict(cls="star", x=x, y=y, flux=1e6))
    # streaks (incl. short/fat ones)
    for i in range(60):
        L = rng.choice([rng.uniform(12, 25), rng.uniform(25, 140)])
        th = rng.uniform(0, np.pi)
        x, y = rng.uniform(150, W - 150), rng.uniform(150, H - 150)
        fpp = 10 ** rng.uniform(1.8, 3.2)   # flux per px along track
        sig = rng.uniform(1.0, 1.5)
        n = int(L * 2)
        pts = []
        for t in np.linspace(-L / 2, L / 2, n):
            px, py = x + t * np.cos(th), y + t * np.sin(th)
            stamp(img, px, py, fpp * L / n * 1.0, sig)
            pts.append((px, py))
        gt.append(dict(cls="streak", x=x, y=y, length=L, theta=th, flux_per_px=fpp,
                       pts=[(float(a), float(b)) for a, b in pts[::4]]))
    # short, fat "streak-like" blobs (defocused, elongation ~1.4) -> should stay stars
    for _ in range(15):
        x, y = rng.uniform(50, W - 50), rng.uniform(50, H - 50)
        stamp(img, x, y, 10 ** rng.uniform(4.0, 5.0), 2.6)
        gt.append(dict(cls="star", x=x, y=y, flux=1e5))
    # hot pixels / cosmic rays (should be rejected)
    for _ in range(300):
        x, y = int(rng.uniform(0, W)), int(rng.uniform(0, H))
        img[y, x] += rng.uniform(300, 4000)
    img = np.clip(img, 0, 65535).astype(np.uint16)
    fits.PrimaryHDU(img).writeto(os.path.join(out, "synthetic_0.fits"), overwrite=True)
    json.dump(gt, open(os.path.join(out, "synthetic_0_gt.json"), "w"))
    print("wrote", out, "n_gt =", len(gt))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
