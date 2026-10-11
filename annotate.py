#!/usr/bin/env python3
"""
SSA star/streak annotation pipeline (FITS -> pixel masks -> YOLO-Ultralytics segmentation).

Sub-commands
------------
  inspect  <file.fits>               print stats, write a preview crop + downsampled full view
  run      <fits_dir> <out_dir>      full pipeline for every *.fits in a folder
  repatch  <out_dir>                 stitch tiles back to full size and verify lossless round trip

Classes (YOLO ids):  0 = star (blob / point source)   1 = streak (space object)

Pipeline (all steps are deterministic and parameterised in CONFIG):
  1. Read FITS (astropy applies BZERO/BSCALE, so uint16 comes back as uint16).
  2. Background mesh (sigma-clipped block mean, 128 px) + local noise mesh (sigma-clipped std); subtract.
     (Clipped mean/std, not median/MAD: real frames are low-count integer images where MAD collapses to 0.)
  3. Light Gaussian smoothing (matched filter for ~PSF-sized sources) -> per-pixel SNR map.
  4. Hysteresis threshold (seed >= snr_high, grow to >= snr_low) -> binary detection mask.
  5. Hot-pixel / cosmic-ray rejection (neighbours of the peak pixel must carry signal), minimum-area filter.
  6. Connected components -> shape statistics -> class (streak vs blob) via elongation + length.
     Fragmented faint streaks are re-joined if their fragments are collinear.
  7. Detection is done on the FULL frame, then tiles are cut from the label map, so no object
     is ever split by tile boundaries before it is measured.
  8. Frame is padded (reflect) to a multiple of 1024 -> 10 x 7 tiles; padding is tracked and cropped
     on repatching, so the round trip is bit-exact.
  9. Polygons are traced per instance per tile (cv2.findContours), written as YOLO-seg txt.
 10. Exact per-tile pixel masks (uint8: 0 background, 1 star, 2 streak) are written to masks/.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
from astropy.io import fits
from scipy import ndimage as ndi
from skimage import measure
from skimage.filters import apply_hysteresis_threshold

CONFIG = dict(
    tile=1024,
    mesh=128,              # background mesh block size (px)
    smooth_sigma=1.0,      # Gaussian sigma (px) of the detection filter
    snr_high=6.0,          # hysteresis seed (on smoothed SNR map); 5.0 over-detected on the reference image
    snr_low=2.5,           # hysteresis grow limit
    min_area=4,            # px, below this -> noise
    hot_neigh_snr=2.5,     # hot pixel / cosmic ray: the 8 neighbours of the peak pixel carry < this many sigma
    # streak vs blob
    streak_min_len=14.0,       # px (major axis length)
    streak_min_aspect=3.0,     # major / minor
    streak_long_len=40.0,      # long objects may be a little less elongated
    streak_long_aspect=2.0,
    core_frac=0.30,            # core = pixels above this fraction of the object's own peak SNR
    core_min_len=7.0,          # px, core length needed for the core-shape rule
    core_min_aspect=1.9,       # core major/minor needed for the core-shape rule
    core_min_peak_snr=9.0,     # core rule only for well-detected objects (faint clumps are noise-shaped)
    edge_px=5,                 # a component lying >= edge_frac inside this border strip is a
    edge_frac=0.8,             #   hot row/column or border artefact, not an object
    # re-joining of fragmented streaks
    join_radius=3,             # px dilation used to find collinear fragments
    join_min_len=30.0,
    join_min_aspect=4.5,
    min_fragment_area=3,       # drop sliver polygons smaller than this in a tile
    # visual stretch
    stretch_zmax=40.0,
)
CLASSES = {0: "star", 1: "streak"}


# --------------------------------------------------------------------------- IO
def read_fits(path):
    # memmap=False: uint16 FITS carries BZERO=32768, which astropy cannot memory-map
    with fits.open(path, memmap=False) as hdul:
        hdu = next(h for h in hdul if h.data is not None and h.data.ndim == 2)
        data = np.asarray(hdu.data)
    return data


# ------------------------------------------------------------------ background
def _block_stats(img, B, rows_per_band=4):
    """Per-block sky level and noise sigma = sigma-clipped MEAN and STD (see notes in the report).
    A MAD-based sigma collapses to 0 on low-count integer data where >50% of pixels share one value,
    and a median is biased low on skewed Poisson data -- both produced garbage SNR on real frames.
    Computed a few block-rows at a time so that memory use stays small."""
    H, W = img.shape
    nby, nbx = -(-H // B), -(-W // B)
    mean_m = np.empty((nby, nbx), np.float32)
    sd_m = np.empty((nby, nbx), np.float32)
    for by0 in range(0, nby, rows_per_band):
        by1 = min(by0 + rows_per_band, nby)
        band = img[by0 * B: min(by1 * B, H)].astype(np.float32)
        ph, pw = (by1 - by0) * B - band.shape[0], nbx * B - band.shape[1]
        if ph or pw:
            band = np.pad(band, ((0, ph), (0, pw)), mode="reflect")
        blk = band.reshape(by1 - by0, B, nbx, B).transpose(0, 2, 1, 3).reshape(by1 - by0, nbx, B * B)
        mean = np.median(blk, axis=2)
        sd = blk.std(axis=2)
        for _ in range(4):  # clip stars / hot pixels, then re-measure
            w = (np.abs(blk - mean[..., None]) <= 3.0 * np.maximum(sd, 1e-3)[..., None]).astype(np.float32)
            n = np.maximum(w.sum(axis=2), 1.0)
            mean = (blk * w).sum(axis=2) / n
            sd = np.sqrt((((blk - mean[..., None]) ** 2) * w).sum(axis=2) / n)
        mean_m[by0:by1], sd_m[by0:by1] = mean, sd
    return mean_m, sd_m


def _upsample(mesh, B, shape):
    m = ndi.median_filter(mesh, size=3, mode="nearest")
    big = cv2.resize(m.astype(np.float32), (m.shape[1] * B, m.shape[0] * B),
                     interpolation=cv2.INTER_LINEAR)
    return big[: shape[0], : shape[1]]


def background_and_noise(img, B):
    med, mad = _block_stats(img, B)
    return _upsample(med, B, img.shape), _upsample(mad, B, img.shape)


# ------------------------------------------------------------------- detection
def _clipped_std(x, n=4, k=3.0):
    x = np.asarray(x, np.float64).ravel()
    m, sd = x.mean(), x.std()
    for _ in range(n):
        keep = np.abs(x - m) <= k * max(sd, 1e-6)
        m, sd = x[keep].mean(), x[keep].std()
    return sd


def snr_map(img, cfg):
    bg, sig = background_and_noise(img, cfg["mesh"])
    # floor: quantisation noise (1/sqrt(12)) and, relative, half the typical noise -- stops dead/masked
    # regions (noise ~ 0) from turning a single stray count into an enormous SNR
    sig = np.maximum(sig, max(0.29, 0.5 * float(np.median(sig[::8, ::8]))))
    resid = img.astype(np.float32)
    resid -= bg
    del bg                 # not needed any more (saves 245 MB)
    sm = cv2.GaussianBlur(resid, (0, 0), cfg["smooth_sigma"])
    # noise reduction factor of the filter, measured empirically on the frame (clipped std, in units of local sigma)
    sub = (slice(None, None, 4), slice(None, None, 4))
    factor = _clipped_std(sm[sub] / sig[sub]) / max(_clipped_std(resid[sub] / sig[sub]), 1e-6)
    sm /= sig          # in place: no extra full-frame temporaries
    sm /= factor
    return sm, resid, sig


def _props(lbl, snr):
    return measure.regionprops(lbl, intensity_image=snr)


def _core_shape(p, cfg):
    """Length / aspect of the object's core (pixels >= core_frac * own peak SNR).
    A PSF-limited star keeps a round core at any brightness; a short, fat streak keeps an
    elongated core even when its faint wings make the full mask look round-ish."""
    im, msk = p.image_intensity, p.image
    peak = float(im[msk].max())
    core = msk & (im >= max(cfg["core_frac"] * peak, cfg["snr_low"]))
    ys, xs = np.nonzero(core)
    if len(ys) < 5:
        return 0.0, 1.0
    c = np.cov(np.vstack([xs, ys]))
    ev = np.sort(np.linalg.eigvalsh(c))[::-1]
    # +1/12 px^2: variance of a pixel's own extent (keeps 1-px-wide lines finite)
    L = 4 * np.sqrt(max(ev[0], 0) + 1 / 12)
    w = 4 * np.sqrt(max(ev[1], 0) + 1 / 12)
    return float(L), float(L / w)


def _classify(p, cfg):
    L = float(p.axis_major_length)
    w = max(float(p.axis_minor_length), 1.0)
    aspect = L / w
    by_mask = (L >= cfg["streak_min_len"] and aspect >= cfg["streak_min_aspect"]) or \
              (L >= cfg["streak_long_len"] and aspect >= cfg["streak_long_aspect"])
    cL, cA = _core_shape(p, cfg)
    by_core = (cL >= cfg["core_min_len"] and cA >= cfg["core_min_aspect"]
               and float(p.intensity_max) >= cfg["core_min_peak_snr"])
    return (1 if (by_mask or by_core) else 0), L, aspect, cL, cA


def _hysteresis_labels(snr, low, high):
    """Connected components (8-connectivity) of snr > low that contain at least one pixel with snr > high,
    returned as a sequentially-numbered int32 label map. Same result as skimage's
    apply_hysteresis_threshold + label, but touches only the 'high' pixels (skimage sorts all 61 Mpx and
    needs ~1.2 GB extra memory)."""
    labels, n = ndi.label(snr > low, structure=np.ones((3, 3), bool))
    keep = np.zeros(n + 1, bool)
    keep[labels[snr > high]] = True
    keep[0] = False
    lut = np.zeros(n + 1, np.int32)
    lut[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return lut[labels]


def detect(img, cfg=CONFIG):
    """Return (label_map int32, class_by_label dict, objects list, 8-bit display image)."""
    snr, resid, sig = snr_map(img, cfg)
    lbl = _hysteresis_labels(snr, cfg["snr_low"], cfg["snr_high"])

    # ---- reject hot pixels / cosmic rays and tiny noise specks
    drop = np.zeros(lbl.max() + 1, bool)
    ring = cv2.boxFilter(resid, -1, (3, 3), normalize=False) - resid   # sum of the 8 neighbours
    for p in _props(lbl, snr):
        if p.area < cfg["min_area"]:
            drop[p.label] = True
            continue
        ys, xs = p.coords[:, 0], p.coords[:, 1]
        k = np.argmax(resid[ys, xs])
        y, x = ys[k], xs[k]
        # a real source (PSF spans several pixels) always lights up the neighbours of its peak pixel;
        # a hot pixel / cosmic-ray hit is a delta function, so its neighbours are just noise
        if ring[y, x] / (sig[y, x] * np.sqrt(8.0)) < cfg["hot_neigh_snr"]:
            drop[p.label] = True
    # ---- sensor-edge defects (hot rows/columns, border lines)
    H, W = lbl.shape
    e = cfg["edge_px"]
    for p in _props(lbl, snr):
        if p.axis_major_length >= 100:   # long + hugging the border: hot row/column, not a star cut by the edge
            ys, xs = p.coords[:, 0], p.coords[:, 1]
            near = (ys < e) | (ys >= H - e) | (xs < e) | (xs >= W - e)
            if near.mean() >= cfg["edge_frac"]:
                drop[p.label] = True
    lbl[drop[lbl]] = 0
    del ring

    # ---- re-join collinear fragments of faint streaks
    bin_ = lbl > 0
    k = 2 * cfg["join_radius"] + 1
    grp, _ = ndi.label(cv2.dilate(bin_.astype(np.uint8), np.ones((k, k), np.uint8)) > 0, structure=np.ones((3, 3), bool))
    del bin_
    nlab = int(lbl.max())
    if nlab:
        # group id of every fragment, and each group's bounding box -- all work below stays inside
        # the bounding box, never over the whole 61-Mpx frame (that was the slow part)
        # fragment -> group lookup using only the detected pixels (never sorts the whole frame)
        yy_, xx_ = np.nonzero(lbl)
        frag_grp = np.zeros(nlab, np.int64)
        frag_grp[lbl[yy_, xx_] - 1] = grp[yy_, xx_]
        order = np.argsort(frag_grp, kind="stable")
        gslices = ndi.find_objects(grp)
        i = 0
        while i < len(order):
            g = frag_grp[order[i]]
            j = i
            while j < len(order) and frag_grp[order[j]] == g:
                j += 1
            frs = (order[i:j] + 1).tolist()
            i = j
            if g <= 0 or len(frs) < 2:
                continue
            ys0, xs0 = gslices[g - 1]
            sub = lbl[ys0, xs0]
            sel = np.isin(sub, frs)
            ys, xs = np.nonzero(sel)
            if len(ys) < 6:
                continue
            c = np.cov(np.vstack([xs, ys]))
            ev = np.sort(np.linalg.eigvalsh(c))[::-1]
            L, w = 4 * np.sqrt(max(ev[0], 0)), 4 * np.sqrt(max(ev[1], 1e-6))
            if L >= cfg["join_min_len"] and L / max(w, 1.0) >= cfg["join_min_aspect"]:
                closed = cv2.morphologyEx(sel.astype(np.uint8), cv2.MORPH_CLOSE,
                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
                sub[(closed > 0) & ((sub == 0) | sel)] = frs[0]

    # ---- final labels, classes and catalogue
    lbl, _, _ = _relabel(lbl)
    objects, cls_by = [], {}
    for p in _props(lbl, snr):
        c, L, asp, cL, cA = _classify(p, cfg)
        cls_by[p.label] = c
        cy, cx = p.centroid
        objects.append(dict(id=int(p.label), cls=CLASSES[c], cx=float(cx), cy=float(cy),
                            area=int(p.area), length=L, aspect=asp, core_length=cL, core_aspect=cA,
                            peak_snr=float(p.intensity_max)))
    img8 = stretch_from_resid(resid, sig, cfg)
    return lbl, cls_by, objects, img8


def _relabel(lbl):
    from skimage.segmentation import relabel_sequential
    return relabel_sequential(lbl)


# -------------------------------------------------------------------- stretch
def stretch_from_resid(resid, sig, cfg=CONFIG, band=512):
    """8-bit display image: asinh of the background-subtracted frame in units of local noise.
    Done in row bands so that no full-frame float temporaries are created."""
    out = np.empty(resid.shape, np.uint8)
    scale = 1.0 / np.arcsinh(cfg["stretch_zmax"] / 2.0)
    for y in range(0, resid.shape[0], band):
        z = resid[y:y + band] / np.maximum(sig[y:y + band], 1e-3)
        z += 1.0
        np.clip(z, 0, None, out=z)
        z *= 0.5
        np.arcsinh(z, out=z)
        z *= scale
        np.clip(z, 0, 1, out=z)
        out[y:y + band] = (z * 255).astype(np.uint8)
    return out


def stretch8(img, bg, sig, cfg=CONFIG):
    return stretch_from_resid(img.astype(np.float32) - bg, sig, cfg)


# --------------------------------------------------------------------- tiling
def tile_grid(shape, T):
    H, W = shape
    return int(np.ceil(H / T)), int(np.ceil(W / T))


def pad_to_grid(a, T):
    H, W = a.shape[:2]
    ph, pw = (-H) % T, (-W) % T
    if a.ndim == 2:
        return np.pad(a, ((0, ph), (0, pw)), mode="reflect")
    return np.pad(a, ((0, ph), (0, pw), (0, 0)), mode="reflect")


def pad_zero(a, T):
    H, W = a.shape
    return np.pad(a, ((0, (-H) % T), (0, (-W) % T)), mode="constant")


def get_tile(a, ty, tx, T, mode="reflect"):
    """Tile (ty, tx) of a 2-D array; edge tiles are padded to T x T (reflect for images, zeros for labels)."""
    t = a[ty * T:(ty + 1) * T, tx * T:(tx + 1) * T]
    ph, pw = T - t.shape[0], T - t.shape[1]
    return np.pad(t, ((0, ph), (0, pw)), mode=mode) if (ph or pw) else t


def _is_simple(p):
    """True if the closed polygon p (n x 2) has no properly crossing edges (vectorised O(n^2))."""
    n = len(p)
    if n < 4:
        return True
    a, b = p, np.roll(p, -1, axis=0)

    def orient(u, v, w):   # u, v, w: broadcastable (..., 2)
        return (v[..., 0] - u[..., 0]) * (w[..., 1] - u[..., 1]) - (v[..., 1] - u[..., 1]) * (w[..., 0] - u[..., 0])
    A_, B_ = a[:, None, :], b[:, None, :]
    C_, D_ = a[None, :, :], b[None, :, :]
    d1, d2 = orient(C_, D_, A_), orient(C_, D_, B_)
    d3, d4 = orient(A_, B_, C_), orient(A_, B_, D_)
    cross = (d1 * d2 < 0) & (d3 * d4 < 0)
    i = np.arange(n)
    nonadj = ((i[:, None] - i[None, :]) % n > 1) & ((i[None, :] - i[:, None]) % n > 1)
    return not bool((cross & nonadj).any())


def _finalize(c, off, T):
    """Pixel-contour -> normalised, 6-decimal polygon in YOLO coordinates (what is written to disk)."""
    return np.round(np.clip(c.astype(np.float64) + off + 0.5, 0, T) / T, 6)


def _valid_poly(q, T):
    """q: normalised, already rounded. Valid = >=3 distinct vertices, non-zero area, no crossing edges
    (checked on the rounded values, because touching vertices turn into crossings after rounding)."""
    if len(q) < 3 or len(np.unique(q, axis=0)) != len(q):
        return False
    x, y = q[:, 0], q[:, 1]
    if abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5 * T * T < 0.25:
        return False
    return _is_simple(q * T)


def _polygon_for_contour(c0, off, T):
    """Best valid polygon for one contour. Order: simplified (0.4 px) -> (0.15 px) -> raw contour ->
    convex hull -> oriented box (1 px margin) for degenerate one-pixel-wide shapes. approxPolyDP can make a
    1-px spur (contour goes out and back along the same pixels) cross itself, hence the fallbacks."""
    cands = [cv2.approxPolyDP(c0, 0.4, True), cv2.approxPolyDP(c0, 0.15, True), c0]
    # coarser simplifications keep the shape's concavities, unlike the convex hull (which over-covers long,
    # bent or star-bearing streaks by up to 2x); the hull is the last resort
    cands += [cv2.approxPolyDP(c0, eps, True) for eps in (0.8, 1.2, 2.0, 3.0)]
    cands.append(cv2.convexHull(c0))
    for c in cands:
        q = _finalize(c.reshape(-1, 2), off, T)
        if _valid_poly(q, T):
            return q
    (cx, cy), (w, h), ang = cv2.minAreaRect(c0.reshape(-1, 2).astype(np.float32))
    box = cv2.boxPoints(((cx, cy), (w + 1.0, h + 1.0), ang))
    q = _finalize(box, off, T)
    return q if _valid_poly(q, T) else None


def polygons_for_tile(lbl_tile, cls_by, T, min_frag):
    """YOLO-seg lines for one tile. Each object is traced inside its own bounding box only
    (re-scanning the whole tile once per object is O(objects x tile) and very slow on dense frames)."""
    lines = []
    for lab0, sl in enumerate(ndi.find_objects(lbl_tile)):
        if sl is None:
            continue
        lab = lab0 + 1
        m = np.pad((lbl_tile[sl] == lab).astype(np.uint8), 1)   # 1-px zero border keeps edge contours closed
        off = np.array([sl[1].start - 1, sl[0].start - 1], np.float32)
        if int(m.sum()) < min_frag:
            continue
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c0 in cnts:
            q = _polygon_for_contour(c0, off, T)
            if q is not None:
                lines.append(f"{cls_by[int(lab)]} " + " ".join(f"{v:.6f}" for v in q.reshape(-1)))
    return lines


# ----------------------------------------------------------------------- run
def process_image(path, out_dir, cfg=CONFIG, keep_empty=True, save_labels=False, force=False):
    import time
    T = cfg["tile"]
    stem = os.path.splitext(os.path.basename(path))[0]
    done = os.path.join(out_dir, "full", stem + ".done")
    if os.path.exists(done) and not force:
        print(f"[{stem}] already processed -> skipped (use --force to redo)")
        with open(os.path.join(out_dir, "full", stem + "_meta.json")) as f:
            return json.load(f)["objects"]
    t0 = time.time()

    def tick(msg):
        print(f"[{stem}]   {msg}  ({time.time() - t0:.0f} s elapsed)", flush=True)

    raw = read_fits(path)
    H, W = raw.shape
    print(f"[{stem}] {W}x{H} dtype={raw.dtype}", flush=True)
    lbl, cls_by, objects, img8 = detect(raw, cfg)
    n_star = sum(o["cls"] == "star" for o in objects)
    n_str = sum(o["cls"] == "streak" for o in objects)
    print(f"[{stem}] detected {n_star} stars, {n_str} streaks", flush=True)
    tick("detection done")

    nty, ntx = tile_grid(raw.shape, T)

    for sub in ("images", "labels", "tiles_raw16", "masks", "full"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    # class lookup for the pixel-level masks: 0 = background, 1 = star/blob, 2 = streak (exact detection map,
    # not the polygon approximation; every pixel has one label, so classes never overlap)
    mask_lut = np.zeros(int(lbl.max()) + 1, np.uint8)
    for k_, c_ in cls_by.items():
        mask_lut[k_] = c_ + 1
    n_written = 0
    for ty in range(nty):
        for tx in range(ntx):
            name = f"{stem}_r{ty:02d}_c{tx:02d}"
            lines = polygons_for_tile(get_tile(lbl, ty, tx, T, "constant"), cls_by, T, cfg["min_fragment_area"])
            if not lines and not keep_empty:
                continue
            cv2.imwrite(os.path.join(out_dir, "images", name + ".png"), get_tile(img8, ty, tx, T))
            cv2.imwrite(os.path.join(out_dir, "tiles_raw16", name + ".png"), get_tile(raw, ty, tx, T).astype(np.uint16))
            with open(os.path.join(out_dir, "labels", name + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            cv2.imwrite(os.path.join(out_dir, "masks", name + ".png"), mask_lut[get_tile(lbl, ty, tx, T, "constant")])
            n_written += 1
    tick(f"{n_written} tiles written")

    meta = dict(stem=stem, height=H, width=W, tile=T, ty=nty, tx=ntx, padded=[nty * T, ntx * T])
    with open(os.path.join(out_dir, "full", stem + "_meta.json"), "w") as f:
        json.dump(dict(meta=meta, objects=objects, config=cfg), f, indent=1)
    if save_labels:   # 260 MB per frame, only needed for evaluation / debugging
        np.save(os.path.join(out_dir, "full", stem + "_labels.npy"), lbl)

    # full-size overlay for visual inspection
    overlay_full(img8, lbl, cls_by, os.path.join(out_dir, "full", stem + "_overlay.png"))
    cv2.imwrite(os.path.join(out_dir, "full", stem + "_stretched.png"), img8)
    tick("overlay + stretched image written")
    open(done, "w").close()
    return objects


def overlay_full(img8, lbl, cls_by, out_path, alpha=0.6):
    rgb = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    lut = np.zeros((int(lbl.max()) + 1, 3), np.uint8)
    for k, c in cls_by.items():
        lut[k] = (255, 120, 60) if c == 0 else (60, 60, 255)   # BGR: star = blue-ish, streak = red
    m = lbl > 0
    col = lut[lbl[m]].astype(np.float32)
    rgb[m] = (rgb[m] * (1 - alpha) + col * alpha).astype(np.uint8)
    cv2.imwrite(out_path, rgb)


# -------------------------------------------------------------------- repatch
def repatch(out_dir):
    ok = True
    for mp in sorted(glob.glob(os.path.join(out_dir, "full", "*_meta.json"))):
        meta = json.load(open(mp))["meta"]
        stem, T = meta["stem"], meta["tile"]
        big16 = np.zeros(meta["padded"], np.uint16)
        big8 = np.zeros(meta["padded"], np.uint8)
        have = 0
        for ty in range(meta["ty"]):
            for tx in range(meta["tx"]):
                n = f"{stem}_r{ty:02d}_c{tx:02d}"
                p16 = os.path.join(out_dir, "tiles_raw16", n + ".png")
                if not os.path.exists(p16):
                    continue
                sl = (slice(ty * T, (ty + 1) * T), slice(tx * T, (tx + 1) * T))
                big16[sl] = cv2.imread(p16, cv2.IMREAD_UNCHANGED)
                big8[sl] = cv2.imread(os.path.join(out_dir, "images", n + ".png"), cv2.IMREAD_UNCHANGED)
                have += 1
        big16, big8 = big16[: meta["height"], : meta["width"]], big8[: meta["height"], : meta["width"]]
        cv2.imwrite(os.path.join(out_dir, "full", stem + "_repatched16.png"), big16)
        cv2.imwrite(os.path.join(out_dir, "full", stem + "_repatched8.png"), big8)
        # verify against source stretched image if present
        sp = os.path.join(out_dir, "full", stem + "_stretched.png")
        same = np.array_equal(big8, cv2.imread(sp, cv2.IMREAD_UNCHANGED)) if os.path.exists(sp) else None
        print(f"[{stem}] repatched {have} tiles -> {big16.shape[1]}x{big16.shape[0]}; 8-bit identical to source stretch: {same}")
        ok &= bool(same in (True, None))
    return ok


def verify_against_fits(fits_dir, out_dir):
    ok = True
    for mp in sorted(glob.glob(os.path.join(out_dir, "full", "*_repatched16.png"))):
        stem = os.path.basename(mp).replace("_repatched16.png", "")
        src = read_fits(os.path.join(fits_dir, stem + ".fits"))
        same = np.array_equal(cv2.imread(mp, cv2.IMREAD_UNCHANGED), src.astype(np.uint16))
        print(f"[{stem}] 16-bit repatch == original FITS data: {same}")
        ok &= same
    return ok


def write_dataset_yaml(out_dir):
    with open(os.path.join(out_dir, "dataset.yaml"), "w") as f:
        f.write("path: .\ntrain: images\nval: images\nnames:\n  0: star\n  1: streak\n")


# ------------------------------------------------------------------- inspect
def inspect(path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    raw = read_fits(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    bg, sig = background_and_noise(raw, CONFIG["mesh"])
    print(f"shape={raw.shape} dtype={raw.dtype} min={raw.min()} max={raw.max()} "
          f"median={np.median(raw):.1f} sat_frac={(raw >= 65535).mean():.2e}")
    print(f"background mesh range {bg.min():.1f}..{bg.max():.1f}; noise sigma {np.median(sig):.2f} "
          f"(range {sig.min():.2f}..{sig.max():.2f})")
    img8 = stretch8(raw, bg, sig)
    cv2.imwrite(os.path.join(out_dir, stem + "_tile_r00_c00.png"), img8[:1024, :1024])
    small = cv2.resize(img8, (raw.shape[1] // 6, raw.shape[0] // 6), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(out_dir, stem + "_overview.png"), small)
    print("wrote preview crop + overview to", out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("inspect"); a.add_argument("fits"); a.add_argument("--out", default="inspect_out")
    r = sp.add_parser("run"); r.add_argument("fits_dir"); r.add_argument("out_dir")
    r.add_argument("--drop-empty", action="store_true", help="skip tiles with no objects")
    r.add_argument("--save-labels", action="store_true", help="also save the full-frame label map (.npy, 260 MB/frame)")
    r.add_argument("--verify", action="store_true", help="after the run, re-stitch tiles and check they equal the FITS data")
    r.add_argument("--force", action="store_true", help="redo frames that were already processed")
    for k in ("snr_high", "snr_low", "smooth_sigma", "hot_neigh_snr"):
        r.add_argument("--" + k.replace("_", "-"), type=float)
    p = sp.add_parser("repatch"); p.add_argument("out_dir"); p.add_argument("--fits-dir")
    args = ap.parse_args()

    if args.cmd == "inspect":
        inspect(args.fits, args.out)
    elif args.cmd == "run":
        cfg = dict(CONFIG)
        for k in ("snr_high", "snr_low", "smooth_sigma", "hot_neigh_snr"):
            if getattr(args, k) is not None:
                cfg[k] = getattr(args, k)
        files = sorted(glob.glob(os.path.join(args.fits_dir, "*.fit*")))
        if not files:
            sys.exit("no FITS files found")
        allobj = {}
        for fpath in files:
            allobj[os.path.basename(fpath)] = process_image(fpath, args.out_dir, cfg, not args.drop_empty, args.save_labels, args.force)
        write_dataset_yaml(args.out_dir)
        if args.verify:
            repatch(args.out_dir)
            verify_against_fits(args.fits_dir, args.out_dir)
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump({k: dict(stars=sum(o["cls"] == "star" for o in v),
                               streaks=sum(o["cls"] == "streak" for o in v)) for k, v in allobj.items()}, f, indent=1)
    elif args.cmd == "repatch":
        repatch(args.out_dir)
        if args.fits_dir:
            verify_against_fits(args.fits_dir, args.out_dir)


if __name__ == "__main__":
    main()
