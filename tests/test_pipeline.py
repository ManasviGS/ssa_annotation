"""Run with:  python -m pytest tests -q     (from the project root)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import annotate as A
import validate_dataset as V

T = 1024


def _stitch(a, T):
    ny, nx = A.tile_grid(a.shape, T)
    big = np.zeros((ny * T, nx * T), a.dtype)
    for ty in range(ny):
        for tx in range(nx):
            big[ty * T:(ty + 1) * T, tx * T:(tx + 1) * T] = A.get_tile(a, ty, tx, T)
    return big[:a.shape[0], :a.shape[1]], (ny, nx)


def test_grid_for_assessment_frame():
    assert A.tile_grid((6380, 9568), T) == (7, 10)     # 70 tiles; 9568=9*1024+352, 6380=6*1024+236


def test_tiling_is_lossless_no_loss_no_duplication():
    rng = np.random.default_rng(0)
    for shape in [(6380 // 4, 9568 // 4), (1024, 1024), (1025, 2048), (300, 5000)]:
        a = rng.integers(0, 65536, shape, dtype=np.uint16)
        back, (ny, nx) = _stitch(a, T)
        assert back.shape == a.shape and np.array_equal(back, a)
        # every source pixel is covered by exactly one tile origin
        cover = np.zeros(shape, int)
        for ty in range(ny):
            for tx in range(nx):
                cover[ty * T:(ty + 1) * T, tx * T:(tx + 1) * T] += 1
        assert (cover == 1).all()


def test_label_padding_is_zero():
    lbl = np.ones((1100, 1100), np.int32)
    t = A.get_tile(lbl, 1, 1, T, "constant")
    assert t.shape == (T, T) and t[:1100 - T, :1100 - T].all() and t[1100 - T:].sum() == 0 and t[:, 1100 - T:].sum() == 0


def _small_frame(seed=1):
    rng = np.random.default_rng(seed)
    H, W = 700, 900
    img = rng.normal(500, 20, (H, W))
    yy, xx = np.mgrid[0:H, 0:W]
    for (x, y) in [(200, 150), (600, 400), (300, 600)]:                  # stars
        img += 4000 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.3 ** 2))
    t = np.linspace(-40, 40, 400)                                         # one 80 px streak
    for s in t:
        x, y = 450 + s * np.cos(0.5), 300 + s * np.sin(0.5)
        img += 300 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.2 ** 2)) * (80 / 400)
    return np.clip(img, 0, 65535).astype(np.uint16)


def test_detects_planted_star_and_streak_with_right_classes():
    raw = _small_frame()
    lbl, cls_by, objects, img8 = A.detect(raw, A.CONFIG)
    stars = [o for o in objects if o["cls"] == "star"]
    streaks = [o for o in objects if o["cls"] == "streak"]
    assert len(stars) >= 3 and len(streaks) >= 1
    for (x, y) in [(200, 150), (600, 400), (300, 600)]:
        assert any(abs(o["cx"] - x) < 3 and abs(o["cy"] - y) < 3 for o in stars)
    assert any(abs(o["cx"] - 450) < 6 and abs(o["cy"] - 300) < 6 for o in streaks)
    assert img8.dtype == np.uint8 and img8.shape == raw.shape


def test_hot_pixels_are_rejected():
    """Single-pixel spikes (250 sigma) must almost always be rejected by the neighbour test.
    The rule is statistical (a spike survives if its 8 noisy neighbours happen to sum to >= 2.5 sigma,
    ~0.6 % per spike), so test the rate over several seeds instead of one fixed noise realisation."""
    survived, total = 0, 0
    pts = [(50, 60), (200, 200), (350, 300)]
    for seed in range(4, 9):
        rng = np.random.default_rng(seed)
        img = rng.normal(500, 20, (400, 400))
        for (x, y) in pts:
            img[y, x] += 5000
        _, _, objects, _ = A.detect(np.clip(img, 0, 65535).astype(np.uint16), A.CONFIG)
        total += len(pts)
        survived += sum(any(abs(o["cx"] - x) < 2 and abs(o["cy"] - y) < 2 for o in objects) for (x, y) in pts)
    assert survived <= 1, f"{survived}/{total} hot pixels survived"


def test_exported_labels_pass_validator_and_mask_alignment():
    raw = _small_frame()
    lbl, cls_by, objects, _ = A.detect(raw, A.CONFIG)
    with tempfile.TemporaryDirectory() as d:
        tile = A.get_tile(lbl, 0, 0, T, "constant")
        lines = A.polygons_for_tile(tile, cls_by, T, A.CONFIG["min_fragment_area"])
        assert lines
        p = os.path.join(d, "t_r00_c00.txt")
        open(p, "w").write("\n".join(lines) + "\n")
        problems, counts = V.check_label_file(p, T, valid_w=900, valid_h=700)
        assert problems == [], problems
        assert counts[0] >= 3 and counts[1] >= 1
        # polygons rasterised back must overlap their source mask (pixel alignment)
        import cv2
        ras = np.zeros((T, T), np.uint8)
        for l in lines:
            v = np.array(l.split()[1:], float).reshape(-1, 2) * T
            cv2.fillPoly(ras, [np.round(v - 0.5).astype(np.int32)], 1)
        inter = ((ras > 0) & (tile > 0)).sum()
        assert inter / max((tile > 0).sum(), 1) > 0.9


def test_validator_catches_bad_labels():
    with tempfile.TemporaryDirectory() as d:
        bad = {
            "class": "2 0.1 0.1 0.2 0.1 0.2 0.2\n",
            "range": "0 0.1 0.1 1.2 0.1 0.2 0.2\n",
            "few": "0 0.1 0.1 0.2 0.1\n",
            "bowtie": "1 0.1 0.1 0.3 0.3 0.3 0.1 0.1 0.3\n",
            "pad": "0 0.8 0.1 0.95 0.1 0.95 0.2\n",
        }
        for k, v in bad.items():
            p = os.path.join(d, k + ".txt")
            open(p, "w").write(v)
            probs, _ = V.check_label_file(p, T, valid_w=700, valid_h=1024)
            assert probs, f"validator missed the '{k}' case"
        good = os.path.join(d, "good.txt")
        open(good, "w").write("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")
        assert V.check_label_file(good, T, 1024, 1024)[0] == []
