# SSA star / streak annotation (Digantara AI/ML Data Annotation Intern assessment)

Scripted pipeline: FITS -> background/noise model -> SNR map -> candidate detection -> star/streak classification
-> 1024x1024 tiles -> YOLO-Ultralytics **segmentation** labels -> full-size overlays and repatched mosaics.
Classes: `0 = star (blob)`, `1 = streak`. The output is automatic candidate detection; review it before relying on it.
The written response is `report.pdf` (source `report.md`).

## Setup (Python 3.9+)
    pip install -r requirements.txt

## Run on the 10 FITS files
    # 0. (optional) look at one frame: stats, preview crop, downsampled overview
    python annotate.py inspect Datasets_Assessment/<file>.fits --out inspect_out

    # 1. annotate every FITS file in the folder (default seed SNR 6; already-finished frames are skipped, --force redoes them)
    python annotate.py run Datasets_Assessment results

    # 2. repatch tiles to full size and check the 16-bit mosaic equals the FITS data exactly
    python annotate.py repatch results --fits-dir Datasets_Assessment

    # 3. validate every YOLO label file (class ids, ranges, polygon validity, padding, pairing)
    python validate_dataset.py results

    # 3b. check the exact pixel masks written by annotate.py (0 bg / 1 star / 2 streak) + write manifest.csv; also run by package_submission.py
    python make_masks.py results

    # 4. unit tests (tiling lossless, padding, detection, hot pixels, label export, validator)
    python -m pytest tests -q

    # 5. build the submission ZIP (validates first; makes the repatched mosaics if missing)
    python package_submission.py results --report report.pdf --out digantara_submission.zip --repo-url <your GitHub URL>
    #    optional, large: --include-raw16 (lossless tiles)  --include-repatched16 (lossless mosaics)

Tuning (command line): `--snr-high 6 --snr-low 2.5 --smooth-sigma 1.0 --hot-neigh-snr 2.5`. Other thresholds are in `CONFIG` at the top of `annotate.py`.
`--drop-empty` skips tiles with no objects; `--save-labels` also stores the full-frame label map (260 MB/frame, only needed for `evaluate_synthetic.py`); `--verify` runs step 2 at the end of `run`.
A frame takes 1-7 minutes depending on the machine and how many objects it has; peak RAM is a few GB for one frame. Delete the half-written frame's files if you stop a run mid-frame.

If a frame gives odd counts (0 or thousands), find out where candidates are lost (seconds, one crop, writes nothing):

    python diagnose.py path/to/frame.fits [--x0 1000 --y0 1000 --size 2048]

## Output layout (`results/`)
    images/        1024x1024 8-bit tiles (asinh stretch)            <frame>_r<row>_c<col>.png
    masks/         exact uint8 masks 0/1/2 (written by annotate.py);  manifest.csv (make_masks.py)  one row per tile (source, row/col, pixel box, padding, paths)
    labels/        YOLO-seg txt, one per tile: `cls x1 y1 x2 y2 ...` normalised to the tile
    tiles_raw16/   lossless 16-bit tiles of the raw data
    full/          per frame: _overlay.png (stretched frame + masks), _stretched.png, _meta.json (object catalogue + config),
                   _repatched8.png / _repatched16.png (after `repatch`), .done marker
    dataset.yaml   Ultralytics dataset file;  summary.json  counts for the frames processed in the last `run`

Tiles: 9568x6380 -> 10 x 7 grid, reflect-padded to 10240x7168; repatching crops back exactly. Labels are never placed in padding.

## Reviewing / correcting
Import `images/` + `labels/` into CVAT as "Ultralytics YOLO Segmentation 1.0" (class order: star, streak). Each object's
`length`, `aspect`, `core_length`, `core_aspect`, `peak_snr` is in `full/<frame>_meta.json`: sort by low `peak_snr` and by values near the
streak thresholds to find the ambiguous ones first.
No model has been trained. To train on the exported dataset: `yolo segment train data=results/dataset.yaml model=yolo11n-seg.pt imgsz=1024`.

## Testing without the real data
    python make_synthetic.py syn/                      # 9568x6380 uint16 FITS + ground truth (add --edge for a dark region with a sharp edge)
    python annotate.py run syn out --save-labels
    python evaluate_synthetic.py out syn/synthetic_0_gt.json
Note: the real frames are low-count images with strong vignetting and masked regions; the synthetic frame is easier.
Do not commit the confidential FITS data to a public repository (`.gitignore` excludes `*.fits` and `results/`).
