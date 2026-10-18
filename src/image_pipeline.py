#!/usr/bin/env python3
"""
Reference implementation of the image-processing workflow (Table 2) with the
explicit quality-control rules stated in the revised Methods.

Use it to (1) re-verify Table 3 from the raw photomicrographs, (2) process the
replicated image set (specimen x field) and (3) run the image-processing
parameter sensitivity analysis requested by the reviewer.

Input layout for replicated data:
    images/<Species_name>/<voucher_id>/<field_id>.tif
Usage:
    python image_pipeline.py images/ out/            # descriptors per field
    python image_pipeline.py images/ out/ --sweep    # + parameter sensitivity
Requires: Python >= 3.11, numpy, pandas, opencv-python 4.10, scikit-image >= 0.22
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from skimage.morphology import skeletonize

DEFAULT = dict(blur=5, block=31, C=5, morph=3, canny_lo=50, canny_hi=150, min_area=50)


def descriptors(path: Path, p: dict = DEFAULT) -> dict:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(path)
    # 1. Grayscale: Y = 0.299 R + 0.587 G + 0.114 B (OpenCV ITU-R BT.601)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # 2. Contrast normalization: linear min-max stretch to 0-255
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    # 3. Gaussian smoothing (sigma derived from kernel size by OpenCV)
    g = cv2.GaussianBlur(g, (p["blur"], p["blur"]), 0)
    # 4-5. Adaptive Gaussian threshold; tissue (cell walls, darker) = foreground
    A = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, p["block"], p["C"])
    # 6. Morphological opening then closing, square kernel
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (p["morph"], p["morph"]))
    A = cv2.morphologyEx(cv2.morphologyEx(A, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    # QC: remove connected objects smaller than min_area pixels
    n, lab, stats, _ = cv2.connectedComponentsWithStats(A, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= p["min_area"]
    A = np.where(keep[lab], 255, 0).astype(np.uint8)
    fg = A > 0
    N = A.size
    PD = fg.sum() / N
    # 7. External contours; QC: exclude zero-area/zero-perimeter contours and
    #    contours touching the image border
    cs, _ = cv2.findContours(A, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    h, w = A.shape
    iso = []
    for c in cs:
        x, y, cw, ch = cv2.boundingRect(c)
        if x == 0 or y == 0 or x + cw >= w or y + ch >= h:
            continue
        a, per = cv2.contourArea(c), cv2.arcLength(c, True)
        if a > 0 and per > 0:
            iso.append(4 * np.pi * a / per ** 2)
    RG = float(np.mean(iso)) if iso else np.nan
    # 8. Continuity: largest 8-connected component / total foreground
    n, lab, stats, _ = cv2.connectedComponentsWithStats(A, connectivity=8)
    CO = stats[1:, cv2.CC_STAT_AREA].max() / fg.sum() if n > 1 else np.nan
    # 9. Skeleton (scikit-image, Zhang-Suen-type thinning); branch point =
    #    skeleton pixel with >= 3 skeleton neighbours (8-neighbourhood)
    sk = skeletonize(fg)
    nb = convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - sk
    LI = ((nb >= 3) & sk).sum() / sk.sum() if sk.sum() else np.nan
    # 10. Canny edges (aperture 3, L1 gradient)
    E = cv2.Canny(g, p["canny_lo"], p["canny_hi"])
    ED = (E > 0).sum() / N
    return dict(PD=PD, OSF=1 - PD, RG=RG, CO=CO, LI=LI, ED=ED,
                n_contours_used=len(iso), width=w, height=h)


def collect(root: Path, p=DEFAULT) -> pd.DataFrame:
    rows = []
    for f in sorted(root.rglob("*")):
        if f.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
            continue
        rel = f.relative_to(root).parts
        sp, spec = (rel + ("NA", "NA"))[:2] if len(rel) >= 3 else (rel[0], "single")
        rows.append({"species": sp.replace("_", " "), "specimen": spec,
                     "field": f.stem, **descriptors(f, p)})
    return pd.DataFrame(rows)


def specimen_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Specimen is the biological unit: average fields within specimen first,
    then report species mean, SD and range across specimens, and the
    between- vs within-specimen variance components."""
    cols = ["PD", "RG", "CO", "LI", "ED"]
    spec = df.groupby(["species", "specimen"])[cols].mean()
    out = spec.groupby("species").agg(["mean", "std", "min", "max", "count"])
    vc = []
    for (sp), g in df.groupby("species"):
        for c in cols:
            within = g.groupby("specimen")[c].var(ddof=1).mean()
            between = g.groupby("specimen")[c].mean().var(ddof=1)
            vc.append({"species": sp, "descriptor": c,
                       "var_between_specimen": between, "var_within_specimen": within})
    return out, pd.DataFrame(vc)


def sweep(root: Path) -> pd.DataFrame:
    grid = dict(blur=[3, 5, 7], block=[21, 31, 41], C=[3, 5, 7],
                canny_lo=[30, 50, 70], canny_hi=[120, 150, 180])
    rows = []
    for vals in itertools.product(*grid.values()):
        p = dict(DEFAULT, **dict(zip(grid, vals)))
        d = collect(root, p).groupby("species")[["PD", "RG", "CO", "ED"]].mean()
        rows.append({**dict(zip(grid, vals)), **{f"{s}|{c}": d.at[s, c]
                     for s in d.index for c in d.columns}})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    df = collect(a.images)
    df.to_csv(a.out / "field_level_descriptors.csv", index=False)
    summ, vc = specimen_summary(df)
    summ.to_csv(a.out / "species_summary_specimen_unit.csv")
    vc.to_csv(a.out / "variance_components.csv", index=False)
    if a.sweep:
        sweep(a.images).to_csv(a.out / "processing_sensitivity.csv", index=False)
    print(f"Processed {len(df)} images from {df['species'].nunique()} species.")
