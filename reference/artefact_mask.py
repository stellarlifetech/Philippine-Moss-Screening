#!/usr/bin/env python3
"""
REFERENCE / SENSITIVITY CODE - not used for Table 3.

Automatic masking of the two optical artefacts visible in the photomicrographs:
  * the eyepiece reticle (two long, thin, straight scale lines with tick marks
    and numerals), detected with a probabilistic Hough transform and masked as a
    band around each line;
  * air bubbles and round debris (isolated, compact, near-circular dark
    outlines; circularity > 0.70, disc fill > 0.60, radius 8 px to 12% of width),
    masked as filled discs. Cell-wall networks are rejected because they are
    not isolated compact outlines. A second detector confirms circular Hough
    candidates whose rim is darker than both the inside and the outside.
    That second detector is switched off by default because it produced many
    false positives on cellular tissue. Reticle masking is reliable; bubble
    masking is CONSERVATIVE AND INCOMPLETE (large or elliptical bubbles and
    bubbles touching tissue are missed). Inspect the QC overlays, and prefer
    manual masks (for example drawn in ImageJ/Fiji) for publication-grade work.
Descriptors are then computed on the unmasked pixels only, using the reference
workflow in image_pipeline.py, so that each image can be analysed with and
without masking under one declared implementation.

Usage:  python reference/artefact_mask.py images_dir out_dir
Writes descriptors_masked_vs_unmasked.csv and a QC overlay PNG per image.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from skimage.morphology import skeletonize

P = dict(blur=5, block=31, C=5, morph=3, canny_lo=50, canny_hi=150, min_area=50)
BAND = 0.05           # reticle band half-width (covers ticks and numerals), fraction of width
RMIN = 8              # minimum bubble radius in pixels
RING_T = 6            # minimum rim darkness (grey levels) relative to inside and outside
USE_RING_DETECTOR = False  # off: produced many false positives on cellular tissue (see QC)


def _line_mask(shape, rho, theta, half):
    h, w = shape
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    L = 3 * max(h, w)
    p1 = (int(x0 - L * b), int(y0 + L * a)); p2 = (int(x0 + L * b), int(y0 - L * a))
    m = np.zeros(shape, np.uint8)
    cv2.line(m, p1, p2, 255, 2 * half)
    return m


def _ring_ok(g, x, y, r, t=RING_T):
    """True if the circle (x, y, r) is a dark rim between brighter inside and outside."""
    h, w = g.shape
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)

    def samp(rr):
        xs = np.clip((x + rr * np.cos(ang)).astype(int), 0, w - 1)
        ys = np.clip((y + rr * np.sin(ang)).astype(int), 0, h - 1)
        return g[ys, xs].astype(float)
    rim = np.minimum(samp(r), np.minimum(samp(r - 1.5), samp(r + 1.5)))
    inn, out = samp(0.6 * r), samp(r + max(4, 0.3 * r))
    return (np.median(inn - rim) > t and np.median(out - rim) > t
            and np.mean((inn - rim > t * 0.8) & (out - rim > t * 0.8)) > 0.5)


def artefact_mask(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    m = np.zeros((h, w), np.uint8)
    # reticle: thin dark lines enhanced by a black-hat filter; keep the
    # strongest line and the strongest line roughly perpendicular to it
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    _, b = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    L = cv2.HoughLines(b, 1, np.pi / 720, threshold=int(0.35 * min(h, w)))
    half = max(8, int(BAND * w))
    chosen = []
    if L is not None:
        for rho, th in L.reshape(-1, 2):
            if not chosen:
                chosen.append((rho, th))
            elif len(chosen) == 1:
                d = abs(((th - chosen[0][1]) + np.pi / 2) % np.pi - np.pi / 2)
                if d > np.radians(60):
                    chosen.append((rho, th)); break
    for rho, th in chosen:
        m |= _line_mask((h, w), rho, th, half)
    # bubbles: isolated, closed, near-circular dark rings. Cell walls form one
    # connected network and are rejected because their filled outline is not
    # a compact disc.
    bh2 = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    _, rim = cv2.threshold(bh2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rim[m > 0] = 0                                   # ignore reticle band
    cs, hier = cv2.findContours(rim, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    n_circ = 0
    if hier is not None:
        for idx, c in enumerate(cs):
            if hier[0][idx][3] != -1:                # outer contours only
                continue
            area = cv2.contourArea(c)
            (cx, cy), r = cv2.minEnclosingCircle(c)
            if not (RMIN <= r <= 0.12 * w) or area <= 0:
                continue
            per = cv2.arcLength(c, True)
            circ_ = 4 * np.pi * area / per ** 2
            fill = area / (np.pi * r * r)
            # compact, near-circular closed outline (bubble rim or round debris)
            if circ_ > 0.70 and fill > 0.60:
                cv2.circle(m, (int(cx), int(cy)), int(r * 1.10) + 3, 255, -1)
                n_circ += 1
    # second bubble detector: circular Hough candidates confirmed by a ring profile
    gb = cv2.GaussianBlur(g, (3, 3), 0)
    cand = cv2.HoughCircles(cv2.medianBlur(g, 5), cv2.HOUGH_GRADIENT, dp=1.2, minDist=15,
                            param1=100, param2=25, minRadius=10, maxRadius=int(0.12 * w))
    if USE_RING_DETECTOR and cand is not None:
        for x, y, r in cand.reshape(-1, cand.shape[-1])[:, :3]:
            if m[int(min(y, h - 1)), int(min(x, w - 1))]:
                continue
            if _ring_ok(gb, x, y, r):
                cv2.circle(m, (int(x), int(y)), int(r * 1.15) + 3, 255, -1)
                n_circ += 1
    return m > 0, len(chosen), n_circ


def descriptors(bgr, invalid=None):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    g = cv2.GaussianBlur(g, (P["blur"], P["blur"]), 0)
    A = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, P["block"], P["C"])
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (P["morph"], P["morph"]))
    A = cv2.morphologyEx(cv2.morphologyEx(A, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    valid = np.ones(A.shape, bool) if invalid is None else ~invalid
    A[~valid] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(A, connectivity=8)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= P["min_area"]
    A = np.where(keep[lab], 255, 0).astype(np.uint8)
    fg = A > 0
    nvalid = valid.sum()
    PD = fg.sum() / nvalid
    # contours fully inside valid area and not touching the image border
    cs, _ = cv2.findContours(A, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    h, w = A.shape
    inval_d = cv2.dilate((~valid).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    iso = []
    for c in cs:
        x, y, cw, ch = cv2.boundingRect(c)
        if x == 0 or y == 0 or x + cw >= w or y + ch >= h:
            continue
        pts = c[:, 0, :]
        if inval_d[pts[:, 1], pts[:, 0]].any():
            continue
        a, per = cv2.contourArea(c), cv2.arcLength(c, True)
        if a > 0 and per > 0:
            iso.append(4 * np.pi * a / per ** 2)
    n, lab, st, _ = cv2.connectedComponentsWithStats(A, connectivity=8)
    CO = st[1:, cv2.CC_STAT_AREA].max() / fg.sum() if n > 1 else np.nan
    sk = skeletonize(fg)
    nb = convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - sk
    LI = ((nb >= 3) & sk).sum() / sk.sum() if sk.sum() else np.nan
    E = cv2.Canny(g, P["canny_lo"], P["canny_hi"]) > 0
    ED = (E & valid).sum() / nvalid
    return dict(PD=PD, RG=float(np.mean(iso)) if iso else np.nan, CO=CO, LI=LI, ED=ED,
                valid_fraction=nvalid / A.size, n_contours=len(iso))


def main(src, out):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(src.glob("*.jpg")) + sorted(src.glob("*.png")) + sorted(src.glob("*.tif")):
        bgr = cv2.imread(str(f))
        inv, nl, nc = artefact_mask(bgr)
        u = descriptors(bgr)
        mk = descriptors(bgr, inv)
        rows.append({"image": f.stem, "width": bgr.shape[1], "height": bgr.shape[0],
                     "reticle_lines": nl, "bubbles": nc,
                     **{f"{k}_unmasked": v for k, v in u.items() if k != "valid_fraction"},
                     **{f"{k}_masked": v for k, v in mk.items()}})
        ov = bgr.copy()
        ov[inv] = (0.45 * ov[inv] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(str(out / f"QC_{f.stem}.png"), ov)
    df = pd.DataFrame(rows)
    df.to_csv(out / "descriptors_masked_vs_unmasked.csv", index=False)
    return df


if __name__ == "__main__":
    print(main(Path(sys.argv[1]), Path(sys.argv[2])).round(4).to_string())
