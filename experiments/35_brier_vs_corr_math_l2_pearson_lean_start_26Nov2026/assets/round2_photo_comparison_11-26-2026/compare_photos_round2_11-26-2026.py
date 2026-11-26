#!/usr/bin/env python3
"""Align each 13:01 photograph onto its 12:03 counterpart and mark which ink is in only one photo.

For each pair: both photographs are shrunk to half size, SIFT keypoints are matched (ratio test 0.75),
a homography is fitted with RANSAC (reprojection threshold 6 px), the new photograph is warped onto the old
one, and 'ink' is any pixel at least 38 grey levels darker than its local background (Gaussian blur, sigma 25).
Ink present in only one photograph (after dilating the other by a 9 by 9 window to tolerate small
misalignment) is drawn red (old photograph only) or blue (new photograph only); ink in both is grey.
The ruled lines, the paper edge, the desk and the pens also show up in the diff; the page curl and the
different camera position make them misalign, so they are not writing changes.

Run from the assets folder:  python3 -I round2_photo_comparison_11-26-2026/compare_photos_round2_11-26-2026.py
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.dirname(HERE)
PAIRS = {
    "page1": ("derivation_page_1_ruled_notebook_11-26-2026.png", "derivation_page_1_extended_ruled_notebook_round2_11-26-2026.png"),
    "page2": ("derivation_page_2_loose_leaf_11-26-2026.png", "derivation_page_2_loose_leaf_rephotographed_round2_11-26-2026.png"),
}


def load(p, s=0.5):
    im = cv2.imread(p)
    return cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def ink(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(g, (0, 0), 25)
    return (cv2.subtract(bg, g) > 38).astype(np.uint8) * 255


def align(name, old_p, new_p):
    o, n = load(old_p), load(new_p)
    go, gn = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY), cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(6000)
    ko, do = sift.detectAndCompute(go, None)
    kn, dn = sift.detectAndCompute(gn, None)
    m = cv2.BFMatcher().knnMatch(dn, do, k=2)
    good = [a for a, b in m if a.distance < 0.75 * b.distance]
    src = np.float32([kn[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    dst = np.float32([ko[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    w = cv2.warpPerspective(n, H, (o.shape[1], o.shape[0]))
    io, iw = ink(o), ink(w)
    k = np.ones((9, 9), np.uint8)
    only_new = cv2.bitwise_and(iw, cv2.bitwise_not(cv2.dilate(io, k)))
    only_old = cv2.bitwise_and(io, cv2.bitwise_not(cv2.dilate(iw, k)))
    vis = np.full(o.shape, 255, np.uint8)
    vis[io > 0] = (190, 190, 190)
    vis[only_old > 0] = (0, 0, 255)
    vis[only_new > 0] = (255, 0, 0)
    out = os.path.join(HERE, f"{name}_old_vs_new_ink_diff.jpg")
    cv2.imwrite(out, vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{name}: matches {len(good)}, RANSAC inliers {int(mask.sum())}; ink px old {int((io > 0).sum())}, "
          f"new {int((iw > 0).sum())}, only in new {int((only_new > 0).sum())}, only in old {int((only_old > 0).sum())}")


if __name__ == "__main__":
    for name, (old, new) in PAIRS.items():
        align(name, os.path.join(ASSETS, old), os.path.join(ASSETS, new))
