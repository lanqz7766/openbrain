# SPDX-License-Identifier: MIT
"""Q4 — Within-ROI T1w intensity coefficient of variation.

Definition (paper Appendix H.4): for each ROI L with at least
``MIN_ROI_VOXELS`` voxels, compute the within-ROI mean and standard
deviation of the T1w intensity. The per-ROI CV is std / mean. Q4 is the
median across applicable ROIs of this CV. Lower is better.

Faithful port of ``q10_intensity_homogeneity.py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

NAME = "Q4"
DIRECTION = "lower_is_better"
DESCRIPTION = "median across ROIs of T1w intensity CV (std/mean)"

MIN_ROI_VOXELS = 50

def compute(image: np.ndarray, label_ms: np.ndarray, include_ids) -> float:
    from scipy import ndimage as ndi # lazy

    img = image.astype(np.float32)
    lbls = list(include_ids)
    means = ndi.mean(img, labels=label_ms, index=lbls)
    stds = ndi.standard_deviation(img, labels=label_ms, index=lbls)
    counts = ndi.sum(np.ones_like(img, dtype=np.int32), labels=label_ms, index=lbls)
    cvs = []
    for m, s, n in zip(means, stds, counts):
        if n < MIN_ROI_VOXELS or not np.isfinite(m) or m <= 1e-6:
            continue
        cvs.append(float(s) / float(m))
    if not cvs:
        return float("nan")
    return float(np.median(cvs))

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=DESCRIPTION)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--label", type=Path, required=True)
    ap.add_argument("--include-ids", type=Path, default=None)
    args = ap.parse_args(argv)

    from openbrain.qc.nifti import load_nifti_float, load_nifti_int, included_label_ids

    img = load_nifti_float(args.image)
    lbl = load_nifti_int(args.label)
    include = included_label_ids(args.include_ids)
    print(f"Q4={compute(img, lbl, include):.6f}")

if __name__ == "__main__":
    main()
