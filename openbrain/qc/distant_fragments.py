# SPDX-License-Identifier: MIT
"""Q3 — Distant connected components.

Definition (paper Appendix H.3): for each ROI L of size at least
``MIN_ROI_VOXELS``, find connected components under 26-connectivity.
Treat the largest CC as primary; classify any other CC whose centroid is
farther than alpha * V_primary^(1/3) from the primary centroid as
"distant". Q3 is the total voxel count of distant CCs divided by the
total foreground voxel count over the included ROIs. Lower is better.

This is a faithful port of ``the distant-components QC module`` from the paper's
``qc_tools`` package, restructured for the public release with no
behaviour change.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

NAME = "Q3"
DIRECTION = "lower_is_better"
DESCRIPTION = (
    "fraction of voxels in connected components whose centroid is more "
    "than alpha * V_primary^(1/3) from the primary CC centroid"
)

MIN_ROI_VOXELS = 50
ALPHA = 1.5

def compute(label_ms: np.ndarray, include_ids) -> float:
    from scipy import ndimage as ndi # lazy

    struct = ndi.generate_binary_structure(3, 3) # 26-connectivity

    total_distant = 0
    total_voxels = 0
    slices = ndi.find_objects(label_ms)
    for L in include_ids:
        if L - 1 >= len(slices):
            continue
        sl = slices[L - 1]
        if sl is None:
            continue
        crop = label_ms[sl]
        mask_crop = crop == L
        n = int(mask_crop.sum())
        if n < MIN_ROI_VOXELS:
            continue
        total_voxels += n
        cc, num = ndi.label(mask_crop, structure=struct)
        if num <= 1:
            continue
        sizes = np.bincount(cc.ravel())[1:]
        primary_idx = int(sizes.argmax()) + 1
        primary_size = int(sizes[primary_idx - 1])
        primary_radius = primary_size ** (1.0 / 3.0)
        thr = ALPHA * primary_radius
        centroids = ndi.center_of_mass(mask_crop, cc, range(1, num + 1))
        primary_c = np.asarray(centroids[primary_idx - 1])
        for cc_id in range(1, num + 1):
            if cc_id == primary_idx:
                continue
            cc_size = int(sizes[cc_id - 1])
            if cc_size == 0:
                continue
            c = np.asarray(centroids[cc_id - 1])
            d = float(np.sqrt(((c - primary_c) ** 2).sum()))
            if d > thr:
                total_distant += cc_size
    if total_voxels == 0:
        return float("nan")
    return float(total_distant / total_voxels)

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=DESCRIPTION)
    ap.add_argument("--label", type=Path, required=True)
    ap.add_argument("--include-ids", type=Path, default=None)
    args = ap.parse_args(argv)

    from openbrain.qc.nifti import load_nifti_int, included_label_ids

    lbl = load_nifti_int(args.label)
    include = included_label_ids(args.include_ids)
    print(f"Q3={compute(lbl, include):.6f}")

if __name__ == "__main__":
    main()
