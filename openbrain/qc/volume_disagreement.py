# SPDX-License-Identifier: MIT
"""Q7 — Committee per-ROI volume CV.

Definition (paper Appendix H.7): for each ROI L, compute the per-model
voxel counts V_M1, V_M2, V_M3. The per-ROI volume CV is std / mean,
restricted to ROIs with mean voxel count >= ``MIN_VOL``. Q7 is the mean
of these CVs across applicable ROIs. Lower is better. (Q7 captures
volumetric agreement in a position-agnostic way; Q5 captures spatial
overlap.)

Faithful port of ``q12_committee_volume_cv.py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

NAME = "Q7"
DIRECTION = "lower_is_better"
DESCRIPTION = "mean across ROIs of CV(volume) across 3 committee models"

NUM_LABELS = 111
MIN_VOL = 50

def compute(
    label_m1: np.ndarray,
    label_m2: np.ndarray,
    label_m3: np.ndarray,
    include_ids,
) -> float:
    v_m1 = np.bincount(label_m1.ravel(), minlength=NUM_LABELS)
    v_m2 = np.bincount(label_m2.ravel(), minlength=NUM_LABELS)
    v_m3 = np.bincount(label_m3.ravel(), minlength=NUM_LABELS)
    cvs: list[float] = []
    for L in include_ids:
        vols = np.array([v_m1[L], v_m2[L], v_m3[L]], dtype=np.float64)
        m = float(vols.mean())
        if m < MIN_VOL:
            continue
        cvs.append(float(vols.std()) / (m + 1e-9))
    if not cvs:
        return float("nan")
    return float(np.mean(cvs))

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=DESCRIPTION)
    ap.add_argument("--m1", type=Path, required=True)
    ap.add_argument("--m2", type=Path, required=True)
    ap.add_argument("--m3", type=Path, required=True)
    ap.add_argument("--include-ids", type=Path, default=None)
    args = ap.parse_args(argv)

    from openbrain.qc.nifti import load_nifti_int, included_label_ids

    m1 = load_nifti_int(args.m1)
    m2 = load_nifti_int(args.m2)
    m3 = load_nifti_int(args.m3)
    include = included_label_ids(args.include_ids)
    print(f"Q7={compute(m1, m2, m3, include):.6f}")

if __name__ == "__main__":
    main()
