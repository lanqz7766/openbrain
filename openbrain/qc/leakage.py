# SPDX-License-Identifier: MIT
"""Q1 — Brain-mask leakage.

Definition (paper Appendix H.1):

    Q1 = | { v : seg(v) > 0 AND brainmask(v) == 0 } | / | { v : seg(v) > 0 } |

i.e., the fraction of foreground-segmented voxels that fall outside the
SynthStrip-derived brain mask. Lower is better. NaN if the segmentation
has no foreground voxels.

This module is a faithful re-implementation of the inline lambda used in
``the QC pipeline driver``; it is reconstructed for the public release as a
standalone, importable function rather than verbatim-copied.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

NAME = "Q1"
DIRECTION = "lower_is_better"
DESCRIPTION = "fraction of foreground segmented voxels outside the brain mask"

def compute(label_ms: np.ndarray, brainmask: np.ndarray) -> float:
    if label_ms.shape != brainmask.shape:
        raise ValueError(
            f"label and brainmask shape mismatch: {label_ms.shape} vs {brainmask.shape}"
        )
    fg = label_ms > 0
    n_fg = int(fg.sum())
    if n_fg == 0:
        return float("nan")
    leak = int((fg & (brainmask == 0)).sum())
    return float(leak / n_fg)

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=DESCRIPTION)
    ap.add_argument("--label", type=Path, required=True, help="M1 label NIfTI")
    ap.add_argument("--brainmask", type=Path, required=True, help="SynthStrip mask NIfTI")
    args = ap.parse_args(argv)

    from openbrain.qc.nifti import load_nifti_int

    lbl = load_nifti_int(args.label)
    msk = load_nifti_int(args.brainmask)
    print(f"Q1={compute(lbl, msk):.6f}")

if __name__ == "__main__":
    main()
