# SPDX-License-Identifier: MIT
"""Q6 — Voxel-level committee entropy.

Definition (paper Appendix H.6): at each voxel three committee models cast
one label each. Define the per-voxel Shannon entropy of the 3-vote
distribution:

    all 3 agree -> H = 0
    exactly 2 agree -> H = -(2/3)ln(2/3) - (1/3)ln(1/3) ≈ 0.6365
    all 3 differ -> H = ln 3 ≈ 1.0986

Q6 is the mean entropy across foreground voxels (i.e. voxels where at
least one of the three models predicts a non-background label). Lower
is better.

Faithful port of ``q11_committee_entropy.py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

NAME = "Q6"
DIRECTION = "lower_is_better"
DESCRIPTION = "mean committee-vote entropy across foreground voxels"

H_TWO = 0.6365141682948128
H_THREE = 1.0986122886681098

def compute(
    label_m1: np.ndarray,
    label_m2: np.ndarray,
    label_m3: np.ndarray,
    include_ids=None, # accepted for signature symmetry; not used by Q6
) -> float:
    same_ms_sw = label_m1 == label_m2
    same_ms_ur = label_m1 == label_m3
    same_sw_ur = label_m2 == label_m3
    all_agree = same_ms_sw & same_ms_ur
    any_two_agree = same_ms_sw | same_ms_ur | same_sw_ur
    two_agree = any_two_agree & ~all_agree
    all_diff = ~any_two_agree

    H = np.zeros_like(label_m1, dtype=np.float32)
    H[two_agree] = H_TWO
    H[all_diff] = H_THREE

    fg = (label_m1 > 0) | (label_m2 > 0) | (label_m3 > 0)
    if fg.sum() == 0:
        return float("nan")
    return float(H[fg].mean())

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=DESCRIPTION)
    ap.add_argument("--m1", type=Path, required=True)
    ap.add_argument("--m2", type=Path, required=True)
    ap.add_argument("--m3", type=Path, required=True)
    args = ap.parse_args(argv)

    from openbrain.qc.nifti import load_nifti_int

    m1 = load_nifti_int(args.m1)
    m2 = load_nifti_int(args.m2)
    m3 = load_nifti_int(args.m3)
    print(f"Q6={compute(m1, m2, m3):.6f}")

if __name__ == "__main__":
    main()
