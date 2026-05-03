# SPDX-License-Identifier: MIT
"""Q5 — Committee Dice (3-model pairwise, 25th percentile).

Definition (paper Appendix H.5): for the three committee models
M1 (ModernSurfer), M2 (Swin UNETR), M3 (UNETR), compute per-ROI Dice on
each of the three model pairs (M1, M2), (M1, M3), (M2, M3). For each
ROI L with at least ``MIN_VOXEL_FLOOR`` voxels in *both* models of a
pair, the pair contributes one Dice value. Q5 is the 25th percentile of
the resulting (3 * len(included_labels))-pool of valid (ROI, pair)
Dice values. Lower is better (high disagreement -> low Q5 percentile).

Faithful re-implementation distilled from ``the QC committee module`` (the paper's
internal script, which also exposed `mean` and `min` aggregations as
ablations). Only the canonical 25-pctile aggregation is exposed here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

NAME = "Q5"
DIRECTION = "lower_is_better"
DESCRIPTION = "25th percentile of pairwise per-ROI Dice across 3 committee models"

NUM_LABELS = 111
MIN_VOXEL_FLOOR = 50

def _pairwise_pool(
 a: np.ndarray, b: np.ndarray, include_ids, num_labels: int = NUM_LABELS
) -> list[float]:
 """Return per-ROI Dice values for one model-pair, restricted to valid ROIs.

 Uses a single 111x111 joint-label bincount (O(V)) → constant-time per-ROI
 intersection lookup; matches the fast path in the original code.
 """
 flat_a = a.ravel().astype(np.int64, copy=False)
 flat_b = b.ravel().astype(np.int64, copy=False)
 joint = flat_a * num_labels + flat_b
 C2 = np.bincount(joint, minlength=num_labels * num_labels).reshape(
 num_labels, num_labels
 )
 row_sums = C2.sum(axis=1) # |A_L|
 col_sums = C2.sum(axis=0) # |B_L|
 diag = np.diag(C2) # |A_L ∩ B_L|
 out: list[float] = []
 for L in include_ids:
 na = int(row_sums[L])
 nb = int(col_sums[L])
 if na < MIN_VOXEL_FLOOR or nb < MIN_VOXEL_FLOOR:
 continue
 denom = na + nb
 if denom <= 0:
 continue
 out.append(2.0 * float(diag[L]) / denom)
 return out

def compute(
 label_m1: np.ndarray,
 label_m2: np.ndarray,
 label_m3: np.ndarray,
 include_ids,
) -> float:
 if not (label_m1.shape == label_m2.shape == label_m3.shape):
 raise ValueError(
 f"shape mismatch M1/M2/M3: "
 f"{label_m1.shape}, {label_m2.shape}, {label_m3.shape}"
 )
 pool: list[float] = []
 pool.extend(_pairwise_pool(label_m1, label_m2, include_ids))
 pool.extend(_pairwise_pool(label_m1, label_m3, include_ids))
 pool.extend(_pairwise_pool(label_m2, label_m3, include_ids))
 if not pool:
 return float("nan")
 return float(np.percentile(pool, 25))

def main(argv: list[str] | None = None) -> None:
 ap = argparse.ArgumentParser(description=DESCRIPTION)
 ap.add_argument("--m1", type=Path, required=True, help="ModernSurfer label NIfTI")
 ap.add_argument("--m2", type=Path, required=True, help="Swin UNETR label NIfTI")
 ap.add_argument("--m3", type=Path, required=True, help="UNETR label NIfTI")
 ap.add_argument("--include-ids", type=Path, default=None)
 args = ap.parse_args(argv)

 from qc._io import load_nifti_int, included_label_ids

 m1 = load_nifti_int(args.m1)
 m2 = load_nifti_int(args.m2)
 m3 = load_nifti_int(args.m3)
 include = included_label_ids(args.include_ids)
 print(f"Q5={compute(m1, m2, m3, include):.6f}")

if __name__ == "__main__":
 main()
