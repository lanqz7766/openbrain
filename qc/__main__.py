# SPDX-License-Identifier: MIT
"""``python -m qc`` — single-case Q1..Q7 driver.

Given a directory layout

 case_dir/<case_id>/image.nii.gz
 case_dir/<case_id>/brain_mask.nii.gz
 m1_dir/<case_id>.nii.gz
 m2_dir/<case_id>.nii.gz # optional (for Q5/Q6/Q7)
 m3_dir/<case_id>.nii.gz # optional (for Q5/Q6/Q7)

compute Q1..Q7 for each case and emit a wide CSV.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

def _parse_args(argv: list[str] | None) -> argparse.Namespace:
 ap = argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--case-dir", type=Path, required=True,
 help="directory of <case_id>/ subdirectories with image.nii.gz + brain_mask.nii.gz")
 ap.add_argument("--m1-dir", type=Path, required=True)
 ap.add_argument("--m2-dir", type=Path, default=None)
 ap.add_argument("--m3-dir", type=Path, default=None)
 ap.add_argument("--include-ids", type=Path, default=None,
 help="path to a label-ID exclusion file (one ID per line; bg=0 always excluded)")
 ap.add_argument("--output", type=Path, required=True,
 help="output CSV path")
 return ap.parse_args(argv)

def _per_case_paths(case_dir: Path, m1_dir: Path,
 m2_dir: Path | None, m3_dir: Path | None,
 case_id: str) -> dict[str, Path]:
 return dict(
 image=case_dir / case_id / "image.nii.gz",
 mask=case_dir / case_id / "brain_mask.nii.gz",
 m1=m1_dir / f"{case_id}.nii.gz",
 m2=(m2_dir / f"{case_id}.nii.gz") if m2_dir else None,
 m3=(m3_dir / f"{case_id}.nii.gz") if m3_dir else None,
 )

def main(argv: list[str] | None = None) -> int:
 args = _parse_args(argv)

 # Lazy imports — keep `python -m qc --help` fast.
 import pandas as pd
 from qc._io import (
 included_label_ids,
 load_nifti_float,
 load_nifti_int,
 )
 from qc import (
 q1_brain_leakage as Q1,
 q3_distant_components as Q3,
 q4_intensity_cv as Q4,
 q5_committee_dice as Q5,
 q6_voxel_entropy as Q6,
 q7_volume_cv as Q7,
 )

 include = included_label_ids(args.include_ids)
 case_ids = sorted(p.name for p in args.case_dir.iterdir() if p.is_dir())
 rows: list[dict] = []
 for cid in case_ids:
 paths = _per_case_paths(
 args.case_dir, args.m1_dir, args.m2_dir, args.m3_dir, cid
 )
 if not paths["image"].is_file() or not paths["m1"].is_file():
 print(f"[skip] {cid}: missing image or M1 label", file=sys.stderr)
 continue
 image = load_nifti_float(paths["image"])
 m1 = load_nifti_int(paths["m1"])
 row = {"case_id": cid}
 # Q1 needs brainmask
 if paths["mask"].is_file():
 mask = load_nifti_int(paths["mask"])
 try:
 row["Q1"] = Q1.compute(m1, mask)
 except ValueError:
 row["Q1"] = float("nan")
 else:
 row["Q1"] = float("nan")
 # Q2 is computed cohort-wide; left NaN here, see core7.q2_volume_outlier
 row["Q2"] = float("nan")
 # Q3, Q4 only need M1 (+ image for Q4)
 row["Q3"] = Q3.compute(m1, include)
 row["Q4"] = Q4.compute(image, m1, include)
 # Q5/Q6/Q7 need M2 and M3
 row["Q5"] = float("nan")
 row["Q6"] = float("nan")
 row["Q7"] = float("nan")
 if paths["m2"] and paths["m3"] and paths["m2"].is_file() and paths["m3"].is_file():
 m2 = load_nifti_int(paths["m2"])
 m3 = load_nifti_int(paths["m3"])
 row["Q5"] = Q5.compute(m1, m2, m3, include)
 row["Q6"] = Q6.compute(m1, m2, m3)
 row["Q7"] = Q7.compute(m1, m2, m3, include)
 rows.append(row)
 print(
 f"[ok] {cid}: "
 + " ".join(f"{k}={v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"{k}=NaN"
 for k, v in row.items() if k != "case_id")
 )

 args.output.parent.mkdir(parents=True, exist_ok=True)
 pd.DataFrame(rows).to_csv(args.output, index=False)
 print(f"wrote {args.output} ({len(rows)} cases)")
 return 0

if __name__ == "__main__":
 sys.exit(main())
