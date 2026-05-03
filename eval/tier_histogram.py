# SPDX-License-Identifier: MIT
"""Stage-C tier histogram: HighConf / Standard / Flagged counts on the
35,838-case manifest.

Used by the reviewer-replication walkthrough to verify that a local
copy of the released manifest matches the headline counts in the paper.
The script reads ``openbrain_v1_0_qc_manifest.csv``, recomputes the
within-source ECDF on R, and reports the tier counts both globally and
per source.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qc.combine_R import (
 DEFAULT_K_FLAGGED_LOWER,
 DEFAULT_K_HIGHCONF,
 ecdf_percentile,
)

def main(argv: list[str] | None = None) -> int:
 ap = argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--manifest", type=Path, required=True)
 ap.add_argument("--output", type=Path, required=True)
 ap.add_argument("--k-high", type=float, default=DEFAULT_K_HIGHCONF)
 ap.add_argument("--k-flag", type=float, default=DEFAULT_K_FLAGGED_LOWER)
 args = ap.parse_args(argv)

 df = pd.read_csv(args.manifest, low_memory=False)
 if "source" not in df.columns and "source_dataset" in df.columns:
 df = df.rename(columns={"source_dataset": "source"})
 if "R" not in df.columns:
 raise SystemExit("manifest is missing required column 'R'")

 def _tier(g: pd.DataFrame) -> pd.DataFrame:
 ecdf_R = ecdf_percentile(g["R"].to_numpy())
 g = g.assign(_ecdf=ecdf_R)
 g["tier"] = "Standard"
 g.loc[g["_ecdf"] <= args.k_high, "tier"] = "HighConf"
 g.loc[g["_ecdf"] >= args.k_flag, "tier"] = "Flagged"
 return g.drop(columns=["_ecdf"])

 out = df.groupby("source", group_keys=False).apply(_tier)
 counts_global = out["tier"].value_counts().to_dict()
 counts_per_src = (
 out.groupby(["source", "tier"]).size().unstack(fill_value=0).reset_index()
 )

 args.output.parent.mkdir(parents=True, exist_ok=True)
 counts_per_src.to_csv(args.output, index=False)
 print(f"[ok] wrote {args.output}")
 print(f"[global] {counts_global}")
 return 0

if __name__ == "__main__":
 raise SystemExit(main())
