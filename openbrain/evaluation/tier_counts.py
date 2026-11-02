# SPDX-License-Identifier: MIT
"""Tier counts: HighConf / Standard / Flagged counts on the
35,838-case manifest.

Used by the reproduction guide (docs/reproduce.md) to verify that a local
copy of the released manifest matches the headline counts in the paper.
The script reads ``quality.csv``, re-derives the tiers from R with the
same quantile cuts as the release (lowest 25% HighConf, top 25% Flagged),
checks them against the ``tier`` column, and reports the counts globally and
per source.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from openbrain.qc.combine import (
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

    out = df.copy()
    ecdf_R = ecdf_percentile(out["R"].to_numpy())
    out["tier_recomputed"] = "Standard"
    out.loc[ecdf_R <= args.k_high, "tier_recomputed"] = "HighConf"
    out.loc[ecdf_R >= args.k_flag, "tier_recomputed"] = "Flagged"
    if "tier" in out.columns:
        agree = (out["tier"] == out["tier_recomputed"]).mean()
        print(f"[check] recomputed tiers agree with the tier column on {agree:.4%} of cases")
    else:
        out["tier"] = out["tier_recomputed"]
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
