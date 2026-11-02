# SPDX-License-Identifier: MIT
"""Combined-risk R = mean(r_1..r_7) and tier assignment.

Each axis Q_k is converted to a percentile risk by ranking the case against
every case in the input table (the QC pool), with mid-ranks for ties:

    r_k(c) = ECDF(Q_k)(Q_k(c))      if direction is lower_is_better
           = 1 - ECDF(Q_k)(Q_k(c))  if direction is higher_is_better

so that r_k in [0, 1] is "risk-aligned": higher = riskier. The combined
risk is R(c) = mean_k r_k(c). Tiers are the lowest-risk K_high fraction
(HighConf, default 25%), the highest-risk 1 - K_flag fraction (Flagged,
default 25%), and Standard in between, cut on the quantiles of R over the
same table.

This is how the released quality.csv was computed (over the QC pool before
release-eligibility filtering). Only the Q2 volume fences are source-specific
(see volume_outliers.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Direction map for the seven canonical axes.
DIRECTIONS = {
    "Q1": "lower_is_better",
    "Q2": "lower_is_better",
    "Q3": "lower_is_better",
    "Q4": "lower_is_better",
    "Q5": "higher_is_better", # higher Dice = better, so flip
    "Q6": "lower_is_better",
    "Q7": "lower_is_better",
}

DEFAULT_K_HIGHCONF = 0.25
DEFAULT_K_FLAGGED_LOWER = 0.75

def ecdf_percentile(x: np.ndarray) -> np.ndarray:
    """Empirical CDF on ``x`` with mid-ranks for ties; rank/n in [0, 1].
    NaN -> NaN."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full_like(x, np.nan)
    finite = ~np.isnan(x)
    if finite.sum() == 0:
        return out
    import pandas as pd  # lazy

    ranks = pd.Series(x[finite]).rank(method="average").to_numpy()
    out[finite] = ranks / finite.sum()
    return out

def per_axis_risk(values: np.ndarray, direction: str) -> np.ndarray:
    p = ecdf_percentile(values)
    if direction == "lower_is_better":
        return p
    if direction == "higher_is_better":
        return 1.0 - p
    raise ValueError(f"unknown direction: {direction!r}")

def assign_tier(
    R: np.ndarray,
    k_high: float = DEFAULT_K_HIGHCONF,
    k_flag: float = DEFAULT_K_FLAGGED_LOWER,
) -> np.ndarray:
    """Bucket a 1-D array of combined risks into HighConf/Standard/Flagged."""
    out = np.full(R.shape, "Standard", dtype=object)
    out[R <= k_high] = "HighConf"
    out[R >= k_flag] = "Flagged"
    return out

def combine(
    df, # pandas.DataFrame with columns case_id, source, Q1..Q7
    k_high: float = DEFAULT_K_HIGHCONF,
    k_flag: float = DEFAULT_K_FLAGGED_LOWER,
):
    """Compute r1..r7, R, and tier over the whole table, on a copy of ``df``."""
    out = df.copy()
    for q, direction in DIRECTIONS.items():
        out[q.replace("Q", "r")] = per_axis_risk(out[q].to_numpy(), direction)
    r_cols = [q.replace("Q", "r") for q in DIRECTIONS]
    out["R"] = out[r_cols].mean(axis=1)
    # Tier cuts: quantiles of R over the complete cases of the whole table.
    complete = out["R"].notna()
    q_high = out.loc[complete, "R"].quantile(k_high)
    q_flag = out.loc[complete, "R"].quantile(k_flag)
    out["tier"] = "Standard"
    out.loc[complete & (out["R"] <= q_high), "tier"] = "HighConf"
    out.loc[complete & (out["R"] >= q_flag), "tier"] = "Flagged"
    out.loc[~complete, "tier"] = "NA"
    return out

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Combine per-case Q1..Q7 into r1..r7, R, and tier."
    )
    ap.add_argument("--input", type=Path, required=True,
                    help="CSV with columns case_id, source, Q1..Q7")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--k-high", type=float, default=DEFAULT_K_HIGHCONF)
    ap.add_argument("--k-flag", type=float, default=DEFAULT_K_FLAGGED_LOWER)
    args = ap.parse_args(argv)

    import pandas as pd

    df = pd.read_csv(args.input, low_memory=False)
    out = combine(df, k_high=args.k_high, k_flag=args.k_flag)
    out.to_csv(args.output, index=False)
    print(
        f"wrote {args.output} n={len(out)} "
        f"sources={out['source'].nunique() if 'source' in out else 'n/a'} "
        f"tiers={out['tier'].value_counts().to_dict()}"
    )

if __name__ == "__main__":
    main()
