# SPDX-License-Identifier: MIT
"""Combined-risk R = mean(r_1..r_7) and tier assignment.

Each axis Q_k is converted to a within-source percentile risk

 r_k(c, D) = ECDF_D(Q_k)(Q_k(c)) if direction is lower_is_better
 = 1 - ECDF_D(Q_k)(Q_k(c)) if direction is higher_is_better

so that r_k in [0, 1] is "risk-aligned": higher = riskier. The combined
risk for a case c in source D is

 R(c, D) = mean_k r_k(c, D).

The case is then bucketed into one of three tiers using two cuts on R:

 R <= K_high -> HighConf (default K_high = 0.25)
 K_high < R < K_flag -> Standard
 R >= K_flag -> Flagged (default K_flag = 0.75)

K cuts are computed on the within-source ECDF of R (so they correspond
to per-source 25th and 75th percentiles by construction). This is the
canonical Stage-C combiner described in the paper.
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
 """Empirical CDF on ``x``; returns rank/n in [0, 1]. NaN -> NaN."""
 x = np.asarray(x, dtype=np.float64)
 out = np.full_like(x, np.nan)
 finite = ~np.isnan(x)
 if finite.sum() == 0:
 return out
 sub = x[finite]
 order = np.argsort(sub, kind="mergesort")
 ranks = np.empty_like(order)
 ranks[order] = np.arange(1, sub.size + 1)
 out[finite] = ranks / sub.size
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
 """Compute r1..r7, R, and tier per source, in-place on a copy of ``df``."""
 import pandas as pd # lazy

 out = df.copy()
 for q, direction in DIRECTIONS.items():
 out[q.replace("Q", "r")] = np.nan
 grouped = out.groupby("source", group_keys=False)

 def _process(g: pd.DataFrame) -> pd.DataFrame:
 for q, direction in DIRECTIONS.items():
 g[q.replace("Q", "r")] = per_axis_risk(g[q].to_numpy(), direction)
 return g

 out = grouped.apply(_process)
 r_cols = [q.replace("Q", "r") for q in DIRECTIONS]
 out["R"] = out[r_cols].mean(axis=1)
 # Tier cuts: per-source ECDF on R (interprets K_high / K_flag as
 # within-source percentiles).
 def _tier(g: pd.DataFrame) -> pd.DataFrame:
 ecdf_R = ecdf_percentile(g["R"].to_numpy())
 g["tier"] = "Standard"
 g.loc[ecdf_R <= k_high, "tier"] = "HighConf"
 g.loc[ecdf_R >= k_flag, "tier"] = "Flagged"
 return g

 out = out.groupby("source", group_keys=False).apply(_tier)
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
 f"sources={out['source'].nunique()} "
 f"tiers={out['tier'].value_counts().to_dict()}"
 )

if __name__ == "__main__":
 main()
