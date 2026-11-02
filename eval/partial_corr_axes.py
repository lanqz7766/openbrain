# SPDX-License-Identifier: MIT
"""F-table: per-axis partial Spearman correlation with macro-Dice (controlling
for source dataset).

For each axis Q_k and each evaluation cohort, compute the Spearman
correlation between Q_k and macro-Dice, conditioned on source dataset
identity. We use the standard "residualise-then-rank" formulation: regress
``rank(Q_k)`` on a one-hot of the source dataset, regress
``rank(macro_dice)`` on the same one-hot, then take the Spearman of the
two residual vectors.

This produces the F_axes_partial_corr table referenced in the paper. The
small structures variant is computed from ``small_struct_dice`` when
available.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

R_COLS = {
    "Q1": "r1", "Q2": "r2", "Q3": "r3", "Q4": "r4",
    "Q5": "r5", "Q6": "r6", "Q7": "r7",
}

def _residuals(values: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Return residuals of rank(values) minus per-source mean rank."""
    ranks = pd.Series(values).rank(method="average").to_numpy()
    df = pd.DataFrame({"rank": ranks, "source": source})
    means = df.groupby("source")["rank"].transform("mean")
    return ranks - means.to_numpy()

def partial_spearman(x: np.ndarray, y: np.ndarray, source: np.ndarray) -> float:
    rx = _residuals(x, source)
    ry = _residuals(y, source)
    if np.allclose(rx.std(), 0) or np.allclose(ry.std(), 0):
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--risk-dir", type=Path, required=True)
    ap.add_argument("--cohorts", nargs="+", default=["IXI", "BrainLat", "Mindboggle-101"])
    ap.add_argument("--source-col", default="source_dataset",
                    help="column in <cohort>_risk.csv naming the source dataset")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    rows = []
    for ds in args.cohorts:
        f = args.risk_dir / f"{ds}_risk.csv"
        if not f.is_file():
            print(f"[skip] {ds}: missing {f}")
            continue
        df = pd.read_csv(f)
        if args.source_col not in df.columns:
            print(f"[skip] {ds}: missing source column {args.source_col!r}")
            continue
        src = df[args.source_col].astype(str).to_numpy()
        for q_name, r_col in R_COLS.items():
            if r_col not in df.columns:
                continue
            for outcome in ("macro_dice", "small_struct_dice"):
                if outcome not in df.columns:
                    continue
                rho = partial_spearman(
                    df[r_col].to_numpy(),
                    df[outcome].to_numpy(),
                    src,
                )
                rows.append(dict(
                    dataset=ds, q_axis=q_name, outcome=outcome,
                    partial_spearman=rho, n=len(df),
                ))
    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[ok] wrote {args.output} rows={len(out)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
