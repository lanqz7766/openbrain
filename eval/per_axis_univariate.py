# SPDX-License-Identifier: MIT
"""Per-axis univariate analysis: is each Q_k alone discriminative?

For each axis Q_k and several strictness cutoffs in {0.50, 0.60, 0.70,
0.80, 0.90, 0.95}, split cases into passed (r_k < cutoff) and flagged
(r_k >= cutoff) and report:

- n_passed, n_flagged, pct_flagged
- mean macro_dice / cortical_dice / subcortical_dice / small_struct_dice
  per group
- delta (passed - flagged) with 5000-boot 95% CI
- Mann-Whitney U two-sided p-value

Faithful port of ``paper/scripts/analysis/per_qk_univariate.py``; the
hardcoded cohort list and CSV name pattern are now CLI-configurable.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

CUTOFFS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
R_COLS = {
    "Q1": "r1", "Q2": "r2", "Q3": "r3", "Q4": "r4",
    "Q5": "r5", "Q6": "r6", "Q7": "r7",
}
OUTCOMES = ["macro_dice", "cortical_dice", "subcortical_dice", "small_struct_dice"]

def det_seed(*parts) -> int:
    return int.from_bytes(
        hashlib.sha256("/".join(map(str, parts)).encode()).digest()[:4], "big"
    )

def boot_delta(x_pass: np.ndarray, x_flag: np.ndarray, n_boot: int,
               rng: np.random.Generator) -> tuple[float, float, float, int, int]:
    xp = np.asarray(x_pass, dtype=np.float64)
    xp = xp[~np.isnan(xp)]
    xf = np.asarray(x_flag, dtype=np.float64)
    xf = xf[~np.isnan(xf)]
    if xp.size < 2 or xf.size < 2:
        return float("nan"), float("nan"), float("nan"), xp.size, xf.size
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = (
            xp[rng.integers(0, xp.size, size=xp.size)].mean()
            - xf[rng.integers(0, xf.size, size=xf.size)].mean()
        )
    return (
        float(xp.mean() - xf.mean()),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
        xp.size, xf.size,
    )

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--risk-dir", type=Path, required=True,
                    help="directory with <cohort>_risk.csv files")
    ap.add_argument("--cohorts", nargs="+", default=["IXI", "BrainLat", "Mindboggle-101"])
    ap.add_argument("--cutoffs", type=float, nargs="+", default=CUTOFFS)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in args.cohorts:
        risk_path = args.risk_dir / f"{ds}_risk.csv"
        if not risk_path.is_file():
            print(f"[skip] {ds}: missing {risk_path}")
            continue
        df = pd.read_csv(risk_path)
        for q_name, r_col in R_COLS.items():
            if r_col not in df.columns:
                continue
            if df[r_col].isna().any():
                print(f"[warn] {ds} {r_col} has NaN; skipping axis")
                continue
            for cutoff in args.cutoffs:
                passed_mask = df[r_col] < cutoff
                passed = df[passed_mask]
                flagged = df[~passed_mask]
                row = {
                    "dataset": ds, "q_axis": q_name, "cutoff": cutoff,
                    "n_passed": int(passed_mask.sum()),
                    "n_flagged": int((~passed_mask).sum()),
                    "pct_flagged": float((~passed_mask).mean()),
                }
                for outcome in OUTCOMES:
                    if outcome not in df.columns:
                        row[f"{outcome}_passed_mean"] = float("nan")
                        row[f"{outcome}_flagged_mean"] = float("nan")
                        row[f"{outcome}_delta"] = float("nan")
                        row[f"{outcome}_ci_lo"] = float("nan")
                        row[f"{outcome}_ci_hi"] = float("nan")
                        row[f"{outcome}_mwu_p"] = float("nan")
                        continue
                    rng = np.random.default_rng(det_seed(ds, q_name, cutoff, outcome))
                    d, lo, hi, _np, _nf = boot_delta(
                        passed[outcome].values, flagged[outcome].values,
                        args.n_boot, rng,
                    )
                    row[f"{outcome}_passed_mean"] = (
                        float(passed[outcome].mean()) if len(passed) else float("nan")
                    )
                    row[f"{outcome}_flagged_mean"] = (
                        float(flagged[outcome].mean()) if len(flagged) else float("nan")
                    )
                    row[f"{outcome}_delta"] = d
                    row[f"{outcome}_ci_lo"] = lo
                    row[f"{outcome}_ci_hi"] = hi
                    xp_ = passed[outcome].dropna().values
                    xf_ = flagged[outcome].dropna().values
                    if xp_.size >= 2 and xf_.size >= 2:
                        try:
                            _u, pval = mannwhitneyu(xp_, xf_, alternative="two-sided")
                            row[f"{outcome}_mwu_p"] = float(pval)
                        except Exception:
                            row[f"{outcome}_mwu_p"] = float("nan")
                    else:
                        row[f"{outcome}_mwu_p"] = float("nan")
                rows.append(row)

    out = pd.DataFrame(rows)
    out_csv = args.output_dir / "per_axis_univariate.csv"
    out.to_csv(out_csv, index=False)
    print(f"[ok] wrote {out_csv} rows={len(out)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
