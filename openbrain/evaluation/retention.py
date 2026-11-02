# SPDX-License-Identifier: MIT
"""Retention sweep — continuous risk scoring + top-K% retention tradeoff.

For each (cohort, risk_def, retention_K%):

    1. sort cases ascending by combined risk (safest first)
    2. retain top-K% lowest-risk subset
    3. compute macro-Dice (mean per-case macro Dice over the retained set)
    4. compare to the baseline (no retention).

Risk definitions are configurable on the CLI; the defaults reproduce the
panels in Figure 2 / Appendix C of the paper.

Inputs (per evaluation cohort):
    - ``<analysis-root>/risk/<cohort>_risk.csv`` with at least the
      columns ``case_id`` and the per-axis percentile risks
      ``r1, r2, ..., r7`` (and optionally ``r8_25pctile, r9, r10, r11, r12, r6p``).
    - ``<analysis-root>/dice/<cohort>_macro_dice.csv`` long format with
      columns ``case_id, roi_id, dice, small_structure_flag``.

This script *intentionally avoids the multiplicative collapse* of the
AND-gate T1 rule by using a continuous mean-of-r combiner.

the
hardcoded paths have been replaced with CLI flags.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RETENTIONS = [90, 80, 70, 60, 50, 40, 30, 25, 20, 15, 10]

# Per-axis risk-column names. Older internal CSVs used ``r8_25pctile`` for
# the committee-Dice axis; we keep that alias for compatibility with the
# paper's intermediate manifests.
R_ALL = {f"q{i}": f"r{i}" for i in range(1, 8)}
R_ALL["Q5"] = "r8_25pctile"

DEFAULT_RISK_DEFS: dict[str, list[str]] = {
    **{f"{qk}_only": [rcol] for qk, rcol in R_ALL.items()},
    "mean_r1_to_r7": [R_ALL[f"q{i}"] for i in range(1, 8)],
    "mean_r1_to_r7_plus_r8": [R_ALL[f"q{i}"] for i in range(1, 8)] + ["r8_25pctile"],
    "mean_r2_r4_r5": ["r2", "r4", "r5"],
    "mean_r2_r4_r5_r8": ["r2", "r4", "r5", "r8_25pctile"],
}

def case_dice(roi_long: pd.DataFrame, included_ids: set[int], small_only: bool) -> pd.Series:
    sub = roi_long[roi_long["roi_id"].isin(included_ids)]
    if small_only and "small_structure_flag" in sub.columns:
        sub = sub[sub["small_structure_flag"].astype(bool)]
    sub = sub.dropna(subset=["dice"])
    return sub.groupby("case_id")["dice"].mean()

def boot_mean(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if x.size < 2:
        return float("nan"), float("nan"), float("nan")
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = x[rng.integers(0, x.size, size=x.size)].mean()
    return float(x.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def det_seed(*parts) -> int:
    return int.from_bytes(
        hashlib.sha256("/".join(map(str, parts)).encode()).digest()[:4], "big"
    )

def _load_excluded_ids(path: Path | None) -> set[int]:
    excl = {0}
    if path is None:
        return excl
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tok = s.split()[0].split(",")[0].strip()
        if tok.isdigit():
            excl.add(int(tok))
    return excl

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--risk-dir", type=Path, required=True,
                    help="directory with <cohort>_risk.csv files")
    ap.add_argument("--dice-dir", type=Path, required=True,
                    help="directory with <cohort>_macro_dice.csv files (long format)")
    ap.add_argument("--cohorts", nargs="+", default=["IXI", "BrainLat", "Mindboggle-101"])
    ap.add_argument("--exclude-ids", type=Path, default=None)
    ap.add_argument("--retentions", type=int, nargs="+", default=DEFAULT_RETENTIONS)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    excl = _load_excluded_ids(args.exclude_ids)
    included = {L for L in range(1, 111) if L not in excl}
    print(f"[cfg] included={len(included)} foreground labels")

    rows = []
    for ds in args.cohorts:
        risk_path = args.risk_dir / f"{ds}_risk.csv"
        dice_path = args.dice_dir / f"{ds}_macro_dice.csv"
        if not risk_path.is_file() or not dice_path.is_file():
            print(f"[skip] {ds}: missing inputs ({risk_path}, {dice_path})")
            continue
        risk = pd.read_csv(risk_path)
        roi_long = pd.read_csv(dice_path)
        dice_macro = case_dice(roi_long, included, small_only=False)
        dice_small = case_dice(roi_long, included, small_only=True)
        n_total = len(risk)

        all_cases = set(risk["case_id"])
        m = boot_mean(
            dice_macro[dice_macro.index.isin(all_cases)].values,
            args.n_boot,
            np.random.default_rng(det_seed("base", ds, "macro")),
        )
        s = boot_mean(
            dice_small[dice_small.index.isin(all_cases)].values,
            args.n_boot,
            np.random.default_rng(det_seed("base", ds, "small")),
        )
        base_macro, base_small = m[0], s[0]
        rows.append(dict(dataset=ds, risk_def="baseline", retention_pct=100,
                         n_retained=n_total, macro_mean=m[0], macro_ci_lo=m[1],
                         macro_ci_hi=m[2], delta_macro=0.0,
                         small_mean=s[0], small_ci_lo=s[1], small_ci_hi=s[2],
                         delta_small=0.0))

        for risk_name, r_cols in DEFAULT_RISK_DEFS.items():
            if any(c not in risk.columns for c in r_cols):
                continue
            if risk[r_cols].isna().any().any():
                # skip the risk_def for this cohort but don't abort the run
                print(f"[warn] {ds} {risk_name}: NaN in {r_cols}; skipping")
                continue
            combined = risk[r_cols].mean(axis=1)
            order = np.argsort(combined.values)
            sorted_cases = risk["case_id"].values[order]
            for K in args.retentions:
                k = int(round(n_total * K / 100.0))
                if k <= 0:
                    continue
                retained = set(sorted_cases[:k])
                arr_m = dice_macro[dice_macro.index.isin(retained)].values
                arr_s = dice_small[dice_small.index.isin(retained)].values
                m = boot_mean(arr_m, args.n_boot,
                              np.random.default_rng(det_seed(risk_name, ds, K, "macro")))
                s = boot_mean(arr_s, args.n_boot,
                              np.random.default_rng(det_seed(risk_name, ds, K, "small")))
                rows.append(dict(
                    dataset=ds, risk_def=risk_name, retention_pct=K, n_retained=k,
                    macro_mean=m[0], macro_ci_lo=m[1], macro_ci_hi=m[2],
                    delta_macro=(m[0] - base_macro) if not np.isnan(m[0]) else float("nan"),
                    small_mean=s[0], small_ci_lo=s[1], small_ci_hi=s[2],
                    delta_small=(s[0] - base_small) if not np.isnan(s[0]) else float("nan"),
                ))
        print(f"{ds}: done ({n_total} cases)")

    out = pd.DataFrame(rows)
    out_csv = args.output_dir / "retention.csv"
    out.to_csv(out_csv, index=False)
    print(f"[ok] saved {out_csv} total_rows={len(out)}")

    pd.set_option("display.width", 240)
    for ds in args.cohorts:
        sub = out[out["dataset"] == ds]
        if sub.empty:
            continue
        piv_m = sub.pivot_table(index="risk_def", columns="retention_pct",
                                values="delta_macro", aggfunc="first")
        piv_s = sub.pivot_table(index="risk_def", columns="retention_pct",
                                values="delta_small", aggfunc="first")
        cols = sorted(piv_m.columns, reverse=True)
        piv_m = piv_m[cols]
        piv_s = piv_s[cols]
        piv_m.to_csv(args.output_dir / f"{ds}_delta_macro_pivot.csv")
        piv_s.to_csv(args.output_dir / f"{ds}_delta_small_pivot.csv")
        print(f"\n=== {ds} delta_macro ===")
        print(piv_m.round(4).to_string())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
