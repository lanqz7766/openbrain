# SPDX-License-Identifier: MIT
"""Stage-A small-structure (Def-A 16-ROI) sensitivity sweep.

For each cohort and each retention K in {10, 25, 50, 75, 90, 100}:

 1. resolve the 16 Def-A FreeSurfer aseg IDs to OpenBrain internal IDs
 by name match against ``data/label_map.json``;
 2. rank cases by combined risk R = mean(r-cols) ascending;
 3. retain bottom K%, compute the small-structure macro Dice (mean
 per-case mean Dice over the 16 Def-A internal IDs, dropping NaN);
 4. baseline = all-cases mean;
 5. delta_def_a = retained_mean - baseline;
 6. 2000-boot 95% CI on retained_mean.

Faithful port of ``paper/scripts/analysis/small_struct_defA/T_A_small_struct_defA.py``;
hardcoded paths now CLI flags.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RETENTIONS = [90, 75, 50, 25, 10]

# Def-A FreeSurfer aseg IDs and their FreeSurfer-style names.
DEF_A_FS = {
 10: "Left-Thalamus-Proper",
 49: "Right-Thalamus-Proper",
 11: "Left-Caudate",
 50: "Right-Caudate",
 12: "Left-Putamen",
 51: "Right-Putamen",
 13: "Left-Pallidum",
 52: "Right-Pallidum",
 17: "Left-Hippocampus",
 53: "Right-Hippocampus",
 18: "Left-Amygdala",
 54: "Right-Amygdala",
 26: "Left-Accumbens-area",
 58: "Right-Accumbens-area",
 28: "Left-VentralDC",
 60: "Right-VentralDC",
}

# Some OpenBrain label maps use "Left-Thalamus" rather than
# "Left-Thalamus-Proper". The alias table covers the known divergences.
NAME_ALIASES = {
 "Left-Thalamus-Proper": ["Left-Thalamus-Proper", "Left-Thalamus"],
 "Right-Thalamus-Proper": ["Right-Thalamus-Proper", "Right-Thalamus"],
}

DEFAULT_R_COLS = ["r1", "r2", "r3", "r4", "r5", "r6", "r7"]

def resolve_def_a(label_map_path: Path) -> tuple[list[tuple[int, str, int, str]], list[tuple[int, str]]]:
 """Resolve Def-A FreeSurfer IDs to OpenBrain internal IDs by name."""
 j = json.loads(Path(label_map_path).read_text())
 name_to_id = {k: int(v) for k, v in j["labels"].items()}
 out, missing = [], []
 for fs_id, fs_name in DEF_A_FS.items():
 candidates = NAME_ALIASES.get(fs_name, [fs_name])
 hit = None
 for n in candidates:
 if n in name_to_id:
 hit = (n, name_to_id[n])
 break
 if hit is None:
 missing.append((fs_id, fs_name))
 continue
 out.append((fs_id, fs_name, hit[1], hit[0]))
 return out, missing

def def_a_per_case(roi_long: pd.DataFrame, internal_ids: list[int]) -> pd.Series:
 sub = roi_long[roi_long["roi_id"].isin(internal_ids)].copy()
 sub = sub.dropna(subset=["dice"])
 return sub.groupby("case_id")["dice"].mean()

def boot_mean_ci(x: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
 x = np.asarray(x, dtype=np.float64)
 x = x[~np.isnan(x)]
 if x.size < 2:
 return float("nan"), float("nan"), float("nan")
 rng = np.random.default_rng(seed)
 boots = np.empty(n_boot)
 for i in range(n_boot):
 boots[i] = x[rng.integers(0, x.size, size=x.size)].mean()
 return (
 float(x.mean()),
 float(np.percentile(boots, 2.5)),
 float(np.percentile(boots, 97.5)),
 )

def main(argv: list[str] | None = None) -> int:
 ap = argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--label-map", type=Path, default=Path("data/label_map.json"))
 ap.add_argument("--risk-dir", type=Path, required=True,
 help="directory with <cohort>_risk.csv files")
 ap.add_argument("--dice-dir", type=Path, required=True,
 help="directory with <cohort>_macro_dice.csv long files")
 ap.add_argument("--cohorts", nargs="+", default=["IXI", "BrainLat", "Mindboggle-101"])
 ap.add_argument("--retentions", type=int, nargs="+", default=DEFAULT_RETENTIONS)
 ap.add_argument("--r-cols", nargs="+", default=DEFAULT_R_COLS,
 help="combined-R risk columns (default = mean of r1..r7)")
 ap.add_argument("--n-boot", type=int, default=2000)
 ap.add_argument("--seed", type=int, default=42)
 ap.add_argument("--output", type=Path, required=True)
 args = ap.parse_args(argv)

 resolved, missing = resolve_def_a(args.label_map)
 if missing:
 print("[warn] missing Def-A names in label map:")
 for fs_id, fs_name in missing:
 print(f" fs_id={fs_id} name={fs_name}")
 print(f"[ok] resolved {len(resolved)}/16 Def-A labels")
 internal_ids = [t[2] for t in resolved]

 rows = []
 base_seed = args.seed
 for ds in args.cohorts:
 risk_path = args.risk_dir / f"{ds}_risk.csv"
 dice_path = args.dice_dir / f"{ds}_macro_dice.csv"
 if not risk_path.is_file() or not dice_path.is_file():
 print(f"[skip] {ds}: missing inputs")
 continue
 risk = pd.read_csv(risk_path)
 roi = pd.read_csv(dice_path)
 case_dice = def_a_per_case(roi, internal_ids)
 n_total = len(risk)

 all_cases = set(risk["case_id"])
 x_base = case_dice[case_dice.index.isin(all_cases)].values
 m, lo, hi = boot_mean_ci(x_base, args.n_boot, base_seed + hash(("base", ds)) % (2**31))
 baseline = m
 rows.append(dict(dataset=ds, retention_pct=100, n_retained=n_total,
 baseline_def_a=baseline, retained_mean_def_a=m,
 delta_def_a=0.0, ci_lo=lo, ci_hi=hi))

 missing_cols = [c for c in args.r_cols if c not in risk.columns]
 if missing_cols:
 print(f"[skip] {ds}: missing r-cols {missing_cols}")
 continue

 combined = risk[args.r_cols].mean(axis=1)
 order = np.argsort(combined.values)
 sorted_cases = risk["case_id"].values[order]

 for K in args.retentions:
 k = int(round(n_total * K / 100.0))
 if k <= 0:
 continue
 retained = set(sorted_cases[:k])
 x = case_dice[case_dice.index.isin(retained)].values
 m, lo, hi = boot_mean_ci(x, args.n_boot,
 base_seed + hash((ds, K)) % (2**31))
 rows.append(dict(dataset=ds, retention_pct=K, n_retained=k,
 baseline_def_a=baseline,
 retained_mean_def_a=m,
 delta_def_a=(m - baseline) if not np.isnan(m) else float("nan"),
 ci_lo=lo, ci_hi=hi))

 out = pd.DataFrame(rows)
 args.output.parent.mkdir(parents=True, exist_ok=True)
 out.to_csv(args.output, index=False)
 print(f"[ok] wrote {args.output} rows={len(out)}")
 return 0

if __name__ == "__main__":
 raise SystemExit(main())
