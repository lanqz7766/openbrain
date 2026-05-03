# SPDX-License-Identifier: MIT
"""Q2 — Volumetric outlier (per-source IQR fences).

Definition (paper Appendix H.2). For each (source dataset D, label L)
fit Tukey 1.5-IQR fences from the within-source distribution of voxel
counts V_D,L:

 lower_D,L = max(0, Q1_D,L - 1.5 * (Q3_D,L - Q1_D,L))
 upper_D,L = Q3_D,L + 1.5 * (Q3_D,L - Q1_D,L)

Q2 for a case c in source D is the fraction of *applicable* labels for
which V_c,L falls outside [lower_D,L, upper_D,L]. Labels are applicable
when they are present in at least 20% of cases in the source cohort, the
source has >= 30 cases (otherwise we fall back to the global 35,838-case
fences), and ``L`` is in the included label list. Lower is better.

This module exposes the fence-fitting helper plus the per-case scoring
function. A typical pipeline computes per-case volume vectors with
``vols_per_case`` (a 1-D bincount over ``label.ravel()``), aggregates them
with ``fit_fences_per_source``, and then scores each case with
``score_case``.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

NAME = "Q2"
DIRECTION = "lower_is_better"
DESCRIPTION = "fraction of labels whose volume falls outside per-source IQR fences"

DEFAULT_MIN_SOURCE_N = 30
DEFAULT_MIN_LABEL_PRESENCE = 0.20

@dataclass(frozen=True)
class Fence:
 lo: float
 hi: float
 n: int

def vols_per_case(label_ms: np.ndarray, num_labels: int = 111) -> np.ndarray:
 """Return a length-``num_labels`` voxel-count vector for ``label_ms``."""
 return np.bincount(label_ms.ravel(), minlength=num_labels)

def fit_fences(
 vols: np.ndarray, # (n_cases, num_labels)
 include_ids: Iterable[int],
 min_label_presence: float = DEFAULT_MIN_LABEL_PRESENCE,
) -> dict[int, Fence]:
 """Fit Tukey 1.5-IQR fences per label across a cohort."""
 n = vols.shape[0]
 fences: dict[int, Fence] = {}
 for L in include_ids:
 col = vols[:, L]
 mask = col > 0
 present = float(mask.mean())
 if present < min_label_presence or mask.sum() < 5:
 continue
 sub = col[mask].astype(np.float64)
 q1, q3 = np.percentile(sub, [25, 75])
 iqr = q3 - q1
 fences[int(L)] = Fence(
 lo=float(max(0.0, q1 - 1.5 * iqr)),
 hi=float(q3 + 1.5 * iqr),
 n=int(mask.sum()),
 )
 return fences

def fit_fences_per_source(
 vols_by_source: Mapping[str, np.ndarray],
 include_ids: Iterable[int],
 min_source_n: int = DEFAULT_MIN_SOURCE_N,
 min_label_presence: float = DEFAULT_MIN_LABEL_PRESENCE,
) -> tuple[dict[str, dict[int, Fence]], dict[int, Fence]]:
 """Fit per-source fences and a global fallback.

 Sources with fewer than ``min_source_n`` cases get an empty per-source
 fence map; their cases will fall back to the global fences.
 """
 include_ids = list(include_ids)
 per_source: dict[str, dict[int, Fence]] = {}
 for src, vols in vols_by_source.items():
 if vols.shape[0] < min_source_n:
 per_source[src] = {}
 continue
 per_source[src] = fit_fences(vols, include_ids, min_label_presence)
 all_vols = np.vstack(list(vols_by_source.values())) if vols_by_source else np.empty((0, 0))
 global_fences = (
 fit_fences(all_vols, include_ids, min_label_presence)
 if all_vols.size
 else {}
 )
 return per_source, global_fences

def score_case(
 case_vols: np.ndarray,
 fences: Mapping[int, Fence],
 global_fences: Mapping[int, Fence] | None = None,
) -> float:
 """Score one case against its source fences (with global fallback)."""
 if not fences and not global_fences:
 return float("nan")
 use = fences if fences else (global_fences or {})
 n_total = 0
 n_out = 0
 for L, f in use.items():
 v = float(case_vols[L])
 if v <= 0:
 continue
 n_total += 1
 if v < f.lo or v > f.hi:
 n_out += 1
 if n_total == 0:
 return float("nan")
 return float(n_out / n_total)

def main(argv: list[str] | None = None) -> None:
 ap = argparse.ArgumentParser(description=DESCRIPTION)
 ap.add_argument("--volumes-csv", type=Path, required=True,
 help="CSV with columns case_id, source, vol_L_1..vol_L_110")
 ap.add_argument("--include-ids", type=Path, default=None)
 ap.add_argument("--output", type=Path, required=True)
 args = ap.parse_args(argv)

 import pandas as pd
 from qc._io import included_label_ids

 df = pd.read_csv(args.volumes_csv, low_memory=False)
 include = included_label_ids(args.include_ids)
 vol_cols = [f"vol_L_{L}" for L in range(1, 111)]
 by_source = {
 src: g[vol_cols].to_numpy(dtype=np.int64)
 for src, g in df.groupby("source", sort=False)
 }
 # pad to 111-wide bincount-style vector
 by_source = {src: np.pad(v, ((0, 0), (1, 0))) for src, v in by_source.items()}
 per_src, glob = fit_fences_per_source(by_source, include)
 rows = []
 for _, row in df.iterrows():
 src = row["source"]
 vec = np.zeros(111, dtype=np.int64)
 for L in range(1, 111):
 vec[L] = int(row.get(f"vol_L_{L}", 0))
 rows.append(
 dict(case_id=row["case_id"], source=src,
 Q2=score_case(vec, per_src.get(src, {}), glob))
 )
 pd.DataFrame(rows).to_csv(args.output, index=False)
 print(f"wrote {args.output}")

if __name__ == "__main__":
 main()
