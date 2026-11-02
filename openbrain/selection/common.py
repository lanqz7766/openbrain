# SPDX-License-Identifier: MIT
"""Shared helpers for subset selectors.

The three selectors all start from the same release quality manifest produced by
``openbrain.qc.combine``, optionally enriched with an ``image_quality_score``
(``IQ``) column from a metadata CSV. The selectors share the joint-score
construction and the per-source diversity bookkeeping.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

R_COLS_DEFAULT = [f"r{i}" for i in range(1, 8)]
DEFAULT_SOURCE_COL = "source_dataset"

def load_manifest(
    manifest: Path,
    metadata: Path | None,
    r_cols: Iterable[str] = R_COLS_DEFAULT,
    require_iq: bool = True,
) -> pd.DataFrame:
    """Load the release quality manifest and optional metadata join.

    The manifest is the output of ``openbrain.qc.combine``; it must have columns
    ``case_id``, ``source_dataset``, ``R``, and the per-axis ``r1..r7``.

    The metadata file (optional) is a wide CSV that we join on
    ``case_id`` to add an ``image_quality_score`` column. If
    ``require_iq=True`` and the metadata file is provided, rows with NaN
    IQ are dropped.
    """
    df = pd.read_csv(manifest, low_memory=False)
    if metadata is not None:
        meta = pd.read_csv(metadata, low_memory=False)
        if "case_id" not in meta.columns and "unique_id" in meta.columns:
            meta = meta.assign(
                case_id=meta["unique_id"].astype(str).str.replace(".nii.gz", "", regex=False)
            )
        keep_cols = [c for c in ("case_id", "image_quality_score", "age", "sex")
                     if c in meta.columns]
        df = df.merge(meta[keep_cols], on="case_id", how="left")
    needed = list(r_cols) + ["R"]
    if require_iq and "image_quality_score" in df.columns:
        needed.append("image_quality_score")
    df = df.dropna(subset=needed).reset_index(drop=True)
    return df

def add_joint_score(
    df: pd.DataFrame,
    use_iq: bool = True,
    iq_col: str = "image_quality_score",
) -> pd.DataFrame:
    """Add a `joint_score` column = rank(R) + rank(-IQ).

    Lower joint_score == lower combined risk + higher image quality.
    """
    out = df.copy()
    out["rank_R"] = out["R"].rank(method="average")
    if use_iq and iq_col in out.columns:
        out["rank_IQ_inv"] = (-out[iq_col]).rank(method="average")
        out["joint_score"] = out["rank_R"] + out["rank_IQ_inv"]
    else:
        out["joint_score"] = out["rank_R"]
    return out

def greedy_capped_select(
    df_sorted: pd.DataFrame,
    n: int,
    cap: int,
    source_col: str = DEFAULT_SOURCE_COL,
) -> pd.DataFrame:
    """Walk df_sorted; accept a row iff its source bucket has < cap picks.

    Returns at most ``n`` rows. Used to enforce per-source diversity.
    """
    counts: Counter = Counter()
    picked = []
    for idx, row in df_sorted.iterrows():
        src = row[source_col]
        if counts[src] < cap:
            picked.append(idx)
            counts[src] += 1
        if len(picked) >= n:
            break
    return df_sorted.loc[picked].copy()

def round_robin_select(
    df_sorted: pd.DataFrame,
    n: int,
    source_col: str = DEFAULT_SOURCE_COL,
) -> pd.DataFrame:
    """Round-robin pick: at cap=1 take first one per source going down the
    sorted list; at cap=2 add a second per source; ... until ``n`` reached.

    Maximises the number of unique sources in the result.
    """
    counts: Counter = Counter()
    picked: list[int] = []
    cap = 0
    while len(picked) < n:
        cap += 1
        added_this_pass = False
        for idx, row in df_sorted.iterrows():
            if idx in picked:
                continue
            src = row[source_col]
            if counts[src] < cap:
                picked.append(idx)
                counts[src] += 1
                added_this_pass = True
            if len(picked) >= n:
                break
        if not added_this_pass:
            # we exhausted the pool before reaching n
            break
    return df_sorted.loc[picked].copy()

def summarise_arm(arm: pd.DataFrame, name: str, source_col: str = DEFAULT_SOURCE_COL) -> str:
    n_src = arm[source_col].nunique()
    if len(arm) == 0:
        return f"[{name}] empty"
    top_pct = 100 * arm[source_col].value_counts().iloc[0] / len(arm)
    bits = [f"n={len(arm)}", f"sources={n_src}", f"top_source_pct={top_pct:.1f}%"]
    if "R" in arm.columns:
        bits.append(f"R_mean={arm['R'].mean():.3f}")
    if "image_quality_score" in arm.columns:
        bits.append(f"IQ_mean={arm['image_quality_score'].mean():.3f}")
    return f"[{name}] " + " ".join(bits)
