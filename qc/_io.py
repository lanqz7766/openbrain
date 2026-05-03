# SPDX-License-Identifier: MIT
"""Shared I/O helpers for the QC QC pipeline.

Lazy-imports nibabel so that the package remains importable on machines
without a NIfTI stack (e.g. the CI smoke job that only exercises the pure
NumPy paths in tests).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

def load_exclude_ids(path: str | Path) -> set[int]:
 """Load a one-id-per-line label-ID exclusion file (background = 0 always).

 Lines beginning with ``#`` are comments. The first whitespace- or
 comma-separated token on each line is parsed as an integer label ID.
 """
 ids: set[int] = {0}
 for line in Path(path).read_text().splitlines():
 stripped = line.strip()
 if not stripped or stripped.startswith("#"):
 continue
 tok = stripped.split()[0].split(",")[0].strip()
 if tok.isdigit():
 ids.add(int(tok))
 return ids

def included_label_ids(
 exclude_path: str | Path | None, num_labels: int = 111
) -> list[int]:
 """Return the sorted list of foreground label IDs after exclusion."""
 if exclude_path is None:
 return list(range(1, num_labels))
 excl = load_exclude_ids(exclude_path)
 return [L for L in range(1, num_labels) if L not in excl]

def load_nifti_int(path: str | Path) -> np.ndarray:
 """Load a NIfTI label volume as a 3D int32 NumPy array."""
 import nibabel as nib # lazy

 arr = np.asanyarray(nib.load(str(path)).dataobj).astype(np.int32, copy=False)
 if arr.ndim == 4:
 arr = arr[..., 0]
 return arr

def load_nifti_float(path: str | Path) -> np.ndarray:
 """Load a NIfTI image as a 3D float32 NumPy array."""
 import nibabel as nib # lazy

 arr = np.asanyarray(nib.load(str(path)).dataobj).astype(np.float32, copy=False)
 if arr.ndim == 4:
 arr = arr[..., 0]
 return arr

def normalize_case_id(fname: str) -> str:
 """Strip a leading ``TestSet_`` prefix and the .nii.gz suffix."""
 stem = fname[: -len(".nii.gz")] if fname.endswith(".nii.gz") else fname
 if stem.startswith("TestSet_"):
 stem = stem[len("TestSet_"):]
 return stem

def assert_same_shape(*arrays: np.ndarray, names: Iterable[str] | None = None) -> None:
 """Raise if any of the provided arrays have differing shapes."""
 shapes = [a.shape for a in arrays]
 if len(set(shapes)) > 1:
 if names is None:
 names = [f"arr{i}" for i in range(len(arrays))]
 raise ValueError(
 "shape mismatch: "
 + ", ".join(f"{n}={s}" for n, s in zip(names, shapes))
 )
