# SPDX-License-Identifier: MIT
"""Load OpenBrain cases from the Hugging Face release.

The release stores one uncompressed tar per OpenNeuro source dataset at
``cases/<source>.tar``. ``load_case`` downloads the tar of the case's source
once (cached by huggingface_hub), extracts the case folder, and returns the
five images plus the case's metadata and quality rows.

    from openbrain import load_case
    case = load_case("ds000001__sub-01__ses-DEFAULT__sub-01_T1w")
    case.t1w, case.parcellation, case.quality["tier"]
"""
from __future__ import annotations

import tarfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ID = "lanqz7766/OpenBrain"
FILES = {
    "t1w": "t1w_brain.nii.gz",
    "brain_mask": "brain_mask.nii.gz",
    "parcellation": "parcellation_modernsurfer.nii.gz",
    "swinunetr": "parcellation_swinunetr.nii.gz",
    "unetr": "parcellation_unetr.nii.gz",
}


@dataclass
class Case:
    case_id: str
    source: str
    path: Path
    quality: Any = None   # pandas Series from metadata/quality.csv
    meta: Any = None      # pandas Series from metadata/cases.csv
    _images: dict = field(default_factory=dict, repr=False)

    def _load(self, key: str):
        if key not in self._images:
            import nibabel as nib
            self._images[key] = nib.load(self.path / FILES[key])
        return self._images[key]

    @property
    def t1w(self):
        return self._load("t1w")

    @property
    def brain_mask(self):
        return self._load("brain_mask")

    @property
    def parcellation(self):
        return self._load("parcellation")

    @property
    def committee(self) -> dict:
        return {"swinunetr": self._load("swinunetr"), "unetr": self._load("unetr")}


def source_of(case_id: str) -> str:
    return case_id.split("__", 1)[0]


def _hf_download(filename: str, revision: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(REPO_ID, filename, repo_type="dataset", revision=revision))


@lru_cache(maxsize=None)
def metadata(revision: str | None = None):
    """Return (cases, quality) DataFrames indexed by case_id."""
    import pandas as pd
    cases = pd.read_csv(_hf_download("metadata/cases.csv", revision), low_memory=False).set_index("case_id")
    quality = pd.read_csv(_hf_download("metadata/quality.csv", revision)).set_index("case_id")
    return cases, quality


def download_source(source: str, revision: str | None = None) -> Path:
    """Download (or reuse the cached) tar of one OpenNeuro source dataset."""
    return _hf_download(f"cases/{source}.tar", revision)


def extract_case(tar_path: Path, case_id: str, out_dir: Path) -> Path:
    """Extract one case folder from a source tar; skip if already present."""
    dest = out_dir / case_id
    if all((dest / f).exists() for f in FILES.values()):
        return dest
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        members = [m for m in tar.getmembers() if m.name.startswith(case_id + "/")]
        if not members:
            raise KeyError(f"{case_id} not found in {tar_path.name}")
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(out_dir, members=members, **kwargs)
    return dest


def load_case(case_id: str, out_dir: str | Path | None = None, revision: str | None = None) -> Case:
    """Download the case's source tar if needed, extract the case, return it.

    ``out_dir`` defaults to ``~/.cache/openbrain/cases``.
    """
    out_dir = Path(out_dir) if out_dir else Path.home() / ".cache" / "openbrain" / "cases"
    source = source_of(case_id)
    tar_path = download_source(source, revision)
    path = extract_case(tar_path, case_id, out_dir)
    quality = meta = None
    try:
        cases, qual = metadata(revision)
        meta = cases.loc[case_id] if case_id in cases.index else None
        quality = qual.loc[case_id] if case_id in qual.index else None
    except Exception:  # metadata is optional for loading images
        pass
    return Case(case_id=case_id, source=source, path=path, quality=quality, meta=meta)
