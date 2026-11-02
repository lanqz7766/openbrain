"""Create a quick PNG preview for one OpenBrain sample case.

The script renders the center axial, coronal, and sagittal slices from
t1w_brain.nii.gz and overlays parcellation_modernsurfer.nii.gz.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def _normalize_slice(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=float)
    lo, hi = np.percentile(x[finite], [1, 99])
    if hi <= lo:
        return np.zeros_like(x, dtype=float)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def _slice(volume: np.ndarray, axis: int) -> np.ndarray:
    idx = volume.shape[axis] // 2
    return np.take(volume, idx, axis=axis)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.35)
    args = parser.parse_args()

    image_path = args.case_dir / "t1w_brain.nii.gz"
    label_path = args.case_dir / "parcellation_modernsurfer.nii.gz"
    mask_path = args.case_dir / "brain_mask.nii.gz"
    for path in [image_path, label_path, mask_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    image = np.asarray(nib.load(image_path).dataobj, dtype=np.float32)
    label = np.asarray(nib.load(label_path).dataobj)
    mask = np.asarray(nib.load(mask_path).dataobj) > 0

    planes = [("Sagittal", 0), ("Coronal", 1), ("Axial", 2)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    fig.suptitle(args.case_dir.name, fontsize=10)

    for ax, (title, axis) in zip(axes, planes):
        img_slice = np.rot90(_normalize_slice(_slice(image, axis)))
        lab_slice = np.rot90(_slice(label, axis))
        mask_slice = np.rot90(_slice(mask, axis))

        ax.imshow(img_slice, cmap="gray", interpolation="nearest")
        overlay = np.ma.masked_where(lab_slice <= 0, lab_slice)
        ax.imshow(overlay, cmap="tab20", alpha=args.alpha, interpolation="nearest")
        contour = np.ma.masked_where(~mask_slice, mask_slice)
        ax.contour(contour, levels=[0.5], colors="white", linewidths=0.4)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
