# SPDX-License-Identifier: MIT
"""OpenBrain: quality checks, training-set selection, and evaluation code."""

__version__ = "1.0.0"

from openbrain.data import Case, download_source, extract_case, load_case  # noqa: E402

__all__ = ["Case", "download_source", "extract_case", "load_case", "__version__"]
