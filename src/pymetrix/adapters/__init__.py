"""Helpers for components backed by other libraries.

The rule: our specification, data handling and result types; the library
does the numerical work. Libraries are optional and imported on first use.
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import Any

EXTRAS = {"linearmodels": "panel", "arch": "arch", "sklearn": "ml", "lightgbm": "lightgbm", "xgboost": "xgboost", "shap": "shap","mcp": "mcp", "openpyxl": "excel",
          "docx": "report", "PySide6": "qt", "ipywidgets": "jupyter"}


def require(module: str) -> Any:
    """Import a module, or explain which extra installs it."""
    top = module.split(".")[0]
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        if top not in EXTRAS:  # a core dependency (statsmodels, matplotlib) missing: a broken install
            raise ImportError(f"this feature uses {top}, which pymetrix requires: pip install {top}") from exc
        raise ImportError(f"this feature uses {top}: pip install pymetrix[{EXTRAS[top]}]") from exc
    except OSError as exc:  # installed, but its compiled library would not load (LightGBM without OpenMP on a Mac)
        raise ImportError(f"{top} is installed but its compiled library could not be loaded ({exc}). LightGBM and "
                          "XGBoost need the OpenMP runtime: on macOS, brew install libomp") from exc


def backend(module: str) -> str:
    """``"statsmodels 0.15.0"``, for provenance records."""
    top = module.split(".")[0]
    try:
        return f"{top} {version(top)}"
    except PackageNotFoundError:
        return top
