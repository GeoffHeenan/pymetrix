"""Forecast accuracy."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..output import Table


def forecast_accuracy(actual: pd.Series, forecasts: pd.Series | Mapping[str, pd.Series]) -> Table:
    """RMSE, MAE, MAPE and Theil's U for one or more forecasts of the same series.

    Observations where the actual or a forecast is missing are skipped.
    """
    if isinstance(forecasts, pd.Series):
        forecasts = {forecasts.name or "forecast": forecasts}
    rows, names = [], []
    for name, f in forecasts.items():
        both = pd.concat([actual.rename("a"), f.rename("f")], axis=1).dropna()
        a, f_ = both["a"].to_numpy(float), both["f"].to_numpy(float)
        e = a - f_
        rmse = np.sqrt(np.mean(e**2))
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = 100 * np.mean(np.abs(e / a))
        theil = rmse / (np.sqrt(np.mean(a**2)) + np.sqrt(np.mean(f_**2)))
        rows.append([len(e), rmse, np.mean(np.abs(e)), mape, theil, np.mean(e)])
        names.append(str(name))
    return Table(["n", "RMSE", "MAE", "MAPE", "Theil U", "mean error"], rows, index=names, title="Forecast accuracy")
