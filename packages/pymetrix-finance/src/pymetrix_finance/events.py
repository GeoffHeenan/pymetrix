"""Event studies of returns: abnormal returns around events, and whether they differ from zero.

    ev = abnormal_returns(data, {"a": "2015-03-02", "b": ["2016-05-10", "2018-01-15"]}, market="mkt")
    ev.tests(); ev.car(); ev.plot()                # the tests for the window; each event's CAR; the average path

Each event's normal return is estimated over the ``estimation`` window before it (relative periods, -250 to -11 by
default): the market model r = a + b m (``market``), a factor model (``factors``), or the mean return (neither). The
abnormal return in the event ``window`` is the actual return less the normal one; their average across events
(AAR) and its cumulative sum (CAAR) are tested over ``car_window`` by
    - the cross-sectional t test: mean CAR over its cross-sectional standard error, t(N - 1);
    - Patell (1976): each abnormal return standardised by its prediction error, sum_i CSAR_i / sqrt(sum_i (L_i - k)
      / (L_i - k - 2)), with L_i estimation periods and k parameters; normal;
    - Boehmer, Musumeci and Poulsen (1991): the cross-sectional t test of the standardised CARs, robust to the
      events changing the variance; t(N - 1);
    - Corrado (1989) and Corrado-Zivney's rank test over the window, U = rank / (L + 1) - 1/2 within each event's
      estimation and event periods; normal.
CSAR_i is the sum of event i's standardised abnormal returns over the window divided by the root of its length.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from pymetrix.extension import api
from pymetrix.errors import SpecificationError
from pymetrix.output import Table
from pymetrix.extension import plugin_context
from . import PLUGIN

if TYPE_CHECKING:
    from pymetrix.data import Dataset
    from pymetrix.figures import Figure


def _event_list(events: Any) -> list[tuple[str, str]]:
    if isinstance(events, Mapping):
        return [(a, str(d)) for a, ds in events.items() for d in ([ds] if isinstance(ds, (str, pd.Period)) else ds)]
    return [(str(a), str(d)) for a, d in events]


@dataclass(frozen=True, eq=False)
class AbnormalReturns:
    """Abnormal returns around events, their averages and the tests that they are zero."""

    events: tuple[tuple[str, str], ...]
    model: str
    ar: pd.DataFrame  # relative period by event
    sar: pd.DataFrame = field(repr=False)  # standardised by the prediction error
    est_obs: np.ndarray = field(repr=False)  # each event's estimation periods
    k: int = 1
    ranks: pd.DataFrame = field(repr=False, default=None)  # U by relative period (estimation and event) and event
    car_window: tuple[int, int] = (0, 0)

    stage = "finance"

    @property
    def labels(self) -> list[str]:
        return list(self.ar.columns)

    def _window(self, window: tuple[int, int] | None) -> pd.DataFrame:
        lo, hi = window or self.car_window
        return self.ar.loc[lo:hi]

    def car(self, window: tuple[int, int] | None = None) -> pd.Series:
        """Each event's cumulative abnormal return over ``window`` (default: the tested window)."""
        return self._window(window).sum().rename("CAR")

    def average(self) -> pd.DataFrame:
        """AAR and CAAR by relative period, with the AAR's cross-sectional t statistic."""
        N = self.ar.shape[1]
        aar = self.ar.mean(axis=1)
        t = aar / (self.ar.std(axis=1, ddof=1) / np.sqrt(N))
        cum = self.ar.cumsum()
        band = 1.96 * cum.std(axis=1, ddof=1) / np.sqrt(N)
        return pd.DataFrame({"AAR": aar, "t": t, "CAAR": aar.cumsum(), "CAAR lower": aar.cumsum() - band,
                             "CAAR upper": aar.cumsum() + band})

    def statistics(self, window: tuple[int, int] | None = None) -> dict[str, tuple[float, float]]:
        """Each test's statistic and two-sided p-value for the CAR over ``window``."""
        lo, hi = window or self.car_window
        N, Lc = self.ar.shape[1], hi - lo + 1
        car = self.car((lo, hi)).to_numpy()
        cs_t = car.mean() / (car.std(ddof=1) / np.sqrt(N))
        csar = self.sar.loc[lo:hi].sum().to_numpy() / np.sqrt(Lc)
        dof = self.est_obs - self.k
        patell = csar.sum() / np.sqrt(np.sum(dof / (dof - 2)))
        bmp = csar.mean() / (csar.std(ddof=1) / np.sqrt(N))
        ubar = self.ranks.mean(axis=1)
        s = np.sqrt(np.mean(ubar**2))
        rank = ubar.loc[lo:hi].sum() / (np.sqrt(Lc) * s)
        two = lambda z: float(2 * sps.norm.sf(abs(z)))  # noqa: E731
        return {"cross-sectional t": (float(cs_t), float(2 * sps.t.sf(abs(cs_t), N - 1))), "Patell Z": (float(patell), two(patell)),
                "BMP t": (float(bmp), float(2 * sps.t.sf(abs(bmp), N - 1))), "Corrado rank": (float(rank), two(rank))}

    def tests(self, window: tuple[int, int] | None = None) -> Table:
        lo, hi = window or self.car_window
        stats = self.statistics((lo, hi))
        car = self.car((lo, hi))
        return Table(["statistic", "p-value"], [list(v) for v in stats.values()], index=list(stats), index_name="test",
                     title=f"Abnormal returns, periods {lo} to {hi}",
                     notes=(f"{len(car)} events; mean CAR {car.mean():.6g}, median {car.median():.6g}, {int((car > 0).sum())} positive.",
                            f"Normal returns: the {self.model}, estimated over {int(self.est_obs.min())} to {int(self.est_obs.max())} periods before each event."))

    def summary(self) -> Table:
        return self.tests()

    def plot(self) -> Figure:
        from pymetrix.figures import Band, Bars, Figure, Line, Panel, Rule

        a = self.average()
        return Figure((Panel((Band(a["CAAR lower"], a["CAAR upper"], "95% band"), Line(a["CAAR"], "CAAR"), Rule(0.0)),
                             title="Cumulative average abnormal return", xlabel="periods from the event"),
                       Panel((Bars(a["AAR"], "AAR"), Rule(0.0)), title="Average abnormal return", xlabel="periods from the event")))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            from pymetrix.extension import views_frame

            return views_frame(self, view)
        return self.ar

    def __repr__(self) -> str:
        return f"<AbnormalReturns of {len(self.events)} events, {self.model}>"


with plugin_context(PLUGIN):
    @api("finance:abnormal_returns")
    def abnormal_returns(data: Dataset, events: Any, market: str | None = None, factors: Sequence[str] | None = None,
                         estimation: tuple[int, int] = (-250, -11), window: tuple[int, int] = (-10, 10),
                         car_window: tuple[int, int] | None = None) -> AbnormalReturns:
        """Abnormal returns of assets around their events (a mapping of asset to dates, or (asset, date) pairs), against
        the market model, a factor model or the mean return, with the cross-sectional, Patell, BMP and rank tests over
        ``car_window`` (default: the whole event window)."""
        pairs = _event_list(events)
        if not pairs:
            raise SpecificationError("give at least one event: {asset: date} or [(asset, date), ...]")
        if data.time_axis is None or data.kind != "time_series":
            raise SpecificationError("an event study needs returns over time")
        if market and factors:
            raise SpecificationError("give the market or the factors, not both")
        regressors = [market] if market else list(factors or [])
        needed = list(dict.fromkeys([a for a, _ in pairs] + regressors))
        missing = [s for s in needed if s not in data.names]
        if missing:
            raise SpecificationError(f"no series {', '.join(map(repr, missing))} in the dataset")
        e0, e1 = map(int, estimation)
        w0, w1 = map(int, window)
        if not (e0 < e1 < w0 <= 0 <= w1):
            raise SpecificationError("the estimation window must end before the event window, which must include period 0")
        c0, c1 = map(int, car_window) if car_window else (w0, w1)
        if not w0 <= c0 <= c1 <= w1:
            raise SpecificationError("the tested window must lie within the event window")
        frame = data.frame
        index = frame.index
        k = 1 + len(regressors)
        model = ("market model" if market else f"factor model on {', '.join(regressors)}") if regressors else "mean return"
        AR, SAR, U, obs, labels = {}, {}, {}, [], []
        for asset, date in pairs:
            try:
                pos = index.get_loc(pd.Period(date, index.freq))
            except (KeyError, ValueError):
                raise SpecificationError(f"the event date {date!r} of {asset} is not in the data") from None
            if pos + e0 < 0 or pos + w1 >= len(index):
                raise SpecificationError(f"the event of {asset} on {date} is too near the data's start or end for the windows")
            est = frame.iloc[pos + e0:pos + e1 + 1][[asset, *regressors]].dropna()
            evt = frame.iloc[pos + w0:pos + w1 + 1][[asset, *regressors]]
            L = len(est)
            if L <= k + 2:
                raise SpecificationError(f"too few estimation periods with data for {asset} on {date}")
            X = np.column_stack([np.ones(L), est[regressors].to_numpy(float)])
            y = est[asset].to_numpy(float)
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ coef
            s2 = resid @ resid / (L - k)
            Xe = np.column_stack([np.ones(len(evt)), evt[regressors].to_numpy(float)])
            ar = evt[asset].to_numpy(float) - Xe @ coef
            lev = np.einsum("ij,jk,ik->i", Xe, np.linalg.inv(X.T @ X), Xe)
            label = f"{asset} {date}"
            if label in AR:
                raise SpecificationError(f"the event {label} is given twice")
            rel = range(w0, w1 + 1)
            AR[label], SAR[label] = pd.Series(ar, index=rel), pd.Series(ar / np.sqrt(s2 * (1 + lev)), index=rel)
            both = np.concatenate([resid, ar])
            ranks = sps.rankdata(np.where(np.isnan(both), np.inf, both))
            u = np.where(np.isnan(both), np.nan, ranks / (np.isfinite(both).sum() + 1) - 0.5)
            est_rel = (index.get_indexer(est.index) - pos).tolist()
            U[label] = pd.Series(u, index=est_rel + list(rel))
            obs.append(L)
            labels.append(label)
        ar_frame, sar_frame = pd.DataFrame(AR), pd.DataFrame(SAR)
        ar_frame.index.name = sar_frame.index.name = "period"
        if ar_frame.isna().any().any():
            raise SpecificationError("some events have missing returns in the event window")
        return AbnormalReturns(tuple(pairs), model, ar_frame, sar_frame, np.array(obs, float), k, pd.DataFrame(U).sort_index(), (c0, c1))
