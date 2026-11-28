"""Reading Dynare model files (``.mod``): the model, its parameters, steady-state guesses and shocks.

    dyn = read_mod("nk.mod")             # or read_mod(text)
    dyn.model                            # a pymetrix Model (calibrated with the file's parameter values)
    sol = dyn.solve()                    # first-order rational-expectations solution (QZ, Blanchard-Kahn)
    sol.irf(20, shock_sd=dyn.shock_sd, shock_corr=dyn.shock_corr).plot()

What is read: ``var``, ``varexo`` and ``parameters`` (with ``$...$`` and ``(long_name=...)`` decorations
ignored), parameter assignments (numbers or expressions of earlier parameters), the ``model`` block
(timing ``x(+1)`` and ``x(-1)``, ``#`` model-local variables, ``[name=...]`` tags, equations without
``=`` meaning ``= 0``), ``initval`` and ``steady_state_model`` values (as starting values for the
steady state), and ``shocks``: ``stderr`` or variances (on a ``varobs`` variable, its measurement error, in
``measurement_sd``), and correlations, ``corr e1, e2 = 0.3;`` or as a
covariance ``var e1, e2 = 0.01;`` (divided by the two standard deviations), kept in ``shock_corr`` as
``{("e1", "e2"): 0.3}`` and used wherever the shocks' sizes are (see ``pymetrix.models.covariance``: the filter's
covariance, draws, moments; impulse responses and variance decompositions orthogonalised by Cholesky in the
order of ``varexo``, as ``stoch_simul``). Macro-processor directives (``@#define``, ``@#if``, ``@#for``,
``@#include``, ``@{...}``...) are carried out first, as Dynare's preprocessor does: see
``pymetrix.models.dynare_macros``; ``@#include`` needs the model read from its file. Commands such as ``steady``,
``check`` and ``stoch_simul`` are listed in ``commands`` but not run. Not read: correlations of measurement errors,
generalised prior bounds, and stoch_simul's options (``solve(order=2)`` gives the second order, ``solve(order=3)`` the third).

For estimation: ``estimated_params`` (all three forms: initial values and bounds, priors, or both;
``stderr`` for shock sizes, named ``sd_<shock>`` in pymetrix, and for a ``varobs`` variable's measurement error,
``sd_<variable>``; ``corr e1, e2`` for correlations of shocks, named ``corr_<e1>_<e2>`` with the two in ``varexo``
order, within (-1, 1)), ``varobs``, and the ``estimation(...)``
options (kept in ``options``; ``datafile``, ``first_obs``, ``nobs``, ``presample``, ``lik_init`` (1, 2 or 3), ``loglinear``
and ``logdata`` are used; ``dsge_var`` and ``dsge_varlag`` by ``dsge_var()``, a DSGE-VAR with its prior weight fixed or
estimated as ``dsge_prior_weight``).
Dynare's prior shapes map to pymetrix's: ``normal_pdf`` -> ``Normal``, ``beta_pdf`` -> ``Beta``, ``gamma_pdf``
-> ``Gamma``, ``inv_gamma_pdf``/``inv_gamma1_pdf`` -> ``InvGamma1``, ``inv_gamma2_pdf`` -> ``InvGamma2``,
``uniform_pdf`` -> ``Uniform`` (from its mean and standard deviation, or its bounds).

    dyn = read_mod("fs2000.mod")
    fit = dyn.estimate()                 # posterior mode, on the file's datafile, priors and observables
    post = fit.mcmc(draws=20000)         # Metropolis-Hastings, as estimation(mh_replic=...)

With ``loglinear`` (in ``estimation`` or ``stoch_simul``), every endogenous variable is written ``exp(x)`` in
the equations, so the model is solved in logs as Dynare's log-linearisation is: steady-state values and
responses are logs. The data are logged too unless ``logdata`` is given. ``.m`` data files are read for simple
MATLAB: matrix and vector literals, column selection ``data(:,k)``, element-wise arithmetic and ``diff``,
``log``, ``exp``, ``sqrt``, ``cumsum``; undated, so periods are numbered from ``start=`` (default ``0001Q1``).

Dynare writes equations in any form; pymetrix's models solve each equation for one variable. Each
equation is paired with one endogenous variable that appears in it unlagged (a bipartite matching);
an equation whose left-hand side is not simply its variable is rewritten ``v = v + (lhs) - (rhs)``,
which has the same solution.
"""

from __future__ import annotations

import keyword
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from .._api import api
from ..errors import SpecificationError
from .dynare_macros import expand_macros, has_macros
from .safe_eval import safe_eval
from .solvers import ModelError

__all__ = ["DynareModel", "read_mod"]

_FUNCS = {"exp": math.exp, "log": math.log, "ln": math.log, "log10": math.log10, "sqrt": math.sqrt, "abs": abs, "max": max, "min": min,
          "inf": math.inf, "Inf": math.inf, "NaN": math.nan, "nan": math.nan, "pi": math.pi, "erf": math.erf,
          "normcdf": lambda x, mu=0.0, sigma=1.0: 0.5 * math.erfc(-(x - mu) / (sigma * math.sqrt(2.0))),
          "normpdf": lambda x, mu=0.0, sigma=1.0: math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))}
_PRIORS = {"normal_pdf": "Normal", "beta_pdf": "Beta", "gamma_pdf": "Gamma", "inv_gamma_pdf": "InvGamma1",
           "inv_gamma1_pdf": "InvGamma1", "inv_gamma2_pdf": "InvGamma2", "uniform_pdf": "Uniform"}


def _shape(field: str) -> str | None:
    """The pymetrix prior for a Dynare prior-shape keyword (case does not matter, as in Dynare), or None."""
    return _PRIORS.get(field.strip().lower())


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"(?m)^\s*%[^\n]*", "", text)


# a declaration's decoration in parentheses, (long_name='...') and similar: quoted text inside it may hold parentheses,
# as in long_name='Output gap (% of potential)'
DECORATION = r"\((?:[^()'\"]|'[^']*'|\"[^\"]*\")*\)"


def _names(declaration: str) -> list[str]:
    declaration = re.sub(r"\$[^$]*\$", " ", declaration)  # LaTeX names
    declaration = re.sub(DECORATION, " ", declaration)
    return [n for n in re.split(r"[\s,]+", declaration) if n]


def _decorations(declaration: str) -> dict[str, tuple[str, str]]:
    """``{name: (symbol, long_name)}`` for the names a declaration decorates: ``alpha $\\alpha$ (long_name='capital
    share')``. The symbol is turned from LaTeX into Unicode (``Param.symbol``)."""
    from ..notation import from_latex

    out = {}
    pattern = r"([A-Za-z_]\w*)\s*(\$[^$]*\$)?\s*(" + DECORATION + ")?"
    for m in re.finditer(pattern, re.sub(r"\s+", " ", declaration)):
        symbol = from_latex(m.group(2)) if m.group(2) else ""
        long_name = re.search(r"long_name\s*=\s*(['\"])(.*?)\1", m.group(3) or "")
        if symbol or long_name:
            out[m.group(1)] = (symbol, long_name.group(2) if long_name else "")
    return out


def _safe(name: str) -> str:
    return f"{name}_" if keyword.iskeyword(name) else name  # Dynare names such as del are Python keywords


def _evaluate(expr: str, values: dict[str, float], what: str) -> float:
    expr = re.sub(r"\b[A-Za-z_]\w*\b", lambda m: _safe(m.group(0)), expr.replace("^", "**"))
    try:
        return float(safe_eval(expr, {_safe(k): v for k, v in values.items()}, _FUNCS))  # checked: numbers, names, arithmetic
    except Exception as exc:
        raise ModelError(f"cannot evaluate {what} = {expr}: {exc}") from None


@dataclass(frozen=True, eq=False)
class DynareModel:
    """A model read from a Dynare file, as a pymetrix ``Model`` with its calibration, guesses and shocks."""

    endogenous: tuple[str, ...]
    exogenous: tuple[str, ...]
    params: dict[str, float]
    equations: tuple[str, ...]  # as pymetrix equation text
    initval: dict[str, float] = field(default_factory=dict)
    shock_sd: dict[str, float] = field(default_factory=dict)
    commands: tuple[str, ...] = ()
    source: str = ""
    model: Any = field(default=None, repr=False)
    priors: dict[str, Any] = field(default_factory=dict)  # from estimated_params; shock sizes as sd_<shock>
    estimated: tuple[str, ...] = ()  # the parameters and shock sizes in estimated_params, in order
    start: dict[str, float] = field(default_factory=dict)  # initial values given in estimated_params
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)  # bounds given in estimated_params
    varobs: tuple[str, ...] = ()
    options: dict[str, Any] = field(default_factory=dict)  # the estimation(...) options
    loglinear: bool = False  # the model is written in the logs of its variables
    labels: dict[str, str] = field(default_factory=dict)  # long_name descriptions of the variables and shocks
    measurement_sd: dict[str, float] = field(default_factory=dict)  # measurement errors: stderr of a varobs variable in shocks
    shock_corr: dict[tuple[str, str], float] = field(default_factory=dict)  # correlations of the shocks (corr in shocks)

    @property
    def guess(self) -> dict[str, float]:
        """Steady-state starting values for the model's variables (logs, for a log-linear model)."""
        return {k: v for k, v in self.initval.items() if k in self.endogenous}

    def steady_state(self) -> Any:
        return self.model.steady_state(guess=self.guess or None)

    def linearize(self) -> Any:
        return self.model.linearize(shocks=list(self.exogenous), guess=self.guess or None)

    def data(self, path: str | Path | None = None, start: str = "0001Q1") -> Any:
        """The observed series (``varobs``) from the ``datafile`` of ``estimation(...)`` (or ``path``): the sample
        ``first_obs`` and ``nobs`` select, logged for a log-linear model unless ``logdata`` is given. A ``Dataset``
        with periods numbered from ``start``."""
        import numpy as np
        import pandas as pd

        from ..data import Dataset

        if path is None:
            name = self.options.get("datafile")
            if not name:
                raise ModelError("no datafile in estimation(...): give the data, or data(path)")
            base = Path(self.source).parent if self.source else Path(".")
            candidates = [base / f"{name}{ext}" for ext in ("", ".m", ".mat", ".csv")]
            path = next((c for c in candidates if c.is_file()), None)
            if path is None:
                raise ModelError(f"the datafile {name} was not found next to the model file ({base})")
        path = Path(path)
        if path.suffix == ".csv":
            values = {k: np.asarray(v, dtype=float) for k, v in pd.read_csv(path).items()}
        elif path.suffix == ".mat":
            values = read_mat_data(path)
        else:
            values = read_m_data(path)
        missing = [v for v in self.varobs if v not in values]
        if missing:
            raise ModelError(f"{path.name} has no series {missing}")
        if len({len(values[v]) for v in self.varobs}) != 1:
            raise ModelError(f"the observed series in {path.name} have different lengths")
        frame = pd.DataFrame({v: values[v] for v in self.varobs})
        first = int(self.options.get("first_obs", 1)) - 1
        nobs = int(self.options.get("nobs", len(frame) - first))
        frame = frame.iloc[first:first + nobs]
        if self.loglinear and "logdata" not in self.options:
            if (frame <= 0).any().any():
                raise ModelError("a log-linear model needs positive data (or the logdata option, for data in logs)")
            frame = np.log(frame)
        frame.index = pd.period_range(start, periods=len(frame), freq="Q", name="time")
        return Dataset.from_frame(frame)

    @api()
    def estimate(self, data: Any = None, estimate: Any = None, priors: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        """Estimate the model as the file's ``estimation`` would: the ``estimated_params`` (or ``estimate``) with
        their priors and bounds, fitted to ``varobs`` in ``data`` (default: the datafile). A ``FittedModel``: the
        posterior mode with priors (then ``.mcmc(...)``), maximum likelihood without."""
        from .model import Model

        if not self.varobs:
            raise ModelError("no varobs in the file: name the observed variables")
        data = self.data() if data is None else data
        names = list(estimate) if estimate is not None else list(self.estimated)
        if not names:
            raise ModelError("no estimated_params in the file: name what to estimate with estimate=[...]")
        chosen = {n: p for n, p in {**self.priors, **dict(priors or {})}.items() if n in names}
        bounds = {**{n: b for n, b in self.bounds.items() if n in names}, **dict(kwargs.pop("bounds", None) or {})}
        model = self.model
        structural = {n: v for n, v in self.start.items() if n in model.parameters.names}
        if structural:
            model = Model(model.equations, params=model.parameters.with_values(structural))
        shock_corr = self._corr(self.start)
        start, presample = self._filter_options()
        kwargs.setdefault("start", start)
        kwargs.setdefault("presample", presample)
        kwargs.setdefault("shock_corr", shock_corr or None)
        return model.estimate(data, self._measurement(self.start), shocks=list(self.exogenous), shock_sd=self._shock_sd(self.start),
                              estimate=names, priors=chosen or None, bounds=bounds or None, guess=self.guess or None, **kwargs)

    @api()
    def dsge_var(self, data: Any = None, estimate: Any = None, priors: dict[str, Any] | None = None, lags: int | None = None,
                 lambda_: Any = None, **kwargs: Any) -> Any:
        """A DSGE-VAR as the file's ``estimation(dsge_var=..., dsge_varlag=...)`` asks (a ``FittedDSGEVAR``; see
        ``pymetrix.models.dsge_var``): ``dsge_var=<value>`` fixes lambda; ``dsge_var`` alone estimates it as the
        parameter ``dsge_prior_weight`` of ``estimated_params``, with its prior; ``dsge_varlag`` gives the lags (4 by
        default). ``lags`` and ``lambda_`` override them. The rest as ``estimate``; the filter options do not apply."""
        from .dsge_var import LAMBDA
        from .model import Model

        if not self.varobs:
            raise ModelError("no varobs in the file: name the observed variables")
        data = self.data() if data is None else data
        weight = "dsge_prior_weight"
        names = list(estimate) if estimate is not None else list(self.estimated)
        option = self.options.get("dsge_var")
        if lambda_ is None:
            if weight in names:
                prior = {**self.priors, **dict(priors or {})}.get(weight)
                if prior is None:
                    raise ModelError(f"{weight} is estimated: give it a prior in estimated_params (or lambda_=)")
                lambda_ = prior
            elif option is None or option is True:
                raise ModelError("the file's estimation has no dsge_var=<value> (nor dsge_prior_weight in estimated_params): "
                                 "give lambda_=")
            else:
                lambda_ = float(option)
        names = [n for n in names if n != weight]
        if not names:
            raise ModelError("no estimated_params in the file besides the prior weight: name what to estimate with estimate=[...]")
        chosen = {n: p for n, p in {**self.priors, **dict(priors or {})}.items() if n in names}
        bounds = {**{n: b for n, b in self.bounds.items() if n in names}, **dict(kwargs.pop("bounds", None) or {})}
        model = self.model
        structural = {n: v for n, v in self.start.items() if n in model.parameters.names and n != weight}
        if structural:
            model = Model(model.equations, params=model.parameters.with_values(structural))
        kwargs.setdefault("shock_corr", self._corr(self.start) or None)
        lags = int(self.options.get("dsge_varlag", 4)) if lags is None else lags
        estimate_names = names + ([LAMBDA] if hasattr(lambda_, "logpdf") else [])
        return model.dsge_var(data, self._measurement(self.start), lags, lambda_, shocks=list(self.exogenous),
                              shock_sd=self._shock_sd(self.start), estimate=estimate_names, priors=chosen or None,
                              bounds=bounds or None, guess=self.guess or None, **kwargs)

    def _measurement(self, point: dict[str, float]) -> Any:
        """``varobs`` observed as they are, with the measurement errors of the ``shocks`` block, or ``point``'s
        ``sd_<variable>`` for those estimated."""
        from .measurement import MeasurementMap

        errors = {**self.measurement_sd, **{n[3:]: v for n, v in point.items() if n.startswith("sd_") and n[3:] in self.varobs}}
        return MeasurementMap.direct(list(self.varobs), errors)

    def _shock_sd(self, point: dict[str, float]) -> dict[str, float]:
        return {**self.shock_sd, **{n[3:]: v for n, v in point.items() if n.startswith("sd_") and n[3:] in self.exogenous}}

    def _corr(self, values: dict[str, float]) -> dict[tuple[str, str], float]:
        """The shocks' correlations: the shocks block's, then any ``corr_<a>_<b>`` in ``values``."""
        from .covariance import corr_pairs

        pairs = corr_pairs(self.exogenous)
        return {**getattr(self, "shock_corr", {}), **{pairs[n]: float(v) for n, v in values.items() if n in pairs}}

    def _filter_options(self) -> tuple[str, int]:
        """The Kalman filter's start and presample from the estimation options: Dynare's lik_init (1 stationary,
        2 scaled: variance 10 I, 3 diffuse) and presample."""
        if int(self.options.get("prefilter", 0)):
            raise ModelError("prefilter=1 (demeaned data) is not supported: give the data demeaned, or add constants to the model")
        start = "auto"
        if "lik_init" in self.options:
            starts = {1: "stationary", 2: "scaled", 3: "diffuse"}
            if int(self.options["lik_init"]) not in starts:
                raise ModelError(f"lik_init={self.options['lik_init']} is not supported (1, 2 or 3)")
            start = starts[int(self.options["lik_init"])]
        return start, int(self.options.get("presample", 0))

    def mode_file(self, path: str | Path | None = None) -> dict[str, float]:
        """The values in a Dynare mode file (``xparam1`` in a ``.mat`` file; default: the ``mode_file`` option, next to
        the model file), named as ``estimated``: shock sizes as ``sd_<shock>``, in the order of ``estimated_params``."""
        import numpy as np
        from scipy import io

        if path is None:
            name = self.options.get("mode_file")
            if not name:
                raise ModelError("no mode_file in estimation(...): give the path")
            base = Path(self.source).parent if self.source else Path(".")
            path = next((c for c in (base / f"{name}.mat", base / str(name)) if c.is_file()), None)
            if path is None:
                raise ModelError(f"the mode file {name} was not found next to the model file ({base})")
        x = np.ravel(io.loadmat(str(path))["xparam1"]).astype(float)
        if len(x) != len(self.estimated):
            raise ModelError(f"the mode file has {len(x)} values for {len(self.estimated)} estimated parameters")
        return dict(zip(self.estimated, map(float, x)))

    def log_posterior(self, values: dict[str, float] | None = None, data: Any = None) -> float:
        """The log posterior (log likelihood plus log priors) at these values of the estimated parameters (default:
        their initial values), as Dynare reports it with ``mode_compute=0``: e.g. ``log_posterior(mode_file())``.
        The filter's start and presample follow the estimation options; without priors, the log likelihood."""
        from .model import Model

        if not self.varobs:
            raise ModelError("no varobs in the file: name the observed variables")
        point = {**self.start, **dict(values or {})}
        structural = {n: v for n, v in point.items() if n in self.model.parameters.names}
        model = Model(self.model.equations, params=self.model.parameters.with_values(structural), roles=self.model.roles)
        solution = model.linearize(model.steady_state(guess=self.guess or None), shocks=list(self.exogenous)).solve()
        system = solution.system(self._measurement(point), self._shock_sd(point), shock_corr=self._corr(point) or None)
        start, presample = self._filter_options()
        loglike = system.loglike_observed(*system.observations(self.data() if data is None else data), start=start, presample=presample)
        return float(loglike + sum(prior.logpdf(point[n]) for n, prior in self.priors.items()))

    @api()
    def solve(self, order: int = 1) -> Any:
        """The rational-expectations solution, with every ``varexo`` as a shock: first order, or second order
        with the file's shock sizes (as ``stoch_simul(order=2)``), or third order (as ``stoch_simul(order=3)``)."""
        if order == 2:
            from .perturbation import second_order

            return second_order(self.model, self.exogenous, self.shock_sd, steady=self.steady_state(), shock_corr=self.shock_corr or None)
        if order == 3:
            from .third_order import third_order

            return third_order(self.model, self.exogenous, self.shock_sd, steady=self.steady_state(), shock_corr=self.shock_corr or None)
        if order != 1:
            raise SpecificationError("order is 1, 2 or 3")
        return self.linearize().solve()

    def __repr__(self) -> str:
        return (f"<DynareModel {self.source or 'from text'}: {len(self.endogenous)} variables, "
                f"{len(self.exogenous)} shocks, {len(self.params)} parameters>")


@api("pymetrix:read_mod")
def read_mod(source: str | Path) -> DynareModel:
    """Read a Dynare ``.mod`` file (a path, or the file's text)."""
    from ..estimation import Equation
    from ..metadata import Param
    from .model import Model

    path = Path(source) if not isinstance(source, str) or ("\n" not in source and source.endswith(".mod")) else None
    text = path.read_text(encoding="utf-8") if path is not None else str(source)
    text = _strip_comments(text)
    if has_macros(text):  # Dynare's macro processor first: see pymetrix.models.dynare_macros
        text = _strip_comments(expand_macros(text, base=path.parent if path is not None else None)[0])
    endo: list[str] = []
    exo: list[str] = []
    param_names: list[str] = []
    params: dict[str, float] = {}
    initval: dict[str, float] = {}
    shock_sd: dict[str, float] = {}
    measurement_sd: dict[str, float] = {}
    pairs: list[tuple[str, str, str, float]] = []  # corr and covariances in the shocks block
    commands: list[str] = []
    raw_equations: list[str] = []
    estimated_lines: list[str] = []
    steady_lines: list[str] = []  # initval and steady_state_model: evaluated once every parameter has a value
    varobs: list[str] = []
    options: dict[str, Any] = {}
    decorated: dict[str, tuple[str, str]] = {}  # symbols and long names from the declarations
    statements = iter(_statements(text))
    for st in statements:
        head = st.split(None, 1)[0] if st.split() else ""
        rest = st[len(head):].strip()
        if head in ("var", "varexo", "parameters"):
            decorated.update(_decorations(rest))
        if head == "var":
            endo += _names(rest)
        elif head == "varexo":
            exo += _names(rest)
        elif head == "parameters":
            param_names += _names(rest)
        elif re.match(r"^model\b", st):
            for inner in statements:
                if inner == "end":
                    break
                raw_equations.append(inner)
        elif head in ("initval", "steady_state_model", "endval"):
            for inner in statements:
                if inner == "end":
                    break
                if head != "endval" and inner.partition("=")[0].strip():
                    steady_lines.append(inner)
        elif head == "estimated_params":
            for inner in statements:
                if inner == "end":
                    break
                if inner:
                    estimated_lines.append(inner)
        elif head == "varobs":
            varobs += _names(rest)
        elif head == "shocks":
            current = None
            for inner in statements:
                if inner == "end":
                    break
                pair = re.match(r"^(corr|var)\s+(\w+)\s*,\s*(\w+)\s*=\s*(.+)$", inner)
                if pair:  # corr e1, e2 = correlation; var e1, e2 = covariance
                    kind, a, b = pair.group(1), pair.group(2), pair.group(3)
                    value = _evaluate(pair.group(4), params, f"{'correlation' if kind == 'corr' else 'covariance'} of {a} and {b}")
                    pairs.append((kind, a, b, value))
                    continue
                m = re.match(r"^var\s+(\w+)\s*(?:=\s*(.+))?$", inner)
                if m:
                    current = m.group(1)
                    sizes = measurement_sd if current in endo else shock_sd  # an endogenous variable's: measurement error
                    if m.group(2):
                        sizes[current] = math.sqrt(_evaluate(m.group(2), params, f"variance of {current}"))
                elif inner.startswith("stderr") and current:
                    sizes[current] = _evaluate(inner[len("stderr"):], params, f"stderr of {current}")
        elif "=" in st and st.split("=")[0].strip() in param_names:
            name, _, expr = st.partition("=")
            params[name.strip()] = _evaluate(expr, params, name.strip())
        elif st and st != "end":
            commands.append(st)
            m = re.match(r"^estimation\s*\((.*)\)\s*$", st)
            if m:
                options = _options(m.group(1))
    for line in estimated_lines:  # as Dynare: a parameter without a calibrated value starts at its initial value
        fields = _split_top(line)
        name = fields[0].split()[0] if fields[0].split() else ""
        if name in param_names and name not in params and len(fields) > 1 and not _shape(fields[1]):
            params[name] = _evaluate(fields[1], params, name)
    missing = [p for p in param_names if p not in params]  # an error only if used (Dynare leaves them NaN)
    for line in steady_lines:
        name, _, expr = line.partition("=")
        initval[name.strip()] = _evaluate(expr, {**params, **initval}, name.strip())
    if not raw_equations:
        raise ModelError("no model block")
    loglinear = "loglinear" in options or any(re.match(r"^stoch_simul\s*\(.*\bloglinear\b", c) for c in commands)
    equations = _convert(raw_equations, endo, exo, param_names)
    used = sorted({p for _, _, used_params in equations for p in used_params} & set(missing))
    if used:
        raise ModelError(f"no values for the parameters {used}")
    param_names = [p for p in param_names if p not in missing]
    if loglinear:  # every endogenous variable as exp(its log), as Dynare's log-linearisation
        equations = [(_logs(lhs, endo), _logs(rhs, endo), used) for lhs, rhs, used in equations]
        for name in [k for k in initval if k in endo]:
            if initval[name] <= 0:
                raise ModelError(f"loglinear: the steady-state value of {name} is {initval[name]:g}; logs need positive values")
            initval[name] = math.log(initval[name])
    if len(equations) != len(endo):
        raise ModelError(f"{len(equations)} equations for {len(endo)} variables")
    targets = _match(equations, endo)
    texts, eqs = [], []
    for (lhs, rhs, used_params), target in zip(equations, targets):
        natural = _single_current(lhs, endo + exo)
        body = (lhs, rhs) if natural == target else (target, f"{target} + ({lhs}) - ({rhs})")
        text_eq = f"{body[0]} {'~' if used_params else '='} {body[1]}"
        texts.append(text_eq)
        eqs.append(Equation(text_eq, params=used_params) if used_params else Equation(text_eq))
    shock_corr = _shock_corr(pairs, exo, shock_sd)
    priors, estimated, start, bounds = _estimated_params(estimated_lines, param_names, exo, params, endo)
    used_params = {p for e in eqs for p in e.param_names}
    named = ([p for p in param_names if p in used_params] + [f"sd_{x}" for x in exo] + [f"sd_{v}" for v in varobs if v in endo]
             + [n for n in estimated if n.startswith("corr_")])
    specs = [Param(n, label=decorated.get(n, ("", ""))[1], symbol=decorated.get(n, ("", ""))[0] or None,
                   bounds=bounds.get(n, (-math.inf, math.inf)), start=start.get(n), prior=priors.get(n))
             for n in named if n in decorated or n in priors or n in bounds or n in start]
    model = Model(eqs, params={p: params[p] for p in param_names if p in used_params} or None, specs=specs)
    unknown = [v for v in varobs if v not in endo]
    if unknown:
        raise ModelError(f"varobs names {unknown}, which are not variables of the model")
    unobserved = sorted({*measurement_sd, *(n[3:] for n in estimated if n.startswith("sd_") and n[3:] in endo)} - set(varobs))
    if unobserved:
        raise ModelError(f"measurement errors (stderr) on {unobserved}, which are not in varobs")
    labels = {n: long_name for n, (_, long_name) in decorated.items() if long_name and (n in endo or n in exo)}
    return DynareModel(tuple(endo), tuple(exo), params, tuple(texts), initval, shock_sd, tuple(commands),
                       str(path) if path is not None else "", model, priors, tuple(estimated), start, bounds, tuple(varobs),
                       options, loglinear, labels, measurement_sd, shock_corr)


def _split_top(text: str) -> list[str]:
    """Split at commas outside brackets."""
    out, depth, current = [], 0, ""
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(current.strip())
            current = ""
        else:
            current += ch
    out.append(current.strip())
    return out


def _options(text: str) -> dict[str, Any]:
    """``order=1, datafile=fs2000_data, loglinear`` -> {"order": 1, "datafile": "fs2000_data", "loglinear": True}."""
    out: dict[str, Any] = {}
    for item in _split_top(text):
        if not item:
            continue
        key, sep, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if not sep:
            out[key] = True
            continue
        try:
            out[key] = int(value)
        except ValueError:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value.strip("'\"")
    return out


def _logs(expr: str, endo: list[str]) -> str:
    """Each endogenous variable, with any lag or lead, as exp(...) of itself."""
    names = sorted(endo, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")\b((?:\[[+-]\d+\])?)")
    return pattern.sub(lambda m: f"exp({m.group(1)}{m.group(2)})", expr)


def _estimated_params(lines: list[str], param_names: list[str], exo: list[str], values: dict[str, float],
                      endo: Sequence[str] = ()) -> tuple[dict[str, Any], list[str], dict[str, float], dict[str, tuple[float, float]]]:
    """Priors, names (shock sizes as sd_<shock>, measurement errors as sd_<variable>), initial values and bounds from
    ``estimated_params`` lines."""
    from ..estimation import priors as P

    priors: dict[str, Any] = {}
    names: list[str] = []
    start: dict[str, float] = {}
    bounds: dict[str, tuple[float, float]] = {}

    def number(fields: list[str], i: int, what: str) -> float | None:
        return None if i >= len(fields) or fields[i].strip() == "" else _evaluate(fields[i], values, what)

    for line in lines:
        fields = _split_top(line)
        head = fields[0].split()
        if head[0] == "corr":  # corr e1, e2, ...: the correlation of two shocks, corr_<e1>_<e2> (in varexo order)
            if len(head) != 2 or len(fields) < 2 or len(fields[1].split()) != 1:
                raise ModelError(f"corr names two shocks (varexo): {line!r}")
            a, b = head[1], fields[1].strip()
            name = corr_name(a, b, exo)
            fields = [f"corr {a}", *fields[2:]]
            bounds.setdefault(name, (-1.0, 1.0))  # as Dynare: within (-1, 1) unless narrower bounds are given
        elif head[0] == "stderr":
            if len(head) != 2 or head[1] not in (*exo, *endo):  # an endogenous variable's: its measurement error
                raise ModelError(f"stderr must name a shock (varexo) or an observed variable (varobs): {line!r}")
            name = f"sd_{head[1]}"
        else:
            name = head[0]
            if name not in param_names:
                raise ModelError(f"estimated_params names {name}, which is not a parameter")
        rest = fields[1:]
        shape_at = next((i for i in (0, 3) if i < len(rest) and _shape(rest[i])), None)
        if shape_at is None and len(rest) > 3:  # prior fields, but no shape this reader knows: never drop a prior silently
            raise ModelError(f"estimated_params for {name}: {rest[3].strip()!r} is not a prior shape this reader knows "
                             f"({', '.join(sorted(_PRIORS))}, in any case)")
        if shape_at != 0:  # initial value and bounds (then perhaps a prior)
            init, low, high = number(rest, 0, name), number(rest, 1, name), number(rest, 2, name)
            if init is not None:
                start[name] = init
            if low is not None or high is not None:
                bounds[name] = (-math.inf if low is None else low, math.inf if high is None else high)
                if name.startswith("corr_"):
                    bounds[name] = (max(bounds[name][0], -1.0), min(bounds[name][1], 1.0))
        if shape_at is not None:
            shape = _shape(rest[shape_at])
            spec = rest[shape_at + 1:]
            mean, sd, p3, p4 = (number(spec, i, name) for i in range(4))
            if shape == "Uniform":
                if p3 is not None and p4 is not None:
                    prior = P.Uniform(p3, p4)
                elif mean is not None and sd is not None:
                    prior = P.Uniform(mean - math.sqrt(3) * sd, mean + math.sqrt(3) * sd)
                else:
                    raise ModelError(f"uniform prior for {name}: give its mean and standard deviation, or its bounds")
            else:
                if mean is None or sd is None:
                    raise ModelError(f"the prior for {name} needs a mean and a standard deviation")
                if any(v is not None and not math.isnan(v) for v in (p3, p4)):
                    raise ModelError(f"generalised prior bounds (third and fourth prior parameters) for {name} are not supported yet")
                prior = getattr(P, shape)(mean, sd)
            priors[name] = prior
        names.append(name)
    return priors, names, start, bounds


def corr_name(a: str, b: str, shocks: list[str] | tuple[str, ...]) -> str:
    """The name of the correlation of two shocks as an estimated parameter: ``corr_<first>_<second>``, the two in the
    order of ``shocks``. Refused for a shock that is not one, or a shock with itself."""
    unknown = [s for s in (a, b) if s not in shocks]
    if unknown:
        raise ModelError(f"corr names {', '.join(unknown)}, which {'is not a shock' if len(unknown) == 1 else 'are not shocks'} "
                         f"(varexo: {', '.join(shocks)})")
    if a == b:
        raise ModelError(f"corr {a}, {b}: a correlation is between two different shocks (stderr {a} for its size)")
    first, second = sorted((a, b), key=list(shocks).index)
    return f"corr_{first}_{second}"


def _shock_corr(pairs: list[tuple[str, str, str, float]], exo: list[str], shock_sd: dict[str, float]) -> dict[tuple[str, str], float]:
    """The correlations of the shocks block: ``corr e1, e2 = r`` as given, ``var e1, e2 = c`` divided by the two
    standard deviations (as Dynare, wherever in the block they are given). Each within (-1, 1)."""
    out: dict[tuple[str, str], float] = {}
    for kind, a, b, value in pairs:
        corr_name(a, b, exo)  # refuses unknown shocks and a shock with itself
        first, second = sorted((a, b), key=exo.index)
        if kind == "var":
            sizes = [shock_sd.get(s) for s in (a, b)]
            if any(sd is None or sd <= 0 for sd in sizes):
                raise ModelError(f"var {a}, {b} = {value:g}: a covariance needs the variances of {a} and {b} in the shocks block")
            value = value / (sizes[0] * sizes[1])
        if not -1 < value < 1:
            raise ModelError(f"the correlation of {a} and {b} is {value:g}, outside (-1, 1)")
        out[(first, second)] = float(value)
    return out


def read_mat_data(path: str | Path) -> dict[str, Any]:
    """The numeric variables of a MATLAB ``.mat`` file (format 5 to 7.2, read by SciPy), vectors as 1-d arrays."""
    import numpy as np
    from scipy import io

    try:
        raw = io.loadmat(str(path))
    except NotImplementedError:
        raise ModelError(f"{Path(path).name} is a MATLAB 7.3 file (HDF5): save it with -v7 in MATLAB, or as CSV") from None
    out = {}
    for name, value in raw.items():
        if name.startswith("__") or not isinstance(value, np.ndarray) or value.dtype.kind not in "fiu":
            continue
        out[name] = value.ravel().astype(float) if 1 in value.shape or value.ndim == 1 else value.astype(float)
    return out


def read_m_data(path: str | Path) -> dict[str, Any]:
    """Variables of a simple MATLAB data file: matrix and vector literals, then assignments using column selection
    ``x(:,k)``, element-wise arithmetic, ``diff``, ``log``, ``exp``, ``sqrt`` and ``cumsum``. Vectors as 1-d arrays."""
    import numpy as np

    name_of_file = Path(path).name
    text = re.sub(r"%[^\n]*", "", Path(path).read_text(encoding="utf-8", errors="replace"))
    values: dict[str, Any] = {}
    funcs = {"diff": np.diff, "log": np.log, "exp": np.exp, "sqrt": np.sqrt, "cumsum": np.cumsum, "abs": np.abs}
    assign = re.compile(r"\s*([A-Za-z_]\w*)\s*=\s*")
    pos = 0
    while text[pos:].strip():
        m = assign.match(text, pos)
        if not m:
            snippet = text[pos:].strip().splitlines()[0][:40]
            raise ModelError(f"cannot read this MATLAB in {name_of_file}: {snippet!r} (only simple data files are read)")
        name, pos = m.group(1), m.end()
        if text[pos] == "[":
            end = text.index("]", pos)
            rows = [r for r in re.split(r"[;\n]", text[pos + 1:end]) if r.strip()]
            matrix = np.array([[float(v) for v in re.split(r"[\s,]+", r.strip())] for r in rows])
            values[name] = matrix.ravel() if 1 in matrix.shape else matrix
            if text[end + 1:end + 2] == "'":
                raise ModelError(f"cannot read this MATLAB in {name_of_file}: transposes ({name} = [...]')")
            pos = end + 1 + re.match(r"\s*;?", text[end + 1:]).end()
            continue
        stop = re.search(r"[;\n]", text[pos:])
        expr = text[pos:pos + stop.start()] if stop else text[pos:]
        pos += stop.end() if stop else len(text) - pos
        if "'" in expr:
            raise ModelError(f"cannot read this MATLAB in {name_of_file}: transposes in {expr.strip()!r}")
        py = re.sub(r"\b([A-Za-z_]\w*)\(\s*:\s*,\s*(\d+)\s*\)", lambda k: f"{k.group(1)}[:, {int(k.group(2)) - 1}]", expr)
        py = py.replace(".^", "**").replace("^", "**").replace("./", "/").replace(".*", "*")
        try:
            result = np.asarray(safe_eval(py, values, funcs, columns=True), dtype=float)  # checked: arithmetic on the file's arrays
        except Exception as exc:
            raise ModelError(f"cannot read this MATLAB in {name_of_file}: {expr.strip()!r} ({exc})") from None
        values[name] = result.ravel() if result.ndim == 2 and 1 in result.shape else result
    return values


def _statements(text: str) -> list[str]:
    return [" ".join(s.split()) for s in text.split(";")]


def _convert(raw: list[str], endo: list[str], exo: list[str], param_names: list[str]) -> list[tuple[str, str, list[str]]]:
    """Dynare equations to pymetrix syntax: timing, model-local variables, tags; as (lhs, rhs, parameters used)."""
    variables = sorted(endo + exo, key=len, reverse=True)
    timing = re.compile(r"\b(" + "|".join(map(re.escape, variables)) + r")\(\s*([+-]?\d+)\s*\)") if variables else None
    local: dict[str, str] = {}
    out = []
    for eq in raw:
        eq = re.sub(r"^\[[^\]]*\]\s*", "", eq.strip())  # [name='...'] tags
        if not eq:
            continue
        if timing is not None:
            eq = timing.sub(lambda m: f"{m.group(1)}[{int(m.group(2)):+d}]" if int(m.group(2)) else m.group(1), eq)
        for name, expr in local.items():
            eq = re.sub(rf"\b{re.escape(name)}\b", f"({expr})", eq)
        if eq.startswith("#"):
            name, _, expr = eq[1:].partition("=")
            local[name.strip()] = expr.strip()
            continue
        lhs, sep, rhs = eq.partition("=")
        if not sep:
            lhs, rhs = eq, "0"
        used = [p for p in param_names if re.search(rf"\b{re.escape(p)}\b", eq)]
        out.append((lhs.strip(), rhs.strip(), used))
    return out


def _current(expr: str, names: list[str]) -> set[str]:
    """Names appearing without a lag or lead."""
    return {n for n in names if re.search(rf"\b{re.escape(n)}\b(?!\s*\[)", expr)}


def _single_current(lhs: str, names: list[str]) -> str | None:
    found = _current(lhs, names)
    return next(iter(found)) if len(found) == 1 else None


def _match(equations: list[tuple[str, str, list[str]]], endo: list[str]) -> list[str]:
    """Pair each equation with an endogenous variable that appears in it unlagged, preferring its left-hand side."""
    naturals = [_single_current(lhs, endo) for lhs, _, _ in equations]
    taken = {v for v in naturals if v}
    candidates = []
    for (lhs, rhs, _), natural in zip(equations, naturals):
        # its own left-hand side first; then variables no other equation has on its left
        others = sorted(_current(lhs + " " + rhs, endo) - {natural}, key=lambda v: (v in taken, v))
        candidates.append(([natural] if natural else []) + others)
    owner: dict[str, int] = {}

    def assign(i: int, seen: set[str]) -> bool:
        for v in candidates[i]:
            if v in seen:
                continue
            seen.add(v)
            if v not in owner or assign(owner[v], seen):
                owner[v] = i
                return True
        return False

    for i in range(len(equations)):
        if not assign(i, set()):
            raise ModelError(f"equation {i + 1} ({equations[i][0]} = {equations[i][1]}) has no endogenous variable left to solve for")
    by_equation = {i: v for v, i in owner.items()}
    return [by_equation[i] for i in range(len(equations))]
