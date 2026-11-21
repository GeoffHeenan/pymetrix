"""TRAMO-SEATS through JDemetra+ 3, checked against other implementations.

- TRAMO's estimates: the airline model's MA parameters and the trading-day and leap-year coefficients equal
  statsmodels' exact maximum likelihood on the same (differenced) regARIMA model.
- SEATS: the adjusted series equals the SEATS in the US Census Bureau's x13as, given the same model (TRAMO's estimates,
  fixed), monthly and quarterly, in logs and levels. Two implementations: they agree to about 2e-4 (relative), most
  closely mid-sample.
- Each setting reaches JDemetra+'s specification; outliers planted are found where they were put.
- Eurostat's own R interface to JDemetra+ (rjd3tramoseats), on five of R's datasets with each predefined
  specification: the same adjusted series, model, logs and outliers. The same JDemetra+ code, so this checks pymetrix's
  bridge (the series' dates, the specification, what is read back), not the method. R's values come from
  fixtures/reference/make_rjd3tramoseats.R: the CI workflow jdemetra-reference.yml runs it weekly and on pull requests
  that touch seasonal adjustment, then these tests with ``RJD3_REFERENCE`` naming its output. Elsewhere they run when
  that file, or a copy saved as fixtures/reference/rjd3tramoseats.json, is there.

Skipped without JPype, Java 21 or the JDemetra+ jars (``JDPLUS_PATH`` or ``pymetrix_seasonal.jdemetra.install()``); the SEATS
comparison also needs x13as (``X13PATH``)."""

import hashlib
import io
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pymetrix_seasonal import TramoSeats, jdemetra, x13_program

from pymetrix import Dataset
from pymetrix.errors import SpecificationError
from pymetrix.forms import build
from pymetrix.plugins import registry
from pymetrix.seasonal import calendar_effects

RJD3 = Path(os.environ.get("RJD3_REFERENCE") or Path(__file__).parent / "fixtures" / "reference" / "rjd3tramoseats.json")
_RJD3 = json.loads(RJD3.read_text(encoding="utf-8")) if RJD3.is_file() else {"cases": [], "series": {}}


def _java():
    try:
        return jdemetra.classes()
    except SpecificationError as exc:
        pytest.skip(str(exc))


def _airline(freq, n, seed, sd=0.02):
    """A series from (1 - 0.4B)(1 - 0.6B^s) innovations, integrated by (1 - B)(1 - B^s), plus a seasonal pattern."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sd, n + freq + 1)
    a = e[freq + 1:] - 0.4 * e[freq:-1] - 0.6 * e[1:-freq] + 0.24 * e[:-freq - 1]
    z = np.zeros(n + freq + 1)
    for t in range(freq + 1, n + freq + 1):
        z[t] = z[t - 1] + z[t - freq] - z[t - freq - 1] + a[t - freq - 1]
    return z[freq + 1:] + 0.1 * np.sin(2 * np.pi * np.arange(n) / freq)


def _run(J, y, method, freq=12, year=2001):
    import jpype

    ts = J("jdplus.toolkit.base.r.timeseries.TsUtility").of(freq, year, 1, jpype.JArray(jpype.JDouble)(np.asarray(y, float)))
    context = J("jdplus.toolkit.base.api.timeseries.regression.ModellingContext").getActiveContext()
    return J("jdplus.tramoseats.base.r.TramoSeats").process(ts, method.specification(J, freq), context)


def _get(J, result, key):
    return result.getData(key, J("java.lang.Object"))


def _twice_differenced(v, s):
    return (v[s + 1:] - v[s:-1]) - (v[1:-s] - v[:-s - 1])


def test_airline_parameters_match_statsmodels():
    J = _java()
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    x = _airline(12, 192, seed=1)
    result = _run(J, 100 * np.exp(x), TramoSeats(spec="rsa0", log=True))
    theirs = SARIMAX(_twice_differenced(x, 12), order=(0, 0, 1), seasonal_order=(0, 0, 1, 12), trend="c").fit(
        disp=False, method="bfgs", maxiter=500).params
    ours = [float(_get(J, result, k).getCoefficient()) for k in ("arima.theta(1)", "arima.btheta(1)")]
    assert np.allclose(ours, theirs[1:3], atol=2e-4)


def test_trading_day_and_leap_year_effects_match_statsmodels():
    J = _java()
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    idx = pd.period_range("2000-01", periods=192, freq="M")
    X = calendar_effects(idx, "six", easter=None, leap_year=True).to_numpy()
    beta = np.array([0.004, 0.002, 0.003, -0.001, 0.005, -0.006, 0.02])
    log_y = _airline(12, 192, seed=2, sd=0.01) + X @ beta
    result = _run(J, 100 * np.exp(log_y), TramoSeats(spec="rsa0", log=True, trading="trading"), year=2000)
    assert list(map(str, _get(J, result, "regression.details.description"))) == ["const", "mon", "tue", "wed", "thu", "fri", "sat", "Leap year"]
    ours = np.array(_get(J, result, "regression.details.coefficients"))[1:]
    Xd = np.column_stack([_twice_differenced(X[:, j], 12) for j in range(7)])
    theirs = SARIMAX(_twice_differenced(log_y, 12), exog=Xd, order=(0, 0, 1), seasonal_order=(0, 0, 1, 12), trend="c").fit(
        disp=False, method="bfgs", maxiter=1000).params[1:8]
    assert np.allclose(ours, theirs, atol=1e-5)
    assert np.allclose(ours[:6], beta[:6], atol=0.002) and abs(ours[6] - beta[6]) < 0.01  # and near the truth


@pytest.mark.parametrize("freq, log", [(12, True), (4, True), (12, False), (4, False)])
def test_seats_matches_x13as_seats_on_the_same_model(freq, log, tmp_path):
    J = _java()
    folder = x13_program()
    if folder is None:
        pytest.skip("x13as is not installed (set X13PATH)")
    n = 160 if freq == 12 else 100
    x = _airline(freq, n, seed=4)
    y = 100 * np.exp(x) if log else 100 + 50 * x
    idx = pd.period_range("2001-01" if freq == 12 else "2001Q1", periods=n, freq="M" if freq == 12 else "Q", name="time")
    method = TramoSeats(spec="rsa0", log=log)
    result = _run(J, y, method, freq)
    theta, btheta = (float(_get(J, result, k).getCoefficient()) for k in ("arima.theta(1)", "arima.btheta(1)"))
    mean = float(np.array(_get(J, result, "regression.details.coefficients"))[0])
    out = Dataset.from_frame(pd.DataFrame({"y": y}, index=idx)).seasonal_adjust("y", method, derived=False)
    ours = out["y_sa"].to_numpy()
    values = "\n".join(repr(float(v)) for v in y)
    (tmp_path / "h.spc").write_text(
        f"series{{ start=2001.1 period={freq} data=(\n{values}\n) }}\ntransform{{function={'log' if log else 'none'}}}\n"
        f"regression{{variables=(const) b=({mean!r}f)}}\narima{{model=(0 1 1)(0 1 1) ma=({-theta!r}f {-btheta!r}f)}}\n"
        "estimate{}\nseats{save=(s11) qmax=1000}\n")  # qmax: keep the model given (SEATS re-estimates a poor fit)
    exe = next(p for p in (f"{folder}/x13as", f"{folder}/x13as.exe") if __import__("os").path.isfile(p))
    subprocess.run([exe, "h"], cwd=tmp_path, capture_output=True, timeout=300)
    theirs = np.array([float(line.split()[1]) for line in (tmp_path / "h.s11").read_text().splitlines()[2:] if line.strip()])
    assert np.allclose(ours, theirs, rtol=5e-4)
    middle = slice(2 * freq, -2 * freq)
    assert np.allclose(ours[middle], theirs[middle], rtol=2e-4)
    factor = out["y_seasonal"].to_numpy()
    assert np.allclose(factor, y / ours if log else y - ours)
    extra = out.variables["y_sa"].extra
    assert extra["tramo_model"] == "(0 1 1)(0 1 1)" and extra["tramo_log"] == ("logs" if log else "levels")
    assert extra["backend"].startswith("TRAMO-SEATS (JDemetra+ 3.9.0")


def test_planted_outliers_are_found_and_recorded():
    _java()
    rng = np.random.default_rng(3)
    n = 180
    y = 100 * np.exp(0.003 * np.arange(n)) * (1 + 0.08 * np.sin(2 * np.pi * np.arange(n) / 12)) * np.exp(rng.normal(0, 0.012, n))
    y[60] *= 1.15
    y[100:] *= 1.1
    data = Dataset.from_frame(pd.DataFrame({"y": y}, index=pd.period_range("2005-01", periods=n, freq="M", name="time")))
    extra = data.seasonal_adjust("y", TramoSeats(), derived=False).variables["y_sa"].extra
    assert extra["tramo_outliers"] == "AO (2010-01), LS (2013-05)" and extra["backend_options"] == "spec=rsafull"
    for method in (TramoSeats(outlier=False), TramoSeats(spec="rsa0")):
        assert data.seasonal_adjust("y", method, derived=False).variables["y_sa"].extra["tramo_outliers"] == "not looked for"
    derived = data.seasonal_adjust("y", TramoSeats(spec="rsa3"))  # a derived series, recomputed from its recipe
    assert np.isfinite(derived["y_sa"].to_numpy()).all()


def test_each_setting_reaches_the_specification():
    J = _java()
    spec = TramoSeats(spec="rsa3", log=False, outlier_types="ao ls", critical=3.5, trading="working", easter=True, easter_days=8,
                      arima="(2 1 0)(0 1 1)", approximation="noisy", seats_method="kalman", trend_boundary=0.6,
                      seasonal_boundary=0.7, seasonal_boundary_pi=0.75, seasonal_tolerance=3.0, xl=0.97).specification(J, 12)
    tramo, seats = spec.getTramo(), spec.getSeats()
    assert str(tramo.getTransform().getFunction()) == "None"
    o = tramo.getOutliers()
    assert (o.isAo(), o.isLs(), o.isTc(), o.isSo(), o.getCriticalValue()) == (True, True, False, False, 3.5)
    a = tramo.getArima()
    assert (a.getP(), a.getD(), a.getQ(), a.getBp(), a.getBd(), a.getBq()) == (2, 1, 0, 0, 1, 1) and not tramo.isUsingAutoModel()
    cal = tramo.getRegression().getCalendar()
    assert str(cal.getTradingDays().getTradingDaysType()) == "TD2" and str(cal.getTradingDays().getRegressionTestType()) == "Separate_T"
    assert cal.getEaster().isUsed() and cal.getEaster().getDuration() == 8
    assert (str(seats.getApproximationMode()), str(seats.getMethod())) == ("Noisy", "KalmanSmoother")
    assert (seats.getTrendBoundary(), seats.getSeasBoundary(), seats.getSeasBoundaryAtPi(), seats.getSeasTolerance(),
            seats.getXlBoundary()) == (0.6, 0.7, 0.75, 3.0, 0.97)
    plain = TramoSeats().specification(J, 12)  # the defaults: RSAfull as JDemetra+ defines it
    assert plain.equals(J("jdplus.tramoseats.base.api.tramoseats.TramoSeatsSpec").RSAfull)
    off = TramoSeats(spec="rsa5", trading="none", easter=False).specification(J, 12).getTramo().getRegression().getCalendar()
    assert not off.getTradingDays().isUsed() and not off.getEaster().isUsed()
    auto = TramoSeats(spec="rsa3", trading="auto").specification(J, 12).getTramo().getRegression().getCalendar().getTradingDays()
    assert auto.isAutomatic() and str(auto.getAutomaticMethod()) == "FTEST"


def test_refusals():
    for bad, message in (({"spec": "rsa9"}, "spec is one of"), ({"trading": "td"}, "trading is"), ({"arima": "(0 1 1)"}, "arima is"),
                         ({"outlier_types": "ao xx"}, "outlier_types"), ({"outlier": False, "critical": 3.0}, "tick outliers")):
        with pytest.raises(SpecificationError, match=message):
            TramoSeats(**bad)
    J = _java()
    with pytest.raises(SpecificationError, match="RSA0 detects no outliers"):
        TramoSeats(spec="rsa0", critical=3.0).specification(J, 12)
    with pytest.raises(SpecificationError, match="no Easter effect"):
        TramoSeats(spec="rsa0", easter_days=8).specification(J, 12)
    constant = Dataset.from_frame(pd.DataFrame({"y": np.ones(48)}, index=pd.period_range("2005-01", periods=48, freq="M", name="time")))
    with pytest.raises(ValueError, match="TRAMO-SEATS could not adjust y: JDemetra\\+ returned no result"):
        constant.seasonal_adjust("y", TramoSeats(), derived=False)


def test_settings_from_the_dialog():
    method = build(registry("seasonal").get("TramoSeats"), "spec='rsa4', trading='trading', easter=True, critical=4.0")
    assert (method.spec, method.trading, method.easter, method.critical) == ("rsa4", "trading", True, 4.0)
    assert method.describe() == "TRAMO-SEATS (RSA4, trading=trading, easter=True, critical=4.0)"


def test_missing_pieces_are_explained(monkeypatch, tmp_path):
    monkeypatch.setenv("PYMETRIX_HOME", str(tmp_path))
    monkeypatch.delenv("JDPLUS_PATH", raising=False)
    assert jdemetra.jar_folder() is None
    with pytest.raises(SpecificationError, match="TRAMO-SEATS .*pymetrix_seasonal.jdemetra.install\\(\\), or JDPLUS_PATH"):  # JPype or the jars missing
        jdemetra.classes()


def test_install_checks_each_download(monkeypatch, tmp_path):
    good, bad = b"a jar", b"a tampered jar"
    monkeypatch.setattr(jdemetra, "JARS", {"x.jar": ("g/x/1", hashlib.sha1(good).hexdigest())})
    served = {"body": good}
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout: io.BytesIO(served["body"]))
    folder = jdemetra.install(str(tmp_path / "jars"))
    assert (tmp_path / "jars" / "x.jar").read_bytes() == good and jdemetra.jar_folder(folder) == folder
    served["body"] = bad
    with pytest.raises(SpecificationError, match="does not match its recorded SHA-1"):
        jdemetra.install(str(tmp_path / "other"))
    assert not (tmp_path / "other" / "x.jar").exists() and not list((tmp_path / "other").glob("*.part"))


def test_install_waits_when_maven_central_is_busy(monkeypatch, tmp_path):
    import urllib.error

    good = b"a jar"
    monkeypatch.setattr(jdemetra, "JARS", {"x.jar": ("g/x/1", hashlib.sha1(good).hexdigest())})
    answers = [429, 503, None]
    waits = []

    def serve(url, timeout):
        code = answers.pop(0)
        if code:
            raise urllib.error.HTTPError(url, code, "busy", {}, None)
        return io.BytesIO(good)

    monkeypatch.setattr("urllib.request.urlopen", serve)
    monkeypatch.setattr(jdemetra.time, "sleep", waits.append)
    assert jdemetra.jar_folder(jdemetra.install(str(tmp_path / "jars")))
    assert waits == [5, 10]
    answers[:] = [404]
    with pytest.raises(SpecificationError, match="could not be fetched from Maven Central .*404"):
        jdemetra.install(str(tmp_path / "missing"))


@pytest.mark.parametrize("case", _RJD3["cases"], ids=lambda c: f"{c['series']}-{c['spec']}")
def test_rjd3tramoseats_gives_the_same_adjustment(case):
    _java()
    series = _RJD3["series"][case["series"]]
    period, (year, sub) = series["period"], series["start"]
    start = f"{year}-{sub:02d}" if period == 12 else f"{year}Q{sub}"
    index = pd.period_range(start, periods=len(series["values"]), freq="M" if period == 12 else "Q", name="time")
    out = Dataset.from_frame(pd.DataFrame({"y": series["values"]}, index=index)).seasonal_adjust(
        "y", TramoSeats(spec=case["spec"]), derived=False)
    extra = out.variables["y_sa"].extra
    assert extra["tramo_model"] == "({} {} {})({} {} {})".format(*case["model"])
    assert extra["tramo_log"] == ("logs" if case["log"] else "levels")
    regressors = [case["regressors"]] if isinstance(case["regressors"], str) else case["regressors"]
    found = [r for r in regressors if re.match(r"(AO|LS|TC|SO) \(", r)]
    assert extra["tramo_outliers"] in ((", ".join(found),) if found else ("none", "not looked for"))
    ours, theirs = out["y_sa"].to_numpy(), np.asarray(case["sa"], dtype=float)
    gap = float(np.max(np.abs(ours - theirs) / np.abs(theirs)))
    assert gap < 1e-8, f"adjusted series differ by {gap:.2e} (relative), with {_RJD3.get('packages')}"


def test_shutdown_stops_java_only_if_it_was_started(monkeypatch):
    import sys
    import types

    monkeypatch.delitem(sys.modules, "jpype", raising=False)
    jdemetra.shutdown()  # JPype never imported: nothing to do, and it is not imported now
    assert "jpype" not in sys.modules
    stopped = []
    for started in (False, True):
        fake = types.SimpleNamespace(isJVMStarted=lambda s=started: s, shutdownJVM=lambda: stopped.append(1))
        monkeypatch.setitem(sys.modules, "jpype", fake)
        jdemetra.shutdown()
    assert stopped == [1]


def test_the_app_stops_java_when_its_loop_ends(monkeypatch):
    pytest.importorskip("PySide6")
    from pymetrix.ui import qt

    order = []
    monkeypatch.setattr(jdemetra, "shutdown", lambda: order.append("java stopped"))
    app = type("App", (), {"exec": lambda self: order.append("loop ended") or 3})()
    assert qt._run(app) == 3 and order == ["loop ended", "java stopped"]


def test_java_leaves_the_main_thread_its_whole_stack(tmp_path):
    """Once TRAMO-SEATS has started Java, the main thread can still go as deep as before. On Linux, Java started from
    the main thread takes that thread's stack to be no larger than -Xss (1 MB by default) and puts its guard pages
    there: Python calls through C went about 1,000 deep rather than 11,000, and an overflow ended the process with no
    report at all. Run in a process of its own, since an overflow would end it."""
    import sys

    if not sys.platform.startswith("linux"):
        pytest.skip("Java cuts the main thread's stack on Linux only")
    import resource

    soft, _ = resource.getrlimit(resource.RLIMIT_STACK)
    if soft != resource.RLIM_INFINITY and soft < 6 * 1024 * 1024:
        pytest.skip(f"a main-thread stack of {soft // 1024} KB is too small to tell")
    code = (
        "import os, sys\n"
        "from pymetrix.errors import SpecificationError\n"
        "import pymetrix_seasonal.jdemetra as jdemetra\n"
        "try:\n"
        "    jdemetra.classes()\n"
        "except SpecificationError as exc:\n"
        "    print(exc); sys.exit(77)\n"
        "sys.setrecursionlimit(100_000)\n"
        "def down(n):\n"
        "    return n if n >= 4000 else list(map(lambda _: down(n + 1), [0]))[0]\n"  # each level through C (map)
        "print(down(0))\n"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    if done.returncode == 77:
        pytest.skip(done.stdout.strip())
    assert done.returncode == 0 and done.stdout.split()[-1] == "4000", (done.returncode, done.stdout[-500:], done.stderr[-2000:])


def test_the_java_stack_size_is_the_main_threads(monkeypatch):
    import sys

    resource = pytest.importorskip("resource")  # not on Windows

    monkeypatch.setattr(sys, "platform", "linux")
    for limit, option in ((8 * 1024 * 1024, "-Xss8192k"), (16 * 1024 * 1024, "-Xss16384k"),
                          (resource.RLIM_INFINITY, "-Xss8192k"), (512 * 1024, "-Xss1024k")):
        monkeypatch.setattr(resource, "getrlimit", lambda which, limit=limit: (limit, resource.RLIM_INFINITY))
        assert jdemetra._jvm_options() == [option]
    monkeypatch.setattr(sys, "platform", "win32")
    assert jdemetra._jvm_options() == []  # Java does not cut the main thread there
