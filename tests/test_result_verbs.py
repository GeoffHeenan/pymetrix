"""Generic verbs and tidy tables on every estimate (plan item 12), and the record a saved result keeps (item 7)."""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymetrix import VAR, Dataset, Equation, System, SUR, load_result, save_result
from pymetrix.errors import NotSupportedError
from pymetrix.results import TABLES, VERBS

sys.path.insert(0, str(Path(__file__).parent))
from _gallery import BUILDERS  # noqa: E402

ESTIMATES = sorted(n for n in BUILDERS if n.startswith("Fitted") or n in ("RollingFit", "QuantileProcess"))
COEFFICIENT_COLUMNS = ["term", "estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high"]


@pytest.fixture(scope="module")
def built():
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in ESTIMATES:
            out[name] = BUILDERS[name]()
    return out


@pytest.mark.parametrize("name", ESTIMATES)
def test_every_estimate_has_the_verbs_or_says_what_to_use(built, name):
    import dataclasses

    obj = built[name]
    fields = {f.name for f in dataclasses.fields(obj)} if dataclasses.is_dataclass(obj) else set()
    for verb in (*VERBS, *TABLES, "to_record"):
        assert verb in fields or hasattr(type(obj), verb), f"{name} has no {verb}"
    for verb in ("params", "resid", "fitted"):
        try:
            getattr(obj, verb)
        except NotSupportedError as exc:
            assert isinstance(exc, AttributeError) and name in str(exc)
    try:
        obj.predict()
    except NotSupportedError as exc:
        assert "has no predict" in str(exc)
    except Exception:  # noqa: BLE001 - a real predict that needs arguments
        pass


@pytest.mark.parametrize("name", ESTIMATES)
def test_tidy_tables(built, name):
    obj = built[name]
    coefs = obj.coefficients()
    if len(coefs) and "std_error" in coefs.columns:
        assert [c for c in coefs.columns if c != "equation"] == COEFFICIENT_COLUMNS
        assert np.allclose(coefs["estimate"].to_numpy(float), np.asarray(obj.params.values, float), equal_nan=True)
    stats = obj.statistics()
    assert list(stats.columns) == ["statistic", "value", "definition"]
    try:
        obs = obj.observations()
    except NotSupportedError:
        return
    assert {"actual", "fitted", "resid"} <= set(obs.columns)


@pytest.mark.parametrize("name", ESTIMATES)
def test_the_record_is_json_and_survives_saving(built, name, tmp_path):
    obj = built[name]
    record = obj.to_record()
    json.dumps(record, allow_nan=True)  # no library objects
    assert record["kind"] == type(obj).__name__
    try:
        path = save_result(obj, tmp_path / name)
    except Exception:  # noqa: BLE001 - results without views are not saved this way
        pytest.skip(f"{name} is not saved as a single result")
    saved = load_result(path)
    if "coefficients" in record and len(obj.coefficients()):
        pd.testing.assert_frame_equal(saved.coefficients().reset_index(drop=True), obj.coefficients().reset_index(drop=True),
                                      check_dtype=False)
    if "observations" in record:
        live, back = obj.observations(), saved.observations()
        assert back.index.equals(live.index) and isinstance(back.index.get_level_values(-1), pd.PeriodIndex)
        assert np.allclose(back["resid"].to_numpy(float), live["resid"].to_numpy(float), equal_nan=True)


def test_inference_follows_the_results_own_distribution():
    from scipy import stats as sps

    d = Dataset.example()
    fit = Equation("c ~ y").fit(d)
    coefs = fit.coefficients()
    t = coefs["estimate"] / coefs["std_error"]
    assert np.allclose(coefs["p_value"], 2 * sps.t(fit.stats["df_resid"]).sf(np.abs(t)))
    q = sps.t(fit.stats["df_resid"]).ppf(0.975)
    assert np.allclose(coefs["ci_low"], coefs["estimate"] - q * coefs["std_error"])
    narrow = fit.coefficients(alpha=0.5)
    assert (narrow["ci_high"] - narrow["ci_low"] < coefs["ci_high"] - coefs["ci_low"]).all()


def test_multi_equation_results_name_the_equation():
    d = Dataset.example()
    var = VAR(["c", "y"], lags=1).fit(d)
    coefs = var.coefficients()
    assert list(coefs.columns[:2]) == ["equation", "term"] and set(coefs["equation"]) == {"c", "y"}
    assert set(coefs.loc[coefs["equation"] == "c", "term"]) == {"const", "c[-1]", "y[-1]"}
    obs = var.observations()
    assert set(obs["equation"]) == {"c", "y"} and np.allclose(obs["actual"], obs["fitted"] + obs["resid"])
    system = System([Equation("c ~ y"), Equation("i ~ y")]).fit(d, method=SUR())
    assert set(system.coefficients()["equation"]) == {"c", "i"}
    var.plot()  # actual and fitted, one panel per equation


def test_statistics_carry_their_definitions():
    fit = Equation("c ~ y").fit(Dataset.example())
    table = fit.statistics().set_index("statistic")
    assert table.loc["aic", "definition"] == "aic.total" and table.loc["aic", "value"] == fit.stats["aic"]


def test_saved_results_from_earlier_versions_say_they_have_no_record(tmp_path):
    import zipfile

    fit = Equation("c ~ y").fit(Dataset.example())
    path = save_result(fit, tmp_path / "old")
    stripped = tmp_path / "older.pymxr"
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(stripped, "w") as dst:
        for item in src.namelist():
            if item != "record.json":
                dst.writestr(item, src.read(item))
    with pytest.raises(NotSupportedError, match="saved before pymetrix recorded them"):
        load_result(stripped).coefficients()
