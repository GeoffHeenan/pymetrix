"""The bundle reader against bundles written by hand to the format (docs/design/eviews-plugin.md, section 4), not by
the reader or by EViews: every value and every metadata field is checked against the literal fixture, and what does
not read must be refused with its place.

fixtures/macro is a workfile of four pages: quarterly (NA and empty fields, alpha series with an empty value and a
quoted comma, a value map, a label with a quote and a backslash, groups, samples, scalars, strings, a vector, a
matrix with a missing value, equations with their estimates, and a model), Annual, a panel (unbalanced) and an
undated cross-section with names that are Python keywords. The equations' "EViews estimates" were computed with
NumPy's least squares on the literal data (test_the_fixture_estimates_are_least_squares checks it), so
compare_estimates is checked against an independent computation, not against itself.
"""

import json
import math
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymetrix_eviews as pe
from pymetrix import Dataset, Matrix, Sample, Selection
from pymetrix.errors import DataError, NotFoundError

FIXTURES = Path(__file__).parent / "fixtures"
MACRO = FIXTURES / "macro" / "macro.evbundle"

INC = [100.5, 101.25, 102.125, 103.0625, 104.03125, 105.015625, 106.0078125, 107.00390625, 108.001953125, 109.0009765625,
       110.00048828125, 111.000244140625]
CONS = [80.1, 81.3, 81.9, 82.8, 83.95, 84.7, 85.85, 86.6, 87.45, 88.5, 89.2, 90.15]


def _read(path=MACRO, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = Dataset.read(str(path), **kw)
    return data, [str(w.message) for w in caught if issubclass(w.category, pe.EViewsBundleNote)]


def _same(values, expected):
    got = list(values)
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        if e is None:
            assert g is None or g is pd.NA or (isinstance(g, float) and math.isnan(g)), (got, expected)
        else:
            assert g == e, (got, expected)


# -- the schema ----------------------------------------------------------------------------------------------------


def test_every_well_formed_fixture_matches_the_schema_and_the_newer_one_does_not():
    jsonschema = pytest.importorskip("jsonschema")
    schema = pe.bundle_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    for name, encoding in (("macro/macro", "utf-8"), ("missing_column/missing", "utf-8"), ("semiannual/semi", "utf-8"),
                           ("weekly/weekly", "utf-8"), ("cp1252/fr", "cp1252")):
        manifest = json.loads((FIXTURES / f"{name}.evbundle").read_bytes().decode(encoding))
        assert not list(validator.iter_errors(manifest)), name
    newer = json.loads((FIXTURES / "newer" / "newer.evbundle").read_text())
    assert any(e.path and list(e.path) == ["format_version"] for e in validator.iter_errors(newer))
    bad = json.loads(MACRO.read_text())
    bad["pages"][0]["series"][0]["kind"] = "text"
    bad["pages"][2].pop("id_columns")
    bad["pages"][0]["file"] = "../outside.csv"
    places = {tuple(e.path) for e in validator.iter_errors(bad)}
    assert ("pages", 0, "series", 0, "kind") in places and ("pages", 0, "file") in places
    assert any(e.validator == "required" and "id_columns" in e.message for e in validator.iter_errors(bad))


def test_the_program_version_and_the_schema_agree_with_the_reader():
    schema = pe.bundle_schema()
    assert schema["properties"]["format"]["const"] == pe.FORMAT
    assert schema["properties"]["format_version"]["const"] == max(pe.FORMAT_VERSIONS) == 1
    assert '"program_version": "1.0"' in pe.export_program() and pe.PROGRAM_VERSION == "1.0"


# -- the pages -----------------------------------------------------------------------------------------------------


def test_bundle_pages_lists_each_page_without_reading_its_data():
    assert pe.bundle_pages(MACRO) == [
        {"name": "quarterly", "structure": "dated", "frequency": "Q", "series": 6, "refused": None},
        {"name": "Annual", "structure": "dated", "frequency": "A", "series": 2, "refused": None},
        {"name": "panel", "structure": "panel", "frequency": "A", "series": 2, "refused": None},
        {"name": "cross", "structure": "undated", "frequency": None, "series": 4, "refused": None}]
    assert pe.bundle_pages(FIXTURES / "macro") == pe.bundle_pages(MACRO)  # the folder finds its manifest
    assert pe.bundle_pages(FIXTURES / "semiannual" / "semi.evbundle")[0]["refused"].startswith("semi-annual")


def test_a_dated_page_every_value():
    q, notes = _read(page="quarterly")
    assert notes == []
    assert [type(a).__name__ for a in q.axes] == ["TimeAxis"] and q.time_axis.name == "time"
    assert list(q.frame.index) == list(pd.period_range("1980Q1", "1982Q4", freq="Q"))
    assert list(q.frame.columns) == ["gdp", "cons", "inc", "inv", "sector", "region"]
    _same(q.frame["gdp"], [1000.1234567890123, 1010.5, None, 1031.25, 1040.0, 1052.75, None, 1070.0, 1081.5, 1090.25, 1101.0,
                           1112.125])  # NA and an empty field are both missing; 17 digits kept; 1.112125E3 read
    _same(q.frame["cons"], CONS)
    _same(q.frame["inc"], INC)
    _same(q.frame["inv"], [20.25, 20.5, 20.75, 21.0, 21.25, 21.5, 21.75, 22.0, 22.25, 22.5, 22.75, 23.0])
    _same(q.frame["sector"], [1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 3.0, 1.0, 2.0])
    assert str(q.frame["region"].dtype) == "string"
    _same(q.frame["region"], ["north", "south", None, "east", "north", "west, upper", 'say "hi"', "south", "NA", "east", "north",
                              "south"])  # an empty alpha value is missing; the text NA is text


def test_a_dated_page_every_metadata_field():
    q, _ = _read(page="quarterly")
    v = q.variables
    assert (v["gdp"].label, v["gdp"].description) == ('Real GDP, "chained" at C:\\data', 'Real GDP, "chained" at C:\\data')
    assert (v["gdp"].units, v["gdp"].source, v["gdp"].source_key) == ("bn 2017 prices", "Statistics Office", "gdp")
    assert dict(v["gdp"].extra) == {"remarks": "seasonally adjusted", "attributes": {"base_year": "2017"}}
    assert (v["cons"].label, v["cons"].description, v["cons"].units) == ("Consumption", "Household consumption", "bn")
    assert dict(v["cons"].extra) == {"display_name": "Consumption"}  # the label is the display name; both kept
    assert (v["inc"].label, v["inc"].description, v["inc"].units, dict(v["inc"].extra)) == ("", "", "", {})
    assert (v["inv"].label, v["inv"].units, dict(v["inv"].extra)) == ("", "", {})
    assert v["sector"].label == "Sector"
    assert dict(v["sector"].extra) == {"value_map": {"name": "vm_sector", "labels": {"1": "Manufacturing", "2": "Services, private",
                                                                                      "3": "Government"}}}
    assert (v["region"].label, v["region"].description) == ("Region (état)", "Region (état)")
    assert dict(v["region"].extra) == {"eviews_kind": "alpha", "remarks": "line one\nline two"}
    assert all(var.retrieved for var in v.values())
    assert q.meta.title == "EViews macro: page quarterly"
    loads = q.meta.extra["loads"]
    assert loads[-1]["source"] == str(MACRO) and loads[-1]["options"] == {"page": "quarterly"}
    assert q.meta.vintage["policy"] == "file"


def test_a_page_keeps_its_eviews_objects_in_meta():
    q, _ = _read(page="quarterly")
    ev = q.meta.extra["eviews"]
    assert ev["format_version"] == 1 and ev["page"] == "quarterly" and ev["structure"] == "dated" and ev["frequency"] == "Q"
    assert ev["exporter"] == {"program_version": "1.0", "eviews_version": "13.0", "exported": "2026-11-05T14:03:00"}
    assert ev["workfile"] == {"name": "macro", "path": "C:\\work\\macro.wf1", "active_page": "quarterly"}
    assert ev["bundle"] == str(MACRO)
    assert (ev["range"], ev["sample"]) == (["1980Q1", "1982Q4"], "1980q1 1982q2")
    assert ev["groups"] == {"g1": ["gdp", "cons"], "g2": ["gdp", "log(cons)"]}
    assert ev["samples"] == {"pre": "1980q1 1981q4", "whole": "@all", "late": "1981:3 @last", "rich": "1980q1 1982q4 if inc > 105"}
    assert ev["scalars"] == {"alpha": 0.3, "unset": None}
    assert ev["strings"] == {"region_s": "north"}
    assert ev["arrays"] == {"w": {"type": "vector", "rows": 3, "cols": 1, "values": [[0.25], [0.5], [0.25]]},
                            "m2": {"type": "matrix", "rows": 2, "cols": 3, "values": [[1.5, None, 3.0], [4.0, 5.25, 6.0]]}}
    est = [-13.872477213762297, 0.938413540386753]
    se = [1.5768404322686491, 0.014927073024990207]
    assert ev["equations"] == [
        {"name": "eq_cons", "command": "ls cons c inc", "method": "LS", "sample": "1980q1 1982q4", "ncoef": 2,
         "coefficients": ["c(1)", "c(2)"], "estimates": est, "std_errors": se},
        {"name": "eq_rev", "command": "ls cons inc c", "method": "LS", "sample": "1980q1 1982q4", "ncoef": 2,
         "coefficients": ["c(1)", "c(2)"], "estimates": est[::-1], "std_errors": se[::-1]},
        {"name": "eq_white", "command": "ls(cov=white) cons c inc", "method": "LS", "sample": "1980q1 1982q4", "ncoef": 2,
         "coefficients": ["c(1)", "c(2)"], "estimates": est, "std_errors": [1.6, 0.015]},
        {"name": "eq_iv", "command": "tsls cons c inc @ c inv", "method": "TSLS", "sample": "1980q1 1982q4", "ncoef": None}]
    assert ev["models"] == {"m": "' the consumption model\n:EQ_CONS\n@IDENTITY Y = CONS + INV\n"}
    assert ev["skipped"] == [{"name": "gr1", "type": "graph", "reason": "objects of this type are not exported"}]
    assert (ev["renamed"], ev["notes"], ev["unknown"]) == ({}, [], {})
    json.dumps(ev)  # plain JSON values: kept in Parquet files and projects


def test_an_annual_page_in_the_same_workfile():
    a, notes = _read(page="annual")  # pages by name in any case, as in EViews
    assert notes == [] and a.time_axis.freq.startswith("Y")
    assert list(a.frame.index) == list(pd.period_range("1980", "1983", freq="Y"))
    _same(a.frame["gdp_a"], [4100.5, 4200.75, None, 4400.0])
    _same(a.frame["pop"], [10.25, 10.5, 10.75, 11.0])
    assert (a.variables["gdp_a"].units, a.variables["pop"].units) == ("bn", "million")
    assert a.meta.extra["eviews"]["page"] == "Annual" and a.meta.extra["eviews"]["sample"] == "@all"
    assert a.meta.extra["eviews"]["groups"] == {} and a.meta.extra["eviews"]["equations"] == []


def test_a_panel_page_unbalanced():
    p, notes = _read(page="panel")
    assert notes == []
    assert [a.name for a in p.axes] == ["crossid", "time"] and p.time_axis.freq.startswith("Y")
    assert list(p.frame.index) == [(1, pd.Period("2000", "Y")), (1, pd.Period("2001", "Y")), (1, pd.Period("2002", "Y")),
                                   (2, pd.Period("2000", "Y")), (2, pd.Period("2002", "Y"))]
    _same(p.frame["y"], [1.5, 2.5, None, 4.25, 5.75])
    _same(p.frame["x"], [10.0, 11.0, 12.0, 20.0, None])


def test_an_undated_page_with_names_that_are_python_keywords():
    c, notes = _read(page="cross")
    assert notes == ["names changed to be valid Python names: in -> in_v, Lambda -> lambda_v"]
    assert [a.name for a in c.axes] == ["obs"] and list(c.frame.index) == [1, 2, 3, 4]
    assert list(c.frame.columns) == ["wage", "in_v", "lambda_v", "sex"]
    _same(c.frame["wage"], [12.5, 15.75, None, 9.0])
    _same(c.frame["in_v"], [1.0, 0.0, 1.0, 0.0])
    _same(c.frame["lambda_v"], [0.5, 0.25, 0.125, 0.001])
    _same(c.frame["sex"], ["f", "m", None, "f"])
    assert dict(c.variables["in_v"].extra) == {"eviews_name": "in"} and c.variables["lambda_v"].source_key == "Lambda"
    assert c.meta.extra["eviews"]["renamed"] == {"in": "in_v", "Lambda": "lambda_v"}


def test_weekly_and_daily_pages_and_what_is_noted():
    w, notes = _read(FIXTURES / "weekly" / "weekly.evbundle", page="w")
    assert w.time_axis.freq == "W-SUN"  # weeks starting on Monday (the dates' weekday), as pandas anchors them
    assert [str(p) for p in w.frame.index] == ["2024-01-01/2024-01-07", "2024-01-08/2024-01-14", "2024-01-15/2024-01-21"]
    _same(w.frame["claims"], [210.5, 215.0, None])
    assert notes == ["the bundle was written by export program 0.9; this plugin ships 1.0 (Get the export program): export again "
                     "with it for what it adds",
                     "keys this reader does not know, kept in meta['eviews']['unknown']: colour, exporter.machine, pages[0].layout, "
                     "pages[0].series[0].seasonal"]
    assert w.meta.extra["eviews"]["unknown"] == {"colour": "blue", "exporter.machine": "pc-17", "pages[0].layout": {"x": 1},
                                                 "pages[0].series[0].seasonal": "sa"}
    d5, _ = _read(FIXTURES / "weekly" / "weekly.evbundle", page="d5")
    assert d5.time_axis.freq == "B" and [str(p) for p in d5.frame.index] == ["2024-01-04", "2024-01-05", "2024-01-08"]
    d7, _ = _read(FIXTURES / "weekly" / "weekly.evbundle", page="d7")
    assert d7.time_axis.freq == "D" and [str(p) for p in d7.frame.index] == ["2024-01-06", "2024-01-07"]


def test_a_file_not_in_utf8_falls_back_to_windows_1252_with_a_note():
    m, notes = _read(FIXTURES / "cp1252" / "fr.evbundle")  # one page: no page= needed
    assert notes == ["fr.evbundle is not UTF-8 (byte 317): read as Windows-1252",
                     "pages/m.csv is not UTF-8 (byte 42): read as Windows-1252"]
    assert (m.variables["prix"].description, m.variables["prix"].units) == ("Prix à la consommation", "€ de 2015")
    _same(m.frame["prix"], [100.5, 101.0])
    _same(m.frame["ville"], ["Orléans", "Saint-Étienne"])
    assert m.time_axis.freq == "M" and [str(p) for p in m.frame.index] == ["2020-01", "2020-02"]
    assert m.meta.extra["eviews"]["notes"] == notes


def test_semi_annual_is_refused_with_the_reason_and_reads_as_undated():
    path = FIXTURES / "semiannual" / "semi.evbundle"
    with pytest.raises(DataError, match=r"semi\.evbundle: pages\[0\] \(halves\) has frequency 'S', semi-annual: pandas has no "
                                        r"half-year period.*undated=True"):
        Dataset.read(str(path), page="halves")
    s, notes = _read(path, page="halves", undated=True)
    assert notes == ["page halves (S) read as undated: rows numbered from 1, its dates in the text series date"]
    assert [a.name for a in s.axes] == ["obs"] and list(s.frame.index) == [1, 2, 3]
    _same(s.frame["gdp"], [10.5, 11.25, 12.0])
    _same(s.frame["date"], ["1990-01-01", "1990-07-01", "1991-01-01"])
    assert s.variables["gdp"].units == "bn"


@pytest.mark.parametrize(("code", "why"), [("2A", "multi-year"), ("5Y", "multi-year"), ("H", "intraday"), ("4H", "intraday"),
                                          ("MIN", "intraday"), ("BM", "bimonthly"), ("F", "fortnightly"), ("T", "ten-day"),
                                          ("D", "custom week"), ("X", "not a frequency")])
def test_other_frequencies_pandas_has_no_period_for_are_refused(tmp_path, code, why):
    root = _copy(tmp_path, "semiannual")
    _edit(root / "semi.evbundle", lambda m: m["pages"][0].update(frequency=code))
    with pytest.raises(DataError, match=why):
        Dataset.read(str(root / "semi.evbundle"))


# -- the other objects ---------------------------------------------------------------------------------------------


def test_eviews_objects_as_pymetrix_objects():
    q, _ = _read(page="quarterly")
    objects = pe.eviews_objects(q)
    assert isinstance(objects["g1"], Selection) and list(objects["g1"].names) == ["gdp", "cons"]
    assert objects["pre"] == Sample(start="1980Q1", end="1981Q4") and objects["whole"] == Sample()
    assert objects["late"] == Sample(start="1981Q3")
    assert objects["alpha"] == 0.3 and math.isnan(objects["unset"]) and objects["region_s"] == "north"
    assert isinstance(objects["w"], Matrix) and objects["w"].values.tolist() == [[0.25], [0.5], [0.25]]
    np.testing.assert_array_equal(objects["m2"].values, [[1.5, np.nan, 3.0], [4.0, 5.25, 6.0]])
    assert objects["_left_out"] == {"g2": "group: log(cons) is not series of the page",
                                    "rich": "sample '1980q1 1982q4 if inc > 105': only @all and one pair of ends translate"}
    assert type(objects["m"]).__name__ == "Template"


@pytest.mark.parametrize(("text", "freq", "expected"), [
    ("@all", "Q", Sample()), ("1980q1 2007q4", "Q", Sample(start="1980Q1", end="2007Q4")), ("@first 2000q4", "Q", Sample(end="2000Q4")),
    ("1990:1 @last", "Q", Sample(start="1990Q1")), ("1990:3 1991:12", "M", Sample(start="1990-03", end="1991-12")),
    ("1990m3 1991m12", "M", Sample(start="1990-03", end="1991-12")), ("1990 2000", "A", Sample(start="1990", end="2000")),
    ("1990 2000", "Q", None), ("1990:5 1991:1", "Q", None), ("1980q1 1990q4 2000q1 2010q4", "Q", None), ("@all if x > 0", "Q", None)])
def test_eviews_samples(text, freq, expected):
    assert pe.eviews_sample(text, freq) == expected


def test_a_model_is_read_by_read_eviews_with_its_equation_fitted():
    q, _ = _read(page="quarterly")
    template = pe.bundle_model(q, "m")
    model = template.model
    assert len(model.equations) == 2
    fitted, identity = model.equations
    np.testing.assert_allclose(fitted.params.values, [-13.872477213762297, 0.938413540386753], rtol=1e-9)
    assert list(fitted.params.names) == ["const", "inc"]
    assert str(identity).replace(" ", "").lower().count("y=cons+inv") == 1
    from pymetrix_eviews.bundle import _model_program

    assert _model_program(q.meta.extra["eviews"], "m") == (
        "smpl 1980q1 1982q4\nequation eq_cons.ls cons c inc\nmodel m\nm.merge eq_cons\nm.append @IDENTITY Y = CONS + INV\n")
    with pytest.raises(NotFoundError, match="no model 'other'; its models: m"):
        pe.bundle_model(q, "other")


def test_the_fixture_estimates_are_least_squares():
    """The fixture's "EViews estimates", recomputed here by the textbook formulas from the literal data."""
    X = np.column_stack([np.ones(12), INC])
    y = np.asarray(CONS)
    b = np.linalg.solve(X.T @ X, X.T @ y)
    resid = y - X @ b
    se = np.sqrt(np.diag(resid @ resid / (12 - 2) * np.linalg.inv(X.T @ X)))
    rows = (FIXTURES / "macro" / "objects" / "quarterly" / "eq_cons_est.csv").read_text().splitlines()[1:]
    # X'X is ill-conditioned (income near 100), so the normal equations' last digits move with the NumPy and BLAS
    # (1.7e-12 relative with the oldest supported NumPy): 1e-9 still pins every printed digit of the fixture
    np.testing.assert_allclose([float(r.split(",")[1]) for r in rows], b, rtol=1e-9)
    np.testing.assert_allclose([float(r.split(",")[2]) for r in rows], se, rtol=1e-9)


def test_compare_estimates_refits_each_least_squares_equation():
    q, _ = _read(page="quarterly")
    table = pe.compare_estimates(q).to_frame()
    assert list(table["equation"]) == ["eq_cons", "eq_cons", "eq_rev", "eq_rev", "eq_white", "eq_iv"]
    assert list(table["coefficient"][:4]) == ["c(1) (const)", "c(2) (inc)", "c(1) (inc)", "c(2) (const)"]  # c(i) in EViews's order
    assert list(table["result"][:4]) == ["matches"] * 4
    np.testing.assert_allclose(table["pymetrix"][:4].astype(float), [-13.872477213762297, 0.938413540386753, 0.938413540386753,
                                                                    -13.872477213762297], rtol=1e-9)
    np.testing.assert_allclose(table["pymetrix s.e."][:4].astype(float), [1.5768404322686491, 0.014927073024990207,
                                                                         0.014927073024990207, 1.5768404322686491], rtol=1e-9)
    assert table["result"].iloc[4].startswith("not compared: options (cov=white)")
    assert table["result"].iloc[5] == "not compared: method TSLS: only least squares (ls) is compared in this version"
    ev = dict(q.meta.extra["eviews"])
    ev["equations"] = [dict(ev["equations"][0], estimates=[-13.9, 0.938413540386753])]  # an estimate EViews would not give
    other = Dataset(q.frame, q.axes, q.variables, q.flags, type(q.meta)(extra={"eviews": ev}))
    result = list(pe.compare_estimates(other).to_frame()["result"])
    assert result[0].startswith("differs: relative 0.00198") and result[1] == "matches"


# -- refusals, at their place --------------------------------------------------------------------------------------


def _copy(tmp_path, name):
    shutil.copytree(FIXTURES / name, tmp_path / name)
    return tmp_path / name


def _edit(path, change):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    change(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_a_folder_without_its_manifest_is_an_incomplete_export():
    with pytest.raises(DataError, match="has no manifest .*the export did not finish"):
        pe.read_bundle(FIXTURES / "no_manifest")
    with pytest.raises(DataError, match="no manifest .*missing.evbundle: .*incomplete export"):
        Dataset.read(str(FIXTURES / "no_manifest" / "missing.evbundle"))


def test_a_newer_format_version_asks_for_an_update():
    with pytest.raises(DataError, match=r"newer.evbundle: format_version 2 is newer than this pymetrix-eviews reads \(1\): update "
                                        r"pymetrix-eviews"):
        Dataset.read(str(FIXTURES / "newer" / "newer.evbundle"))


def test_a_csv_without_a_column_the_manifest_lists():
    with pytest.raises(DataError, match=r"pages/q.csv: no column 'cons' \(the series cons in pages\[0\]\.series\[1\]\); the file has "
                                        r"pmx_date, gdp"):
        Dataset.read(str(FIXTURES / "missing_column" / "missing.evbundle"))


@pytest.mark.parametrize(("csv", "message"), [
    ("PMX_DATE,GDP,CONS,INC,INV,SECTOR,REGION\n1980-01-01,abc,1,2,3,1,x\n", r"pages/quarterly.csv, line 2, column GDP: 'abc' is not a number"),
    ("PMX_DATE,GDP,CONS,INC,INV,SECTOR,REGION\n1980-01-01,inf,1,2,3,1,x\n", r"line 2, column GDP: 'inf' is not a number"),
    ("PMX_DATE,GDP,CONS,INC,INV,SECTOR,REGION\n1980-01-01,1,2,3,4,1,x\n1980Q2,1,2,3,4,1,x\n",
     r"pages/quarterly.csv, line 3, column PMX_DATE: '1980Q2' is not a date written YYYY-MM-DD"),
    ("PMX_DATE,GDP,CONS,INC,INV,SECTOR,REGION\n1980-02-01,1,2,3,4,1,x\n",
     r"line 2, column PMX_DATE: 1980-02-01 is not the first day of a Q period \(it falls in 1980Q1\)"),
    ("PMX_DATE,GDP,CONS,INC,INV,SECTOR,REGION\n1980-01-01,1,2,3,4,1,x\n1980-01-01,1,2,3,4,1,x\n", r"the period 1980Q1 appears twice"),
    ("PMX_DATE,GDP,CONS,INC,INV,SECTOR,REGION\n1980-01-01,1,2,3,4,1\n", r"pages/quarterly.csv, line 2: 6 fields; the header has 7"),
    ("PMX_DATE,GDP,gdp,CONS,INC,INV,SECTOR,REGION\n", r"line 1: a column name is given twice"),
    ("", r"pages/quarterly.csv is empty"),
])
def test_a_page_csv_that_does_not_read(tmp_path, csv, message):
    root = _copy(tmp_path, "macro")
    (root / "pages" / "quarterly.csv").write_text(csv, encoding="utf-8")
    with pytest.raises(DataError, match=message):
        Dataset.read(str(root / "macro.evbundle"), page="quarterly")


def test_a_panel_row_given_twice(tmp_path):
    root = _copy(tmp_path, "macro")
    (root / "pages" / "panel.csv").write_text("pmx_date,pmx_crossid,y,x\n2000-01-01,1,1,1\n2001-01-01,1,1,1\n2000-01-01,1,2,2\n")
    with pytest.raises(DataError, match="pages/panel.csv, line 4: the row's time and crossid appear on an earlier line"):
        Dataset.read(str(root / "macro.evbundle"), page="panel")


@pytest.mark.parametrize(("change", "message"), [
    (lambda m: m.update(format="something-else"), r"not a pymetrix EViews bundle"),
    (lambda m: m.update(format_version="1"), r"the manifest.format_version should be a whole number, not str"),
    (lambda m: m.pop("workfile"), r"the manifest has no 'workfile'"),
    (lambda m: m["pages"][0].pop("frequency"), r"pages\[0\] has no 'frequency'"),
    (lambda m: m["pages"][0].update(structure="stacked"), r"pages\[0\]\.structure is 'stacked'"),
    (lambda m: m["pages"][2].update(id_columns=[]), r"pages\[2\]\.id_columns: a panel page names its cross-section id column"),
    (lambda m: m["pages"][0]["series"][1].update(kind="date"), r"pages\[0\]\.series\[1\]\.kind is 'date'"),
    (lambda m: m["pages"][0]["series"][0]["attributes"].update(base_year=2017), r"series\[0\]\.attributes\.base_year should be text"),
    (lambda m: m["pages"][0]["equations"][0].pop("command"), r"pages\[0\]\.equations\[0\] has no 'command'"),
    (lambda m: m["pages"][0]["scalars"][0].update(value="0.3"), r"pages\[0\]\.scalars\[0\]\.value should be a number or null"),
    (lambda m: m["pages"].append(dict(m["pages"][1], name="QUARTERLY")), r"pages\[4\]: a second page called 'QUARTERLY'"),
    (lambda m: m["pages"][0].update(file="../macro/pages/quarterly.csv"), r"pages\[0\]\.file = .*files must be inside the bundle"),
    (lambda m: m["pages"][0].update(file="C:/work/quarterly.csv"), r"files must be inside the bundle"),
    (lambda m: m["pages"][0].update(file="pages/q2.csv"), r"pages\[0\]\.file names pages/q2.csv, which is not in the bundle"),
    (lambda m: m["pages"][0]["series"][4].update(value_map="vm_other"), r"series\[4\]\.value_map names 'vm_other'"),
    (lambda m: m["pages"][0]["equations"][0].update(ncoef=3), r"eq_cons_est.csv: 2 coefficients; the manifest's ncoef is 3"),
    (lambda m: m["pages"][0]["arrays"][1].update(rows=3), r"m2.csv: 2 rows; the manifest says 3"),
    (lambda m: m["pages"][0]["arrays"][1].update(cols=2), r"m2.csv, line 1: 3 values; the manifest says 2 columns"),
])
def test_a_manifest_that_does_not_read_is_refused_at_its_place(tmp_path, change, message):
    root = _copy(tmp_path, "macro")
    _edit(root / "macro.evbundle", change)
    with pytest.raises(DataError, match=message):
        Dataset.read(str(root / "macro.evbundle"), page="quarterly")


def test_json_errors_give_their_line_and_column(tmp_path):
    root = _copy(tmp_path, "macro")
    text = (root / "macro.evbundle").read_text()
    (root / "macro.evbundle").write_text(text.replace('"na": "NA",', '"na": "NA"'))
    with pytest.raises(DataError, match=r"macro.evbundle, line 7, column 3: not valid JSON \(Expecting ',' delimiter\)"):
        Dataset.read(str(root / "macro.evbundle"), page="quarterly")
    (root / "macro.evbundle").write_text(text.replace('"value": 0.3', '"value": NaN'))
    with pytest.raises(DataError, match="NaN is not JSON"):
        Dataset.read(str(root / "macro.evbundle"), page="quarterly")
    (root / "macro.evbundle").write_text(text.replace('"na": "NA",', '"na": "NA", "na": "-",'))
    with pytest.raises(DataError, match="the key 'na' is given twice"):
        Dataset.read(str(root / "macro.evbundle"), page="quarterly")


def test_the_page_must_be_named_when_there_are_several():
    with pytest.raises(DataError, match=r"macro.evbundle has 4 pages \(quarterly, Annual, panel, cross\): give page="):
        Dataset.read(str(MACRO))
    with pytest.raises(NotFoundError, match="has no page 'monthly'; its pages: quarterly, Annual, panel, cross"):
        Dataset.read(str(MACRO), page="monthly")
    with pytest.raises(DataError, match="page 'panel' is a panel: it cannot be read as undated"):
        Dataset.read(str(MACRO), page="panel", undated=True)


def test_value_map_and_estimate_files_are_checked(tmp_path):
    root = _copy(tmp_path, "macro")
    (root / "valmaps" / "quarterly" / "vm_sector.csv").write_text("code,text\n1,a\n")
    with pytest.raises(DataError, match=r"vm_sector.csv, line 1: a value map's columns are value,label \(found code,text\)"):
        Dataset.read(str(root / "macro.evbundle"), page="quarterly")
    (root / "valmaps" / "quarterly" / "vm_sector.csv").write_text("value,label\n1,a\n1,b\n")
    with pytest.raises(DataError, match="vm_sector.csv, line 3: the value '1' is mapped twice"):
        Dataset.read(str(root / "macro.evbundle"), page="quarterly")


# -- recorded and replayed -----------------------------------------------------------------------------------------


def test_typed_reads_are_recorded_and_replay(tmp_path):
    from pymetrix.commands import Session, load_project, save_project

    s = Session()
    s.run_code(f"quarterly = Dataset.read({str(MACRO)!r}, page='quarterly')")
    record = s.history[-1]
    assert record.ok and type(record.command).__name__ == "Call" and record.command.function == "pymetrix:dataset.read"
    assert dict(record.command.arguments)["kw"] == {"page": "quarterly"}
    assert record.code == f"quarterly = Dataset.read({str(MACRO)!r}, page='quarterly')"
    ns: dict = {}
    exec(s.script(), ns)
    pd.testing.assert_frame_equal(ns["quarterly"].frame, s["quarterly"].frame)
    path = save_project(s, tmp_path / "p.pymx")
    again = load_project(path)
    pd.testing.assert_frame_equal(again["quarterly"].frame, s["quarterly"].frame)
    assert again["quarterly"].meta.extra["eviews"]["equations"] == s["quarterly"].meta.extra["eviews"]["equations"]


def test_a_page_saved_as_parquet_keeps_its_metadata(tmp_path):
    q, _ = _read(page="quarterly")
    q.write(str(tmp_path / "q.parquet"))
    back = Dataset.read(str(tmp_path / "q.parquet"))
    pd.testing.assert_frame_equal(back.frame, q.frame)
    assert back.meta.extra["eviews"] == q.meta.extra["eviews"]
    for name, var in q.variables.items():
        assert (back.variables[name].label, back.variables[name].units, dict(back.variables[name].extra)) == \
            (var.label, var.units, dict(var.extra))


# -- the export program --------------------------------------------------------------------------------------------


def test_the_export_program_says_it_is_experimental_and_keeps_its_points_to_check(tmp_path):
    text = pe.export_program()
    head = text[:1200]
    assert "EXPERIMENTAL. UNTESTED IN EVIEWS." in head and "CSV or Excel" in head and "supported way in" in head
    assert text.count("(verify") >= 30  # every point the design marks (verify) is a comment in the program
    lines = text.splitlines()
    for point in ("@pagelist", "@pagefreq", "@pagesmpl", "pagesave(type=text", "@datestr(@date", "@wlookup", ".@attr(",
                  "@command", "@method", "@displayname", "@chr", "@ver", "{%pmx_o}.text", "@members", "shell(h) mkdir"):
        at = [i for i, line in enumerate(lines) if point in line]
        assert at and any("(verify" in line for i in at for line in lines[max(i - 6, 0):i + 1]), point
    for kind in ("series", "alpha", "group", "scalar", "string", "vector", "matrix", "sym", "coef", "valmap", "equation", "model", "sample"):
        assert kind in text
    saved = Path(pe.save_export_program(tmp_path))
    assert saved == tmp_path / "pymetrix_export.prg"
    raw = saved.read_bytes()
    assert raw.count(b"\r\n") == text.count("\n") and b"\r\r" not in raw  # Windows line ends, as EViews expects
    assert pe.save_export_program(tmp_path / "x.prg") == str(tmp_path / "x.prg")
