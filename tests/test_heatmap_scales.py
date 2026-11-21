"""Heatmaps of tables whose columns are in different units: ``scale`` "table" (one scale, as before), "column" (each
column its own) and "auto" (by the columns' sizes), and annotations that read on any cell.

A posterior summary mixes means and HPD bounds (0 to 1), R-hat (near 1) and effective draws (in the hundreds): on one
scale the draws took every colour and the rest were white, with pale grey numbers on pale cells. The checks: each
column's smallest and largest value land on the colour map's ends (from the image's own array and colours); the
text's contrast with its cell is worked out here from the WCAG 2 formula, independently of the code's helper; and the
recorded step and the figure's matplotlib code replay to the same chart."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymetrix import Dataset, Equation, Model, heatmap_figure
from pymetrix.errors import SpecificationError
from pymetrix.estimation.priors import Gamma, Normal
from pymetrix.figures import Heatmap, _heatmap_auto_scale, _text_colour
from pymetrix.metadata import Role

pytest.importorskip("matplotlib")

# a posterior summary as the owner's DSGE table had it: parameter values, R-hat near 1, effective draws in the hundreds
POSTERIOR = pd.DataFrame(
    [[0.12, 0.13, 0.12, 0.03, 0.08, 0.18, 1.01, 227.0],
     [0.80, 0.79, 0.80, 0.04, 0.73, 0.85, 1.00, 133.0],
     [0.17, 0.17, 0.17, 0.01, 0.15, 0.19, 1.02, 160.0],
     [1.52, 1.49, 1.50, 0.21, 1.15, 1.83, 1.00, 102.0]],
    index=["kappa", "rho", "sd_ei", "phi_pi"],
    columns=["mode", "mean", "median", "std dev", "HPD 90% low", "HPD 90% high", "R-hat", "eff. draws"])


@pytest.fixture(scope="module")
def data():
    return Dataset.read(str(Path(__file__).parents[1] / "examples" / "sample_macro.parquet"))


def _drawn(fig):
    drawn = fig.draw()
    drawn.canvas.draw()
    return drawn


def _wcag(rgb):
    """Relative luminance (WCAG 2), written out here rather than taken from the code under test."""
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb[:3]]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(a, b):
    from matplotlib.colors import to_rgb

    hi, lo = sorted((_wcag(to_rgb(a)), _wcag(to_rgb(b))), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _cell_colours(ax):
    """Each annotation's text, its colour and the colour of the cell under it, from the drawn image."""
    image = ax.get_images()[0]
    rgba = image.to_rgba(image.get_array())
    return [(t.get_text(), t.get_color(), rgba[int(round(t.get_position()[1])), int(round(t.get_position()[0]))]) for t in ax.texts]


def test_the_table_scale_is_unchanged():
    table = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["a", "b"], columns=["x", "y"])
    assert heatmap_figure(table).png() == heatmap_figure(table, scale="table").png()  # the default: one scale
    assert Heatmap(table).scale == "table"
    fig = _drawn(heatmap_figure(POSTERIOR))
    image = fig.axes[0].get_images()[0]
    assert np.allclose(image.get_array(), POSTERIOR.to_numpy()) and image.get_clim() == (0.01, 227.0)  # one scale over all
    assert len(fig.axes) == 2  # with its colour bar
    with pytest.raises(SpecificationError, match="scale is 'table', 'column' or 'auto'"):
        Heatmap(table, scale="rows")


def test_column_scale_maps_each_columns_range_onto_the_colour_maps_ends():
    from matplotlib import colormaps

    fig = _drawn(heatmap_figure(POSTERIOR, scale="column"))
    ax = fig.axes[0]
    image = ax.get_images()[0]
    shades = np.asarray(image.get_array())
    v = POSTERIOR.to_numpy()
    for j in range(v.shape[1]):  # each column's smallest value at 0, its largest at 1, in between in proportion
        lo, hi = v[:, j].min(), v[:, j].max()
        assert np.allclose(shades[:, j], (v[:, j] - lo) / (hi - lo))
    blues = colormaps["Blues"]
    rgba = image.to_rgba(shades)
    for j in range(v.shape[1]):
        assert np.allclose(rgba[v[:, j].argmin(), j], blues(0.0)) and np.allclose(rgba[v[:, j].argmax(), j], blues(1.0))
    assert [t.get_text() for t in ax.texts][:8] == ["0.12", "0.13", "0.12", "0.03", "0.08", "0.18", "1.01", "227.00"]  # real values
    assert len(fig.axes) == 1 and ax.get_xlabel() == "each column coloured on its own scale"  # a note, not a colour bar
    assert np.allclose(image.pymetrix_values, v)
    constant = _drawn(heatmap_figure(pd.DataFrame({"a": [1.0, 1.0], "b": [0.0, np.nan]}), scale="column")).axes[0]
    shades = np.ma.filled(constant.get_images()[0].get_array(), np.nan)
    assert np.allclose(shades[:, 0], 0.5) and shades[0, 1] == 0.5 and np.isnan(shades[1, 1])  # one value: the middle


def test_auto_picks_column_for_a_posterior_and_one_scale_for_tables_of_one_kind(data):
    from pymetrix import VAR

    assert _heatmap_auto_scale(POSTERIOR) == "column"  # effective draws are 100 times the parameters
    # without them, R-hat (about 1) is less than ten times the largest std dev (0.21): one scale; with that std dev
    # 0.03, R-hat is 30 times it
    assert _heatmap_auto_scale(POSTERIOR.drop(columns=["eff. draws"])) == "table"
    assert _heatmap_auto_scale(POSTERIOR.drop(columns=["eff. draws"]).drop(index="phi_pi")) == "column"
    signed = POSTERIOR.copy()  # coefficients of both signs: still parameters beside counts of draws
    signed.loc["rho", ["mode", "mean", "median", "HPD 90% low", "HPD 90% high"]] *= -1
    assert _heatmap_auto_scale(signed) == "column"
    residuals = pd.DataFrame({"level": [120.0, -80.0, 15.0], "rate": [0.02, -0.01, 0.005]})
    assert _heatmap_auto_scale(residuals) == "table"  # deviations about zero, every column: one scale
    params = POSTERIOR[["mode", "mean", "median", "HPD 90% low", "HPD 90% high"]]
    assert _heatmap_auto_scale(params) == "table"  # parameter values alone: one kind
    q = pd.DataFrame({"AC": [0.6, 0.35, 0.2, 0.1], "PAC": [0.6, -0.05, 0.02, -0.01], "Q": [37.0, 49.5, 53.6, 54.6],
                      "p-value": [0.0, 0.0, 0.0, 0.0]}, index=[1, 2, 3, 4])
    assert _heatmap_auto_scale(q) == "column"  # a correlogram: Q statistics beside autocorrelations
    d = data.with_series(gy="400*dlog(y)", gc="400*dlog(c)", gi="400*dlog(i)")
    corr = d.plot(["gy", "gc", "gi"], kind="correlation").panels[0].layers[0].values
    assert _heatmap_auto_scale(corr) == "table"
    cov = pd.DataFrame([[400.0, 2.0], [2.0, 0.01]], index=["level", "rate"], columns=["level", "rate"])
    assert _heatmap_auto_scale(cov) == "table"  # a square matrix keeps one scale, whatever its sizes
    var = VAR(["gy", "gc", "gi"], lags=1).fit(d)
    irf = var.irf(8, draws=0)
    from pymetrix.views import sections

    tables = dict(sections(irf))
    assert _heatmap_auto_scale(irf.heatmap().panels[0].layers[0].values) == "table"  # responses by horizon
    for tab in ("Selected horizons", "All horizons"):  # responses by shock, dying away
        assert _heatmap_auto_scale(heatmap_figure(tables[tab]).panels[0].layers[0].values) == "table", tab
    fevd = var.fevd(8)
    assert _heatmap_auto_scale(fevd.heatmap().panels[0].layers[0].values) == "table"
    for tab, table in sections(fevd):
        if hasattr(table, "to_frame") and not hasattr(table, "panels"):  # shares: each variable's add up to 100
            assert _heatmap_auto_scale(heatmap_figure(table).panels[0].layers[0].values) == "table", tab
    shares = pd.DataFrame([[0.95, 0.04, 0.01], [0.10, 0.89, 0.01]], columns=["s1", "s2", "s3"])
    assert _heatmap_auto_scale(shares) == "table"  # as fractions too
    fig = heatmap_figure(POSTERIOR, scale="auto")
    assert fig.panels[0].layers[0].scale == "auto" and len(_drawn(fig).axes) == 1  # drawn column by column


def test_annotations_read_on_every_cell():
    from matplotlib import colormaps

    for cmap in ("Blues", "RdBu_r", "viridis", "magma", "Greys", "YlOrRd"):  # every shade of each colour map
        for x in np.linspace(0, 1, 257):
            cell = colormaps[cmap](x)
            assert _contrast(_text_colour(cell), cell) >= 4.5, (cmap, x)
    tables = [POSTERIOR, pd.DataFrame(np.linspace(-1, 1, 25).reshape(5, 5)), pd.DataFrame(np.arange(20.0).reshape(4, 5))]
    for table in tables:
        for scale in ("table", "column"):
            ax = _drawn(heatmap_figure(table, scale=scale)).axes[0]
            for text, colour, cell in _cell_colours(ax):
                assert _contrast(colour, cell) >= 4.5, (text, colour, cell)
    ax = _drawn(heatmap_figure(POSTERIOR, scale="column")).axes[0]
    colours = {text: colour for text, colour, _ in _cell_colours(ax)}
    assert colours["227.00"] == "#ffffff" and colours["102.00"] == "#1a1a1a"  # white on dark, dark on light


def test_diagnostics_can_be_left_out():
    from pymetrix.output import Table

    table = Table(list(POSTERIOR.columns), POSTERIOR.to_numpy().tolist(), index=list(POSTERIOR.index), title="Posterior")
    assert list(heatmap_figure(table, diagnostics=False).panels[0].layers[0].values.columns) == list(POSTERIOR.columns[:6])
    assert list(heatmap_figure(table).panels[0].layers[0].values.columns) == list(POSTERIOR.columns)
    mcse = pd.DataFrame({"mean": [0.1, 0.2], "MCSE mean": [0.01, 0.02], "ESS bulk": [400.0, 380.0]})
    assert list(heatmap_figure(mcse, diagnostics=False).panels[0].layers[0].values.columns) == ["mean"]
    repeated = Table(["gy", "(se)", "gc", "(se)"], [[0.5, 0.1, 0.2, 0.05], [0.3, 0.1, 0.4, 0.04]], index=["a", "b"])
    assert heatmap_figure(repeated).panels[0].layers[0].values.shape == (2, 4)  # a VAR's Summary: it failed on "(se)"
    only = pd.DataFrame({"R-hat": [1.0, 1.01]})
    assert list(heatmap_figure(only, diagnostics=False).panels[0].layers[0].values.columns) == ["R-hat"]  # nothing else to show


def test_column_heatmaps_save_and_draw_in_matplotlib_alone():
    from test_figures import _run_standalone

    from pymetrix.figures import readout
    from pymetrix.saved import _decode_view, _encode_view

    for scale in ("column", "auto", "table"):
        fig = heatmap_figure(POSTERIOR, label="value", scale=scale)
        assert _decode_view(json.loads(json.dumps(_encode_view(fig)))).png() == fig.png()  # a saved result draws the same
        ours, theirs = _drawn(fig), _run_standalone(fig.matplotlib_code())
        a, b = ours.axes[0], theirs.axes[0]
        assert len(ours.axes) == len(theirs.axes)
        assert np.allclose(np.ma.filled(a.get_images()[0].get_array(), np.nan), np.ma.filled(b.get_images()[0].get_array(), np.nan))
        assert [(t.get_text(), t.get_color()) for t in a.texts] == [(t.get_text(), t.get_color()) for t in b.texts]
        assert a.get_xlabel() == b.get_xlabel()
    ax = _drawn(heatmap_figure(POSTERIOR, scale="column")).axes[0]
    px, py = ax.transData.transform((7, 0))  # the readout gives the value, not its shade
    assert readout(ax, px, py).endswith(": 227")


@pytest.fixture(scope="module")
def posterior():
    rng = np.random.default_rng(1)
    x = np.zeros(120)
    for t in range(1, 120):
        x[t] = 0.7 * x[t - 1] + 0.5 * rng.normal()
    data = Dataset.from_frame(pd.DataFrame({"x": x}, index=pd.period_range("1950Q1", periods=120, freq="Q", name="time")))
    model = Model([Equation("x ~ rho*x[-1] + e", params=["rho"])], params={"rho": 0.3}, roles={"e": Role.SHOCK})
    fit = model.estimate(data, ["x"], estimate=["rho", "sd_e"], priors={"rho": Normal(0.5, 0.2), "sd_e": Gamma(0.5, 0.2)})
    return fit.mcmc(draws=150, chains=2, seed=3)


def test_the_heatmap_button_colours_by_column_and_its_step_replays(posterior, first_run, monkeypatch):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("qtconsole")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymetrix.commands import Session, TableHeatmap
    from pymetrix.ui import plugins_qt
    from pymetrix.ui.qt import main_window

    monkeypatch.setattr(plugins_qt, "ask_new_plugins", lambda *a, **k: None)
    session = Session()
    session.objects["post"] = posterior
    window = main_window(session)
    window.show_object("post")
    sub = next(s for s in window.mdi.subWindowList() if s.windowTitle().startswith("post"))
    sub.widget().heatmap_buttons["Summary"].click()
    step = session.history[-1]
    assert isinstance(step.command, TableHeatmap) and step.command.scale == "auto"
    assert step.code == "post_heatmap = heatmap_figure(post, view='Summary', scale='auto', diagnostics=False)"
    layer = session["post_heatmap"].panels[0].layers[0]
    assert layer.scale == "auto" and "R-hat" not in layer.values.columns and "eff. draws" not in layer.values.columns
    ns = {"post": posterior}
    exec("from pymetrix import *\n" + step.code, ns)
    assert ns["post_heatmap"].png() == session["post_heatmap"].png()  # the recorded step replays
    assert TableHeatmap("post", "Summary").scale == "auto"  # the step's own default, when its dialog is opened
    session.dirty = False
    window.close()
