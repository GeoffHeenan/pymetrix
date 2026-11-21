"""Fan charts and forecast intervals start at the last actual value, and their one legend sits above the first panel,
clear of the data and of the dates (the forecast round's graph, Lab 3's headline figure, had the legend over both)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from pymetrix.figures import Fan, Line, legend_place  # noqa: E402


@pytest.fixture(scope="module")
def round_():
    from pymetrix import demo

    return demo("forecast_round")


def test_the_forecast_rounds_fan_starts_at_the_last_actual(round_):
    rnd = round_["forecast_round"]
    fig = rnd.plot()
    for panel in fig.panels:
        fan = next(layer for layer in panel.layers if isinstance(layer, Fan))
        actual = next(layer for layer in panel.layers if isinstance(layer, Line) and layer.label == "actual")
        last = actual.y.index[-1]
        assert fan.center.index[0] == last and fan.center.index[1] == rnd.paths.index[0]
        assert fan.center.iloc[0] == pytest.approx(actual.y.iloc[-1])  # joined: no gap between history and forecast
        for lo, hi in zip(fan.lower, fan.upper):  # the bands open from the known last value
            assert lo.iloc[0] == pytest.approx(hi.iloc[0]) == pytest.approx(actual.y.iloc[-1])
        assert np.allclose(fan.center.iloc[1:].to_numpy(), rnd.paths[panel.title].to_numpy())  # the forecast unchanged
    assert [p.legend for p in fig.panels] == [True, False, False] and fig.panels[0].legend_loc == "above"


def test_the_legend_is_above_the_panel_clear_of_data_and_dates(round_):
    fig = round_["forecast_round"].plot()
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    mpl = fig.draw()
    canvas = FigureCanvasAgg(mpl)
    canvas.draw()
    renderer = canvas.get_renderer()
    axes = [ax for ax in mpl.axes if ax.get_visible()]
    legend = axes[0].get_legend()
    box = legend.get_window_extent(renderer)
    for ax in axes:
        assert box.y0 >= ax.get_window_extent(renderer).y1 - 1  # above every panel's plotting area
        for label in ax.get_xticklabels():
            if label.get_text():
                assert not box.overlaps(label.get_window_extent(renderer))
    assert [t.get_text() for t in legend.get_texts()] == ["90%", "60%", "30%", "forecast", "actual"]
    assert sum(ax.get_legend() is not None for ax in axes) == 1


def test_a_forecasts_interval_joins_the_history_and_has_one_legend(round_):
    from pymetrix.figures import Band, forecast_figure

    fc, data = round_["fc"], round_["data"]
    fig = forecast_figure(fc, data)
    for panel in fig.panels:
        band = next(layer for layer in panel.layers if isinstance(layer, Band))
        central = next(layer for layer in panel.layers if isinstance(layer, Line) and layer.label == "forecast")
        actual = next(layer for layer in panel.layers if isinstance(layer, Line) and layer.label == "actual")
        assert central.y.index[0] == actual.y.index[-1] == band.lower.index[0]
        assert central.y.iloc[0] == band.lower.iloc[0] == band.upper.iloc[0] == actual.y.iloc[-1]
    assert [p.legend for p in fig.panels] == [True] + [False] * (len(fig.panels) - 1)
    alone = forecast_figure(fc)  # no data: the forecast alone, as before
    assert alone.panels[0].layers[-1].y.index[0] == fc.frame.index[0]


def test_the_figure_code_places_the_legend_as_drawn(round_):
    from pymetrix.figure_code import matplotlib_code

    code = matplotlib_code(round_["forecast_round"].plot())
    assert "loc='lower right', bbox_to_anchor=(1.0, 1.0), ncols=5" in code
    namespace: dict = {}
    exec(code, namespace)  # the code runs
    assert legend_place("", 3) == {} and legend_place("upper left", 2) == {"loc": "upper left"}


def test_a_simulation_fan_joins_its_data():
    from pymetrix.forecasting import fan_figure

    periods = pd.period_range("2020Q1", periods=8, freq="Q")
    history = pd.DataFrame({"y": np.arange(8.0)}, index=periods)
    from pymetrix import Dataset

    ahead = pd.period_range("2022Q1", periods=4, freq="Q")
    paths = pd.DataFrame({"y": [8.0, 9.0, 10.0, 11.0], "y_se": [0.5, 0.7, 0.9, 1.1]}, index=ahead)
    fig = fan_figure(paths, Dataset.from_frame(history.rename_axis("time")))
    fan = fig.panels[0].layers[0]
    assert str(fan.center.index[0]) == "2021Q4" and fan.center.iloc[0] == 7.0
    assert all(lo.iloc[0] == hi.iloc[0] == 7.0 for lo, hi in zip(fan.lower, fan.upper))
    assert fan.upper[0].iloc[1] > 8.0  # the bands beyond, from the standard errors


def test_a_saved_figure_keeps_its_legend_place(round_):
    import json

    from pymetrix.saved import _decode_view, _encode_view

    fig = round_["forecast_round"].plot()
    saved = _decode_view(json.loads(json.dumps(_encode_view(fig))))
    assert [p.legend_loc for p in saved.panels] == ["above"] * 3
