"""Demo projects: a session built step by step from the app's own commands, ready to explore.

    s = pymetrix.demo("forecast_round")      # File > Open demo project in the app
    s.script()                               # every step as Python
    demos()                                  # the names

Each demo runs the same commands as its course material, so its history, generated script, views and
provenance are those of a real session, and it can be saved, reopened and replayed like any project.
"""

from __future__ import annotations

from typing import Any

from .errors import NotFoundError

__all__ = ["demo", "demos", "headline"]

_DEMOS = {
    "forecast_round": "Lab 3: a quarterly VAR forecast round with judgement (sample_macro, 2025Q1-2026Q4).",
    "consumption": "Lecture 1: a consumption function, its diagnostics and a dynamic version (sample_macro).",
    "nk_estimation": "A New Keynesian model estimated on (simulated) data: posterior mode, Metropolis-Hastings, identification "
                     "and the contribution of each observed series.",
    "policy_and_nonlinear": "The zero lower bound (OccBin), optimal policy under commitment and discretion, the best simple rule, "
                            "a parameter sweep, and a growth model at second order.",
}

_HEADLINES = {  # the windows the app opens with a demo, the last on top, each on the view that shows the work; a note
    "forecast_round": ((("var", "Summary"), ("forecast_round", "Graph")),
                       "the round's fan chart is open (the VAR behind it); its Judgement view shows the adjustment"),
    "consumption": ((("static", "Summary"), ("dynamic", "Summary")),
                    "the dynamic consumption function is open (the static one behind it); see its Diagnostics"),
    "nk_estimation": ((("estimated", "Summary"), ("posterior", "Summary")),
                      "the posterior is open (the mode behind it); identification and contributions are in Objects"),
    "policy_and_nonlinear": ((("commitment", "Graph"), ("zlb_episode", "Graph")),
                             "the zero-lower-bound episode is open (optimal policy behind it); the rest is in Objects"),
}

_NK_DATA = '''nk_truth = nk.solve().simulate(120, nk.shock_sd, seed=11)
nk_truth.index = pd.period_range("1995Q1", periods=120, freq="Q", name="time")
data = nk.measurement.apply(Dataset.from_frame(nk_truth))'''



def demos() -> dict[str, str]:
    """The demo projects: name -> what it shows."""
    return dict(_DEMOS)


def headline(name: str) -> tuple[tuple[tuple[str, str], ...], str]:
    """What a demo shows first: its result windows, as (object, view) pairs to open in order (the last on top), and a
    note on what is open and where to go next. Nothing for an unknown demo."""
    return _HEADLINES.get(name, ((), ""))


def demo(name: str = "forecast_round") -> Any:
    """A new session with the demo's steps already run."""
    from .commands import AdjustRound, FitEquation, FitVAR, Forecast, GenerateSeries, LoadExample, MakeRound, Session

    if name not in _DEMOS:
        raise NotFoundError(f"no demo {name!r}; demos: {', '.join(_DEMOS)}")
    s = Session()
    s.demo = name
    if name in ("nk_estimation", "policy_and_nonlinear"):
        _models(s, name)
        s.dirty = False  # as opened: nothing changed yet
        return s
    s.run(LoadExample())
    if name == "forecast_round":
        for series, formula in (("ly", "log(y)"), ("lc", "log(c)"), ("li", "log(i)")):
            s.run(GenerateSeries(series, formula))
        s.run(FitVAR("ly, lc, li", lags="2"))
        s.run(Forecast("var", "2025Q1", "2026Q4", name="fc"))
        s.run(MakeRound("fc", "2024Q4 round"))
        s.run(AdjustRound("forecast_round", "ly", "2025Q1=-0.004, 2025Q2=-0.002", "port strike cuts output in early 2025"))
    else:
        s.run(FitEquation("c ~ y", name="static"))
        s.run(FitEquation("c ~ y + c[-1]", name="dynamic"))
    s.dirty = False  # as opened: nothing changed yet
    return s


def _models(s: Any, name: str) -> None:
    """The modelling demos: each step a dialog command (data simulated by a typed step), as in the app."""
    from .commands import (DataContributions, EstimateModel, LoadTemplate, OccasionallyBinding, OptimalSimpleRule, ParameterSweep,
                           SolveOptimalPolicy, SolveSecondOrder)
    from .commands.builtin import CheckIdentification, RunMCMC

    if name == "nk_estimation":
        s.run(LoadTemplate("new_keynesian", name="nk"))
        s.run_code(_NK_DATA)
        s.run(EstimateModel("nk", estimate="kappa, rho, sd_ei",
                            priors="kappa=Gamma(0.15, 0.05); rho=Beta(0.7, 0.1); sd_ei=InvGamma1(0.2, inf)", name="estimated"))
        s.run(RunMCMC("estimated", draws=1000, chains=2, seed=1, name="posterior"))
        s.run(CheckIdentification("estimated"))
        s.run(DataContributions("estimated", horizon=8))
        return
    s.run(LoadTemplate("new_keynesian", params="lower_bound=0", name="nk_zlb"))
    s.run(OccasionallyBinding("nk_zlb", "ey[1:4]=-2.5", periods=20, anticipated=True, name="zlb_episode"))
    s.run(LoadTemplate("new_keynesian", name="nk"))
    s.run(SolveOptimalPolicy("nk", "i", "pi=1, y=0.25", kind="commitment", name="commitment"))
    s.run(SolveOptimalPolicy("nk", "i", "pi=1, y=0.25", kind="discretion", name="discretion"))
    s.run(OptimalSimpleRule("nk", "phi_pi, phi_y", "pi=1, y=0.25", lower="phi_pi=1.01, phi_y=0", upper="phi_pi=10, phi_y=5"))
    # output's response to a cost-push shock doubles across phi_pi; to a demand shock it barely moves (see the help)
    s.run(ParameterSweep("nk", "phi_pi", "1.2, 1.5, 2, 3", what="irf", periods=12, response="y", shock="epi"))
    s.run(LoadTemplate("rbc", name="rbc"))
    s.run(SolveSecondOrder("rbc", name="rbc_second"))
