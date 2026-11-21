# Start here: building and estimating models

This page is for people who build macro models: in Dynare, IRIS or by hand. It goes from equations to an
estimated, forecasting model in one sitting, with every step in code. The same steps are dialogs in the app
(**Model** menu), and the app writes them as code in the **Script** window. The [user guide](guide.md) has the
detail on each area, and [validation](validation.md) has what the numbers are checked against: Dynare 6, IRIS
conventions, closed forms.

The code blocks run in order, as one script; a test runs them as written, so they stay correct.
The data are simulated from the model, so nothing needs downloading.

## 1. A model from equations

```python
import pandas as pd
from pymetrix import Dataset, Equation, Model

model = Model([
    Equation("y ~ y[+1] - (1/sigma)*(i - pi[+1]) + ey", params=["sigma"], name="IS"),
    Equation("pi ~ beta*pi[+1] + kappa*y + epi", params=["beta", "kappa"], name="Phillips"),
    Equation("i ~ rho*i[-1] + (1 - rho)*(phi_pi*pi + phi_y*y) + ei", params=["rho", "phi_pi", "phi_y"], name="policy"),
], params=dict(sigma=1, beta=0.99, kappa=0.1, rho=0.8, phi_pi=1.5, phi_y=0.5))

model.check(shocks=["ey", "epi", "ei"])          # every equation, the steady state and Blanchard-Kahn, one row each
solution = model.linearize(shocks=["ey", "epi", "ei"]).solve()
irf = solution.irf(20)                           # irf.plot() draws them
```

`x[-1]` is a lag and `x[+1]` an expectation. Names that are not parameters and appear in no equation's left side
are exogenous; `shocks=` says which are shocks. `solution.rules()` shows the decision rules, as Dynare's
`stoch_simul` prints them.

## 2. Templates, Dynare and IRIS files

The same model ships as a template, with measurement equations that link it to data:

```python
from pymetrix import read_iris, read_mod, templates

nk = templates.new_keynesian()                   # nk.model is the Model above; nk.measurement maps it to data
mod = read_mod("""
var y pi i; varexo ey epi ei;
parameters sigma beta kappa rho phi_pi phi_y;
sigma = 1; beta = 0.99; kappa = 0.1; rho = 0.8; phi_pi = 1.5; phi_y = 0.5;
model(linear);
y = y(+1) - (1/sigma)*(i - pi(+1)) + ey;
pi = beta*pi(+1) + kappa*y + epi;
i = rho*i(-1) + (1 - rho)*(phi_pi*pi + phi_y*y) + ei;
end;
shocks; var ey; stderr 0.5; var epi; stderr 0.3; var ei; stderr 0.2; end;
""")
iris = read_iris("""
!transition_variables y, pi, i
!transition_shocks ey, epi, ei
!parameters sigma = 1, beta = 0.99, kappa = 0.1, rho = 0.8, phi_pi = 1.5, phi_y = 0.5
!transition_equations
    y = y{+1} - (1/sigma)*(i - pi{+1}) + ey;
    pi = beta*pi{+1} + kappa*y + epi;
    i = rho*i{-1} + (1 - rho)*(phi_pi*pi + phi_y*y) + ei;
""")
assert abs(mod.solve().irf(8).values - iris.solve().irf(8).values).max() < 1e-12
```

`read_mod` reads `.mod` files: the model, `steady_state_model`, `shocks`, `estimated_params`, `varobs` and the
estimation options; `read_iris` reads `.model` files, with `!for`, `!substitutions`, `&x`, `!if` and `!dtrends`.
Both give a template, so everything below works on them too. Pass a file path instead of the text.

## 3. Data, estimation and the posterior

```python
from pymetrix.estimation.priors import Beta, Gamma, InvGamma1

truth = nk.solve().simulate(120, nk.shock_sd, seed=11)
truth.index = pd.period_range("1995Q1", periods=120, freq="Q", name="time")
data = nk.measurement.apply(Dataset.from_frame(truth))     # inflation, the policy rate and GDP growth, in percent

fit = nk.estimate(data, estimate=["kappa", "rho", "sd_ei"],
                  priors={"kappa": Gamma(0.15, 0.05), "rho": Beta(0.7, 0.1), "sd_ei": InvGamma1(0.2, float("inf"))})
print(fit.summary())                             # the posterior mode with standard errors
post = fit.mcmc(draws=300, chains=2, seed=1)     # random-walk Metropolis-Hastings; use thousands of draws in practice
print(post.summary())                            # mean, HPD intervals, R-hat, effective draws, marginal density
ident = fit.identification()                     # can the data tell the parameters apart?
```

Priors are given by mean and standard deviation, as in Dynare (`InvGamma1(0.2, inf)` is `inv_gamma_pdf, 0.2, inf`).
On Dynare's `fs2000` the log posterior, mode, standard errors and Laplace density match Dynare 6.0, and
the posterior means are within 0.08 standard deviations.

## 4. Forecasts and decompositions

```python
forecast = fit.forecast(("2025Q1", "2026Q4"))   # from the smoothed state, observed series in data units
shocks = fit.decompose()                         # each shock's contribution to the filtered history
by_series = fit.contributions(horizon=8)         # each observed series' contribution, history and forecast
moments = fit.moments()                          # theoretical moments of the estimated model
```

Shock decompositions and data contributions add up to the filtered paths exactly. `fit.forecast(span, Scenario(...))`
takes a simulation plan (exogenize, endogenize, anticipated or not), and `tunes=` imposes judgement in the filter.

## 5. Policy, bounds and nonlinearity

```python
from pymetrix import occbin, optimal_policy, osr, second_order, sweep

loss = {"pi": 1.0, "y": 0.25}
commitment = optimal_policy(nk, "i", loss, kind="commitment")
discretion = optimal_policy(nk, "i", loss, kind="discretion")
best_rule = osr(nk, ["phi_pi", "phi_y"], loss, bounds={"phi_pi": (1.01, 10), "phi_y": (0, 5)})
assert commitment.loss() < discretion.loss() < best_rule.loss

zlb = templates.new_keynesian(lower_bound=0)     # the policy rate cannot go below 0% a year
episode = occbin(zlb, {"ey": {t: -2.5 for t in (1, 2, 3, 4)}}, periods=20, anticipated=True)
assert episode.binding.to_numpy().any()          # which periods the bound binds

runs = sweep(nk, {"phi_pi": [1.2, 1.5, 2.0, 3.0]}, "irf", periods=12)
rbc = templates.rbc()
rbc2 = second_order(rbc.model, rbc.shocks, rbc.shock_sd)   # Dynare order=2: ghx, ghu, ghxx, ghxu, ghuu, ghs2
```

OccBin matches the nonlinear perfect-foresight solution, second order matches Dynare to 1e-14, and optimal
policy matches Galí's closed forms. `model.solve(data, span, solver=PerfectForesight())` solves nonlinear,
anticipated paths; `PerfectForesight(terminal="growth")` ends them on a balanced growth path.

## 6. The same in the app, and back to code

`pymetrix-gui --demo nk_estimation` (or **File → Open demo project…**) opens sections 3 and 4 as a project, built one dialog at a time;
`pymetrix-gui --demo policy_and_nonlinear` opens section 5. **History** lists the steps, **Script** shows them as
code like the above, and **File → Export script…** saves it. The script runs in plain Python, with no display:

```python
from pymetrix import demo

session = demo("policy_and_nonlinear")           # what the app builds, without the app
script = session.script()                        # the code the Script window shows
assert "occbin" in script.lower() or "OccasionallyBinding" in script
```

Where next: [forward-looking models](guide.md#forward-looking-models) and
[measurement and estimation](guide.md#measurement-state-space-and-template-models) in the guide;
[known gaps](known-gaps.md) for what is not there yet.
