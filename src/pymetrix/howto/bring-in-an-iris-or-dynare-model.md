---
summary: read an IRIS .model or Dynare .mod file, check the steady state and the Blanchard-Kahn conditions, and solve it.
audience: IRIS and Dynare users
order: 4
---

# Bring an IRIS or Dynare model across, then check and solve it

Read a model file written for IRIS or Dynare, check that pymetrix reads it as you meant (the equations, parameters,
shocks and their sizes), and solve it to first order under rational expectations. Neither IRIS nor Dynare (nor
MATLAB) is needed: the file is read as text.

## Before you start

- The model file. This page uses a three-equation New Keynesian model, in IRIS's syntax as `nk.model`:

  ```text nk.model
  % A three-equation New Keynesian model
  !transition_variables
      "Output gap" y, "Inflation" pi, "Policy rate" i
  !transition_shocks
      ey, epi, ei
  !parameters
      sig = 1, bet = 0.99, kap = 0.1, rho = 0.8, phi
      std_ey = 0.5, std_epi = 0.3, std_ei = 0.2
  !measurement_variables
      obs_pi
  !transition_equations
      "IS curve" y = y{+1} - sig*(i - pi{+1}) + ey;
      "Phillips curve" pi = bet*pi{+1} + kap*y + epi;
      "Policy rule" i = rho*i{-1} + (1-rho)*(phi*pi + 0.5*y) + ei;
  !measurement_equations
      obs_pi = 2 + 4*pi;
  ```

  and in Dynare's as `nk.mod`:

  ```text nk.mod
  // The same model in Dynare
  var y pi i;
  varexo ey epi ei;
  parameters sig bet kap rho phi;
  sig = 1; bet = 0.99; kap = 0.1; rho = 0.8; phi = 1.5;
  model;
  y = y(+1) - sig*(i - pi(+1)) + ey;
  pi = bet*pi(+1) + kap*y + epi;
  i = rho*i(-1) + (1-rho)*(phi*pi + 0.5*y) + ei;
  end;
  shocks;
  var ey; stderr 0.5;
  var epi; stderr 0.3;
  var ei; stderr 0.2;
  end;
  steady;
  check;
  stoch_simul(order=1, irf=20) y pi i;
  ```

- The values of parameters the IRIS file leaves out. IRIS models often set them in code (`m.phi = 1.5`); here
  `phi` has no value in the file.

## In the app

1. Model → Import → IRIS file…, with IRIS file: nk.model, Parameters: phi=1.5 and Save as: nk. A parameter
   still without a value stops the step and is named.
2. In the model's window, read the views: Equations (as pymetrix reads them, `y{+1}` written `y[+1]`), Parameters,
   Measurement (`obs_pi`) and Shocks (their standard deviations, from the `std_` parameters).
3. Model → Solution → Rational expectations…, with Equations: nk and Save as: sol. Leave Shocks and Parameters
   empty: an imported model has its own. The solve finds the steady state and checks the Blanchard-Kahn conditions;
   if there is no stable solution, or more than one, the step stops and says which.
4. In the solution's window: Decision rules (each variable on the predetermined variables and the shocks), Graph
   (the responses to a one-unit shock), Steady state and Roots (with the Blanchard-Kahn count).
5. For the Dynare file: Model → Import → Dynare file…, with Dynare file: nk.mod and Save as: mod, then Rational
   expectations… with Equations: mod. The model's window lists the file's commands (steady, check, stoch_simul)
   under Commands: they are read, not run.

To look for problems in the file itself before importing it: Model → Model editor…, Syntax: IRIS (.model) or Dynare
(.mod), and paste the text. Problems lists each at its line and column; Check solution finds the steady state and the
roots and draws the responses to one-standard-deviation shocks.

## In code

```python
import numpy as np
from pymetrix import read_iris, read_mod

nk = read_iris("nk.model", params={"phi": 1.5})       # values the file leaves out
nk.check()                                              # equations, parameters, timing, steady state, Blanchard-Kahn
nk.steady_state()                                       # zero: the model is in deviations
sol = nk.solve()                                        # first order: the unique stable solution
sol.roots()                                             # 1 stable root for 1 predetermined variable
sol.rules()                                             # y, pi and i on i[-1] and the three shocks
irf = sol.irf(20, shock_sd=nk.shock_sd)                 # one standard deviation of each shock
irf.plot()
```

The Dynare file gives the same model:

```python
mod = read_mod("nk.mod")
mod.commands                                            # ('steady', 'check', 'stoch_simul(order=1, irf=20) y pi i')
dyn = mod.solve()
assert np.allclose(dyn.irf(20).values, sol.irf(20).values)
```

## Check the result

- The Roots view: "Blanchard-Kahn: stable roots 1 (unit roots included), predetermined variables 1". The stable
  root is 0.54; the other two are outside the unit circle.
- The Decision rules against IRIS's or Dynare's own: the response of `y` to `ei` on impact is -1.97 for a unit
  shock. Dynare's `stoch_simul` prints the same numbers as its policy and transition functions, with the
  variables as columns.
- The impulse responses against IRIS's `srf` or Dynare's `stoch_simul` (both one standard deviation): `y` to `ei`
  on impact is -0.394 (-1.97 times 0.2).
- The steady state against IRIS's `sstate` or Dynare's `steady`: here zero for every variable.

## Pitfalls

- **One unit or one standard deviation.** `sol.irf(20)` and the solution's Graph view give responses to a one-unit
  shock; IRIS's `srf` and Dynare's `stoch_simul` to one standard deviation. Pass `shock_sd=nk.shock_sd` to compare,
  or use the model editor's Check solution.
- **"The solution is not unique".** Too many stable roots: indeterminacy, often a policy rule that responds less
  than one for one to inflation (`phi=0.8` here gives it). Too few: no stable solution, often a sign error.
- **`!log_variables`.** Those variables are solved in logs, so their responses are log deviations, as in IRIS; a
  Dynare model of the same variables in levels gives level deviations.
- **Commands in a `.mod` file are not run.** `estimated_params`, `varobs` and `estimation` are read for estimation
  (Model → Estimation → Estimate model…); `stoch_simul`'s options are not applied.
- **Decision rules in Dynare's own output are in its order.** `oo_.dr.ghx` and `oo_.dr.ghu` have their rows in
  `oo_.dr.order_var`'s order, not the declaration order pymetrix uses.

## See also

- [Forward-looking models](pymetrix:topic/forward-looking-models) and
  [Coming from MATLAB/IRIS and Dynare](pymetrix:topic/coming-from-iris), in the help: where `sstate`, `filter`,
  `estimate`, `shock_decomposition` and the other commands are.
- [The user guide's IRIS models](../../../docs/guide.md#iris-models) and
  [Dynare models](../../../docs/guide.md#dynare-models): `!dtrends`, `!substitutions`, `!for` loops,
  `!reporting_equations`, and what a `.mod` file's blocks become.
- [Building and estimating models](../../../docs/modelling-quickstart.md): from a solved model to estimation.
