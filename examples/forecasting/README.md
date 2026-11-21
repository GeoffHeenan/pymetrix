---
title: Forecasting course pack
course: Forecasting course pack (pymetrix)
links: [forecast-workflow, commands/RunBacktest, commands/CompareBacktests, commands/CompareManyForecasts, commands/CombineForecasts]
references:
  - Elliott, G. and Timmermann, A. (2016). Economic Forecasting. Princeton University Press.
---

# Forecasting course pack

Six lab notebooks for a graduate course in economic forecasting. They follow the topics of Elliott and
Timmermann's *Economic Forecasting* (2016), with the emphasis on how forecasts are evaluated and compared. The
book is not reproduced here and is not needed to run the labs.

Each lab simulates its own data from a model it states, so the right answers are known. Each also checks
pymetrix against a calculation done another way: by hand, by numerical integration, or with statsmodels. Students
can see where every number comes from. The notebooks run from start to finish in the pymetrix test suite.

## The labs

| Lab | Topics | Chapters |
|---|---|---|
| [F1. Loss functions and the properties of a good forecast](F1_loss_and_efficiency.ipynb) | Squared, lin-lin and linex loss, and the best forecast under each; Mincer-Zarnowitz efficiency; overlapping errors and Newey-West | 2, 3, 15, 16 |
| [F2. Comparing forecasts](F2_comparing_forecasts.ipynb) | Diebold-Mariano; nested models and Clark-West, with a Monte Carlo of their size; Giacomini-White; encompassing; SPA, the model confidence set and StepM | 17 |
| [F3. Combining forecasts](F3_combination.ipynb) | Equal, inverse-MSE and regression weights in real time; the combination puzzle; the linear pool; Bayesian model averaging with the g-prior | 14 |
| [F4. Density and probability forecasts](F4_densities.ipynb) | Log score and CRPS; PITs, Berkowitz and Christoffersen; expanding and rolling windows under a volatility break; event probabilities: Brier score, ROC, Kuipers, Pesaran-Timmermann | 12, 13, 16, 18 |
| [F5. Forecasting with many predictors](F5_many_predictors.ipynb) | Principal components and the Bai-Ng criteria; diffusion indexes made in real time; ridge and lasso with cross-validation; bagging | 5, 6, 10, 11 |
| [F6. Breaks and nonlinearity](F6_breaks_and_nonlinearity.ipynb) | Chow and CUSUM; rolling, expanding and averaged windows through a break; threshold autoregressions; Hansen's and Teräsvirta's linearity tests; nonlinear forecasts out of sample | 8, 19 |

Each lab takes about two hours with its questions.

**What you need:**
- pymetrix, which brings statsmodels and matplotlib.
- Lab F2's multiple-comparison tests need `arch` (`pip install pymetrix[arch]`).
- Lab F5's ridge and lasso need scikit-learn (`pip install pymetrix[ml]`).

Without those packages, the labs skip those cells and say why.

The VAR, Bayesian VAR, forecast-round, nowcasting and real-time-vintage material is in the
example course (`examples/course`, Lab 3), and the same tools are in the app's **Forecast** menu.

## Coverage by chapter

How pymetrix covers each chapter of *Economic Forecasting*. **Lab** is where a lab here, or Lab 3 of the
example course, uses it. Chapter numbers follow the 2016 edition.

| Ch. | Topic | In pymetrix | Lab | Not yet |
|---|---|---|---|---|
| 2 | Loss functions | Squared, absolute, lin-lin and linex losses in backtests and comparisons | F1 | Losses that depend on the outcome as well as the error; estimating the loss from forecasts |
| 3 | The forecasting problem | Expanding and rolling backtests, from any fit or function | F1 | |
| 4 | Classical estimation | Least squares, maximum likelihood and IV, with robust and HAC covariances; IV-GMM through linearmodels | | |
| 5 | Bayesian forecasting | Bayesian VARs (Minnesota prior, simulated predictive densities); Bayesian model averaging | F3, course Lab 3 | General Bayesian regression and predictive densities beyond VARs |
| 6 | Model selection | Information criteria; ridge, lasso and elastic net with time-series cross-validation; Bayesian model averaging | F3, F5 | Selection by out-of-sample performance as a command |
| 7 | Univariate linear models | Autoregressions; equations with ARMA errors (differencing by expression, `d(y)`); exponential smoothing (simple, Holt, damped, Holt-Winters); unobserved components | F2 | |
| 8 | Univariate nonlinear models | Two-regime threshold AR; Markov switching; Hansen's and Teräsvirta's tests | F6 | Estimating smooth-transition models; more than two regimes |
| 9 | VARs | VARs, Bayesian VARs, VECMs; conditional forecasts (Waggoner-Zha), with ranges and structural scenarios | course Lab 3 | Density conditions (a whole distribution for a variable) |
| 10 | Many predictors | Principal components, the Bai-Ng criteria, dynamic factor models for nowcasting | F5, course Lab 3 | |
| 11 | Nonparametric methods | Random forests, gradient boosting, neural networks, nearest neighbours (scikit-learn); bagging | F5 | Sieve and kernel regression |
| 12 | Binary forecasts | Logit and probit; Brier, log score, ROC, Kuipers; Pesaran-Timmermann | F4 | Backtests of probability forecasts from logit and probit |
| 13 | Volatility and density forecasts | GARCH family (`arch`); normal, simulated and mixture densities in backtests | F4 | Realized volatility (HAR); multivariate GARCH |
| 14 | Forecast combination | Equal, inverse-MSE and regression weights in real time; linear pools; Bayesian model averaging | F3 | Shrinkage and time-varying weights |
| 15 | Properties of optimal forecasts | Mincer-Zarnowitz efficiency, with Newey-West for overlapping errors | F1 | Tests across horizons (forecast revisions) |
| 16 | Evaluating individual forecasts | Accuracy by horizon; rolling and expanding windows; interval tests | F1, F4 | |
| 17 | Comparing forecasts | Diebold-Mariano (HLN), Clark-West, Giacomini-White, encompassing; SPA, MCS, StepM | F2 | |
| 18 | Evaluating density forecasts | Log score, CRPS, PIT; Berkowitz, Christoffersen | F4 | Tests of PITs beyond one step ahead; weighted scores |
| 19 | Model instability | Chow, CUSUM, CUSUM of squares; rolling windows and averages over windows | F6 | Tests for several unknown breaks (Bai-Perron); forecasting with break models |
| 20 | Trending variables | Unit-root and cointegration tests; VECMs | | Forecast evaluation specific to trending series |

Real-time data and revisions (vintages, real-time backtests, forecast rounds and their revisions) are in the
example course's Lab 3.

## Getting help in the labs

- **Any function:** in a lab, type a pymetrix function's name and a question mark in a cell (`backtest?`), or press
  Shift+Tab inside the brackets of a call. This shows its full documentation.
- **Every function in a lab:** the last cell of each lab lists every pymetrix function, class and method it uses,
  with their arguments and what they do.

## For instructors

- **Start here:** `00_start_here.ipynb` is this page as a notebook, because Jupyter opens notebooks formatted. It
  is made from this README: edit the README, then run `python tests/test_forecasting_course.py` to rebuild it.
  The same command rebuilds each lab's list of functions.
- **Run every lab** from start to finish, from any folder, with `pytest tests/test_forecasting_course.py`. The
  tests also check:
  - each lab's conclusions;
  - that the lists of functions are current;
  - that no lab has been saved with its outputs.
- **In the app's help:** **Help → Course material…** adds this folder to the app's help. Each lab then appears on
  the help pages it links to.
- **Links:** run `pymetrix.testing.check_course("examples/forecasting")` to check them.
- **Your own labs:** `pymetrix.testing.write_api_reference(notebook)` adds the same list of functions to a lab of
  your own, and `check_api_reference` tells you when it needs updating.
- **Seeds:** each lab draws its data from a fixed seed. Changing the seed is a good exercise: which conclusions
  survive?
