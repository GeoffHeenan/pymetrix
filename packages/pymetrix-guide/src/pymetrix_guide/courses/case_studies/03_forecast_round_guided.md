---
title: Lab 3, guided. A quarterly forecast round
summary: The December round for a simulated economy, from the data to the report: a VAR baseline, judgement, revisions since June and the track record
course: Case studies (pymetrix-guide)
level: introductory
stages: [Data, Model, Baseline, Judgement, Revisions, Evaluate, Report]
links: [forecast-workflow, var, bayesian-var, reports, commands/MakeRound]
---

# Lab 3, guided. A quarterly forecast round

## Overview: the forecast workflow

A forecast round is the work a forecasting team does each quarter to produce a forecast, publish it, and later
account for it. Most rounds pass through the same stages, and this case study takes them in order:

1. **Data.** Bring in the latest data and put them in the form the model needs (here, logs). Note the last
   period the data reach: the forecast starts after it. With real data, note also what was revised since the last
   round, and keep the data as they were (the vintage).
2. **Model.** Choose the model, estimate it, and check it: is it stable, and is there dynamics left in its
   residuals?
3. **Baseline.** Forecast with the model alone, with no conditions or adjustments: the unconditional forecast, a
   central path with bands that widen with the horizon.
4. **Judgement.** Add what the model cannot know: an event after the data, a policy already announced, or
   information from other sources. This is done by adjusting the path, or by conditioning the forecast on an
   assumed path for one variable. Forecasters do it because a model sees only its data; the reason is recorded, so
   the published forecast can always be separated from the model's.
5. **Revisions** and scenarios. Set the forecast against alternatives: the last round (what changed, and why?) or
   another assumption (what if?).
6. **Evaluate.** Ask how good the model has been: forecast the past as if in real time (a backtest) and compare it
   with an alternative model on the same forecasts.
7. **Report.** Publish the forecast: fan charts, tables, the judgement and its reasons, and where each number came
   from.

## How the app keeps the record

The forecast is only part of a round. The rest is the record: what was done, with which data, and why.

- **Each step** you run is in the session's history. **File → Export script** gives the steps as Python, and a
  saved project runs them again when it is opened.
- **A forecast round** (**Forecast → Make round**) holds the forecast paths and their uncertainty, when the round
  was made, the last period of data it used, its source and notes. Each adjustment is kept with the value before
  it and the reason (**Forecast → Adjust round**).
- **A forecast store** (**Forecast → Save round**) is a folder that keeps every round unchanged, so a later round
  can be compared with it (**Forecast → Compare rounds**) and its forecasts judged against what happened.
- **A backtest** (**Forecast → Backtest**) gives a model's track record, and **Forecast → Compare backtests** tests
  whether one model's forecasts are better than another's by more than chance.
- **A report** (**File → Export report**) puts the round, its judgement and the track record in one document, with
  an appendix of the data, sample and method behind the numbers.

## This scenario

**The data.** `sample_macro` is a simulated quarterly economy from 2000Q1 to 2024Q4: consumption `c`, investment
`i`, government spending `g` and output `y = c + i + g`. Being simulated, it has a single vintage: past figures are
never revised.

**The round.** The data end in 2024Q4, so this is the December round: the forecast starts in 2025Q1 and runs
eight quarters, to 2026Q4.

**The model.** A VAR with two lags in the logs of output, consumption and investment. Each variable depends on the
last two quarters of all three, so the model needs no view of what drives what, and its forecasts of the three
hang together. Two lags keep it small (**Estimate → VARs and local projections → VAR lag selection** is how you
would check the choice).

**The judgement.** A port strike will cut output in early 2025, which data to 2024Q4 cannot show. The round lowers
log output by 0.004 in 2025Q1 and 0.002 in 2025Q2 (about 0.4 and 0.2 per cent), as the effect fades once the
strike ends. Only output's path changes: consumption and investment keep the model's.

**The comparisons.** The December round against the June round, made with the data to 2024Q2: what was revised,
and why? Then the VAR against a Bayesian VAR (a Minnesota prior) in a backtest from 2015Q1 to 2022Q4, four
quarters ahead: which forecasts output better, and is the difference more than chance?

**By the end** you should be able to explain what the baseline's bands include, what the judgement changed and how
it is recorded, what drove the revision since June, how far to trust the VAR's forecasts, and where each number in
the report came from.

## Using the guide

Start from an empty project (**File → New session**). Each step opens its dialog filled in (**Open…**): look at
the settings, change them if you like, and press OK. **Show me** runs the step as written. A step ticks itself
when its result is there; **Undo** takes the tick back and **Redo** gives it again. Everything you run is an
ordinary step: the History, **File → Export script** and the saved project keep it.

Each step says why it is done and what to look for in its result. The strip above the step shows where it is in
the workflow. The `forecast_round` demo project (**File → Open demo project…**) is the December round as step 8
leaves it, if you want to look ahead.

---

First, the data: load the example dataset `sample_macro` (**File → Load example data**).

```pymetrix-step
title: Load the example data
stage: Data
command: LoadExample
settings: {dataset: sample_macro, name: data}
check: object data is Dataset
why: Every forecast starts from the data as they are now; their last period sets where the forecast begins.
look: The four series c, i, g and y, quarterly from 2000Q1 to 2024Q4: 2024Q4 is the last period of data, so the forecast will start in 2025Q1.
```

The VAR is in logs. Add log output, `ly = log(y)`, with **Data → Generate series**.

```pymetrix-step
title: Log output
stage: Data
command: GenerateSeries
settings: {series: ly, formula: "log(y)"}
check: series ly in data
why: In logs a change of 0.01 is about 1 per cent, so the judgement and the revisions read as percentages, and steady growth is a straight line.
look: ly is now in data, next to y.
```

The same for consumption...

```pymetrix-step
title: Log consumption
stage: Data
command: GenerateSeries
settings: {series: lc, formula: "log(c)"}
check: series lc in data
why: The VAR forecasts consumption with output and investment, so it needs the log of each.
look: lc is in data.
```

... and investment.

```pymetrix-step
title: Log investment
stage: Data
command: GenerateSeries
settings: {series: li, formula: "log(i)"}
check: series li in data
why: The third of the model's variables.
look: li is in data: the three series the VAR will use are ready.
```

Estimate a VAR in `ly`, `lc` and `li` with 2 lags (**Estimate → VARs and local projections → VAR**).

```pymetrix-step
title: Estimate the VAR
stage: Model
command: FitVAR
settings: {variables: "ly, lc, li", lags: "2", name: var}
check: object var is FittedVAR
why: This is the model behind the baseline, so check it before forecasting with it.
look: The Summary (the sample starts in 2000Q3, as two quarters go to the lags); the Roots tab (all inside the unit circle, so the VAR is stable, though the largest is close to one, as for trending series in logs); and the Residual correlogram (few correlations outside the 95 per cent band, about what chance gives, so little dynamics is left).
```

Forecast two years past the data, 2025Q1 to 2026Q4 (**Forecast → Forecast**).

```pymetrix-step
title: Forecast 2025-2026
stage: Baseline
command: Forecast
settings: {fit: var, start: 2025Q1, end: 2026Q4, name: fc}
check: object fc is Dataset
why: The baseline: the model's own forecast, before any judgement, with its uncertainty.
look: For each variable the path (ly), its standard error (ly_se) and a 95 per cent interval (ly_lower, ly_upper); the standard error grows with the horizon. A least-squares VAR's intervals allow for the shocks only, not for uncertainty about its coefficients, so they are too narrow.
```

Make the forecast the "2024Q4 round" (**Forecast → Make round**). A round records when it was made, the last
period of data it used, and its source.

```pymetrix-step
title: Make the December round
stage: Baseline
command: MakeRound
settings: {forecast: fc, round_name: "2024Q4 round", name: forecast_round}
check: object forecast_round is ForecastRound
why: A round turns a forecast into a record that can be adjusted, stored, compared with later rounds and published.
look: The Paths table ("data to 2024Q4") and the Graph tab: a fan chart of each variable after its recent history, with 30, 60 and 90 per cent bands.
```

A port strike will cut output in early 2025. Lower `ly` by 0.004 in 2025Q1 and 0.002 in 2025Q2 with
**Forecast → Adjust round**, giving the reason.

```pymetrix-step
title: Add the judgement
stage: Judgement
command: AdjustRound
settings: {round: forecast_round, variable: ly, changes: "2025Q1=-0.004, 2025Q2=-0.002", reason: "a port strike cuts output in early 2025"}
check: step AdjustRound ran on forecast_round
why: The model has only the data to 2024Q4 and cannot know about the strike; the judgement adds it, and its reason becomes part of the round's record.
look: The round's Judgement tab: for each period, the value before and after, the change and the reason. In the Graph tab, the output path is lower in 2025Q1 and Q2 and moves the fan with it, by much less than the fan's width; consumption and investment are unchanged.
```

What has changed since the last round? The June round used the data to 2024Q2. Rebuild it: a VAR on the data to
2024Q2 (the **End** field of the VAR dialog)...

```pymetrix-step
title: The June VAR
stage: Revisions
command: FitVAR
settings: {variables: "ly, lc, li", lags: "2", end: 2024Q2, name: var_june}
check: object var_june is FittedVAR
why: To compare with the last round, rebuild it as it was: the same model on the data it had.
look: The Summary: the sample ends in 2024Q2, two quarters before the December VAR's.
```

... its forecast from 2024Q3...

```pymetrix-step
title: The June forecast
stage: Revisions
command: Forecast
settings: {fit: var_june, start: 2024Q3, end: 2026Q4, name: fc_june}
check: object fc_june is Dataset
why: The June round forecast from the quarter after its data, to the same end as December's.
look: The forecast starts in 2024Q3; its 2024Q3 and 2024Q4 values are forecasts of quarters that the December round already has as data.
```

... and its round.

```pymetrix-step
title: Make the June round
stage: Revisions
command: MakeRound
settings: {forecast: fc_june, round_name: "2024Q2 round", name: june_round}
check: object june_round is ForecastRound
why: As a round it can be stored and compared with December's.
look: The Paths table says "data to 2024Q2"; the December round's says 2024Q4.
```

A forecast store is a folder that keeps every round. Save the June round in it (**Forecast → Save round**; the
folder `forecasts` is next to your project)...

```pymetrix-step
title: Save the June round
stage: Revisions
command: SaveRound
settings: {round: june_round, folder: forecasts, overwrite: true}
check: step SaveRound ran on june_round
why: The store keeps each round as it was, so comparisons and later evaluation use the forecast that was made, not a rebuilt one.
look: A folder forecasts next to the project, with the round's paths and its record: when it was made, the data it used, its notes and judgement.
```

... and the December round.

```pymetrix-step
title: Save the December round
stage: Revisions
command: SaveRound
settings: {round: forecast_round, folder: forecasts, overwrite: true}
check: step SaveRound ran on forecast_round
why: Both rounds must be in the store to be compared; the December round goes in with its judgement.
look: Overwrite is ticked so that the guide can be run again; in a real store a different round under an existing name is refused, as overwriting loses the earlier record.
```

Compare the two rounds for output (**Forecast → Compare rounds**). The December round is higher than the June
round all the way out. Is that the new data, the judgement, or both?

```pymetrix-step
title: Compare the rounds
stage: Revisions
command: CompareRounds
settings: {old: "2024Q2 round", new: "2024Q4 round", folder: forecasts, variables: ly, name: revisions}
check: object revisions exists
why: Every revision needs an explanation: new data, judgement, or a changed model.
look: For each quarter, both rounds' paths for ly and the revision (December less June). The judgement lowered only 2025Q1 and Q2: compare those quarters' revisions with the rest.
```

How good has the VAR been? A backtest re-estimates the model at every forecast origin from 2015 to 2022 and
forecasts four quarters ahead (**Forecast → Backtest**).

```pymetrix-step
title: Backtest the VAR
stage: Evaluate
command: RunBacktest
settings: {fit: var, start: 2015Q1, end: 2022Q4, horizon: 4, name: bt_var}
check: object bt_var is Backtest
why: A forecast is worth as much as the model's track record; the backtest forecasts each quarter from the data before it, as if in real time.
look: The Accuracy tab: RMSE, MAE and mean error by variable and horizon. Errors grow with the horizon, and a mean error far from zero is a bias. With real data, give the data as they were at each origin (Vintages): revised data flatter the model.
```

Would a Bayesian VAR have done better? Estimate one: tick **Bayesian** in the VAR dialog (a Minnesota prior,
tightness 0.2). Fewer posterior draws than the default keep the backtest quick.

```pymetrix-step
title: Estimate a Bayesian VAR
stage: Evaluate
command: FitVAR
settings: {variables: "ly, lc, li", lags: "2", bayesian: true, tightness: 0.2, draws: 500, name: bvar}
check: object bvar is FittedBVAR
why: The alternative: the Minnesota prior pulls each equation towards a random walk, which often forecasts better than an unrestricted VAR with many coefficients and a short sample.
look: Compare its coefficients with the VAR's: the prior pulls them towards a random walk, the second lags most.
```

Backtest it in the same way.

```pymetrix-step
title: Backtest the Bayesian VAR
stage: Evaluate
command: RunBacktest
settings: {fit: bvar, start: 2015Q1, end: 2022Q4, horizon: 4, name: bt_bvar}
check: object bt_bvar is Backtest
why: The same origins and horizon as the VAR's backtest, so the two models forecast the same quarters.
look: Its Accuracy tab next to the VAR's: which has the smaller RMSE for ly four quarters ahead?
```

Which forecasts output better four quarters ahead, and is the difference more than chance? The Diebold-Mariano
test (**Forecast → Compare backtests**) asks.

```pymetrix-step
title: Compare the backtests
stage: Evaluate
command: CompareBacktests
settings: {first: bt_var, second: bt_bvar, variable: ly, horizon: 4, name: dm}
check: object dm exists
why: A smaller RMSE over 32 origins may be luck; the test asks whether the difference in squared errors is more than chance, allowing for overlapping four-quarter forecasts.
look: The statistic and its p-value: a negative statistic favours the first backtest (the VAR). With 32 origins the test has limited power, and failing to reject does not show that the two are equally good.
```

Export the round and the VAR's track record with the `forecast_round` template in the `policy` theme
(**File → Export report**; Markdown, HTML, Word or Excel).

```pymetrix-step
title: Export the report
stage: Report
command: ExportReport
settings: {objects: "forecast_round, bt_var", path: lab3_report.html, title: "Quarterly forecast, 2024Q4", template: forecast_round, theme: policy}
check: step ExportReport ran on forecast_round
why: The round is published with its judgement and the model's track record, so a reader can see what is the model's and what is judgement.
look: Open lab3_report.html: the fan chart, the paths, the judgement with its reason, the track record, and the appendix of where the numbers come from (method, sample, data vintages, pymetrix version). Could a colleague reproduce them?
```

You have made a forecast round from start to finish: the data in logs, a VAR checked before use, a baseline with its
uncertainty, judgement recorded with its reason, the revisions since June explained, the model's track record
against an alternative, and a report that says where each number came from.

**File → Export script** gives every step you ran as Python: compare it with the notebook. Save the project, close
it and open it again: the steps run again and the round is unchanged.

**Where to go next**

- The lab notebook (`examples/course/03_forecast_round.ipynb`) does the same in code, and goes further: a
  structural fiscal scenario stored as a round, so it reads as a revision to the baseline, and a nowcast of 2025Q1
  from monthly data (**Estimate → Mixed frequency**).
- The forecasting course pack (`examples/forecasting`): F2 on comparing forecasts (Diebold-Mariano and, for nested
  models, Clark-West), F3 on combining them, F4 on density forecasts (are the fan's bands right?) and F6 on breaks.
- Related dialogs: **Forecast → Conditional forecast** (judgement as an assumed path for one variable, with the
  others following), **Forecast → Combine forecasts** and **Forecast → Compare many forecasts**.
- The help page "The forecast workflow", and the `forecast_round` demo project (**File → Open demo project…**).
