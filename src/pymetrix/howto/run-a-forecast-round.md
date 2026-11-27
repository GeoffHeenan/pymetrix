---
summary: make a baseline forecast round, adjust it with judgement and a reason, add a scenario, keep the rounds in a store, compare them and write the report.
audience: forecasters in an institution
order: 5
---

# Run a forecast round: baseline, judgement, scenarios, report

Turn a model's forecast into a forecast round: a record of the paths, when they were made, the last period of data
used, notes, and every judgemental change with its reason. Add a scenario as a round of its own, keep both in a
forecast store so later rounds can be compared with them and judged against what happened, and write the round's
report.

## Before you start

- A model that forecasts. This page uses the VAR of [Estimate a VAR, then impulse responses and a
  forecast](estimate-a-var.md), with one lag: output growth `dly` and investment growth `dli`, per cent a quarter,
  from the example `sample_macro` (data to 2024Q4).
- The round's name, the same in every place it is kept: here "2025Q1 baseline".
- The judgement to add, with its reason, and the scenario: here a strike that takes 0.3 points off output growth in
  2025Q1 and 0.1 in 2025Q2, and a scenario in which investment falls 2 per cent a quarter through 2025.
- A folder for the store. Every round goes in it, and stays as saved.

## In the app

1. File → Load example data…, with Example dataset: sample_macro and Save as: data. Data → Derived series…, with
   Data: data, New series name: dly and Formula: `100*dlog(y)`; again with New series name: dli and Formula:
   `100*dlog(i)`.
2. Estimate → VARs and local projections → VAR…, with Series: dly, dli, Lags: 1 and Save as: var. Forecast →
   Forecast…, with Estimated model: var, Forecast start: 2025Q1, Forecast end: 2026Q4 and Save as: fc.
3. Forecast → Make round…, with Forecast or solution: fc, Round name: 2025Q1 baseline, Notes: VAR(1) in output and
   investment growth, and Save as: baseline. The round's window shows Paths, Graph (a fan chart) and Judgement.
4. Forecast → Adjust round…, with Round: baseline, Variable: dly, Changes by period: 2025Q1=-0.3, 2025Q2=-0.1,
   Reason: a strike in January, and Adjust by: Add to the path. The Judgement view lists each change with the value
   before and after.
5. The scenario: Forecast → Conditional forecast…, with Estimated VAR: var, Forecast start: 2025Q1, Forecast end:
   2026Q4, Conditions: dli[2025Q1:2025Q4]=-2 and Save as: weak. Then Forecast → Make round…, with Forecast or
   solution: weak, Round name: 2025Q1 weak investment and Save as: scenario.
6. Forecast → Save round…, with Round: baseline and Store folder: forecasts; again with Round: scenario.
7. Forecast → Compare rounds…, with Earlier round: 2025Q1 baseline, Later round: 2025Q1 weak investment, Store
   folder: forecasts and Save as: revisions: both paths of each variable, and the difference, period by period.
8. File → Export report…, with Results: baseline, scenario, Write to: round.html, Report title: Forecast round 2025Q1
   and Views: forecast_round (fan charts, paths, judgement). Word (`.docx`) and Excel (`.xlsx`) work the same way.

## In code

```python
from pymetrix import VAR, Dataset, ForecastRound, ForecastStore, Report

data = Dataset.example("sample_macro").derive(dly="100*dlog(y)", dli="100*dlog(i)")
var = VAR(["dly", "dli"], lags=1).fit(data)
fc = var.forecast(data, ("2025Q1", "2026Q4"))
baseline = ForecastRound.from_forecast("2025Q1 baseline", fc, data=data, notes="VAR(1) in output and investment growth")
baseline = baseline.adjust("dly", {"2025Q1": -0.3, "2025Q2": -0.1}, "a strike in January")
baseline.to_frame()["dly"].iloc[0]                    # 0.33: the model's 0.63, less 0.3
baseline.judgement()                                  # each change: before, after, reason
```

The scenario, the store and the report:

```python
weak = var.conditional_forecast(data, ("2025Q1", "2026Q4"),
                                {"dli": {"2025Q1": -2.0, "2025Q2": -2.0, "2025Q3": -2.0, "2025Q4": -2.0}})
scenario = ForecastRound.from_forecast("2025Q1 weak investment", weak, data=data)
scenario.to_frame()["dly"].iloc[0]                    # -0.79: output growth, given investment's fall
store = ForecastStore("forecasts")
store.save(baseline)
store.save(scenario)
revisions = store.compare("2025Q1 baseline", "2025Q1 weak investment")   # dly revision -1.12 in 2025Q1
report = Report.from_template("forecast_round", {"baseline": baseline, "scenario": scenario},
                              title="Forecast round 2025Q1")
report.save("round.html")
```

## Check the result

- The baseline's last period of data: 2024Q4 (`baseline.data_end`). A round made from data that ends earlier than
  you think is the commonest mistake in a round.
- The Judgement view: two rows, dly in 2025Q1 from 0.63 to 0.33 and in 2025Q2 from 0.92 to 0.82, each with the
  reason. The paths before judgement are kept (`baseline.base`), so the judgement's effect can be measured later.
- The scenario's conditions hold: `dli` is -2 in every quarter of 2025; from 2026 the VAR's dynamics take over
  (1.72 in 2026Q1). Output growth is -0.79 in 2025Q1: the VAR's most likely path of the other shocks given the conditions
  (Waggoner and Zha), so it does not depend on the ordering of the variables.
- The store folder has a Parquet file of each round's paths and a JSON record of its notes, data end and judgement:
  `2025Q1_baseline.parquet` and `.json`, and the same for the scenario.
- The revisions table: dly's revision in 2025Q1 is -1.12, the scenario's -0.79 less the baseline's 0.33.

## Pitfalls

- **A scenario made from the model does not carry the baseline's judgement.** The weak-investment round starts from
  the VAR, not from the adjusted baseline: the strike is not in it. Adjust the scenario's round too, or compare it
  with the baseline before judgement (`baseline.base`).
- **A saved round is not changed.** Saving a different round under a name the store has is refused unless Overwrite
  is ticked, which loses the earlier record. Give a new round a new name.
- **Adjust by: Set the path replaces values.** Add to the path (the default) adds the changes; Set replaces the
  path in those periods, whatever the model said.
- **The intervals are the model's.** Judgement moves the central path, not the bands; a scenario's intervals are
  conditional on the conditions holding exactly.
- **Rounds are compared by name in the store.** Compare rounds… reads the store folder, not the session: save both
  rounds first.

## See also

- [The forecast workflow](pymetrix:topic/forecast-workflow), in the help.
- [Check a forecast's record (backtest)](check-a-forecasts-record.md): how the model forecast in the past.
- [Make a report](make-a-report.md): the report outline, built as you go.
- [The user guide's Forecast rounds and backtests](../../../docs/guide.md#forecast-rounds-and-backtests) and
  [Conditional forecasts](../../../docs/guide.md#conditional-forecasts): the store's accuracy table, fan charts and
  structural scenarios.
