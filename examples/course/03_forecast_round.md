---
title: Lab 3. A quarterly forecast round
course: Example course (pymetrix)
links: [forecast-workflow, var, bayesian-var, reports, commands/MakeRound]
---

# Lab 3. A quarterly forecast round

A forecasting team's quarter, in the app or in the [lab notebook](03_forecast_round.ipynb). With the
`pymetrix-guide` plugin, **View → Guide** takes you through it step by step ("Lab 3, guided").

1. **Load** the example data (**File → Load example data**, `sample_macro`) and add the logs
   `ly = log(y)`, `lc = log(c)` and `li = log(i)` with **Data → Generate series**.
2. **Forecast**: estimate a VAR in `ly, lc, li` with 2 lags (**Estimate → VARs and local projections → VAR**), forecast 2025Q1-2026Q4
   (**Forecast → Forecast**), and make it the "2024Q4 round" (**Forecast → Make round**).
3. **Judgement**: a port strike will cut output in early 2025. Lower `ly` by 0.004 in 2025Q1 and 0.002 in
   2025Q2 with **Forecast → Adjust round**, giving the reason. Look at the round's Judgement tab.
4. **Revisions**: save the round (**Forecast → Save round**) next to a June round made from the data to
   2024Q2, and compare the two (**Forecast → Compare rounds**). What drove the revision?
5. **Track record**: backtest the VAR from 2015Q1 to 2022Q4, four quarters ahead (**Forecast → Backtest**),
   then a Bayesian VAR (tick **Bayesian** in **Estimate → VARs and local projections → VAR**). Which forecasts output better, and is the
   difference significant?
6. **Report**: **File → Export report** with the round, the revisions and the backtest, as Word, Excel,
   HTML or Markdown; choose the `forecast_round` template and the `policy` theme for the standard layout.
   Read the appendix: where did each number come from?
7. **Save** the project, close it, open it again and check the round is unchanged. **File → Export script**
   gives every step as Python.
8. **Nowcast**: load `sample_monthly` (**File → Load example data**), add monthly growth rates with
   **Data → Generate series**, and estimate **Estimate → Mixed frequency → Bridge equation** and **Estimate → Mixed frequency →
   Dynamic factor model (nowcasting)** for GDP growth in 2025Q1. In code, `monthly.as_of("2025-02-15")` shows what was
   known in mid-February, and the factor model's `news` shows which figures moved its nowcast.
