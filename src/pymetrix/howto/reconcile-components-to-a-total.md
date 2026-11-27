---
summary: make quarterly components add up to their quarterly total, keeping each component's annual totals and holding some fixed.
audience: forecasters and national accountants
order: 14
---

# Reconcile national accounts components to a total

Quarterly estimates of the components of GDP, each made on its own (often by temporal disaggregation from annual
data), rarely add up to quarterly GDP. Move them as little as possible until they do, every quarter, while each
keeps its annual totals and the components you trust are not moved at all.

## Before you start

- The quarterly components and their total, in one dataset. This page uses three years of quarterly consumption,
  investment and government spending, and GDP, as `quarterly.csv`. The components were estimated one by one, and
  miss GDP by up to 5.2 in a quarter:

  ```text quarterly.csv
  quarter,cons,inv,gov,gdp
  2021Q1,152.4,39.1,30.0,220.0
  2021Q2,148.9,42.4,30.3,223.7
  2021Q3,147.5,43.1,30.6,225.8
  2021Q4,158.4,42.2,30.9,226.3
  2022Q1,153.4,40.2,31.2,226.5
  2022Q2,154.4,40.8,31.5,228.1
  2022Q3,158.1,42.3,31.8,231.4
  2022Q4,160.5,45.0,32.1,235.3
  2023Q1,159.6,46.3,32.4,238.0
  2023Q2,160.6,45.2,32.7,238.8
  2023Q3,160.6,43.7,33.0,238.9
  2023Q4,164.8,43.5,33.3,240.0
  ```

- The components' annual totals, which the quarters must keep, as `annual.csv`. Each year's components add up to the
  year's GDP; without that, no reconciliation can meet both:

  ```text annual.csv
  year,cons,inv,gov
  2021,607.2,166.8,121.8
  2022,626.4,168.3,126.6
  2023,645.6,178.7,131.4
  ```

- Which components to hold. Government spending comes from the government's own accounts, quarter by quarter, so it
  is held fixed here; consumption and investment take the gaps.

## In the app

1. File → Load data…, with Data file: quarterly.csv, Time column: quarter, Frequency: Quarterly and Save as: q.
   Again with Data file: annual.csv, Time column: year, Frequency: Annual and Save as: annual.
2. Data → Transform → Reconcile components…, with Components: cons, inv, gov, Total (high frequency): gdp,
   High-frequency data: q, Low-frequency totals: annual, Components held fixed: gov, Method: Denton (proportional),
   Conversion: Sum (flows) and Save as: rec. Replace the components in the data stays ticked: `cons` and `inv` in
   `q` become the reconciled series.
3. In the result's window: Summary (each component's largest and mean adjustment, and the largest gap before and
   after), Graph, Series and Adjustments (each component's change, quarter by quarter).

## In code

```python
from pymetrix import Dataset, reconcile

q = Dataset.read("quarterly.csv", time="quarter", freq="Q")
annual = Dataset.read("annual.csv", time="year", freq="Y")
rec = reconcile(q, ["cons", "inv", "gov"], "gdp", low=annual, fixed=["gov"])
rec.diagnostics()                    # the largest gap: 5.2 before, nothing after
rec.adjustments["cons"]["2021Q4"]    # -4.85: consumption takes most of 2021Q4's gap of 5.2
q = q.with_series(cons=rec.series["cons"], inv=rec.series["inv"])
```

## Check the result

- Every quarter, `cons + inv + gov` equals `gdp`, to rounding: the Summary's largest gap after is of the order of
  1e-13.
- Each year, the quarters of `cons` and of `inv` add up to `annual`'s value: 607.2 and 166.8 in 2021.
- `gov` has not moved: its adjustments are zero.
- The adjustments are spread smoothly over the quarters: consumption, nearly four times as large as investment,
  takes most of each gap (-4.85 of 2021Q4's -5.2). Proportional Denton keeps each component's
  quarter-to-quarter growth as close as it can to its own; the same problem solved directly, as a least-squares
  problem under the adding-up and annual constraints, gives the same series.

## Pitfalls

- **The annual totals must add up.** If the components' annual totals do not add to the total's (or a fixed
  component does not fit them), the constraints contradict each other and the step is refused, saying by how much.
  Correct the data, or hold fewer components fixed.
- **Proportional Denton needs components without zeros.** For a series that is zero, or changes sign (changes in
  inventories, net exports), choose Denton (additive), or Weighted least squares (in proportion to the weights) with
  a weight for each component (Component weights, under Advanced settings).
- **The conversion must match the data.** Sum (flows) for flows such as GDP; Average (rates) for rates and indices;
  First or Last for stocks.
- **Components that enter with a minus sign.** Imports subtract from GDP: reconcile with the imports as a negative
  series, and prefer the weighted method, which moves a negative component as it moves a positive one.
- **The gaps are information.** Large or systematic gaps say the component estimates disagree with the total: look at
  them (the Summary's gaps before) before letting them be spread away.

## See also

- [Temporal disaggregation](pymetrix:topic/temporal-disaggregation), in the help.
- [Seasonally adjust a series](seasonally-adjust-a-series.md): adjusted components need reconciling too.
- [The user guide's Temporal disaggregation](../../../docs/guide.md#temporal-disaggregation): Chow-Lin, Denton and
  the others, from annual data to quarters, before the components are reconciled.
