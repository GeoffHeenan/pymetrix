---
title: Lab 2. From estimates to a model, and back again
course: Example course (pymetrix)
links: [model-solution, least-squares, diagnostic-tests, commands/Solve model]
---

# Lab 2. From estimates to a model, and back again

An exercise from start to finish, in the app or in the [lab notebook](02_lab.ipynb).

1. **Load** the example data: **File → Load example data**, `sample_macro`.
2. **Estimate** consumption, `c ~ y + c[-1]` with `hac` standard errors, and investment, `i ~ y[-1]`, in
   **Estimate → Equation**. Test the consumption residuals with **Test → Run test**, `breusch_godfrey`.
3. **Build a model**: define the identity `y = c + i + g` in **Model → Define equation**, then solve the
   two fits and the identity over 2015Q1-2024Q4 in **Model → Solution → Solve model**. How closely does simulated output
   follow the data?
4. **Scenario**: solve 2020Q1-2024Q4 again in **Model → Solution → Solve model**, with **Shift** `g=5` in its Scenario section,
   saved as `more_g`; compare `y` in the two solutions. What is the multiplier, and why does it grow?
5. **Save** the project (**File → Save project**), close it, **open** it again, and check the results are
   unchanged. The History window shows each step; **File → Export script** gives the same steps as Python.
