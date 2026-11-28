"""The plugin's help topic. Phase 1 is code only: there is no dialog yet (docs/design/heterogeneous-agents.md)."""

from __future__ import annotations

from pymetrix.extension import add_help_topic

REFERENCES = (
    "Auclert, A., Bardóczy, B., Rognlie, M. and Straub, L. (2021). Using the sequence-space Jacobian to solve and "
    "estimate heterogeneous-agent models. Econometrica 89(5), 2375-2408.",
    "Carroll, C. D. (2006). The method of endogenous gridpoints for solving dynamic stochastic optimization problems. "
    "Economics Letters 91(3), 312-320.",
    "Krusell, P. and Smith, A. A. (1998). Income and wealth heterogeneity in the macroeconomy. Journal of Political "
    "Economy 106(5), 867-896.",
    "Young, E. R. (2010). Solving the incomplete markets model with aggregate uncertainty using the Krusell-Smith "
    "algorithm and non-stochastic simulations. Journal of Economic Dynamics and Control 34(1), 36-41.",
)

add_help_topic("heterogeneous-agents", "Heterogeneous-agent models (pymetrix-ha)", """
    In a heterogeneous-agent model households differ in income and wealth, and save against income risk they cannot
    insure, so that who holds wealth changes how the economy responds to a shock. Krusell and Smith (1998) is the
    standard example. The pymetrix-ha plugin solves such models with sequence-jacobian, the package of Auclert,
    Bardóczy, Rognlie and Straub (2021), and returns pymetrix results. Install it with
    `pip install "pymetrix[ha]"` (in a clone, `pip install -e packages/pymetrix-ha`). It brings numba, which
    compiles the household routines the first time they run in a session (a few seconds).

    **The method.** A model is a set of blocks:
    - *Simple blocks* are aggregate equations, written as a Python function (`K(-1)` for a lag, `r(+1)` for a lead).
    - A *het block* is the household problem, given as one backward step. The endogenous grid method (Carroll 2006)
      iterates it to the steady-state policies. Young's (2010) histogram method then finds the distribution
      over income states and asset points, with no simulation noise.

    The steady state solves the calibration: unknowns, such as the discount factor, are found to meet targets, such
    as an interest rate. Linearised dynamics use sequence-space Jacobians. Each block's T x T Jacobian is the
    response of an output at each date to an input at each date; the household's comes from the fake-news
    algorithm. Chained along the model, they give the general-equilibrium Jacobians G. G maps any path of a shock
    to every output's path, so an impulse response is a matrix product. A nonlinear perfect-foresight transition
    after an unexpected shock is found by Newton's method on the unknowns' paths, with the steady state's
    Jacobian.

    **In Python** (there is no dialog yet):
    ```
    from pymetrix_ha.examples import krusell_smith
    ks = krusell_smith()                    # an HAModel: the blocks, calibration, steady state, equilibrium, shocks
    ss = ks.steady_state()                  # ss.table(), ss["K"], ss.checks()
    J = ks.jacobian("household", ["r", "w"], T=5, steady=ss)    # J["C", "r"], J.table("C", "r")
    lin = ks.linearize(ss)                  # lin.jacobian("K", "Z") is G
    lin.irf(40).plot()                      # a pymetrix IRF
    path = ks.solve(steady=ss)              # the nonlinear transition, a Solution; path.linear beside it
    ```
    Your own model is `HAModel(blocks, calibration=..., steady_state={"unknowns": ..., "targets": ...},
    equilibrium={"unknowns": [...], "targets": [...]}, shocks={"Z": AR1(rho, sd)})`. Its blocks are written with
    sequence-jacobian's decorators, or with `pymetrix_ha.simple` and `pymetrix_ha.het`, which check them.

    **Writing blocks.** sequence-jacobian reads a block's outputs from the text of its last `return` line.
    - Assign each output to a name and return the names on one line (`return r, w, Y`). A returned expression is
      refused when the model is made, with the block named.
    - A block defined in text run by `exec` has no source text to read. Put such blocks in a .py file, or run the
      text with `pymetrix_ha.blocks_from_source(text)`.

    **Results.** The steady state is an `HASteadyState`, with a table of the aggregates, the unknowns found and
    the targets' residuals. Its checks give the distribution's mass and the aggregates against the distribution.
    Jacobians are T x T arrays by output and input. Impulse responses are an `IRF`, as for a DSGE model.
    Transitions are a `Solution` in levels, dated from `start` (labels: period 0 is the shock's date).
    Provenance records:
    - the calibration, and each block with a fingerprint of its source;
    - the grids, their sizes and a hash of their points;
    - every tolerance passed to the library (`HAOptions`: sequence-jacobian 1.0.0's defaults, passed explicitly);
    - the steady state's evaluations and each Newton iteration's error;
    - the library's version, with numba's and NumPy's.
    Nothing is random in these results, so no seed is recorded. The library's printing is kept in provenance, not
    shown.

    **Checks.** The Krusell-Smith example reproduces the numbers sequence-jacobian's Krusell-Smith notebook prints:
    the household Jacobian of consumption on the interest rate (8 digits, with the notebook's grid, `agrid`), the
    firm's Jacobian, and the Newton iterations' errors. Every result equals sequence-jacobian called directly. A
    representative-agent model written as simple blocks gives pymetrix's own first-order responses.

    **Not yet.** Estimation, standard household blocks, one- and two-asset HANK examples, decompositions,
    distributional results, a dialog and model text are later phases (docs/known-gaps.md). The responses are first
    order in aggregate shocks; transitions are nonlinear but perfect foresight.
    """, REFERENCES, ("forward-looking-models",))
