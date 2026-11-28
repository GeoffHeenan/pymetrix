# pymetrix-ha

Heterogeneous-agent macro models for pymetrix: households that differ in income and wealth, as in Krusell and Smith
(1998). It wraps sequence-jacobian, the sequence-space Jacobian method of Auclert, Bardóczy, Rognlie and Straub
(2021), and returns pymetrix results with provenance:

- the steady state, with its distribution, as a table;
- block and general-equilibrium Jacobians;
- impulse responses as pymetrix `IRF` objects;
- nonlinear perfect-foresight transitions as a `Solution`.

It is a plugin because it brings numba, a large dependency that supports new Python versions late. It is not in
`pymetrix[all]`.

## Install

```bash
pip install "pymetrix[ha]"                 # or, from the repository:
pip install -e packages/pymetrix-ha
```

It installs sequence-jacobian 1.0, numba, NumPy and SciPy. sequence-jacobian's release declares no dependencies of its
own, so the plugin declares them. numba compiles the household routines the first time they run in a session (a few
seconds).

## Use

This first version is code only; a dialog comes later (docs/design/heterogeneous-agents.md).

```python
from pymetrix_ha.examples import krusell_smith

ks = krusell_smith()                              # the model of sequence-jacobian's Krusell-Smith notebook
ss = ks.steady_state()                            # ss.table(), ss["K"], ss.checks()
J = ks.jacobian("household", ["r", "w"], T=5, steady=ss)
J["C", "r"]                                       # the household Jacobian the notebook prints
lin = ks.linearize(ss)                            # the general-equilibrium Jacobians G
lin.irf(40).plot()                                # a pymetrix IRF
path = ks.solve(steady=ss)                        # the nonlinear transition (a Solution); path.linear beside it
path.provenance.resolved["iterations"]            # each Newton iteration's error
```

Your own model takes sequence-jacobian's blocks:

```python
from pymetrix_ha import AR1, HAModel, simple

@simple
def firm(K, L, Z, alpha, delta):
    r = alpha * Z * (K(-1) / L) ** (alpha - 1) - delta
    w = (1 - alpha) * Z * (K(-1) / L) ** alpha
    Y = Z * K(-1) ** alpha * L ** (1 - alpha)
    return r, w, Y

model = HAModel([household, firm, mkt_clearing], calibration={...},
                steady_state={"unknowns": {"beta": 0.98, "Z": 0.85, "K": 3.0}, "targets": {"r": 0.01, "Y": 1.0, "asset_mkt": 0.0}},
                equilibrium={"unknowns": ["K"], "targets": ["asset_mkt"]}, shocks={"Z": AR1(rho=0.8, sd=0.01)})
```

sequence-jacobian reads a block's outputs from the text of its last `return` line:

- Return names, on one line. A returned expression is refused when the model is made, with the block named.
- Blocks defined in text run by `exec` have no source to read. Put them in a .py file, or run the text with
  `pymetrix_ha.blocks_from_source(text)`.

Every tolerance is passed to the library from `HAOptions`, whose defaults are sequence-jacobian 1.0.0's own. The
library's printing is kept in provenance.

## How it is checked

- **The notebook's numbers.** The Krusell-Smith example reproduces what sequence-jacobian's notebook prints, with its
  grid (`grids.agrid`):
  - the household Jacobian `J['C']['r']`, to 8 digits;
  - the firm's Jacobian and their product;
  - the error at each Newton iteration of two transitions.
- **The library called directly.** The steady state, G, the impulse responses and the transitions equal
  sequence-jacobian's own calls, with its defaults, to 1e-12.
- **Hand checks:**
  - capital, wage and TFP from the calibration;
  - the distribution's mass and sums, and Walras' law;
  - the stationary distribution as the eigenvector of Young's transition matrix;
  - G by the notebook's step-by-step chain rule;
  - the nonlinear transition approaching the linear one at second order.
- **The representative-agent limit.** sequence-jacobian's RBC model gives pymetrix's own first-order responses
  (QZ), to 1e-8.

Without sequence-jacobian or numba, the tests are skipped.

License: BSD-3-Clause. sequence-jacobian is MIT; the example's blocks are adapted from its notebook
(THIRD-PARTY-NOTICES.md).
