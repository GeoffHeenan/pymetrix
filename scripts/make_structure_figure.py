"""Draw docs/design/pymetrix-structure.png: the core and its add-ins, built and planned, with each one's audience,
functionality and added dependencies (docs/design/plugin-split.md). Rerun after the plan changes:

    python scripts/make_structure_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, PAPER, ACCENT, MUTED, PLAN = "#1b2a2f", "#f4efe6", "#e07a2f", "#5f6f72", "#fbfaf7"
fig = plt.figure(figsize=(17, 11.2), dpi=110)
ax = fig.add_axes((0, 0, 1, 1)); ax.set_xlim(0, 170); ax.set_ylim(0, 112); ax.axis("off")
fig.patch.set_facecolor("white")

def node(x, y, w, h, name, status, audience, does, deps, planned=False, core=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.2",
                                fc=PAPER if not planned else PLAN, ec=ACCENT if core else INK,
                                lw=2.6 if core else 1.6, ls=(0, (5, 3)) if planned else "-"))
    ax.text(x + 2, y + h - 2.4, name, fontsize=13.5, fontweight="bold", color=INK, va="top")
    ax.text(x + w - 2, y + h - 2.6, status, fontsize=8.5, color=ACCENT if planned else MUTED, va="top", ha="right",
            fontweight="bold")
    rows = [("Who", audience), ("What", does), ("Adds", deps)]
    yy = y + h - 7.4
    for label, text in rows:
        ax.text(x + 2, yy, label, fontsize=8.6, color=MUTED, va="top", fontweight="bold")
        ax.text(x + 8.6, yy, text, fontsize=8.6, color=INK, va="top", linespacing=1.3)
        yy -= 1.95 * (text.count("\n") + 1) + 1.1

def link(x1, y1, x2, y2, planned=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, lw=1.2,
                                 color=MUTED, ls=(0, (4, 3)) if planned else "-"))

ax.text(4, 109.5, "PyMetrix: the core and its add-ins", fontsize=20, fontweight="bold", color=INK, va="top")
ax.text(4, 104.6, "Packages by audience, extras by dependency (docs/design/plugin-split.md).  Solid: built.  "
        "Dashed: planned.\nEvery add-in builds on the core alone (none depends on another), through the extension API, and appears in the app, the help and scripts.",
        fontsize=10, color=MUTED, va="top")

# core
node(4, 4, 162, 27, "pymetrix (core)", "BUILT · 0.1.0a19",
     "applied economists, central-bank and ministry analysts, students and lecturers",
     "data and connectors (FRED, SDMX, World Bank), expressions, econometrics (OLS, IV, panel, limited dependent,\n"
     "GMM, nonparametric), VARs and BVARs, state space, forecasting and forecast rounds with judgement, nowcasting,\n"
     "reports with provenance, the desktop app, notebooks, the MCP server for AI assistants, the plugin system",
     "numpy, pandas, scipy, pyarrow, statsmodels, matplotlib\n"
     "extras: [qt] app · [panel] linearmodels · [arch] · [ml] scikit-learn · [mcp] · [report]/[excel] · [jax] switch\n"
     "planned extras: statsforecast, LightGBM/XGBoost, pyfixest, doubleml, DBnomics, DuckDB/Polars", core=True)

# built add-ins
W, H = 39.5, 31
xs = [4, 45, 86, 127]
built = [
    ("pymetrix-finance", "BUILT", "finance and risk analysts,\nmarkets and stability teams",
     "volatility models, copulas and\nvines, VaR/ES, portfolios, credit",
     "arch\nplanned: [interactive] plotly;\n[full] QuantLib, cvxpy, pyvinecopulib"),
    ("pymetrix-labbook", "BUILT", "researchers; reviewers,\njournals and funders",
     "hash-chained record, verify by\nreplay, brief and plan, who acted,\nwhat was tried, auditor's MCP tools",
     "nothing\n[audit]: mcp"),
    ("pymetrix-psets", "BUILT", "users of PSETS databases\n(small-economy statistics)",
     "psets:// addresses, vintages,\nprovenance, publisher issues",
     "PSETS (from its repository)"),
    ("pymetrix-guide", "BUILT", "trainees, lecturers,\ncentral-bank trainers",
     "guided case studies in the app:\nsteps that open real dialogs\nand tick themselves off",
     "nothing"),
]
for x, (n, st, who, what, deps) in zip(xs, built):
    planned = not st.startswith("BUILT")
    node(x, 35.5, W, H, n, st, who, what, deps, planned=planned)

# planned, moving out of core
xs2 = [4, 45, 86, 127]
planned = [
    ("pymetrix-seasonal", "BUILT · from core", "statistics offices,\ndata teams",
     "X-13ARIMA-SEATS, JDemetra+\n(TRAMO-SEATS); classical and\nSTL stay in core",
     "Java, JPype, the x13as program"),
    ("pymetrix-macro", "LATER · from core", "modelling teams;\nDynare and IRIS users",
     "DSGE, OccBin, model estimation,\noptimal policy, sweeps; later\nscenario sets, HANK satellites",
     "[jax]: JAX, NumPyro/BlackJAX\n(NUTS, autodiff, particle filters)\n[full]: sequence-jacobian, oct2py"),
    ("pymetrix-assistant", "BUILT · stage 2", "analysts working with\nAI assistants",
     "the open app's session served\nto an outside assistant; Ask\nClaude or a local model",
     "mcp, uvicorn, anthropic,\nkeyring"),
    ("Third-party add-ins", "OPEN", "institutions, course authors,\nother developers",
     "their own estimators, data\nconnectors, models, course packs,\nreport templates",
     "their own; installed with pip,\nenabled in Help → Plugins"),
]
for x, (n, st, who, what, deps) in zip(xs2, planned):
    node(x, 70, W, 29, n, st, who, what, deps, planned=not st.startswith("BUILT"))

OUT = Path(__file__).resolve().parents[1] / "docs" / "design" / "pymetrix-structure.png"
fig.savefig(OUT, facecolor="white")
print(f"wrote {OUT}")
