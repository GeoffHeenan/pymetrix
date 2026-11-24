"""What phase 1 leaves out, and why: the report's reasons for the commands of the later phases (design note,
section 2: phase 2, estimation, tests and models; phase 3, control flow; phase 4, what stays in the report), and the
expression functions pymetrix has no equivalent of (the audit, docs/known-gaps.md).

Every statement is either translated or left out with one of these reasons: nothing is dropped silently.
"""

from __future__ import annotations

__all__ = ["GAP_FUNCTIONS", "PROGRAM_FUNCTIONS", "PHASE2", "PHASE2_COMMANDS", "PHASE2_METHODS", "PHASE3", "PHASE3_WORDS", "PHASE4_COMMANDS",
           "PHASE4_METHODS", "STATISTICS", "left_reason"]

PHASE2 = "phase 2 (estimation, tests and models)"
PHASE3 = "phase 3 (control flow)"

# phase 2: the estimation commands and what pymetrix calls them
PHASE2_COMMANDS = {
    "ls": "Equation(\"y ~ x\").fit(data) (Model → Estimate equation)",
    "tsls": "Equation(...).fit(data, method=IV(endog=[...], instruments=[...]))",
    "gmm": "Equation(...).fit(data, method=GMM(...))",
    "liml": "Equation(...).fit(data, method=LIML(...))",
    "arch": "Equation(..., variance=GARCH(arch=1, garch=1)).fit(data, method=ML())",
    "binary": "Equation(..., family=Logit()) or Probit()",
    "logit": "Equation(..., family=Logit())",
    "probit": "Equation(..., family=Probit())",
    "ordered": "OrderedProbit or OrderedLogit",
    "censored": "Tobit",
    "count": "Poisson or NegativeBinomial",
    "qreg": "Quantile",
    "heckit": "Heckman",
    "ardl": "ARDL",
    "nls": "Equation in explicit form",
    "stepls": "none (stepwise least squares)",
    "breakls": "bai_perron",
    "threshold": "ThresholdAR",
    "switchreg": "MarkovSwitching",
    "glm": "Equation(..., family=...)",
    "equation": "Equation(...) (Model → Define equation, Estimate equation)",
    "var": "VAR([...], lags=..., exog=[...])",
    "system": "System([...]) with SUR, ThreeSLS, FIML",
    "model": "read_eviews(text, data=...)",
    "sspace": "read_eviews_sspace(text)",
    "coef": "",
}

PHASE2_METHODS = {
    **{m: PHASE2_COMMANDS.get(m, "Equation(...).fit(data)") for m in
       ("ls", "tsls", "gmm", "liml", "arch", "binary", "ordered", "censored", "count", "qreg", "heckit", "ardl", "nls",
        "stepls", "breakls", "threshold", "switchreg", "glm", "logit", "probit")},
    "ec": "VECM", "sur": "System(...) with SUR", "3sls": "System(...) with ThreeSLS", "fiml": "System(...) with FIML",
    "wald": "eq.wald(\"x = 0\")", "auto": "eq.test(\"breusch_godfrey\", lags=...)", "hettest": "eq.test(\"breusch_pagan\") or \"white\"",
    "archtest": "eq.test(\"arch\")", "reset": "eq.test(\"reset\")", "chow": "eq.test(\"chow\")", "hist": "eq.test(\"jarque_bera\")",
    "rls": "eq.recursive_residuals()", "testdrop": "eq.wald(\"x = 0\")", "testadd": "none (omitted variables: a gap)",
    "uroot": "data.test(\"adf\", \"x\", ...)", "coint": "data.johansen([...])", "forecast": "eq.forecast(data, span, dynamic=True)",
    "fit": "eq.forecast(data, span, dynamic=False)", "makeresids": "data.with_series(r=eq.resid)",
    "append": "read_eviews(text, data=...)", "merge": "read_eviews(text, data=...)",
    "solve": "model.solve(data, span, solver=Newton())", "solveopt": "model.solve(..., solver=Newton() or GaussSeidel())",
    "addassign": "Model.add_factors(data, span)", "addinit": "Model.add_factors(data, span)",
    "scenario": "Scenario(set=..., fix=...)", "override": "Scenario(set=...)", "exclude": "Scenario(fix=...)",
    "makegroup": "the model's endogenous variables (model.endogenous)", "makemodel": "read_eviews(text, data=...)",
    "results": "eq.summary()", "output": "eq.summary()", "spec": "the equation's text",
}

# phase 3: control flow, subroutines and included files
PHASE3_WORDS = {
    "for": "a for loop", "next": "the end of a for loop", "while": "a while loop", "wend": "the end of a while loop",
    "if": "an if block", "else": "an else branch", "endif": "the end of an if block", "subroutine": "a subroutine",
    "endsub": "the end of a subroutine", "call": "a subroutine call", "include": "an included program",
    "exitloop": "leaving a loop", "return": "leaving a subroutine", "stop": "stopping the program",
}

# phase 4: what stays in the report, with the reason and the pymetrix equivalent
_GRAPHS = ("graphs are not translated: their formatting has no one-to-one mapping", "data.plot(...), Figure, Data → Plot series")
_TABLES = ("tables are made by results, not cell by cell", "Table, a result's summary()")
_DISPLAY = ("display only", "the Results area; print(...) in a script")
_SETTINGS = ("a program setting or display", "none needed")
_EXTERNAL = ("runs an external program", "none")
PHASE4_COMMANDS = {
    "graph": _GRAPHS, "table": _TABLES, "spool": ("output collections", "Report, File → Export report"),
    "freeze": ("a view becomes an object in EViews; in pymetrix, results already are objects", "the result itself, by name"),
    "show": _DISPLAY, "print": _DISPLAY, "output": _DISPLAY, "close": _DISPLAY, "logmode": _SETTINGS, "tic": _SETTINGS,
    "toc": _SETTINGS, "statusline": _SETTINGS, "setmaxerrs": _SETTINGS, "pon": _SETTINGS, "poff": _SETTINGS, "cd": _SETTINGS,
    "mode": _SETTINGS, "logmsg": _SETTINGS, "logclear": _SETTINGS, "logsave": _SETTINGS, "seterr": _SETTINGS, "wfstats": _DISPLAY,
    "shell": _EXTERNAL, "spawn": _EXTERNAL, "xopen": _EXTERNAL, "xrun": _EXTERNAL, "xget": _EXTERNAL, "xput": _EXTERNAL,
    "xclose": _EXTERNAL, "exec": ("runs another program", "translate that program too, then run its script first"),
    "run": ("runs another program", "translate that program too, then run its script first"),
    "text": ("text objects are notes", "a note in the project"),
}
PHASE4_METHODS = {
    **dict.fromkeys(("line", "bar", "scat", "scatpair", "xyline", "area", "spike", "pie", "boxplot", "dot", "errbar", "hilo",
                     "distplot", "qqplot", "cdfplot", "kdensity", "addtext", "setelem", "axis", "legend", "align", "options",
                     "draw", "name", "datelabel", "scale", "addarrow", "setfont", "setobslabel", "template", "displayname_graph"),
                    _GRAPHS),
    **dict.fromkeys(("setformat", "setwidth", "setjust", "setlines", "setfillcolor", "settextcolor", "setmerge", "setindent",
                     "title", "save", "comment"), _TABLES),
    "sheet": _DISPLAY, "stats": _DISPLAY, "statby": _DISPLAY, "freq": _DISPLAY, "correl": _DISPLAY, "cor": _DISPLAY,
    "cov": _DISPLAY, "show": _DISPLAY, "print": _DISPLAY, "display": _DISPLAY,
}

# EViews functions that phase 1 translates only as a scalar's whole right-hand side: the sample's statistics, and
# pandas' method that gives each (over the sample's rows, missing values skipped)
STATISTICS = {"@mean": "mean", "@sum": "sum", "@min": "min", "@max": "max", "@stdev": "std", "@median": "median",
              "@obs": "count"}

# the audit (design note, phase 1): EViews expression functions found in the validation programs or named in the
# design that pymetrix's expressions have no equivalent of. Each is a gap in docs/known-gaps.md.
GAP_FUNCTIONS = {
    "@year": "the period's year", "@quarter": "the period's quarter", "@month": "the period's month", "@day": "the day",
    "@date": "the period's date number", "@obsnum": "the observation number", "@obsid": "the observation number",
    "@round": "rounding", "@floor": "rounding down", "@ceiling": "rounding up", "@nan": "replacing missing values",
    "@isna": "testing for a missing value", "@iif": "a choice by a condition", "nrnd": "normal random numbers",
    "rnd": "uniform random numbers", "@rnorm": "normal random numbers", "@runif": "uniform random numbers",
    "@cumsum": "a cumulative sum over the sample", "@sumsby": "sums by group", "@minsby": "minimums by group",
    "@maxsby": "maximums by group", "@meansby": "means by group", "@obssmpl": "the sample's number of observations",
    "@trend": "a trend from the data's dates (eviews_series makes it for a model; not for data work yet)",
    "@seas": "a seasonal dummy from the data's dates (eviews_series makes it for a model; not for data work yet)",
    "@elem": "a value at a date (eviews_series makes it for a model; not for data work yet)",
    "@log10": "the base-10 logarithm",
}


# EViews's programming-language functions (strings, dates as numbers, objects, matrices): phase 3
PROGRAM_FUNCTIONS = frozenset({
    "@val", "@str", "@word", "@wcount", "@replace", "@isobject", "@wfind", "@wfindnc", "@right", "@left", "@mid", "@lower",
    "@upper", "@len", "@length", "@instr", "@trim", "@ltrim", "@rtrim", "@dateval", "@datestr", "@dateadd", "@datediff",
    "@dtoo", "@otod", "@strdate", "@seriesname", "@count", "@endog", "@exog", "@rows", "@columns", "@rowextract",
    "@columnextract", "@sort", "@resample", "@wlookup", "@wsplit", "@wjoin", "@wkeep", "@wdrop", "@wintersect",
    "@wunion", "@wnotin", "@wsort", "@winterleave", "@wcross", "@pagelist", "@pagefreq", "@pagesmpl", "@pagename",
    "@wfname", "@getnextname", "@env", "@fileexist", "@folderexist", "@linepath", "@evpath", "@tablenames", "@vernum",
    "@ver", "@errorcount", "@lasterrstr", "@isvalidname", "@makevalidname", "@chr", "@hasoption", "@equaloption",
    "@option", "@inner", "@transpose", "@inverse", "@identity", "@vec", "@unvec", "@filledmatrix", "@filledvector",
    "@convert", "@zeros", "@ones", "@getmaindiagonal", "@makediagonal", "@rank", "@det", "@trace", "@cholesky",
})


def left_reason(why: str, equivalent: str = "") -> str:
    """A reason as the report gives it: why, then pymetrix's equivalent where there is one."""
    return f"{why}; in pymetrix: {equivalent}" if equivalent and equivalent != "none" else why
