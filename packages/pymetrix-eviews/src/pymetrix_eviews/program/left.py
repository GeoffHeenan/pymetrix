"""What the translation leaves out, and why: the report's reasons for what stays in the report (design note, section
2, phase 4), and the expression functions pymetrix has no equivalent of (the audit, docs/known-gaps.md); the words
of control flow (phase 3, ``control.py``), named in the report when one is out of place. Phase 2's own refusals (an estimator's option not mapped, testadd, the
extreme-value link) are given where they are met, in ``estimation.py`` and ``models.py``.

Every statement is either translated or left out with one of these reasons: nothing is dropped silently.
"""

from __future__ import annotations

from pymetrix.translation import left_reason  # a reason as the report gives it (core, extension API 1.11)

__all__ = ["GAP_FUNCTIONS", "PROGRAM_FUNCTIONS", "PHASE2_COMMANDS", "PHASE3_WORDS", "PHASE4_COMMANDS", "PHASE4_METHODS",
           "STATISTICS", "left_reason"]

# state-space objects: read whole by the core reader, not line by line
PHASE2_COMMANDS = {"sspace": "read_eviews_sspace(text)"}

# phase 3: control flow, subroutines and included files (control.py translates them)
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
# The rest of the audit now translates (the core reader's render_eviews_expression with data=True): @year, @quarter,
# @month, @day, @obsnum, @obsid, @round, @floor, @ceiling, @log10, @nan, @isna, @iif, na, nrnd, rnd, @rnorm, @runif,
# @cumsum and @sumsby, @meansby, @minsby, @maxsby (by a series' name).
GAP_FUNCTIONS = {
    "@date": "the period's date number: use @year, @quarter, @month and @day",
    "@obssmpl": "the sample's number of observations, a number: give a scalar its value first (!n = @obssmpl)",
    "@trend": "a trend from the data's dates (eviews_series makes it for a model; not for data work yet)",
    "@seas": "a seasonal dummy from the data's dates (eviews_series makes it for a model; not for data work yet)",
    "@elem": "a value at a date (eviews_series makes it for a model; not for data work yet)",
}


# EViews's programming-language functions (strings, dates as numbers, objects, matrices): program values, not series;
# control.py translates the string and word functions and a group's members, the rest are gaps
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
