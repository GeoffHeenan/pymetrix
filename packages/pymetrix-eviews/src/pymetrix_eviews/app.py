"""The plugin in the app: File → Import EViews workfile…, File → Translate EViews program…, the EViews translation
panel (``panel.py``) and the help topics, registered through the extension API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from pymetrix.extension import Command, Widget, add_help_topic, command, plugin_context

from . import PLUGIN
from .bundle import SUFFIX, _py_name, bundle_pages, save_export_program
from .panel import create_panel

if TYPE_CHECKING:
    from pymetrix.commands import Session

BUNDLE = "workfile bundle"
PROGRAM = "get the export program (experimental)"
_BUNDLE_ONLY = ("what", (BUNDLE,))

with plugin_context(PLUGIN):
    @command(menu="File/Import EViews workfile...", id="eviews:import_eviews")
    @dataclass(frozen=True)
    class ImportEViews(Command):
        """Read an EViews workfile's pages through a bundle, or save a copy of the program that writes one.

        A bundle is what the export program, pymetrix_export.prg, writes when it is run in EViews on an open workfile:
        each page's series as CSV, and a manifest (.evbundle) with their labels and units, the groups, samples,
        scalars, matrices, equations and models. Each page becomes a dataset named after it, read by one
        Dataset.read(...) line in the script. Replaying the step needs this plugin; ticking Save the data as Parquet
        writes each page next to the bundle as Parquet and reads it from there, so the script needs only pymetrix.

        The export program is EXPERIMENTAL: it was written from EViews's documentation and has not been run in
        EViews. Saving the workfile from EViews as CSV or Excel, and loading that with File > Load data, is the
        supported way in until it has been checked; Get the export program saves a copy with every point to check
        marked (verify).
        """

        topics: ClassVar[tuple[str, ...]] = ("eviews-import",)

        what: Annotated[Literal["workfile bundle", "get the export program (experimental)"],
                        Widget("choice", label="Import",
                               help="a workfile bundle the export program wrote; or save a copy of that program "
                                    "(experimental: untested in EViews)")] = BUNDLE
        path: Annotated[str, Widget("file", source="EViews workfile bundles (*.evbundle);;All files (*)", show_if=_BUNDLE_ONLY,
                                    label="Bundle", help="the bundle's manifest, <workfile>.evbundle, in the folder the "
                                                         "export program wrote")] = ""
        pages: Annotated[str, Widget("text", show_if=_BUNDLE_ONLY,
                                     help="the pages to read, separated by commas, as EViews names them; empty: every page")] = ""
        parquet: Annotated[bool, Widget("checkbox", show_if=_BUNDLE_ONLY, label="Save the data as Parquet",
                                        help="write each page next to the bundle as <page>.parquet and read it from there, "
                                             "so replaying the script does not need this plugin")] = False
        undated: Annotated[bool, Widget("checkbox", show_if=_BUNDLE_ONLY, advanced=True, label="Read dated pages as undated",
                                        help="rows numbered from 1, the dates kept as text: for frequencies pandas has no "
                                             "period for, such as semi-annual")] = False
        save_to: Annotated[str, Widget("savefile", source="EViews programs (*.prg);;All files (*)", show_if=("what", (PROGRAM,)),
                                       label="Save the program as",
                                       help="where to save pymetrix_export.prg (experimental: untested in EViews)")] = ""

        def _pages(self) -> list[str]:
            chosen = [p.strip() for p in self.pages.split(",") if p.strip()]
            return chosen or [p["name"] for p in bundle_pages(self.path)]

        def _plan(self) -> list[tuple[str, str, str]]:
            """(page, the dataset's name, the Parquet copy's path) for each page read."""
            folder = Path(self.path).parent
            return [(page, _py_name(page), str(folder / f"{_py_name(page)}.parquet")) for page in self._pages()]

        def validate(self, session: Session) -> list[str]:
            from pymetrix.errors import PyMetrixError

            if self.what == PROGRAM:
                return [] if self.save_to.strip() else ["choose where to save the export program"]
            if not self.path.strip():
                return [f"choose the bundle ({SUFFIX}) the export program wrote"]
            try:
                known = {p["name"].lower(): p for p in bundle_pages(self.path)}
                plan = self._plan()
            except PyMetrixError as exc:
                return [str(exc)]
            errors = []
            for page, _, _ in plan:
                info = known.get(page.lower())
                if info is None:
                    errors.append(f"the bundle has no page {page!r}; its pages: {', '.join(p['name'] for p in known.values())}")
                elif info["refused"] and not self.undated:
                    errors.append(f"page {info['name']} ({info['frequency']}): {info['refused']}; tick Read dated pages as undated")
            names = [name for _, name, _ in plan]
            twice = sorted({n for n in names if names.count(n) > 1})
            if twice:
                errors.append(f"two pages would be called {', '.join(twice)}: read them one at a time")
            return errors

        def _calls(self) -> list[Any]:
            from pymetrix.extension import make_call

            options = {"undated": True} if self.undated else {}
            if self.parquet:
                return [make_call("pymetrix:dataset.read", output=name, recorded_by=self.command_id, source=copy)
                        for _, name, copy in self._plan()]
            return [make_call("pymetrix:dataset.read", output=name, recorded_by=self.command_id, source=self.path, page=page, **options)
                    for page, name, _ in self._plan()]

        def run(self, session: Session) -> Any:
            from pymetrix.data import Dataset

            if self.what == PROGRAM:
                return save_export_program(self.save_to)
            made = None
            for (page, name, copy), call in zip(self._plan(), self._calls()):
                if self.parquet and Path(self.path).is_file():  # the copy follows the bundle while the bundle is there
                    Dataset.read(self.path, page=page, **({"undated": True} if self.undated else {})).write(copy)
                made = session.objects[name] = call.invoke(session.objects)
            return made

        def to_code(self) -> str:
            if self.what == PROGRAM:
                return f"from pymetrix_eviews import save_export_program\nsave_export_program({self.save_to!r})"
            return "\n".join(call.to_code() for call in self._calls())


PROGRAMS = "EViews programs (*.prg);;All files (*)"
DATA = "Workfile data (*.evbundle *.csv *.xlsx *.xls);;All files (*)"


with plugin_context(PLUGIN):
    @command(menu="File/Translate EViews program...", id="eviews:translate_program")
    @dataclass(frozen=True)
    class TranslateEViewsProgram(Command):
        """Translate an EViews program (.prg) into a readable pymetrix script and a report on every line.

        The translation opens in the EViews translation panel: each line of the program with its status (translated,
        translated with a note, left out, error) and the reason, the program and the script side by side, and Run as
        steps, which makes the program's work steps of this session, each recorded and undoable. The step itself
        only translates: it reads the program, keeps it, its script and its report in the project, and changes no
        data.

        Workfile data is what the program's wfopen opens: the bundle the export program wrote, or a CSV or Excel file
        saved from the workfile in EViews (give its date column and frequency). The program's arguments are what
        follows its name in EViews's exec (exec macro.prg us 2020q1): they become %0, %1, ..., the script's
        parameters.
        """

        topics: ClassVar[tuple[str, ...]] = ("eviews-programs",)

        path: Annotated[str, Widget("file", source=PROGRAMS, label="Program", help="the EViews program to translate (.prg); "
                                    "programs it includes are read from its folder")] = ""
        data: Annotated[str, Widget("file", source=DATA, label="Workfile data",
                                    help="what the program's wfopen opens: the bundle the export program wrote, or a CSV or "
                                         "Excel file saved from the workfile in EViews; empty for a program that makes its "
                                         "own pages (wfcreate)")] = ""
        time: Annotated[str, Widget("text", label="Date column", help="a CSV or Excel file's column of dates (obs, say); "
                                                                      "empty: as Load data finds it")] = ""
        freq: Annotated[Literal["", "Y", "Q", "M", "W", "D"], Widget("choice", label="Frequency",
                                                                     help="the dates' frequency in a CSV or Excel file: Y "
                                                                          "annual, Q quarterly, M monthly, W weekly, D "
                                                                          "daily; empty: as Load data finds it")] = ""
        args: Annotated[str, Widget("text", label="Program arguments",
                                    help="what follows the program's name in EViews's exec, separated by spaces: %0, %1, "
                                         "..., the script's parameters (arg0, arg1)")] = ""

        def options(self) -> dict[str, Any]:
            """translate_program's keyword arguments."""
            out: dict[str, Any] = {}
            if self.data.strip():
                out["bundle"] = self.data.strip()
                read = {k: v for k, v in (("time", self.time.strip()), ("freq", self.freq)) if v}
                if read and not self.data.strip().lower().endswith(".evbundle"):
                    out["read_options"] = read
            if self.args.strip():
                out["args"] = " ".join(self.args.split())
            return out

        def validate(self, session: Session) -> list[str]:
            if not self.path.strip():
                return ["choose the EViews program (.prg) to translate"]
            if not Path(self.path).is_file():
                return [f"there is no program {self.path}"]
            if self.data.strip() and not Path(self.data.strip()).is_file():
                return [f"there is no file {self.data.strip()}: the bundle the export program wrote, or a CSV or Excel file"]
            return []

        def translate(self) -> Any:
            from .program import translate_program

            return translate_program(self.path, **self.options())

        def run(self, session: Session) -> Any:
            translation = self.translate()
            _TRANSLATIONS[_key(self)] = translation
            files = session.attachments.setdefault("pymetrix-eviews", {})
            stem = Path(self.path).stem
            for name, text in translation.sources.items():
                files[f"{stem}/{name}"] = text.encode("utf-8")  # the program as translated: the panel's copy if it moves
            files[f"{stem}.py"] = translation.script.encode("utf-8")
            files[f"{stem}.txt"] = translation.report.to_text().encode("utf-8")
            return translation

        def to_code(self) -> str:
            options = "".join(f", {k}={v!r}" for k, v in self.options().items())
            return (f"from pymetrix_eviews import translate_program\n"
                    f"{_py_name(Path(self.path).stem)}_translation = translate_program({self.path!r}{options})")


_TRANSLATIONS: dict[tuple[Any, ...], Any] = {}  # the translations made, by their step's fields (the panel's)


def _key(cmd: Any) -> tuple[Any, ...]:
    return (cmd.path, cmd.data, cmd.time, cmd.freq, cmd.args)


def translation_of(session: Any, cmd: Any) -> Any:
    """The translation a step made: kept from the step, or made again (from the program the project keeps, when the
    file is no longer there)."""
    key = _key(cmd)
    if key not in _TRANSLATIONS:
        from .program import translate_program

        if Path(cmd.path).is_file():
            _TRANSLATIONS[key] = cmd.translate()
        else:
            kept = session.attachments.get("pymetrix-eviews", {}).get(f"{Path(cmd.path).stem}/{Path(cmd.path).name}")
            if kept is None:
                return None
            options = cmd.options()
            _TRANSLATIONS[key] = translate_program(kept.decode("utf-8"), folder=Path(cmd.path).parent, **options)
    return _TRANSLATIONS[key]


def translations(session: Any) -> list[tuple[Any, Any]]:
    """The session's translations, from its History's Translate EViews program steps: (step, translation)."""
    out = []
    for record in getattr(session, "history", []):
        cmd = getattr(record, "command", None)
        if isinstance(cmd, TranslateEViewsProgram):
            t = translation_of(session, cmd)
            if t is not None:
                out.append((cmd, t))
    return out


def _help() -> None:
    add_help_topic("eviews-import", "EViews workfiles (the pymetrix-eviews plugin)", """
        pymetrix does not read EViews's binary workfiles (.wf1): their format is not published, and a misread column
        of numbers looks like data. There are two ways in.

        **Supported: save from EViews as CSV or Excel.** In EViews, save the page (File > Save As, or
        wfsave(type=...)) as CSV or Excel, then File > Load data in pymetrix, with the date column as Time and the
        page's frequency. Labels and other metadata do not come with it.

        **Experimental: the export program.** File > Import EViews workfile > Get the export program saves
        pymetrix_export.prg. In EViews, open the workfile and run it: exec pymetrix_export.prg C:\\work\\macro-export.
        It writes a bundle: each page's series as CSV, with the series' labels, units and sources, the groups,
        samples, scalars, strings, vectors and matrices, value maps, equations (with EViews's estimates) and models,
        and a manifest, <workfile>.evbundle, which File > Import EViews workfile (Workfile bundle) reads. The program was
        written from EViews's documentation and has NOT been run in EViews: the project has no licence. Every point
        to check is marked (verify) in it. If it fails, or its bundle is refused, use CSV or Excel and report what
        happened.

        Each page becomes a dataset named after it, read by Dataset.read(path, page=...), so replaying needs the
        plugin; Save the data as Parquet reads each page from a Parquet copy instead, so the script needs only
        pymetrix. In code, eviews_objects(data) gives the page's groups (as selections), samples, scalars, matrices
        and models (read by read_eviews); compare_estimates(data) refits each least-squares equation and sets
        pymetrix's estimates beside EViews's, saying where they match.

        What is refused, with its place: a folder without its manifest (an incomplete export), a bundle of a newer
        format version (update pymetrix-eviews), a CSV without a column the manifest lists, and pages of a frequency
        pandas has no period for (semi-annual, multi-year, intraday): those read as undated. Value maps are kept in
        each series' metadata (Variable.extra) but are not labels of its values: pymetrix has no value labels yet.

        EViews programs (.prg) are translated with File > Translate EViews program, or in code with
        translate_program: see the topic on EViews programs.
        """, see_also=("eviews-programs",))
    add_help_topic("eviews-programs", "EViews programs (the pymetrix-eviews plugin)", """
        File > Translate EViews program asks for the program, the workfile data its wfopen opens (the bundle, or a
        CSV or Excel file saved from EViews, with its date column and frequency) and the program's arguments (%0,
        %1). The translation opens in the EViews translation panel (View > EViews translation): a row for each line,
        its status in words and in colour, the reason; the program and the script side by side, both following the
        row selected; a filter for each status, with its count; Copy (the selected rows, or the whole report) and
        Copy script; and Run as steps, which runs the program's work in this session one step at a time, each
        recorded and undoable, stopping at the first step that fails and marking its row.

        In code, translate_program turns an EViews program (.prg) into three things, from one reading of it:

            from pymetrix_eviews import translate_program
            t = translate_program("macro.prg", bundle="macro-export/macro.evbundle")
            print(t.script)        # Python, each EViews line as a comment above what it became
            t.report.summary()     # a row per line, with its status and the reason
            t.run(session)         # the same work as steps, recorded in the session's History

        The script uses pymetrix's public API only, and reads like the program. Running it, or running the steps,
        gives the same datasets and values. The steps are recorded as if made in dialogs; the script and the
        report are kept in the project.

        **What the statuses mean.** Translated: the line became the Python shown. Translated with a note: it did,
        and the note says something to know (a name changed, a value put in, a formula worked out period by
        period). Left out: the line is not translated, and the reason says why and what to use in pymetrix.
        Error: the line does not read, or names something the program has not made. Every line with text has a
        row: nothing is dropped silently.

        **What is translated (phase 1, data work).** wfopen (of the bundle the export program wrote, or of a CSV
        or Excel file saved from the workfile, given as bundle=), wfcreate and pagecreate, pageselect; smpl and
        sample, with dates, @first and @last, offsets, if conditions and named samples; series, genr and plain
        assignments, within the current sample; frml as a derived series; group; label, displayname and setattr;
        rename, delete and fill; scalars, !numbers and %strings as Python variables, with {%s} and {!i} put in where
        their value is known before the program runs; @mean, @sum, @min, @max, @stdev, @median and @obs as a
        scalar's value over the sample; vectors and matrices.

        **What is translated (phase 2, estimation, tests and models).** Equations by ls (with cov=white or hac, and
        ar and ma terms), tsls, gmm, liml, arch, binary, ordered, censored, count, qreg, heckit and ardl, each a
        fitted Equation (or ARDL), with the options that map; list specifications (y c x y(-1) becomes y ~ x +
        y[-1]) and expression specifications (y = c(1) + c(2)*x, with parameters c_1, c_2). Each estimate keeps
        its coefficient map, EViews's c(i) against pymetrix's coefficient, which later lines use: eq.@coefs(2),
        @stderrs, @tstats, @r2, @se, @dw, @regobs, @aic (per observation, as EViews), and c(2) after an
        estimation. VARs (var v.ls 1 4 y1 y2 @ c) and VECs (v.ec); systems (sur, 3sls, fiml, tsls); wald,
        testdrop, auto, hettest, archtest, reset, chow, hist and rls; uroot (adf, pp, kpss, dfgls, ers, np) and
        coint; forecast, fit and makeresids, the series added to the page. Models: model, append, merge,
        @identity, @add and @innov collected into one text, read by read_eviews before the model is used (its
        refusals in the report at their lines); solve and solveopt, addassign and addinit (Model.add_factors),
        scenarios, override and exclude, the solution's series added to the page under the scenario's alias
        (y_0, y_1).

        Where EViews's defaults differ from pymetrix's (the HAC bandwidth, the ARMA and GARCH estimators, the
        default solver), the line is translated with a note saying what to check. An option that is not mapped
        leaves the line out, naming it: an estimate that ran without it would differ without saying so.

        **What is translated (phase 3, control flow).** for !i = 1 to n step k, for %s a b c, while ... wend, and
        if ... else ... endif stay loops and ifs in the script; exitloop, return and stop become break, return and
        the end of the program. Subroutines become functions, defined at the top of the script, with string and
        scalar parameters; EViews passes arguments by reference, so a parameter the subroutine assigns goes back to
        the caller's variable; a local subroutine shares none of the program's names (its data and estimation
        lines are left out, as EViews deletes what it makes). include translates the included program in line,
        its rows under its name. {%s} and {!i} with a value known only as the program runs make the name then (an
        f-string in the script). Conditions read the program's strings and numbers, @word, @wcount, @wfind,
        @wlookup, @isobject, @lower, @replace, @mid and the other string functions, a group's @count and
        @seriesname, the sample's statistics, and an equation's @r2 and other values, so "if eq.@r2 > 0.5" is
        decided by the equation just estimated. The program's arguments (%0, %1) are the script's parameters
        (arg0, arg1). Run as steps, the loops and conditions run in Python and only the steps are recorded, each
        one undoable; a while loop stops after 10,000 passes.

        **What is left out.** stepls, breakls, threshold, switchreg, glm and nls (each naming pymetrix's
        equivalent); testadd and the extreme-value link (gaps); graphs, tables, spools and display; EViews
        databases (fetch and store); add-ins (what they make can be given to translate_program as objects=);
        subroutine parameters that are objects; names with $ in them; and the expression functions pymetrix has no
        equivalent of: @date, @trend, @seas and @elem in data work, @obssmpl inside a series' formula, @strdate,
        and @sumsby and the like grouped by anything but a series' name (docs/known-gaps.md in pymetrix lists them).

        **Expression functions.** @log10, @floor, @ceiling, @round, @isna, @nan, @iif (and @recode), na, @year,
        @quarter, @month, @day, @obsnum, @obsid, @cumsum, @sumsby, @meansby, @minsby and @maxsby translate to
        pymetrix's functions (log10, floor, ceil, round, isna, fillna, iif, na, year(), ..., obsnum(), cumsum, sum(x,
        by=g) ...). pymetrix computes @cumsum and the by-group statistics over all the page's rows, EViews over the
        current sample: a line under a smpl says so. Random numbers (nrnd, rnd, @rnorm, @runif) become normrnd and
        unifrnd with a seed: rndseed n gives the draws after it n*1000 + 1, n*1000 + 2 ..., and without a rndseed
        the program's draws are seeded 1, 2 ... in order. pymetrix's numbers are never EViews's, so each such line is
        translated with a note "to check".

        **EViews's dates.** @dateval, @datestr (with a format), @dateadd, @datediff, @dtoo and @otod act on the
        program's strings and numbers, so they are worked out as Python values, with EViews's date numbers (days
        since 1 January A.D. 1, that day 0): now, when the values are known before the program runs, else when the
        script runs (it imports pymetrix_eviews.dates). FRB/US's dateshift subroutine translates as it is.

        A program's wfopen needs the workfile's data: export it with the export program (experimental), or save it
        from EViews as CSV or Excel and pass that file, with read_options={"time": "obs", "freq": "Q"} for
        Dataset.read.
        """, see_also=("eviews-import",))


with plugin_context(PLUGIN):
    _help()
    from pymetrix.extension import AppPanel, register as _add

    _add("panels", AppPanel("EViews translation", create_panel, area="bottom"), name="eviews_translation")  # View → EViews translation
