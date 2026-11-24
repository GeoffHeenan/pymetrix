"""The plugin in the app: File → Import EViews workfile… and its help topic, registered through the extension API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from pymetrix.extension import Command, Widget, add_help_topic, command, plugin_context

from . import PLUGIN
from .bundle import SUFFIX, _py_name, bundle_pages, save_export_program

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
        """)


with plugin_context(PLUGIN):
    _help()
