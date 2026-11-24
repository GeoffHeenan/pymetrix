"""pymetrix-eviews: EViews work into pymetrix (docs/design/eviews-plugin.md).

This version reads EViews workfiles through a *bundle*: a folder that ``pymetrix_export.prg``, run once in the
user's own EViews, writes from the open workfile (each page's series as CSV, and a JSON manifest with the metadata,
groups, samples, scalars, matrices, equations and models). The binary ``.wf1`` is not read.

    from pymetrix import Dataset

    quarterly = Dataset.read("C:/work/macro-export/macro.evbundle", page="quarterly")   # recorded, and replayed
    annual = Dataset.read("C:/work/macro-export/macro.evbundle", page="annual")

    from pymetrix_eviews import bundle_pages, compare_estimates, eviews_objects, save_export_program
    bundle_pages("C:/work/macro-export/macro.evbundle")    # the pages, their frequencies and series
    eviews_objects(quarterly)                              # groups, samples, scalars, matrices and models
    compare_estimates(quarterly)                           # least-squares equations refitted, against EViews's numbers
    save_export_program("C:/work")                         # a copy of the export program

**The export program is experimental.** It was written from EViews's public documentation and has never been run in
EViews: the project has no EViews licence. Until someone with one has run it on the two most recent major versions,
saving the workfile from EViews as CSV or Excel, which pymetrix reads with File → Load data…, is the supported way
in. The reader is tested against bundles written by hand to the format, and the manifest's JSON Schema.

EViews programs (``.prg``) are translated into a readable script, a report with a row per line, and the same work as
recorded steps: data work, estimation, tests and models, and control flow (loops, ``if`` blocks, subroutines, included
programs, the program's arguments); what has no pymetrix equivalent is left out with the reason:

    from pymetrix_eviews import translate_program
    t = translate_program("C:/work/macro.prg", bundle="C:/work/macro-export/macro.evbundle")
    print(t.script); t.report.summary(); t.run(session)

In the app: **File → Import EViews workfile…** (a workfile bundle, or a copy of the export program), **File → Translate
EViews program…** and the **EViews translation** panel (the report, the program beside the script, Run as steps).
Running EViews through COM is later work.
"""

from __future__ import annotations

from pymetrix.extension import PluginInfo

from ._version import __version__

PLUGIN = PluginInfo(
    name="pymetrix-eviews",
    version=__version__,
    license="BSD-3-Clause",
    maintainer="Geoffrey Heenan",
    description="EViews workfiles into pymetrix through a bundle the (experimental) export program writes: pages, "
                "series with their labels, groups, samples, scalars, matrices, equations and models",
)

from .bundle import (  # noqa: E402
    FORMAT, FORMAT_VERSIONS, PROGRAM_VERSION, SUFFIX, EViewsBundleNote, bundle_model, bundle_pages, bundle_schema,
    compare_estimates, eviews_objects, eviews_sample, export_program, read_bundle, save_export_program,
)

from .program import ProgramTranslation, ReportRow, TranslationReport, translate_program  # noqa: E402

_registered = False


def register() -> None:
    """The entry point (``pymetrix.plugins``, ``pymetrix.io``, ``pymetrix.commands``): ``.evbundle`` files read
    through ``read_bundle``, the File → Import EViews workfile… dialog and the help topic. Importing the package registers
    them too; a second call does nothing."""
    global _registered
    if _registered:
        return
    _registered = True
    from pymetrix.extension import plugin_context
    from pymetrix.io.registry import register_reader

    with plugin_context(PLUGIN):
        register_reader(SUFFIX, read_bundle)
    # imported outside the plugin's context: it imports pymetrix's own dialogs first, which must stay pymetrix's;
    # it registers its dialog and help topic inside the context itself
    from . import app  # noqa: F401


register()

__all__ = ["FORMAT", "FORMAT_VERSIONS", "PLUGIN", "PROGRAM_VERSION", "SUFFIX", "EViewsBundleNote", "__version__", "bundle_model",
           "bundle_pages", "bundle_schema", "compare_estimates", "eviews_objects", "eviews_sample", "export_program",
           "read_bundle", "register", "save_export_program", "ProgramTranslation", "ReportRow", "TranslationReport", "translate_program"]
