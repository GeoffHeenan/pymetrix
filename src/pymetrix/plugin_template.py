"""A new pymetrix plugin, laid out as the finance plugin is: ``pymetrix-new-plugin pymetrix-myname``.

    from pymetrix.plugin_template import new_plugin
    new_plugin("pymetrix-volatility", description="Volatility estimators", author="A. N. Author")

writes a folder ``pymetrix-volatility`` with a package ``pymetrix_volatility`` that works at once: a recorded call
(``volatility:describe``) returning a result with a summary, a graph and a table, a dialog under its own menu, a help
topic, entry points for each group it extends, tests (an independent check, the dialog, the public-surface guard), a
tutorial whose code a test runs, a validation table, and a BSD-3-Clause ``LICENSE`` in the author's name that the sdist
and wheel include (``license-files``). Replace the example with the plugin's own work, keeping the
pattern: a call, a result class with views, a dialog, and a test against something other than the plugin.
"""

from __future__ import annotations

import datetime
import re
import sys
from pathlib import Path

__all__ = ["new_plugin", "main"]

_FILES = {
    "pyproject.toml": '''[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[project]
name = "{dist}"
dynamic = ["version"]
description = "{description}"
readme = "README.md"
license = "BSD-3-Clause"  # change it, LICENSE and PLUGIN's licence together
license-files = ["LICENSE"]  # in the sdist and the wheel (*.dist-info/licenses/)
requires-python = ">=3.11"
authors = [{{ name = "{author}" }}]
dependencies = ["pymetrix>={pymetrix_version}"]

[project.optional-dependencies]
test = ["pytest>=9.0.3"]

# importing the package registers everything; an entry point in each group it extends loads it when that is needed
[project.entry-points."pymetrix.plugins"]
{short} = "{module}:register"
[project.entry-points."pymetrix.commands"]
{short} = "{module}:register"
[project.entry-points."pymetrix.views"]
{short} = "{module}:register"
[project.entry-points."pymetrix.menus"]
{short} = "{module}:register"

[tool.hatch.version]
path = "src/{module}/_version.py"

[tool.hatch.build.targets.wheel]
packages = ["src/{module}"]

[tool.pytest.ini_options]
pythonpath = ["src"]
''',
    "LICENSE": '''BSD 3-Clause License

Copyright (c) {year}, {author}

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
''',
    "README.md": '''# {dist}

{description}. A plugin for [pymetrix](https://github.com/GeoffHeenan/pymetrix).

```
pip install -e .           # in this folder; then the app's Help → Plugins lists it
python -m pytest           # its tests
```

```python
from pymetrix import Dataset
from {module} import describe

describe(Dataset.example("sample_macro"), "c").summary()
```

In the app: **{menu} → Describe…**. The [tutorial](docs/tutorial.md) runs as written (a test checks it), and
[VALIDATION.md](VALIDATION.md) lists what each number is checked against.
''',
    "VALIDATION.md": '''# Validation

What each number this plugin produces is checked against, in its tests (the kinds of reference are pymetrix's: a
library, another package's output, a published table, a formula, or recovery from simulation).

| Method | Checked against | Agreement |
| --- | --- | --- |
| `describe`: mean, standard deviation, quantiles | numpy's mean, std (ddof=1) and quantile | 1e-12 |
''',
    "docs/tutorial.md": '''# {dist}: a tutorial

The code blocks run in order, as one script; `tests/test_tutorial.py` runs them as written.

```python
from pymetrix import Dataset
from {module} import describe

data = Dataset.example("sample_macro")
summary = describe(data, "c")
summary.summary()
```

`summary.plot()` draws the series' histogram; `summary.to_frame()` is the table as a DataFrame.
''',
    "src/{module}/_version.py": '__version__ = "0.1.0"\n',
    "src/{module}/__init__.py": '''"""{description} (``{dist}``): a pymetrix plugin.

Importing the package registers its calls, results' views, dialogs, menu and help, under its ``PLUGIN``.
"""

from __future__ import annotations

from pymetrix.extension import PluginInfo

from ._version import __version__

PLUGIN = PluginInfo(name="{dist}", version=__version__, license="BSD-3-Clause", maintainer="{author}",
                    description="{description}")

from .core import Description, describe  # noqa: E402 - registers under PLUGIN
from . import app  # noqa: E402,F401 - the dialog, views, menu and help

__all__ = ["PLUGIN", "Description", "describe"]


def register() -> None:
    """The entry point: importing the package has registered everything."""
''',
    "src/{module}/core.py": '''"""The plugin's work: here an example, the description of one series.

    d = describe(data, "y")
    d.summary(); d.plot(); d.to_frame()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pymetrix.errors import SpecificationError
from pymetrix.extension import api, plugin_context, views_frame
from pymetrix.output import Table

from . import PLUGIN

if TYPE_CHECKING:
    from pymetrix.data import Dataset
    from pymetrix.figures import Figure


@dataclass(frozen=True, eq=False)
class Description:
    """A series' mean, spread and quantiles."""

    series: str
    values: pd.Series = field(repr=False)

    stage = "explore"  # where the app shows it: explore, estimate, test, forecast, or the plugin's own workspace

    def stats(self) -> dict[str, float]:
        v = self.values.to_numpy(float)
        q = np.quantile(v, [0.05, 0.5, 0.95])
        return {{"observations": float(len(v)), "mean": float(v.mean()), "standard deviation": float(v.std(ddof=1)),
                "5%": float(q[0]), "median": float(q[1]), "95%": float(q[2])}}

    def summary(self) -> Table:
        s = self.stats()
        return Table(["value"], [[v] for v in s.values()], index=list(s), index_name="", title=f"{{self.series}}: description")

    def plot(self) -> Figure:
        from pymetrix.figures import Figure, Histogram, Panel

        return Figure((Panel((Histogram(self.values, label=self.series),), title=f"{{self.series}}: distribution"),))

    def to_frame(self, view: str | None = None) -> pd.DataFrame:
        if view is not None:
            return views_frame(self, view)
        return self.summary().to_frame()

    def __repr__(self) -> str:
        return f"<Description of {{self.series}}>"


with plugin_context(PLUGIN):
    @api("{prefix}:describe")
    def describe(data: Dataset, series: str) -> Description:
        """The mean, standard deviation and quantiles of ``series``."""
        if series not in data.names:
            raise SpecificationError(f"no series {{series!r}} in the dataset")
        values = data[series].dropna()
        if len(values) < 2:
            raise SpecificationError(f"{{series}} has fewer than two values")
        return Description(series, values)
''',
    "src/{module}/app.py": '''"""The plugin in the app: its dialog, its result's window, its menu and its help."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from pymetrix.extension import (Command, MenuLayout, Ref, Widget, add_help_topic, command, figure_views, make_call,
                                plugin_context, register, register_view)

from . import PLUGIN
from .core import Description

if TYPE_CHECKING:
    from pymetrix.commands import Session

with plugin_context(PLUGIN):
    @command(menu="{menu}/Describe...")
    @dataclass(frozen=True)
    class Describe(Command):
        """Describe a series: its mean, standard deviation and quantiles, with a histogram.

        Each field's help is shown in the dialog; the docstring's first line is its title.
        """

        topics: ClassVar[tuple[str, ...]] = ("{prefix}-describe",)

        series: Annotated[str, Widget("series", depends_on="data", help="the series to describe")]
        data: Annotated[str, Widget("object", source="Dataset", help="the dataset")] = "data"
        name: str = "described"

        def to_call(self) -> Any:
            return make_call("{prefix}:describe", output=self.name, recorded_by=self.command_id, data=Ref(self.data),
                             series=self.series)

        def run(self, session: Session) -> Any:
            return self.to_call().run(session)

        def to_code(self) -> str:
            return self.to_call().to_code()


def _views(obj: Description) -> list[Any]:
    return [("Summary", obj.summary()), *figure_views(obj.plot)]


register_view(Description, _views, plugin=PLUGIN)
register("menus", MenuLayout("{menu}", ["Describe"]), name="{short}", plugin=PLUGIN)
add_help_topic("{prefix}-describe", "Describing a series", """
    The mean and standard deviation summarise a series' level and spread; the quantiles, its shape without assuming a
    distribution. Replace this topic with the plugin's own: what the method does, when to use it, what to watch for.
    """, ())
''',
    "tests/conftest.py": '''"""This plugin's tests skip where it cannot be imported."""

import importlib.util

collect_ignore_glob = [] if importlib.util.find_spec("{module}") is not None else ["test_*.py"]
''',
    "tests/test_{short}.py": '''"""{dist}, checked independently: the description against numpy; the dialog records the call."""

import numpy as np
import pytest

from pymetrix import Dataset
from pymetrix.errors import SpecificationError

from {module} import describe


def test_the_description_against_numpy():
    data = Dataset.example("sample_macro")
    s = describe(data, "c").stats()
    c = data["c"].dropna().to_numpy(float)
    assert s["mean"] == pytest.approx(c.mean(), rel=1e-12) and s["standard deviation"] == pytest.approx(c.std(ddof=1), rel=1e-12)
    assert [s["5%"], s["median"], s["95%"]] == pytest.approx(np.quantile(c, [0.05, 0.5, 0.95]), rel=1e-12)
    with pytest.raises(SpecificationError, match="no series"):
        describe(data, "nope")


def test_the_dialog_records_the_call():
    from pymetrix.commands import Session
    from pymetrix.commands.registry import COMMANDS
    from pymetrix.views import sections

    s = Session()
    s.objects["data"] = Dataset.example("sample_macro")
    s.run(COMMANDS["Describe"].cls("c"))
    assert s.history[-1].code == "described = describe(data, 'c')"
    assert [t for t, _ in sections(s.objects["described"])][0] == "Summary"
''',
    "tests/test_tutorial.py": '''"""The tutorial (docs/tutorial.md) runs as written: its code blocks, in order, as one script."""

import re
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs" / "tutorial.md"


def test_the_tutorial_runs_as_written():
    blocks = re.findall(r"```python\\n(.*?)```", DOC.read_text(encoding="utf-8"), flags=re.S)
    assert blocks
    scope: dict = {{"__name__": "tutorial"}}
    for i, block in enumerate(blocks, 1):
        exec(compile(block, f"{{DOC.name}}, block {{i}}", "exec"), scope)
''',
    "tests/test_public_surface.py": '''"""The plugin imports from pymetrix only through the extension API and the stable modules, and no private name:
so pymetrix's internals can change without breaking it."""

import ast
from pathlib import Path

from pymetrix.extension import STABLE

SRC = Path(__file__).resolve().parents[1] / "src" / "{module}"


def test_only_the_public_surface():
    bad = []
    for f in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 and node.module.split(".")[0] == "pymetrix":
                if node.module not in STABLE:
                    bad.append(f"{{f.name}}: from {{node.module}}")
                bad += [f"{{f.name}}: {{node.module}}.{{a.name}}" for a in node.names if a.name.startswith("_")]
            elif isinstance(node, ast.Import):
                bad += [f"{{f.name}}: import {{a.name}}" for a in node.names if a.name.split(".")[0] == "pymetrix" and a.name not in STABLE]
    assert not bad, "imports outside the public surface:\\n" + "\\n".join(bad)
''',
}


def new_plugin(folder: str | Path, description: str = "", author: str = "", menu: str = "") -> Path:
    """Write a working plugin at ``folder`` (named like ``pymetrix-myname``); returns the folder."""
    from .errors import SpecificationError
    from ._version import __version__

    root = Path(folder)
    dist = root.name
    if not re.fullmatch(r"pymetrix-[a-z][a-z0-9]*(-[a-z0-9]+)*", dist):
        raise SpecificationError(f"name the plugin's folder like pymetrix-myname (lower case, digits and dashes), not {dist!r}")
    if root.exists() and any(root.iterdir()):
        raise SpecificationError(f"{root} exists and is not empty")
    short = dist.removeprefix("pymetrix-").replace("-", "_")
    fill = {"dist": dist, "module": dist.replace("-", "_"), "short": short, "prefix": short,
            "description": description or f"The {short.replace('_', ' ')} plugin", "author": author or "the authors",
            "menu": menu or short.replace("_", " ").title(), "pymetrix_version": __version__,
            "year": datetime.date.today().year}
    for template, text in _FILES.items():
        path = root / template.format(**fill)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.format(**fill), encoding="utf-8")
    return root


def main(argv: list[str] | None = None) -> int:
    """``pymetrix-new-plugin pymetrix-myname [description]``."""
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in ("-h", "--help"):
        print(__doc__.split("\n\n")[0])
        print("usage: pymetrix-new-plugin pymetrix-myname [description]")
        return 0 if args else 2
    root = new_plugin(args[0], " ".join(args[1:]))
    print(f"wrote {root}: cd there, pip install -e ., python -m pytest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
