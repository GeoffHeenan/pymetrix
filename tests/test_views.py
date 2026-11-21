"""Views belong to the engine: results, reports and the test kit never need the UI package."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SRC = Path(__file__).parents[1] / "src"


def test_the_engine_never_imports_the_ui(tmp_path):
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(SRC)!r})
        from pymetrix import Dataset, Equation, Report
        from pymetrix.testing import fingerprint
        from pymetrix.views import view_sections

        data = Dataset.example("sample_macro")
        fit = Equation("c ~ y").fit(data)
        assert [t for t, _ in view_sections(fit)][0] == "Summary" and fingerprint(fit)
        Report("x").add(fit, sections=["Summary", "Coefficients"]).save({str(tmp_path / "r.html")!r})
        loaded = sorted(m for m in sys.modules if m.startswith("pymetrix.ui"))
        assert not loaded, loaded
    """)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_the_old_import_path_still_works():
    from pymetrix import views
    from pymetrix.ui import render

    assert render.register_view is views.register_view and render.view_sections is views.view_sections


def test_every_section_is_a_table_or_a_figure():
    """Windows and reports need Tables and Figures; views that hand back pandas objects are converted."""
    pytest.importorskip("statsmodels")
    from pymetrix import Dataset, MarkovSwitching
    from pymetrix.figures import Figure
    from pymetrix.output import Table
    from pymetrix.views import view_sections

    data = Dataset.example("sample_macro").with_series(gy="400*dlog(y)")
    regimes = MarkovSwitching("gy ~ 1", switching_variance=True, search=2, seed=1).fit(data)
    parts = dict(view_sections(regimes))
    assert all(isinstance(v, (Table, Figure)) for v in parts.values()), {t: type(v).__name__ for t, v in parts.items()}
    assert parts["Transition probabilities"].to_frame().shape == (2, 2)
