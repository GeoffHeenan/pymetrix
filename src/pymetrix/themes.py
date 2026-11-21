"""Themes: how figures, tables and reports look, by name.

    set_theme("policy")                                 # the default for this session
    fig.with_options(theme="monochrome")                # one figure
    Report("Forecast", theme="publication")             # a report: its figures and the digits in its tables
    themes()                                            # the names

A theme changes appearance only: colours, line styles and widths, font sizes, grids, band shading and the
number of decimals shown in report tables. The numbers behind a figure or table are the same in every theme.
Colours that results give their layers (the first colour for a central path, the second for an alternative)
are mapped to the theme's own palette, so a fan chart is navy in ``policy`` and black in ``monochrome``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .errors import NotFoundError

__all__ = ["Theme", "get_theme", "set_theme", "themes"]

# Okabe-Ito: distinguishable with colour-vision deficiencies, and in grey. M. Okabe and K. Ito (2008), "Color Universal
# Design (CUD): how to make figures and presentations that are friendly to colorblind people", jfly.uni-koeln.de/color.
# The fixed order: blue, vermillion, bluish green, reddish purple, orange, sky blue, yellow, black.
OKABE_ITO = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000")


@dataclass(frozen=True)
class Theme:
    """Appearance settings for figures, tables and reports."""

    name: str
    palette: tuple[str, ...] = OKABE_ITO
    line_styles: tuple[str, ...] = ("-",)  # cycled over a panel's lines; more than one for black and white
    line_width: float = 1.6
    font_size: float = 8.0  # ticks, labels and legends
    title_size: float = 9.5
    font_family: str = ""  # empty: matplotlib's default
    grid: bool = True
    band_alpha: float = 0.15
    digits: int = 4  # decimals in report tables
    version: int = 1
    rc: tuple[tuple[str, Any], ...] = ()  # matplotlib settings (rcParams) applied while drawing, e.g. from a style sheet

    @classmethod
    def from_mplstyle(cls, path: str, name: str | None = None) -> Theme:
        """A theme from a matplotlib style sheet (``.mplstyle``), registered under ``name`` (default: the file's name).

        The sheet's colour cycle becomes the palette, and its font, font size, title size, line width and grid set
        those of the theme; every other setting in the sheet (spines, ticks, background, ...) is kept and applied
        while drawing, as matplotlib would apply the sheet. Use it as ``theme="name"`` on any figure or report."""
        from pathlib import Path as _Path

        from matplotlib import rc_params_from_file

        source = _Path(path)
        if not source.is_file():
            raise NotFoundError(f"no style sheet {path!r}")
        rc = dict(rc_params_from_file(str(source), use_default_template=False, fail_on_error=True))
        cycle = rc.get("axes.prop_cycle")
        palette = tuple(cycle.by_key().get("color", ())) if cycle is not None else ()
        family = rc.get("font.family")
        family = family[0] if isinstance(family, (list, tuple)) and family else (family or "")
        size = rc.get("font.size")
        title = rc.get("axes.titlesize")
        theme = cls(name or source.stem,
                    palette=palette or OKABE_ITO,
                    line_width=float(rc.get("lines.linewidth", 1.6)),
                    font_size=float(size) if size is not None else 8.0,
                    title_size=float(title) if isinstance(title, (int, float)) else (float(size) * 1.2 if size else 9.5),
                    font_family=str(family),
                    grid=bool(rc.get("axes.grid", True)),
                    rc=tuple((k, v) for k, v in rc.items() if k != "axes.prop_cycle"))
        _THEMES[theme.name] = theme
        return theme

    def with_options(self, **changes: Any) -> Theme:
        """A copy with some settings changed: ``get_theme("policy").with_options(digits=1)``."""
        return replace(self, **changes)


_THEMES: dict[str, Theme] = {
    "default": Theme("default"),
    "publication": Theme("publication", line_width=1.2, font_size=8.0, title_size=9.0, font_family="serif", grid=False, digits=3),
    "teaching": Theme("teaching", line_width=2.2, font_size=10.0, title_size=12.0, band_alpha=0.2, digits=3),
    "policy": Theme("policy", palette=("#1F3A5F", "#C8102E", "#4F7CAC", "#E0A100", "#2E8B57", "#8C8C8C", "#7A4E9E", "#000000"),
                    line_width=1.8, font_size=8.5, title_size=10.0, band_alpha=0.18, digits=2),
    "monochrome": Theme("monochrome", palette=("#000000", "#555555", "#888888", "#333333", "#AAAAAA", "#222222", "#777777", "#000000"),
                        line_styles=("-", "--", ":", "-."), line_width=1.4, band_alpha=0.12, digits=3),
}
_current = "default"


def themes() -> list[str]:
    """The names of the themes: the built-in ones and those added since (``Theme.from_style``, ``set_theme``)."""
    return list(_THEMES)


def get_theme(theme: str | Theme | None = None) -> Theme:
    """A theme by name, the theme itself, or (``None``) the session's default."""
    if isinstance(theme, Theme):
        return theme
    name = theme or _current
    if name not in _THEMES:
        raise NotFoundError(f"no theme {name!r}; themes: {', '.join(_THEMES)}")
    return _THEMES[name]


def set_theme(theme: str | Theme) -> str | Theme:
    """Make a theme the default for figures and reports that do not name one; returns the previous default."""
    global _current
    previous = _current
    if isinstance(theme, Theme):
        _THEMES[theme.name] = theme
        theme = theme.name
    get_theme(theme)
    _current = theme
    return previous
