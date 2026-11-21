"""Draw the app's icon: run once, offline, after changing the design.

    python scripts/make_icon.py        # writes src/pymetrix_app/icons/pymetrix.png and pymetrix.ico

The icon is the first step from data to information: a field of scattered data points, and a magnifying glass that
finds the function behind them, f(x). Ink on paper, with the x in amber. It is drawn with matplotlib at 1024 pixels
and scaled down: a 256-pixel PNG for the window and the Linux launcher, and a Windows .ico holding every size Windows
asks for. Below 32 pixels the dots and f( ) would only blur, so the smallest sizes show the glass and the x.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch
from PIL import Image

INK, PAPER, ACCENT, MUTED = "#1b2a2f", "#f4efe6", "#e07a2f", "#9aa7a9"
CX, CY, R = 0.58, 0.58, 0.25  # the lens
# the data: a jittered grid, fixed here so the icon does not change with a random number generator
DOTS = [(0.191, 0.168), (0.486, 0.155), (0.609, 0.155), (0.735, 0.135), (0.134, 0.266), (0.268, 0.214),
        (0.463, 0.277), (0.611, 0.219), (0.733, 0.262), (0.250, 0.338), (0.150, 0.420), (0.179, 0.528),
        (0.274, 0.485), (0.134, 0.639), (0.157, 0.760), (0.862, 0.735), (0.129, 0.860), (0.244, 0.866),
        (0.381, 0.874), (0.759, 0.862), (0.862, 0.475)]
SIZES = [16, 20, 24, 32, 40, 48, 64, 128, 256]
OUT = Path(__file__).resolve().parents[1] / "src" / "pymetrix_app" / "icons"


def draw(pixels: int = 1024, small: bool = False) -> Image.Image:
    """The icon at ``pixels`` square; ``small`` draws it for 16 to 24 pixels: no dots, heavier lines and a bold x
    alone in the lens, since the dots and f( ) would only blur."""
    weight = 1.4 if small else 1.0
    scale = pixels / 512  # line widths below are for a 512-pixel drawing
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    fig = plt.figure(figsize=(pixels / 100, pixels / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.04, 0.04), 0.92, 0.92, boxstyle="round,pad=0,rounding_size=0.2",
                                fc=PAPER, ec=INK, lw=14 * scale * weight))
    if not small:
        x, y = np.array(DOTS).T
        ax.scatter(x, y, s=150 * scale**2, c=MUTED, lw=0, zorder=3)
    ax.add_patch(Circle((CX, CY), R, fc="white", ec=INK, lw=20 * scale * weight, zorder=4))
    angle = np.deg2rad(-45)
    x0, y0 = CX + R * np.cos(angle) * 1.08, CY + R * np.sin(angle) * 1.08
    ax.plot([x0, x0 + 0.19 * np.cos(angle)], [y0, y0 + 0.19 * np.sin(angle)], color=INK, lw=38 * scale * weight,
            solid_capstyle="round", zorder=4)
    size = 64 * scale
    if small:
        parts = [(0.0, 0.0, r"$\mathbf{x}$", size * 2.2, ACCENT)]
    else:
        parts = [(-0.105, 0.0, r"$f($", size, INK), (0.012, -0.008, r"$x$", size, ACCENT), (0.105, 0.0, r"$)$", size, INK)]
    dx, dy = _ink_offset(pixels, parts)  # the glyphs' ink, not their font boxes, centred in the lens
    for x, y, text, fontsize, color in parts:
        ax.text(CX + x + dx, CY + y + dy, text, ha="center", va="center", fontsize=fontsize, color=color, zorder=5)
    return _image(fig)


def _ink_offset(pixels: int, parts: list[tuple[float, float, str, float, str]]) -> tuple[float, float]:
    """How far to move the lens's text so that its inked pixels are centred on the lens: the parts are drawn alone,
    on a blank figure, and the box of their ink measured (a font's box includes room for accents and descenders, so
    centring the boxes leaves the letters visibly off centre)."""
    fig = plt.figure(figsize=(pixels / 100, pixels / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for x, y, text, fontsize, _ in parts:
        ax.text(CX + x, CY + y, text, ha="center", va="center", fontsize=fontsize, color="black")
    alpha = np.asarray(_image(fig))[:, :, 3]
    rows, cols = np.nonzero(alpha > 32)
    width, height = alpha.shape[1], alpha.shape[0]
    ink_x = (cols.min() + cols.max() + 1) / 2 / width
    ink_y = 1 - (rows.min() + rows.max() + 1) / 2 / height  # image rows run downwards; the axes run upwards
    return CX - ink_x, CY - ink_y


def _image(fig: plt.Figure) -> Image.Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", transparent=True)
    plt.close(fig)
    return Image.open(buffer).convert("RGBA")


def at(size: int) -> Image.Image:
    """The icon as shown at ``size`` pixels: the full drawing scaled down, or, below 32, the small drawing."""
    big = draw(small=size < 32)
    return big.resize((size, size), Image.LANCZOS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    at(256).save(OUT / "pymetrix.png", optimize=True)
    images = [at(size) for size in SIZES]
    images[-1].save(OUT / "pymetrix.ico", sizes=[(s, s) for s in SIZES], append_images=images[:-1])
    print(f"wrote {OUT / 'pymetrix.png'} and {OUT / 'pymetrix.ico'}")


if __name__ == "__main__":
    main()
