"""The Fama-French example (``Dataset.example("french")``), downloaded from Kenneth R. French's data library on first use.

The data library's files are Eugene F. Fama's and Kenneth R. French's copyright and state no licence to redistribute
them, so the plugin does not ship them: the first ``Dataset.example("french")`` downloads five CSV files from the
library, builds the dataset (the columns and months the plugin's tutorial uses) and keeps it in the user's cache,
``~/.pymetrix/cache/datasets/french.parquet`` (``$PYMETRIX_HOME/cache/...`` when that is set). Later loads read the
cached file and need no network. Offline, the first load fails with a ``DataError`` that says what to do.

Built from the library's value-weighted monthly returns, as fractions (the files give percent):

- ``F-F_Research_Data_Factors``: ``mkt_rf``, ``smb``, ``hml`` and ``rf``; ``F-F_Momentum_Factor``: ``mom``;
- ``12_Industry_Portfolios``: the 12 industries (``nodur`` ... ``other``);
- ``25_Portfolios_5x5``: sizes 1, 3 and 5 by book-to-market 1, 3 and 5 (``s1v1`` ... ``s5v5``);
- ``25_Portfolios_ME_Prior_12_2``: sizes 1, 3 and 5 by prior return 1, 3 and 5 (``s1m1`` ... ``s5m5``);

over January 1949 to March 2017, the span the plugin used to ship (from linearmodels' copy of the library). The
library revises its history when CRSP does, so the numbers can differ a little from release to release.
"""

from __future__ import annotations

import io
import os
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

import pandas as pd

LIBRARY = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"
BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FIRST, LAST = "1949-01", "2017-03"
TIMEOUT = 60
TITLE = "Fama-French factors and portfolios, monthly, 1949-2017"
DESCRIPTION = ("monthly returns as fractions: the market's excess return mkt_rf, smb, hml, mom and the risk-free rate "
               "rf; 12 industry portfolios; size by value (s1v1 ... s5v5) and size by momentum (s1m1 ... s5m5) "
               "portfolios")
LICENSE = ("Kenneth R. French's data library (copyright Eugene F. Fama and Kenneth R. French), downloaded on first use "
           f"from {LIBRARY}; not redistributed by pymetrix")

_INDUSTRIES = ["NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq", "Telcm", "Utils", "Shops", "Hlth", "Money", "Other"]
_INDUSTRY_LABELS = ["non-durables", "durables", "manufacturing", "energy", "chemicals", "business equipment",
                    "telecoms", "utilities", "retail", "health care", "finance", "other"]


def _sorts(col: str, letter: str, low: str, high: str) -> dict[str, str]:
    """The library's column names for sizes 1, 3, 5 by sorts 1, 3, 5 (``SMALL LoBM``, ``ME3 BM3``...) -> s1v1..."""
    out = {}
    for i in (1, 3, 5):
        for j in (1, 3, 5):
            if i in (1, 5) and j in (1, 5):
                name = f"{'SMALL' if i == 1 else 'BIG'} {low if j == 1 else high}"
            else:
                name = f"ME{i} {col}{j}"
            out[name] = f"s{i}{letter}{j}"
    return out


# file stem -> {the library's column: pymetrix's name}
FILES: dict[str, dict[str, str]] = {
    "F-F_Research_Data_Factors": {"Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml", "RF": "rf"},
    "F-F_Momentum_Factor": {"Mom": "mom"},
    "12_Industry_Portfolios": {c: c.lower() for c in _INDUSTRIES},
    "25_Portfolios_5x5": _sorts("BM", "v", "LoBM", "HiBM"),
    "25_Portfolios_ME_Prior_12_2": _sorts("PRIOR", "m", "LoPRIOR", "HiPRIOR"),
}
COLUMNS = ["mkt_rf", "smb", "hml", "mom", "rf", *[c.lower() for c in _INDUSTRIES],
           *[f"s{i}v{j}" for i in (1, 3, 5) for j in (1, 3, 5)], *[f"s{i}m{j}" for i in (1, 3, 5) for j in (1, 3, 5)]]
LABELS = {"mkt_rf": "the market's return over the risk-free rate", "smb": "size factor (small minus big)",
          "hml": "value factor (high minus low book-to-market)", "mom": "momentum factor", "rf": "risk-free rate",
          **{c.lower(): f"industry: {label}" for c, label in zip(_INDUSTRIES, _INDUSTRY_LABELS)},
          **{f"s{i}v{j}": f"size {i}, book-to-market {j}" for i in (1, 3, 5) for j in (1, 3, 5)},
          **{f"s{i}m{j}": f"size {i}, prior return {j}" for i in (1, 3, 5) for j in (1, 3, 5)}}


def cache_path() -> Path:
    """Where the downloaded dataset is kept: ``cache/datasets/french.parquet`` in ``~/.pymetrix`` (or ``$PYMETRIX_HOME``)."""
    base = os.environ.get("PYMETRIX_HOME") or os.path.join(Path.home(), ".pymetrix")
    return Path(base) / "cache" / "datasets" / "french.parquet"


def _get(url: str) -> bytes:
    """Fetch a URL (tests replace this, so that they need no network)."""
    from .._version import __version__

    request = urllib.request.Request(url, headers={"User-Agent": f"pymetrix-finance/{__version__}"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:  # noqa: S310 - fixed https URLs
        return response.read()


def monthly_table(text: str) -> pd.DataFrame:
    """The first table of monthly rows (``YYYYMM, v1, v2...``) in one of the library's CSV files, in percent as
    printed, with -99.99 and -999 (the library's missing values) as NaN."""
    lines = text.splitlines()
    for start, line in enumerate(lines):  # the header is the line before the first YYYYMM row
        if line.split(",")[0].strip().isdigit() and len(line.split(",")[0].strip()) == 6:
            break
    else:
        raise ValueError("no monthly rows")
    header = [c.strip() for c in lines[start - 1].split(",")][1:]
    rows = []
    for line in lines[start:]:
        cells = [c.strip() for c in line.split(",")]
        if not (len(cells[0]) == 6 and cells[0].isdigit()):
            break  # the monthly table ends at the first blank line or annual heading
        rows.append(cells)
    frame = pd.DataFrame([r[1:] for r in rows], columns=header, index=[r[0] for r in rows]).astype(float)
    frame.index = pd.PeriodIndex([f"{d[:4]}-{d[4:]}" for d in frame.index], freq="M", name="time")
    return frame.mask(frame.isin([-99.99, -999.0]))


def build(get: Callable[[str], bytes] | None = None) -> pd.DataFrame:
    """The dataset's frame, from the library's five zipped CSV files fetched with ``get`` (``_get`` by default)."""
    get = get or _get
    parts = []
    for stem, columns in FILES.items():
        url = f"{BASE}{stem}_CSV.zip"
        try:
            with zipfile.ZipFile(io.BytesIO(get(url))) as archive:
                text = archive.read(archive.namelist()[0]).decode("latin-1")
            table = monthly_table(text)
        except (zipfile.BadZipFile, ValueError, KeyError, IndexError) as exc:
            from pymetrix.errors import DataError

            raise DataError(f"Kenneth French's data library sent {url} in a form pymetrix does not read ({exc})") from None
        missing = [c for c in columns if c not in table.columns]
        if missing:
            from pymetrix.errors import DataError

            raise DataError(f"{url} has no column {', '.join(missing)}: the data library's layout has changed")
        parts.append(table[list(columns)].rename(columns=columns))
    frame = pd.concat(parts, axis=1).loc[FIRST:LAST, COLUMNS] / 100
    return frame


def fetch(get: Callable[[str], bytes] | None = None) -> Path:
    """The cached dataset's path, downloading and building it first if it is not in the cache."""
    path = cache_path()
    if path.is_file():
        return path
    try:
        frame = build(get)
    except (OSError, urllib.error.URLError) as exc:  # no network, a proxy, the site down (HTTPError is a URLError)
        from pymetrix.errors import DataError

        reason = getattr(exc, "reason", None) or exc
        raise DataError(
            f"the 'french' example is downloaded from Kenneth R. French's data library on first use ({LIBRARY}), "
            f"and the download failed: {reason}. Connect to the internet and try again; or download it on another "
            f"computer and copy its french.parquet to {path}") from None
    from pymetrix import Dataset
    from pymetrix.metadata import DatasetMeta, Variable

    variables = {c: Variable(c, label=LABELS[c]) for c in frame.columns}
    data = Dataset.from_frame(frame, variables=variables, meta=DatasetMeta(title=TITLE, description=DESCRIPTION, license=LICENSE))
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.stem}.{os.getpid()}.tmp.parquet")
    data.write(str(partial))
    os.replace(partial, path)  # whole or not at all, so a failed write never leaves a broken cache
    return path


class Download(os.PathLike):
    """The example's path for ``ExampleDataset``: the cached file, fetched when the dataset is first read
    (``os.fspath`` downloads it then). ``source`` says where it comes from without downloading anything."""

    source = LIBRARY

    def __fspath__(self) -> str:
        return str(fetch())

    def __repr__(self) -> str:
        return f"<downloaded on first use from {LIBRARY} to {cache_path()}>"
