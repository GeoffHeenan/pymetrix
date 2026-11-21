"""Example datasets shipped with pymetrix, for teaching, documentation and tests.

    from pymetrix import Dataset
    data = Dataset.example("sample_macro")

They are installed with the package, so notebooks find them from any folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import NotFoundError

if TYPE_CHECKING:
    from ..data import Dataset

__all__ = ["available", "load", "path"]

_DESCRIPTIONS = {
    "sample_macro": "a simulated quarterly economy, 2000Q1-2024Q4: consumption c, investment i, "
                    "government spending g and output y = c + i + g",
    "sample_monthly": "monthly indicators of the same economy, 2000-01 to 2025-03, for nowcasting: industrial "
                      "production ip, retail sales retail, employment emp and a survey pmi, each with its publication lag",
    # cross sections and panels (scripts/make_examples.py builds them; each dataset's license says where it is from)
    "mroz": "married women's labour supply in 1975 (Mroz 1987), 753 women: hours, participation inlf, wages, education, "
            "children; Tobit, Heckman and probit examples",
    "card": "returns to schooling (Card 1995), 3,010 men: wage, educ and nearc4, growing up near a four-year college, as "
            "an instrument",
    "wage_panel": "young men's wages, 545 men, 1980-1987 (Vella and Verbeek 1998): lwage, union, experience; panel models",
    "jobtraining": "Michigan job-training grants, 157 firms, 1987-1989 (Holzer et al. 1993): scrap rates, grant receipt; "
                   "panel and difference-in-differences",
    "nile": "the annual flow of the Nile at Aswan, 1871-1970: the classic structural break, a fall from 1898",
    "randhie": "the RAND Health Insurance Experiment, 20,190 person-years: mdvis, doctor visits, and insurance terms; "
               "count models",
    "anes96": "the 1996 American National Election Study, 944 respondents: PID, party identification on a 0 to 6 scale; "
              "ordered and multinomial models",
    "grunfeld": "Grunfeld's investment data, 11 US firms, 1935-1954: invest, value and capital; panels and systems",
    "policy_panel": "simulated: 50 states, 2000-2019, half adopting a policy in 2010 with a true effect of 2.0; "
                    "difference-in-differences and event studies",
    "rd_sample": "simulated: 2,000 people with a score; those at or above 0 get a programme raising y by 1.5; "
                 "regression discontinuity",
    "synth_panel": "simulated: 20 regions, 1990-2015; region 1 adopts a policy in 2005 that lowers y by 3.0; "
                   "synthetic control",
}


def _plugins() -> dict[str, Any]:
    """Example datasets from plugins (``register("datasets", ExampleDataset(...))``), by name."""
    from ..extension import ExampleDataset
    from ..plugins import load, registry

    load("datasets")
    return {e.name: e.obj for e in registry("datasets").entries(available=True) if isinstance(e.obj, ExampleDataset)}


def available() -> dict[str, str]:
    """The example datasets and what they hold: pymetrix's, then plugins'."""
    out = dict(_DESCRIPTIONS)
    for name, ds in _plugins().items():
        out.setdefault(name, ds.description)
    return out


def path(name: str) -> Path:
    """Where an example dataset's Parquet file is installed. A plugin's ``ExampleDataset.path`` may be any path-like
    object: one that downloads the data on first use (the finance plugin's ``french``) fetches it here."""
    if name in _DESCRIPTIONS:
        return Path(__file__).with_name(f"{name}.parquet")
    extra = _plugins()
    if name not in extra:
        raise NotFoundError(f"no example dataset {name!r}; available: {', '.join(available())}")
    return Path(extra[name].path)


def load(name: str) -> Dataset:
    from ..data import Dataset

    return Dataset.read(str(path(name)))
