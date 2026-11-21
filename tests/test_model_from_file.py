"""Model.from_file: dispatch by extension to the Dynare, IRIS and EViews readers."""

from pathlib import Path

import pytest

from pymetrix.errors import SpecificationError
from pymetrix.models import Model
from pymetrix.models.dynare import read_mod
from pymetrix.models.eviews import read_eviews, read_eviews_xml

from test_iris import IRIS_NK

FIXTURES = Path(__file__).parent / "fixtures"


def test_dynare_mod():
    path = FIXTURES / "dynare" / "rbc_order2.mod"
    model = Model.from_file(path)
    assert isinstance(model, Model)
    assert model.endogenous == read_mod(path).model.endogenous


def test_iris_model_with_reader_options(tmp_path):
    path = tmp_path / "nk.model"
    path.write_text(IRIS_NK, encoding="utf-8")
    model = Model.from_file(str(path), params={"phi": 1.5})
    assert isinstance(model, Model) and model.endogenous == ("y", "pi", "i", "r")


def test_eviews_program(tmp_path):
    path = tmp_path / "small.PRG"  # the extension in any case
    path.write_text((FIXTURES / "eviews" / "small_model.prg").read_text(encoding="utf-8"), encoding="utf-8")
    model = Model.from_file(path)
    assert isinstance(model, Model)
    assert model.endogenous == ("cap", "ur", "q", "i", "k", "prod", "le", "rhi", "co", "fd", "m", "x")
    assert model.endogenous == read_eviews(path.read_text(encoding="utf-8")).model.endogenous


def test_eviews_xml():
    path = FIXTURES / "eviews" / "frbus_excerpt.xml"
    model = Model.from_file(path)
    assert isinstance(model, Model) and model.endogenous == read_eviews_xml(path).model.endogenous


def test_errors(tmp_path):
    with pytest.raises(SpecificationError, match=r"cannot tell the format of 'm\.txt'.*\.mod, \.model, \.prg, \.xml"):
        Model.from_file(tmp_path / "m.txt")
    with pytest.raises(FileNotFoundError, match="no model file"):
        Model.from_file(tmp_path / "missing.mod")
    with pytest.raises(SpecificationError, match="read_mod takes no options"):
        Model.from_file(FIXTURES / "dynare" / "rbc_order2.mod", params={"a": 1})
