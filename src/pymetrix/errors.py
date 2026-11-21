"""pymetrix's errors: one hierarchy, so that ``except PyMetrixError`` catches everything pymetrix raises about its own
inputs and results, while each error is still the built-in kind code expects (``except ValueError`` keeps working).

    PyMetrixError
      DataError (ValueError)            the data: missing series values, frequencies, too few observations, gaps
      NotFoundError (KeyError)          a series, object, test, template or other name that does not exist
      SpecificationError (ValueError)   a formula, option or argument that cannot be used as given
        ExpressionError                 a formula that cannot be parsed or evaluated
      ModelError (ValueError)           a model that cannot be solved, linearised or estimated as specified
      CommandError (ValueError)         a dialog step whose fields do not validate
      MigrationError (ValueError)       a saved step that cannot be brought up to the current version
      PlannedError (NotImplementedError)  a feature that is planned but not built yet

Errors from other libraries (pandas, NumPy, SciPy, statsmodels) are not wrapped: they pass through as they are.
"""

from __future__ import annotations

__all__ = ["DataError", "NotFoundError", "PlannedError", "PluginUnavailable", "PyMetrixError", "SpecificationError"]


class PyMetrixError(Exception):
    """Base class of every error pymetrix raises itself."""


class DataError(PyMetrixError, ValueError):
    """Something about the data: a series' values, frequency, sample, or too few observations."""


class NotFoundError(PyMetrixError, KeyError):
    """A name that does not exist: a series, an object, a test, a template, an equation."""

    def __str__(self) -> str:  # the message as given, not KeyError's quoted repr
        return str(self.args[0]) if self.args else ""


class PluginUnavailable(NotFoundError):
    """Something a plugin provides, asked for while that plugin is disabled or not installed."""


class SpecificationError(PyMetrixError, ValueError):
    """A formula, option or argument that cannot be used as given."""


class PlannedError(PyMetrixError, NotImplementedError):
    """A feature that is planned but not built yet."""


class NotSupportedError(PyMetrixError, AttributeError):
    """A generic verb (``resid``, ``predict``...) that this kind of result does not have; the message says what to
    use instead. It is an ``AttributeError``, so ``hasattr`` and duck typing behave as before."""
