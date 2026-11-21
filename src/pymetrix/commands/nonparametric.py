"""The nonparametric dialogs: kernel densities and kernel regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from .registry import Command, Widget, command

if TYPE_CHECKING:
    from .session import Session


def _bandwidth(text: str, words: tuple[str, ...]) -> Any:
    """'cv' -> 'cv'; '0.4' -> 0.4; '0.4, 0.5' -> [0.4, 0.5]."""
    from ..errors import SpecificationError

    text = text.strip().lower()
    if text in words:
        return text
    try:
        values = [float(v) for v in text.replace(";", ",").split(",") if v.strip()]
    except ValueError:
        raise SpecificationError(f"the bandwidth is {', '.join(words)} or numbers, not {text!r}") from None
    return values[0] if len(values) == 1 else values


@command(menu="Estimate/Kernel density...")
@dataclass(frozen=True)
class FitKernelDensity(Command):
    """Estimate a series' density by kernel, without assuming its shape.

    The bandwidth sets the smoothing: Silverman's or Scott's rule of thumb, least-squares cross-validation (cv), or a
    number. The window draws the density beside the normal with the same mean and variance.

    The bandwidth matters far more than the kernel: too small and the estimate shows spurious bumps, too large and it
    hides real features such as skewness, fat tails or a second mode. The rules of thumb are tuned to a roughly normal
    density and tend to oversmooth one with several modes; cross-validation adapts to the data but can be erratic in
    small samples. Compare the estimate with the normal overlay to judge departures from normality, and try a few
    bandwidths before reading much into a bump.
    """

    topics: ClassVar[tuple[str, ...]] = ("nonparametric",)

    variable: Annotated[str, Widget("series", depends_on="data", help="the series whose distribution to estimate, e.g. returns or residuals")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    bandwidth: Annotated[str, Widget("text", help="silverman, scott, cv, or a number; smaller shows more detail and more noise")] = "silverman"
    kernel: Annotated[Literal["gaussian", "epanechnikov"], Widget("choice", help="the kernel's shape (cross-validation needs gaussian)")] = "gaussian"
    name: str = "density"
    points: Annotated[int, Widget("number", min=16, max=10_000, step=64, advanced=True, help="points of the grid the density is drawn on")] = 256
    cut: Annotated[float, Widget("number", min=0, max=10, step=0.5, advanced=True,
                                 help="the grid's reach beyond the data, in bandwidths")] = 3.0

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        return make_call("pymetrix:kde", output=self.name, recorded_by=self.command_id, data=Ref(self.data), variable=self.variable,
                         bandwidth=_bandwidth(self.bandwidth, ("silverman", "scott", "cv")), kernel=self.kernel,
                         **{k: getattr(self, k) for k, d in (("points", 256), ("cut", 3.0)) if getattr(self, k) != d})

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()


@command(menu="Estimate/Kernel regression...")
@dataclass(frozen=True)
class FitKernelRegression(Command):
    """Estimate E[y | x] by kernel regression, without a functional form.

    Local linear fits a weighted line around each point (less biased at the edges); Nadaraya-Watson a weighted
    average. Bandwidths, one for each regressor, by leave-one-out cross-validation (cv), the normal reference rule, or
    given as numbers separated by commas. The window draws the fitted curve against the first regressor.

    Without a functional form the estimate can reveal nonlinearities a linear regression would miss, but it needs many
    observations, and the more so the more regressors: with several regressors the curse of dimensionality makes the
    fit noisy, so keep to one or two. The bandwidths trade bias against variance; a regressor with a very large
    cross-validated bandwidth enters almost linearly (local linear) or barely matters (local constant). The mean
    gradient (local linear only) is the average slope, comparable to a linear coefficient. The plot holds the other
    regressors at their medians. The fit describes the conditional mean, not a causal effect.
    """

    topics: ClassVar[tuple[str, ...]] = ("nonparametric",)

    y: Annotated[str, Widget("series", depends_on="data", help="the series whose conditional mean E[y | x] is estimated")]
    x: Annotated[str, Widget("series", depends_on="data", multiple=True, help="the conditioning series; one or two, as the fit gets noisy quickly with more")]
    data: Annotated[str, Widget("object", source="Dataset")] = "data"
    kind: Annotated[Literal["ll", "lc"], Widget("choice", help="ll: local linear; lc: Nadaraya-Watson (local constant)")] = "ll"
    bandwidth: Annotated[str, Widget("text", help="cv, normal, or a number for each regressor")] = "cv"
    name: str = "kreg"

    def to_call(self) -> Any:
        from ..calls import make_call
        from ..codec import Ref

        x = [s.strip() for s in self.x.split(",") if s.strip()]
        return make_call("pymetrix:kernel_regression", output=self.name, recorded_by=self.command_id, data=Ref(self.data), y=self.y, x=x,
                         kind=self.kind, bandwidth=_bandwidth(self.bandwidth, ("cv", "normal")))

    def run(self, session: Session) -> Any:
        return self.to_call().run(session)

    def to_code(self) -> str:
        return self.to_call().to_code()
