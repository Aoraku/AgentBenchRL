"""Named measurement curves and their finite-segment trapezoid areas."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class CurvePoint:
    """One coordinate of a named measurement curve."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Curve:
    """Points whose axes retain the measurement and budget semantics."""

    x_axis: str
    y_axis: str
    points: tuple[CurvePoint, ...]


@dataclass(frozen=True, slots=True)
class CurveArea:
    """An area-under-curve result that keeps its named axes."""

    name: str
    x_axis: str
    y_axis: str
    value: float


def build_curve(
    records: Iterable[Mapping[str, float] | CurvePoint], *, x_axis: str, y_axis: str
) -> Curve:
    """Build an x-sorted curve from raw records and explicit axis names."""
    _validate_axes(x_axis, y_axis)
    points: list[CurvePoint] = []
    for record in records:
        if isinstance(record, CurvePoint):
            point = record
        else:
            try:
                point = CurvePoint(x=float(record[x_axis]), y=float(record[y_axis]))
            except KeyError as error:
                raise ValueError(f"curve record is missing axis {error.args[0]!r}") from error
        points.append(point)
    return Curve(
        x_axis=x_axis,
        y_axis=y_axis,
        points=tuple(sorted(points, key=lambda point: point.x)),
    )


def trapezoid_auc(curve: Curve) -> CurveArea:
    """Integrate finite adjacent points while retaining the integrated axis name."""
    if not isinstance(curve, Curve):
        raise TypeError("trapezoid_auc requires a named Curve")
    _validate_axes(curve.x_axis, curve.y_axis)
    area = 0.0
    for left, right in zip(curve.points, curve.points[1:], strict=False):
        if not all(isfinite(value) for value in (left.x, left.y, right.x, right.y)):
            continue
        area += (right.x - left.x) * (left.y + right.y) / 2.0
    return CurveArea(
        name=f"AUC_{curve.x_axis}",
        x_axis=curve.x_axis,
        y_axis=curve.y_axis,
        value=area,
    )


def _validate_axes(x_axis: str, y_axis: str) -> None:
    if (
        not isinstance(x_axis, str)
        or not isinstance(y_axis, str)
        or not x_axis.strip()
        or not y_axis.strip()
    ):
        raise ValueError("curves require non-empty named axes")
