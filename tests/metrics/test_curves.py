from __future__ import annotations

import math

import pytest

from rlbench.metrics import Curve, build_curve, trapezoid_auc


def test_named_curve_auc_integrates_only_finite_adjacent_points() -> None:
    """Joining across a missing measurement fabricates area not supported by facts."""
    curve = build_curve(
        [
            {"learning_env_steps": 0.0, "score": 0.5},
            {"learning_env_steps": 1.0, "score": 0.7},
            {"learning_env_steps": 2.0, "score": math.nan},
            {"learning_env_steps": 3.0, "score": 0.9},
            {"learning_env_steps": 4.0, "score": 1.0},
        ],
        x_axis="learning_env_steps",
        y_axis="score",
    )

    area = trapezoid_auc(curve)

    assert area.x_axis == "learning_env_steps"
    assert area.y_axis == "score"
    assert area.name == "AUC_learning_env_steps"
    assert area.value == pytest.approx(1.55)


def test_trapezoid_auc_rejects_an_unnamed_bare_sequence() -> None:
    """An unlabeled AUC cannot state the budget axis it integrates over."""
    with pytest.raises(TypeError, match="Curve"):
        trapezoid_auc([(0.0, 0.5), (1.0, 0.7)])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("x_axis", "y_axis"), [("", "score"), ("learning_env_steps", "")]
)
def test_trapezoid_auc_rejects_directly_constructed_curve_without_axes(
    x_axis: str, y_axis: str
) -> None:
    """Direct construction must not bypass the named-axis AUC contract."""
    with pytest.raises(ValueError, match="named axes"):
        trapezoid_auc(Curve(x_axis=x_axis, y_axis=y_axis, points=()))
