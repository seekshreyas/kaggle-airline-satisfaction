"""Runs the cases declared in tests/cases/diagnostics.yaml."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.diagnostics import ladder
from tests.case_loader import load_cases


def _board_and_folds(case: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = {"parent": ("r1_parent", None, case["parent_folds"]), "child": ("r2_child", "parent", case["child_folds"])}
    board = pd.DataFrame(
        [
            {"run_id": rid, "name": name, "parent": parent, "n_features": 1,
             "cv_mean": np.mean(f), "cv_std": np.std(f), "fit_seconds": 1.0}
            for name, (rid, parent, f) in runs.items()
        ]
    )
    folds = pd.DataFrame(
        [{"run_id": rid, "fold": k, "roc_auc": v} for rid, _, f in runs.values() for k, v in enumerate(f)]
    )
    return board, folds


@pytest.mark.parametrize("case", load_cases("diagnostics.yaml", "ladder"))
def test_ladder_verdict(case):
    board, folds = _board_and_folds(case)
    table = ladder(board, case["keep_rule"], folds).set_index("name")
    assert table.loc["parent", "verdict"] == "reference"
    assert table.loc["child", "verdict"] == case["expected"]


def test_per_run_tables():
    import yaml

    from src.modeling import diagnostics as dg
    from tests.case_loader import CASES_DIR

    case = yaml.safe_load((CASES_DIR / "diagnostics.yaml").read_text())["per_run"]
    y, p, labels, exp = np.array(case["y"]), np.array(case["p"]), tuple(case["labels"]), case["expected"]

    roc = dg.roc_points(y, p)
    assert roc.iloc[0].tolist() == exp["roc_starts_at"] and roc.iloc[-1].tolist() == exp["roc_ends_at"]
    assert np.trapezoid(roc["tpr"], roc["fpr"]) == pytest.approx(exp["auc"], abs=0.01)

    assert dg.calibration_table(y, p)["n"].sum() == exp["calibration_rows"]

    cm = dg.confusion_table(y, p, labels).pivot(index="actual", columns="predicted", values="n")
    assert cm.to_dict(orient="index") == exp["confusion"]

    hist = dg.score_histogram(y, p, labels).groupby("label")["share"].sum()
    assert np.allclose(hist, exp["histogram_share_per_class"])

    seg = dg.score_by_segment(y, p, pd.DataFrame({"g": ["a"] * 6}))
    assert seg.loc[0, "auc"] == pytest.approx(exp["auc"])


def test_grid_table():
    import yaml

    from src.modeling.diagnostics import grid_table
    from tests.case_loader import CASES_DIR

    case = yaml.safe_load((CASES_DIR / "diagnostics.yaml").read_text())["grid_table"]
    board = pd.DataFrame(case["board"], columns=["run_id", "name", "cv_mean"]).assign(cv_std=0.002, fit_seconds=1.0)
    t = grid_table(board, "lr_08").sort_values("C")
    assert t[["C", "penalty", "solver"]].to_dict("records") == case["expected"]
