"""Runs the cases declared in tests/cases/experiment.yaml."""

import pytest

from src.modeling.experiment import expand_grid
from tests.case_loader import load_cases


@pytest.mark.parametrize("case", load_cases("experiment.yaml", "expand_grid"))
def test_expand_grid(case):
    cfg = {"name": "x", "model": {"name": "logreg", "params": {"max_iter": 10}, "grid": case["grid"]}}
    out = expand_grid(cfg)
    assert [c["name"] for c in out] == case["expected_names"]
    for c in out:
        assert c["model"]["params"]["max_iter"] == 10, "base params kept"
        assert c["model"].get("grid") is None or case["grid"] is None


def test_append_leaderboard_aligns_new_columns(tmp_path):
    import pandas as pd

    from src.modeling.experiment import append_leaderboard

    board = tmp_path / "leaderboard.csv"
    append_leaderboard(board, {"run_id": "r1", "cv_mean": 0.9})
    append_leaderboard(board, {"run_id": "r2", "cv_mean": 0.91, "submission_path": "x.csv"})
    lb = pd.read_csv(board)
    assert list(lb.columns) == ["run_id", "cv_mean", "submission_path"]
    assert lb["submission_path"].isna().tolist() == [True, False]
