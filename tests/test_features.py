"""Runs the cases declared in tests/cases/features.yaml."""

import numpy as np
import pandas as pd
import pytest
import yaml

import src.features  # noqa: F401
from src.features.build import build_preprocessor
from src.features.registry import get_derive
from src.modeling.experiment import build_pipeline, feature_columns
from src.modeling.metrics import get_metrics
from src.modeling.validate import cross_validate_oof
from src.utils.config import load_config
from src.utils.paths import project_root
from tests.case_loader import CASES_DIR, load_cases

CASES = "features.yaml"
SPEC = yaml.safe_load((CASES_DIR / CASES).read_text())
EXPERIMENTS = project_root() / "configs" / "experiments"


def synthetic(cfg: dict, n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.Series]:
    """Competition-shaped random data with a target that depends on a few columns."""
    syn, schema = SPEC["synthetic"], cfg["schema"]
    df = pd.DataFrame({c: rng.choice(syn["ordinal_values"], n) for c in schema["ordinal"]})
    df["age"] = rng.integers(7, 85, n)
    df["flight_distance"] = rng.integers(30, 5000, n)
    df["departure_delay_in_minutes"] = rng.exponential(15, n).round()
    arr = rng.exponential(15, n).round()
    arr[rng.random(n) < 0.02] = np.nan
    df["arrival_delay_in_minutes"] = arr
    for col, values in syn["categorical_values"].items():
        df[col] = rng.choice(values, n)
    logit = (df["online_boarding"] - 2.5) + (df["class"] == "Business") - df["age"].sub(45).abs() / 20
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int), name="target")
    return df[feature_columns(cfg)], y


@pytest.mark.parametrize("case", load_cases(CASES, "derive"))
def test_derive_block(case):
    df = pd.DataFrame(case["rows"])
    out = get_derive(case["block"])(df, **case.get("params", {}))
    assert list(out.columns) == list(case["expected"]), "exactly the declared columns"
    for col, values in case["expected"].items():
        if isinstance(values[0], str):
            assert out[col].tolist() == values, col
        else:
            np.testing.assert_allclose(out[col].to_numpy(dtype=float), np.array(values, dtype=float), rtol=1e-6)
    assert out.index.equals(df.index)


@pytest.mark.filterwarnings("ignore:Found unknown categories")
@pytest.mark.parametrize("name", SPEC["pipelines"])
def test_pipeline_train_test_consistent(name):
    cfg = load_config(EXPERIMENTS / f"{name}.yaml")
    rng = np.random.default_rng(0)
    X, y = synthetic(cfg, SPEC["synthetic"]["n_rows"], rng)
    X_test, _ = synthetic(cfg, 300, rng)
    for col, value in SPEC["synthetic"]["unseen_in_test"].items():
        X_test.loc[X_test.index[:10], col] = value

    pre = build_preprocessor(cfg).fit(X, y)
    a, b = pre.transform(X), pre.transform(X_test)
    assert list(a.columns) == list(b.columns), "same feature columns on train and test"
    assert not a.isna().any().any() and not b.isna().any().any(), "no NaNs after encoding"


def test_cv_oof_covers_each_row_once_and_is_reproducible():
    cfg = load_config(EXPERIMENTS / "lr_00_baseline.yaml")
    X, y = synthetic(cfg, SPEC["synthetic"]["n_rows"], np.random.default_rng(1))
    folds = np.arange(len(y)) % cfg["cv"]["n_splits"]
    metrics = get_metrics(["roc_auc"])
    r1 = cross_validate_oof(build_pipeline(cfg), X, y, folds, metrics)
    r2 = cross_validate_oof(build_pipeline(cfg), X, y, folds, metrics)
    assert not np.isnan(r1.oof).any(), "every row gets exactly one OOF prediction"
    assert len(r1.fold_scores) == cfg["cv"]["n_splits"]
    np.testing.assert_allclose(r1.oof, r2.oof)
    assert r1.fold_scores["roc_auc"].mean() > 0.6, "learns the synthetic signal"


def test_no_leakage_scaler_learns_train_fold_only():
    """The fitted scaler's mean must equal the TRAIN-fold mean, not the full-data mean."""
    cfg = load_config(EXPERIMENTS / "lr_00_baseline.yaml")
    X, y = synthetic(cfg, SPEC["synthetic"]["n_rows"], np.random.default_rng(2))
    train_idx = np.arange(len(y)) % 4 != 0
    pre = build_preprocessor(cfg).fit(X[train_idx], y[train_idx])
    scaler = pre.named_steps["encode"].named_transformers_["continuous"][-1]
    expected = X.loc[train_idx, "age"].mean()
    assert scaler.mean_[list(scaler.feature_names_in_).index("age")] == pytest.approx(expected)
    assert expected != pytest.approx(X["age"].mean())
