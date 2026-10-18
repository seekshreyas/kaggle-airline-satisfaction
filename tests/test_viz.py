"""Smoke test: every plot in src.viz.CATALOG builds from the synthetic inputs in tests/cases/viz.yaml."""

import inspect

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import yaml

import src.viz as viz
from src.modeling import diagnostics as dg
from tests.case_loader import CASES_DIR

SPEC = yaml.safe_load((CASES_DIR / "viz.yaml").read_text())


def _run(name: str, seed: int) -> dg.RunDiagnostics:
    spec, rng = SPEC["run"], np.random.default_rng(seed)
    n = spec["n_rows"]
    y = rng.integers(0, 2, n)
    p = np.clip(0.3 * y + 0.7 * rng.random(n), 0, 1)
    segs = pd.DataFrame({c: [v[i % len(v)] for i in range(n)] for c, v in spec["segments"].items()})
    coefs = pd.DataFrame({"feature": spec["features"], "coef": rng.normal(size=len(spec["features"]))})
    coefs["abs_coef"] = coefs["coef"].abs()
    labels = ("not satisfied", "satisfied")
    return dg.RunDiagnostics(
        name=name, run_id=f"r_{name}",
        scores={"roc_auc": 0.8, "log_loss": 0.5, "accuracy": 0.7, "brier": 0.2, "n_rows": n, "n_features": 4},
        roc=dg.roc_points(y, p), calibration=dg.calibration_table(y, p),
        histogram=dg.score_histogram(y, p, labels), confusion=dg.confusion_table(y, p, labels),
        segments=dg.score_by_segment(y, p, segs),
        folds=pd.DataFrame({"fold": [0, 1, 2, 3], "roc_auc": [0.80, 0.81, 0.79, 0.80]}),
        coefs=coefs.sort_values("abs_coef", ascending=False).reset_index(drop=True),
    )


INPUTS = {
    "run": lambda: (_run("model", 0),),
    "ladder": lambda: (pd.DataFrame(SPEC["ladder"], columns=["name", "parent", "cv_mean", "cv_std", "delta", "verdict"]).assign(n_features=10),),
    "deltas": lambda: (pd.DataFrame(SPEC["deltas"], columns=["name", "parent", "fold", "delta"]),),
    "grid": lambda: (pd.DataFrame(SPEC["grid"], columns=["C", "penalty", "cv_mean", "cv_std"]),),
}


@pytest.mark.parametrize("name", list(viz.CATALOG))
def test_plot_builds(name):
    fn, kind = viz.CATALOG[name]
    kwargs = dict(SPEC["kwargs"].get(name, {}))
    if "adopted" in kwargs:
        kwargs["adopted"] = set(kwargs["adopted"])
    fig = fn(*INPUTS[kind](), **kwargs)
    assert isinstance(fig, go.Figure) and len(fig.data) > 0


@pytest.mark.parametrize("name", [n for n, (fn, kind) in viz.CATALOG.items() if kind == "run" and "other" in inspect.signature(fn).parameters])
def test_plot_with_comparison_run(name):
    fn, _ = viz.CATALOG[name]
    fig = fn(_run("model", 0), _run("parent", 1))
    assert any("parent" in (t.name or "") for t in fig.data), "comparison run is drawn"


def test_every_figure_function_is_catalogued():
    """New *_figure functions must be added to CATALOG so they are discoverable and tested."""
    catalogued = {fn for fn, _ in viz.CATALOG.values()}
    for module in (viz.classification, viz.experiments, viz.explain, viz.segments):
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if name.endswith("_figure") and fn.__module__ == module.__name__:
                assert fn in catalogued, f"{module.__name__}.{name} is not in src.viz.CATALOG"


def test_diagnostic_figures_set():
    figs = viz.diagnostic_figures(_run("model", 0), _run("parent", 1))
    assert list(figs) == ["ROC", "Calibration", "Score distribution", "Confusion", "Folds", "Segments", "Coefficients"]
