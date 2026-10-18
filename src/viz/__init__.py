#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plotting library. Diagnostics compute tables; these functions draw them; reports and
notebooks compose the figures.

Conventions for every plot function:
- data in (tidy DataFrame or RunDiagnostics), `plotly.graph_objects.Figure` out
- finish with `theme.style(fig, title)`; colors only from `theme`
- no file I/O (use `export.save_png` to write images)
- add it to CATALOG below; tests/test_viz.py then builds it from synthetic data

    from src.viz import roc_figure, save_png
    from src.viz import catalog; catalog()   # what's available
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

import pandas as pd

from src.viz.classification import calibration_figure, confusion_figure, roc_figure, score_distribution_figure
from src.viz.dashboards import diagnostic_figures
from src.viz.experiments import folds_figure, gains_figure, grid_figure, ladder_figure, paired_deltas_figure
from src.viz.explain import coefficients_figure, drivers_figure
from src.viz.export import for_slides, save_png
from src.viz.segments import segment_auc_figure, segments_figure

# Every public plot: name -> (function, input kind). Input kinds:
#   "run"      RunDiagnostics (+ optional comparison run)
#   "ladder"   ladder table from modeling.diagnostics.ladder
#   "deltas"   per-fold deltas from modeling.diagnostics.paired_deltas
#   "grid"     one row per grid point: C, penalty, cv_mean, cv_std
CATALOG = {
    "roc": (roc_figure, "run"),
    "calibration": (calibration_figure, "run"),
    "score_distribution": (score_distribution_figure, "run"),
    "confusion": (confusion_figure, "run"),
    "folds": (folds_figure, "run"),
    "segments": (segments_figure, "run"),
    "segment_auc": (segment_auc_figure, "run"),
    "coefficients": (coefficients_figure, "run"),
    "drivers": (drivers_figure, "run"),
    "ladder": (ladder_figure, "ladder"),
    "gains": (gains_figure, "ladder"),
    "paired_deltas": (paired_deltas_figure, "deltas"),
    "grid": (grid_figure, "grid"),
}


def catalog() -> pd.DataFrame:
    """Table of available plots: name, module, input kind, description."""
    return pd.DataFrame(
        [
            {
                "name": name,
                "module": fn.__module__.removeprefix("src.viz."),
                "input": kind,
                "description": (fn.__doc__ or "").strip().split("\n")[0],
            }
            for name, (fn, kind) in CATALOG.items()
        ]
    )


__all__ = [
    "CATALOG",
    "catalog",
    "calibration_figure",
    "coefficients_figure",
    "confusion_figure",
    "diagnostic_figures",
    "drivers_figure",
    "folds_figure",
    "for_slides",
    "gains_figure",
    "grid_figure",
    "ladder_figure",
    "paired_deltas_figure",
    "roc_figure",
    "save_png",
    "score_distribution_figure",
    "segment_auc_figure",
    "segments_figure",
]
