#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standard sets of figures, composed from the other viz modules.
"""

__author__ = "Shreyas"
__email__ = "seekshreyas@gmail.com"

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from src.viz.classification import calibration_figure, confusion_figure, roc_figure, score_distribution_figure
from src.viz.experiments import folds_figure
from src.viz.explain import coefficients_figure
from src.viz.segments import segments_figure

if TYPE_CHECKING:
    from src.modeling.diagnostics import RunDiagnostics


def diagnostic_figures(run: "RunDiagnostics", other: "RunDiagnostics | None" = None) -> dict[str, go.Figure]:
    """All diagnostic figures for a run, keyed by short name (for tabs or PNG export)."""
    return {
        "ROC": roc_figure(run, other),
        "Calibration": calibration_figure(run, other),
        "Score distribution": score_distribution_figure(run),
        "Confusion": confusion_figure(run),
        "Folds": folds_figure(run, other),
        "Segments": segments_figure(run, other),
        "Coefficients": coefficients_figure(run),
    }
